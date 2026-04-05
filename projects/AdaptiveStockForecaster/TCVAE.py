import torch
import torch.nn as nn
import torch.nn.functional as F

class TCVAE(nn.Module):
    """
    时序条件变分自编码器 (Temporal Conditional Variational Autoencoder)
    网络结构:
        - 编码器: 使用 stride=2 的 Conv1d 层进行时序下采样
        - 潜在层: 高斯分布 (mu, logvar)
        - 解码器: 使用 ConvTranspose1d 层进行时序上采样
    参数:
        feature_dim: 输入特征维度
        latent_dim: 潜在空间维度
        hidden_dim: 隐藏层维度
        seq_len: 预期序列长度（用于形状验证和采样）
    输入:
        x: (batch_size, seq_len, feature_dim) 时间序列数据
    输出:
        x_recon: (batch_size, seq_len, feature_dim) 重建的时间序列
        mu: (batch_size, latent_dim, compressed_seq_len) 潜在分布均值
        logvar: (batch_size, latent_dim, compressed_seq_len) 潜在分布对数方差
    """

    def __init__(self, feature_dim, latent_dim=64, hidden_dim=128, seq_len=None):
        super(TCVAE, self).__init__()

        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # 编码器: (B, F, T) -> (B, hidden, T/4)
        # 通过两次 stride=2 卷积实现 4 倍时序压缩
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # VAE 潜在空间投影层
        # 使用 1x1 卷积将隐藏表示投影到潜在空间
        self.fc_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.fc_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        # 解码器输入层: (B, latent, T/4) -> (B, hidden, T/4)
        self.decoder_input = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
        )

        # 解码器: (B, hidden, T/4) -> (B, F, T)
        # 通过转置卷积进行上采样，恢复原始时间分辨率
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),

            nn.Conv1d(hidden_dim // 2, feature_dim, kernel_size=3, stride=1, padding=1),
            # 输出激活函数根据数据范围选择:
            # nn.Sigmoid()  # 数据在 [0, 1] 范围时使用
            # nn.Tanh()     # 数据在 [-1, 1] 范围时使用
        )

    def encode(self, x):
        """
        将输入编码为潜在分布参数
        参数:
            x: (B, T, F) 输入时间序列
        返回:
            mu: (B, latent_dim, T') 潜在分布的均值
            logvar: (B, latent_dim, T') 潜在分布的对数方差
        """
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T) 转换为 Conv1d 格式
        h = self.encoder(x)  # (B, hidden_dim, T') 编码后的隐藏表示

        mu = self.fc_mu(h)  # 均值
        logvar = self.fc_logvar(h)  # 对数方差

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧: z = mu + std * epsilon
        允许梯度通过随机采样过程反向传播
        参数:
            mu: 分布均值
            logvar: 分布对数方差
        返回:
            z: 采样的潜在变量
        """
        std = torch.exp(0.5 * logvar)  # 标准差 = exp(0.5 * log(var))
        eps = torch.randn_like(std)  # 从标准正态分布采样
        return mu + eps * std  # 重参数化采样

    def decode(self, z):
        """
        将潜在变量解码为重建序列
        参数:
            z: (B, latent_dim, T') 潜在变量
        返回:
            x_recon: (B, T, F) 重建的时间序列
        """
        h = self.decoder_input(z)  # 解码器输入处理
        x_recon = self.decoder(h)  # (B, F, T) 解码重建
        return x_recon.transpose(1, 2)  # (B, T, F) 转回原始格式

    def forward(self, x):
        """
        VAE 前向传播
        参数:
            x: (B, T, F) 输入时间序列
        返回:
            x_recon: (B, T, F) 重建的时间序列
            mu: (B, latent_dim, T') 潜在均值
            logvar: (B, latent_dim, T') 潜在对数方差
        """
        mu, logvar = self.encode(x)  # 编码
        z = self.reparameterize(mu, logvar)  # 采样
        x_recon = self.decode(z)  # 解码

        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar, kl_weight=1.0):
        """
        VAE 损失函数 = 重建损失 + KL 散度
        参数:
            x: 原始输入
            x_recon: 重建输入
            mu: 潜在均值
            logvar: 潜在对数方差
            kl_weight: KL 项权重 (beta-VAE 的 beta 参数)
        返回:
            total_loss: 总损失
            recon_loss: 重建损失
            kl_loss: KL 散度损失
        """
        # 重建损失 (均方误差)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL 散度: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # 衡量编码分布与标准正态分布的差异
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # 总损失
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """
        从先验分布中采样生成新序列
        参数:
            num_samples: 生成样本数量
            device: 生成设备
        返回:
            samples: (num_samples, seq_len, feature_dim) 生成的时间序列
        """
        # 需要知道压缩后的序列长度
        if self.seq_len is None:
            raise ValueError("seq_len 必须在初始化时指定才能采样")

        # 计算压缩后的长度（假设压缩率为 4）
        compressed_len = self.seq_len // 4

        # 从标准正态分布采样
        z = torch.randn(num_samples, self.latent_dim, compressed_len).to(device)

        # 解码生成样本
        samples = self.decode(z)
        return samples


class TCVAEConceptEncoder(nn.Module):
    """
    基于 TCVAE 的概念编码器，支持注意力池化
    将变长时间序列映射为固定长度的概念向量
    参数:
        feature_dim: 输入特征维度
        latent_dim: 潜在空间维度
        hidden_dim: 隐藏层维度
        pooling: 池化策略 ('mean', 'max', 'attention', 'last')
        use_variance: 是否同时返回方差估计
    """

    def __init__(self, feature_dim, latent_dim=64, hidden_dim=128,
                 pooling='attention', use_variance=False):
        super(TCVAEConceptEncoder, self).__init__()

        self.tcv_encoder = TCVAE(feature_dim, latent_dim, hidden_dim)
        self.pooling = pooling
        self.use_variance = use_variance

        # 注意力池化机制
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.Tanh(),
                nn.Linear(latent_dim // 2, 1)
            )

    def forward(self, x, return_distribution=False):
        """
        从时间序列中提取概念表示
        参数:
            x: (B, T, F) 时间序列输入
            return_distribution: 是否分别返回 mu 和 logvar
        返回:
            concept: (B, latent_dim) 概念表示向量
            或 (concept_mu, concept_logvar) 如果 return_distribution=True
        """
        mu, logvar = self.tcv_encoder.encode(x)  # (B, latent_dim, T')

        if return_distribution:
            # 分别返回分布参数
            if self.pooling == 'mean':
                concept_mu = mu.mean(dim=2)
                concept_logvar = logvar.mean(dim=2)
            elif self.pooling == 'max':
                concept_mu = mu.max(dim=2)[0]
                concept_logvar = logvar.max(dim=2)[0]
            elif self.pooling == 'last':
                concept_mu = mu[:, :, -1]
                concept_logvar = logvar[:, :, -1]
            elif self.pooling == 'attention':
                # 时间维度上的注意力池化
                mu_t = mu.transpose(1, 2)  # (B, T', latent_dim)
                attn_weights = F.softmax(self.attention(mu_t), dim=1)  # (B, T', 1)
                concept_mu = (mu_t * attn_weights).sum(dim=1)  # (B, latent_dim)

                logvar_t = logvar.transpose(1, 2)
                concept_logvar = (logvar_t * attn_weights).sum(dim=1)
            else:
                raise ValueError(f"未知的池化方式: {self.pooling}")

            return concept_mu, concept_logvar

        else:
            # 采样后进行池化
            z = self.tcv_encoder.reparameterize(mu, logvar)  # (B, latent_dim, T')

            # 时序池化
            if self.pooling == 'mean':
                concept = z.mean(dim=2)
            elif self.pooling == 'max':
                concept = z.max(dim=2)[0]
            elif self.pooling == 'last':
                concept = z[:, :, -1]
            elif self.pooling == 'attention':
                z_t = z.transpose(1, 2)  # (B, T', latent_dim)
                attn_weights = F.softmax(self.attention(z_t), dim=1)  # (B, T', 1)
                concept = (z_t * attn_weights).sum(dim=1)  # (B, latent_dim)
            else:
                raise ValueError(f"未知的池化方式: {self.pooling}")

            if self.use_variance:
                # 同时返回方差估计
                temporal_variability = z.var(dim=2)
                return concept, temporal_variability

            return concept


