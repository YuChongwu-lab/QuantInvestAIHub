# proceed_diffusion_augmentation.py
# -*- coding: utf-8 -*-
"""
Proceed + Diffusion 数据增强集成模块

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import random


# ═══════════════════════════════════════════════════════════════════
#  Diffusion 模型
# ═══════════════════════════════════════════════════════════════════

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimeSeriesDenoiser(nn.Module):
    """
    时间序列去噪网络
    """
    def __init__(self, seq_len, feature_dim, hidden_dim, context_dim, num_steps):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_steps),
            nn.Linear(num_steps, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 输入投影
        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        # 时序卷积
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 自注意力
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 上下文投影层
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm4 = nn.LayerNorm(hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x_t, t, context):
        """
        Args:
            x_t: (B, T, F) 噪声数据
            t: (B,) 时间步
            context: (B, C) 上下文向量
        Returns:
            noise: (B, T, F) 预测的噪声
        """
        # 时间步嵌入
        t_emb = self.time_mlp(t)  # (B, H)

        # 输入投影
        h = self.input_proj(x_t)  # (B, T, H)

        # 时序卷积
        h_conv = h.transpose(1, 2)  # (B, H, T)
        h_conv = F.gelu(self.conv1(h_conv))
        h_conv = h_conv + t_emb.unsqueeze(-1)
        h_conv = F.gelu(self.conv2(h_conv))
        h = h_conv.transpose(1, 2)
        h = self.norm1(h)

        # 自注意力
        h_attn, _ = self.self_attn(h, h, h)
        h = h + h_attn
        h = self.norm2(h)

        # 上下文交叉注意力
        context_expanded = self.context_proj(context).unsqueeze(1)  # (B, 1, H)
        h_cross, _ = self.cross_attn(h, context_expanded, context_expanded)
        h = h + h_cross
        h = self.norm3(h)

        # 输出
        noise = self.output_proj(h)
        return noise


class SimpleDDPM(nn.Module):
    """
    简化的 Diffusion 模型（修复版 DDIM 采样）
    """

    def __init__(self,
                 seq_len: int = 24,
                 feature_dim: int = 10,
                 hidden_dim: int = 64,
                 context_dim: int = 64,
                 num_steps: int = 50,
                 sampling_steps: int = 10):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.num_steps = num_steps
        self.sampling_steps = sampling_steps

        # 噪声调度
        self.register_noise_schedule()

        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(seq_len * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, context_dim)
        )

        # 去噪网络
        self.denoiser = TimeSeriesDenoiser(
            seq_len=seq_len,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            num_steps=num_steps
        )

    def register_noise_schedule(self, beta_start=1e-4, beta_end=0.02):
        """注册噪声调度（添加数值稳定性）"""
        betas = torch.linspace(beta_start, beta_end, self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 【修复】添加数值稳定性
        alphas_cumprod = torch.clamp(alphas_cumprod, min=1e-8, max=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=1e-8)))

    def forward_diffusion(self, x_0, t, noise=None):
        """前向加噪"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        return x_t, noise

    @torch.no_grad()
    def ddim_sample(self, context, n_samples=1, eta=0.0):
        """
        Args:
            context: 上下文向量
            n_samples: 每个上下文生成的样本数
            eta: 随机性参数 (0=确定性, 1=DDPM)
        """
        B = context.size(0)
        device = context.device

        # 扩展上下文
        context = context.repeat_interleave(n_samples, dim=0)

        # 从纯噪声开始
        x_t = torch.randn(B * n_samples, self.seq_len, self.feature_dim, device=device)

        # 选择采样时间步
        timesteps = torch.linspace(
            self.num_steps - 1, 0, self.sampling_steps, dtype=torch.long, device=device
        )

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B * n_samples,), t, device=device, dtype=torch.long)

            # 预测噪声
            predicted_noise = self.denoiser(x_t, t_batch, context)

            # 获取 alpha 值
            alpha_t = self.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # 【修复】正确的 DDIM 更新公式
            # 预测 x_0（添加数值稳定性）
            sqrt_alpha_t = torch.sqrt(torch.clamp(alpha_t, min=1e-8))
            sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1 - alpha_t, min=1e-8))

            x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t

            if i < len(timesteps) - 1:
                # 计算方差（可选的随机性）
                sigma_t = eta * torch.sqrt(
                    (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
                )

                # 方向性噪声
                direction = torch.sqrt(torch.clamp(
                    1 - alpha_t_prev - sigma_t ** 2, min=0
                )) * predicted_noise

                # 随机噪声（如果 eta > 0）
                noise = torch.randn_like(x_t) if eta > 0 else 0

                # 更新
                x_t = torch.sqrt(alpha_t_prev) * x_0_pred + direction + sigma_t * noise
            else:
                x_t = x_0_pred

        return x_t

    def training_loss(self, x_0, context):
        """训练损失"""
        B = x_0.size(0)
        device = x_0.device

        # 随机时间步
        t = torch.randint(0, self.num_steps, (B,), device=device)

        # 前向加噪
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_diffusion(x_0, t, noise)

        # 预测噪声
        predicted_noise = self.denoiser(x_t, t, context)

        # 损失
        loss = F.mse_loss(predicted_noise, noise)
        return loss


# ═══════════════════════════════════════════════════════════════════
# Diffusion 增强器
# ═══════════════════════════════════════════════════════════════════

class DiffusionAugmentor:
    """
    Diffusion 数据增强器
    """

    def __init__(self, model: SimpleDDPM, mode='faithful', adversarial_strength=0.3):
        """
        Args:
            model: SimpleDDPM 模型
            mode: 增强模式
                - 'faithful': 忠实复现原始数据分布
                - 'adversarial': 生成对抗样本
                - 'mixed': 混合模式
            adversarial_strength: 对抗强度 (0-1)
        """
        self.model = model
        self.mode = mode
        self.adversarial_strength = adversarial_strength
        self.device = next(model.parameters()).device

    def pretrain(self, X_train, epochs=10, batch_size=32, lr=1e-3, save_path=None):
        """
        预训练 Diffusion 模型
        Args:
            X_train: (N, T, F) 训练数据
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            save_path: 模型保存路径
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        n_samples = len(X_train)
        n_batches = (n_samples + batch_size - 1) // batch_size

        print(f"开始预训练 (epochs={epochs}, batch_size={batch_size}, samples={n_samples})")

        for epoch in range(epochs):
            epoch_loss = 0.0
            # 随机打乱
            indices = torch.randperm(n_samples)

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]

                # 获取批次数据
                x_batch = torch.FloatTensor(X_train[batch_indices]).to(self.device)

                # 编码上下文
                B, T, F = x_batch.shape
                with torch.no_grad():
                    context = self.model.context_encoder(x_batch.view(B, -1))

                # 训练损失
                loss = self.model.training_loss(x_batch, context)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # 保存模型
        if save_path:
            self.save_pretrained(save_path)
            print(f"✓ 模型已保存: {save_path}")

        self.model.eval()

    def save_pretrained(self, save_path):
        """保存预训练模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'seq_len': self.model.seq_len,
            'feature_dim': self.model.feature_dim,
            'num_steps': self.model.num_steps,
            'sampling_steps': self.model.sampling_steps,
            'mode': self.mode,
            'adversarial_strength': self.adversarial_strength
        }
        torch.save(checkpoint, save_path)

    def load_pretrained(self, load_path):
        """
        【修复】加载预训练模型（自动处理配置不匹配）

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(load_path):
            print(f"⚠️ 文件不存在: {load_path}")
            return False

        try:
            print(f"尝试加载模型: {load_path}")
            ckpt = torch.load(load_path, map_location=self.device, weights_only=False)

            # 【修复】检查配置是否匹配
            ckpt_feature_dim = ckpt.get('feature_dim', None)
            ckpt_seq_len = ckpt.get('seq_len', None)

            config_mismatch = False
            mismatch_reasons = []

            if ckpt_feature_dim != self.model.feature_dim:
                mismatch_reasons.append(
                    f"feature_dim: 期望 {self.model.feature_dim}, 实际 {ckpt_feature_dim}"
                )
                config_mismatch = True

            if ckpt_seq_len != self.model.seq_len:
                mismatch_reasons.append(
                    f"seq_len: 期望 {self.model.seq_len}, 实际 {ckpt_seq_len}"
                )
                config_mismatch = True

            if config_mismatch:
                print(f"⚠️ 配置不匹配:")
                for reason in mismatch_reasons:
                    print(f"  - {reason}")

                # 【关键修复】删除不匹配的checkpoint
                print(f"  → 删除不匹配的checkpoint")
                try:
                    os.remove(load_path)
                    print(f"  ✓ 已删除，需要重新训练")
                except Exception as e:
                    print(f"  ✗ 删除失败: {e}")

                return False  # 返回False表示需要重新训练

            # 配置匹配，加载模型
            self.model.load_state_dict(ckpt['model_state_dict'])

            # 恢复其他配置
            if 'mode' in ckpt:
                self.mode = ckpt['mode']
            if 'adversarial_strength' in ckpt:
                self.adversarial_strength = ckpt['adversarial_strength']

            print(f"✓ 成功加载模型 (feature_dim={self.model.feature_dim}, seq_len={self.model.seq_len})")
            self.model.eval()
            return True

        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            print(f"  → 尝试删除损坏的checkpoint")
            try:
                os.remove(load_path)
                print(f"  ✓ 已删除损坏的checkpoint")
            except:
                pass
            return False

    @torch.no_grad()
    def augment(self, X_batch, n_augment=None):
        """
        生成增强样本

        Args:
            X_batch: (B, T, F) 真实样本
            n_augment: 每个样本生成几个增强样本（默认=B）

        Returns:
            X_aug: (B*n, T, F) 增强样本
        """
        self.model.eval()
        B, T, F = X_batch.shape
        device = X_batch.device

        if n_augment is None:
            n_augment = B

        # 编码上下文
        context = self.model.context_encoder(X_batch.view(B, -1))

        # 根据模式生成样本
        if self.mode == 'faithful':
            # 忠实模式：低随机性
            X_aug = self.model.ddim_sample(context, n_samples=n_augment, eta=0.0)

        elif self.mode == 'adversarial':
            # 对抗模式：扰动上下文
            noise_scale = self.adversarial_strength
            context_perturbed = context + torch.randn_like(context) * noise_scale
            X_aug = self.model.ddim_sample(context_perturbed, n_samples=n_augment, eta=0.5)

        elif self.mode == 'mixed':
            # 混合模式：部分忠实，部分对抗
            n_faithful = n_augment // 2
            n_adversarial = n_augment - n_faithful

            X_faithful = self.model.ddim_sample(context, n_samples=n_faithful, eta=0.0)

            noise_scale = self.adversarial_strength
            context_perturbed = context + torch.randn_like(context) * noise_scale
            X_adversarial = self.model.ddim_sample(context_perturbed, n_samples=n_adversarial, eta=0.5)

            X_aug = torch.cat([X_faithful, X_adversarial], dim=0)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return X_aug

    def evaluate_quality(self, X_real, n_samples=100):
        """
        评估生成质量
        Args:
            X_real: (N, T, F) 真实样本
            n_samples: 生成样本数
        Returns:
            metrics: 质量指标字典
        """
        self.model.eval()

        # 随机选择样本
        n_real = min(n_samples, len(X_real))
        indices = np.random.choice(len(X_real), n_real, replace=False)
        X_batch = torch.FloatTensor(X_real[indices]).to(self.device)

        # 生成样本
        with torch.no_grad():
            X_gen = self.augment(X_batch, n_augment=1)

        X_real_np = X_batch.cpu().numpy()
        X_gen_np = X_gen.cpu().numpy()

        # 计算统计量
        metrics = {}

        # KS 统计量
        from scipy import stats
        ks_stats = []
        for f in range(X_real_np.shape[-1]):
            ks_stat, _ = stats.ks_2samp(
                X_real_np[:, :, f].flatten(),
                X_gen_np[:, :, f].flatten()
            )
            ks_stats.append(ks_stat)
        metrics['ks_statistic'] = np.mean(ks_stats)

        # 均值和标准差差异
        metrics['mean_diff'] = np.abs(X_real_np.mean() - X_gen_np.mean())
        metrics['std_diff'] = np.abs(X_real_np.std() - X_gen_np.std())

        # 自相关（可选）
        try:
            real_autocorr = np.mean([
                np.corrcoef(X_real_np[i, :-1, 0], X_real_np[i, 1:, 0])[0, 1]
                for i in range(min(100, len(X_real_np)))
            ])
            gen_autocorr = np.mean([
                np.corrcoef(X_gen_np[i, :-1, 0], X_gen_np[i, 1:, 0])[0, 1]
                for i in range(min(100, len(X_gen_np)))
            ])
            metrics['autocorr_diff'] = np.abs(real_autocorr - gen_autocorr)
        except:
            pass

        return metrics


# ═══════════════════════════════════════════════════════════════════
# 数据集类
# ═══════════════════════════════════════════════════════════════════

class ProceedDiffusionDataset(Dataset):
    """
    Proceed + Diffusion 数据集
    """

    def __init__(self,
                 Xw_all,
                 Yw_all,
                 diffusion_augmentor: Optional[DiffusionAugmentor] = None,
                 batch_size=32,
                 augment_prob=0.35,
                 augment_ratio=1.0,
                 use_concept_only=True):
        """
        Args:
            Xw_all: (N, T, F) 窗口数据
            Yw_all: (N, P) 标签
            diffusion_augmentor: Diffusion 增强器（可选）
            batch_size: 批次大小
            augment_prob: 使用增强的概率
            augment_ratio: 生成数据比例（相对于真实数据）
            use_concept_only: 是否只在概念编码时使用增强
        """
        self.Xw_all = Xw_all
        self.Yw_all = Yw_all
        self.augmentor = diffusion_augmentor
        self.batch_size = batch_size
        self.augment_prob = augment_prob
        self.augment_ratio = augment_ratio
        self.use_concept_only = use_concept_only

        self.n_total = len(Xw_all)
        self.n_batches = (self.n_total + batch_size - 1) // batch_size

    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.n_total)

        Xb = self.Xw_all[start_idx:end_idx]
        Yb = self.Yw_all[start_idx:end_idx]

        # 是否使用增强
        use_augment = (
                self.augmentor is not None and
                random.random() < self.augment_prob
        )

        if use_augment:
            try:
                n_real = len(Xb)
                n_gen = max(1, int(n_real * self.augment_ratio))

                # 生成增强样本
                Xb_tensor = torch.FloatTensor(Xb).to(next(self.augmentor.model.parameters()).device)
                Xb_aug = self.augmentor.augment(Xb_tensor, n_augment=n_gen).cpu().numpy()

                if self.use_concept_only:
                    # 只在概念编码时使用增强数据
                    # 返回混合数据 + 真实数据数量标记
                    Xb_mixed = np.concatenate([Xb, Xb_aug], axis=0)
                    return torch.FloatTensor(Xb_mixed), torch.FloatTensor(Yb), n_real
                else:
                    # 增强数据也参与训练
                    Yb_aug = Yb  # 复制标签（简化处理）
                    Xb_mixed = np.concatenate([Xb, Xb_aug], axis=0)
                    Yb_mixed = np.concatenate([Yb, Yb_aug], axis=0)
                    return torch.FloatTensor(Xb_mixed), torch.FloatTensor(Yb_mixed)

            except Exception as e:
                print(f"⚠️ 增强失败: {e}，使用原始数据")
                return torch.FloatTensor(Xb), torch.FloatTensor(Yb)

        return torch.FloatTensor(Xb), torch.FloatTensor(Yb)


# ═══════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════

def setup_diffusion_augmentor(Xw_train, seq_len, feature_dim,
                              mode='mixed', device='cuda',
                              pretrain_epochs=5, save_path=None):
    """
    便捷函数：初始化并预训练 Diffusion 增强器

    根据 load_pretrained 的返回值决定是否训练
    """
    print("\n" + "=" * 70)
    print("初始化 Diffusion 数据增强器")
    print("=" * 70)

    # 创建模型
    diffusion_model = SimpleDDPM(
        seq_len=seq_len,
        feature_dim=feature_dim,
        hidden_dim=64,
        context_dim=64,
        num_steps=50,
        sampling_steps=10
    ).to(device)

    # 创建增强器
    augmentor = DiffusionAugmentor(
        diffusion_model,
        mode=mode,
        adversarial_strength=0.3
    )

    # 预训练或加载
    need_training = True

    if save_path and os.path.exists(save_path):
        print(f"检测到已有模型: {save_path}")
        loaded = augmentor.load_pretrained(save_path)

        if loaded:
            print(f" 成功加载已有模型")
            need_training = False
        else:
            print(f" 加载失败或配置不匹配，将重新训练")
            need_training = True

    if need_training:
        print("开始预训练 Diffusion 模型...")
        augmentor.pretrain(
            Xw_train,
            epochs=pretrain_epochs,
            batch_size=32,
            lr=1e-3,
            save_path=save_path
        )

    # 评估质量
    print("\n评估生成质量...")
    metrics = augmentor.evaluate_quality(Xw_train[:100])
    if metrics:
        print(f"KS 统计量: {metrics.get('ks_statistic', 0):.4f}")
        print(f"均值差异: {metrics.get('mean_diff', 0):.4f}")
        print(f"标准差差异: {metrics.get('std_diff', 0):.4f}")
        if 'autocorr_diff' in metrics:
            print(f"自相关差异: {metrics['autocorr_diff']:.4f}")

    print(" Diffusion 增强器准备就绪！")
    print("=" * 70 + "\n")

    return augmentor


