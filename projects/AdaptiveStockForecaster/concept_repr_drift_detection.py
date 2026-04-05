
"""
概念表示与分布漂移检测
功能：
1.提取概念表示，检查异常值，统计性描述
2.计算概念漂移，进行统计性描述
3.计算相邻概念相似度（余弦相似度），统计性描述
4.可视化：包含概念向量的分布、delta范数随时间变化、delta范数分布、相邻概念相似度、每个维度方差、PCA降维可视化

"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from TCVAE import TCVAEConceptEncoder

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("概念表示与分布漂移检测")
print("=" * 80)
print()
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ------------------- 基本配置 -------------------
import copy
from config import BASE_CFG
CFG = copy.deepcopy(BASE_CFG)
CFG["max_assets"] = 1  # 分析脚本只取1只股票

# =================== 1. 加载数据 ===================
print("[1/7] 加载数据...")
from data import load_real_data, make_windows_xy

try:
    price, bench, panel_data = load_real_data(
        CFG["data_path"],
        max_assets=2,
        max_days=1000
    )
    print(f"  数据加载成功")

    stock_id = panel_data['ts_code'].unique()[0]
    df_one = panel_data[panel_data['ts_code'] == stock_id].sort_values('trade_date')
    print(f"  测试股票: {stock_id}")
    print(f"  数据长度: {len(df_one)} 天")
except Exception as e:
    print(f"  数据加载失败: {e}")
    sys.exit(1)

# =================== 2. 特征工程 ===================
print("\n[2/7] 特征工程...")
seq_len = CFG["seq_len"]
pred_len = CFG["pred_len"]


feature_cols = [c for c in df_one.columns if c not in ["trade_date", "ts_code", "close"]]
cont_cols = [c for c in feature_cols if df_one[c].nunique() > 20]#如果不同的取值数大于20，则识别为连续特征
disc_cols = [c for c in feature_cols if c not in cont_cols]#除连续特征外，其他识别为离散特征

X_cont=df_one[cont_cols]
# 离散特征：检查常数，并过滤掉
if len(disc_cols) > 0:
    disc_data = []
    for col in disc_cols:
        if df_one[col].nunique() > 1:
            disc_data.append(df_one[col].fillna(0).values.reshape(-1, 1))
    X_disc = np.concatenate(disc_data, axis=1) if disc_data else np.zeros((len(df_one), 0))
else:
    X_disc = np.zeros((len(df_one), 0))

X_all = np.concatenate([X_cont, X_disc], axis=1)
y_full = np.log(df_one["close"]).diff().shift(-1).fillna(0).values

# 数据集划分
n = len(X_all)
if n < (seq_len + pred_len + 5):
    raise ValueError("数据太短，无法切窗口。")
n_train = int(n * 0.8)

# 标准化
mu = X_all[:n_train].mean(axis=0, keepdims=True)
std = X_all[:n_train].std(axis=0, keepdims=True) + 1e-6
X_all = (X_all - mu) / std

X_train_all = X_all[:n_train]
y_train_all = y_full[:n_train]
X_test_all = X_all[n_train:]
y_test_all = y_full[n_train:]

# 创建时间窗口
Xw_tr, Yw_tr = make_windows_xy(X_train_all, y_train_all, seq_len, pred_len)
Xw_test, Yw_test = make_windows_xy(X_test_all, y_test_all, seq_len, pred_len)


feature_dim = Xw_tr.shape[-1]

print(f"数据准备完成")
print(f"训练集: {len(Xw_tr)} 样本")
print(f"测试集: {len(Xw_test)} 样本")
print(f"特征维度: {feature_dim}")


# =================== 3. 初始化编码器 ===================
print("\n[3/7] 初始化概念编码器...")

latent_dim = 64
hidden_dim = 128

encoder = TCVAEConceptEncoder(
    feature_dim=feature_dim,
    latent_dim=latent_dim,
    hidden_dim=hidden_dim,
    pooling='attention'
).to(DEVICE)

print(f"特征维度: {feature_dim}")
print(f"概念维度: {latent_dim}")
print(f"隐藏层维度: {hidden_dim}")
print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")

# =================== 4. 提取概念向量 ===================
print("\n[4/7] 提取概念向量...")

encoder.eval()
with torch.no_grad():
    # 将所有窗口转换为tensor
    Xw_tensor = torch.tensor(Xw_tr, dtype=torch.float32).to(DEVICE)

    # 提取概念向量
    concepts = encoder(Xw_tensor).cpu().numpy()  # (N, latent_dim)

    print(f"概念向量形状: {concepts.shape}")

# 统计分析
print(f"\n  概念向量统计:")
print(f"均值: {concepts.mean():.6f}")
print(f"标准差: {concepts.std():.6f}")
print(f"最小值: {concepts.min():.6f}")
print(f"最大值: {concepts.max():.6f}")
print(f"中位数: {np.median(concepts):.6f}")

# 检查是否有异常值
zero_ratio = (np.abs(concepts) < 1e-6).mean()
inf_ratio = np.isinf(concepts).mean()
nan_ratio = np.isnan(concepts).mean()

print(f"\n  异常值检查:")
print(f"近零值比例: {zero_ratio:.2%}")
print(f"Inf值比例: {inf_ratio:.2%}")
print(f"NaN值比例: {nan_ratio:.2%}")

# =================== 5. 计算概念漂移 ===================
print("\n[5/7] 计算概念漂移（delta）...")

# 模拟Proceed的滚动锚点
window_size = 10  # 锚点窗口大小
deltas = []
delta_norms = []

for i in range(window_size, len(concepts)):
    # 当前概念
    c_t = concepts[i]
    # 锚点：前window_size个的平均
    c_anchor = concepts[i - window_size:i].mean(axis=0)
    # 漂移
    delta = c_t - c_anchor
    deltas.append(delta)
    delta_norms.append(np.linalg.norm(delta))

deltas = np.array(deltas)
delta_norms = np.array(delta_norms)

print(f" 计算了 {len(deltas)} 个漂移向量")
print(f"\n  概念漂移统计:")
print(f" Delta均值: {deltas.mean():.6f}")
print(f" Delta标准差: {deltas.std():.6f}")
print(f" Delta范数均值: {delta_norms.mean():.6f}")
print(f" Delta范数标准差: {delta_norms.std():.6f}")
print(f" Delta范数最大值: {delta_norms.max():.6f}")

# =================== 6. 概念向量的区分度 ===================
print("\n[6/7] 分析概念向量区分度...")

# 计算相邻概念的余弦相似度
similarities = []
for i in range(len(concepts) - 1):
    sim = cosine_similarity(concepts[i:i + 1], concepts[i + 1:i + 2])[0, 0]
    similarities.append(sim)

similarities = np.array(similarities)

print(f" 相邻概念相似度:")
print(f" 均值: {similarities.mean():.6f}")
print(f" 标准差: {similarities.std():.6f}")
print(f"  最小值: {similarities.min():.6f}")
print(f"  最大值: {similarities.max():.6f}")
print(f"  中位数: {np.median(similarities):.6f}")

# 计算概念向量的方差（每个维度）
per_dim_var = concepts.var(axis=0)
print(f"\n  每维度方差:")
print(f"    均值: {per_dim_var.mean():.4f}")
print(f"    最小值: {per_dim_var.min():.4f}")
print(f"    最大值: {per_dim_var.max():.4f}")

# 检查是否有维度方差过小（未激活）
inactive_dims = (per_dim_var < 0.001).sum()
if inactive_dims > 0:
    print(f"警告: {inactive_dims}/{latent_dim} 个维度方差<0.001（可能未激活）")
else:
    print(f" 所有维度都被激活")

# =================== 7. 可视化 ===================
print("\n[7/7] 生成可视化...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'概念表示与分布漂移检测 - {stock_id}', fontsize=16, fontweight='bold')

    # 图1: 概念向量的分布（第一维和第二维）
    ax = axes[0, 0]
    ax.scatter(concepts[:, 0], concepts[:, 1], c=np.arange(len(concepts)),
               cmap='viridis', alpha=0.6, s=20)
    ax.set_xlabel('Concept Dim 1')
    ax.set_ylabel('Concept Dim 2')
    ax.set_title('概念空间分布（前2维）')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('时间顺序')

    # 图2: Delta范数的时间序列
    ax = axes[0, 1]
    ax.plot(delta_norms, linewidth=1, alpha=0.7)
    ax.axhline(delta_norms.mean(), color='r', linestyle='--',
               label=f'均值={delta_norms.mean():.3f}')
    ax.set_xlabel('时间步')
    ax.set_ylabel('Delta范数')
    ax.set_title('概念漂移幅度')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图3: Delta范数的分布
    ax = axes[0, 2]
    ax.hist(delta_norms, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(delta_norms.mean(), color='r', linestyle='--',
               label=f'均值={delta_norms.mean():.3f}')
    ax.set_xlabel('Delta范数')
    ax.set_ylabel('频数')
    ax.set_title('概念漂移分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图4: 相邻概念相似度
    ax = axes[1, 0]
    ax.plot(similarities, linewidth=1, alpha=0.7)
    ax.axhline(similarities.mean(), color='r', linestyle='--',
               label=f'均值={similarities.mean():.3f}')
    ax.set_xlabel('时间步')
    ax.set_ylabel('余弦相似度')
    ax.set_title('相邻概念相似度')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图5: 每维度方差
    ax = axes[1, 1]
    ax.bar(range(len(per_dim_var)), per_dim_var, alpha=0.7)
    ax.axhline(0.001, color='r', linestyle='--', label='阈值=0.001')
    ax.set_xlabel('概念维度')
    ax.set_ylabel('方差')
    ax.set_title(f'每维度方差（{inactive_dims}个低方差维度）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 图6: PCA降维可视化
    ax = axes[1, 2]
    if len(concepts) > 2:
        pca = PCA(n_components=2)
        concepts_2d = pca.fit_transform(concepts)
        scatter = ax.scatter(concepts_2d[:, 0], concepts_2d[:, 1],
                             c=np.arange(len(concepts)), cmap='viridis',
                             alpha=0.6, s=20)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_title('PCA降维可视化')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='时间顺序')

    plt.tight_layout()

    # 保存图片
    save_path = 'checkpoints/concept_encoder_verification.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 可视化已保存: {save_path}")

    # 如果在notebook环境，显示图片
    try:
        plt.show()
    except:
        pass

except Exception as e:
    print(f"  可视化失败: {e}")

