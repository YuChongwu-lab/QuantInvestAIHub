# config.py
"""
统一配置文件

使用说明：
  from config import BASE_CFG
  import copy
  cfg = copy.deepcopy(BASE_CFG)   # 必须深拷贝，避免各模块互相污染

"""

BASE_CFG = {
    "seq_len": 48,
    "pred_len": 1,

    "pretrain": {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 10,
    },

    "adapter_train": {
        "lr": 5e-4,
        "batch_size": 64,
        "epochs": 10,
        "bottleneck_dim": 96,
        "concept_dim": 64,
        "d_model": 128,
        "reg": 1e-4,
        "scale_mult": 0.05,
        "synth_prob": 0.35,
        "synth_modes": ["noise", "scale", "shift"],
        "synth_strength": 0.02,
    },

    "tcvae": {
        "latent_dim": 64,      # 潜在空间维度（概念向量维度）
        "hidden_dim": 128,     # 隐藏层维度
        "pooling": "mean",     # 池化策略：'mean', 'max', 'last', 'attention'
    },

    "diffusion": {
        "enable": True,                              # 设为 False 则不增强
        "mode": "faithful",                          # 'faithful', 'adversarial', 'mixed'
        "pretrain_epochs": 5,
        "augment_prob": 0.35,
        "augment_ratio": 1.0,
        "adversarial_strength": 0.3,
        "hidden_dim": 64,
        "context_dim": 64,
        "save_dir": "checkpoints/diffusion_models",
    },

    "patchtst": {
        "d_model": 256,
        "nhead": 8,
        "nlayers": 4,
        "dropout": 0.1,
        "patch_len": 6,
        "stride": 3,
        "head_hidden": 128,
    },

    "online_ft": {
        "enable": True,
        "lr": 1e-4,
        "steps": 1,
        "batch_size": 8,
        "use_train_tail": 32,
        "finetune_layers": ["head_in", "fc_out"],
    },

    "loss": {
        "type": "directional",  # 使用 DirectionalLoss
        "alpha": 1.0,           # 方向正确时的权重
        "beta": 2.0,            # 方向错误时的权重（加重惩罚）
        "use_soft": True,       # 使用软方向匹配
    },

    "data_path": "CSI100_Top30.pkl",
    "max_assets": 2,
    "max_days": 1000,
    "seed": 42,
}
