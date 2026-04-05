
"""
快速超参数搜索（简化版）
========================================
用于快速测试和验证搜索流程
- 更少的参数
- 更快的速度
- 适合初次使用

搜索空间：
1. diffusion.mode: ['faithful', 'adversarial', 'mixed']
2. tcvae.pooling: ['mean', 'max', 'last', 'attention']
3. seq_len: [24, 48, 60, 96]
"""

import warnings
import numpy as np
import pandas as pd
import optuna
from data import load_real_data
from run_experiment import run_stock

warnings.filterwarnings('ignore')

# =================== 简化配置 ===================
import copy
from config import BASE_CFG
BASE_CFG = copy.deepcopy(BASE_CFG)
BASE_CFG["max_assets"] = 1  # 快速搜索只跑1只股票
# 注意：seq_len / tcvae.pooling / diffusion.mode 会在搜索时被覆盖


# =================== 精简搜索空间（仅搜索3个参数）===================
def simple_search_space(trial):
    """
    只搜索3个关键参数：
    1. diffusion.mode
    2. tcvae.pooling
    3. seq_len
    """
    # 搜索 diffusion.mode
    diffusion_mode = trial.suggest_categorical(
        "diffusion_mode",
        ["faithful", "adversarial", "mixed"]
    )

    # 搜索 tcvae.pooling
    tcvae_pooling = trial.suggest_categorical(
        "tcvae_pooling",
        ["mean", "max", "last", "attention"]
    )

    # 搜索 seq_len
    seq_len = trial.suggest_categorical(
        "seq_len",
        [24, 48, 60, 96]
    )

    return {
        "seq_len": seq_len,  # 更新seq_len
        "tcvae": {
            "latent_dim": 64,
            "hidden_dim": 128,
            "pooling": tcvae_pooling,  # 更新pooling
        },
        "diffusion": {
            "enable": True,
            "mode": diffusion_mode,  #更新mode
            "pretrain_epochs": 5,
            "augment_prob": 0.35,
            "augment_ratio": 1.0,
            "adversarial_strength": 0.3,
            "hidden_dim": 64,
            "context_dim": 64,
            "save_dir": "checkpoints/diffusion_models",
        },
    }


# =================== 目标函数 ===================
def objective(trial):
    """快速测试目标函数"""
    #  深拷贝BASE_CFG，避免污染原始配置
    import copy
    cfg = copy.deepcopy(BASE_CFG)

    # 获取搜索参数
    search_params = simple_search_space(trial)

    # 更新配置（使用深度更新）
    cfg["seq_len"] = search_params["seq_len"]
    cfg["tcvae"].update(search_params["tcvae"])
    cfg["diffusion"].update(search_params["diffusion"])

    # 同步tcvae_pooling到顶层
    cfg["tcvae_pooling"] = search_params["tcvae"]["pooling"]

    try:
        price, bench, panel_data = load_real_data(
            cfg["data_path"], cfg["max_assets"], cfg["max_days"]
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return 0.0

    da_scores = []
    for stock_id, df_one in panel_data.groupby("ts_code"):
        try:
            res = run_stock(df_one.sort_values("trade_date"), stock_id, cfg)
            if res is not None:
                da_scores.append(res["DA"])
        except Exception as e:
            print(f"股票 {stock_id} 失败: {e}")
            continue

    return float(np.mean(da_scores)) if da_scores else 0.0


# =================== 快速搜索 ===================
def quick_search(n_trials=10):
    """快速搜索（10次试验，约10-15分钟）"""

    print("=" * 60)
    print("快速超参数搜索")
    print("=" * 60)
    print(f"试验次数: {n_trials}")
    print(f"搜索参数:")
    print(f"  1. diffusion.mode: ['faithful', 'adversarial', 'mixed']")
    print(f"  2. tcvae.pooling: ['mean', 'max', 'last', 'attention']")
    print(f"  3. seq_len: [24, 48, 60, 96]")
    print(f"股票数量: {BASE_CFG['max_assets']} (加速测试)")
    print("=" * 60 + "\n")

    study = optuna.create_study(
        study_name="quick_search",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 输出结果
    print("\n" + "=" * 60)
    print("搜索完成！")
    print("=" * 60)
    print(f"\n最佳DA: {study.best_value:.4f}")
    print(f"\n最佳参数:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 保存结果
    df = study.trials_dataframe()
    df.to_csv("checkpoints/quick_search_results.csv", index=False)
    print(f"\n结果已保存: checkpoints/quick_search_results.csv")

    return study


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    study = quick_search(args.trials)
