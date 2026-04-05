
"""
===============================================================================
多目标超参数搜索
===============================================================================
功能：
1. 同时优化 MSE (最小化) 和 DA (最大化)
2. 搜索参数：diffusion.mode, tcvae.pooling, seq_len
3. 自动生成帕累托前沿和推荐策略
4. 可选可视化功能
===============================================================================
"""
import warnings
import numpy as np
import pandas as pd
import optuna
import os
import sys
from pathlib import Path

# 确保checkpoints目录存在
Path("checkpoints").mkdir(exist_ok=True)

warnings.filterwarnings('ignore')

# =================== 导入主系统 ===================
try:
    from data import load_real_data
    from run_experiment import run_stock

    print("✓ 成功导入主系统模块")
except ImportError as e:
    print(f"导入失败: {e}")
    print("请确保 run_experiment.py 和 data.py 在当前目录")
    sys.exit(1)

# =================== 全局配置 ===================
import copy
from config import BASE_CFG
BASE_CFG = copy.deepcopy(BASE_CFG)
BASE_CFG["max_assets"] = 1  # 搜索时只跑1只股票，加快速度


# =================== 搜索空间定义 ===================
def simple_search_space(trial):
    """
    定义超参数搜索空间
    搜索参数：
    1. diffusion_mode: 数据增强策略
    2. tcvae_pooling: 概念编码池化方式
    3. seq_len: 输入序列长度
    """
    diffusion_mode = trial.suggest_categorical(
        "diffusion_mode",
        ["faithful", "adversarial", "mixed"]
    )

    tcvae_pooling = trial.suggest_categorical(
        "tcvae_pooling",
        ["mean", "max", "last", "attention"]
    )

    seq_len = trial.suggest_categorical(
        "seq_len",
        [24, 48, 60, 96]
    )

    return {
        "seq_len": seq_len,
        "tcvae": {
            "latent_dim": 64,
            "hidden_dim": 128,
            "pooling": tcvae_pooling,
        },
        "diffusion": {
            "enable": True,
            "mode": diffusion_mode,
            "pretrain_epochs": 5,
            "augment_prob": 0.35,
            "augment_ratio": 1.0,
            "adversarial_strength": 0.3,
            "hidden_dim": 64,
            "context_dim": 64,
            "save_dir": "checkpoints/diffusion_models",
        },
    }


# =================== 多目标函数 ===================
def objective_multi(trial):
    """
    多目标优化函数
    Returns:
        tuple: (MSE, -DA)
            - MSE: 均方误差（越小越好）
            - -DA: 负的方向准确率（用于统一最小化，原始DA越大越好）
    """
    import copy

    # 深拷贝配置，避免污染
    cfg = copy.deepcopy(BASE_CFG)

    # 获取当前试验的超参数
    search_params = simple_search_space(trial)

    # 更新配置
    cfg["seq_len"] = search_params["seq_len"]
    cfg["tcvae"].update(search_params["tcvae"])
    cfg["diffusion"].update(search_params["diffusion"])
    cfg["tcvae_pooling"] = search_params["tcvae"]["pooling"]  # 兼容主系统

    # 加载数据
    try:
        price, bench, panel_data = load_real_data(
            cfg["data_path"],
            cfg["max_assets"],
            cfg["max_days"]
        )
    except Exception as e:
        print(f"数据加载失败: {e}")
        return float('inf'), float('inf')

    # 收集各股票的MSE和DA
    mse_scores = []
    da_scores = []

    for stock_id, df_one in panel_data.groupby("ts_code"):
        try:
            # 运行单只股票
            res = run_stock(df_one.sort_values("trade_date"), stock_id, cfg)

            if res is not None:
                # 提取RMSE并计算MSE
                if "RMSE" in res and not np.isnan(res["RMSE"]):
                    mse = res["RMSE"] ** 2
                    mse_scores.append(mse)

                # 提取DA
                if "DA" in res and not np.isnan(res["DA"]):
                    da_scores.append(res["DA"])

        except Exception as e:
            print(f"股票 {stock_id} 运行失败: {e}")
            continue

    # 检查是否有有效结果
    if not mse_scores or not da_scores:
        print(f" Trial {trial.number}: 无有效结果")
        return float('inf'), float('inf')

    # 确保长度一致（取较短的）
    min_len = min(len(mse_scores), len(da_scores))
    mse_scores = mse_scores[:min_len]
    da_scores = da_scores[:min_len]

    # 计算平均值
    avg_mse = float(np.mean(mse_scores))
    avg_da = float(np.mean(da_scores))

    # 实时输出
    print(f" Trial {trial.number}: MSE={avg_mse:.6f}, DA={avg_da:.4f} "
          f"[{search_params['diffusion']['mode']}, "
          f"{search_params['tcvae']['pooling']}, "
          f"len={search_params['seq_len']}]")

    # 返回两个目标（Optuna统一最小化，所以DA取负）
    return avg_mse, -avg_da


# =================== 辅助函数：保存所有试验 ===================
def _save_all_trials(study, pareto_trials):
    """
    保存所有试验结果到CSV
    Args:
        study: Optuna study对象
        pareto_trials: 帕累托最优解列表
    """
    all_trials_data = []# 用于存储所有试验的数据
    pareto_numbers = {t.number for t in pareto_trials}# 获取帕累托最优解的试验编号

    for trial in study.trials:
        if trial.values is not None:
            all_trials_data.append({
                "trial_number": trial.number,
                "mse": trial.values[0],
                "da": -trial.values[1],  # 恢复正值
                "is_pareto": trial.number in pareto_numbers,# 是否为帕累托最优解
                "state": trial.state.name,
                **trial.params
            })

    if all_trials_data:
        all_df = pd.DataFrame(all_trials_data)
        all_df.to_csv("checkpoints/multi_objective_all_trials.csv", index=False)
        print(f"所有试验已保存: checkpoints/multi_objective_all_trials.csv")
        return all_df
    else:
        print("没有有效试验结果可保存")
        return None


# =================== 辅助函数：打印推荐策略 ===================
def _print_recommendations(pareto_trials):
    """打印三种推荐策略"""
    if not pareto_trials:# 如果没有帕累托最优解，直接返回
        return

    print("\n" + "=" * 70)
    print("推荐策略")
    print("=" * 70)

    # 策略1: MSE最优
    try:
        best_mse_trial = min(pareto_trials, key=lambda t: t.values[0])# 找到MSE最小的帕累托解
        print("\n【策略1】MSE最优 - 精确预测")
        print(f"  适用场景: 量化回测、精确预测")
        print(f"  MSE: {best_mse_trial.values[0]:.6f}")
        print(f"  DA:  {-best_mse_trial.values[1]:.4f}")
        print(f"  参数:")
        for k, v in best_mse_trial.params.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"\n无法计算MSE最优: {e}")

    # 策略2: DA最优
    try:
        best_da_trial = max(pareto_trials, key=lambda t: -t.values[1]) # 找到DA最大的帕累托解
        print("\n【策略2】DA最优 - 交易信号")
        print(f"  适用场景: 实盘交易、涨跌判断")
        print(f"  MSE: {best_da_trial.values[0]:.6f}")
        print(f"  DA:  {-best_da_trial.values[1]:.4f}")
        print(f"  参数:")
        for k, v in best_da_trial.params.items():
            print(f"    {k}: {v}")
    except Exception as e:
        print(f"\n无法计算DA最优: {e}")

    # 策略3: 平衡策略
    if len(pareto_trials) > 1: # 如果有多个帕累托解
        try:
            mse_values = np.array([t.values[0] for t in pareto_trials])
            da_values = np.array([-t.values[1] for t in pareto_trials])

            mse_range = mse_values.max() - mse_values.min()
            da_range = da_values.max() - da_values.min()

            if mse_range > 1e-8 and da_range > 1e-8: # 如果MSE和DA的差异足够大，则归一化
                mse_norm = (mse_values - mse_values.min()) / mse_range
                da_norm = (da_values.max() - da_values) / da_range

                # 综合得分（等权重）
                combined_score = mse_norm + da_norm # 综合评分
                best_balanced_idx = np.argmin(combined_score) # 选择综合得分最小的策略
                best_balanced_trial = pareto_trials[best_balanced_idx]

                print("\n【策略3】平衡策略 - 综合最优")
                print(f"  适用场景: 综合应用、研究报告")
                print(f"  MSE: {best_balanced_trial.values[0]:.6f}")
                print(f"  DA:  {-best_balanced_trial.values[1]:.4f}")
                print(f"  参数:")
                for k, v in best_balanced_trial.params.items():
                    print(f"    {k}: {v}")
            else:
                print("\n帕累托解差异太小，无法计算平衡策略")
        except Exception as e:
            print(f"\n平衡策略计算失败: {e}")
    elif len(pareto_trials) == 1:
        print("\n只有1个帕累托解，所有策略相同")


# =================== 辅助函数：打印可视化指南 ===================
def _print_visualization_guide():
    """打印可视化使用指南"""
    print("\n" + "=" * 70)
    print("可视化指南")
    print("=" * 70)
    print("""
方法1：运行时添加 --visualize 参数
    python hypersearch_multi_obj.py --trials 20 --visualize

方法2：在Python中调用
    from hypersearch_multi_obj import plot_pareto_front
    plot_pareto_front()

方法3：手动绘图
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('checkpoints/multi_objective_all_trials.csv')
    pareto = df[df['is_pareto'] == True]

    plt.scatter(df['mse'], df['da'], alpha=0.3, label='All')
    plt.scatter(pareto['mse'], pareto['da'], color='red', label='Pareto')
    plt.xlabel('MSE')
    plt.ylabel('DA')
    plt.legend()
    plt.show()
    """)


# =================== 可视化函数 ===================
def plot_pareto_front(csv_path="checkpoints/multi_objective_all_trials.csv",
                      save_path="checkpoints/pareto_front.png"):
    """
    绘制帕累托前沿图

    Args:
        csv_path: 试验结果CSV路径
        save_path: 图片保存路径
    """
    # 检查matplotlib
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 无显示环境
    except ImportError:
        print("matplotlib未安装，无法生成可视化")
        print("安装命令: pip install matplotlib")
        return

    # 检查文件
    if not os.path.exists(csv_path):
        print(f" 文件不存在: {csv_path}")
        return

    # 读取数据
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f" 读取CSV失败: {e}")
        return

    # 检查列
    required_cols = ['mse', 'da', 'is_pareto']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f" CSV缺少必要列: {missing_cols}")
        return

    # 分离帕累托解和非帕累托解
    pareto_df = df[df['is_pareto'] == True]
    non_pareto_df = df[df['is_pareto'] == False]

    if len(pareto_df) == 0:
        print("没有帕累托解可绘制")
        return

    # 创建图表
    plt.figure(figsize=(12, 7))

    # 绘制非帕累托解
    if len(non_pareto_df) > 0:
        plt.scatter(non_pareto_df['mse'], non_pareto_df['da'],
                    alpha=0.4, s=80, label='Non-Pareto Solutions',
                    color='skyblue', edgecolors='steelblue', linewidths=0.5)

    # 绘制帕累托解
    plt.scatter(pareto_df['mse'], pareto_df['da'],
                color='red', label='Pareto Front', s=200, marker='*',
                edgecolors='darkred', linewidths=1.5, zorder=5)

    # 连线帕累托前沿
    if len(pareto_df) > 1:
        pareto_sorted = pareto_df.sort_values('mse')
        plt.plot(pareto_sorted['mse'], pareto_sorted['da'],
                 'r--', alpha=0.5, linewidth=1.5, zorder=4,
                 label='Pareto Boundary')

    # 标注帕累托解
    for idx, row in pareto_df.iterrows():
        label = f"#{row.get('trial_number', idx)}"
        plt.annotate(label,
                     xy=(row['mse'], row['da']),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color='darkred', fontweight='bold')

    # 图表美化
    plt.xlabel('MSE (Mean Squared Error)', fontsize=12, fontweight='bold')
    plt.ylabel('DA (Directional Accuracy)', fontsize=12, fontweight='bold')
    plt.title('Pareto Front: MSE vs DA Trade-off',
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')

    # 添加统计信息
    stats_text = (
        f"Total Trials: {len(df)}\n"
        f"Pareto Solutions: {len(pareto_df)}\n"
        f"MSE Range: [{pareto_df['mse'].min():.6f}, {pareto_df['mse'].max():.6f}]\n"
        f"DA Range: [{pareto_df['da'].min():.4f}, {pareto_df['da'].max():.4f}]"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存图片
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n 帕累托前沿图已保存: {save_path}")
    except Exception as e:
        print(f" 图片保存失败: {e}")
    finally:
        plt.close()


# =================== 主搜索函数 ===================
def multi_objective_search(n_trials=20):
    """
    多目标超参数搜索主函数

    Args:
        n_trials: 搜索试验次数（建议≥20）

    Returns:
        tuple: (study, pareto_df)
            - study: Optuna study对象
            - pareto_df: 帕累托解DataFrame
    """
    print("=" * 70)
    print(" 多目标超参数搜索 (MSE + DA)")
    print("=" * 70)
    print(f"试验次数: {n_trials}")
    print(f"数据路径: {BASE_CFG['data_path']}")
    print(f"股票数量: {BASE_CFG['max_assets']}")
    print(f"数据天数: {BASE_CFG['max_days']}")
    print(f"\n搜索参数:")
    print(f"  1. diffusion.mode: ['faithful', 'adversarial', 'mixed']")
    print(f"  2. tcvae.pooling: ['mean', 'max', 'last', 'attention']")
    print(f"  3. seq_len: [24, 48, 60, 96]")
    print(f"\n优化目标:")
    print(f"  Objective 0: MSE (最小化)")
    print(f"  Objective 1: DA (最大化)")
    print("=" * 70 + "\n")

    # 创建多目标优化study
    study = optuna.create_study(
        study_name="multi_objective_search",
        directions=["minimize", "minimize"],  # 两个都最小化（DA已取负）
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    )

    # 开始优化
    try:
        study.optimize(objective_multi, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\n搜索被用户中断（Ctrl+C）")
    except Exception as e:
        print(f"\n 搜索过程出错: {e}")
        import traceback
        traceback.print_exc()

    # =================== 结果分析 ===================
    print("\n" + "=" * 70)
    print("搜索完成！")
    print("=" * 70)

    # 检查是否有有效试验
    valid_trials = [t for t in study.trials if t.values is not None]
    if not valid_trials:
        print("\n 没有成功完成的试验！")
        return study, None

    print(f"\n有效试验数: {len(valid_trials)} / {len(study.trials)}")

    # 获取帕累托前沿
    try:
        pareto_trials = study.best_trials
    except Exception as e:
        print(f"\n无法获取帕累托前沿: {e}")
        pareto_trials = []

    if not pareto_trials:
        print("\n 没有找到帕累托最优解")
        all_df = _save_all_trials(study, [])
        return study, all_df

    print(f"\n 帕累托最优解数量: {len(pareto_trials)}")
    print("\n" + "=" * 70)
    print("帕累托前沿解")
    print("=" * 70)

    # 保存帕累托解详情
    pareto_data = []
    for i, trial in enumerate(pareto_trials, 1):
        mse = trial.values[0]
        da = -trial.values[1]  # 恢复正值
        params = trial.params

        print(f"\n解 {i} (Trial #{trial.number}):")
        print(f"  MSE: {mse:.6f}")
        print(f"  DA:  {da:.4f}")
        print(f"  参数:")
        for k, v in params.items():
            print(f"    {k}: {v}")

        pareto_data.append({
            "solution_id": i,
            "trial_number": trial.number,
            "mse": mse,
            "da": da,
            **params
        })

    # 保存帕累托解
    pareto_df = pd.DataFrame(pareto_data)
    pareto_df.to_csv("checkpoints/pareto_solutions.csv", index=False)
    print(f"\n 帕累托解已保存: checkpoints/pareto_solutions.csv")

    # 保存所有试验
    all_df = _save_all_trials(study, pareto_trials)

    # 打印推荐策略
    _print_recommendations(pareto_trials)

    # 打印可视化指南
    _print_visualization_guide()

    return study, pareto_df


# =================== 主程序入口 ===================
def main():
    """主程序入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="多目标超参数搜索 (MSE + DA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  基础搜索（20次试验）:
    python hypersearch_multi_obj.py --trials 20

  快速测试（5次试验）:
    python hypersearch_multi_obj.py --trials 5

  完整搜索 + 可视化（30次试验）:
    python hypersearch_multi_obj.py --trials 30 --visualize

  自定义数据和股票数:
    python hypersearch_multi_obj.py --trials 20 --max_assets 20
        """
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="搜索试验次数（建议≥20，默认20）"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="是否生成帕累托前沿可视化图"
    )

    parser.add_argument(
        "--max_assets",
        type=int,
        default=None,
        help="最大股票数量（默认使用配置文件设置）"
    )

    parser.add_argument(
        "--max_days",
        type=int,
        default=None,
        help="最大数据天数（默认使用配置文件设置）"
    )

    args = parser.parse_args()

    # 更新配置
    if args.max_assets is not None:
        BASE_CFG["max_assets"] = args.max_assets
        print(f" 股票数量设置为: {args.max_assets}")

    if args.max_days is not None:
        BASE_CFG["max_days"] = args.max_days
        print(f" 数据天数设置为: {args.max_days}")

    # 运行搜索
    print(f"\n开始搜索...\n")
    study, pareto_df = multi_objective_search(args.trials)

    # 可视化
    if args.visualize:
        print("\n生成可视化图表...")
        plot_pareto_front()

    # 完成提示
    print("\n" + "=" * 70)
    print(" 所有任务完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. checkpoints/pareto_solutions.csv - 帕累托最优解")
    print("  2. checkpoints/multi_objective_all_trials.csv - 所有试验结果")
    if args.visualize:
        print("  3. checkpoints/pareto_front.png - 帕累托前沿图")
    print("\n下一步:")
    print("  1. 查看推荐策略，选择合适的参数配置")
    print("  2. 在主系统中使用选定的参数进行完整训练")
    print("  3. 如需可视化，运行: plot_pareto_front()")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
