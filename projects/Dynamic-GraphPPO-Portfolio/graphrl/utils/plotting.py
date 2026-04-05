# =============================================================================
# graphrl/utils/plotting.py —— 回测结果可视化
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

plt.rcParams["font.family"]       = ["SimHei"]
plt.rcParams['font.sans-serif']   = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_enhanced_results(
    results: Dict,
    portfolio_values: List,
    benchmark_levels: List,
    save_path: Optional[str] = None
):
    """
    绘制交易成本图和净值曲线

    Parameters
    ----------
    results          : EnhancedBacktestEngine.calculate_enhanced_metrics 的返回
    portfolio_values : 引擎的累积组合价值序列（长度 T+1）
    benchmark_levels : 引擎记录的基准净值/价格水平（长度 T）
    save_path        : 若非 None 则保存图像
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        fig.suptitle('Graph-RL 回测结果', fontsize=14, fontweight='bold')

        # 交易成本柱状图
        ax_cost = axes[0]
        cost_data = {
            '总交易成本':   float(results.get('total_transaction_costs', 0.0)),
            '平均交易成本': float(results.get('avg_transaction_cost',    0.0)),
            '成本率(%)':    float(results.get('transaction_cost_ratio',  0.0)) * 100.0,
        }
        bars = ax_cost.bar(list(cost_data.keys()), list(cost_data.values()))
        for b, v in zip(bars, cost_data.values()):
            ax_cost.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}',
                         ha='center', va='bottom')
        ax_cost.set_title('交易成本')
        ax_cost.grid(True, alpha=0.3)

        # 净值曲线
        ax_nav = axes[1]
        if (not portfolio_values) or (not benchmark_levels):
            ax_nav.text(0.5, 0.5, '无净值数据', ha='center', va='center',
                        transform=ax_nav.transAxes)
        else:
            vals = np.asarray(portfolio_values, dtype=float)  # [T+1]
            bl   = np.asarray(benchmark_levels,  dtype=float)  # [T]

            if len(vals) > len(bl):
                strat_nav = vals / vals[0]
                bench_nav = np.concatenate([[1.0], bl / bl[0]])
            elif len(vals) == len(bl):
                strat_nav = vals / vals[0]
                bench_nav = bl   / bl[0]
            else:
                strat_nav = vals / vals[0]
                bench_nav = bl[:len(vals)] / bl[0]

            L = min(len(strat_nav), len(bench_nav))
            strat_nav = strat_nav[:L]
            bench_nav = bench_nav[:L]

            ax_nav.plot(strat_nav, linewidth=2,   label='策略净值')
            ax_nav.plot(bench_nav, linewidth=1.8, label='基准净值')
            ax_nav.set_title('净值曲线')
            ax_nav.legend()
            ax_nav.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Plot] 保存图像：{save_path}")
        plt.show()

    except Exception as e:
        print(f"[Plot] 绘图失败: {e}")
