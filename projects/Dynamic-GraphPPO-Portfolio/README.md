# Dynamic-GraphPPO-Portfolio

**基于图神经网络与强化学习的量化投资组合管理**

将动态图 Transformer（E-DyGFormer）与 PPO 强化学习结合，通过课程学习（Curriculum Learning）和多目标强化学习（MORL）训练股票组合策略。

---

致谢：本项目使用的数据 CSI100_index.pkl、CSI100_stock_data_with_factors.pkl 均来自 Tushare 金融数据接口，特此致谢。

## 系统架构

```
数据（面板）
    │
    ▼
图数据集构建（时间边 + 相关边）
    │
    ▼
E-DyGFormer 编码器（iTransformer / Vanilla）
    │
    ▼
PolicyNet（Actor-Critic，Dirichlet 分布）
    │
    ├─── MORL 辅助任务（收益预测 IC + MSE）
    │
    ▼
PPO 训练（含课程学习三阶段）
    │
    ▼
EnhancedBacktestEngine（含交易成本 / 冲击成本）
```

---

## 快速开始

### 1. 环境安装

```bash
git clone https://github.com/yourname/Dynamic-GraphPPO-Portfolio.git
cd Dynamic-GraphPPO-Portfolio

conda env create -f environment.yml
conda activate dynamic-graphppo
pip install -e .
```

> **Windows 用户**：如遇 OpenMP 冲突报错，`scripts/train.py` 已在顶部设置
> `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` 自动处理。

### 2. 直接回测（使用预训练权重，无需训练）

```bash
python scripts/backtest.py \
    --config configs/experiments/itransformer_morl.yaml \
    --checkpoint pretrained/itransformer_morl.pth \
    --split both
```

或打开 `notebooks/01_quickstart.ipynb` 跟随交互演示。

### 3. 自行训练

```bash
# 主实验（iTransformer + MORL + 课程学习）
python scripts/train.py --config configs/experiments/itransformer_morl.yaml

# 对照实验（Vanilla Baseline）
python scripts/train.py --config configs/experiments/vanilla_baseline.yaml

# 消融：去掉课程学习
python scripts/train.py --config configs/experiments/ablation_no_curriculum.yaml
```

---

## 数据格式

### 股票面板数据（必须）

支持 `.pkl` / `.csv` / `.xlsx` 格式，须包含以下列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `trade_date` | datetime / str | 交易日期，如 `2020-01-02` |
| `ts_code` | str | 股票代码，如 `000001.SZ` |
| `close` | float | 后复权收盘价（建议用后复权，避免分红跳空影响收益计算） |
| 其余数值列 | float | 因子特征，全部自动纳入（如 `pe`, `pb`, `momentum_20`, `volatility` 等） |

**数据结构示例**（每行一条股票×日期的记录）：

```
trade_date    ts_code     close    pe      pb    momentum_20
2020-01-02   000001.SZ   14.21   12.3   1.05      0.032
2020-01-02   000002.SZ   29.80   18.6   2.31     -0.011
2020-01-03   000001.SZ   14.35   12.4   1.06      0.041
...
```

**注意事项**：
- 非数值列（字符串、类别）会被自动跳过，不影响运行
- 允许有缺失值，内部会做前向填充（`ffill`）+ 后向填充（`bfill`）
- 建议数据至少覆盖 **3 年以上**（默认按 7:2:1 划分训练/验证/测试集）
- 股票数量建议 **50 支以上**，过少会导致图结构稀疏，效果变差

**快速验证数据格式**：

```python
import pandas as pd
df = pd.read_pickle('your_data.pkl')
print(df.columns.tolist())          # 确认含 trade_date, ts_code, close
print(df.dtypes)                    # 确认各列类型
print(df[['trade_date','ts_code','close']].head())
print(f"股票数: {df['ts_code'].nunique()}, 交易日数: {df['trade_date'].nunique()}")
```

### 指数基准数据（可选，强烈推荐）

若不提供，系统会退而使用**股票池等权均值**作为基准（基准与策略候选池重叠，IR 会偏高，参考意义有限）。

推荐提供真实指数（如 CSI100、沪深300）：

| 列名 | 类型 | 说明 |
|---|---|---|
| `trade_date` | datetime / str | 交易日期 |
| `close` | float | 指数收盘点位 |

使用方式（在配置文件中指定，或直接传参）：

```yaml
# configs/default.yaml
index_data_path: "CSI100_index.pkl"   # 新增此行
```

或在代码中：

```python
from graphrl.data.loader import load_real_data
price, bench, panel = load_real_data(
    'stock_data.pkl',
    index_data_path='CSI100_index.pkl'   # 传入指数文件
)
```

---

## 项目结构

```
Dynamic-GraphPPO-Portfolio/
├── graphrl/               # 核心包（可 pip install -e . 安装）
│   ├── data/              # 数据加载与图数据集构建
│   ├── models/            # E-DyGFormer 模型
│   ├── rl/                # PolicyNet、PortfolioEnv、PPOTrainer
│   ├── backtest/          # 回测引擎
│   ├── curriculum/        # 课程学习配置
│   └── utils/             # 工具函数（指标、绘图、种子）
├── configs/               # YAML 配置文件
│   ├── default.yaml
│   ├── curriculum/
│   └── experiments/       # 可复现的实验配置
├── scripts/               # 命令行入口
│   ├── train.py
│   └── backtest.py
├── notebooks/             # Jupyter 演示
│   ├── 01_quickstart.ipynb
│   ├── 02_data_and_graph.ipynb
│   └── 03_results_analysis.ipynb
├── pretrained/            # 预训练权重（通过 Releases 分发）
├── results/               # 实验结果存档
└── runs/                  # 训练输出（.gitignore）
```

---

## 核心组件说明

### E-DyGFormer
边感知动态图 Transformer，支持两种编码模式：
- `itransformer`：在资产维度做注意力，捕捉截面关系（推荐）
- `vanilla`：标准 Transformer + 时序位置编码 + 边偏置

### 课程学习（Curriculum Learning）
三阶段渐进训练，从宽松约束逐步过渡到严格约束：

| 阶段 | max_pos | 交易成本 | 换手惩罚 |
|---|---|---|---|
| Warmup | 30% | 0.10% | 0.02 |
| Balance | 25% | 0.20% | 0.04 |
| Finetune | 20% | 0.30% | 0.05 |

### 多目标强化学习（MORL）
PPO 主任务（组合收益）+ 辅助任务（收益预测 IC），通过 PCGrad 梯度手术处理任务间冲突。

### Benchmark 说明
基准计算位于 `graphrl/data/loader.py`。优先级：
1. 外部指数文件（`index_data_path` 参数，推荐）
2. 完整股票池等权均值（未提供指数时的退而求其次）

基准计算在 `max_assets` 截断之前完成，确保基准代表完整股票池而非截断后的子集。

---

## License

MIT
