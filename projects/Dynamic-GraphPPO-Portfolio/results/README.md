# 实验结果

## 指标说明

| 指标 | 说明 |
|---|---|
| Ann. Return | 年化收益率 |
| Sharpe | 夏普比率（无风险利率=0） |
| IR | 信息比率（超额收益 / 跟踪误差） |
| MDD | 最大回撤 |
| Turnover | 日均换手率 |

## 实验对比（测试集）

> 请在完成训练后将结果填入此表，便于 review 和复现

| 实验 | Ann. Return | Sharpe | IR | MDD | 配置文件 |
|---|---|---|---|---|---|
| iTransformer + MORL + Curriculum | — | — | — | — | `itransformer_morl.yaml` |
| Vanilla Baseline | — | — | — | — | `vanilla_baseline.yaml` |
| 消融：无课程学习 | — | — | — | — | `ablation_no_curriculum.yaml` |

## 如何生成 summary_table.csv

```python
import json, pandas as pd, glob

rows = []
for path in glob.glob('runs/*/results.json'):
    with open(path) as f:
        r = json.load(f)
    name = path.split('/')[1]
    test = r.get('test', {})
    rows.append({'experiment': name, **{k: test.get(k) for k in
        ['annual_return', 'sharpe_ratio', 'information_ratio', 'max_drawdown']}})

pd.DataFrame(rows).to_csv('results/summary_table.csv', index=False)
```
