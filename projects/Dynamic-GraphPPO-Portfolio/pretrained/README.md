# 预训练权重说明

## 可用权重

| 文件名 | 配置 | 训练数据范围 | 资产池 | 验证集 Sharpe | 测试集 Sharpe |
|---|---|---|---|---|---|
| `itransformer_morl.pth` | `configs/experiments/itransformer_morl.yaml` | 见下 | CSI100（前190支） | — | — |
| `vanilla_baseline.pth` | `configs/experiments/vanilla_baseline.yaml` | 见下 | CSI100（前190支） | — | — |

> 训练数据时间范围与测试集划分取决于你的 `CSI100_stock_data_with_factors.pkl` 文件，
> 权重本身不含数据。按 70/20/10 划分：最后 10% 为测试集，倒数 10%~30% 为验证集。

## 使用方式

```bash
# 加载预训练权重直接回测
python scripts/backtest.py \
    --config configs/experiments/itransformer_morl.yaml \
    --checkpoint pretrained/itransformer_morl.pth \
    --split both \
    --output_dir runs/pretrained_eval
```

或在 Notebook 中：

```python
import torch
from graphrl.models.edygformer import E_DyGFormer
from graphrl.rl.policy import PolicyNet, PolicyModelAdapter

encoder = E_DyGFormer(node_feat_dim=..., num_assets=..., seq_len=60,
                       encoder_type='itransformer', hidden_dim=128, num_layers=4)
policy  = PolicyNet(encoder, hidden=256, alpha_scale=0.5, temperature=10)
policy.load_state_dict(torch.load('pretrained/itransformer_morl.pth', map_location='cpu'))
policy.eval()
```

## 注意事项

- 权重文件通过 GitHub Releases 分发，不直接提交到仓库（见 `.gitignore`）
- 如需自行训练，请参考 `notebooks/01_quickstart.ipynb`
