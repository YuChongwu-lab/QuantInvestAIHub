# 分布漂移自适应股价预测系统

基于 **Proceed** 框架的股价预测系统，核心思想是在不重新训练模型的前提下，通过检测市场分布漂移并动态调制骨干网络，使模型持续适应市场机制的变化。

本项目是在 Zhao and Shen (KDD 2025) 的 PROCEED 框架基础上的改进版本，主要改进之处为：（1）使用TCVAE 概念编码器；（2）可选的 DDPM 数据增强，（3）可选带验证集。

致谢：本项目使用的数据CSI100_Top30.pkl来自 Tushare 金融数据接口，特此致谢。

---

## 核心设计

```
市场数据 → TCVAE 概念编码 → 漂移检测（delta）→ ProceedAdapter 门控调制 → PatchTST 预测
                                                                         ↑
                                                               在线微调（带验证回滚）
```

- **PatchTST Backbone**：将时序数据分块后经 Transformer 提取特征
- **TCVAE 概念编码器**：把任意长度时序压缩成固定维度的"市场状态向量"，支持 mean / max / last / attention 四种池化
- **ProceedAdapter**：接收源域与目标域的概念差异 `delta`，生成覆盖 Backbone 每一层的乘法门控参数，实现轻量级分布适应
- **扩散模型增强**：可选的 DDPM 数据增强，支持 faithful / adversarial / mixed 三种模式
- **在线微调**：每步推理前对预测头进行一步微调，带验证集回滚，防止负迁移

---

## 快速上手

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

数据文件为 `pkl` / `csv` / `xlsx` 格式，需包含以下列：

| 列名 | 类型 | 说明 |
|---|---|---|
| `trade_date` | datetime | 交易日期 |
| `ts_code` | str | 股票代码 |
| `close` | float | 收盘价 |
| 其他特征列 | float / int | 可选，系统自动区分连续/离散特征 |

在 `config.py` 中修改数据路径：

```python
"data_path": "your_data.pkl",
```

### 3. 运行实验

```bash
python run_experiment.py
```

在 `run_experiment.py` 的 `EXPERIMENT_CONFIG` 中调整实验参数：

```python
EXPERIMENT_CONFIG = {
    "tcvae_pooling": "attention",   # 'mean' | 'max' | 'last' | 'attention'
    "diffusion_enable": False,       # 是否启用扩散数据增强
    "diffusion_mode": "mixed",       # 'faithful' | 'adversarial' | 'mixed'
}
```

### 4. 超参数搜索

```bash
# 完整多目标搜索（MSE + DA 帕累托前沿）
python hypersearch_multi_obj.py --trials 30

# 搜索 + 生成可视化图
python hypersearch_multi_obj.py --trials 30 --visualize

# 快速验证流程（少量试验）
python quick_search.py --trials 5
```

### 5. 漂移分析

```bash
python drift_analysis.py
```

---

## 主要超参数

所有参数集中在 `config.py`，以下列出关键项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `seq_len` | 48 | 输入序列长度（须为 4 的倍数） |
| `pred_len` | 1 | 预测步长 |
| `tcvae.latent_dim` | 64 | 概念向量维度 |
| `tcvae.pooling` | `"mean"` | 池化策略 |
| `patchtst.d_model` | 256 | Transformer 隐藏维度 |
| `patchtst.nlayers` | 4 | Transformer 层数 |
| `adapter_train.concept_dim` | 64 | 须与 `tcvae.latent_dim` 相等 |
| `adapter_train.scale_mult` | 0.05 | 门控幅度上限（tanh × scale_mult） |
| `diffusion.mode` | `"faithful"` | 扩散增强模式 |
| `online_ft.lr` | 1e-4 | 在线微调学习率 |

---

## 评估指标

| 指标 | 说明 |
|---|---|
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| sMAPE | 对称平均绝对百分比误差 |
| DA | 方向准确率（Direction Accuracy），核心指标 |
| R² | 决定系数 |

---

## 注意事项

- `seq_len` 必须为 4 的倍数（TCVAE 编码器做 4 倍时序压缩）
- `tcvae.latent_dim` 必须与 `adapter_train.concept_dim` 保持相等
- 在线微调带验证回滚：若微调后验证 loss 不降，自动回滚参数
- 超参数搜索脚本默认 `max_assets=1`（单股票快速测试），正式实验在 `config.py` 中设置
