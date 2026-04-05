"""
分布漂移自适应股价预测系统 - 实验入口
═══════════════════════════════════════════════════════════════════════════
功能：整合预测系统
注意：
1.seq_len 必须为4的倍数
2.在线微调使用DirectionalLoss
"""

import os, random, copy, collections
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# =================== 导入自定义模块 ===================
from TCVAE import TCVAEConceptEncoder
from losses import DirectionalLoss
from proceed_diffusion_augmentation import setup_diffusion_augmentor

from train_utils import (
    pretrain_backbone_with_validation,
    train_adapter_with_validation,
    online_finetune_with_validation
)
from proceed_diffusion_augmentation_with_validation import train_adapter_proceed_with_diffusion_validated

from data import load_real_data, make_windows_xy, set_seed
from models import PatchTSTBackbone, ProceedAdapter

# ------------------- 基本配置 -------------------
from config import BASE_CFG
CFG = copy.deepcopy(BASE_CFG)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(CFG["diffusion"]["save_dir"], exist_ok=True)


# ---------------- 可复现性 & 设备 ----------------
set_seed(CFG["seed"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============== 在线微调 ==============
def build_online_model(base_model: PatchTSTBackbone, finetune_layers: List[str]):
    online_model = copy.deepcopy(base_model).to(DEVICE)
    for p in online_model.parameters(): p.requires_grad = False
    name2layer = {
        "head_in": online_model.head_in,
        "fc_out": online_model.fc_out,
    }
    params = []
    for k in finetune_layers:
        if k in name2layer:
            for p in name2layer[k].parameters():
                p.requires_grad = True
                params.append(p)
    return online_model, params


def online_finetune_step(online_model, params, X_ft, Y_ft, lr=1e-4, steps=1, batch_size=8):
    """在线微调（使用DirectionalLoss）"""
    if len(X_ft) == 0:
        return

    opt = torch.optim.Adam(params, lr=lr)

    # 使用DirectionalLoss
    loss_cfg = CFG.get("loss", {})
    loss_fn = DirectionalLoss(
        alpha=loss_cfg.get("alpha", 1.0),
        beta=loss_cfg.get("beta", 2.0),
        use_soft=loss_cfg.get("use_soft", True)
    )

    ds = TensorDataset(torch.tensor(X_ft), torch.tensor(Y_ft))
    dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    online_model.train()
    for _ in range(steps):
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            pred = online_model(xb)
            loss = loss_fn(pred.view(pred.size(0), -1), yb.view(yb.size(0), -1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    online_model.eval()


# ============== 单股票运行 ==============
def run_stock(df_one, stock_id, cfg):
    """
    运行单个股票的完整流程
    Args:
        df_one: 单个股票的DataFrame
        stock_id: 股票ID
        cfg: 配置字典
    Returns:
        metrics_dict: 包含所有指标和训练历史的字典
    """
    # =================== 初始化历史记录变量（防止 UnboundLocalError） ===================
    history_pretrain = {'best_epoch': 0, 'best_val_loss': 0.0, 'best_val_da': 0.0}
    history_adapter = {'best_epoch': 0, 'best_val_loss': 0.0, 'best_val_da': 0.0}

    print("\n" + "=" * 80)
    print(f"开始处理股票: {stock_id}")
    print("=" * 80)

    seq_len = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    # =================== 1.数据预处理 ===================
    print("\n阶段1: 数据预处理")
    print("-" * 80)

    # 特征工程
    feature_cols = [c for c in df_one.columns if c not in ["trade_date", "ts_code", "close"]]
    cont_cols = [c for c in feature_cols if df_one[c].nunique() > 20]#如果不同的取值数大于20，则识别为连续特征
    disc_cols = [c for c in feature_cols if c not in cont_cols]#除连续特征外，其他识别为离散特征

    X_cont=df_one[cont_cols] #连续特征取原始值
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

    if len(Xw_tr) == 0 or len(Xw_test) == 0:
        print("Not enough data for windows.")
        return None

    feature_dim = Xw_tr.shape[-1]

    print(f"数据准备完成")
    print(f"训练集: {len(Xw_tr)} 样本")
    print(f"测试集: {len(Xw_test)} 样本")
    print(f"特征维度: {feature_dim}")

    # =================== 2创建模型 ===================
    print("阶段2: 创建模型")
    print("-" * 80)

    # 创建损失函数
    loss_fn = nn.MSELoss()

    # 创建 PatchTST Backbone
    P = cfg["patchtst"]
    base_model = PatchTSTBackbone(
        seq_len=seq_len, feature_dim=feature_dim, pred_len=pred_len,
        d_model=P["d_model"], nhead=P["nhead"], nlayers=P["nlayers"],
        dropout=P["dropout"], patch_len=P["patch_len"], stride=P["stride"],
        head_hidden=P["head_hidden"]
    ).to(DEVICE)

    print(f"PatchTST Backbone 创建成功")
    print(f"参数量: {sum(p.numel() for p in base_model.parameters()):,}")

    # =================== 3.预训练PatchTST Backbone（带Validation） ===================
    print("阶段3: 预训练 PatchTST Backbone（带Validation）")
    print("-" * 80)

    pretrain_save_path = f'checkpoints/best_pretrain_{stock_id}.pth'

    try:
        history_pretrain = pretrain_backbone_with_validation(
            model=base_model,
            X_train=Xw_tr,
            Y_train=Yw_tr,
            cfg=cfg["pretrain"],
            loss_fn=loss_fn,
            device=DEVICE,
            val_ratio=0.2,  # 20% 用作验证集
            save_path=pretrain_save_path
        )

        print("预训练完成！")
        print(f"最佳Epoch: {history_pretrain['best_epoch'] + 1}")
        print(f"最佳Val Loss: {history_pretrain['best_val_loss']:.6f}")
        if 'best_val_metric' in history_pretrain:
            print(f"最佳Val DA: {history_pretrain['best_val_metric']:.4f}")

    except Exception as e:
        print(f"\n预训练失败: {e}")
        import traceback
        traceback.print_exc()
        history_pretrain = None

    # =================== 4.训练 Adapter（带Validation） ====================
    print("阶段4: 训练概念编码器和适配器（带Validation）")
    print("-" * 80)

    A = cfg["adapter_train"]

    tcvae_cfg = cfg.get("tcvae", {
        "latent_dim": 64,
        "hidden_dim": 128,
        "pooling": "mean"
    })
    pooling_mode = cfg.get("tcvae_pooling", tcvae_cfg.get("pooling", "mean"))

    print(f"\nTCVAE配置:")
    print(f"  - feature_dim: {feature_dim}")
    print(f"  - latent_dim: {tcvae_cfg.get('latent_dim', 64)}")
    print(f"  - hidden_dim: {tcvae_cfg.get('hidden_dim', 128)}")
    print(f"  - pooling: {pooling_mode}")

    # 使用TCVAE编码器，池化策略可选('mean', 'max', 'attention', 'last')
    enc_src = TCVAEConceptEncoder(
        feature_dim,
        latent_dim=tcvae_cfg.get("latent_dim", 64),
        hidden_dim=tcvae_cfg.get("hidden_dim", 128),
        pooling=pooling_mode
    )
    enc_tgt = TCVAEConceptEncoder(
        feature_dim,
        latent_dim=tcvae_cfg.get("latent_dim", 64),
        hidden_dim=tcvae_cfg.get("hidden_dim", 128),
        pooling=pooling_mode
    )

    assert tcvae_cfg.get("latent_dim", 64) == A["concept_dim"], \
        f"TCVAE的latent_dim({tcvae_cfg.get('latent_dim', 64)}) 必须等于 Adapter的concept_dim({A['concept_dim']})"

    adapter = ProceedAdapter(
        concept_dim=A["concept_dim"],
        bottleneck_dim=A["bottleneck_dim"],
        layer_meta=base_model.get_layer_meta(),
        scale_mult=A.get("scale_mult", 0.05)
    )

    # 选择增强方法
    diff_cfg = cfg.get("diffusion", {})

    # 判断是否使用 Diffusion 增强
    use_diffusion = diff_cfg.get("enable", True)

    try:
        if use_diffusion:
            print(f"\n使用 Diffusion 数据增强")
            print(f"模式: {diff_cfg.get('mode', 'mixed')}")
            print(f"增强概率: {diff_cfg.get('augment_prob', 0.35)}")

            diffusion_save_path = os.path.join(
                diff_cfg.get("save_dir", "checkpoints"),
                f"diffusion_{stock_id}.pth"
            )

            augmentor = setup_diffusion_augmentor(
                Xw_tr,
                seq_len=cfg["seq_len"],
                feature_dim=feature_dim,
                mode=diff_cfg.get("mode", "mixed"),
                device=DEVICE,
                pretrain_epochs=diff_cfg.get("pretrain_epochs", 5),
                save_path=diffusion_save_path
            )

            A_modified = A.copy()
            A_modified["synth_prob"] = diff_cfg.get("augment_prob", 0.35)

            history_adapter = train_adapter_proceed_with_diffusion_validated(
                enc_src, enc_tgt, adapter, base_model,
                Xw_tr, Yw_tr, A_modified, augmentor, DEVICE,
                val_ratio=0.2,
                save_best_path='checkpoints/best.pth'
            )

            print(f"最佳DA: {history_adapter['best_val_da']:.4f}")

        else:
            print(f"\n不使用数据增强（但带Validation）")
            print(f"增强概率: {A.get('synth_prob', 0.35)}")
            print(f"增强模式: {A.get('synth_modes', [])}")

            # 使用带validation的adapter训练，但是没有数据增强，
            history_adapter = train_adapter_with_validation(
                enc_src=enc_src,
                enc_tgt=enc_tgt,
                adapter=adapter,
                base_model=base_model,
                X_train=Xw_tr,
                Y_train=Yw_tr,
                cfg=A,
                loss_fn=loss_fn,
                device=DEVICE,
                val_ratio=0.2,
                use_diffusion=False
            )

            if history_adapter:
                print("\n" + "=" * 80)
                print("Adapter训练完成！")
                print("=" * 80)
                print(f"最佳Epoch: {history_adapter['best_epoch'] + 1}")
                print(f"最佳Val Loss: {history_adapter['best_val_loss']:.6f}")

    except Exception as e:
        print(f"\nAdapter训练失败: {e}")
        import traceback
        traceback.print_exc()
        history_adapter = None

    # =================== 在线微调和推理（带Validation） ===================
    print("\n" + "=" * 80)
    print("阶段6: 在线测试和推理")
    print("=" * 80)

    online_cfg = cfg["online_ft"]
    online_model, ft_params = build_online_model(base_model, online_cfg["finetune_layers"])

    preds = []
    trues = []
    rolling_anchor = collections.deque(maxlen=1)
    H = pred_len

    # 统计在线微调的接受/拒绝次数
    finetune_accepted = 0
    finetune_rejected = 0
    finetune_skipped = 0

    with torch.no_grad():
        xi_anchor = torch.tensor(Xw_tr[-1:], dtype=torch.float32).to(DEVICE)
        c_anchor = enc_src(xi_anchor).detach()
        rolling_anchor.append(c_anchor.squeeze(0))

    print(f"开始在线测试（共 {len(Xw_test)} 步）...")

    for i in range(len(Xw_test)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"进度: {i + 1}/{len(Xw_test)}", end='\r')

        xj = torch.tensor(Xw_test[i:i + 1], dtype=torch.float32).to(DEVICE)
        y_true = Yw_test[i]

        # 在线微调（带验证）
        if online_cfg["enable"]:
            X_ft_list, Y_ft_list = [], []
            safe_idx = i - H
            if safe_idx >= 0:
                X_ft_list.append(Xw_test[safe_idx:safe_idx + 1])
                Y_ft_list.append(Yw_test[safe_idx:safe_idx + 1])
            ut = online_cfg.get("use_train_tail", 0)
            if ut > 0:
                X_ft_list.append(Xw_tr[-min(ut, len(Xw_tr)):])
                Y_ft_list.append(Yw_tr[-min(ut, len(Yw_tr)):])

            if len(X_ft_list) > 0:
                X_ft = np.concatenate(X_ft_list, axis=0)
                Y_ft = np.concatenate(Y_ft_list, axis=0)

                # 使用带验证的微调
                accepted = online_finetune_with_validation(
                    online_model=online_model,
                    params=ft_params,
                    X_ft=X_ft,
                    Y_ft=Y_ft,
                    loss_fn=loss_fn,
                    lr=online_cfg["lr"],
                    steps=online_cfg["steps"],
                    batch_size=online_cfg["batch_size"],
                    val_ratio=0.2,
                    min_samples=10,
                    device=DEVICE
                )

                if accepted:
                    finetune_accepted += 1
                elif accepted is False:
                    finetune_rejected += 1
                else:
                    finetune_skipped += 1
            else:
                finetune_skipped += 1

        # 推理
        with torch.no_grad():
            if i >= (H - 1):
                xi_th = torch.tensor(Xw_test[i - (H - 1):i - (H - 1) + 1],
                                     dtype=torch.float32).to(DEVICE)
                c_th = enc_src(xi_th).detach().squeeze(0)
                rolling_anchor.clear()
                rolling_anchor.append(c_th)

            cj = enc_tgt(xj)
            ci = rolling_anchor[-1].unsqueeze(0).to(DEVICE)
            delta = cj - ci
            mod = adapter(delta)
            y_hat = online_model(xj, mod=mod).cpu().numpy().flatten()

        preds.append(y_hat)
        trues.append(y_true)

    print()  # 换行

    if online_cfg["enable"]:
        print(f"\n在线微调统计:")
        print(f"接受: {finetune_accepted}")
        print(f"拒绝: {finetune_rejected}")
        print(f"跳过: {finetune_skipped}")
        accept_rate = finetune_accepted / max(1, finetune_accepted + finetune_rejected) * 100
        print(f"接受率: {accept_rate:.1f}%")

    # =================== 指标计算 ===================
    print("\n" + "=" * 80)
    print("阶段7: 指标计算")
    print("=" * 80)

    preds_arr = np.array(preds).reshape(len(preds), -1)
    trues_arr = np.array(trues).reshape(len(trues), -1)
    Hh = preds_arr.shape[1]

    mae_all = float(np.mean(np.abs(preds_arr - trues_arr)))
    rmse_all = float(np.sqrt(np.mean((preds_arr - trues_arr) ** 2)))

    mae_h = np.mean(np.abs(preds_arr - trues_arr), axis=0)
    rmse_h = np.sqrt(np.mean((preds_arr - trues_arr) ** 2, axis=0))

    eps = 1e-8
    smape_h = 2.0 * np.mean(
        np.abs(preds_arr - trues_arr) / (np.abs(preds_arr) + np.abs(trues_arr) + eps),
        axis=0
    ) * 100.0
    smape_all = float(np.mean(
        2.0 * np.abs(preds_arr - trues_arr) / (np.abs(preds_arr) + np.abs(trues_arr) + eps)
    ) * 100.0)

    da = float(np.mean(np.sign(preds_arr) == np.sign(trues_arr)))
    r2 = float(1 - np.sum((preds_arr - trues_arr) ** 2) /
               (np.sum((trues_arr - trues_arr.mean()) ** 2) + 1e-12))

    # =================== 打印最终结果 ===================
    method_tag = "Diffusion" if diff_cfg.get("enable", False) else "No Augment "

    print("\n" + "=" * 80)
    print(f"最终结果 - {stock_id}")
    print("=" * 80)
    print(f"方法: {method_tag}增强 + DirectionalLoss")
    print(f"训练框架: Validation Framework (Early Stopping)")
    print()
    print(f"整体指标:")
    print(f"MAE:   {mae_all:.6f}")
    print(f"RMSE:  {rmse_all:.6f}")
    print(f"sMAPE: {smape_all:.2f}%")
    print(f"DAR:    {da:.3f}")
    print(f"R²:    {r2:.3f}")



    print(f"\n各步长指标:")
    for h in range(Hh):
        print(f"  H{h + 1:02d} MAE={mae_h[h]:.6f} RMSE={rmse_h[h]:.6f} "
              f"sMAPE={smape_h[h]:.2f}%")

    # =================== 返回详细结果 ===================
    results = {
        # 基础指标
        "MAE": mae_all, "RMSE": rmse_all, "sMAPE": smape_all, "DA": da,
        "R2": r2, "MAE_H": mae_h, "RMSE_H": rmse_h, "sMAPE_H": smape_h,

        # 方法标记
        "method": method_tag,
        "loss_type": "DirectionalLoss",
        "use_validation": True,

        # 训练历史
        "history_pretrain": history_pretrain,
        "history_adapter": history_adapter,

        # 在线微调统计
        "online_finetune_stats": {
            "accepted": finetune_accepted,
            "rejected": finetune_rejected,
            "skipped": finetune_skipped,
            "accept_rate": accept_rate if online_cfg["enable"] else 0
        }
    }

    print("\n" + "=" * 80)
    print(f"股票 {stock_id} 处理完成！")
    print("=" * 80)

    return results


# ============== 主函数 ==============
if __name__ == "__main__":
    # =================== 实验配置 ===================
    EXPERIMENT_CONFIG = {
        "tcvae_pooling": "attention",  # 可选: 'mean', 'max', 'last', 'attention'
        "diffusion_enable": False,
        "diffusion_mode": "mixed",#可选：'faithful', 'adversarial', 'mixed'
    }

    print("\n" + "=" * 80)
    print("实验配置")
    print("=" * 80)
    print(f"TCVAE池化模式: {EXPERIMENT_CONFIG['tcvae_pooling']}")
    print(f"Diffusion增强: {'启用' if EXPERIMENT_CONFIG['diffusion_enable'] else '禁用'}")
    if EXPERIMENT_CONFIG['diffusion_enable']:
        print(f"Diffusion模式: {EXPERIMENT_CONFIG['diffusion_mode']}")
    print("=" * 80 + "\n")

    # =================== 更新全局配置 ===================
    # 将实验配置同步到CFG
    CFG["tcvae"]["pooling"] = EXPERIMENT_CONFIG["tcvae_pooling"]  # 更新默认池化
    CFG["tcvae_pooling"] = EXPERIMENT_CONFIG["tcvae_pooling"]  # 用于run_stock优先级覆盖
    CFG["diffusion"]["enable"] = EXPERIMENT_CONFIG["diffusion_enable"]
    if EXPERIMENT_CONFIG["diffusion_enable"]:
        CFG["diffusion"]["mode"] = EXPERIMENT_CONFIG["diffusion_mode"]

    # =================== 加载数据 ===================
    price, bench, panel_data = load_real_data(
        CFG["data_path"],
        CFG["max_assets"],
        CFG["max_days"]
    )

    # =================== 运行单组实验 ===================
    rows = []
    method_name = "Diffusion" if EXPERIMENT_CONFIG["diffusion_enable"] else "No_Augment"
    if EXPERIMENT_CONFIG["diffusion_enable"]:
        method_name += f"_{EXPERIMENT_CONFIG['diffusion_mode']}"
    method_name += f"_TCVAE_{EXPERIMENT_CONFIG['tcvae_pooling']}"

    print("\n" + "=" * 80)
    print(f"开始实验: {method_name}")
    print("=" * 80 + "\n")

    for stock_id, df_one in tqdm(
            panel_data.groupby("ts_code"),
            desc=f"处理进度"
    ):
        try:
            res = run_stock(
                df_one.sort_values("trade_date"),
                stock_id,
                CFG  # 传递完整的CFG
            )

            if res is None:
                continue

            row = {
                "stock_id": stock_id,
                "method": method_name,
                "tcvae_pooling": EXPERIMENT_CONFIG["tcvae_pooling"],
                "tcvae_latent_dim": CFG["tcvae"]["latent_dim"],  # 【新增】记录维度配置
                "tcvae_hidden_dim": CFG["tcvae"]["hidden_dim"],  # 【新增】
                "diffusion_enable": EXPERIMENT_CONFIG["diffusion_enable"],
                "diffusion_mode": EXPERIMENT_CONFIG["diffusion_mode"] if EXPERIMENT_CONFIG[
                    "diffusion_enable"] else "N/A",
                "loss_type": res.get("loss_type", "DirectionalLoss"),
                "MAE": res["MAE"],
                "RMSE": res["RMSE"],
                "sMAPE": res["sMAPE"],
                "DA": res["DA"],
                "R2": res["R2"]
            }

            H = len(res["MAE_H"])
            for h in range(H):
                row[f"MAE@H{h + 1}"] = res["MAE_H"][h]
                row[f"RMSE@H{h + 1}"] = res["RMSE_H"][h]
                row[f"sMAPE@H{h + 1}"] = res["sMAPE_H"][h]

            rows.append(row)

        except Exception as e:
            print(f"\n[ERROR] 股票 {stock_id} 跳过，原因: {e}")
            import traceback

            traceback.print_exc()

    # =================== 保存结果 ===================
    if len(rows) == 0:
        print("\n警告：没有成功处理任何股票！")
        exit(0)

    out = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("实验结果统计")
    print("=" * 80)
    print(f"\n实验配置: {method_name}")
    print(out[['MAE', 'RMSE', 'sMAPE', 'DA', 'R2']].describe())

    result_filename = f"checkpoints/results_{method_name}.csv"
    out.to_csv(result_filename, index=False)

    print(f"\n✓ 结果已保存！")
    print(f"  - 结果文件: {result_filename}")
    print(f"  - 成功处理: {len(rows)} 个股票")
