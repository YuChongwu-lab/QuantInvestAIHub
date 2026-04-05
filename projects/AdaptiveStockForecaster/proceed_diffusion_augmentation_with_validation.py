
"""
Proceed + Diffusion 数据增强集成模块（带完整Validation）
═══════════════════════════════════════════════════════════════════════════
1. 添加训练/验证集划分
2.  验证集不使用数据增强
3. Early Stopping机制
4. 保存和恢复最佳模型
5. 返回完整训练历史++
6. 监控验证集DA指标


"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict
import random

from validation_framework import split_train_val
from proceed_diffusion_augmentation import ProceedDiffusionDataset


# ═══════════════════════════════════════════════════════════════════
# 核心训练函数（带Validation）
# ═══════════════════════════════════════════════════════════════════

def train_adapter_proceed_with_diffusion_validated(
    enc_src,
    enc_tgt,
    adapter,
    base_model,
    Xw_all,
    Yw_all,
    cfg,
    diffusion_augmentor,
    device,
    val_ratio: float = 0.2,
    save_best_path: Optional[str] = None
) -> Dict:
    """
    使用 Diffusion 增强的 Proceed 训练（带Validation）

    1. 训练/验证集划分（80/20）
    2. 验证集不使用数据增强
    3. Early Stopping（patience=10）
    4. 保存和恢复最佳模型
    5. 监控验证集DA和Loss
    6. 返回完整训练历史
    
    Args:
        enc_src: 源域概念编码器
        enc_tgt: 目标域概念编码器
        adapter: Adapter模块
        base_model: PatchTST骨干网络
        Xw_all: 全部窗口数据 (N, T, F)
        Yw_all: 全部标签 (N, P)
        cfg: 配置字典 {'lr', 'batch_size', 'epochs', 'reg', 'synth_prob'}
        diffusion_augmentor: Diffusion增强器
        device: 训练设备
        val_ratio: 验证集比例
        save_best_path: 保存最佳模型路径（可选）
        
    Returns:
        history: 训练历史字典
            - 'train_loss': 每轮训练loss
            - 'val_loss': 每轮验证loss
            - 'val_da': 每轮验证DA
            - 'best_epoch': 最佳epoch
            - 'best_val_loss': 最佳验证loss
            - 'best_val_da': 最佳验证DA
    """
    print("\n" + "="*70)
    print("开始训练Adapter（Proceed + Diffusion + Validation）")
    print("="*70)
    
    # =================== 1. 数据集划分 ===================
    Xw_tr, Xw_val, Yw_tr, Yw_val = split_train_val(
        Xw_all, Yw_all,
        val_ratio=val_ratio,
        shuffle=False  # 时间序列不打乱
    )
    
    print(f"训练集大小: {len(Xw_tr)}")
    print(f"验证集大小: {len(Xw_val)}")
    
    # =================== 2. 创建数据集 ===================
    # 训练集：使用Diffusion增强
    train_ds = ProceedDiffusionDataset(
        Xw_tr, Yw_tr,
        diffusion_augmentor=diffusion_augmentor,
        batch_size=cfg["batch_size"],
        augment_prob=cfg.get("synth_prob", 0.35),
        augment_ratio=1.0,
        use_concept_only=True
    )
    
    # 验证集：不使用增强
    val_ds = ProceedDiffusionDataset(
        Xw_val, Yw_val,
        diffusion_augmentor=None,  # 验证集不增强
        batch_size=cfg["batch_size"],
        augment_prob=0.0,  # 确保不增强
        augment_ratio=0.0,
        use_concept_only=True
    )
    
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, drop_last=False)
    
    # =================== 3. 模型和优化器设置 ===================
    enc_src.to(device)
    enc_tgt.to(device)
    adapter.to(device)
    base_model.to(device).eval()

    trainable_params = (
        list(enc_src.parameters()) +
        list(enc_tgt.parameters()) +
        list(adapter.parameters())
    )
    opt = torch.optim.Adam(trainable_params, lr=cfg["lr"])
    mse = nn.MSELoss()
    reg_w = cfg.get("reg", 1e-4)
    
    # =================== 4. 训练状态跟踪 ===================
    best_val_loss = float('inf')
    best_val_da = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = cfg.get("patience", 10)
    
    best_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_da': []
    }
    
    # =================== 5. 定义训练和验证函数 ===================
    def train_epoch(dataloader, prev_Xb_buffer):
        """训练一个epoch"""
        enc_src.train()
        enc_tgt.train()
        adapter.train()
        base_model.eval()
        
        total_loss = 0.0
        nstep = 0
        
        for batch in dataloader:
            # 处理不同格式的批次
            if len(batch) == 3:  # 混合批次（带增强）
                Xb_mixed, Yb, n_real = batch
                Xb_mixed = Xb_mixed.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)
                n_real = n_real.item()
                
                Xb_real = Xb_mixed[:n_real]
                Xb_all = Xb_mixed
            else:  # 普通批次
                Xb, Yb = batch
                Xb = Xb.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)
                Xb_real = Xb
                Xb_all = Xb
            
            # 目标概念（使用全部数据包括增强）
            cj = enc_tgt(Xb_all).mean(dim=0, keepdim=True)
            
            # 源概念（使用buffer）
            if prev_Xb_buffer is None:
                prev_Xb_buffer = Xb_real[:min(len(Xb_real), 32)].detach().clone()
            
            with torch.no_grad():
                ci_avg = enc_src(prev_Xb_buffer).mean(dim=0, keepdim=True)
            
            ci = ci_avg.expand(len(Xb_real), -1)
            cj_expanded = cj.expand(len(Xb_real), -1)
            delta = cj_expanded - ci
            
            # Adapter
            mod = adapter(delta)
            
            # 预测（只用真实数据）
            pred = base_model(Xb_real, mod=mod)
            
            # 损失
            loss_pred = mse(pred.view(pred.size(0), -1), Yb.view(Yb.size(0), -1))
            
            # 正则化
            reg = 0.0
            for k, t in mod["patch_embed"].items():
                reg = reg + t.pow(2).mean()
            for g in mod["enc"]:
                for k, t in g.items():
                    reg = reg + t.pow(2).mean()
            for k, t in mod["head_in"].items():
                reg = reg + t.pow(2).mean()
            for k, t in mod["fc_out"].items():
                reg = reg + t.pow(2).mean()
            
            loss = loss_pred + reg_w * reg
            
            # 反向传播
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            
            # 更新buffer
            prev_Xb_buffer = Xb_real[:min(len(Xb_real), 32)].detach().clone()
            
            total_loss += float(loss.item())
            nstep += 1
        
        avg_loss = total_loss / max(1, nstep)
        return avg_loss, prev_Xb_buffer
    
    def validate_epoch(dataloader, prev_Xb_buffer):
        """验证一个epoch"""
        enc_src.eval()
        enc_tgt.eval()
        adapter.eval()
        base_model.eval()
        
        total_loss = 0.0
        nstep = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 验证集不使用增强，所以batch格式简单
                if len(batch) == 3:
                    Xb, Yb, _ = batch
                else:
                    Xb, Yb = batch
                
                Xb = Xb.squeeze(0).to(device)
                Yb = Yb.squeeze(0).to(device)
                
                # 目标概念
                cj = enc_tgt(Xb).mean(dim=0, keepdim=True)
                
                # 源概念
                if prev_Xb_buffer is None:
                    prev_Xb_buffer = Xb[:min(len(Xb), 32)].detach().clone()
                
                ci_avg = enc_src(prev_Xb_buffer).mean(dim=0, keepdim=True)
                ci = ci_avg.expand(len(Xb), -1)
                cj_expanded = cj.expand(len(Xb), -1)
                delta = cj_expanded - ci
                
                # Adapter
                mod = adapter(delta)
                
                # 预测
                pred = base_model(Xb, mod=mod)
                
                # 损失
                loss = mse(pred.view(pred.size(0), -1), Yb.view(Yb.size(0), -1))
                
                total_loss += float(loss.item())
                nstep += 1
                
                # 收集预测用于计算DA
                all_preds.append(pred.cpu().numpy())
                all_targets.append(Yb.cpu().numpy())
                
                # 更新buffer
                prev_Xb_buffer = Xb[:min(len(Xb), 32)].detach().clone()
        
        avg_loss = total_loss / max(1, nstep)
        
        # 计算DA（方向准确率）
        all_preds = np.concatenate(all_preds, axis=0).flatten()
        all_targets = np.concatenate(all_targets, axis=0).flatten()
        val_da = np.mean(np.sign(all_preds) == np.sign(all_targets))
        
        return avg_loss, val_da
    
    # =================== 6. 训练循环 ===================
    prev_Xb = None
    
    for ep in range(cfg.get("epochs", 10)):
        # 训练
        train_loss, prev_Xb = train_epoch(train_dl, prev_Xb)
        history['train_loss'].append(train_loss)
        
        # 验证
        val_loss, val_da = validate_epoch(val_dl, prev_Xb)
        history['val_loss'].append(val_loss)
        history['val_da'].append(val_da)
        
        # 检查改善
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_da = val_da
            best_epoch = ep
            epochs_no_improve = 0
            
            # 保存最佳模型状态
            best_state = {
                'enc_src': copy.deepcopy(enc_src.state_dict()),
                'enc_tgt': copy.deepcopy(enc_tgt.state_dict()),
                'adapter': copy.deepcopy(adapter.state_dict())
            }
            
            # 保存到文件（如果提供路径）
            if save_best_path:
                os.makedirs(os.path.dirname(save_best_path) if os.path.dirname(save_best_path) else '.', exist_ok=True)
                torch.save(best_state, save_best_path)
            
            print(f"Epoch {ep+1}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val DA: {val_da:.4f} ⭐")
        else:
            epochs_no_improve += 1
            print(f"Epoch {ep+1}/{cfg['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Val DA: {val_da:.4f}")
        
        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"\n Early stopping at epoch {ep+1}")
            print(f"   No improvement for {patience} epochs")
            break
    
    # =================== 7. 恢复最佳模型 ===================
    if best_state is not None:
        enc_src.load_state_dict(best_state['enc_src'])
        enc_tgt.load_state_dict(best_state['enc_tgt'])
        adapter.load_state_dict(best_state['adapter'])
        print(f"\nRestored best model from epoch {best_epoch+1}")
    
    # =================== 8. 整理训练历史 ===================
    history['best_epoch'] = best_epoch
    history['best_val_loss'] = best_val_loss
    history['best_val_da'] = best_val_da
    
    print("\n" + "="*70)
    print(" Adapter训练完成（Validation版）")
    print("="*70)
    print(f"   最佳Epoch: {best_epoch+1}")
    print(f"   最佳Val Loss: {best_val_loss:.6f}")
    print(f"   最佳Val DA: {best_val_da:.4f}")
    print("="*70 + "\n")
    
    return history


# ═══════════════════════════════════════════════════════════════════
# 便捷包装函数
# ═══════════════════════════════════════════════════════════════════

def train_adapter_proceed_with_diffusion(
    enc_src, enc_tgt, adapter, base_model,
    Xw_all, Yw_all, cfg, diffusion_augmentor,
    device,
    use_validation=True,
    val_ratio=0.2,
    save_best_path=None
):
    """

    Args:
        use_validation: 是否使用validation版本（默认True）
        val_ratio: 验证集比例
        save_best_path: 保存最佳模型路径

    """
    if use_validation:
        return train_adapter_proceed_with_diffusion_validated(
            enc_src, enc_tgt, adapter, base_model,
            Xw_all, Yw_all, cfg, diffusion_augmentor,
            device, val_ratio, save_best_path
        )
    else:
        print("建议使用validation版本！现在使用简化训练...")
        pass


