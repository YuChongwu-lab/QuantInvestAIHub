
import os
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from validation_framework import Trainer, split_train_val


# =================== 核心函数1: 预训练骨干网络（带验证） ===================

def pretrain_backbone_with_validation(
    model: nn.Module,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: Dict,
    loss_fn: nn.Module,
    device: str = 'cuda',
    val_ratio: float = 0.2,
    save_path: Optional[str] = None
) -> Dict:
    """
    预训练PatchTST骨干网络（带验证集和Early Stopping）
    Args:
        model: PatchTST模型
        X_train: 训练数据 (N, seq_len, feature_dim)
        Y_train: 训练标签 (N, pred_len)
        cfg: 配置字典 {'lr': ..., 'batch_size': ..., 'epochs': ...}
        loss_fn: 损失函数（DirectionalLoss或MSE）
        device: 设备
        val_ratio: 验证集比例
        save_path: 保存最佳模型的路径
        
    Returns:
        history: 训练历史
            - 'train_loss': 每轮训练loss
            - 'val_loss': 每轮验证loss
            - 'val_da': 每轮验证DA
            - 'best_epoch': 最佳epoch
            - 'best_val_loss': 最佳验证loss
    """
    print("\n" + "="*70)
    print("开始预训练（带Validation）")
    print("="*70)
    
    # 1. 划分训练/验证集
    X_tr, X_val, Y_tr, Y_val = split_train_val(
        X_train, Y_train, 
        val_ratio=val_ratio, 
        shuffle=False  # 时间序列不打乱
    )
    
    print(f"训练集大小: {len(X_tr)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 2. 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32)
        ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=True,
        drop_last=False,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val, dtype=torch.float32)
        ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=False,
        drop_last=False
    )
    
    # 3. 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 1e-3))
    
    # 4. 定义评估指标（DA）
    def compute_da(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算方向准确率"""
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        return np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))
    
    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=cfg.get('patience', 10),  # 10轮不改善就停止
        min_delta=cfg.get('min_delta', 1e-4),
        metric_fn=compute_da,  # 监控DA
        metric_mode='max',  # DA越大越好
        verbose=True
    )
    
    # 6. 训练
    if save_path is None:
        save_path = "checkpoints/best_pretrain_model.pth"
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.get('epochs', 100),  # 最大轮数
        save_path=save_path
    )
    
    # 7. 设置模型为评估模式并冻结参数
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("\n预训练完成！")
    print(f"最佳Epoch: {history['best_epoch']+1}")
    print(f"最佳Val Loss: {history['best_val_loss']:.6f}")
    if 'best_val_metric' in history:
        print(f"最佳Val DA: {history['best_val_metric']:.4f}")
    
    return history


# =================== 核心函数2: 训练Adapter（带验证） ===================

def train_adapter_with_validation(
    enc_src: nn.Module,
    enc_tgt: nn.Module,
    adapter: nn.Module,
    base_model: nn.Module,
    X_train: np.ndarray,  # 原 Xw_all
    Y_train: np.ndarray,  # 原 Yw_all
    cfg: Dict,
    loss_fn: nn.Module,
    device: str = 'cuda',
    val_ratio: float = 0.2,
    use_diffusion: bool = False,
    diffusion_augmentor = None
) -> Dict:
    """
    功能：训练Adapter（带验证集和Early Stopping）
    1. 划分训练/验证集
    2. 不使用数据增强
    3. Early Stopping
    4. 返回完整训练历史
    Args:
        enc_src, enc_tgt: 概念编码器
        adapter: Adapter模块
        base_model: PatchTST骨干网络
        X_train, Y_train: 训练数据和标签
        cfg: 配置字典
        loss_fn: 损失函数
        device: 设备
        val_ratio: 验证集比例
        use_diffusion: 是否使用Diffusion增强
        diffusion_augmentor: Diffusion增强器
        
    Returns:
        history: 训练历史
    """
    print("\n" + "="*70)
    print("开始训练Adapter（带Validation）")
    print("="*70)
    
    # 1. 划分训练/验证集
    X_tr, X_val, Y_tr, Y_val = split_train_val(X_train, Y_train, val_ratio=val_ratio, shuffle=False)
    print(f"训练集大小: {len(X_tr)}")
    print(f"验证集大小: {len(X_val)}")
    
    # 2. 移动模型到设备
    enc_src.to(device)
    enc_tgt.to(device)
    adapter.to(device)
    base_model.to(device).eval()
    
    # 3. 设置可训练参数
    trainable_params = (
        list(enc_src.parameters()) +
        list(enc_tgt.parameters()) +
        list(adapter.parameters())
    )
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.get('lr', 5e-4))
    
    # 4. 创建训练数据加载器（不使用数据增强）
    # 训练集加载器
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(Y_tr, dtype=torch.float32)),
        batch_size=cfg.get('batch_size', 64),
        shuffle=True,
        drop_last=False
    )
    
    # 验证集加载器
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32) ),
        batch_size=cfg.get('batch_size', 64),
        shuffle=False,
        drop_last=False
    )
    
    # 5. 定义自定义训练和验证函数（因为需要特殊的forward逻辑）
    class AdapterTrainer:
        def __init__(self, enc_src, enc_tgt, adapter, base_model, loss_fn, optimizer, device, reg_w=1e-4):
            self.enc_src = enc_src
            self.enc_tgt = enc_tgt
            self.adapter = adapter
            self.base_model = base_model
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            self.device = device
            self.reg_w = reg_w
            
            self.best_val_loss = float('inf')
            self.best_epoch = 0
            self.best_state = None
            self.epochs_no_improve = 0
            self.history = {'train_loss': [], 'val_loss': [], 'val_da': []}
        
        def train_epoch(self, train_loader):
            """训练一个epoch"""
            self.enc_src.train()
            self.enc_tgt.train()
            self.adapter.train()
            self.base_model.eval()
            
            total_loss = 0.0
            count = 0
            
            prev_Xb = None
            
            for Xb, Yb in train_loader:
                Xb = Xb.to(self.device)
                Yb = Yb.to(self.device)
                
                # 目标域编码
                cj = self.enc_tgt(Xb)
                
                # 源域编码（使用上一个batch的平均）
                if prev_Xb is None:
                    prev_Xb = Xb.detach()
                
                with torch.no_grad():
                    ci_avg = self.enc_src(prev_Xb).mean(dim=0, keepdim=True)
                ci = ci_avg.expand_as(cj)


                # Adapter调制
                delta = cj - ci
                mod = self.adapter(delta)
                
                # 预测
                pred = self.base_model(Xb, mod=mod)
                
                # 损失
                loss_pred = self.loss_fn(
                    pred.view(pred.size(0), -1),
                    Yb.view(Yb.size(0), -1)
                )
                
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
                
                loss = loss_pred + self.reg_w * reg
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.enc_src.parameters()) + 
                    list(self.enc_tgt.parameters()) + 
                    list(self.adapter.parameters()),
                    1.0
                )
                self.optimizer.step()
                
                prev_Xb = Xb.detach()
                
                total_loss += loss.item() * Xb.size(0)
                count += Xb.size(0)
            
            return total_loss / max(1, count)
        
        def validate_epoch(self, val_loader):
            """验证一个epoch"""
            self.enc_src.eval()
            self.enc_tgt.eval()
            self.adapter.eval()
            self.base_model.eval()
            
            total_loss = 0.0
            count = 0
            all_preds = []
            all_targets = []
            
            prev_Xb = None
            
            with torch.no_grad():
                for Xb, Yb in val_loader:
                    Xb = Xb.to(self.device)
                    Yb = Yb.to(self.device)
                    
                    # 与训练相同的forward逻辑
                    cj = self.enc_tgt(Xb)
                    
                    if prev_Xb is None:
                        prev_Xb = Xb
                    
                    # ci采用均值
                    ci_avg = self.enc_src(prev_Xb).mean(dim=0, keepdim=True)
                    ci = ci_avg.expand_as(cj)
                    
                    delta = cj - ci
                    mod = self.adapter(delta)
                    
                    pred = self.base_model(Xb, mod=mod)
                    
                    loss = self.loss_fn(
                        pred.view(pred.size(0), -1),
                        Yb.view(Yb.size(0), -1)
                    )
                    
                    total_loss += loss.item() * Xb.size(0)
                    count += Xb.size(0)
                    
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(Yb.cpu().numpy())
                    
                    prev_Xb = Xb
            
            val_loss = total_loss / max(1, count)
            
            # 计算DA
            all_preds = np.concatenate(all_preds, axis=0).flatten()
            all_targets = np.concatenate(all_targets, axis=0).flatten()
            val_da = np.mean(np.sign(all_preds) == np.sign(all_targets))
            
            return val_loss, val_da
        
        def fit(self, train_loader, val_loader, epochs, patience=10):
            """完整训练循环"""
            for epoch in range(epochs):
                # 训练
                train_loss = self.train_epoch(train_loader)
                self.history['train_loss'].append(train_loss)
                
                # 验证
                val_loss, val_da = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_da'].append(val_da)
                
                # 检查改善
                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.best_state = {
                        'enc_src': copy.deepcopy(self.enc_src.state_dict()),
                        'enc_tgt': copy.deepcopy(self.enc_tgt.state_dict()),
                        'adapter': copy.deepcopy(self.adapter.state_dict())
                    }
                    self.epochs_no_improve = 0
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Val DA: {val_da:.4f} ⭐")
                else:
                    self.epochs_no_improve += 1
                    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | Val DA: {val_da:.4f}")
                
                # Early Stopping
                if self.epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            
            # 恢复最佳模型
            if self.best_state is not None:
                self.enc_src.load_state_dict(self.best_state['enc_src'])
                self.enc_tgt.load_state_dict(self.best_state['enc_tgt'])
                self.adapter.load_state_dict(self.best_state['adapter'])
                print(f"\n Restored best model from epoch {self.best_epoch+1}")
            
            self.history['best_epoch'] = self.best_epoch
            self.history['best_val_loss'] = self.best_val_loss
            
            return self.history
    
    # 6. 训练
    adapter_trainer = AdapterTrainer(
        enc_src, enc_tgt, adapter, base_model,
        loss_fn, optimizer, device,
        reg_w=cfg.get('reg', 1e-4)
    )
    
    history = adapter_trainer.fit(
        train_loader, val_loader,
        epochs=cfg.get('epochs', 10),
        patience=cfg.get('patience', 10)
    )
    
    print("\nAdapter训练完成！")
    print(f"最佳Epoch: {history['best_epoch']+1}")
    print(f"最佳Val Loss: {history['best_val_loss']:.6f}")
    
    return history


# =================== 核心函数3: 在线微调（带验证） ===================

def online_finetune_with_validation(
    online_model: nn.Module,
    params: List,
    X_ft: np.ndarray,
    Y_ft: np.ndarray,
    loss_fn: nn.Module,
    lr: float = 1e-4,
    steps: int = 1,
    batch_size: int = 8,
    val_ratio: float = 0.2,
    min_samples: int = 10,
    device: str = 'cuda'
) -> bool:
    """
    在线微调（带验证检查）
    功能：
    1. 微调前保存模型状态
    2. 划分训练/验证集
    3. 微调后在验证集上检查是否真的改善
    4. 只有改善时才接受微调，否则回滚
    
    Args:
        online_model: 在线模型（需要微调的模型）
        params: 模型的可训练参数列表
        X_ft, Y_ft: 微调数据
        loss_fn: 损失函数
        lr: 学习率
        steps: 微调步数
        batch_size: batch大小
        val_ratio: 验证集比例
        min_samples: 最少样本数，如果样本少于该值，则不进行微调。
        device: 设备
        
    Returns:
        accepted: 是否接受微调（True表示改善，False表示回滚）
    """
    if len(X_ft) < min_samples:
        return False  # 样本太少，不微调
    
    # 1. 保存原始模型状态
    original_state = copy.deepcopy(online_model.state_dict())
    
    # 2. 划分训练/验证集
    if len(X_ft) > min_samples * 2:
        X_tr, X_val, Y_tr, Y_val = split_train_val(
            X_ft, Y_ft,
            val_ratio=val_ratio,
            shuffle=False
        )
    else:
        # 样本太少，不划分验证集，直接用全部数据
        X_tr, Y_tr = X_ft, Y_ft
        X_val, Y_val = None, None
    
    # 3. 计算微调前的验证loss（如果有验证集）
    if X_val is not None:
        online_model.eval()
        with torch.no_grad():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32).to(device)
            pred_before = online_model(X_val_tensor)
            loss_before = loss_fn(
                pred_before.view(pred_before.size(0), -1),
                Y_val_tensor.view(Y_val_tensor.size(0), -1)
            ).item()
    else:
        loss_before = None
    
    # 4. 微调
    optimizer = torch.optim.Adam(params, lr=lr)
    
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32)
        ),
        batch_size=min(batch_size, len(X_tr)),
        shuffle=True,
        drop_last=False
    )
    
    online_model.train()
    for _ in range(steps):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            pred = online_model(xb)
            loss = loss_fn(
                pred.view(pred.size(0), -1),
                yb.view(yb.size(0), -1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 5. 计算微调后的验证loss
    if X_val is not None:
        online_model.eval()
        with torch.no_grad():
            pred_after = online_model(X_val_tensor)
            loss_after = loss_fn(
                pred_after.view(pred_after.size(0), -1),
                Y_val_tensor.view(Y_val_tensor.size(0), -1)
            ).item()
        
        # 6. 判断是否改善
        if loss_after < loss_before:
            # 改善，接受微调

            return True
        else:
            # 没改善，回滚
            online_model.load_state_dict(original_state)

            return False
    else:
        # 没有验证集，无条件接受
        return True


