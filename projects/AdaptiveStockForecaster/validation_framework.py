
"""
训练框架 - 包含验证集、Early Stopping和最佳模型选择
═══════════════════════════════════════════════════════════════════════════
核心功能：
1. Trainer类 - 自动处理验证、Early Stopping、最佳模型保存
2. AugmentationEvaluator - 评估数据增强配置
3. 完整的训练监控和可视化
"""

import os
import copy
import json
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score


# =================== 1. Trainer类 ===================

class Trainer:
    """
    通用训练器 - 自动处理验证、Early Stopping和最佳模型选择
    核心功能：
    - 自动划分训练/验证集
    - 每个epoch在验证集上评估
    - Early Stopping（N轮不改善就停止）
    - 保存验证loss最小的模型
    - 记录完整训练历史
    - 支持自定义评估指标

    """
    
    def __init__(self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        patience: int = 10,
        min_delta: float = 1e-4,
        metric_fn: Optional[Callable] = None,
        metric_mode: str = 'max',  # 'max' for DA/R², 'min' for loss
        verbose: bool = True
    ):
        """
        Args:
            model: 要训练的模型
            loss_fn: 损失函数
            optimizer: 优化器
            device: 设备 ('cuda' or 'cpu')
            patience: Early Stopping的耐心值
            min_delta: 认为"改善"的最小变化量
            metric_fn: 可选的评估指标函数 (y_pred, y_true) -> float
            metric_mode: 'max' 表示指标越大越好，'min' 表示越小越好
            verbose: 是否打印训练信息
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.metric_fn = metric_fn
        self.metric_mode = metric_mode
        self.verbose = verbose
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_metric = float('-inf') if metric_mode == 'max' else float('inf')
        self.best_epoch = 0
        self.best_model_state = None
        self.epochs_no_improve = 0
        
        # 训练历史
        self.history = defaultdict(list)
        
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        eval_every: int = 1,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大训练轮数
            eval_every: 每隔多少轮评估一次
            save_path: 保存最佳模型的路径（可选）
        Returns:
            history: 包含训练历史的字典
                - 'train_loss': 每轮的训练loss
                - 'val_loss': 每轮的验证loss
                - 'val_metric': 每轮的验证指标（如果提供）
                - 'best_epoch': 最佳epoch
                - 'best_val_loss': 最佳验证loss
                - 'stopped_early': 是否提前停止
        """
        for epoch in range(epochs):
            # 训练
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # 验证
            if (epoch + 1) % eval_every == 0:
                val_loss, val_metric = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                if val_metric is not None:
                    self.history['val_metric'].append(val_metric)
                
                # 检查是否改善
                improved = self._check_improvement(val_loss, val_metric)
                
                if improved:
                    self.best_val_loss = val_loss
                    if val_metric is not None:
                        self.best_val_metric = val_metric
                    self.best_epoch = epoch
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    self.epochs_no_improve = 0
                    
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        if self.verbose:
                            print(f" Saved best model to {save_path}")
                else:
                    self.epochs_no_improve += 1
                
                # 打印进度
                if self.verbose:
                    self._print_progress(epoch, epochs, train_loss, val_loss, val_metric)
                
                # Early Stopping
                if self.epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"\n  Early stopping at epoch {epoch+1}")
                        print(f"   Best epoch: {self.best_epoch+1}")
                        print(f"   Best val_loss: {self.best_val_loss:.6f}")
                        if val_metric is not None:
                            print(f"   Best val_metric: {self.best_val_metric:.6f}")
                    self.history['stopped_early'] = True
                    break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"\n Restored best model from epoch {self.best_epoch+1}")
        
        # 整理历史
        self.history['best_epoch'] = self.best_epoch
        self.history['best_val_loss'] = self.best_val_loss
        if self.metric_fn is not None:
            self.history['best_val_metric'] = self.best_val_metric
        
        return dict(self.history)
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        count = 0
        
        for xb, yb in train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            
            # 前向传播
            pred = self.model(xb)
            
            # 确保形状匹配
            if pred.dim() > 2:
                pred = pred.view(pred.size(0), -1)
            if yb.dim() > 2:
                yb = yb.view(yb.size(0), -1)
            
            loss = self.loss_fn(pred, yb)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
        
        return total_loss / max(1, count)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Optional[float]]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        count = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                pred = self.model(xb)
                
                # 确保形状匹配
                if pred.dim() > 2:
                    pred = pred.view(pred.size(0), -1)
                if yb.dim() > 2:
                    yb = yb.view(yb.size(0), -1)
                
                loss = self.loss_fn(pred, yb)
                
                total_loss += loss.item() * xb.size(0)
                count += xb.size(0)
                
                # 收集预测用于计算指标
                if self.metric_fn is not None:
                    all_preds.append(pred.cpu().numpy())
                    all_targets.append(yb.cpu().numpy())
        
        val_loss = total_loss / max(1, count)
        
        # 计算指标
        val_metric = None
        if self.metric_fn is not None and len(all_preds) > 0:
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            val_metric = self.metric_fn(all_preds, all_targets)
        
        return val_loss, val_metric
    
    def _check_improvement(self, val_loss: float, val_metric: Optional[float]) -> bool:
        """检查是否有改善"""
        # 主要看validation loss
        loss_improved = val_loss < (self.best_val_loss - self.min_delta)
        
        # 如果有metric，也考虑metric
        metric_improved = False
        if val_metric is not None:
            if self.metric_mode == 'max':
                metric_improved = val_metric > (self.best_val_metric + self.min_delta)
            else:
                metric_improved = val_metric < (self.best_val_metric - self.min_delta)
        
        # 主要以loss为准，metric作为参考
        return loss_improved or (val_metric is not None and metric_improved)
    
    def _print_progress(self, epoch: int, total_epochs: int, 
                       train_loss: float, val_loss: float, 
                       val_metric: Optional[float]):
        """打印训练进度"""
        msg = f"Epoch {epoch+1}/{total_epochs} | "
        msg += f"Train Loss: {train_loss:.6f} | "
        msg += f"Val Loss: {val_loss:.6f}"
        
        if val_metric is not None:
            msg += f" | Val Metric: {val_metric:.6f}"
        
        if val_loss < self.best_val_loss:
            msg += " ⭐"
        
        print(msg)


# =================== 2. 数据增强评估器 ===================

class AugmentationEvaluator:
    """
    数据增强配置评估器
    
    功能：
    - 对比不同数据增强配置的效果
    - 自动选择最佳配置
    - 生成评估报告

    """
    
    def __init__(
        self,
        model_fn: Callable,  # 返回新模型实例的函数
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        base_config: Dict,
        loss_fn: nn.Module,
        device: str = 'cuda'
    ):
        """
        Args:
            model_fn: 返回新模型实例的函数
            train_data: (X_train, Y_train)
            val_data: (X_val, Y_val)
            base_config: 基础训练配置
            loss_fn: 损失函数
            device: 设备
        """
        self.model_fn = model_fn
        self.X_train, self.Y_train = train_data
        self.X_val, self.Y_val = val_data
        self.base_config = base_config
        self.loss_fn = loss_fn
        self.device = device
        
        self.results = []
    
    def compare_configs(
        self,
        augmentation_configs: List[Tuple[Dict, str]],
        n_runs: int = 3
    ) -> pd.DataFrame:
        """
        对比多个数据增强配置
        
        Args:
            augmentation_configs: [(config_dict, name), ...]
            n_runs: 每个配置运行几次（取平均）
            
        Returns:
            results_df: 包含所有配置结果的DataFrame
        """
        print("\n" + "="*70)
        print(" 数据增强配置评估")
        print("="*70)
        
        for aug_config, config_name in augmentation_configs:
            print(f"\n 测试配置: {config_name}")
            print(f"   参数: {aug_config}")
            
            metrics_runs = []
            
            for run in range(n_runs):
                print(f"   Run {run+1}/{n_runs}...", end=" ")
                
                # 创建新模型
                model = self.model_fn().to(self.device)
                
                # 创建数据加载器（根据配置应用数据增强）
                train_loader = self._create_dataloader(
                    self.X_train, self.Y_train, 
                    aug_config, 
                    shuffle=True
                )
                val_loader = self._create_dataloader(
                    self.X_val, self.Y_val, 
                    {'enable': False},  # 验证集不增强
                    shuffle=False
                )
                
                # 训练
                optimizer = torch.optim.Adam(
                    model.parameters(), 
                    lr=self.base_config.get('lr', 1e-3)
                )
                
                trainer = Trainer(
                    model=model,
                    loss_fn=self.loss_fn,
                    optimizer=optimizer,
                    device=self.device,
                    patience=self.base_config.get('patience', 10),
                    metric_fn=self._compute_da,
                    metric_mode='max',
                    verbose=False
                )
                
                history = trainer.fit(
                    train_loader, 
                    val_loader, 
                    epochs=self.base_config.get('epochs', 20)
                )
                
                # 记录指标
                metrics = {
                    'val_loss': history['best_val_loss'],
                    'val_da': history.get('best_val_metric', 0),
                    'best_epoch': history['best_epoch'],
                    'stopped_early': history.get('stopped_early', False)
                }
                metrics_runs.append(metrics)
                
                print(f"Val Loss: {metrics['val_loss']:.6f}, DA: {metrics['val_da']:.4f}")
            
            # 计算平均指标
            avg_metrics = {
                'config_name': config_name,
                'config': str(aug_config),
                'val_loss_mean': np.mean([m['val_loss'] for m in metrics_runs]),
                'val_loss_std': np.std([m['val_loss'] for m in metrics_runs]),
                'val_da_mean': np.mean([m['val_da'] for m in metrics_runs]),
                'val_da_std': np.std([m['val_da'] for m in metrics_runs]),
                'avg_best_epoch': np.mean([m['best_epoch'] for m in metrics_runs]),
                'early_stop_rate': np.mean([m['stopped_early'] for m in metrics_runs])
            }
            
            self.results.append(avg_metrics)
            
            print(f"    平均结果: Val Loss={avg_metrics['val_loss_mean']:.6f}±{avg_metrics['val_loss_std']:.6f}, "
                  f"DA={avg_metrics['val_da_mean']:.4f}±{avg_metrics['val_da_std']:.4f}")
        
        # 生成结果表
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print(" 评估结果汇总")
        print("="*70)
        print(results_df[['config_name', 'val_loss_mean', 'val_da_mean', 'avg_best_epoch']].to_string(index=False))
        
        return results_df
    
    def get_best_config(self, metric: str = 'val_da_mean') -> Dict:
        """
        获取最佳配置
        
        Args:
            metric: 用于选择的指标 ('val_loss_mean' or 'val_da_mean')
            
        Returns:
            best_config: 最佳配置的字典
        """
        if not self.results:
            raise ValueError("请先运行 compare_configs()")
        
        results_df = pd.DataFrame(self.results)
        
        if 'loss' in metric:
            best_idx = results_df[metric].idxmin()
        else:
            best_idx = results_df[metric].idxmax()
        
        best_result = results_df.iloc[best_idx]
        
        print(f"\n 最佳配置 (按{metric}):")
        print(f"   名称: {best_result['config_name']}")
        print(f"   配置: {best_result['config']}")
        print(f"   Val Loss: {best_result['val_loss_mean']:.6f}±{best_result['val_loss_std']:.6f}")
        print(f"   Val DA: {best_result['val_da_mean']:.4f}±{best_result['val_da_std']:.4f}")
        
        return best_result.to_dict()
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        aug_config: Dict, 
        shuffle: bool
    ) -> DataLoader:
        """创建数据加载器"""
        # 这里简化处理，实际使用时需要根据你的数据增强实现调整
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)
        )
        
        return DataLoader(
            dataset,
            batch_size=self.base_config.get('batch_size', 64),
            shuffle=shuffle,
            drop_last=False
        )
    
    def _compute_da(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算方向准确率"""
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        return np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))


# =================== 3. 辅助函数 ===================

def split_train_val(X, Y, val_ratio=0.2, shuffle=False):
    """
    划分训练集和验证集
    Args:
        X: 特征数组
        Y: 标签数组
        val_ratio: 验证集比例
        shuffle: 是否打乱（时间序列通常不打乱）
        
    Returns:
        X_train, X_val, Y_train, Y_val
    """
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_val
    
    if shuffle:
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
    else:
        # 时间序列：前80%训练，后20%验证
        train_indices = np.arange(n_train)
        val_indices = np.arange(n_train, n_samples)
    
    X_train = X[train_indices]
    X_val = X[val_indices]
    Y_train = Y[train_indices]
    Y_val = Y[val_indices]
    
    return X_train, X_val, Y_train, Y_val


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    计算多个评估指标
    
    Args:
        y_pred: 预测值
        y_true: 真实值
        
    Returns:
        metrics: {'da': ..., 'r2': ..., 'mae': ..., 'rmse': ...}
    """
    y_pred_flat = y_pred.flatten()
    y_true_flat = y_true.flatten()
    
    da = np.mean(np.sign(y_pred_flat) == np.sign(y_true_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = np.mean(np.abs(y_pred_flat - y_true_flat))
    rmse = np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2))
    
    return {
        'da': da,
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }


