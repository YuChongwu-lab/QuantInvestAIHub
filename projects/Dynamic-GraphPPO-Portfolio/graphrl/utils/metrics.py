# =============================================================================
# graphrl/utils/metrics.py —— 信号计算 / 损失函数 / IC 评估
# =============================================================================

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def compute_signal(
    preds: torch.Tensor,
    risks: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """横截面 z-score 信号，可选风险调整"""
    if risks is not None:
        s = preds / (risks + 1e-8)
    else:
        s = preds
    if s.dim() == 2:
        s = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + 1e-8)
    else:
        s = (s - s.mean()) / (s.std() + 1e-8)
    return s


def pearson_corr(
    x: torch.Tensor, y: torch.Tensor,
    dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """按 dim 维计算皮尔逊相关系数（可微）"""
    if dim != -1:
        x = x.transpose(dim, -1)
        y = y.transpose(dim, -1)
    xm = x - x.mean(dim=-1, keepdim=True)
    ym = y - y.mean(dim=-1, keepdim=True)
    num = (xm * ym).sum(dim=-1)
    den = xm.norm(dim=-1) * ym.norm(dim=-1) + eps
    return num / den


def mse_ic_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    risks: torch.Tensor = None,
    alpha: float = 1.0,
    beta: float  = 0.5,
    eps: float   = 1e-8
) -> Tuple[torch.Tensor, Dict]:
    """组合损失 = alpha*MSE + beta*(-Pearson-IC)"""
    s       = compute_signal(preds, risks)
    ic      = pearson_corr(s, targets, dim=1)
    ic_loss = (-ic).mean()
    mse     = torch.mean((preds - targets) ** 2)
    total   = alpha * mse + beta * ic_loss
    return total, {'ic': ic.mean().detach(), 'mse': mse.detach()}


@torch.no_grad()
def evaluate_ic(
    model: nn.Module,
    dataset: List[Data],
    batch_size: int = 16
) -> Dict[str, float]:
    """评估平均日度 Spearman-IC（numpy/pandas 口径）"""
    model.eval()
    ics: List[float] = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        try:
            out   = model(batch)
            preds = out['predictions'].detach().cpu().numpy()
            ys    = np.stack([d.y.detach().cpu().numpy() for d in batch])
        except Exception:
            continue
        for p, y in zip(preds, ys):
            if not (np.isfinite(p).all() and np.isfinite(y).all()):
                continue
            if np.std(p) == 0 or np.std(y) == 0:
                continue
            rx = pd.Series(p).rank(method='average').to_numpy()
            ry = pd.Series(y).rank(method='average').to_numpy()
            ic = np.corrcoef(rx, ry)[0, 1]
            if np.isfinite(ic):
                ics.append(float(ic))
    return {"ic_mean": float(np.mean(ics)) if ics else 0.0, "n": int(len(ics))}
