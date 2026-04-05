# =============================================================================
# graphrl/rl/policy.py —— 策略网络 & 回测适配器
# =============================================================================

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch_geometric.data import Data


class PolicyNet(nn.Module):
    """
    Graph-RL 策略：
    - 编码器：E_DyGFormer.encode_tokens() → asset_tokens[A,H], cls[H]
    - Actor：对每个资产 token 输出 Dirichlet concentration α_i > 0
    - Critic：对 CLS（或资产 token 汇聚）输出状态值 V(s)
    """

    def __init__(self, encoder, hidden: int = 256,
                 alpha_scale: float = 1.0, alpha_min: float = 1e-3,
                 temperature: float = 1.5, max_pos: Optional[float] = 0.2,
                 use_cls_for_value: bool = True, freeze_encoder: bool = False):
        super().__init__()
        self.enc               = encoder
        self.A                 = encoder.num_assets
        self.H                 = encoder.hidden_dim
        self.temperature       = float(temperature)
        self.max_pos           = max_pos
        self.alpha_scale       = float(alpha_scale)
        self.alpha_min         = float(alpha_min)
        self.use_cls_for_value = use_cls_for_value
        self.pred_head         = nn.Linear(self.H, 1)

        self.actor = nn.Sequential(
            nn.Linear(self.H, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.H, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        asset_tokens, cls_token = self.enc.encode_tokens(data)
        raw   = self.actor(asset_tokens).squeeze(-1)
        alpha = F.softplus(raw).clamp(max=20.0) * self.alpha_scale + self.alpha_min
        logits_det = raw.clone()

        if self.use_cls_for_value and cls_token is not None:
            v_in = cls_token
        else:
            v_in = asset_tokens.mean(dim=0)
        V = self.critic(v_in)
        return alpha, V.squeeze(-1), logits_det

    def predict_next_return(self, data: Data) -> torch.Tensor:
        """利用共享编码器回归下一期收益 [A]"""
        asset_tokens, _ = self.enc.encode_tokens(data)
        return self.pred_head(asset_tokens).squeeze(-1)

    @torch.no_grad()
    def act(self, data: Data, stochastic: bool = True
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成动作（权重）：返回 action_raw, action_proj, logp, V"""
        device = next(self.parameters()).device
        alpha, V, logits_det = self.forward(data)
        if stochastic:
            dist   = Dirichlet(alpha)
            a_raw  = dist.sample()
            logp   = dist.log_prob(a_raw)
            a_proj = self._project_to_capped_simplex(a_raw, self.max_pos)
        else:
            z      = logits_det / max(1e-6, self.temperature)
            a_raw  = torch.softmax(z, dim=-1)
            logp   = torch.zeros((), device=device)
            a_proj = self._project_to_capped_simplex(a_raw, self.max_pos)
        return a_raw, a_proj, logp, V

    def weights_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """确定性权重（用于回测评估）"""
        z = logits / max(1e-6, self.temperature)
        w = torch.softmax(z, dim=-1)
        return self._project_to_capped_simplex(w, self.max_pos)

    @staticmethod
    def _project_to_capped_simplex(
        w: torch.Tensor, cap: Optional[float]
    ) -> torch.Tensor:
        """投影到 { y>=0, sum y=1, y_i <= cap } 的单纯形（迭代 water-filling）"""
        w = torch.clamp(w, min=0)
        if cap is None:
            s = w.sum()
            return (w / (s + 1e-8)) if s > 0 else torch.full_like(w, 1.0 / w.numel())
        x = w.clone()
        A = x.numel()
        cap_val = float(cap)
        for _ in range(10):
            x = torch.clamp(x, max=cap_val)
            s = x.sum()
            if s.abs() < 1e-8:
                x = torch.full_like(x, 1.0 / A)
                break
            diff = 1.0 - s.item()
            if abs(diff) < 1e-6:
                break
            mask  = (x < cap_val - 1e-12).float()
            denom = mask.sum().item()
            if denom <= 0:
                x = x / (s + 1e-8)
                break
            x = x + (diff / (denom + 1e-8)) * mask
        x = torch.clamp(x, min=0.0, max=cap_val)
        x = x / (x.sum() + 1e-8)
        return x


class PolicyModelAdapter(nn.Module):
    """
    把策略包装成 run_backtest 所需的 {predictions, risks} 输出格式。
    支持 EMA 权重平滑，避免过度换手。
    """

    def __init__(self, policy: PolicyNet,
                 smooth_weights: bool = True,
                 smooth_rho: float    = 0.3):
        super().__init__()
        self.policy         = policy
        self.smooth_weights = bool(smooth_weights)
        self.smooth_rho     = float(smooth_rho)
        self._ema_w: Optional[torch.Tensor] = None
        self.temperature    = float(getattr(policy, 'temperature', 1.5))

    def reset_state(self):
        self._ema_w = None

    @torch.no_grad()
    def forward(self, data: Data):
        _, _, logits = self.policy(data)
        if self.smooth_weights:
            T = max(1e-6, self.temperature)
            w = torch.softmax(logits / T, dim=-1)
            if self._ema_w is None:
                self._ema_w = w.clone()
            else:
                self._ema_w = self.smooth_rho * self._ema_w + (1.0 - self.smooth_rho) * w
            w_t   = self._ema_w / (self._ema_w.sum() + 1e-8)
            preds = T * torch.log(w_t + 1e-8)
            risks = torch.ones_like(preds)
        else:
            preds = logits
            risks = logits.abs() + 0.01
        return {'predictions': preds.unsqueeze(0), 'risks': risks.unsqueeze(0)}
