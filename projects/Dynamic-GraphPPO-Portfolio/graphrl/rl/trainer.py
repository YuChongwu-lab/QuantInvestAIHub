# =============================================================================
# graphrl/rl/trainer.py —— PPO 训练器 & 课程学习调度
# =============================================================================

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch_geometric.data import Data

from graphrl.curriculum.config import CurriculumConfig
from graphrl.rl.env import PortfolioEnv
from graphrl.rl.policy import PolicyNet
from graphrl.utils.metrics import compute_signal, pearson_corr

logger = logging.getLogger(__name__)


# =============================================================================
# Rollout 数据类
# =============================================================================

class RolloutItem:
    __slots__ = ('data', 'a_raw', 'a_proj', 'logp', 'V', 'V_next', 'r', 'done')

    def __init__(self, data, a_raw, a_proj, logp, V, V_next, r, done):
        self.data   = data
        self.a_raw  = a_raw
        self.a_proj = a_proj
        self.logp   = logp
        self.V      = V
        self.V_next = V_next
        self.r      = r
        self.done   = done


# =============================================================================
# PPO 训练器
# =============================================================================

class PPOTrainer:
    def __init__(self, policy: PolicyNet, env: PortfolioEnv,
                 gamma=0.99, lam=0.95, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01,
                 lr=3e-4, minibatch_size=16, update_epochs=4, device=None,
                 loss_weights=None, morl_enable=False, morl_aux_weight=0.5,
                 morl_alpha: float = 1.0, morl_beta: float = 0.5,
                 use_grad_surgery: bool = True,
                 adaptive_weights: bool = True):
        self.policy = policy
        self.env    = env

        self.gamma      = gamma
        self.lam        = lam
        self.clip_ratio = clip_ratio
        self.vf_coef    = vf_coef
        self.ent_coef   = ent_coef

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.mb  = minibatch_size
        self.update_epochs = update_epochs
        self.device = device or next(policy.parameters()).device

        self.loss_weights    = loss_weights if loss_weights else {'mse': 1.0, 'ic': 1.0, 'return': 1.0}
        self.morl_enable     = morl_enable
        self.morl_aux_weight = morl_aux_weight
        self.morl_alpha      = float(morl_alpha)
        self.morl_beta       = float(morl_beta)
        self.use_grad_surgery  = use_grad_surgery
        self.adaptive_weights  = adaptive_weights
        self.loss_history    = {'rl': [], 'pred': []}

    def _project_conflicting_gradients(
        self, grads_list: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """PCGrad 简化版：对多任务梯度做冲突投影"""
        num_tasks = len(grads_list)
        if num_tasks <= 1:
            return grads_list[0] if grads_list else {}

        param_names  = list(grads_list[0].keys())
        merged_grads = {name: torch.zeros_like(grads_list[0][name]) for name in param_names}

        for i in range(num_tasks):
            grad_i    = grads_list[i]
            projected = {name: grad_i[name].clone() for name in param_names}

            for j in range(num_tasks):
                if i == j:
                    continue
                grad_j = grads_list[j]
                dot_product = sum(
                    (grad_i[name] * grad_j[name]).sum().item()
                    for name in param_names
                    if grad_i[name] is not None and grad_j[name] is not None
                )
                if dot_product < 0:
                    norm_j_sq = sum((grad_j[name] ** 2).sum().item() for name in param_names)
                    if norm_j_sq > 1e-8:
                        for name in param_names:
                            projected[name] = projected[name] - (dot_product / norm_j_sq) * grad_j[name]

            for name in param_names:
                merged_grads[name] += projected[name]

        for name in param_names:
            merged_grads[name] /= num_tasks
        return merged_grads

    def _update_adaptive_weights(self, rl_loss: float, pred_loss: float):
        self.loss_history['rl'].append(rl_loss)
        self.loss_history['pred'].append(pred_loss)
        if len(self.loss_history['rl']) > 20:
            self.loss_history['rl']   = self.loss_history['rl'][-20:]
            self.loss_history['pred'] = self.loss_history['pred'][-20:]
        if len(self.loss_history['rl']) >= 5:
            rl_rate    = np.mean(np.diff(self.loss_history['rl'][-5:]))
            pred_rate  = np.mean(np.diff(self.loss_history['pred'][-5:]))
            total_rate = abs(rl_rate) + abs(pred_rate) + 1e-6
            pred_weight = 0.2 + 0.6 * (abs(rl_rate) / total_rate)
            self.morl_aux_weight = float(np.clip(pred_weight, 0.2, 0.8))

    @torch.no_grad()
    def collect(self, horizon=128, stochastic=True) -> List[RolloutItem]:
        """与环境交互，采样一段轨迹"""
        buf: List[RolloutItem] = []
        s = self.env.reset()
        while len(buf) < horizon:
            a_raw, a_proj, logp, V = self.policy.act(s, stochastic=stochastic)
            s_next, r, done, info  = self.env.step(a_proj.cpu().numpy())
            if (not done) and (s_next is not None):
                _, V_next, _ = self.policy(s_next)
                V_next = V_next.detach()
            else:
                V_next = torch.zeros_like(V)
            buf.append(RolloutItem(
                s, a_raw, a_proj, logp, V, V_next,
                torch.tensor(r, dtype=torch.float32), done
            ))
            s = s_next if not done else self.env.reset()
        return buf

    def _gae(self, buf: List[RolloutItem]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Sequence[Data]
    ]:
        """计算 GAE 优势函数 & TD(λ) 回报"""
        rewards     = torch.stack([it.r     for it in buf]).to(self.device)
        dones       = torch.tensor([it.done for it in buf], dtype=torch.float32, device=self.device)
        values      = torch.stack([it.V     for it in buf]).to(self.device)
        next_values = torch.stack([it.V_next for it in buf]).to(self.device)

        deltas = rewards + self.gamma * next_values * (1.0 - dones) - values
        adv    = torch.zeros_like(rewards)
        gae    = 0.0
        for t in reversed(range(len(buf))):
            gae    = deltas[t] + self.gamma * self.lam * (1.0 - dones[t]) * gae
            adv[t] = gae
        returns = adv + values
        adv     = (adv - adv.mean()) / (adv.std() + 1e-8)

        actions_raw = torch.stack([it.a_raw for it in buf]).to(self.device)
        old_logp    = torch.stack([it.logp  for it in buf]).to(self.device)
        datas       = [it.data for it in buf]
        return actions_raw, old_logp, returns.detach(), adv.detach(), datas

    def update(self, buf: List[RolloutItem]):
        """PPO 更新主循环"""
        actions_raw, old_logp, returns, adv, datas = self._gae(buf)
        T   = actions_raw.size(0)
        idx = torch.randperm(T)

        for epoch in range(self.update_epochs):
            for start in range(0, T, self.mb):
                end = min(start + self.mb, T)
                b   = idx[start:end]

                alphas, values, _ = self._forward_batch([datas[i] for i in b])
                dist  = Dirichlet(alphas)
                logp  = dist.log_prob(actions_raw[b])
                ratio = torch.exp(logp - old_logp[b])

                surr1       = ratio * adv[b]
                surr2       = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv[b]
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss  = F.mse_loss(values, returns[b])
                entropy     = torch.mean(dist.entropy())
                rl_loss     = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                if self.morl_enable:
                    batch_datas = [datas[i] for i in b]
                    preds_list, ys_list = [], []
                    for d in batch_datas:
                        p = self.policy.predict_next_return(d)
                        y = d.y.to(p.device)
                        preds_list.append(p)
                        ys_list.append(y)
                    preds = torch.stack(preds_list, dim=0)
                    ys    = torch.stack(ys_list,    dim=0)

                    mse     = torch.mean((preds - ys) ** 2)
                    s       = compute_signal(preds, risks=None)
                    ic_per  = pearson_corr(s, ys, dim=1)
                    ic      = ic_per.mean()
                    pred_loss = self.morl_alpha * mse + self.morl_beta * (-ic)

                    if self.adaptive_weights and epoch == 0:
                        self._update_adaptive_weights(rl_loss.item(), pred_loss.item())

                    if self.use_grad_surgery:
                        self.opt.zero_grad()
                        rl_loss.backward(retain_graph=True)
                        rl_grads = {
                            name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                            for name, p in self.policy.named_parameters()
                        }
                        self.opt.zero_grad()
                        pred_loss.backward(retain_graph=True)
                        pred_grads = {
                            name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                            for name, p in self.policy.named_parameters()
                        }
                        merged_grads = self._project_conflicting_gradients([rl_grads, pred_grads])
                        self.opt.zero_grad()
                        for name, p in self.policy.named_parameters():
                            if name in merged_grads:
                                p.grad = rl_grads[name] + self.morl_aux_weight * merged_grads[name]
                    else:
                        loss = rl_loss + self.morl_aux_weight * pred_loss
                        self.opt.zero_grad()
                        loss.backward()
                else:
                    self.opt.zero_grad()
                    rl_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

    def _forward_batch(
        self, data_batch: List[Data]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alphas, vals, logits = [], [], []
        for d in data_batch:
            a, v, l = self.policy(d)
            alphas.append(a)
            vals.append(v)
            logits.append(l)
        return torch.stack(alphas), torch.stack(vals), torch.stack(logits)


# =============================================================================
# 课程学习训练调度
# =============================================================================

def run_curriculum_training(
    policy: PolicyNet,
    price_df,
    train_ds: List[Data],
    base_engine_cfg: Dict,
    base_reward_cfg: Dict,
    trainer_kwargs: Dict,
    curriculum_config: Optional[CurriculumConfig] = None,
    horizon: int = 128
):
    """逐阶段训练：根据课程配置渐进提高难度"""
    if curriculum_config is None:
        logger.info("未提供课程配置，使用默认3阶段配置")
        curriculum_config = CurriculumConfig.create_default()

    curriculum_config.validate_progression()
    curriculum_config.print_summary()

    device = next(policy.parameters()).device

    for stage_idx, stage in enumerate(curriculum_config.stages, 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"开始训练 {stage.name} ({stage_idx}/{len(curriculum_config.stages)})")
        logger.info(f"{'=' * 60}")

        policy.max_pos = stage.max_pos

        eng_cfg = dict(base_engine_cfg)
        eng_cfg['max_position_ratio']    = stage.max_pos
        eng_cfg['transaction_cost_rate'] = stage.cost

        rew_cfg = dict(base_reward_cfg)
        rew_cfg['lambda_turnover'] = stage.lambda_turnover

        env = PortfolioEnv(policy, train_ds, price_df, eng_cfg, rew_cfg)
        trainer = PPOTrainer(
            policy, env,
            **trainer_kwargs,
            device=device,
            loss_weights=stage.loss_weight
        )

        for ep in range(1, stage.epochs + 1):
            buf = trainer.collect(horizon=horizon, stochastic=True)
            trainer.update(buf)
            if ep % 5 == 0 or ep == stage.epochs:
                last_V = float(buf[-1].V.detach().cpu().item())
                logger.info(f"  [{stage.name}] Epoch {ep}/{stage.epochs} | V≈{last_V:,.2f}")

        logger.info(f" 完成 {stage.name}")
        logger.info(
            f"  配置: max_pos={stage.max_pos:.2%}, "
            f"cost={stage.cost:.4%}, λ_turn={stage.lambda_turnover:.3f}"
        )
