# =============================================================================
# graphrl/rl/env.py —— 投资组合 RL 环境
# =============================================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
from torch_geometric.data import Data

from graphrl.backtest.engine import EnhancedBacktestEngine
from graphrl.rl.policy import PolicyNet


class PortfolioEnv:
    """把 EnhancedBacktestEngine 包装成 RL 环境（Gym-like）"""

    def __init__(self, policy: PolicyNet, dataset: List[Data],
                 price_df, engine_cfg: Dict, reward_cfg: Dict):
        self.policy     = policy
        self.dataset    = dataset
        self.price_df   = price_df
        self.engine     = EnhancedBacktestEngine(**engine_cfg)
        self.reward_cfg = reward_cfg
        self.reset()

    def reset(self):
        self.engine.reset()
        self.t          = 0
        self.last_value = self.engine.initial_capital
        return self.dataset[self.t]

    def step(
        self, action_weights: np.ndarray
    ) -> Tuple[Optional[Data], float, bool, Dict]:
        date   = self.price_df.index[-len(self.dataset):][self.t]
        prices = self.price_df.loc[date].values

        if self.t == 0:
            current_value = self.engine.initial_capital
        else:
            current_shares = np.array(
                [self.engine.positions.get(j, 0.0) for j in range(len(prices))]
            )
            current_value = self.engine.cash + float(np.sum(current_shares * prices))

        _, trade_details = self.engine.execute_trades(action_weights, prices, current_value)

        pos_shares = np.array(
            [self.engine.positions.get(j, 0.0) for j in range(len(prices))]
        )
        V_t = self.engine.cash + float(np.sum(pos_shares * prices))

        r  = float(np.log(max(V_t, 1e-8)) - np.log(max(self.last_value, 1e-8)))
        r -= float(
            self.reward_cfg.get('lambda_turnover', 0.0)
            * trade_details.get('turnover_rate', 0.0)
        )

        self.last_value = V_t
        self.t         += 1
        done       = (self.t >= len(self.dataset))
        next_state = None if done else self.dataset[self.t]
        info       = {"V": V_t, "turnover": trade_details.get('turnover_rate', 0.0)}
        return next_state, r, done, info
