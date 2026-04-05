# =============================================================================
# graphrl/backtest/engine.py —— 回测引擎
# =============================================================================

import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data


class EnhancedBacktestEngine:
    """
    回测引擎（含交易成本 / 冲击成本 / 持仓约束）

    成员变量：
        positions:         dict[int -> shares]
        cash:              float
        portfolio_value:   List[float]
        returns:           List[float]
        benchmark_returns: List[float]
        benchmark_levels:  List[float]
    """

    def __init__(
        self,
        initial_capital:       float = 1_000_000,
        transaction_cost_rate: float = 0.002,
        impact_cost_factor:    float = 0.001,
        min_trade_amount:      float = 5_000,
        max_position_ratio:    float = 0.2,
    ):
        self.initial_capital       = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.impact_cost_factor    = impact_cost_factor
        self.min_trade_amount      = min_trade_amount
        self.max_position_ratio    = max_position_ratio
        self.reset()

    def reset(self):
        self.portfolio_value          = [self.initial_capital]
        self.positions                = {}
        self.cash                     = self.initial_capital
        self.transaction_costs        = []
        self.returns                  = []
        self.benchmark_returns        = []
        self.benchmark_levels         = []
        self.turnover_rates           = []
        self.holdings_count           = []
        self.holdings_count_effective = []

    def calculate_transaction_cost(
        self, trade_value: float, position_change_ratio: float
    ) -> float:
        base_cost   = abs(trade_value) * self.transaction_cost_rate
        impact_cost = abs(trade_value) * self.impact_cost_factor * abs(position_change_ratio)
        if abs(trade_value) < self.min_trade_amount:
            return base_cost + impact_cost + 5.0
        return base_cost + impact_cost

    def execute_trades(
        self,
        target_weights: np.ndarray,
        current_prices: np.ndarray,
        current_value:  float
    ) -> Tuple[float, Dict]:
        n_assets = len(target_weights)
        target_weights = np.clip(target_weights, 0, self.max_position_ratio)
        total_w = np.sum(target_weights)
        target_weights = target_weights / total_w if total_w > 0 else target_weights

        target_values  = target_weights * current_value
        target_shares  = target_values / np.maximum(current_prices, 1e-8)
        current_shares = np.array([self.positions.get(i, 0.0) for i in range(n_assets)])
        current_values = current_shares * current_prices

        est_trade_values = (target_shares - current_shares) * current_prices
        est_total        = float(np.sum(np.abs(est_trade_values)))
        est_turnover     = est_total / current_value if current_value > 0 else 0.0
        est_cost = (
            est_total * self.transaction_cost_rate
            + est_total * self.impact_cost_factor * est_turnover
        )
        scale = max(0.0, 1.0 - est_cost / max(current_value, 1e-8))
        if scale < 1.0:
            target_shares *= scale

        trade_shares      = target_shares - current_shares
        trade_values      = trade_shares * current_prices
        total_trade_value = float(np.sum(np.abs(trade_values)))
        turnover_rate     = total_trade_value / current_value if current_value > 0 else 0.0
        self.turnover_rates.append(turnover_rate)

        total_cost   = 0.0
        trade_details = {'trades': [], 'total_cost': 0.0, 'turnover_rate': turnover_rate}

        for i in range(n_assets):
            if abs(trade_values[i]) > self.min_trade_amount:
                current_weight       = current_values[i] / current_value if current_value > 0 else 0.0
                target_weight        = target_shares[i] * current_prices[i] / max(current_value, 1e-8)
                position_change_ratio = abs(target_weight - current_weight)
                trade_cost = self.calculate_transaction_cost(trade_values[i], position_change_ratio)
                total_cost += trade_cost
                trade_details['trades'].append({
                    'asset':  i,
                    'shares': float(trade_shares[i]),
                    'value':  float(trade_values[i]),
                    'cost':   float(trade_cost),
                    'price':  float(current_prices[i]),
                })
                self.positions[i] = float(target_shares[i])
            else:
                self.positions[i] = float(current_shares[i])

        trade_details['total_cost'] = float(total_cost)
        self.transaction_costs.append(float(total_cost))
        return float(total_cost), trade_details

    def run_backtest(
        self,
        model: nn.Module,
        test_dataset: List[Data],
        price_data: pd.DataFrame,
        benchmark_data: pd.Series,
        temperature: float = 2.0
    ) -> Dict:
        print(f"开始回测... (T={temperature})")
        model.eval()
        dates = price_data.index[-len(test_dataset):]
        all_trade_details = []
        self.reset()

        with torch.no_grad():
            prev_bench_level = None
            for i, (data, date) in enumerate(zip(test_dataset, dates)):
                try:
                    out         = model(data)
                    predictions = out['predictions'][0].cpu().numpy()
                    risks_list  = out.get('risks', None)
                    risks = (
                        risks_list[0].cpu().numpy() if risks_list is not None
                        else np.abs(predictions) + 0.01
                    )

                    scores     = predictions / (risks + 1e-8)
                    scores     = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                    exp_scores = np.exp(np.clip(scores / max(1e-6, float(temperature)), -20, 20))
                    sum_exp    = np.sum(exp_scores)
                    weights    = exp_scores / sum_exp if sum_exp > 0 else np.zeros_like(exp_scores)

                    current_prices = price_data.loc[date].values
                    if i == 0:
                        current_value = self.initial_capital
                        self.cash     = self.initial_capital
                    else:
                        current_shares = np.array(
                            [self.positions.get(j, 0.0) for j in range(len(current_prices))]
                        )
                        current_value = self.cash + np.sum(current_shares * current_prices)

                    transaction_cost, trade_detail = self.execute_trades(
                        weights, current_prices, current_value
                    )
                    all_trade_details.append(trade_detail)

                    pos_shares         = np.array(
                        [self.positions.get(j, 0.0) for j in range(len(current_prices))]
                    )
                    pos_values         = pos_shares * current_prices
                    holdings_value_now = float(np.sum(pos_values))
                    self.holdings_count.append(int(np.sum(pos_values > 1e-8)))
                    self.holdings_count_effective.append(
                        int(np.sum(np.abs(pos_values) >= self.min_trade_amount))
                    )
                    self.cash = current_value - holdings_value_now - transaction_cost

                    final_value = self.cash + holdings_value_now
                    self.portfolio_value.append(float(final_value))
                    if i > 0:
                        ret = (final_value - self.portfolio_value[-2]) / self.portfolio_value[-2]
                        self.returns.append(float(ret))

                    level = (
                        float(benchmark_data.loc[date]) if date in benchmark_data.index
                        else (self.benchmark_levels[-1] if self.benchmark_levels else 1.0)
                    )
                    self.benchmark_levels.append(level)
                    if prev_bench_level is not None and prev_bench_level > 0:
                        bench_ret = (level - prev_bench_level) / prev_bench_level
                        self.benchmark_returns.append(float(bench_ret))
                    prev_bench_level = level

                except Exception as e:
                    print(f"回测第 {i} 天出错: {e}")
                    print(traceback.format_exc())
                    continue

        return self.calculate_enhanced_metrics(all_trade_details)

    def calculate_enhanced_metrics(self, trade_details: List[Dict]) -> Dict:
        if len(self.returns) == 0:
            return self.get_default_metrics()

        returns           = np.array(self.returns,           dtype=float)
        transaction_costs = np.array(self.transaction_costs, dtype=float)
        turnover_rates    = np.array(self.turnover_rates,    dtype=float)
        holdings_cnt      = np.array(self.holdings_count,    dtype=float) if self.holdings_count else np.array([])
        holdings_cnt_eff  = np.array(self.holdings_count_effective, dtype=float) if self.holdings_count_effective else np.array([])

        total_return  = (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0.0
        annual_vol    = float(np.std(returns) * np.sqrt(252))
        sharpe_ratio  = float(annual_return / annual_vol) if annual_vol > 0 else 0.0

        cumulative   = np.cumprod(1 + returns)
        running_max  = np.maximum.accumulate(cumulative)
        drawdowns    = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        total_tc = float(np.sum(transaction_costs))
        avg_tc   = float(np.mean(transaction_costs)) if len(transaction_costs) else 0.0
        tc_ratio = float(total_tc / self.initial_capital)
        avg_turn = float(np.mean(turnover_rates)) if len(turnover_rates) else 0.0

        if len(self.benchmark_levels) >= 2:
            b0  = float(self.benchmark_levels[0])
            b1  = float(self.benchmark_levels[-1])
            btr = float((b1 - b0) / max(b0, 1e-12))
            bar = (1 + btr) ** (252 / len(returns)) - 1
        else:
            btr = 0.0
            bar = 0.0

        if len(self.benchmark_returns) == len(returns) and len(returns) > 0:
            bench                = np.array(self.benchmark_returns, dtype=float)
            active_daily         = returns - bench
            active_return_annual = float(np.mean(active_daily) * 252)
            tracking_error       = float(np.std(active_daily) * np.sqrt(252))
            information_ratio    = float(active_return_annual / tracking_error) if tracking_error > 0 else 0.0
        else:
            active_return_annual = 0.0
            tracking_error       = 0.0
            information_ratio    = 0.0

        excess_total_return = float((1 + total_return) / (1 + btr) - 1)
        excess_return       = float(annual_return - bar)

        return {
            'total_return':            float(total_return),
            'annual_return':           float(annual_return),
            'annual_volatility':       float(annual_vol),
            'sharpe_ratio':            float(sharpe_ratio),
            'max_drawdown':            float(max_drawdown),
            'win_rate':                float(np.sum(returns > 0) / len(returns)),
            'avg_holdings':            float(np.mean(holdings_cnt)) if holdings_cnt.size else 0.0,
            'avg_holdings_effective':  float(np.mean(holdings_cnt_eff)) if holdings_cnt_eff.size else 0.0,
            'total_transaction_costs': float(total_tc),
            'avg_transaction_cost':    avg_tc,
            'transaction_cost_ratio':  tc_ratio,
            'avg_turnover_rate':       avg_turn,
            'benchmark_total_return':  btr,
            'benchmark_annual_return': bar,
            'excess_total_return':     excess_total_return,
            'excess_return':           excess_return,
            'active_return_annual':    active_return_annual,
            'information_ratio':       float(information_ratio),
            'tracking_error':          float(tracking_error),
            'final_portfolio_value':   float(self.portfolio_value[-1]),
            'total_trades':            int(sum(len(td.get('trades', [])) for td in trade_details)),
        }

    def get_default_metrics(self) -> Dict:
        return {
            'total_return': 0.0, 'annual_return': 0.0, 'annual_volatility': 0.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
            'avg_holdings': 0.0, 'avg_holdings_effective': 0.0,
            'total_transaction_costs': 0.0, 'avg_transaction_cost': 0.0,
            'transaction_cost_ratio': 0.0, 'avg_turnover_rate': 0.0,
            'benchmark_total_return': 0.0, 'excess_total_return': 0.0, 'excess_return': 0.0,
            'information_ratio': 0.0, 'tracking_error': 0.0,
            'final_portfolio_value': float(self.initial_capital), 'total_trades': 0,
        }
