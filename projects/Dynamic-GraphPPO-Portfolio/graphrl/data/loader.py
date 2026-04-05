# =============================================================================
# graphrl/data/loader.py —— 数据加载
# =============================================================================

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_real_data(
    data_file_path: str,
    max_assets: int = 190,
    max_days: int   = 2500,
    index_data_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    加载原始数据，返回 price（透视价格矩阵）、bench（基准净值）、
    panel_data（原始面板，已对齐筛选）

    Parameters
    ----------
    data_file_path  : 股票面板数据路径（.pkl / .csv / .xlsx）
    max_assets      : 最多使用的股票数（按列顺序截取），仅影响策略候选池
    max_days        : 最多使用的交易日数（取最近 N 天）
    index_data_path : 【推荐】指数行情文件路径（.pkl / .csv），含 trade_date 和 close 列。
                      若提供，benchmark 使用真实指数；否则退而求其次用完整股票池等权均值。
    """
    print(f"加载数据: {data_file_path}")
    if not isinstance(data_file_path, str):
        raise TypeError("data_file_path 应为字符串路径")
    if not os.path.exists(data_file_path):
        raise FileNotFoundError(f"未找到数据文件: {data_file_path}")

    if data_file_path.endswith('.pkl'):
        panel_data = pd.read_pickle(data_file_path)
    elif data_file_path.endswith('.csv'):
        panel_data = pd.read_csv(data_file_path)
    else:
        panel_data = pd.read_excel(data_file_path)

    print(f"数据形状: {panel_data.shape}")
    required = ['trade_date', 'ts_code', 'close']
    miss = [c for c in required if c not in panel_data.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    # 透视成价格矩阵（全部股票，未截断）
    price_full = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price_full = price_full.sort_index().ffill().bfill()
    price_full = price_full.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 先截断天数
    if max_days is not None and max_days > 0:
        price_full = price_full.iloc[-max_days:]

    # ---------------------------------------------------------------
    # Benchmark 计算：优先使用外部指数，否则用完整股票池等权均值
    # 关键：基准计算在 max_assets 截断之前，代表完整股票池表现
    # ---------------------------------------------------------------
    if index_data_path and os.path.exists(index_data_path):
        print(f"使用外部指数作为基准: {index_data_path}")
        if index_data_path.endswith('.pkl'):
            idx_df = pd.read_pickle(index_data_path)
        else:
            idx_df = pd.read_csv(index_data_path)
        if not pd.api.types.is_datetime64_any_dtype(idx_df['trade_date']):
            idx_df['trade_date'] = pd.to_datetime(idx_df['trade_date'])
        idx_df     = idx_df.set_index('trade_date').sort_index()
        idx_series = idx_df['close'].reindex(price_full.index).ffill().bfill()
        bench      = (idx_series / idx_series.iloc[0] * 1000.0).rename('Benchmark')
    else:
        print("未提供指数数据，使用完整股票池等权均值作为基准（次优选择）")
        eq_ret = price_full.pct_change().mean(axis=1).fillna(0.0)
        bench  = pd.Series(np.cumprod(1 + eq_ret) * 1000.0,
                           index=price_full.index, name='Benchmark')

    # 截断资产数（仅影响策略候选池，不影响基准）
    if max_assets is not None and max_assets > 0:
        price = price_full.iloc[:, :max_assets]
    else:
        price = price_full

    keep_dates = set(price.index)
    keep_codes = set(price.columns)
    mask       = (panel_data['trade_date'].isin(keep_dates) &
                  panel_data['ts_code'].isin(keep_codes))
    panel_data = panel_data.loc[mask].copy()

    print(f"处理后: 候选股票池 {price.shape[1]} 支，共 {price.shape[0]} 个交易日")
    return price, bench, panel_data
