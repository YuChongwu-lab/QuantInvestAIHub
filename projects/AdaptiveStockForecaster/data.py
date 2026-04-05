"""
数据加载与工具函数
包含：
- set_seed        : 随机种子设置
- load_real_data  : 股票数据加载（支持 pkl / csv / excel）
- make_windows_xy : 滑动窗口构建
"""

import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_real_data(data_file_path: str,
                   max_assets: int = 30,
                   max_days: int = 500) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """加载股票数据，支持pkl、csv、excel三种格式"""
    print(f"加载数据: {data_file_path}")
    if data_file_path.endswith('.pkl'):
        panel_data = pd.read_pickle(data_file_path)
    elif data_file_path.endswith('.csv'):
        panel_data = pd.read_csv(data_file_path)
    else:
        panel_data = pd.read_excel(data_file_path)

    required = ['trade_date', 'ts_code', 'close'] #必需的列
    miss = [c for c in required if c not in panel_data.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])

    panel_data = panel_data.sort_values(['trade_date', 'ts_code']).reset_index(drop=True)
    price = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price = price.sort_index().ffill().bfill()
    price = price.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if max_assets is not None and max_assets > 0:
        price = price.iloc[:, :max_assets]
    if max_days is not None and max_days > 0:
        price = price.iloc[-max_days:]

    eq_ret = price.pct_change().mean(axis=1).fillna(0.0)
    bench = pd.Series(1000 * np.cumprod(1 + eq_ret), index=price.index, name='Benchmark')

    keep_dates = set(price.index)
    keep_codes = set(price.columns)
    mask = panel_data['trade_date'].isin(keep_dates) & panel_data['ts_code'].isin(keep_codes)
    panel_data = panel_data.loc[mask].copy()

    for col in panel_data.columns:
        if panel_data[col].dtype == 'object' or panel_data[col].dtype.name == 'category':
            panel_data[col] = panel_data[col].astype('category').cat.codes

    print(f"price shape: {price.shape}")
    return price, bench, panel_data


def make_windows_xy(X, y, seq_len, pred_len):
    """创建滑动窗口"""
    Xs, Ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i + seq_len])
        Ys.append(y[i + seq_len:i + seq_len + pred_len])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)
