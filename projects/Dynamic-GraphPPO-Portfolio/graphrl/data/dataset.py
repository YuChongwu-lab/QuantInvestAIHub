# =============================================================================
# graphrl/data/dataset.py —— 图数据集构建
# =============================================================================

import traceback
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def _construct_time_edges(
    seq_len: int, n_stocks: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为每个资产的时间序列生成双向边（边权为 0）"""
    time_steps    = torch.arange(seq_len - 1)
    asset_indices = torch.arange(n_stocks)

    src_forward = time_steps.repeat_interleave(n_stocks) * n_stocks + asset_indices.repeat(seq_len - 1)
    dst_forward = (time_steps + 1).repeat_interleave(n_stocks) * n_stocks + asset_indices.repeat(seq_len - 1)

    edge_index = torch.cat([
        torch.stack([src_forward, dst_forward], dim=0),
        torch.stack([dst_forward, src_forward], dim=0),
    ], dim=1)
    edge_attr = torch.zeros(edge_index.size(1), dtype=torch.float32)
    return edge_index, edge_attr


def create_enhanced_dataset(
    panel_data: pd.DataFrame,
    seq_len: int            = 60,
    corr_window: int        = 10,
    corr_threshold: float   = 0.3,
    feature_cols: Optional[List[str]] = None,
    categorical_handling: str = 'drop'
) -> Tuple[List[Data], List[str]]:
    """
    使用原始面板中的数值特征构建图数据集。

    节点 = 资产×时间；节点特征 = 数值特征列，滚动窗口标准化。
    边类型：
      - 时间边：同一资产 t↔t+1 双向边（边权 w=0）
      - 相关边：最近 corr_window 天收益滚动相关，|corr|>阈值的资产对，
                双向边，edge_attr = 带符号相关系数 ∈ [-0.999, 0.999]
    """
    print(f"创建图数据集... (corr_window={corr_window}, corr_threshold={corr_threshold})")

    required = ['trade_date', 'ts_code', 'close']
    miss = [c for c in required if c not in panel_data.columns]
    if miss:
        raise ValueError(f"panel_data 缺少必要列: {miss}")

    if not pd.api.types.is_datetime64_any_dtype(panel_data['trade_date']):
        panel_data = panel_data.copy()
        panel_data['trade_date'] = pd.to_datetime(panel_data['trade_date'])
    panel_data = panel_data.sort_values(['trade_date', 'ts_code'])

    if feature_cols is None:
        feature_cols = [c for c in panel_data.columns if c not in ('trade_date', 'ts_code', 'close')]

    all_feat_cols    = feature_cols[:]
    numeric_cols     = panel_data[all_feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = sorted(list(set(all_feat_cols) - set(numeric_cols)))

    if non_numeric_cols:
        if categorical_handling not in ('drop', 'factorize'):
            raise ValueError(f"未知的 categorical_handling={categorical_handling}（应为 'drop' 或 'factorize'）")
        if categorical_handling == 'drop':
            print(f"[create] 发现非数值列(已跳过)：{non_numeric_cols[:8]}{'  ...' if len(non_numeric_cols) > 8 else ''}")
        else:
            print(f"[create] 发现非数值列(做整数编码)：{non_numeric_cols[:8]}{'  ...' if len(non_numeric_cols) > 8 else ''}")
            for col in non_numeric_cols:
                codes, _ = pd.factorize(panel_data[col], sort=True)
                codes = codes.astype(np.int32)
                codes[codes < 0] = 0
                panel_data[col + "_code"] = codes
            numeric_cols += [c + "_code" for c in non_numeric_cols]

    feature_cols = numeric_cols

    price = panel_data.pivot(index='trade_date', columns='ts_code', values='close')
    price = price.sort_index().replace([np.inf, -np.inf], np.nan).ffill().bfill()
    price = price.dropna(axis=0, how='all').dropna(axis=1, how='all')

    if price.shape[0] < seq_len + 2:
        print(f"[create] 有效交易日不足：T={price.shape[0]}，seq_len+2={seq_len + 2}")
        return [], []
    n_stocks = price.shape[1]
    if n_stocks == 0:
        print("[create] 资产数为 0，退出。")
        return [], []

    dates   = price.index
    returns = price.pct_change().fillna(0.0)

    feat_mats = []
    for col in feature_cols:
        mat = panel_data.pivot(index='trade_date', columns='ts_code', values=col)
        mat = mat.reindex(index=dates, columns=price.columns)
        mat = mat.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        feat_mats.append(mat.values)
    feat_tensor = np.stack(feat_mats, axis=0).transpose(1, 2, 0)  # [T, n_stocks, F]
    feature_dim = feat_tensor.shape[-1]

    dataset        = []
    corr_window    = max(2, int(corr_window))
    corr_threshold = float(corr_threshold)
    first_error    = None

    for i in range(seq_len, len(dates) - 2):
        try:
            # 1) 节点特征（滚动窗口标准化）
            slab       = feat_tensor[i - seq_len:i].astype(np.float32, copy=False)  # [T, A, F]
            node_feats = slab.reshape(-1, slab.shape[-1])                            # [T*A, F]
            start_idx  = max(0, i - corr_window)
            rolling_window_feats = feat_tensor[start_idx:i].reshape(-1, slab.shape[-1])
            rolling_mean = np.mean(rolling_window_feats, axis=0)
            rolling_std  = np.std(rolling_window_feats,  axis=0)
            x_np = (node_feats - rolling_mean) / (rolling_std + 1e-8)
            x    = torch.FloatTensor(x_np)

            # 2) 时间边
            time_edge_index, time_edge_attr = _construct_time_edges(seq_len, n_stocks)

            # 3) 相关边
            recent = returns.iloc[i - corr_window:i].values
            if recent.shape[0] >= 2:
                corr        = np.corrcoef(recent.T)
                triu_indices = np.triu_indices(n_stocks, k=1)
                j_indices    = triu_indices[0]
                k_indices    = triu_indices[1]
                corr_values  = corr[j_indices, k_indices]
                mask         = np.abs(corr_values) > corr_threshold
                valid_j      = j_indices[mask]
                valid_k      = k_indices[mask]
                valid_w      = corr_values[mask]

                if len(valid_j) > 0:
                    last_step_offset  = (seq_len - 1) * n_stocks
                    src_nodes         = last_step_offset + valid_j
                    dst_nodes         = last_step_offset + valid_k
                    valid_w           = np.clip(valid_w, -0.999, 0.999)
                    corr_edge_src     = np.concatenate([src_nodes, dst_nodes])
                    corr_edge_dst     = np.concatenate([dst_nodes, src_nodes])
                    corr_edge_weights = np.concatenate([valid_w, valid_w])
                    corr_edge_index   = torch.LongTensor(np.stack([corr_edge_src, corr_edge_dst]))
                    corr_edge_attr    = torch.FloatTensor(corr_edge_weights)
                    edge_index        = torch.cat([time_edge_index, corr_edge_index], dim=1)
                    edge_attr         = torch.cat([time_edge_attr,  corr_edge_attr],  dim=0)
                else:
                    edge_index = time_edge_index
                    edge_attr  = time_edge_attr
            else:
                edge_index = time_edge_index
                edge_attr  = time_edge_attr

            # 4) 目标：下一天横截面收益
            y = torch.FloatTensor(returns.iloc[i + 1].values).to(torch.float32)

            dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        except Exception as e:
            if first_error is None:
                first_error = f"首个样本构建失败 @i={i}：{e}\n{traceback.format_exc()}"
            continue

    if first_error and len(dataset) == 0:
        raise RuntimeError(first_error)

    print(f"成功创建 {len(dataset)} 个样本（节点特征维度 F={feature_dim}，资产数={n_stocks}）")
    preview = feature_cols[:8] if isinstance(feature_cols, list) else []
    print(f"使用原始特征列: {preview}{'  ...' if isinstance(feature_cols, list) and len(feature_cols) > 8 else ''}")
    return dataset, feature_cols
