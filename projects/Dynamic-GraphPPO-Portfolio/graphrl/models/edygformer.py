# =============================================================================
# core/model.py —— E-DyGFormer 模型定义
# 包含：PositionalEncoding、VariableEmbedding、EdgeAwareMHSA、iTransformerMHSA、
#       EdgeAwareEncoderLayer、iTransformerEncoderLayer、E_DyGFormer
#
# 注意：encode_tokens() 直接作为 E_DyGFormer 的方法定义于类中，
#       删除了原代码中通过 setattr 注入的猴子补丁机制。
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple


# =============================================================================
# 1. 位置编码
# =============================================================================

class PositionalEncoding(nn.Module):
    """正余弦位置编码（与 Transformer 经典实现一致）。输入：[S, B, H]"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# =============================================================================
# 2. iTransformer 变量嵌入
# =============================================================================

class VariableEmbedding(nn.Module):
    """iTransformer 的变量/资产嵌入（替代时间位置编码）"""

    def __init__(self, num_variables: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.var_embedding = nn.Parameter(torch.randn(num_variables, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [A, B, H]
        return self.dropout(x + self.var_embedding.unsqueeze(1))


# =============================================================================
# 3. 边感知多头自注意力（Vanilla Transformer 用）
# =============================================================================

class EdgeAwareMHSA(nn.Module):
    """
    多头自注意力（支持边偏置 + 可选带符号聚合）
    - edge_bias_2d: [S, S] 加性偏置
    - use_signed_agg=True：attn = softmax(logits) * sign(edge_bias)
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1,
                 use_signed_agg: bool = True):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model = d_model
        self.nhead   = nhead
        self.head_dim = d_model // nhead
        self.use_signed_agg = use_signed_agg

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, src: torch.Tensor,
                edge_bias_2d: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        S, B, H = src.shape
        device = src.device

        q = self.q_proj(src); k = self.k_proj(src); v = self.v_proj(src)

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(S, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        q = reshape_heads(q); k = reshape_heads(k); v = reshape_heads(v)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        sign_gate = None
        if edge_bias_2d is not None:
            eb = edge_bias_2d.to(device)
            attn_logits = attn_logits + eb.unsqueeze(0).unsqueeze(0)
            if self.use_signed_agg:
                sign_gate = torch.sign(eb)
                sign_gate = torch.where(sign_gate == 0, torch.ones_like(sign_gate), sign_gate)
                sign_gate = sign_gate.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        if sign_gate is not None:
            attn = attn * sign_gate

        out = torch.matmul(attn, v)
        out = out.permute(2, 0, 1, 3).contiguous().view(S, B, H)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        attn_to_return = attn.mean(dim=1) if need_weights else None
        return out, attn_to_return


# =============================================================================
# 4. iTransformer 多头自注意力（在资产维度做注意力）
# =============================================================================

class iTransformerMHSA(nn.Module):
    """在资产/变量维度做多头自注意力"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.d_model  = d_model
        self.nhead    = nhead
        self.head_dim = d_model // nhead

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, src: torch.Tensor,
                asset_relation_bias: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        # src: [A, B, H]
        A, B, H = src.shape
        q = self.q_proj(src); k = self.k_proj(src); v = self.v_proj(src)

        def reshape_heads(t):
            return t.view(A, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        q = reshape_heads(q); k = reshape_heads(k); v = reshape_heads(v)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if asset_relation_bias is not None:
            attn_logits = attn_logits + asset_relation_bias.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)
        out  = torch.matmul(attn, v)
        out  = out.permute(2, 0, 1, 3).contiguous().view(A, B, H)
        out  = self.out_proj(out)
        out  = self.proj_drop(out)
        attn_to_return = attn.mean(dim=1) if need_weights else None
        return out, attn_to_return


# =============================================================================
# 5. 编码层
# =============================================================================

class EdgeAwareEncoderLayer(nn.Module):
    """标准 Transformer 编码层（带边偏置的 MHSA）"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, use_signed_agg: bool = False):
        super().__init__()
        self.mhsa  = EdgeAwareMHSA(d_model, nhead, dropout, use_signed_agg=use_signed_agg)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act     = nn.ReLU()
        self.drop2   = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop3   = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        attn_out, attn = self.mhsa(src, edge_bias_2d=edge_bias, need_weights=need_weights)
        src = self.norm1(src + self.drop1(attn_out))
        ff  = self.linear2(self.drop2(self.act(self.linear1(src))))
        src = self.norm2(src + self.drop3(ff))
        return src, attn


class iTransformerEncoderLayer(nn.Module):
    """iTransformer 编码层"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.mhsa  = iTransformerMHSA(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.act     = nn.ReLU()
        self.drop2   = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.drop3   = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                asset_relation_bias: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        attn_out, attn = self.mhsa(src, asset_relation_bias, need_weights)
        src = self.norm1(src + self.drop1(attn_out))
        ff  = self.linear2(self.drop2(self.act(self.linear1(src))))
        src = self.norm2(src + self.drop3(ff))
        return src, attn


# =============================================================================
# 6. E-DyGFormer 主模型
# =============================================================================

class E_DyGFormer(nn.Module):
    """
    Edge-aware Dynamic Graph Transformer (E-DyGFormer)

    encode_tokens() 直接定义为类方法，不再使用猴子补丁注入。
    """

    def __init__(self, node_feat_dim=64, hidden_dim=128,
                 num_layers=6, num_heads=8, max_seq_len=8192, dropout=0.1,
                 num_assets=50, seq_len=20,
                 edge_bias_scale=0.3, edge_bias_temp=2.0,
                 use_signed_agg: bool = False,
                 encoder_type: str = 'vanilla'):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.num_assets      = num_assets
        self.seq_len         = seq_len
        self.edge_bias_scale = float(edge_bias_scale)
        self.edge_bias_temp  = float(edge_bias_temp)
        self.encoder_type    = encoder_type.lower()

        if self.encoder_type == 'itransformer':
            self.node_embedding = nn.Linear(node_feat_dim * seq_len, hidden_dim)
            self.var_embedding  = VariableEmbedding(num_assets, hidden_dim, dropout)
            self.layers = nn.ModuleList([
                iTransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
                for _ in range(num_layers)
            ])
            self.pos_encoding = nn.Identity()
        else:
            self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
            self.pos_encoding   = PositionalEncoding(hidden_dim, dropout, max_len=max_seq_len)
            self.layers = nn.ModuleList([
                EdgeAwareEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout,
                                      use_signed_agg=use_signed_agg)
                for _ in range(num_layers)
            ])

        self.cls_token       = nn.Parameter(torch.randn(1, hidden_dim))
        self.asset_head      = nn.Linear(hidden_dim, 1)
        self.asset_risk_head = nn.Linear(hidden_dim, 1)

    # ── 辅助方法 ─────────────────────────────────────────────────────────────

    def _align_node_features(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        N, F = x.size()
        expected_N = self.num_assets * self.seq_len
        if N == expected_N:
            return x
        if N > expected_N:
            return x[-expected_N:]
        pad = torch.zeros(expected_N - N, F, device=device, dtype=x.dtype)
        return torch.cat([pad, x], dim=0)

    def _extract_edge_weights(self, edge_attr, edge_index: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        if edge_attr is None or edge_attr.numel() == 0:
            return torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
        ea = edge_attr.to(device).float()
        w  = ea if ea.dim() == 1 else (ea.squeeze(-1) if ea.size(-1) == 1 else ea[:, 0])
        if w.numel() != edge_index.size(1):
            return torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
        return w

    def _apply_edge_weight_transform(self, w: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        w = w.clamp(-1 + eps, 1 - eps)
        return torch.atanh(w) / max(1e-6, self.edge_bias_temp)

    def _build_edge_bias(self, data, S_total: int,
                         device: torch.device) -> Optional[torch.Tensor]:
        if getattr(data, 'edge_index', None) is None or data.edge_index.numel() == 0:
            return None
        bias = torch.zeros(S_total, S_total, device=device, dtype=torch.float32)
        edge_index = data.edge_index.to(device)
        w = self._extract_edge_weights(getattr(data, 'edge_attr', None), edge_index, device)
        if not torch.allclose(w, torch.ones_like(w)):
            w = self._apply_edge_weight_transform(w)
        src_nodes = (edge_index[0] + 1).clamp_(0, S_total - 1)
        dst_nodes = (edge_index[1] + 1).clamp_(0, S_total - 1)
        bias.index_put_((src_nodes, dst_nodes), w * self.edge_bias_scale, accumulate=True)
        if self.training and bias.abs().sum() > 0:
            mask = (bias != 0)
            if mask.any():
                keep = torch.rand_like(bias) >= 0.2
                bias = torch.where(mask & (~keep), torch.zeros_like(bias), bias)
        return bias

    def _build_asset_relation_bias(self, data,
                                   device: torch.device) -> Optional[torch.Tensor]:
        if getattr(data, 'edge_index', None) is None or data.edge_index.numel() == 0:
            return None
        A = self.num_assets
        bias = torch.zeros(A, A, device=device, dtype=torch.float32)
        edge_index = data.edge_index.to(device)
        w = self._extract_edge_weights(getattr(data, 'edge_attr', None), edge_index, device)
        last_step_offset = (self.seq_len - 1) * A
        mask = (edge_index[0] >= last_step_offset) & (edge_index[1] >= last_step_offset)
        if not mask.any():
            return None
        corr_edges   = edge_index[:, mask]
        corr_weights = w[mask]
        src = corr_edges[0] - last_step_offset
        dst = corr_edges[1] - last_step_offset
        valid_mask = (src >= 0) & (src < A) & (dst >= 0) & (dst < A)
        if not valid_mask.any():
            return None
        src = src[valid_mask]; dst = dst[valid_mask]
        corr_weights = self._apply_edge_weight_transform(corr_weights[valid_mask])
        bias[src, dst] = corr_weights * self.edge_bias_scale
        bias[dst, src] = corr_weights * self.edge_bias_scale
        return bias

    # ── 前向传播 ─────────────────────────────────────────────────────────────

    def forward(self, data_batch, store_attention: bool = False):
        if self.encoder_type == 'itransformer':
            single_fn = self._forward_single_itransformer
        else:
            single_fn = self._forward_single_edgeaware

        if not isinstance(data_batch, list):
            p, r, a = single_fn(data_batch, store_attention)
            out = {'predictions': p.unsqueeze(0), 'risks': r.unsqueeze(0)}
            if a is not None:
                out['attention_weights'] = [a]
            return out

        if len(data_batch) == 0:
            raise ValueError("空的数据批次")

        if len(data_batch) > 1:
            try:
                batch_data = Batch.from_data_list(data_batch)
                if self.encoder_type == 'vanilla':
                    preds, risks, attns = self._forward_batch_edgeaware(
                        batch_data, len(data_batch), store_attention)
                elif self.encoder_type == 'itransformer':
                    preds, risks, attns = self._forward_batch_itransformer(
                        batch_data, len(data_batch), store_attention)
                out = {'predictions': preds, 'risks': risks}
                if attns is not None:
                    out['attention_weights'] = attns
                return out
            except Exception as e:
                print(f"[Warning] 批处理失败，回退: {e}")

        preds, risks, attns = [], [], []
        for data in data_batch:
            p, r, a = single_fn(data, store_attention)
            preds.append(p); risks.append(r)
            if a is not None: attns.append(a)
        out = {'predictions': torch.stack(preds), 'risks': torch.stack(risks)}
        if attns:
            out['attention_weights'] = attns
        return out

    def _forward_single_edgeaware(self, data, store_attention: bool = False):
        device   = next(self.parameters()).device
        x        = self._align_node_features(data.x.to(device), device)
        node_emb = self.node_embedding(x)
        seq = torch.cat([self.cls_token, node_emb], dim=0)
        h   = self.pos_encoding(seq.unsqueeze(1))
        S   = 1 + self.num_assets * self.seq_len
        edge_bias_2d = self._build_edge_bias(data, S_total=S, device=device)
        attn_list = [] if store_attention else None
        for layer in self.layers:
            h, attn = layer(h, edge_bias=edge_bias_2d, need_weights=store_attention)
            if store_attention and attn is not None:
                attn_list.append(attn.detach().cpu())
        tokens_no_cls    = h[1:, 0, :].view(self.seq_len, self.num_assets, -1)
        k                = min(3, self.seq_len)
        last_step_tokens = tokens_no_cls[-k:, :, :].mean(dim=0)
        asset_scores = self.asset_head(last_step_tokens).squeeze(-1)
        asset_risks  = F.softplus(self.asset_risk_head(last_step_tokens)).squeeze(-1)
        attention_info = torch.stack(attn_list) if attn_list else None
        return asset_scores, asset_risks, attention_info

    def _forward_single_itransformer(self, data, store_attention: bool = False):
        device = next(self.parameters()).device
        x = self._align_node_features(data.x.to(device), device)
        x = x.view(self.seq_len, self.num_assets, -1).permute(1, 0, 2)
        x = x.reshape(self.num_assets, -1)
        h = self.node_embedding(x).unsqueeze(1)
        h = self.var_embedding(h)
        asset_relation_bias = self._build_asset_relation_bias(data, device)
        attn_list = [] if store_attention else None
        for layer in self.layers:
            h, attn = layer(h, asset_relation_bias, need_weights=store_attention)
            if store_attention and attn is not None:
                attn_list.append(attn.detach().cpu())
        asset_tokens = h[:, 0, :]
        asset_scores = self.asset_head(asset_tokens).squeeze(-1)
        asset_risks  = F.softplus(self.asset_risk_head(asset_tokens)).squeeze(-1)
        attention_info = torch.stack(attn_list) if attn_list else None
        return asset_scores, asset_risks, attention_info

    def _forward_batch_edgeaware(self, batch_data: Batch, batch_size: int,
                                 store_attention: bool = False):
        device     = next(self.parameters()).device
        expected_N = self.num_assets * self.seq_len
        batch_indices = batch_data.batch.to(device)
        aligned_nodes = []
        for b in range(batch_size):
            mask     = (batch_indices == b)
            sample_x = batch_data.x[mask].to(device)
            N_b = sample_x.size(0)
            if N_b != expected_N:
                if N_b > expected_N:
                    sample_x = sample_x[-expected_N:]
                else:
                    pad = torch.zeros(expected_N - N_b, sample_x.size(1),
                                      device=device, dtype=sample_x.dtype)
                    sample_x = torch.cat([pad, sample_x], dim=0)
            aligned_nodes.append(sample_x)
        x_aligned = torch.cat(aligned_nodes, dim=0)
        node_emb  = self.node_embedding(x_aligned)
        cls_tokens = self.cls_token.repeat(batch_size, 1)
        S          = expected_N + 1
        batch_seq  = torch.zeros(batch_size, S, self.hidden_dim, device=device, dtype=node_emb.dtype)
        for b in range(batch_size):
            batch_seq[b, 0]  = cls_tokens[b]
            start_idx = b * expected_N
            batch_seq[b, 1:] = node_emb[start_idx:start_idx + expected_N]
        h = self.pos_encoding(batch_seq.transpose(0, 1))
        edge_bias_2d = self._build_shared_edge_bias(batch_data, S, device)
        attn_list = [] if store_attention else None
        for layer in self.layers:
            h, attn = layer(h, edge_bias=edge_bias_2d, need_weights=store_attention)
            if store_attention and attn is not None:
                attn_list.append(attn)
        tokens_no_cls    = h[1:, :, :].view(self.seq_len, batch_size, self.num_assets, -1)
        k                = min(3, self.seq_len)
        last_step_tokens = tokens_no_cls[-k:].mean(dim=0)
        flat_tokens = last_step_tokens.reshape(-1, self.hidden_dim)
        asset_scores = self.asset_head(flat_tokens).squeeze(-1).view(batch_size, self.num_assets)
        asset_risks  = F.softplus(self.asset_risk_head(flat_tokens)).squeeze(-1).view(batch_size, self.num_assets)
        attention_info = torch.stack(attn_list) if attn_list else None
        return asset_scores, asset_risks, attention_info

    def _build_shared_edge_bias(self, batch_data: Batch, S_total: int,
                                device: torch.device) -> Optional[torch.Tensor]:
        if batch_data.edge_index.numel() == 0:
            return None
        bias = torch.zeros(S_total, S_total, device=device, dtype=torch.float32)
        edge_index = batch_data.edge_index.to(device)
        w = self._extract_edge_weights(getattr(batch_data, 'edge_attr', None), edge_index, device)
        if not torch.allclose(w, torch.ones_like(w)):
            w = self._apply_edge_weight_transform(w)
        batch_indices = batch_data.batch.to(device)
        edge_batch    = batch_indices[edge_index[0]]
        batch_size    = batch_indices.max().item() + 1
        N_per_sample  = self.num_assets * self.seq_len
        for b in range(batch_size):
            edge_mask    = (edge_batch == b)
            if not edge_mask.any(): continue
            sample_edges = edge_index[:, edge_mask]
            sample_w     = w[edge_mask]
            offset = b * N_per_sample
            src_nodes = (sample_edges[0] - offset + 1).clamp_(0, S_total - 1)
            dst_nodes = (sample_edges[1] - offset + 1).clamp_(0, S_total - 1)
            bias.index_put_((src_nodes, dst_nodes), sample_w * self.edge_bias_scale, accumulate=True)
        bias = bias / max(1, batch_size)
        if self.training:
            mask = (bias != 0)
            if mask.any():
                keep = torch.rand_like(bias) >= 0.2
                bias = torch.where(mask & (~keep), torch.zeros_like(bias), bias)
        return bias

    def _forward_batch_itransformer(self, batch_data: Batch, batch_size: int,
                                    store_attention: bool = False):
        device     = next(self.parameters()).device
        expected_N = self.num_assets * self.seq_len
        batch_indices = batch_data.batch.to(device)
        batch_asset_inputs = []
        for b in range(batch_size):
            mask     = (batch_indices == b)
            sample_x = batch_data.x[mask].to(device)
            N_b, F   = sample_x.size()
            if N_b != expected_N:
                if N_b > expected_N: sample_x = sample_x[-expected_N:]
                else:
                    pad = torch.zeros(expected_N - N_b, F, device=device, dtype=sample_x.dtype)
                    sample_x = torch.cat([pad, sample_x], dim=0)
            x = sample_x.view(self.seq_len, self.num_assets, F).permute(1, 0, 2).reshape(self.num_assets, -1)
            batch_asset_inputs.append(x)
        x_batch = torch.stack(batch_asset_inputs, dim=1)
        h = self.node_embedding(x_batch)
        h = self.var_embedding(h)
        asset_relation_bias = self._build_shared_asset_relation_bias(batch_data, batch_size, device)
        attn_list = [] if store_attention else None
        for layer in self.layers:
            h, attn = layer(h, asset_relation_bias, need_weights=store_attention)
            if store_attention and attn is not None: attn_list.append(attn)
        asset_tokens = h.permute(1, 0, 2)
        flat_tokens  = asset_tokens.reshape(-1, self.hidden_dim)
        asset_scores = self.asset_head(flat_tokens).squeeze(-1).view(batch_size, self.num_assets)
        asset_risks  = F.softplus(self.asset_risk_head(flat_tokens)).squeeze(-1).view(batch_size, self.num_assets)
        attention_info = torch.stack(attn_list) if attn_list else None
        return asset_scores, asset_risks, attention_info

    def _build_shared_asset_relation_bias(self, batch_data: Batch, batch_size: int,
                                          device: torch.device) -> Optional[torch.Tensor]:
        if batch_data.edge_index.numel() == 0:
            return None
        A = self.num_assets
        bias = torch.zeros(A, A, device=device, dtype=torch.float32)
        edge_index = batch_data.edge_index.to(device)
        w = self._extract_edge_weights(getattr(batch_data, 'edge_attr', None), edge_index, device)
        if torch.allclose(w, torch.ones_like(w)):
            return None
        batch_indices    = batch_data.batch.to(device)
        edge_batch       = batch_indices[edge_index[0]]
        N_per_sample     = A * self.seq_len
        last_step_offset = (self.seq_len - 1) * A
        for b in range(batch_size):
            edge_mask = (edge_batch == b)
            if not edge_mask.any(): continue
            sample_edges = edge_index[:, edge_mask]
            sample_w     = w[edge_mask]
            offset    = b * N_per_sample
            local_src = sample_edges[0] - offset
            local_dst = sample_edges[1] - offset
            valid     = (local_src >= last_step_offset) & (local_dst >= last_step_offset)
            if not valid.any(): continue
            asset_src = local_src[valid] - last_step_offset
            asset_dst = local_dst[valid] - last_step_offset
            edge_w    = sample_w[valid]
            valid_assets = (asset_src >= 0) & (asset_src < A) & (asset_dst >= 0) & (asset_dst < A)
            if valid_assets.any():
                asset_src = asset_src[valid_assets]
                asset_dst = asset_dst[valid_assets]
                edge_w    = self._apply_edge_weight_transform(edge_w[valid_assets])
                bias[asset_src, asset_dst] += edge_w * self.edge_bias_scale
                bias[asset_dst, asset_src] += edge_w * self.edge_bias_scale
        bias = bias / max(1, batch_size)
        if self.training and bias.abs().sum() > 0:
            mask = (bias != 0)
            if mask.any():
                keep = torch.rand_like(bias) >= 0.2
                bias = torch.where(mask & (~keep), torch.zeros_like(bias), bias)
        return bias

    # ── encode_tokens（直接作为类方法，无需猴子补丁）────────────────────────

    def encode_tokens(self, data: Data,
                      return_cls: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """返回 (asset_tokens[A,H], cls_token[H] or None)"""
        device = next(self.parameters()).device

        if self.encoder_type == 'itransformer':
            x = self._align_node_features(data.x.to(device), device)
            x = x.view(self.seq_len, self.num_assets, -1).permute(1, 0, 2)
            x = x.reshape(self.num_assets, -1)
            h = self.node_embedding(x).unsqueeze(1)
            h = self.var_embedding(h)
            asset_relation_bias = self._build_asset_relation_bias(data, device)
            for layer in self.layers:
                h, _ = layer(h, asset_relation_bias, need_weights=False)
            return h[:, 0, :], None   # iTransformer 不使用 CLS

        # Vanilla transformer
        x        = self._align_node_features(data.x.to(device), device)
        node_emb = self.node_embedding(x)
        seq = torch.cat([self.cls_token, node_emb], dim=0)
        h   = self.pos_encoding(seq.unsqueeze(1))
        S   = 1 + self.num_assets * self.seq_len
        edge_bias_2d = self._build_edge_bias(data, S_total=S, device=device)
        for layer in self.layers:
            h, _ = layer(h, edge_bias=edge_bias_2d, need_weights=False)
        tokens_no_cls    = h[1:, 0, :].view(self.seq_len, self.num_assets, -1)
        k                = min(3, self.seq_len)
        asset_tokens     = tokens_no_cls[-k:, :, :].mean(dim=0)
        cls_token        = h[0, 0, :].clone() if return_cls else None
        return asset_tokens, cls_token
