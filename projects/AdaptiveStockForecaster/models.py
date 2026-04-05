"""
模型定义
═══════════════════════════════════════════════════════════════════════════
包含：
- TransformerBlock   : 带细粒度门控的 Transformer 块
- PatchTSTBackbone   : PatchTST 骨干网络
- ProceedAdapter     : 门控生成器（概念漂移 → 多层门控参数）
"""

from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== Transformer Block ==============
class TransformerBlock(nn.Module):
    """带细粒度门控的Transformer块"""

    def __init__(self, d_model, nhead, dim_ffn, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.linear2 = nn.Linear(dim_ffn, d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()

    def _mh(self, x):
        B, P, D = x.shape
        return x.view(B, P, self.nhead, self.head_dim).permute(0, 2, 1, 3)

    def _merge(self, x):
        B, H, P, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, P, H * d)

    def forward(self, x,
                alpha_q_in=None, alpha_k_in=None, alpha_v_in=None, beta_o_out=None,
                alpha_l1_in=None, beta_l1_out=None,
                alpha_l2_in=None, beta_l2_out=None):
        # 多头注意力
        x_q = x * (1.0 + alpha_q_in).unsqueeze(1) if alpha_q_in is not None else x
        x_k = x * (1.0 + alpha_k_in).unsqueeze(1) if alpha_k_in is not None else x
        x_v = x * (1.0 + alpha_v_in).unsqueeze(1) if alpha_v_in is not None else x

        q = self._mh(self.q_proj(x_q))
        k = self._mh(self.k_proj(x_k))
        v = self._mh(self.v_proj(x_v))

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self._merge(attn)
        out = self.o_proj(attn)

        if beta_o_out is not None:
            out = out * (1.0 + beta_o_out.unsqueeze(1))

        x = x + self.dropout_attn(out)
        x = self.norm1(x)

        # FFN
        h1 = x * (1.0 + alpha_l1_in).unsqueeze(1) if alpha_l1_in is not None else x
        z1 = self.linear1(h1)
        if beta_l1_out is not None:
            z1 = z1 * (1.0 + beta_l1_out).unsqueeze(1)

        h2 = self.act(z1)
        h2 = h2 * (1.0 + alpha_l2_in).unsqueeze(1) if alpha_l2_in is not None else h2
        z2 = self.linear2(self.dropout_ffn(h2))
        if beta_l2_out is not None:
            z2 = z2 * (1.0 + beta_l2_out).unsqueeze(1)

        x = x + self.dropout_ffn(z2)
        x = self.norm2(x)
        return x


# ============== PatchTST Backbone ==============
class PatchTSTBackbone(nn.Module):
    """PatchTST模型骨干网络"""

    def __init__(self, seq_len, feature_dim, pred_len,
                 d_model=256, nhead=8, nlayers=4, dropout=0.1,
                 patch_len=6, stride=3, head_hidden=128):
        super().__init__()
        assert seq_len >= patch_len, "seq_len 必须 ≥ patch_len"
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.nlayers = nlayers

        self.patch_embed = nn.Conv1d(in_channels=feature_dim, out_channels=d_model,
                                     kernel_size=patch_len, stride=stride, padding=0, bias=True)
        self.padding = 0

        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches < 1:
            raise ValueError("patch 参数不合法，num_patches<1")
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        dim_ffn = d_model * 4
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ffn, dropout=dropout) for _ in range(nlayers)
        ])
        self.dim_ffn = dim_ffn

        self.head_in = nn.Linear(d_model, head_hidden)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(head_hidden, pred_len)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @torch.no_grad()
    def get_layer_meta(self) -> Dict[str, Any]:
        return {
            "patch_embed": {"type": "conv_weight", "ch_in": self.patch_embed.in_channels,
                            "ch_out": self.patch_embed.out_channels},
            "enc": {"type": "enc_layers", "nlayers": self.nlayers,
                    "d_model": self.d_model, "dim_ffn": self.dim_ffn},
            "head_in": {"type": "linear_gate", "in": self.head_in.in_features,
                        "out": self.head_in.out_features},
            "fc_out": {"type": "linear_gate", "in": self.fc_out.in_features,
                       "out": self.fc_out.out_features},
        }

    def _conv_with_weight_scaling(self, x_ft, alpha_in, beta_out):
        """卷积核缩放"""
        B, Cin, T = x_ft.shape
        Cout = self.patch_embed.out_channels
        K = self.patch_embed.kernel_size[0]
        w = self.patch_embed.weight

        alpha_exp = (1.0 + alpha_in).view(B, 1, Cin, 1)
        beta_exp = (1.0 + beta_out).view(B, Cout, 1, 1)
        w_scaled = w.unsqueeze(0) * beta_exp * alpha_exp

        w_grouped = w_scaled.view(B * Cout, Cin, K)
        x_grouped = x_ft.contiguous().view(1, B * Cin, T)

        out_grouped = F.conv1d(x_grouped, w_grouped, bias=None,
                               stride=self.stride, padding=self.padding, groups=B)
        out = out_grouped.view(B, Cout, -1)

        if self.patch_embed.bias is not None:
            out = out + self.patch_embed.bias.view(1, -1, 1)
        return out

    def forward(self, x, mod: Optional[Dict[str, Any]] = None):
        B = x.size(0)
        x_ft = x.transpose(1, 2)

        # Conv权重级缩放
        if mod is not None and "patch_embed" in mod and mod["patch_embed"] is not None:
            alpha_in = mod["patch_embed"].get("alpha_in", None)
            beta_out = mod["patch_embed"].get("beta_out", None)
            if alpha_in is not None and beta_out is not None:
                conv_out = self._conv_with_weight_scaling(x_ft, alpha_in, beta_out)
            else:
                conv_out = self.patch_embed(x_ft)
        else:
            conv_out = self.patch_embed(x_ft)

        tokens = conv_out.transpose(1, 2)
        tokens = tokens + self.pos_embed

        # Transformer layers
        if mod is not None and "enc" in mod and isinstance(mod["enc"], list):
            assert len(mod["enc"]) == len(self.blocks)
            for blk, gates in zip(self.blocks, mod["enc"]):
                tokens = blk(tokens,
                             alpha_q_in=gates.get("alpha_q_in", None),
                             alpha_k_in=gates.get("alpha_k_in", None),
                             alpha_v_in=gates.get("alpha_v_in", None),
                             beta_o_out=gates.get("beta_o_out", None),
                             alpha_l1_in=gates.get("alpha_l1_in", None),
                             beta_l1_out=gates.get("beta_l1_out", None),
                             alpha_l2_in=gates.get("alpha_l2_in", None),
                             beta_l2_out=gates.get("beta_l2_out", None))
        else:
            for blk in self.blocks:
                tokens = blk(tokens)

        h = tokens.mean(dim=1)

        # head_in
        if mod is not None and "head_in" in mod and mod["head_in"] is not None:
            alpha_in = mod["head_in"].get("alpha_in", None)
            beta_out = mod["head_in"].get("beta_out", None)
            if alpha_in is not None:
                h = h * (1.0 + alpha_in)
            y = self.head_in(h)
            if beta_out is not None:
                y = y * (1.0 + beta_out)
            h = self.act(y)
        else:
            h = self.act(self.head_in(h))

        # fc_out
        if mod is not None and "fc_out" in mod and mod["fc_out"] is not None:
            alpha_in = mod["fc_out"].get("alpha_in", None)
            beta_out = mod["fc_out"].get("beta_out", None)
            if alpha_in is not None:
                h = h * (1.0 + alpha_in)
            y = self.fc_out(h)
            if beta_out is not None:
                y = y * (1.0 + beta_out)
            return y
        else:
            return self.fc_out(h)


# ============== Proceed Adapter ==============
class ProceedAdapter(nn.Module):
    """门控生成器：概念漂移→多层门控参数"""

    def __init__(self, concept_dim, bottleneck_dim, layer_meta: Dict[str, Any], scale_mult=0.05):
        super().__init__()
        self.scale_mult = scale_mult
        self.trunk = nn.Sequential(
            nn.Linear(concept_dim, bottleneck_dim), nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim), nn.ReLU()
        )
        self.heads = nn.ModuleDict()

        # patch_embed
        pe = layer_meta["patch_embed"]
        self.heads["pe_alpha_in"] = nn.Linear(bottleneck_dim, pe["ch_in"])
        self.heads["pe_beta_out"] = nn.Linear(bottleneck_dim, pe["ch_out"])

        # Encoder
        enc = layer_meta["enc"]
        L = enc["nlayers"]
        D = enc["d_model"]
        FF = enc["dim_ffn"]

        self.L, self.D, self.FF = L, D, FF

        gate_dims = {
            "alpha_q_in": D, "alpha_k_in": D, "alpha_v_in": D, "beta_o_out": D,
            "alpha_l1_in": D, "beta_l1_out": FF, "alpha_l2_in": FF, "beta_l2_out": D,
        }

        for gate_name, gate_dim in gate_dims.items():
            self.heads[f"enc_{gate_name}"] = nn.Linear(bottleneck_dim, L * gate_dim)

        # Prediction Head
        hi = layer_meta["head_in"]
        self.heads["hi_alpha_in"] = nn.Linear(bottleneck_dim, hi["in"])
        self.heads["hi_beta_out"] = nn.Linear(bottleneck_dim, hi["out"])

        fo = layer_meta["fc_out"]
        self.heads["fo_alpha_in"] = nn.Linear(bottleneck_dim, fo["in"])
        self.heads["fo_beta_out"] = nn.Linear(bottleneck_dim, fo["out"])

    def forward(self, delta):
        B = delta.size(0)
        h = self.trunk(delta)
        sm = self.scale_mult
        out: Dict[str, Any] = {}

        out["patch_embed"] = {
            "alpha_in": torch.tanh(self.heads["pe_alpha_in"](h)) * sm,
            "beta_out": torch.tanh(self.heads["pe_beta_out"](h)) * sm,
        }

        L, D, FF = self.L, self.D, self.FF

        def split(name, d):
            vec = torch.tanh(self.heads[name](h)) * sm
            return vec.view(B, L, d)

        enc_list = []
        a_q = split("enc_alpha_q_in", D)
        a_k = split("enc_alpha_k_in", D)
        a_v = split("enc_alpha_v_in", D)
        b_o = split("enc_beta_o_out", D)
        a_l1 = split("enc_alpha_l1_in", D)
        b_l1 = split("enc_beta_l1_out", FF)
        a_l2 = split("enc_alpha_l2_in", FF)
        b_l2 = split("enc_beta_l2_out", D)

        for l in range(L):
            enc_list.append({
                "alpha_q_in": a_q[:, l, :], "alpha_k_in": a_k[:, l, :],
                "alpha_v_in": a_v[:, l, :], "beta_o_out": b_o[:, l, :],
                "alpha_l1_in": a_l1[:, l, :], "beta_l1_out": b_l1[:, l, :],
                "alpha_l2_in": a_l2[:, l, :], "beta_l2_out": b_l2[:, l, :],
            })
        out["enc"] = enc_list

        out["head_in"] = {
            "alpha_in": torch.tanh(self.heads["hi_alpha_in"](h)) * sm,
            "beta_out": torch.tanh(self.heads["hi_beta_out"](h)) * sm,
        }
        out["fc_out"] = {
            "alpha_in": torch.tanh(self.heads["fo_alpha_in"](h)) * sm,
            "beta_out": torch.tanh(self.heads["fo_beta_out"](h)) * sm,
        }

        return out
