"""VM1: PyTorch port of the official improved-MeanFlow DiT (external/iMeanFlow @ bf60cd7, models/imfDiT.py,
models/embedder.py, models/torch_models.py), adapted from class-conditional 2-D latents to PPG-conditioned 1-D ECG.

Ported unchanged (same structure, init and constants):
  * RoPE multi-head self-attention with QK RMSNorm, no qkv/out bias               (imfDiT.py RoPEAttention)
  * pre-norm block, SwiGLU MLP (ratio 8/3), zero-initialised per-channel residual gates  (TransformerBlock)
  * RMSNorm + zero-initialised linear final layer                                 (FinalLayer)
  * in-context conditioning tokens: learned token groups + broadcast embeddings for h, omega = CFG scale
    (embedded as 1 - 1/omega), CFG interval start / end                           (_build_sequence)
  * shared trunk of depth - aux_head_depth blocks, then a u head and an auxiliary v head of aux_head_depth blocks each
  * "scaled_variance" init N(0, c / sqrt(fan_in)), c = 0.32 for blocks, 1.0 for embedders; tokens N(0, 1 / sqrt(H))
  * the network is conditioned on h = t - r only, never on t                      (imfDiT.__call__ comment)

Adapted for the task (docs/VM1_VANILLA_IMF_PREREGISTRATION.md):
  * 2-D patches -> 1-D patches (Conv1d kernel = stride = patch), flax xavier_uniform fan convention kept
  * the class label -> the PPG window: one learned condition token per PPG patch plus that patch's embedding
    (in-context; the official class-token group has one token per slot plus a broadcast label embedding);
    the "null class" of CFG -> a learned null PPG embedding of the same shape
  * RoPE in its real (cos/sin) form so torch.func.jvp applies; numerically equal to the complex form
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_linear(fan_in: int, fan_out: int, bias: bool, c: float, zero: bool = False) -> nn.Linear:
    lin = nn.Linear(fan_in, fan_out, bias=bias)
    if zero:
        nn.init.zeros_(lin.weight)
    else:
        nn.init.normal_(lin.weight, std=c / math.sqrt(fan_in))
    if bias:
        nn.init.zeros_(lin.bias)
    return lin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int, c: float):
        super().__init__()
        self.w1 = scaled_linear(dim, hidden, False, c)
        self.w3 = scaled_linear(dim, hidden, False, c)
        self.w2 = scaled_linear(hidden, dim, False, c)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def rope_tables(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    return torch.cos(ang), torch.sin(ang)          # [L, head_dim/2] each


def apply_rope(x, cos, sin):
    """x [B, L, nh, hd]; pairs (x[..., 0::2], x[..., 1::2]) rotated = (a + ib) * e^{i ang} (the official complex form)."""
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, c: float):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (scaled_linear(dim, dim, False, c) for _ in range(4))
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        B, L, H = x.shape
        q = apply_rope(self.q_norm(self.q_proj(x).view(B, L, self.n_heads, self.head_dim)), cos, sin)
        k = apply_rope(self.k_norm(self.k_proj(x).view(B, L, self.n_heads, self.head_dim)), cos, sin)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        att = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.head_dim), dim=-1)
        return self.out_proj((att @ v).transpose(1, 2).reshape(B, L, H))


class Block(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, c: float):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = RoPEAttention(dim, n_heads, c)
        self.mlp = SwiGLU(dim, int(dim * mlp_ratio), c)
        self.attn_scale = nn.Parameter(torch.zeros(dim))
        self.mlp_scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin) * self.attn_scale
        return x + self.mlp(self.norm2(x)) * self.mlp_scale


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch: int, out_ch: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = scaled_linear(dim, patch * out_ch, True, 1.0, zero=True)

    def forward(self, x):
        return self.linear(self.norm(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, c: float, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(scaled_linear(freq_dim, dim, True, c), nn.SiLU(), scaled_linear(dim, dim, True, c))

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.reshape(-1, 1).float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class PatchEmbed1d(nn.Module):
    def __init__(self, patch: int, in_ch: int, dim: int):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv1d(in_ch, dim, kernel_size=patch, stride=patch)
        bound = math.sqrt(6.0 / (patch * in_ch + dim))     # flax xavier_uniform(in_axis=(0,1,2), out_axis=-1)
        nn.init.uniform_(self.proj.weight, -bound, bound)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):                                   # [B, C, T] -> [B, T/p, H]
        return self.proj(x).transpose(1, 2)


class ImfDiT1d(nn.Module):
    def __init__(self, seq_len: int = 512, patch: int = 8, in_ch: int = 1, hidden: int = 192, depth: int = 6,
                 n_heads: int = 6, mlp_ratio: float = 8 / 3, aux_head_depth: int = 4,
                 num_time_tokens: int = 4, num_cfg_tokens: int = 4, num_interval_tokens: int = 2,
                 token_init_constant: float = 1.0, embedding_init_constant: float = 1.0,
                 weight_init_constant: float = 0.32):
        super().__init__()
        assert seq_len % patch == 0 and hidden % n_heads == 0 and depth > aux_head_depth
        self.patch, self.in_ch = patch, in_ch
        self.n_patches = seq_len // patch
        ce, ct, cw = embedding_init_constant, token_init_constant, weight_init_constant
        self.x_embedder = PatchEmbed1d(patch, in_ch, hidden)
        self.ppg_embedder = PatchEmbed1d(patch, 1, hidden)
        self.null_ppg = nn.Parameter(torch.randn(self.n_patches, hidden) * (ce / math.sqrt(hidden)))
        self.h_embedder = TimestepEmbedder(hidden, ce)
        self.omega_embedder = TimestepEmbedder(hidden, ce)
        self.cfg_t_start_embedder = TimestepEmbedder(hidden, ce)
        self.cfg_t_end_embedder = TimestepEmbedder(hidden, ce)
        tok = lambda n: nn.Parameter(torch.randn(n, hidden) * (ct / math.sqrt(hidden)))  # noqa: E731
        self.cond_tokens = tok(self.n_patches)         # the official class-token group, one per PPG patch
        self.omega_tokens = tok(num_cfg_tokens)
        self.t_min_tokens = tok(num_interval_tokens)
        self.t_max_tokens = tok(num_interval_tokens)
        self.time_tokens = tok(num_time_tokens)
        self.prefix = self.n_patches + num_cfg_tokens + 2 * num_interval_tokens + num_time_tokens
        cos, sin = rope_tables(hidden // n_heads, self.prefix + self.n_patches)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        blk = lambda: Block(hidden, n_heads, mlp_ratio, cw)  # noqa: E731
        self.shared_blocks = nn.ModuleList([blk() for _ in range(depth - aux_head_depth)])
        self.u_heads = nn.ModuleList([blk() for _ in range(aux_head_depth)])
        self.v_heads = nn.ModuleList([blk() for _ in range(aux_head_depth)])
        self.u_final_layer = FinalLayer(hidden, patch, in_ch)
        self.v_final_layer = FinalLayer(hidden, patch, in_ch)

    def _cond(self, ppg, null):
        """Condition tokens + PPG patch embeddings; rows with null=True get the learned null embedding instead."""
        emb = self.ppg_embedder(ppg)                                            # [B, P, H]
        if null is not None:
            emb = torch.where(null.reshape(-1, 1, 1), self.null_ppg.expand_as(emb), emb)
        return self.cond_tokens + emb

    def forward(self, x, h, omega, t_min, t_max, ppg, null=None, need_v: bool = True):
        """x, ppg: [B, 1, T]; h, omega, t_min, t_max: [B] or [B, 1, 1]; null: [B] bool or None -> (u, v or None)."""
        B = x.shape[0]
        flat = lambda a: a.reshape(B)  # noqa: E731
        seq = torch.cat([
            self._cond(ppg, null),
            self.omega_tokens + self.omega_embedder(1 - 1 / flat(omega))[:, None],
            self.t_min_tokens + self.cfg_t_start_embedder(flat(t_min))[:, None],
            self.t_max_tokens + self.cfg_t_end_embedder(flat(t_max))[:, None],
            self.time_tokens + self.h_embedder(flat(h))[:, None],
            self.x_embedder(x),
        ], dim=1)
        cos, sin = self.rope_cos, self.rope_sin
        for b in self.shared_blocks:
            seq = b(seq, cos, sin)
        u_seq = seq
        for b in self.u_heads:
            u_seq = b(u_seq, cos, sin)
        u = self._unpatch(self.u_final_layer(u_seq[:, self.prefix:]))
        if not need_v:
            return u, None
        v_seq = seq
        for b in self.v_heads:
            v_seq = b(v_seq, cos, sin)
        return u, self._unpatch(self.v_final_layer(v_seq[:, self.prefix:]))

    def _unpatch(self, tokens):                                                 # [B, P, p*C] -> [B, C, P*p]
        B, P, _ = tokens.shape
        return tokens.reshape(B, P, self.patch, self.in_ch).permute(0, 3, 1, 2).reshape(B, self.in_ch, P * self.patch)


def count_params(model: nn.Module, inference: bool = False) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if not (inference and (n.startswith("v_heads") or n.startswith("v_final_layer"))))


ARCHS = {  # S: size-matched to the 4.3 M PENGUIN backbone; B: the official iMF-B/2 trunk (patch 8 in 1-D)
    "S": dict(hidden=192, depth=6, n_heads=6, aux_head_depth=4),
    "B": dict(hidden=768, depth=12, n_heads=12, aux_head_depth=8),
}


def load_official_weights(model: ImfDiT1d, path: str) -> dict:
    """Initialise from the official PyTorch inference checkpoint (Lyy0725/iMF, e.g. iMF-B-2.pth).

    Every tensor whose name and shape match is copied (trunk, u head, h / omega / interval embedders and tokens,
    u final-layer norm). The checkpoint has no auxiliary v head, so v_heads.i is copied from u_heads.i. Shape-
    incompatible or task-specific tensors keep their fresh init: the 2x2x4 image patch embedder and output projection,
    the ImageNet class table and class tokens (replaced by the PPG condition)."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    src = {k.removeprefix("net.").replace("._flax_linear", ""): v for k, v in sd.items()}
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in src.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v.clone(); loaded.append(k)
        else:
            skipped.append(k)
    copied_v = []
    for k in own:
        if k.startswith("v_heads."):
            s = "u_heads." + k[len("v_heads."):]
            if s in src and src[s].shape == own[k].shape:
                own[k] = src[s].clone(); copied_v.append(k)
    model.load_state_dict(own)
    fresh = [k for k in own if k not in loaded and k not in copied_v]
    return {"n_loaded": len(loaded), "n_v_copied_from_u": len(copied_v), "skipped_from_ckpt": skipped, "fresh_init": fresh,
            "params_loaded": int(sum(src[k].numel() for k in loaded)),
            "params_fresh": int(sum(own[k].numel() for k in fresh))}
