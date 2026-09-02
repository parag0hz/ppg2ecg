"""R3 — disentangled target-side rhythm fusion + adaptive rhythm gate.

Frozen by docs/R3_DISENTANGLED_RHYTHM_FUSION_PREREGISTRATION.md (3d779fc). Protocol machinery only: the
RhythmCrossFusionAdapter (section 6), the Adaptive Rhythm Gate and its CONST control (section 7), the
MeanFlowS5 subclass that inserts the fusion at z_e (hook audit), initialisation with a fixed construction
order (section 9), the total verdict function (section 17) and the ORACLE reading (section 17.3).
The scaffold enters u() as channel 1 of the R2 [B,2,T] carrier and is split before any backbone call;
backbone.pre_conv_ppg receives the PPG channel only. No training loop, no data access.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.flow.rhythm_transfer import split_ppg2

FS, T_LEN = 128, 1024
D_MODEL, N_HEADS = 32, 4
TOK_K, TOK_S, TOK_P, N_TOKENS = 7, 4, 3, 256
PE_BASE = 10000.0
GATE_HIDDEN, GATE_POOL, GATE_INIT_P = 16, 33, 0.90
GATE_INIT_BIAS = math.log(GATE_INIT_P / (1.0 - GATE_INIT_P))            # logit(0.90) = ln 9
STEPS, CKPT_STEPS, PREFLIGHT_STEPS, BUDGET_GPU_HOURS = 2200, (0, 550, 1100, 2200), 100, 6.0
LR, WEIGHT_DECAY, BATCH, MICRO_BATCH, SEED = 1e-3, 0.01, 64, 32, 42
BOOT_N, BOOT_SEED = 2000, 20260902
GATE_MIN_EFFECT, GATE_BEATS_DEV_MAX = 0.02, 0.20
NONINFERIORITY_MARGIN, ORACLE_LIFT_MARGIN = -0.005, 0.010
PHASE_SHIFT_SAMPLES = 256
PARAM_BUDGET, PARAM_BUDGET_FRAC, GATE_PARAM_BUDGET = 50_000, 0.011, 128
EXPECTED_TF_PARAMS, EXPECTED_GATE_PARAMS = 12_768, 81
EXPECTED_GTF_PARAMS = EXPECTED_TF_PARAMS + EXPECTED_GATE_PARAMS
GENERATOR_PARAMS = 4_568_707

ARMS = ("B", "ADD", "TF-TRUE", "TF-SHUFFLE", "GTF-TRUE", "GTF-SHUFFLE", "GTF-CONST", "GTF-ORACLE", "ADD-ORACLE")
TRAINED_ARMS = ("tf_true", "tf_shuffle", "gtf_true", "gtf_shuffle", "gtf_const", "gtf_oracle")
ORACLE_ARMS = frozenset({"GTF-ORACLE", "ADD-ORACLE"})
ORACLE_LABEL = "(GT-R leakage; diagnostic only)"
ARM_FAMILY = {"tf_true": "tf", "tf_shuffle": "tf", "gtf_true": "gtf", "gtf_shuffle": "gtf", "gtf_const": "gtf", "gtf_oracle": "gtf"}
ARM_GATE_MODE = {"tf_true": None, "tf_shuffle": None, "gtf_true": "adaptive", "gtf_shuffle": "adaptive", "gtf_const": "const", "gtf_oracle": "adaptive"}
ARM_SCAFFOLD = {"tf_true": "own", "tf_shuffle": "partner", "gtf_true": "own", "gtf_shuffle": "partner", "gtf_const": "own", "gtf_oracle": "oracle"}
ARM_EVAL_NAME = {"tf_true": "TF-TRUE", "tf_shuffle": "TF-SHUFFLE", "gtf_true": "GTF-TRUE", "gtf_shuffle": "GTF-SHUFFLE", "gtf_const": "GTF-CONST", "gtf_oracle": "GTF-ORACLE"}

TF_PARAM_NAMES = ("fusion.tok.weight", "fusion.q.weight",
                  "fusion.attn.q.weight", "fusion.attn.q.bias", "fusion.attn.k.weight", "fusion.attn.k.bias",
                  "fusion.attn.v.weight", "fusion.attn.v.bias", "fusion.attn.o.weight", "fusion.attn.o.bias",
                  "fusion.out.weight", "fusion.out.bias")
GATE_PARAM_NAMES = ("gate.0.weight", "gate.0.bias", "gate.2.weight", "gate.2.bias")
GTF_PARAM_NAMES = TF_PARAM_NAMES + GATE_PARAM_NAMES
FAMILY_PARAM_NAMES = {"tf": TF_PARAM_NAMES, "gtf": GTF_PARAM_NAMES}

EXPECTED_GENERATOR_FILE_SHA_PREFIX, EXPECTED_RHYTHM_FILE_SHA_PREFIX = "557c7054", "bfe76ea6"
EXPECTED_R2_ADD_FILE_SHA_PREFIX, EXPECTED_R2_ORACLE_FILE_SHA_PREFIX = "2d577897", "2802292b"
R2_ADD_CKPT = "outputs/r2_true_adapter_seed42/adapter_step2200.pt"
R2_ORACLE_CKPT = "outputs/r2_oracle_adapter_seed42/adapter_step2200.pt"
EXPECTED_R2_ADD_STATE_SHA_PREFIX = "f98057ca981bb840"
EXPECTED_R2_ORACLE_STATE_SHA_PREFIX = "c8827b1b0a6d065f"
R2_PROBE_HASH = "04aad6ae5ec41798084bc534165335e540f03488f607f88df59d0a36da9f888c"


# ----------------------------------------------------------------------------------------------------------
# positional encoding (section 6.3): fixed sinusoid on a shared sample-index axis
# ----------------------------------------------------------------------------------------------------------
def sinusoidal_pe(positions: torch.Tensor, d: int = D_MODEL, base: float = PE_BASE) -> torch.Tensor:
    """[P] integer sample positions -> [P, d]; even columns sin, odd columns cos."""
    pos = positions.to(torch.float64).unsqueeze(1)
    i = torch.arange(0, d, 2, dtype=torch.float64)
    div = torch.exp(-math.log(base) * i / d)
    pe = torch.zeros(len(positions), d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(torch.float32)


QUERY_POSITIONS = torch.arange(T_LEN)                    # sample t
TOKEN_POSITIONS = torch.arange(0, T_LEN, TOK_S)          # token j centred on sample 4j


# ----------------------------------------------------------------------------------------------------------
# cross-attention (section 6.4): explicit q/k/v/o so that forward-mode JVP is supported
# ----------------------------------------------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(self, d: int = D_MODEL, heads: int = N_HEADS):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d, self.h = d, heads
        self.q, self.k, self.v, self.o = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, qx: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """qx [B,Lq,d], kv [B,Lk,d] -> [B,Lq,d]; softmax(QK^T/sqrt(dh))V per head, no mask."""
        B, Lq, _ = qx.shape
        Lk = kv.shape[1]
        dh = self.d // self.h
        Q = self.q(qx).view(B, Lq, self.h, dh).transpose(1, 2)
        K = self.k(kv).view(B, Lk, self.h, dh).transpose(1, 2)
        V = self.v(kv).view(B, Lk, self.h, dh).transpose(1, 2)
        att = torch.softmax(Q @ K.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        return self.o((att @ V).transpose(1, 2).reshape(B, Lq, self.d))


class RhythmCrossFusionAdapter(nn.Module):
    """scaffold s [B,1,T] + target hidden H_z [B,C,T] -> fusion output [B,C,T] (zero at init)."""

    def __init__(self, c_hidden: int):
        super().__init__()
        self.tok = nn.Conv1d(1, D_MODEL, TOK_K, stride=TOK_S, padding=TOK_P, bias=False)     # 6.1
        self.q = nn.Conv1d(c_hidden, D_MODEL, 1, bias=False)                                    # 6.2
        self.attn = CrossAttention(D_MODEL, N_HEADS)                                             # 6.4
        self.out = nn.Conv1d(D_MODEL, c_hidden, 1, bias=True)                                   # 6.5
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("pe_q", sinusoidal_pe(QUERY_POSITIONS), persistent=False)         # 6.3
        self.register_buffer("pe_k", sinusoidal_pe(TOKEN_POSITIONS), persistent=False)

    def forward(self, hz: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        tok = self.tok(s).transpose(1, 2)                                 # [B,L/4,d]
        q = self.q(hz).transpose(1, 2)                                    # [B,L,d]
        tok = tok + self.pe_k[:tok.shape[1]]                              # positions 4j (identical slicing at L = 1024)
        q = q + self.pe_q[:q.shape[1]]                                    # positions t
        return self.out(self.attn(q, tok).transpose(1, 2))                # [B,C,L]


# ----------------------------------------------------------------------------------------------------------
# adaptive rhythm gate (section 7)
# ----------------------------------------------------------------------------------------------------------
def gate_features(s: torch.Tensor, const: bool = False) -> torch.Tensor:
    """[B,1,T] scaffold -> [B,3,T]: s, 33-sample centred mean (truncated at the edges), |first difference|
    (f3[0] = 0). const=True replaces every feature by its per-window temporal mean broadcast over T.
    Takes the scaffold only: no ECG, no GT R, no site, no validation information."""
    if s.dim() != 3 or s.shape[1] != 1:
        raise ValueError(f"scaffold must be [B,1,T], got {tuple(s.shape)}")
    f2 = F.avg_pool1d(s, GATE_POOL, stride=1, padding=GATE_POOL // 2, count_include_pad=False)
    f3 = torch.cat([torch.zeros_like(s[..., :1]), (s[..., 1:] - s[..., :-1]).abs()], dim=-1)
    feats = torch.cat([s, f2, f3], dim=1)
    if const:
        feats = feats.mean(dim=-1, keepdim=True).expand_as(feats)
    return feats


def make_gate() -> nn.Sequential:
    """gate.0 = Conv1d(3->16, k=1), SiLU, gate.2 = Conv1d(16->1, k=1); final weight zero, bias logit(0.9)."""
    g = nn.Sequential(nn.Conv1d(3, GATE_HIDDEN, 1), nn.SiLU(), nn.Conv1d(GATE_HIDDEN, 1, 1))
    nn.init.zeros_(g[2].weight)
    nn.init.constant_(g[2].bias, GATE_INIT_BIAS)
    return g


class R3Fusion(nn.Module):
    """family 'tf' (ungated) or 'gtf' (gated; gate_mode 'adaptive' | 'const')."""

    def __init__(self, c_hidden: int, family: str, gate_mode: str | None):
        super().__init__()
        if family not in ("tf", "gtf") or (family == "gtf") != (gate_mode in ("adaptive", "const")):
            raise ValueError(f"bad family/gate_mode {family}/{gate_mode}")
        self.family, self.gate_mode = family, gate_mode
        self.fusion = RhythmCrossFusionAdapter(c_hidden)                # constructed FIRST (fixed order)
        self.gate = make_gate() if family == "gtf" else None            # strictly after the out-projection

    def gate_values(self, s: torch.Tensor) -> torch.Tensor | None:
        if self.gate is None:
            return None
        return torch.sigmoid(self.gate(gate_features(s, const=(self.gate_mode == "const"))))

    def forward(self, hz: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self.fusion(hz, s)
        if self.gate is None:
            return out
        return self.gate_values(s) * out


def build_r3_module(family: str, gate_mode: str | None, c_hidden: int = 128, seed: int = SEED) -> R3Fusion:
    """Identical initialisation across arms: seed the CPU RNG immediately before construction, fixed order."""
    torch.manual_seed(seed)
    return R3Fusion(c_hidden, family, gate_mode)


def n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters()))


def param_names(m: nn.Module, prefix: str = "") -> tuple:
    return tuple(prefix + n for n, _ in m.named_parameters())


def params_sha256(m: nn.Module, only_prefix: str | None = None) -> str:
    """sha256 over sorted named_parameters (buffers such as the positional encodings are excluded)."""
    h = hashlib.sha256()
    for n, p in sorted(m.named_parameters(), key=lambda kv: kv[0]):
        if only_prefix and not n.startswith(only_prefix):
            continue
        h.update(n.encode()); h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------
# generator subclass: fusion at z_e (hook audit); pre_conv_ppg untouched; optional direct-route cancellation
# ----------------------------------------------------------------------------------------------------------
class FusionMeanFlowS5(MeanFlowS5):
    def __init__(self, backbone: nn.Module, r3: R3Fusion, cond_mode: str = "h_only", h_scale: float = 1.0):
        super().__init__(backbone, cond_mode=cond_mode, h_scale=h_scale)
        self.r3 = r3
        self.cancel_direct_route = False                                # evaluation-only diagnostic (section 18.2)

    def u(self, z: torch.Tensor, ppg2: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        ppg, s = split_ppg2(ppg2)
        ppg_e = bb.pre_conv_ppg(ppg)                                    # PPG stem: untouched
        z_e = bb.pre_conv_target(z)                                     # hook: target hidden [B,C,T]
        delta = self.r3(z_e, s)
        z_e = z_e + delta
        if self.cond_mode == "h_only":
            cond = bb.timestep_embedder(h.reshape(-1) * self.h_scale)
        else:
            cond = bb.timestep_embedder(t.reshape(-1))
            if self.cond_mode == "t_plus_h":
                cond = cond + bb.timestep_embedder(h.reshape(-1) * self.h_scale)
        all_dx = torch.zeros_like(z_e)
        for blk in bb.flow_ssm_list:
            ppg_e, dx = blk(ppg_e, z_e, cond)
            all_dx = all_dx + dx
        if self.cancel_direct_route:
            all_dx = all_dx - len(bb.flow_ssm_list) * delta
        return bb.final_layer(all_dx, cond)


def trainable_names(net: nn.Module) -> tuple:
    return tuple(sorted(n for n, p in net.named_parameters() if p.requires_grad))


def assert_only_r3_trainable(net: FusionMeanFlowS5) -> None:
    exp = tuple(sorted("r3." + n for n in FAMILY_PARAM_NAMES[net.r3.family]))
    got = trainable_names(net)
    if got != exp:
        raise RuntimeError(f"trainable set must be {exp}, got {got}")


def assert_frozen_have_no_grad(net: FusionMeanFlowS5, tcn: nn.Module | None = None) -> None:
    bad = [n for n, p in net.named_parameters() if not n.startswith("r3.") and p.grad is not None]
    if tcn is not None:
        bad += [f"tcn.{n}" for n, p in tcn.named_parameters() if p.grad is not None]
    if bad:
        raise RuntimeError(f"frozen parameters received gradients: {bad[:5]} ... ({len(bad)})")


def roll_scaffold(s: torch.Tensor, shift: int = PHASE_SHIFT_SAMPLES) -> torch.Tensor:
    return torch.roll(s, int(shift), dims=-1)


# ----------------------------------------------------------------------------------------------------------
# verdict (section 17) — total function of the decision record
# ----------------------------------------------------------------------------------------------------------
VERDICTS = ("UNGATED TARGET-SIDE FUSION SUFFICIENT",
            "ADAPTIVE GATING REQUIRED AND SUPPORTED",
            "EVENT GAIN WITH STRUCTURE TRADE-OFF PERSISTS",
            "TARGET-SIDE RHYTHM FUSION NOT SUPPORTED")
RESIDUAL_CODES = ("D_NO_EVENT_GAIN", "D_SUBTHRESHOLD", "D_TF_U3_FAIL", "D_TF_U6_CATASTROPHE",
                  "D_GTF_G5_NONINFERIORITY_FAIL", "D_GTF_G5_STRUCTURE_VS_CONST_FAIL", "D_GTF_G6_CATASTROPHE")


def decide_verdict_r3(rec: dict) -> dict:
    """rec keys: U1..U6, G1..G6 (bool); ev_tf, ev_gtf, ev_ci_tf, ev_ci_gtf, deg_tf, deg_gtf (bool);
    g5_noninferior, g5_structure (bool); gtf_vs_tf_protects (bool); u4 (bool, = U4).
    Returns {verdict, codes, necessity, prefers_tf_over_gtf}."""
    U = [bool(rec[f"U{i}"]) for i in range(1, 7)]
    G = [bool(rec[f"G{i}"]) for i in range(1, 7)]
    ev_tf, ev_gtf = bool(rec["ev_tf"]), bool(rec["ev_gtf"])
    deg_tf, deg_gtf = bool(rec["deg_tf"]), bool(rec["deg_gtf"])
    protects = bool(rec.get("gtf_vs_tf_protects", False))                     # A tie-break: S4/S5 protection AND no event worsening
    structure = bool(rec.get("gtf_vs_tf_structure", protects))                # B necessity: S4/S5 only (section 17.2 row 2)
    out = {"codes": [], "necessity": None, "prefers_tf_over_gtf": None, "g_all_pass": all(G), "u_all_pass": all(U)}
    if all(U):
        out["verdict"] = VERDICTS[0]
        out["prefers_tf_over_gtf"] = not protects
        return out
    if all(G):
        out["verdict"] = VERDICTS[1]
        out["necessity"] = "SEPARATED" if (not U[3]) or structure else "NOT_SEPARATED"
        out["tf_failed_items"] = [f"U{i+1}" for i in range(6) if not U[i]]
        return out
    ev_arms = [x for x, e in (("tf", ev_tf), ("gtf", ev_gtf)) if e]
    if ev_arms and all({"tf": deg_tf, "gtf": deg_gtf}[x] for x in ev_arms):
        out["verdict"] = VERDICTS[2]
        out["coupling"] = {x: {"tf": deg_tf, "gtf": deg_gtf}[x] for x in ("tf", "gtf")}
        out["ev_arms"] = ev_arms
        return out
    codes = []
    if not ev_arms:
        codes.append("D_NO_EVENT_GAIN")
    if (bool(rec.get("ev_ci_tf")) and not ev_tf) or (bool(rec.get("ev_ci_gtf")) and not ev_gtf):
        codes.append("D_SUBTHRESHOLD")
    if ev_tf and not deg_tf:
        if not U[2]:
            codes.append("D_TF_U3_FAIL")
        if not U[5]:
            codes.append("D_TF_U6_CATASTROPHE")
    if ev_gtf and not deg_gtf:
        if not bool(rec.get("g5_noninferior", True)):
            codes.append("D_GTF_G5_NONINFERIORITY_FAIL")
        if not bool(rec.get("g5_structure", True)):
            codes.append("D_GTF_G5_STRUCTURE_VS_CONST_FAIL")
        if not G[5]:
            codes.append("D_GTF_G6_CATASTROPHE")
    out["verdict"] = VERDICTS[3]
    out["codes"] = codes or ["D_RESIDUAL_UNCLASSIFIED"]
    return out


ORACLE_READINGS = ("LOWERED", "LIFTED", "LIFTED_BUT_COUPLES", "UNCHANGED", "OTHER")


def oracle_reading(v_gtf_vs_add: str, point: float, v_s4_vs_b: str, v_s5_vs_b: str) -> str:
    """Ordered, total reading of GTF-ORACLE vs ADD-ORACLE (F1 excess) with S4/S5 of GTF-ORACLE vs B."""
    if v_gtf_vs_add == "worsens":
        return "LOWERED"
    couples = v_s4_vs_b == "worsens" or v_s5_vs_b == "worsens"
    if v_gtf_vs_add == "improves" and point >= ORACLE_LIFT_MARGIN:
        return "LIFTED_BUT_COUPLES" if couples else "LIFTED"
    if v_gtf_vs_add == "unresolved" or (v_gtf_vs_add == "improves" and point < ORACLE_LIFT_MARGIN):
        return "UNCHANGED"
    return "OTHER"
