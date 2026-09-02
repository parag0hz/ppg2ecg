"""R2 — frozen global rhythm scaffold -> frozen generator, minimal transfer adapter.

Frozen by docs/R2_RHYTHM_SCAFFOLD_TRANSFER_PREREGISTRATION.md (f954e07). Everything here is protocol
machinery: the adapter and the subclass that injects it (section 5), the scaffold (section 4), the SHUFFLE
derangement (section 7), the ORACLE field (section 6), the phase-ablation phi (section 18), the persistence
matcher (section 19) and the total verdict function (section 22). No training loop, no data access.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events, soft_event_field

FS, T_LEN = 128, 1024
SHUFFLE_SALT = "r2-rhythm-shuffle-v1"
STEPS, CKPT_STEPS, PREFLIGHT_STEPS = 2200, (0, 550, 1100, 2200), 100
BUDGET_GPU_HOURS = 6.0
LR, WEIGHT_DECAY, BATCH, MICRO_BATCH = 1e-3, 0.01, 64, 32
SEED = 42
BOOT_N, BOOT_SEED = 2000, 20260902
GATE_MIN_EFFECT, GATE_BEATS_DEV_MAX = 0.02, 0.20
PHASE_SHIFT_SAMPLES = 256                        # +2.0 s at 128 Hz
PERSIST_TOL_MS = 250.0                           # = 32 samples, greedy one-to-one (rpeaks.match_rpeaks)
NFE_PRIMARY, NFE_SECONDARY = 4, (1, 2)
ARMS = ("B", "TRUE", "SHUFFLE", "ORACLE")
TRAINED_ARMS = ("true", "shuffle", "oracle")
GENERATOR_CKPT = "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt"
RHYTHM_CKPT = "outputs/r1_global_tcn_seed42/checkpoint_best.pt"
EXPECTED_GENERATOR_STATE_SHA = "47d7ccb94e5dbf7190d777f852b18f107f3ce2628d160b5e01ff96ef2a1d0d0f"
EXPECTED_RHYTHM_STATE_SHA = "0986a7af1db291336046f3c7e9659aafc7ee77a381745a4e33344a7ac96a3287"
TRAINABLE_NAMES = ("rhythm_adapter.proj.weight",)


# ----------------------------------------------------------------------------------------------------------
# adapter + injected generator (section 5)
# ----------------------------------------------------------------------------------------------------------
class RhythmAdapter(nn.Module):
    """Conv1d(1 -> h_dim, k=1, bias=False), zero-initialised. No activation, no norm, no temporal mixing."""

    def __init__(self, h_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(1, int(h_dim), kernel_size=1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)


def make_ppg2(ppg: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[B,1,T] PPG + [B,1,T] scaffold -> [B,2,T]; the frozen loss/sampler pass it through untouched."""
    if ppg.shape != s.shape:
        raise ValueError(f"ppg {tuple(ppg.shape)} and scaffold {tuple(s.shape)} must match")
    return torch.cat([ppg, s], dim=1)


def split_ppg2(ppg2: torch.Tensor):
    if ppg2.shape[1] != 2:
        raise ValueError(f"expected [B,2,T], got {tuple(ppg2.shape)}")
    return ppg2[:, :1].contiguous(), ppg2[:, 1:2].contiguous()   # contiguous: the stem conv must see the same layout as the frozen path


class RhythmMeanFlowS5(MeanFlowS5):
    """MeanFlowS5 with ppg_e' = pre_conv_ppg(ppg) + RhythmAdapter(s). backbone.* keys are untouched."""

    def __init__(self, backbone: nn.Module, h_dim: int, cond_mode: str = "h_only", h_scale: float = 1.0):
        super().__init__(backbone, cond_mode=cond_mode, h_scale=h_scale)
        self.rhythm_adapter = RhythmAdapter(h_dim)

    def u(self, z: torch.Tensor, ppg2: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        bb = self.backbone
        ppg, s = split_ppg2(ppg2)
        ppg_e = bb.pre_conv_ppg(ppg) + self.rhythm_adapter(s)
        z_e = bb.pre_conv_target(z)
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
        return bb.final_layer(all_dx, cond)


def trainable_names(net: nn.Module) -> list[str]:
    return sorted(n for n, p in net.named_parameters() if p.requires_grad)


def n_trainable(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def assert_only_adapter_trainable(net: nn.Module) -> None:
    got = tuple(trainable_names(net))
    if got != tuple(sorted(TRAINABLE_NAMES)):
        raise RuntimeError(f"trainable set must be {TRAINABLE_NAMES}, got {got}")


def assert_frozen_have_no_grad(net: nn.Module, tcn: nn.Module | None = None) -> None:
    bad = [n for n, p in net.named_parameters() if n not in TRAINABLE_NAMES and p.grad is not None]
    if tcn is not None:
        bad += [f"tcn.{n}" for n, p in tcn.named_parameters() if p.grad is not None]
    if bad:
        raise RuntimeError(f"frozen parameters received gradients: {bad[:5]} ... ({len(bad)})")


# ----------------------------------------------------------------------------------------------------------
# checkpoints (section 2)
# ----------------------------------------------------------------------------------------------------------
def state_dict_sha256(sd: dict) -> str:
    """V1 method: sorted keys, contiguous numpy bytes concatenated."""
    return hashlib.sha256(b"".join(np.ascontiguousarray(sd[k].detach().cpu().numpy()).tobytes()
                                   for k in sorted(sd))).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def build_rhythm_generator(ck: dict) -> RhythmMeanFlowS5:
    """RhythmMeanFlowS5 from a frozen generator checkpoint dict; backbone loaded strict on backbone.* keys."""
    from ppg2ecg.models import build_penguin_backbone  # upstream import (needs mkl warm-up first)
    cfg = ck.get("imf_cfg", {})
    net = RhythmMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), int(ck["model_cfg"]["h_dim"]),
                           cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0))
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or set(missing) != set(TRAINABLE_NAMES):
        raise RuntimeError(f"unexpected checkpoint layout: missing={missing} unexpected={unexpected}")
    net.backbone.requires_grad_(False)
    net.rhythm_adapter.requires_grad_(True)
    return net


def load_generator(path, dev) -> tuple[RhythmMeanFlowS5, dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sdh = state_dict_sha256(ck["state_dict"])
    if sdh != EXPECTED_GENERATOR_STATE_SHA:
        raise RuntimeError(f"generator state_dict sha256 {sdh} != frozen {EXPECTED_GENERATOR_STATE_SHA}")
    net = build_rhythm_generator(ck).to(dev).eval()
    meta = {"path": str(path), "file_sha256": file_sha256(path), "state_dict_sha256": sdh,
            "round": int(ck["epoch"]), "c1_arm": ck.get("args", {}).get("c1_arm"),
            "selection_metric": float(ck["selection"]["value"]), "model_cfg": ck["model_cfg"],
            "imf_cfg": ck.get("imf_cfg", {}), "n_params_total": int(sum(p.numel() for p in net.backbone.parameters())),
            "h_dim": int(ck["model_cfg"]["h_dim"])}
    return net, ck, meta


def load_rhythm_tcn(path, dev) -> tuple[RhythmTCN, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sdh = state_dict_sha256(ck["state_dict"])
    if sdh != EXPECTED_RHYTHM_STATE_SHA:
        raise RuntimeError(f"Global-TCN state_dict sha256 {sdh} != frozen {EXPECTED_RHYTHM_STATE_SHA}")
    tcn = RhythmTCN(ck["dilations"], n_sites=ck["n_sites"]).to(dev).eval()
    tcn.load_state_dict(ck["state_dict"])
    tcn.requires_grad_(False)
    meta = {"path": str(path), "file_sha256": file_sha256(path), "state_dict_sha256": sdh,
            "dilations": list(ck["dilations"]), "n_sites": int(ck["n_sites"]), "params": int(ck["params"]),
            "rf": int(ck["rf"]), "best_epoch": int(ck["epoch"]), "internal_dev_bce": float(ck["internal_dev_bce"]),
            "r1_threshold_recorded_not_used": 0.35}
    return tcn, meta


# ----------------------------------------------------------------------------------------------------------
# scaffolds (sections 4, 6)
# ----------------------------------------------------------------------------------------------------------
@torch.no_grad()
def scaffold_from_ppg(tcn: RhythmTCN, ppg: torch.Tensor) -> torch.Tensor:
    """PPG [B,1,T] -> dense pre-NMS sigmoid field [B,1,T], detached. PPG only: no ECG argument exists."""
    if ppg.dim() != 3 or ppg.shape[1] != 1:
        raise ValueError(f"PPG must be [B,1,T], got {tuple(ppg.shape)}")
    return torch.sigmoid(tcn(ppg)).detach()


def _oracle_one(y: np.ndarray) -> np.ndarray:
    return soft_event_field(R.detect_rpeaks(np.asarray(y, dtype=np.float64), FS), T_LEN)


def oracle_fields(ecg: np.ndarray, workers: int = 12) -> np.ndarray:
    """GT-R soft fields (sigma 12.8 samples), the exact R1 label, for an [N,T] ECG array. LEAKAGE BY DESIGN."""
    ecg = np.asarray(ecg)
    if ecg.ndim != 2 or ecg.shape[1] != T_LEN:
        raise ValueError(f"ECG must be [N,{T_LEN}], got {ecg.shape}")
    if workers <= 1 or len(ecg) < 64:
        return np.stack([_oracle_one(y) for y in ecg]).astype(np.float32)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_oracle_one, list(ecg), chunksize=256))
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------
# SHUFFLE derangement (section 7)
# ----------------------------------------------------------------------------------------------------------
def shuffle_key(subject: str, site: str, window_index: int, salt: str = SHUFFLE_SALT) -> str:
    return hashlib.sha256(f"{salt}|{subject}|{site}|{int(window_index)}".encode()).hexdigest()


def shuffle_partner(subjects, sites, window_index, salt: str = SHUFFLE_SALT) -> np.ndarray:
    """Partner position for every element: within each (subject, site) stratum rank by SHA256 and map
    rank i -> rank (i+1) mod n. Bijective, no fixed points (n >= 2). A singleton stratum raises."""
    subjects, sites, window_index = np.asarray(subjects), np.asarray(sites), np.asarray(window_index)
    n = len(subjects)
    if not (len(sites) == n == len(window_index)):
        raise ValueError("subjects, sites, window_index must have the same length")
    partner = np.full(n, -1, dtype=np.int64)
    for sub in np.unique(subjects):
        for site in np.unique(sites[subjects == sub]):
            m = np.flatnonzero((subjects == sub) & (sites == site))
            if m.size < 2:
                raise RuntimeError(f"SHUFFLE undefined: stratum ({sub}, {site}) has {m.size} element(s)")
            keys = [shuffle_key(str(sub), str(site), int(window_index[i]), salt) for i in m]
            if len(set(keys)) != len(keys):
                raise RuntimeError(f"duplicate window_index inside stratum ({sub}, {site})")
            order = m[np.argsort(keys, kind="stable")]
            partner[order] = np.roll(order, -1)           # rank i -> rank i+1 (mod n)
    if np.any(partner < 0):
        raise RuntimeError("shuffle partner incomplete")
    return partner


def assert_derangement(partner: np.ndarray) -> None:
    p = np.asarray(partner)
    if sorted(p.tolist()) != list(range(len(p))):
        raise RuntimeError("shuffle partner is not a bijection")
    if np.any(p == np.arange(len(p))):
        raise RuntimeError("shuffle partner has a fixed point")


# ----------------------------------------------------------------------------------------------------------
# paired-randomness probe (section 9)
# ----------------------------------------------------------------------------------------------------------
def probe_update(h, idx: torch.Tensor, t: torch.Tensor, r: torch.Tensor, e: torch.Tensor) -> None:
    for x in (idx, t, r, e):
        h.update(np.ascontiguousarray(x.detach().cpu().numpy()).tobytes())


# ----------------------------------------------------------------------------------------------------------
# phase ablation (section 18)
# ----------------------------------------------------------------------------------------------------------
def roll_scaffold(s: torch.Tensor, shift: int = PHASE_SHIFT_SAMPLES) -> torch.Tensor:
    return torch.roll(s, int(shift), dims=-1)


def phase_phi(gt_peaks, shift: int = PHASE_SHIFT_SAMPLES) -> float:
    """frac(shift / mean GT RR in samples); nan with fewer than two GT beats."""
    p = np.asarray(gt_peaks, dtype=np.float64)
    if p.size < 2:
        return float("nan")
    rr = float(np.mean(np.diff(p)))
    return float((shift / rr) % 1.0)


def phi_stratum(phi: float) -> str:
    if not np.isfinite(phi):
        return "undefined"
    if phi < 0.1 or phi >= 0.9:
        return "in_phase"
    if 0.4 <= phi <= 0.6:
        return "anti_phase"
    return "rest"


def scaffold_event_f1(field: np.ndarray, gt_peaks, threshold: float = 0.35, tol_ms: float = 50.0) -> float:
    """The scaffold's own event F1@50 under the frozen R1 rule (exploratory stratification, section 17)."""
    ev = extract_events(field, threshold)
    m, fp, fn = R.match_rpeaks(np.asarray(gt_peaks), ev, FS, tol_ms)
    return float(R.prf(len(m), fp, fn)[2])


# ----------------------------------------------------------------------------------------------------------
# persistence matcher (section 19)
# ----------------------------------------------------------------------------------------------------------
def persistence_deltas(gt_peaks, pred_peaks_by_nfe: dict, tol_ms: float = PERSIST_TOL_MS) -> np.ndarray:
    """[n_gt, n_nfe] signed delta in ms (pred - GT), NaN if unmatched; greedy one-to-one at +-250 ms."""
    gt = np.asarray(gt_peaks)
    nfes = sorted(pred_peaks_by_nfe)
    out = np.full((len(gt), len(nfes)), np.nan)
    for j, k in enumerate(nfes):
        m, _, _ = R.match_rpeaks(gt, np.asarray(pred_peaks_by_nfe[k]), FS, tol_ms)
        for i_ref, j_pred in m:
            out[i_ref, j] = (float(pred_peaks_by_nfe[k][j_pred]) - float(gt[i_ref])) / FS * 1000.0
    return out


# ----------------------------------------------------------------------------------------------------------
# verdict (section 22) — a total function of the decision record
# ----------------------------------------------------------------------------------------------------------
VERDICTS = ("RHYTHM SCAFFOLD TRANSFER SUPPORTED",
            "EVENT GAIN WITH STRUCTURE TRADE-OFF",
            "SCAFFOLD INFORMATIVE, MINIMAL INTERFACE INSUFFICIENT",
            "RHYTHM CONDITIONING NOT SUPPORTED BY THIS INTERFACE")


def decide_verdict(rec: dict) -> tuple[str, str]:
    """rec: item1..item5 (bool), v_OB, v_TB, v_OT, v_SB in {improves, unresolved, worsens}, item1_point,
    item1_ci_positive (bool). Returns (verdict, residual_reason)."""
    items = [bool(rec[f"item{i}"]) for i in range(1, 6)]
    i1, i2, i3, i4, i5 = items
    if all(items):
        return VERDICTS[0], ""
    if i1 and i2 and not i4:
        quals = []
        if not i3:
            quals.append("reliability item 3 failed")
        if not i5:
            quals.append("beat-count catastrophe item 5 failed")
        return VERDICTS[1], "; ".join(quals)
    if rec["v_OB"] == "improves" and rec["v_OT"] == "improves":
        return VERDICTS[2], ""
    reasons = []
    if i1 and i2:
        # TRUE did beat B and SHUFFLE on F1 excess; D's narrative must not claim otherwise
        if not i3:
            reasons.append("reliability item 3 failed")
        if not i5:
            reasons.append("beat-count catastrophe item 5 failed")
    else:
        if not i1 and rec.get("item1_ci_positive") and rec.get("item1_point", 0.0) < GATE_MIN_EFFECT:
            reasons.append("sub-threshold event gain")
        if not i1 and not rec.get("item1_ci_positive"):
            reasons.append("TRUE does not beat B")
        if not i2:
            reasons.append("TRUE does not beat SHUFFLE")
    if rec["v_OB"] == "improves" and rec["v_OT"] != "improves":
        reasons.append("ORACLE ~ TRUE" if rec["v_OT"] == "unresolved" else "TRUE beats ORACLE")
    if rec["v_OB"] == "worsens":
        reasons.append("ORACLE worsens")
    if rec["v_OB"] == "unresolved":
        reasons.append("ORACLE ~ B")
    return VERDICTS[3], "; ".join(reasons) if reasons else "residual"


def oracle_case(v_OB: str, v_TB: str, v_OT: str) -> str:
    if v_OB in ("unresolved", "worsens"):
        return "case4"
    if v_OB == "improves" and v_TB == "improves" and v_OT == "improves":
        return "case1"
    if v_OB == "improves" and v_TB == "improves" and v_OT == "unresolved":
        return "case2"
    if v_OB == "improves" and v_TB in ("unresolved", "worsens") and v_OT == "improves":
        return "case3"
    return "other"
