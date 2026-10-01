"""R1 models (prereg §2).

SOFT-RHYTHM-WW   exactly the WW-DET class (C0-A `ablation.WWDet`, decoder width 72, depth 5); its conditioning channel
                 carries the frozen C0 timing detector's pre-threshold field sigmoid(logits) [B, 512] instead of the
                 Gaussian event raster. Same parameter count as WW-DET.
JOINT-RHYTHMFIELD-WW  one shared C0 encoder; a lightweight rhythm head (1x1 64->64, GELU, 1x1 64->1) gives one logit per
                 ECG sample; p_R = sigmoid(z_R) (not detached, never thresholded) is concatenated with the shared features
                 and decoded by WW-DET's decoder into the 512-sample ECG window.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.coherentbeat import ablation as AB
from ppg2ecg.coherentbeat import model as CM

WW_DEC_CH, WW_N_DEC = 72, 5                  # WW-DET's frozen configuration (C0-A parameter_match.json)
DET_THRESHOLD, DET_REFRACTORY = 0.35, 32     # the frozen C0 detector convention (used only for diagnostics)
RHYTHM_SIGMA_MS = 20.0                        # the frozen C0 detector target convention (Gaussian, BCE with logits)
LAMBDA_RHYTHM = 1.0


def soft_rhythm_ww() -> AB.WWDet:
    return AB.WWDet(WW_DEC_CH, WW_N_DEC)


class RhythmHead(nn.Module):
    def __init__(self, ch: int = CM.CH):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(ch, ch, 1), nn.GELU(), nn.Conv1d(ch, 1, 1))

    def forward(self, h):                     # [B, C, T] -> [B, T] logits
        return self.net(h)[:, 0]


class JointRhythmFieldWW(nn.Module):
    """h = E(PPG); z_R = R(h); p_R = sigmoid(z_R); ECG_hat = D(h, p_R). Returns (ECG_hat [B, T], z_R [B, T])."""

    def __init__(self, dec_ch: int = WW_DEC_CH, n_dec: int = WW_N_DEC, ch: int = CM.CH):
        super().__init__()
        self.encoder = CM.Encoder(ch)
        self.rhythm_head = RhythmHead(ch)
        self.dec_in = nn.Conv1d(ch + 1, dec_ch, 1)
        self.dec = nn.ModuleList([CM.ResBlock(dec_ch, CM.KERNEL, int(AB.WW_DEC_DILATIONS[i])) for i in range(n_dec)])
        self.head = nn.Conv1d(dec_ch, 1, 1)

    def forward(self, x, return_field: bool = False):
        h = self.encoder(x[:, None])
        z = self.rhythm_head(h)
        p = torch.sigmoid(z)                                     # dense, not detached, never thresholded
        q = self.dec_in(torch.cat([h, p[:, None]], dim=1))
        for b in self.dec:
            q = b(q)
        y = self.head(q)[:, 0]
        return (y, z, p) if return_field else (y, z)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_breakdown(m: JointRhythmFieldWW) -> dict:
    head = n_params(m.rhythm_head)
    return {"total": n_params(m), "rhythm_head": head, "waveform": n_params(m) - head}


# ----------------------------------------------------------------------------------------------- patient-macro metrics
def patient_macro_rows(n_tp, n_fp, n_fn, pid) -> dict:
    """Per-patient pooled event metrics (prereg §6): precision_p = TP / (TP + FP) (nan without detections),
    recall_p = TP / (TP + FN), F1_p = 2 TP / (2 TP + FP + FN), FP rate_p = FP / windows. Returns sorted patients + arrays."""
    pid = np.asarray(pid)
    subs = np.unique(pid)
    k = np.searchsorted(subs, pid)
    tp = np.bincount(k, weights=np.asarray(n_tp, float), minlength=subs.size)
    fp = np.bincount(k, weights=np.asarray(n_fp, float), minlength=subs.size)
    fn = np.bincount(k, weights=np.asarray(n_fn, float), minlength=subs.size)
    nw = np.bincount(k, minlength=subs.size).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return {"patients": subs, "precision": prec, "recall": rec, "f1": f1, "fp_rate": fp / nw,
            "pooled": {"tp": int(tp.sum()), "fp": int(fp.sum()), "fn": int(fn.sum())}}


def pooled_prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": p, "recall": r, "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")}


# ----------------------------------------------------------------------------------------------- frozen rules
M_FP_C0, M_RECALL, M_FD_WW, M_CORR = 0.03, 0.005, 1.0, 0.02


def val_gates(d: dict) -> dict:
    """ARCH-VAL gates V1-V5 on paired [point, lo, hi] differences (candidate - reference)."""
    g = {"V1": d["fp_vs_ww"][2] < 0, "V2": d["fp_vs_c0"][2] < M_FP_C0, "V3": d["recall_vs_ww"][1] > -M_RECALL,
         "V4": d["fd_vs_ww"][2] < M_FD_WW and d["fd_vs_c0"][2] < 0, "V5": d["corr_vs_ww"][1] > -M_CORR}
    g = {k: bool(v) for k, v in g.items()}
    g["QUALIFIED"] = all(g[k] for k in ("V1", "V2", "V3", "V4", "V5"))
    return g


def select_candidate(qual: dict, fd: dict, fp: dict) -> str:
    """Prereg §8: NONE / the single qualifier / both -> lower FD; |dFD| < 0.25 -> lower FP; |dFP| < 0.01 -> SOFT."""
    q = [k for k in ("SOFT", "JOINT") if qual[k]]
    if not q:
        return "NONE"
    if len(q) == 1:
        return q[0]
    if abs(fd["SOFT"] - fd["JOINT"]) >= 0.25:
        return "SOFT" if fd["SOFT"] < fd["JOINT"] else "JOINT"
    if abs(fp["SOFT"] - fp["JOINT"]) >= 0.01:
        return "SOFT" if fp["SOFT"] < fp["JOINT"] else "JOINT"
    return "SOFT"


def test_verdict(d: dict) -> dict:
    """Final TEST T1-T5 (same margins); PARTIAL = T1 and T4 pass and exactly one of T2 / T3 / T5 fails within twice its
    margin; FAILED otherwise (incl. recall CI lower bound <= -2 x 0.005, i.e. major recall degradation)."""
    t = {"T1": d["fp_vs_ww"][2] < 0, "T2": d["fp_vs_c0"][2] < M_FP_C0, "T3": d["recall_vs_ww"][1] > -M_RECALL,
         "T4": d["fd_vs_ww"][2] < M_FD_WW and d["fd_vs_c0"][2] < 0, "T5": d["corr_vs_ww"][1] > -M_CORR}
    t = {k: bool(v) for k, v in t.items()}
    narrow = {"T2": d["fp_vs_c0"][2] < 2 * M_FP_C0, "T3": d["recall_vs_ww"][1] > -2 * M_RECALL, "T5": d["corr_vs_ww"][1] > -2 * M_CORR}
    failed_secondary = [k for k in ("T2", "T3", "T5") if not t[k]]
    major_recall = d["recall_vs_ww"][1] <= -2 * M_RECALL
    if all(t.values()):
        v = "STRONG"
    elif t["T1"] and t["T4"] and len(failed_secondary) == 1 and narrow[failed_secondary[0]] and not major_recall:
        v = "PARTIAL"
    else:
        v = "FAILED"
    return t | {"verdict": v}
