"""O3 shared helpers — frozen loaders, supplied-schedule inference and the frozen metric plumbing.

NO TRAINING anywhere: no optimizer is constructed, no gradient is taken and every network is loaded frozen.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2W
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.probes.rhythm_tcn import extract_events

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o3_schedule_tolerance"
O1ART = ROOT / "artifacts/o1_component_extractability"
O2CART = ROOT / "artifacts/o2c_oracle_integer_grid"
O2C_CKPT = ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt"
VAL = ("an0", "k2s")
FS, T_LEN, BATCH = 128, 1024, 64
NFE, SRC_SEED = O3.NFE, O3.SRC_SEED
SRC_BANK_SHA = "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"
REG_TOL = 1e-6
FROZEN_B = {"f1": 0.4367299150733669, "chance_f1": 0.11916804674636086, "f1_excess": 0.31756186832700606,
            "missing": 0.5661580180073833, "spurious": 0.5153896464321268, "beats_ratio_dev": 0.10666361023416689}
FROZEN_O2C = {"f1_excess": 0.8592510052638713, "nAE_T4": 0.40715406781869296, "nAE_T6": 0.40190969840528823,
              "nAE_T7": 0.41699228519117515, "nAE_T8": 0.41387939453125}
HASHES = {"B_file": "557c7054", "B_state": "47d7ccb9", "O2C_file": "5aab09be", "O2C_state": "f1cc44b3",
          "R1_file": "bfe76ea6", "R1_state": "0986a7af", "o2b_warp": "cb4d1866", "o2_warp": "046becfb"}
EVENT_M = ("f1", "chance_f1", "f1_excess", "precision", "recall", "missing", "spurious", "beats_ratio_dev")
STRUCT_M = ("raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err", "qrs_e_dev",
            "p2p_dev", "hf_err")
HIGHER = {"f1", "f1_excess", "precision", "recall", "raw_corr"}
BOOT_M = ("f1_excess", "nAE_T4", "nAE_T6", "nAE_T7", "nAE_T8", "qrs_deriv_rmse", "qrs_curvature_err")
ALIGNED = O3.ALIGNED
NAE = {t: f"nAE_{OT.TARGET_IDS[t]}" for t in ALIGNED}


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m


R2E = _load("r2_evaluate", "scripts/r2_evaluate.py")
S0 = _load("o2_stage0_roundtrip", "scripts/o2_stage0_roundtrip.py")
O1E = _load("o1_evaluate", "scripts/o1_evaluate.py")


def wcsv(p, rows):
    R2E.wcsv(p, rows)


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def macro(vals, subs) -> float:
    v, s = np.asarray(vals, float), np.asarray(subs)
    return float(np.mean([np.nanmean(v[s == u]) for u in sorted(set(s.tolist()))]))


def orient(m: str) -> str:
    return "higher_better" if m in HIGHER else "lower_better"


def load_cohort():
    """The exact frozen O2c primary cohort; the three population facts are asserted."""
    ER.assert_no_test_subjects(VAL)
    X, Y, SUB, SITE, POS, WI = S0.load_cohort()
    Yd = Y.astype(np.float64)
    gt_pk = R2E.pmap(S0._peaks, list(Yd))
    gt_tg = R2E.pmap(S0._targets, list(Yd))
    n_beats = int(sum(len(p) for p in gt_pk))
    cluster = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])
    if len(X) != 2048 or n_beats != 19834 or len(set(cluster.tolist())) != 1922:
        raise RuntimeError(f"frozen cohort facts differ: {len(X)} windows, {n_beats} beats, "
                           f"{len(set(cluster.tolist()))} clusters (STOP)")
    iqr = {t: __import__("json").loads((O1ART / "target_scaling.json").read_text())["targets"][t]["scale_train_IQR"]
           for t in ALIGNED}
    return dict(X=X, Y=Y, Yd=Yd, SUB=SUB, SITE=SITE, POS=POS, WI=WI, gt_pk=gt_pk, gt_tg=gt_tg,
                n_beats=n_beats, cluster=cluster, iqr=iqr)


def source_bank(n: int, seed: int = SRC_SEED) -> torch.Tensor:
    e = torch.randn(n, 1, T_LEN, generator=torch.Generator().manual_seed(int(seed)))
    if seed == SRC_SEED and n == 2048:
        assert hashlib.sha256(e.numpy().tobytes()).hexdigest() == SRC_BANK_SHA
    return e


def load_models(dev, with_r1: bool = False):
    """B, O2c (and optionally the R1 Global-TCN), all frozen. No optimizer is created."""
    _n, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    o2c_ck = torch.load(O2C_CKPT, map_location="cpu", weights_only=False)
    o2c = MeanFlowS5(build_penguin_backbone(**o2c_ck["model_cfg"]), cond_mode=o2c_ck["imf_cfg"]["cond_mode"],
                     h_scale=o2c_ck["imf_cfg"]["h_scale"]).to(dev).eval()
    o2c.load_state_dict(o2c_ck["state_dict"]); o2c.requires_grad_(False)
    assert not any(p.requires_grad for p in base.parameters())
    assert not any(p.requires_grad for p in o2c.parameters())
    assert sum(p.numel() for p in o2c.backbone.parameters()) == 4_568_707
    assert int(o2c_ck["step"]) == 10046 and o2c_ck["state_dict_sha256"].startswith(HASHES["O2C_state"])
    assert fsha(O2C_CKPT).startswith(HASHES["O2C_file"])
    assert gmeta["state_dict_sha256"].startswith(HASHES["B_state"])
    for key, rel in (("o2b_warp", "src/ppg2ecg/evaluation/o2b_warp.py"), ("o2_warp", "src/ppg2ecg/evaluation/o2_warp.py")):
        assert fsha(ROOT / rel).startswith(HASHES[key]), rel
    meta = {"B": {k: gmeta[k] for k in gmeta if k != "model_cfg"},
            "O2C": {"path": str(O2C_CKPT.relative_to(ROOT)), "step": int(o2c_ck["step"]),
                    "file_sha256": fsha(O2C_CKPT), "state_dict_sha256": o2c_ck["state_dict_sha256"],
                    "n_params_total": 4_568_707},
            "operator": {"o2b_warp_sha256": fsha(ROOT / "src/ppg2ecg/evaluation/o2b_warp.py"),
                         "o2_warp_sha256": fsha(ROOT / "src/ppg2ecg/evaluation/o2_warp.py"),
                         "ANCHOR_W": BW.ANCHOR_W, "MIN_INT_SPACING": BW.MIN_INT_SPACING,
                         "CORE_OFFSET_TOL": BW.CORE_OFFSET_TOL, "modification": "none"}}
    if not with_r1:
        return base, o2c, meta
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    assert not any(p.requires_grad for p in tcn.parameters()) and not tcn.training
    assert tmeta["state_dict_sha256"].startswith(HASHES["R1_state"])
    assert fsha(ROOT / RT.RHYTHM_CKPT).startswith(HASHES["R1_file"])
    meta["R1"] = tmeta
    return base, o2c, tcn, meta


@torch.no_grad()
def r1_schedules(tcn, X, dev, threshold: float = O3.R1_THRESHOLD):
    """Frozen R1 Global-TCN -> per-sample probability -> frozen extract_events. No correction of any kind."""
    out = []
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(np.ascontiguousarray(X[i:i + BATCH])).to(dev).unsqueeze(1)
        prob = torch.sigmoid(tcn(pp)).squeeze(1).float().cpu().numpy()
        out += [extract_events(prob[j], threshold) for j in range(prob.shape[0])]
    return out


def build_warps(schedules):
    return [BW.IntegerEventWarp(np.asarray(s, dtype=np.int64)) for s in schedules]


def warp_block(A, warps, direction, dev, batch=512):
    out = np.empty_like(A)
    for i in range(0, len(A), batch):
        t = torch.from_numpy(np.ascontiguousarray(A[i:i + batch])).to(dev).unsqueeze(1)
        out[i:i + batch] = O2W.apply_warp(t, warps[i:i + batch], direction).squeeze(1).cpu().numpy()
    return out


@torch.no_grad()
def gen_canonical(net, Xcan, bank, dev):
    outs, got = [], set()
    for i in range(0, len(Xcan), BATCH):
        pp = torch.from_numpy(np.ascontiguousarray(Xcan[i:i + BATCH])).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, pp, bank[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
        got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {NFE}
    return np.concatenate(outs)


def o2c_predict(o2c, X, warps, bank, dev):
    """PPG --W_S--> canonical -> frozen O2c -> --W_S^-1--> raw. The SAME warp is used in both directions."""
    xcan = warp_block(X, warps, "to_canonical", dev)
    can = gen_canonical(o2c, xcan, bank, dev)
    return warp_block(can, warps, "to_raw", dev), can


def aligned_rows(pred, gt_tg, iqr):
    tg = R2E.pmap(S0._targets, list(pred.astype(np.float64)))
    rows = []
    for i in range(len(pred)):
        row = {}
        for t in ALIGNED:
            a, b = gt_tg[i][t], tg[i][t]
            row[NAE[t]] = float(abs(b - a) / iqr[t]) if np.isfinite(a) and np.isfinite(b) else np.nan
        rows.append(row)
    return rows


def paired_boot(rows_b, al_b, rows_c, al_c, SUB, CLUSTER, metrics=BOOT_M):
    """Positive always means the supplied-schedule O2c arm is better than B. default_rng(20260904)."""
    out = {}
    for m in metrics:
        if m.startswith("nAE_"):
            a = np.array([r[m] for r in al_b], float); b = np.array([r[m] for r in al_c], float)
            d = a - b
        else:
            a = np.array([r[m] for r in rows_b], float); b = np.array([r[m] for r in rows_c], float)
            d = (b - a) if orient(m) == "higher_better" else (a - b)
        res = O1E.cluster_bootstrap(d, SUB, CLUSTER, n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
        out[m] = {"point": res["point"], "lo": res["lo"], "hi": res["hi"], "verdict": res["verdict"],
                  "B": macro(a, SUB), "arm": macro(b, SUB)}      # equal-subject macro, as everywhere else
    return out


def shape_only(pred, Yd, supplied, gt_pk, retained_idx, iqr, SUB, fs=FS):
    """Beat-identity SHAPE-ONLY diagnostic (preregistration section 18). Cannot enter G1-G6.

    `matched_coverage` is the +-50 ms match rate of retained beats; primitive usability (the QRS core fitting
    inside the window, exactly the O1 validity rule) is reported separately so the two are never conflated.
    """
    from ppg2ecg.evaluation import rpeaks as RP
    gen_pk = R2E.pmap(S0._peaks, list(pred.astype(np.float64)))
    subs = sorted(set(np.asarray(SUB).tolist()))
    per = {u: {"ret": 0, "match": 0, "usable": 0, **{t: [] for t in ALIGNED}} for u in subs}
    for i in range(len(pred)):
        s = np.asarray(supplied[i], dtype=np.int64)
        keep = np.asarray(retained_idx[i], dtype=np.int64)          # index into s; GT identity known
        if keep.size == 0:
            continue
        u = per[SUB[i]]
        sub = s[keep[:, 0]]
        gtb = np.asarray(gt_pk[i], dtype=np.int64)[keep[:, 1]]
        u["ret"] += int(sub.size)
        m, _fp, _fn = RP.match_rpeaks(sub, np.asarray(gen_pk[i], dtype=np.int64), fs, 50.0)
        u["match"] += int(len(m))
        for a, b in m:
            pg = O3.beat_primitives(pred[i], int(gen_pk[i][b]), fs)
            pt = O3.beat_primitives(Yd[i], int(gtb[a]), fs)
            if pg is None or pt is None:
                continue
            u["usable"] += 1
            for t in ALIGNED:
                if np.isfinite(pg[t]) and np.isfinite(pt[t]):
                    u[t].append(abs(pg[t] - pt[t]) / iqr[t])
    ret = sum(per[u]["ret"] for u in subs)
    mat = sum(per[u]["match"] for u in subs)
    use = sum(per[u]["usable"] for u in subs)
    out = {"retained_beats": ret, "matched_beats": mat, "primitive_usable_beats": use,
           "matched_coverage": float(np.mean([per[u]["match"] / per[u]["ret"] for u in subs if per[u]["ret"]]))
                               if any(per[u]["ret"] for u in subs) else np.nan,
           "primitive_usable_fraction": (use / mat) if mat else np.nan}
    for t in ALIGNED:
        vals = [float(np.mean(per[u][t])) for u in subs if per[u][t]]
        out[NAE[t]] = float(np.mean(vals)) if vals else np.nan     # equal-subject macro
        out[f"n_{NAE[t]}"] = int(sum(len(per[u][t]) for u in subs))
    return out
