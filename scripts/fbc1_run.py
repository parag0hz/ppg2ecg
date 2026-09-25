"""FBC1 — Functional Budget Calibration: small-calibration static (K, S) allocation
(docs/FBC1_CALIBRATION_ALLOCATION_PREREGISTRATION.md).

Given a frozen generator, the HR functional and a fixed vector-field NFE budget B = K x S, select ONE allocation (K, S)
from a small labeled calibration set and use it for every future window. No per-instance adaptation.

  window loss   l_{p,j,a} = | median_{k<K} HR(x_{p,j,S,k}) - HR*_{p,j} |   (finite draws; none finite -> frozen c_train)
  patient loss  L_{p,a}   = mean_j l_{p,j,a};   risk R_a = mean_p L_{p,a}  (patient-macro, primary)
  FBC-ERM       argmin_a mean_{p in C} L_{p,a}
  FBC-UCB       argmin_a mean + t_{0.90, n-1} * SD / sqrt(n)            (primary method)
  ties          |R_a - R_min| <= 1e-9 -> lower K

Stages (in this order; later stages check earlier artifacts and commits):
  audit    bank shapes, draw / window non-finite rates, availability, sha256 (label-free)       -> audit.json
  grid     candidate grid, fallback constant, calibration subset manifest, prereg manifest (label-free)
  fullval  informativeness gate, full-validation risk landscape, a_fullval, delta_near         -> frozen_method.json
           (no calibration-subset result is computed here)
  nested   calibration subsets -> ERM / UCB -> held-out evaluation inside validation (PRIMARY), population bootstrap,
           criteria, verdict (requires frozen_method.json committed)
  test     legacy VitalDB TEST confirmation of the frozen selections (descriptive; changes nothing)
  latency  compute accounting (NFE, sequential / batched wall-clock, peak GPU memory, per-sample overhead, HR functional)
  figure   figure.png
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/fbc1_run.py <stage>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

ART = ROOT / "artifacts/fbc1_calibration_allocation"
OUT = ROOT / "outputs/fbc1_calibration_allocation"
M1OUT = ROOT / "outputs/m1_pilot_gain_prediction"
DW1RAW = ROOT / "outputs/dw1_raw"
TEST_REF = ROOT / "outputs/sr1_eval/arm_I_seed42.npz"
TRAIN_HR = ROOT / "outputs/db1_hr_regressor/train_hr_labels.npz"
PREREG_DOC = "docs/FBC1_CALIBRATION_ALLOCATION_PREREGISTRATION.md"

SEED = 20260925
MODELS = ("I", "D", "C")
NAME = {"I": "iMF", "D": "CD", "C": "PENGUIN"}
BUDGETS = (32, 16)
GRID = {32: [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)],          # width -> depth (K descending)
        16: [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]}
BALANCED = {32: (8, 4), 16: (4, 4)}
HISTORICAL = {32: {"I": (16, 2), "D": (32, 1), "C": (8, 4)},              # DW1 frozen best cells (result.json)
              16: {"I": (8, 2), "D": (16, 1), "C": (8, 2)}}
NCAL = (5, 10, 25, 50, 100, 200)
NSUB = 200
HEADLINE_N = 25
CONF = 0.90
TIE_TOL = 1e-9
DELTA_FRAC, DELTA_FLOOR, DELTA_CEIL = 0.10, 0.05, 0.25
NI_MARGIN = 0.025
NEAR_TARGET = 0.80
NBOOT_VAL, NBOOT_GATE, NBOOT_TEST = 2000, 2000, 5000
SNAP = 3
N_VAL_PATIENTS, N_TEST_PATIENTS = 289, 1156
FIXED = ("pure_width", "pure_depth", "balanced", "historical", "fullval")
METHODS = ("UCB", "ERM")


# ============================================================================ helpers
def sha(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def snap(x):
    return np.round(np.asarray(x, np.float64), SNAP)


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)


def require_committed(path):
    rel = str(Path(path).relative_to(ROOT))
    assert git("ls-files", "--error-unmatch", rel).returncode == 0, f"{rel} is not committed"
    assert git("diff", "--quiet", "HEAD", "--", rel).returncode == 0, f"{rel} differs from HEAD"


def cells(B):
    return GRID[B]


def arm_index(B, cell):
    return GRID[B].index(tuple(cell))


def argmin_tie(V, tol=TIE_TOL):
    """row-wise argmin over arms ordered K descending; values within tol of the minimum are tied -> lower K (later)."""
    V = np.atleast_2d(V)
    m = V.min(1, keepdims=True)
    mask = V <= m + tol
    return V.shape[1] - 1 - np.argmax(mask[:, ::-1], 1)


@lru_cache(maxsize=None)
def tq(n):
    return float(stats.t.ppf(CONF, n - 1))


def pct(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    return [float(np.nanpercentile(x, q[0])), float(np.nanpercentile(x, q[1]))]


def c_train():
    h = np.load(TRAIN_HR)["hr"]
    return float(snap(np.nanmedian(h)))


# ============================================================================ banks
def bank_files(split):
    f = {}
    for m in MODELS:
        if split == "val":
            f[(m, 1)] = [M1OUT / f"val_hr_{m}_S1.npy", OUT / f"val_hr_{m}_S1_seeds16_31.npy"]
            for S in (2, 4, 8):
                f[(m, S)] = [M1OUT / f"val_hr_{m}_S{S}.npy"]
            f[(m, 16)] = [OUT / f"val_hr_{m}_S16_seeds0_1.npy"]
            f[(m, 32)] = [OUT / f"val_hr_{m}_S32_seeds0_0.npy"]
        else:
            for S in (1, 2, 4, 8, 16, 32):
                f[(m, S)] = [DW1RAW / f"hr_{m}{S}.npy"]
    return f


def load_split(split):
    """HR banks (rows = noise seeds 0..K-1, snapped to 1e-3 bpm), reference HR, patient IDs, in load order."""
    banks = {}
    for (m, S), fs in bank_files(split).items():
        Z = np.vstack([np.load(p) for p in fs]).astype(np.float64)
        banks[(m, S)] = snap(Z[:32 // S])
        assert banks[(m, S)].shape[0] == 32 // S, (split, m, S, Z.shape)
    if split == "val":
        z = np.load(M1OUT / "val_ref.npz"); ref, pid = z["ref_hr"], z["pid"]
    else:
        z = np.load(TEST_REF, allow_pickle=True); ref, pid = z["ref_hr"], z["pid"]
    for v in banks.values():
        assert v.shape[1] == len(ref)
    assert np.isfinite(ref).all()
    return banks, snap(ref), np.asarray(pid)


def cell_estimates(banks, m, B, c_fb):
    """(A, W) consensus estimates with fallback, fallback mask, and the estimates without fallback (NaN)."""
    est, raw, fb = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for K, S in cells(B):
            e = np.nanmedian(banks[(m, S)][:K], 0)
            raw.append(e.copy()); f = ~np.isfinite(e); fb.append(f); e[f] = c_fb; est.append(e)
    return np.array(est), np.array(fb), np.array(raw)


def patient_index(pid):
    ups = np.unique(pid)
    return ups, np.searchsorted(ups, pid)


def aggregate(loss, g, P):
    """loss (A, W) -> patient-macro losses L (P, A), window sums (P, A), window counts (P,)."""
    cnt = np.bincount(g, minlength=P).astype(float)
    sums = np.stack([np.bincount(g, weights=l, minlength=P) for l in loss], 1)
    return sums / cnt[:, None], sums, cnt


def split_tables(split, c_fb):
    banks, ref, pid = load_split(split)
    ups, g = patient_index(pid)
    T = {"ref": ref, "pid": pid, "ups": ups, "g": g}
    for m in MODELS:
        for B in BUDGETS:
            est, fb, raw = cell_estimates(banks, m, B, c_fb)
            loss = np.abs(est - ref[None])
            L, Ws, Wn = aggregate(loss, g, len(ups))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                drop = np.abs(raw - ref[None])
                cnt_ok = np.stack([np.bincount(g, weights=np.isfinite(d).astype(float), minlength=len(ups)) for d in drop], 1)
                sum_ok = np.stack([np.bincount(g, weights=np.nan_to_num(d), minlength=len(ups)) for d in drop], 1)
                Ldrop = sum_ok / cnt_ok
            T[(m, B)] = {"est": est, "fb": fb, "loss": loss, "L": L, "Wsum": Ws, "Wn": Wn, "Ldrop": Ldrop}
    return T


# ============================================================================ core selection / evaluation
def evaluate_subsets(L, M, Wsum=None, Wn=None):
    """L (P, A) patient losses, M (R, P) bool calibration membership. Per subset: calibration mean / UCB, held-out risk."""
    Mf = M.astype(np.float64)
    n = Mf.sum(1)[:, None]
    s1 = Mf @ L
    s2 = Mf @ (L * L)
    mu = s1 / n
    var = np.maximum(s2 - n * mu * mu, 0.0) / (n - 1)
    t = np.array([tq(int(k)) for k in n[:, 0]])[:, None]
    ucb = mu + t * np.sqrt(var / n)
    RE = (L.sum(0)[None] - s1) / (L.shape[0] - n)
    out = {"n": n[:, 0].astype(int), "mu": mu, "sd": np.sqrt(var), "ucb": ucb, "RE": RE,
           "ERM": argmin_tie(mu), "UCB": argmin_tie(ucb), "oracle": argmin_tie(RE), "omin": RE.min(1)}
    if Wsum is not None:
        out["REm"] = (Wsum.sum(0)[None] - Mf @ Wsum) / (Wn.sum() - Mf @ Wn)[:, None]
    return out


def fixed_arms(B, m, a_full):
    A = len(cells(B))
    return {"pure_width": 0, "pure_depth": A - 1, "balanced": arm_index(B, BALANCED[B]),
            "historical": arm_index(B, HISTORICAL[B][m]), "fullval": int(a_full)}


def summarise(ev, groups, fixed, delta, a_full, with_micro=False):
    """group-level summaries (one value per calibration size). groups: list of index arrays into subsets."""
    R = np.arange(len(ev["n"]))
    res = {}
    pols = {k: ev[k] for k in METHODS}
    pols.update({k: np.full(len(R), v) for k, v in fixed.items()})
    reg = {k: ev["RE"][R, s] - ev["omin"] for k, s in pols.items()}
    full_risk = ev["RE"][:, a_full]
    for k, s in pols.items():
        r_ = reg[k]; risk = ev["RE"][R, s]
        res[f"{k}|mean_regret"] = np.array([r_[gi].mean() for gi in groups])
        res[f"{k}|near_rate"] = np.array([(r_[gi] <= delta + 1e-12).mean() for gi in groups])
        res[f"{k}|material_rate"] = np.array([(r_[gi] > 2 * delta + 1e-12).mean() for gi in groups])
        res[f"{k}|mean_risk"] = np.array([risk[gi].mean() for gi in groups])
        if k in METHODS:
            res[f"{k}|median_regret"] = np.array([np.median(r_[gi]) for gi in groups])
            res[f"{k}|oracle_recovery"] = np.array([(s[gi] == ev["oracle"][gi]).mean() for gi in groups])
            res[f"{k}|fullval_recovery"] = np.array([(s[gi] == a_full).mean() for gi in groups])
            res[f"{k}|within_delta_of_fullval"] = np.array([((risk - full_risk)[gi] <= delta + 1e-12).mean() for gi in groups])
            res[f"{k}|minus_fullval"] = np.array([(risk - full_risk)[gi].mean() for gi in groups])
            A = ev["RE"].shape[1]
            ent = []
            for gi in groups:
                p = np.bincount(s[gi], minlength=A) / len(gi); p = p[p > 0]
                ent.append(float(-(p * np.log(p)).sum()))
            res[f"{k}|entropy"] = np.array(ent)
            if with_micro:
                rm = ev["REm"][R, s] - ev["REm"].min(1)
                res[f"{k}|micro_mean_regret"] = np.array([rm[gi].mean() for gi in groups])
    for k in ("ERM",) + FIXED:
        d = reg["UCB"] - reg[k]
        res[f"UCB_minus_{k}|regret"] = np.array([d[gi].mean() for gi in groups])
    d = reg["ERM"] - reg["historical"]
    res["ERM_minus_historical|regret"] = np.array([d[gi].mean() for gi in groups])
    return res, reg, pols


# ============================================================================ stage: audit
def audit_stage():
    ART.mkdir(parents=True, exist_ok=True)
    out = {"date": "2026-09-25", "label_use": "none (only reference finiteness, shapes, non-finite rates, sha256)", "splits": {}}
    rows = []
    for split in ("val", "test"):
        files = bank_files(split)
        if split == "val":
            ref_f = M1OUT / "val_ref.npz"; z = np.load(ref_f); ref, pid = z["ref_hr"], z["pid"]
        else:
            ref_f = TEST_REF; z = np.load(ref_f, allow_pickle=True); ref, pid = z["ref_hr"], z["pid"]
        sp = {"n_windows": int(len(ref)), "n_patients": int(len(np.unique(pid))), "reference_finite": float(np.isfinite(ref).mean()),
              "reference_file": str(ref_f.relative_to(ROOT)), "reference_sha256": sha(ref_f), "banks": {}}
        for (m, S), fs in files.items():
            Z = np.vstack([np.load(p) for p in fs])
            sp["banks"][f"{NAME[m]}_S{S}"] = {"files": [str(p.relative_to(ROOT)) for p in fs], "sha256": [sha(p) for p in fs],
                                            "rows_available": int(Z.shape[0]), "rows_used": 32 // S,
                                            "draw_nonfinite_rate_rows_used": float(np.isnan(Z[:32 // S]).mean())}
        for m in MODELS:
            for B in BUDGETS:
                for K, S in cells(B):
                    Z = np.vstack([np.load(p) for p in files[(m, S)]])[:K]
                    none_finite = float((~np.isfinite(Z)).all(0).mean())
                    gen = split == "val" and ((S == 1 and K > 16) or S in (16, 32))
                    src = ("M1 validation bank (m1_val_bank.py)" + (" + FBC1 seeds 16-31" if S == 1 and K > 16 else "")
                           if split == "val" and S in (1, 2, 4, 8) else "FBC1 generation (fbc1_val_bank.py)" if split == "val"
                           else "DW1 test bank (dw1_depth_width.py; rows 0-15 at S=1,2 and S=1 for CD/PENGUIN copied from AB1)")
                    rows.append({"model": NAME[m], "split": split, "B": B, "K": K, "S": S, "available": bool(Z.shape[0] >= K),
                                 "source": src, "needs_generation": gen, "draw_nonfinite_rate": float(np.isnan(Z).mean()),
                                 "window_no_finite_draw_rate": none_finite})
        out["splits"][split] = sp
    out["cells"] = rows
    chk = OUT / "val_bank_check.json"
    out["validation_generation_check"] = json.loads(chk.read_text()) if chk.exists() else None
    out["train_reference_hr"] = {"file": str(TRAIN_HR.relative_to(ROOT)), "sha256": sha(TRAIN_HR),
                                 "n": int(len(np.load(TRAIN_HR)["hr"])), "median_snapped": c_train()}
    ck = {"iMF": "outputs/v1_vitaldb_armI_seed42/checkpoint_last.pt", "CD": "outputs/cd1_vitaldb_armD_seed42/checkpoint_last.pt",
          "PENGUIN": "outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt"}
    out["generators"] = {k: {"checkpoint": v, "sha256": sha(ROOT / v), "training_seed": 42} for k, v in ck.items()}
    out["samplers"] = {"iMF": "ER.sample_meanflow_schedule, uniform [1/S]*S", "CD": "f_consistency at t=0 then re-noise at t=j/S with "
                       "S-1 extra N(0, I) tensors drawn from the same generator after z0", "PENGUIN": "euler_sample, S steps",
                       "code": "scripts/dw1_depth_width.py make_sampler (batch 512)",
                       "noise": "row k = torch.Generator().manual_seed(k), z0 = randn(n_windows, 1, 512) over the whole split"}
    out["hr_functional"] = "v1_evaluate._hr: neurokit R-peaks, 60*fs*(n-1)/span at 128 Hz; NaN iff < 2 peaks"
    vp = set(np.unique(np.load(M1OUT / "val_ref.npz")["pid"])); tp = set(np.unique(np.load(TEST_REF, allow_pickle=True)["pid"]))
    out["patients_shared_val_test"] = len(vp & tp)
    (ART / "audit.json").write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['model']:8s} {r['split']:4s} B{r['B']:2d} ({r['K']:2d},{r['S']:2d}) avail={r['available']} gen={r['needs_generation']} "
              f"draw-nan {100 * r['draw_nonfinite_rate']:.2f}% window-no-finite {100 * r['window_no_finite_draw_rate']:.3f}%")


# ============================================================================ stage: grid (label-free, before prereg commit)
def grid_stage():
    ART.mkdir(parents=True, exist_ok=True)
    cfb = c_train()
    grid = {"budget_unit": "generative vector-field evaluations (NFE) per window; B = K x S",
            "primary_budget": 32, "secondary_budget": 16,
            "cells": {str(B): [{"K": K, "S": S, "nfe": K * S} for K, S in cells(B)] for B in BUDGETS},
            "arm_order": "K descending (pure width first, pure depth last); ties -> lower K",
            "baselines": {"pure_width": {str(B): list(cells(B)[0]) for B in BUDGETS},
                          "pure_depth": {str(B): list(cells(B)[-1]) for B in BUDGETS},
                          "balanced": {str(B): list(BALANCED[B]) for B in BUDGETS},
                          "historical_static": {str(B): {NAME[m]: list(HISTORICAL[B][m]) for m in MODELS} for B in BUDGETS},
                          "historical_source": "artifacts/dw1_depth_width/result.json best (K, S) per model and budget "
                                               "(DW1 prereg cc902b3, legacy VitalDB test; seed-stable in DW2 at B = 32)"},
            "draws": "noise seeds 0..K-1 at depth S (prefix reuse); HR = v1_evaluate._hr snapped to 1e-3 bpm",
            "consensus": "median of the finite draw HRs; no finite draw -> fallback constant",
            "fallback_constant_bpm": cfb,
            "fallback_source": {"file": str(TRAIN_HR.relative_to(ROOT)), "sha256": sha(TRAIN_HR),
                                "definition": "median reference HR over the VitalDB TRAINING windows (DB1 labels), snapped"},
            "samplers": {"iMF": "uniform MeanFlow schedule [1/S]*S", "CD": "multistep consistency, re-noise at t=j/S",
                         "PENGUIN": "forward Euler, S steps"},
            "checkpoints": {"iMF": "outputs/v1_vitaldb_armI_seed42", "CD": "outputs/cd1_vitaldb_armD_seed42",
                            "PENGUIN": "outputs/v1_vitaldb_armC_seed42"}}
    (ART / "candidate_grid.json").write_text(json.dumps(grid, indent=1))

    ups = np.unique(np.load(M1OUT / "val_ref.npz")["pid"])
    assert len(ups) == N_VAL_PATIENTS
    rng = np.random.default_rng(SEED)
    subs = {}
    for n in NCAL:
        seen, lst = set(), []
        while len(lst) < NSUB:
            s = tuple(sorted(int(i) for i in rng.choice(len(ups), n, replace=False)))
            if s in seen:
                continue
            seen.add(s); lst.append([int(ups[i]) for i in s])
        subs[str(n)] = lst
    man = {"seed": SEED, "n_validation_patients": int(len(ups)),
           "rule": "rng = numpy.random.default_rng(20260925); for n in (5, 10, 25, 50, 100, 200) in this order: repeat "
                   "s = sorted(rng.choice(289, n, replace=False)) over the sorted unique validation patient IDs, skip a subset "
                   "already drawn for this n, until 200; stored as patient IDs. Evaluation set = the other validation patients.",
           "n_subsets_per_size": NSUB, "sizes": list(NCAL), "subsets": subs}
    (ART / "calibration_subsets.json").write_text(json.dumps(man))

    files = {}
    for split in ("val", "test"):
        for (m, S), fs in bank_files(split).items():
            for p in fs:
                files[str(p.relative_to(ROOT))] = sha(p)
    files[str((M1OUT / "val_ref.npz").relative_to(ROOT))] = sha(M1OUT / "val_ref.npz")
    files[str(TEST_REF.relative_to(ROOT))] = sha(TEST_REF)
    files[str(TRAIN_HR.relative_to(ROOT))] = sha(TRAIN_HR)
    pm = {"head_at_freeze": git("rev-parse", "HEAD").stdout.strip(),
          "prereg_doc": PREREG_DOC, "prereg_doc_sha256": sha(ROOT / PREREG_DOC) if (ROOT / PREREG_DOC).exists() else None,
          "audit_json_sha256": sha(ART / "audit.json"), "candidate_grid_sha256": sha(ART / "candidate_grid.json"),
          "calibration_subsets_sha256": sha(ART / "calibration_subsets.json"),
          "code_sha256": {"scripts/fbc1_run.py": sha(Path(__file__)), "scripts/fbc1_val_bank.py": sha(ROOT / "scripts/fbc1_val_bank.py")},
          "input_sha256": files,
          "no_result_computed": "no validation or test risk, loss, selection or regret had been computed at freeze"}
    (ART / "prereg_manifest.json").write_text(json.dumps(pm, indent=1))
    print(f"[fbc1] grid written; fallback c_train = {cfb}; subsets sha256 {pm['calibration_subsets_sha256'][:16]}")


# ============================================================================ stage: fullval (margins; no subset result)
def gate_and_landscape(T, rng):
    """informativeness gate (per model / budget / cell) and full-validation risk landscape with patient bootstrap."""
    ref, g, ups = T["ref"], T["g"], T["ups"]
    P = len(ups)
    const = float(snap(np.median(ref)))
    Lc, _, _ = aggregate(np.abs(const - ref)[None], g, P)
    Lc = Lc[:, 0]
    win = [np.flatnonzero(g == p) for p in range(P)]
    draws = [rng.integers(0, P, P) for _ in range(NBOOT_GATE)]
    wts = np.array([np.bincount(d, minlength=P) for d in draws], float)
    widx = [np.concatenate([win[p] for p in d]) for d in draws]
    rows = []
    for m in MODELS:
        for B in BUDGETS:
            t = T[(m, B)]
            L = t["L"]
            risk_b = wts @ L / P
            diff_b = wts @ (L - Lc[:, None]) / P
            for a, (K, S) in enumerate(cells(B)):
                rho = float(stats.spearmanr(t["est"][a], ref)[0])
                rb = [stats.spearmanr(t["est"][a][idx], ref[idx])[0] for idx in widx]
                mc = float((L[:, a] - Lc).mean())
                rows.append({"model": NAME[m], "B": B, "K": K, "S": S, "nfe": K * S,
                             "risk_macro": float(L[:, a].mean()), "risk_macro_ci": pct(risk_b[:, a]),
                             "risk_micro": float(t["loss"][a].mean()), "risk_macro_dw1_dropnan": float(np.nanmean(t["Ldrop"][:, a])),
                             "fallback_rate": float(t["fb"][a].mean()), "patient_sd": float(L[:, a].std(ddof=1)),
                             "gateA_minus_constant": mc, "gateA_ci": pct(diff_b[:, a]),
                             "gateB_spearman": rho, "gateB_ci": pct(rb)})
                r = rows[-1]
                r["gate_pass"] = bool(r["gateA_ci"][1] < 0 and r["gateB_ci"][0] > 0)
    return rows, const, float(Lc.mean())


def fullval_stage():
    require_committed(ART / "calibration_subsets.json"); require_committed(ART / "candidate_grid.json")
    require_committed(ROOT / PREREG_DOC)
    assert not (ART / "frozen_method.json").exists(), "frozen_method.json already exists"
    cfb = json.loads((ART / "candidate_grid.json").read_text())["fallback_constant_bpm"]
    T = split_tables("val", cfb)
    rows, const, const_mae = gate_and_landscape(T, np.random.default_rng([SEED, 2]))
    ref = {}
    for m in MODELS:
        for B in BUDGETS:
            L = T[(m, B)]["L"]; R = L.mean(0)
            best = int(argmin_tie(R[None])[0])
            order = [int(i) for i in np.argsort(R, kind="stable")]
            second = [i for i in order if i != best][0]
            d = L[:, best] - L[:, second]
            se = float(d.std(ddof=1) / np.sqrt(len(d))); tc = float(stats.t.ppf(0.975, len(d) - 1))
            A = len(cells(B))
            gap_depth = float(R[A - 1] - R[best])
            if gap_depth > 0:
                gap, rule = gap_depth, "R_val(pure depth) - R_val(full-validation best)"
            else:
                gap, rule = float(np.max(np.abs(R - R[best]))), "max_a |R_val(a) - R_val(best)| (pure depth not worse)"
            delta = float(np.clip(DELTA_FRAC * gap, DELTA_FLOOR, DELTA_CEIL))
            rm = T[(m, B)]["loss"].mean(1)
            gate_rows = [r for r in rows if r["model"] == NAME[m] and r["B"] == B]
            ref[f"{NAME[m]}|B{B}"] = {
                "a_fullval": list(cells(B)[best]), "risk_fullval": float(R[best]),
                "risks": {f"({K},{S})": float(R[i]) for i, (K, S) in enumerate(cells(B))},
                "second_best": list(cells(B)[second]), "paired_best_minus_second": {"mean": float(d.mean()), "se": se,
                                                                                   "ci95_t": [float(d.mean() - tc * se), float(d.mean() + tc * se)]},
                "gap_scale": gap, "gap_rule": rule, "delta_near": delta,
                "micro_best": list(cells(B)[int(argmin_tie(rm[None])[0])]),
                "gate_pass_all_cells": bool(all(r["gate_pass"] for r in gate_rows))}
    with open(ART / "validation_full_grid.csv", "w", newline="") as fh:
        flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
        w = csv.DictWriter(fh, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    fr = {"split": "VitalDB validation, all 289 patients / 4,822 windows", "reference_selector": "achievable reference, not an oracle",
          "gate_constant_bpm": const, "gate_constant_mae": const_mae, "gate_bootstrap": f"{NBOOT_GATE} patient replicates, default_rng([{SEED}, 2])",
          "per_model_budget": ref}
    (ART / "fullval_reference.json").write_text(json.dumps(fr, indent=1))
    prereg_commit = git("log", "-n", "1", "--format=%H", "--", PREREG_DOC).stdout.strip()
    frozen = {
        "method": "Functional Budget Calibration (FBC)", "primary": "FBC-UCB", "baseline_variant": "FBC-ERM",
        "prereg_commit": prereg_commit, "prereg_doc": PREREG_DOC, "prereg_doc_sha256": sha(ROOT / PREREG_DOC),
        "candidate_grid": {str(B): [list(c) for c in cells(B)] for B in BUDGETS},
        "candidate_grid_sha256": sha(ART / "candidate_grid.json"),
        "risk": "patient-macro mean over patients of the mean window |median_k HR - HR*| (primary); window-micro secondary",
        "consensus": "median of finite draw HRs (seeds 0..K-1 at depth S); none finite -> fallback constant",
        "fallback_constant_bpm": cfb, "hr_snap_decimals": SNAP,
        "FBC_ERM": "argmin_a mean_{p in C} L_{p,a}",
        "FBC_UCB": "argmin_a mean_{p in C} L_{p,a} + t_{0.90, n-1} * SD_{p in C}(L_{p,a}) / sqrt(n)  (SD ddof 1)",
        "confidence_level_one_sided": CONF, "tie_rule": f"|R_a - R_min| <= {TIE_TOL} -> lower K",
        "calibration_sizes": list(NCAL), "headline_n": HEADLINE_N, "subsets_per_size": NSUB,
        "calibration_subsets_sha256": sha(ART / "calibration_subsets.json"),
        "delta_near": {k: v["delta_near"] for k, v in ref.items()},
        "delta_rule": "clip(0.10 x Gap_scale, 0.05, 0.25); Gap_scale = R_val(pure depth) - R_val(full-val best), or max_a |R_val(a) - R_val(best)| if pure depth is not worse",
        "a_fullval": {k: v["a_fullval"] for k, v in ref.items()},
        "baselines": {"pure_width": {str(B): list(cells(B)[0]) for B in BUDGETS}, "pure_depth": {str(B): list(cells(B)[-1]) for B in BUDGETS},
                      "balanced": {str(B): list(BALANCED[B]) for B in BUDGETS},
                      "historical_static": {str(B): {NAME[m]: list(HISTORICAL[B][m]) for m in MODELS} for B in BUDGETS}},
        "gate_pass": {k: v["gate_pass_all_cells"] for k, v in ref.items()},
        "success_criteria": "preregistration section 12 (A-D STRONG, F1-F4 FAILED, PARTIAL labels, GO rule); non-inferiority margin "
                            f"{NI_MARGIN} bpm, near-optimal target {NEAR_TARGET}",
        "bootstrap": {"validation_population": f"{NBOOT_VAL} replicates, default_rng([{SEED}, 1])",
                      "test": f"{NBOOT_TEST} replicates, default_rng([{SEED}, 3])"},
        "code_sha256": sha(Path(__file__)),
        "status": "frozen before any calibration-subset result and before any VitalDB TEST read"}
    (ART / "frozen_method.json").write_text(json.dumps(frozen, indent=1))
    for k, v in ref.items():
        print(f"{k:14s} a_fullval {tuple(v['a_fullval'])} R {v['risk_fullval']:.3f} gap {v['gap_scale']:.3f} delta {v['delta_near']:.3f} "
              f"gate {v['gate_pass_all_cells']} risks {v['risks']}")


# ============================================================================ stage: nested (primary)
def load_subsets(ups):
    man = json.loads((ART / "calibration_subsets.json").read_text())
    pos = {int(u): i for i, u in enumerate(ups)}
    M, groups, k = [], [], 0
    for n in NCAL:
        lst = man["subsets"][str(n)]
        assert len(lst) == NSUB and all(len(s) == n for s in lst)
        for s in lst:
            row = np.zeros(len(ups), bool); row[[pos[int(p)] for p in s]] = True; M.append(row)
        groups.append(np.arange(k, k + NSUB)); k += NSUB
    return np.array(M), groups


def paired_top2(L, M, ev):
    """per subset: calibration winner (lowest mean) minus runner-up, paired over calibration patients."""
    out = []
    for r in range(len(M)):
        order = np.argsort(ev["mu"][r], kind="stable")
        a1 = int(ev["ERM"][r]); a2 = int([i for i in order if i != a1][0])
        d = L[M[r], a1] - L[M[r], a2]
        n = len(d); se = d.std(ddof=1) / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
        out.append((a1, a2, d.mean(), se, d.mean() - tc * se, d.mean() + tc * se))
    return np.array(out)


def verdict_of(pt, ci, delta, gate):
    """preregistration section 12. pt / ci: dicts keyed (model, B, key) -> arrays over NCAL / CI per n.
    A model that failed the informativeness gate at B = 32 cannot count as meeting A, B or D."""
    iN = {n: i for i, n in enumerate(NCAL)}
    h, h50, h100, h5 = iN[HEADLINE_N], iN[50], iN[100], iN[5]
    B = 32
    crit = {}
    for meth in METHODS:
        A_ = [gate[(m, B)] and pt[(m, B, f"{meth}|near_rate")][h] >= NEAR_TARGET for m in MODELS]
        B_ = [gate[(m, B)] and pt[(m, B, f"{meth}|mean_regret")][h] < min(pt[(m, B, "pure_depth|mean_regret")][h], pt[(m, B, "pure_width|mean_regret")][h])
              for m in MODELS]
        D_ = [gate[(m, B)] and ((pt[(m, B, f"{meth}|near_rate")][h50] >= NEAR_TARGET) or
                                (pt[(m, B, f"{meth}|mean_regret")][h50] <= delta[(m, B)] + 1e-12)) for m in MODELS]
        crit[meth] = {"A_models": {NAME[m]: bool(x) for m, x in zip(MODELS, A_)}, "A": bool(sum(A_) >= 2),
                      "B_models": {NAME[m]: bool(x) for m, x in zip(MODELS, B_)}, "B": bool(sum(B_) >= 2),
                      "D_models": {NAME[m]: bool(x) for m, x in zip(MODELS, D_)}, "D": bool(all(D_))}
    pooled = float(np.mean([pt[(m, B, "UCB_minus_ERM|regret")][h] for m in MODELS]))
    pooled_ci = ci[("pooled", B, "UCB_minus_ERM|regret")][h]
    C = bool(pooled < 0 and pooled_ci[1] < NI_MARGIN)
    F1_ = [(pt[(m, B, "UCB|mean_regret")][h100] > delta[(m, B)]) and
           (pt[(m, B, "UCB|mean_regret")][h100] >= 0.75 * pt[(m, B, "UCB|mean_regret")][h5]) for m in MODELS]
    F2_ = [pt[(m, B, "UCB|material_rate")][h50] > 0.20 for m in MODELS]
    F3_ = [(pt[(m, B, "UCB_minus_historical|regret")][h] > delta[(m, B)]) and (ci[(m, B, "UCB_minus_historical|regret")][h][0] > 0)
           for m in MODELS]
    F4_ = [(pt[(m, B, "UCB|near_rate")][h] < 0.50) and (pt[(m, B, "UCB|mean_regret")][h] > delta[(m, B)]) for m in MODELS]
    fails = {"F1_no_learning": bool(sum(F1_) >= 2), "F2_n50_material_suboptimality": bool(sum(F2_) >= 2),
             "F3_worse_than_historical_static": bool(sum(F3_) >= 2), "F4_unstable_at_n25": bool(sum(F4_) >= 2)}
    detail = {"F1_models": dict(zip([NAME[m] for m in MODELS], map(bool, F1_))), "F2_models": dict(zip([NAME[m] for m in MODELS], map(bool, F2_))),
              "F3_models": dict(zip([NAME[m] for m in MODELS], map(bool, F3_))), "F4_models": dict(zip([NAME[m] for m in MODELS], map(bool, F4_)))}
    ucb_ok = crit["UCB"]["A"] and crit["UCB"]["B"] and crit["UCB"]["D"]
    erm_ok = crit["ERM"]["A"] and crit["ERM"]["B"] and crit["ERM"]["D"]
    labels = []
    if any(fails.values()):
        verdict = "FAILED"
    elif ucb_ok and C:
        verdict = "STRONG"
    else:
        verdict = "PARTIAL"
        if ucb_ok and not C:
            labels.append("ERM works but UCB does not add value (criterion C)")
        if not ucb_ok and erm_ok:
            labels.append("calibration selection works, uncertainty-aware variant unsupported")
        n_models_n25 = sum(crit["UCB"]["A_models"].values())
        if n_models_n25 == 1:
            labels.append("only 1 of 3 models meets the n = 25 near-optimality criterion")
        dec = all(pt[(m, B, "UCB|mean_regret")][iN[200]] <= pt[(m, B, "UCB|mean_regret")][h5] for m in MODELS)
        if not ucb_ok and dec:
            labels.append("sample efficiency improves with n but the n = 25 criteria are not met")
    if verdict == "STRONG":
        go = "GO"
    elif verdict == "PARTIAL" and (ucb_ok or erm_ok):
        go = "CONDITIONAL GO (" + ("FBC-UCB" if ucb_ok else "FBC-ERM only; FBC-UCB not supported") + ")"
    else:
        go = "NO-GO"
    return {"criteria": crit, "C": C, "C_pooled_UCB_minus_ERM_regret": pooled, "C_pooled_ci": pooled_ci,
            "fails": fails, "fail_detail": detail, "UCB_meets_ABD": ucb_ok, "ERM_meets_ABD": erm_ok,
            "verdict": verdict, "partial_labels": labels, "go": go}


def nested_stage():
    require_committed(ART / "frozen_method.json")
    fz = json.loads((ART / "frozen_method.json").read_text())
    cfb = fz["fallback_constant_bpm"]
    T = split_tables("val", cfb)
    M, groups = load_subsets(T["ups"])
    delta = {(m, B): fz["delta_near"][f"{NAME[m]}|B{B}"] for m in MODELS for B in BUDGETS}
    afull = {(m, B): arm_index(B, fz["a_fullval"][f"{NAME[m]}|B{B}"]) for m in MODELS for B in BUDGETS}
    pt, EV, per_rows, sel_rows, pair_rows = {}, {}, [], [], []
    for m in MODELS:
        for B in BUDGETS:
            t = T[(m, B)]
            assert int(argmin_tie(t["L"].mean(0)[None])[0]) == afull[(m, B)]
            ev = evaluate_subsets(t["L"], M, t["Wsum"], t["Wn"])
            EV[(m, B)] = ev
            fx = fixed_arms(B, m, afull[(m, B)])
            res, reg, pols = summarise(ev, groups, fx, delta[(m, B)], afull[(m, B)], with_micro=True)
            for k, v in res.items():
                pt[(m, B, k)] = v
            pr = paired_top2(t["L"], M, ev)
            for gi, n in zip(groups, NCAL):
                for r in gi:
                    per_rows.append({"model": NAME[m], "B": B, "n_cal": n, "subset": int(r - gi[0]),
                                     **{f"{k}_{f}": v for k in METHODS for f, v in (
                                         ("K", cells(B)[ev[k][r]][0]), ("S", cells(B)[ev[k][r]][1]),
                                         ("cal_mean", round(float(ev["mu"][r, ev[k][r]]), 6)),
                                         ("cal_ucb", round(float(ev["ucb"][r, ev[k][r]]), 6)),
                                         ("heldout_risk", round(float(ev["RE"][r, ev[k][r]]), 6)),
                                         ("regret", round(float(reg[k][r]), 6)),
                                         ("near_optimal", bool(reg[k][r] <= delta[(m, B)] + 1e-12)),
                                         ("heldout_micro_risk", round(float(ev["REm"][r, ev[k][r]]), 6)))},
                                     "oracle_K": cells(B)[ev["oracle"][r]][0], "oracle_S": cells(B)[ev["oracle"][r]][1],
                                     "oracle_risk": round(float(ev["omin"][r]), 6),
                                     "fullval_heldout_risk": round(float(ev["RE"][r, afull[(m, B)]]), 6),
                                     **{f"regret_{k}": round(float(reg[k][r]), 6) for k in FIXED}})
                for meth in METHODS:
                    cnt = np.bincount(ev[meth][gi], minlength=len(cells(B)))
                    for a, (K, S) in enumerate(cells(B)):
                        sel_rows.append({"model": NAME[m], "B": B, "n_cal": n, "method": meth, "K": K, "S": S,
                                         "frequency": float(cnt[a] / NSUB), "entropy_nats": float(pt[(m, B, f"{meth}|entropy")][NCAL.index(n)]),
                                         "is_fullval": a == afull[(m, B)]})
                p = pr[gi]
                pair_rows.append({"model": NAME[m], "B": B, "n_cal": n, "mean_paired_diff": float(p[:, 2].mean()),
                                  "median_paired_diff": float(np.median(p[:, 2])), "median_paired_se": float(np.median(p[:, 3])),
                                  "share_ci_upper_below_0": float((p[:, 5] < 0).mean()),
                                  "share_top2_contains_fullval": float(((p[:, 0] == afull[(m, B)]) | (p[:, 1] == afull[(m, B)])).mean())})
    fr = json.loads((ART / "fullval_reference.json").read_text())["per_model_budget"]
    for m in MODELS:
        for B in BUDGETS:
            v = fr[f"{NAME[m]}|B{B}"]["paired_best_minus_second"]
            pair_rows.append({"model": NAME[m], "B": B, "n_cal": 289, "mean_paired_diff": v["mean"], "median_paired_diff": v["mean"],
                              "median_paired_se": v["se"], "share_ci_upper_below_0": float(v["ci95_t"][1] < 0),
                              "share_top2_contains_fullval": 1.0})

    # ---------------- patient-clustered population bootstrap (resample the 289 validation patients, rerun everything)
    rng = np.random.default_rng([SEED, 1])
    keys = sorted({k for (_, _, k) in pt if "micro" not in k})
    boot = {(m, B, k): np.empty((NBOOT_VAL, len(NCAL))) for m in MODELS for B in BUDGETS for k in keys}
    t0 = time.time()
    for b in range(NBOOT_VAL):
        idx = rng.integers(0, N_VAL_PATIENTS, N_VAL_PATIENTS)
        for m in MODELS:
            for B in BUDGETS:
                Lb = T[(m, B)]["L"][idx]
                ev = evaluate_subsets(Lb, M)
                ab = int(argmin_tie(Lb.mean(0)[None])[0])
                res, _, _ = summarise(ev, groups, fixed_arms(B, m, ab), delta[(m, B)], ab)
                for k, v in res.items():
                    boot[(m, B, k)][b] = v
        if b % 200 == 0:
            print(f"[fbc1] bootstrap {b}/{NBOOT_VAL} ({time.time() - t0:.0f}s)", flush=True)
    ci = {}
    for (m, B, k), arr in boot.items():
        ci[(m, B, k)] = [pct(arr[:, i]) for i in range(len(NCAL))]
    for B in BUDGETS:
        for k in ("UCB_minus_ERM|regret", "UCB|mean_regret", "ERM|mean_regret"):
            arr = np.mean([boot[(m, B, k)] for m in MODELS], 0)
            ci[("pooled", B, k)] = [pct(arr[:, i]) for i in range(len(NCAL))]
            pt[("pooled", B, k)] = np.mean([pt[(m, B, k)] for m in MODELS], 0)

    gate = {(m, B): bool(fz["gate_pass"][f"{NAME[m]}|B{B}"]) for m in MODELS for B in BUDGETS}
    ver = verdict_of(pt, ci, delta, gate)
    ver["gate_pass"] = {f"{NAME[m]}|B{B}": gate[(m, B)] for m in MODELS for B in BUDGETS}
    ver16 = verdict_b16(pt, ci, delta)

    # ---------------- artifacts
    with open(ART / "nested_calibration_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_rows[0])); w.writeheader(); w.writerows(per_rows)
    with open(ART / "selection_frequency.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sel_rows[0])); w.writeheader(); w.writerows(sel_rows)
    with open(ART / "paired_risk.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0])); w.writeheader(); w.writerows(pair_rows)
    curve = []
    for m in MODELS:
        for B in BUDGETS:
            for i, n in enumerate(NCAL):
                for pol in METHODS + FIXED:
                    row = {"model": NAME[m], "B": B, "n_cal": n, "policy": pol,
                           "mean_regret": pt[(m, B, f"{pol}|mean_regret")][i], "mean_regret_ci": ci[(m, B, f"{pol}|mean_regret")][i],
                           "near_optimal_rate": pt[(m, B, f"{pol}|near_rate")][i], "near_optimal_rate_ci": ci[(m, B, f"{pol}|near_rate")][i],
                           "material_rate": pt[(m, B, f"{pol}|material_rate")][i],
                           "mean_heldout_risk": pt[(m, B, f"{pol}|mean_risk")][i], "mean_heldout_risk_ci": ci[(m, B, f"{pol}|mean_risk")][i]}
                    if pol in METHODS:
                        g = groups[i]
                        rr = np.array([r[f"{pol}_regret"] for r in per_rows if r["model"] == NAME[m] and r["B"] == B and r["n_cal"] == n])
                        row.update({"median_regret": pt[(m, B, f"{pol}|median_regret")][i], "regret_p2.5_p97.5_across_subsets": pct(rr),
                                    "oracle_recovery": pt[(m, B, f"{pol}|oracle_recovery")][i],
                                    "fullval_recovery": pt[(m, B, f"{pol}|fullval_recovery")][i],
                                    "fullval_recovery_ci": ci[(m, B, f"{pol}|fullval_recovery")][i],
                                    "within_delta_of_fullval": pt[(m, B, f"{pol}|within_delta_of_fullval")][i],
                                    "entropy_nats": pt[(m, B, f"{pol}|entropy")][i],
                                    "micro_mean_regret": pt[(m, B, f"{pol}|micro_mean_regret")][i]})
                        assert len(rr) == len(g)
                    curve.append({k: (json.dumps([round(x, 6) for x in v]) if isinstance(v, list) else
                                      (round(float(v), 6) if isinstance(v, (float, np.floating)) else v)) for k, v in row.items()})
    with open(ART / "regret_curves.csv", "w", newline="") as fh:
        fields = list(dict.fromkeys(k for r in curve for k in r))
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(curve)

    def req_n(m, B, meth):
        for i, n in enumerate(NCAL):
            if pt[(m, B, f"{meth}|near_rate")][i] >= NEAR_TARGET or pt[(m, B, f"{meth}|mean_regret")][i] <= delta[(m, B)] + 1e-12:
                return n
        return ">200"
    near = {"delta_near": {f"{NAME[m]}|B{B}": delta[(m, B)] for m in MODELS for B in BUDGETS},
            "near_optimal_rate": {f"{NAME[m]}|B{B}|{meth}": dict(zip(map(str, NCAL), map(float, pt[(m, B, f"{meth}|near_rate")])))
                                  for m in MODELS for B in BUDGETS for meth in METHODS},
            "required_n (near >= 0.80 or mean regret <= delta)": {f"{NAME[m]}|B{B}|{meth}": req_n(m, B, meth)
                                                                   for m in MODELS for B in BUDGETS for meth in METHODS},
            "verdict_B32": ver, "B16_descriptive": ver16}
    (ART / "near_optimality.json").write_text(json.dumps(near, indent=1, default=float))
    bj = {"validation_population_bootstrap": {"replicates": NBOOT_VAL, "rng": f"default_rng([{SEED}, 1])",
                                              "scheme": "resample the 289 validation patients with replacement into 289 slots; the frozen subset "
                                                        "manifest is applied to slot positions; full procedure rerun (selection, held-out risk, "
                                                        "held-out oracle, full-validation selector); delta_near held fixed",
                                              "note": "duplicated patients can fall into both a calibration set and its held-out set in a replicate "
                                                      "(standard population bootstrap of a procedure); CIs are percentile 95 %",
                                              "ci": {f"{m if m == 'pooled' else NAME[m]}|B{B}|{k}": dict(zip(map(str, NCAL), v)) for (m, B, k), v in ci.items()}}}
    (ART / "bootstrap.json").write_text(json.dumps(bj, indent=1))
    np.savez(OUT / "nested_selections.npz", **{f"{m}_{B}_{k}": EV[(m, B)][k] for m in MODELS for B in BUDGETS for k in ("UCB", "ERM", "oracle")})
    print(json.dumps(ver, indent=1, default=float))
    for m in MODELS:
        for B in BUDGETS:
            print(f"{NAME[m]:8s} B{B} delta {delta[(m, B)]:.3f}")
            for i, n in enumerate(NCAL):
                print(f"   n={n:3d} UCB regret {pt[(m, B, 'UCB|mean_regret')][i]:.4f} near {pt[(m, B, 'UCB|near_rate')][i]:.3f} "
                      f"rec {pt[(m, B, 'UCB|fullval_recovery')][i]:.3f} H {pt[(m, B, 'UCB|entropy')][i]:.2f} | ERM regret "
                      f"{pt[(m, B, 'ERM|mean_regret')][i]:.4f} near {pt[(m, B, 'ERM|near_rate')][i]:.3f} | depth "
                      f"{pt[(m, B, 'pure_depth|mean_regret')][i]:.3f} width {pt[(m, B, 'pure_width|mean_regret')][i]:.3f} "
                      f"hist {pt[(m, B, 'historical|mean_regret')][i]:.3f}")


def verdict_b16(pt, ci, delta):
    """descriptive: criteria A-D evaluated at B = 16 (cannot rescue B = 32)."""
    iN = {n: i for i, n in enumerate(NCAL)}
    h, h50, B = iN[HEADLINE_N], iN[50], 16
    out = {}
    for meth in METHODS:
        out[meth] = {
            "A": sum(pt[(m, B, f"{meth}|near_rate")][h] >= NEAR_TARGET for m in MODELS) >= 2,
            "B": sum(pt[(m, B, f"{meth}|mean_regret")][h] < min(pt[(m, B, "pure_depth|mean_regret")][h], pt[(m, B, "pure_width|mean_regret")][h])
                     for m in MODELS) >= 2,
            "D": all((pt[(m, B, f"{meth}|near_rate")][h50] >= NEAR_TARGET) or (pt[(m, B, f"{meth}|mean_regret")][h50] <= delta[(m, B)] + 1e-12)
                     for m in MODELS)}
    out["C_pooled_UCB_minus_ERM_regret"] = float(pt[("pooled", B, "UCB_minus_ERM|regret")][h])
    out["C_pooled_ci"] = ci[("pooled", B, "UCB_minus_ERM|regret")][h]
    return {k: ({kk: bool(vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in out.items()}


# ============================================================================ stage: test (legacy confirmation)
def test_stage():
    require_committed(ART / "frozen_method.json")
    assert (ART / "near_optimality.json").exists(), "nested stage has not run"
    fz = json.loads((ART / "frozen_method.json").read_text())
    cfb = fz["fallback_constant_bpm"]
    V = split_tables("val", cfb)
    M, groups = load_subsets(V["ups"])
    sv = np.load(OUT / "nested_selections.npz")
    T = split_tables("test", cfb)
    assert len(T["ups"]) == N_TEST_PATIENTS
    rng = np.random.default_rng([SEED, 3])
    W = np.array([np.bincount(rng.integers(0, N_TEST_PATIENTS, N_TEST_PATIENTS), minlength=N_TEST_PATIENTS) for _ in range(NBOOT_TEST)], float)
    res = {"label": "Legacy-test confirmation, not fresh prospective validation",
           "note": "VitalDB TEST was analysed before (DW1, DW2, EXP-B, M1, M2). Selections come from validation calibration subsets "
                   "only and were frozen before this stage; the test oracle is descriptive and not deployable.",
           "bootstrap": f"{NBOOT_TEST} patient replicates, default_rng([{SEED}, 3])", "per_model_budget": {}}
    for m in MODELS:
        for B in BUDGETS:
            v = V[(m, B)]
            ev = evaluate_subsets(v["L"], M)
            for k in ("UCB", "ERM"):
                assert np.array_equal(ev[k], sv[f"{m}_{B}_{k}"]), "validation selections differ from the nested stage"
            t = T[(m, B)]
            L = t["L"]; P = L.shape[0]
            R = L.mean(0); Rb = W @ L / P
            Rm = t["Wsum"].sum(0) / t["Wn"].sum(); Rmb = (W @ t["Wsum"]) / (W @ t["Wn"])[:, None]
            omin_b = Rb.min(1)
            fx = fixed_arms(B, m, arm_index(B, fz["a_fullval"][f"{NAME[m]}|B{B}"]))
            d = {"cells": {f"({K},{S})": {"risk_macro": float(R[a]), "ci": pct(Rb[:, a]), "risk_micro": float(Rm[a]), "ci_micro": pct(Rmb[:, a]),
                                          "risk_macro_dw1_dropnan": float(np.nanmean(t["Ldrop"][:, a])), "fallback_rate": float(t["fb"][a].mean())}
                           for a, (K, S) in enumerate(cells(B))},
                 "test_oracle": {"cell": list(cells(B)[int(argmin_tie(R[None])[0])]), "risk_macro": float(R.min()), "ci": pct(omin_b)}}
            for k, a in fx.items():
                d[k] = {"cell": list(cells(B)[a]), "risk_macro": float(R[a]), "ci": pct(Rb[:, a]), "risk_micro": float(Rm[a]),
                        "regret_vs_test_oracle": float(R[a] - R.min()), "regret_ci": pct(Rb[:, a] - omin_b)}
            for k in METHODS:
                for gi, n in zip(groups, NCAL):
                    f = np.bincount(ev[k][gi], minlength=len(cells(B))) / len(gi)
                    er, erb, erm = float(f @ R), Rb @ f, float(f @ Rm)
                    e = {"expected_risk_macro": er, "ci": pct(erb), "expected_risk_micro": erm, "selection_frequency":
                         {f"({K},{S})": float(f[a]) for a, (K, S) in enumerate(cells(B))},
                         "regret_vs_test_oracle": float(er - R.min()), "regret_ci": pct(erb - omin_b),
                         "near_optimal_rate_on_test": float(np.mean([R[a] - R.min() <= fz["delta_near"][f"{NAME[m]}|B{B}"] + 1e-12 for a in ev[k][gi]]))}
                    for bk, a in fx.items():
                        e[f"minus_{bk}"] = float(er - R[a]); e[f"minus_{bk}_ci"] = pct(erb - Rb[:, a])
                    if k == "UCB":
                        fe = np.bincount(ev["ERM"][gi], minlength=len(cells(B))) / len(gi)
                        e["minus_ERM"] = float(er - fe @ R); e["minus_ERM_ci"] = pct(erb - Rb @ fe)
                    d[f"{k}_n{n}"] = e
            res["per_model_budget"][f"{NAME[m]}|B{B}"] = d
    (ART / "legacy_test_confirmation.json").write_text(json.dumps(res, indent=1))
    bj = json.loads((ART / "bootstrap.json").read_text())
    bj["legacy_test_bootstrap"] = {"replicates": NBOOT_TEST, "rng": f"default_rng([{SEED}, 3])", "scheme": "resample the 1,156 test patients "
                                   "(multiplicity weights); selections held fixed; CIs in legacy_test_confirmation.json"}
    (ART / "bootstrap.json").write_text(json.dumps(bj, indent=1))
    for key, d in res["per_model_budget"].items():
        print(f"{key:12s} oracle {tuple(d['test_oracle']['cell'])} {d['test_oracle']['risk_macro']:.3f} | depth {d['pure_depth']['risk_macro']:.3f} "
              f"width {d['pure_width']['risk_macro']:.3f} bal {d['balanced']['risk_macro']:.3f} hist {d['historical']['risk_macro']:.3f} "
              f"fullval {tuple(d['fullval']['cell'])} {d['fullval']['risk_macro']:.3f} | UCB25 {d['UCB_n25']['expected_risk_macro']:.3f} "
              f"ERM25 {d['ERM_n25']['expected_risk_macro']:.3f}")


# ============================================================================ stage: latency / compute accounting
def other_gpu_procs():
    import os
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True)
    return [ln.strip() for ln in q.stdout.splitlines() if ln.strip() and int(ln.split(",")[0]) != os.getpid()]


def latency_stage():
    import torch
    import dw1_depth_width as DW
    import v1_evaluate as V
    import vm1_evaluate as VM
    before = other_gpu_procs()
    assert not before, f"another GPU job is running, not timing concurrently: {before}"
    dev = torch.device("cuda")
    X, _, _ = VM.load("val")
    x1 = X[:1]
    WARM, REPS = 10, 100
    allcells = sorted({c for B in BUDGETS for c in cells(B)}, key=lambda c: (-c[0] * c[1], -c[0]))
    res = {}
    for m in MODELS:
        sampler = DW.make_sampler(m, dev)

        def run(K, S, mode, _s=sampler):
            if mode == "sequential":
                for k in range(K):
                    _s(x1, k, S)
            else:
                _s(np.repeat(x1, K, 0), 0, S)

        per = {}
        for K, S in allcells:
            for mode in ("sequential", "batched"):
                for _ in range(WARM):
                    run(K, S, mode)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(REPS):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    run(K, S, mode)
                    torch.cuda.synchronize(); ts.append(1000 * (time.perf_counter() - t0))
                per[f"({K},{S})|{mode}"] = {"median_ms": float(np.median(ts)), "p10_ms": float(np.percentile(ts, 10)),
                                            "p90_ms": float(np.percentile(ts, 90)), "mean_ms": float(np.mean(ts)),
                                            "peak_gpu_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}
            print(f"[fbc1-lat] {NAME[m]} ({K},{S}) seq {per[f'({K},{S})|sequential']['median_ms']:.1f} ms "
                  f"batched {per[f'({K},{S})|batched']['median_ms']:.1f} ms", flush=True)
        # per-sample fixed overhead: sequential median T(K, S) = K * (c0 + c1 * S), fitted on T / K
        A = np.array([[1.0, S] for K, S in allcells]); y = np.array([per[f"({K},{S})|sequential"]["median_ms"] / K for K, S in allcells])
        c0, c1 = np.linalg.lstsq(A, y, rcond=None)[0]
        # HR functional (CPU) per sample
        Z = sampler(np.repeat(X[:64], 1, 0), 0, 1)
        th = []
        for z in Z:
            t0 = time.perf_counter(); V._hr(z); th.append(1000 * (time.perf_counter() - t0))
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with FlopCounterMode(display=False) as fc:
                sampler(x1, 0, 1)
            f1 = fc.get_total_flops()
            with FlopCounterMode(display=False) as fc:
                sampler(x1, 0, 2)
            f2 = fc.get_total_flops()
            flops = {"per_step_batch1": int(f2 - f1), "first_step_batch1": int(f1),
                     "tool": "torch.utils.flop_counter.FlopCounterMode (built-in; counts matmul / conv / attention ops only)"}
        except Exception as e:  # noqa: BLE001
            flops = {"error": repr(e)}
        res[NAME[m]] = {"per_cell": per, "sequential_fit_ms": {"per_sample_overhead_c0": float(c0), "per_nfe_c1": float(c1)},
                        "hr_functional_ms_per_sample": {"median": float(np.median(th)), "p90": float(np.percentile(th, 90)), "n": len(th)},
                        "flops": flops}
        del sampler; torch.cuda.empty_cache()
    after = other_gpu_procs()
    out = {"budget_statement": "fixed generative vector-field evaluation budget (B = K x S NFE); not equal total compute",
           "other_gpu_processes_at_start": before, "other_gpu_processes_at_end": after,
           "protocol": f"one validation window (content-independent), {WARM} warm-up + {REPS} timed runs per cell and mode; sequential = K "
                       "batch-1 sampler calls of S steps; batched = one call with the window repeated K times; host-side sampler overhead "
                       "(noise generation, transfer) included; HR functional excluded and timed separately on CPU",
           "device": torch.cuda.get_device_name(0),
           "concurrency_note": "checked with nvidia-smi at start (none) and end (listed above); unrelated CPU-bound processes of the "
                               "same user may have been running on the host",
           "cells": {str(B): [{"K": K, "S": S, "nfe": K * S} for K, S in cells(B)] for B in BUDGETS}, "per_model": res}
    (ART / "compute_accounting.json").write_text(json.dumps(out, indent=1))


# ============================================================================ stage: figure
def figure_stage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(ART / "regret_curves.csv")))
    sel = list(csv.DictReader(open(ART / "selection_frequency.csv")))
    lt = json.loads((ART / "legacy_test_confirmation.json").read_text())["per_model_budget"]
    near = json.loads((ART / "near_optimality.json").read_text())
    col = {"iMF": "#1f4e79", "CD": "#8a5a00", "PENGUIN": "#6a3d9a"}
    fig = plt.figure(figsize=(22, 4.8))
    ax = [fig.add_subplot(1, 5, i + 1) for i in range(5)]
    a0 = ax[0]; a0.axis("off")
    boxes = [(0.90, "small labeled calibration set\n(n VitalDB validation patients)", "#eef"),
             (0.66, "per allocation (K, S), K·S = B:\npatient risk mean + t₀.₉₀·SE", "#ffe"),
             (0.42, "single frozen (K, S)\n(FBC-UCB: lowest upper bound)", "#e8f4e8"),
             (0.18, "deployment: every window uses\nK samples × S steps, median HR", "#f4e8e8")]
    for y, s, c in boxes:
        a0.text(0.5, y, s, ha="center", va="center", fontsize=8.5, bbox=dict(fc=c, ec="#777"))
    for y0, y1 in ((0.83, 0.74), (0.59, 0.50), (0.35, 0.26)):
        a0.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->"))
    a0.text(0.5, 0.03, "held-out evaluation: the other validation patients", ha="center", fontsize=7.5)
    a0.set_title("A. Functional Budget Calibration", fontsize=9)
    for m in ("iMF", "CD", "PENGUIN"):
        for meth, ls in (("UCB", "-"), ("ERM", "--")):
            r = [x for x in rows if x["model"] == m and x["B"] == "32" and x["policy"] == meth]
            n = [int(x["n_cal"]) for x in r]; y = [float(x["mean_regret"]) for x in r]
            c = [json.loads(x["mean_regret_ci"]) for x in r]
            ax[1].errorbar(n, y, yerr=[[max(0.0, a - b[0]) for a, b in zip(y, c)], [max(0.0, b[1] - a) for a, b in zip(y, c)]], ls=ls, marker="o", ms=3,
                           color=col[m], capsize=2, label=f"{m} FBC-{meth}")
            nr = [float(x["near_optimal_rate"]) for x in r]
            ax[2].plot(n, nr, ls=ls, marker="o", ms=3, color=col[m], label=f"{m} FBC-{meth}")
        d = near["delta_near"][f"{m}|B32"]
        ax[1].axhline(d, color=col[m], lw=0.6, ls=":")
    ax[1].set_xscale("log"); ax[1].set_xticks(NCAL); ax[1].set_xticklabels(NCAL)
    ax[1].set_xlabel("calibration patients n"); ax[1].set_ylabel("mean held-out regret (bpm), B = 32")
    ax[1].set_title("B. held-out regret (dotted: δ_near)", fontsize=9); ax[1].legend(fontsize=6.5)
    ax[2].axhline(NEAR_TARGET, color="#555", lw=0.6, ls=":")
    ax[2].set_xscale("log"); ax[2].set_xticks(NCAL); ax[2].set_xticklabels(NCAL); ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel("calibration patients n"); ax[2].set_ylabel("P(regret ≤ δ_near)"); ax[2].set_title("C. near-optimal selection rate", fontsize=9)
    cmap = plt.get_cmap("viridis")
    ylab, yi = [], 0
    for m in ("iMF", "CD", "PENGUIN"):
        for n in NCAL:
            fr = [x for x in sel if x["model"] == m and x["B"] == "32" and x["method"] == "UCB" and int(x["n_cal"]) == n]
            left = 0.0
            for j, x in enumerate(fr):
                w = float(x["frequency"])
                ax[3].barh(yi, w, left=left, color=cmap(j / 5), edgecolor="white", lw=0.5,
                           label=f"({x['K']},{x['S']})" if yi == 0 else None)
                left += w
            ylab.append((yi, f"{m} n={n}")); yi += 1
        yi += 0.5
    ax[3].set_yticks([y for y, _ in ylab]); ax[3].set_yticklabels([s_ for _, s_ in ylab], fontsize=6.5); ax[3].invert_yaxis()
    ax[3].set_xlabel("FBC-UCB selection frequency, B = 32")
    ax[3].legend(fontsize=6.5, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12), title="(K, S)", title_fontsize=6.5, frameon=False)
    ax[3].set_title("D. allocation selection frequencies", fontsize=9)
    pols = [("pure_depth", "pure depth"), ("pure_width", "pure width"), ("historical", "historical static"),
            ("UCB_n25", "FBC-UCB n=25"), ("ERM_n25", "FBC-ERM n=25"), ("fullval", "full-val selector"), ("test_oracle", "test oracle")]
    wbar, allv = 0.11, []
    for j, (k, lab) in enumerate(pols):
        ys, lo, hi = [], [], []
        for m in ("iMF", "CD", "PENGUIN"):
            d = lt[f"{m}|B32"][k]
            v = d.get("expected_risk_macro", d.get("risk_macro")); c = d["ci"]
            ys.append(v); lo.append(max(0.0, v - c[0])); hi.append(max(0.0, c[1] - v)); allv += [c[0], c[1]]
        ax[4].bar(np.arange(3) + (j - 3) * wbar, ys, wbar, yerr=[lo, hi], capsize=1.5, color=plt.get_cmap("tab10")(j), label=lab)
    ax[4].set_xticks(range(3)); ax[4].set_xticklabels(["iMF", "CD", "PENGUIN"]); ax[4].set_ylim(0.95 * min(allv), 1.12 * max(allv))
    ax[4].set_ylabel("legacy VitalDB test HR MAE (bpm, patient-macro)"); ax[4].legend(fontsize=6.5, ncol=2)
    ax[4].set_title("E. legacy-test confirmation (not fresh), B = 32", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ART / "figure.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["audit", "grid", "fullval", "nested", "test", "latency", "figure"])
    st = ap.parse_args().stage
    {"audit": audit_stage, "grid": grid_stage, "fullval": fullval_stage, "nested": nested_stage, "test": test_stage,
     "latency": latency_stage, "figure": figure_stage}[st]()


if __name__ == "__main__":
    main()
