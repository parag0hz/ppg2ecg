"""M1 (pilot-gain prediction) — do pilot-sample statistics predict the gain from disjoint future samples?

Preregistration docs/M1_PILOT_GAIN_PREDICTION_PREREGISTRATION.md (7526201) + dated amendment
docs/M1_PILOT_GAIN_PREDICTION_AMENDMENT.md (deviations D1-D9). No generator training, no new functional.

Stages (in this order; caches under outputs/m1_pilot_gain_prediction/):
  tables-val   feature / target rows for VALIDATION (condition x window x partition)
  fit          patient-grouped CV on validation (row-level OOF MAE), freeze B0-B5, B4 (extended), leave-one-model-out
  sanity       informativeness gate of the ECG->HR functional on the M1 test subset (no predictor involved)
  tables-test  feature / target rows for TEST (only after `fit` and `sanity`)
  test         single evaluation of the frozen predictors; every statistic per partition, then averaged (D1);
               patient-clustered bootstrap (5,000); verdict
  mechanism    cross-fitted rho_{A_p} -> G_{B_p} diagnostic (uses the reference; descriptive)
  cost         cost accounting (NFE of the pilot / extended feature families)
  figure       main figure (reads test_metrics.json, crossfit_mechanism.json, per-partition predictions)
Stage-order guards: tables-test / test / mechanism require the frozen predictor and a passing informativeness gate;
fit refuses to run once test tables or test metrics exist; cached tables carry provenance hashes.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/m1_run.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/m1_pilot_gain_prediction"
ART = ROOT / "artifacts/m1_pilot_gain_prediction"
PREREG = "docs/M1_PILOT_GAIN_PREDICTION_PREREGISTRATION.md (7526201) + docs/M1_PILOT_GAIN_PREDICTION_AMENDMENT.md"
SEED, K, NPART, NBOOT, RBOOT, EPS, SNAP = 20260924, 16, 32, 5000, 64, 1.0, 3
N_TEST_PATIENTS, NWORK = 1156, 8
ARMS = ("I", "D", "C")
NAME = {"I": "iMF", "D": "CD", "C": "PENGUIN"}
STEPS = (1, 2, 4, 8)
COND = [(a, S) for a in ARMS for S in STEPS]           # frozen order iMF -> CD -> PENGUIN, S 1 -> 2 -> 4 -> 8
ARM_OF_COND = np.array([ARMS.index(a) for a, _ in COND])
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
FEATS = ["median_A", "MAD_A", "IQR_A", "SD_A", "range_A", "meanPair_A", "medPair_A", "bootMedInstab_A",
         "splitHalfInstab_A", "logS"]
LOG1P = list(range(1, 9))                              # frozen: spread features 2-9 get log1p
EXT_FEATS = ["median_D", "MAD_D", "medAbs_D", "absMedShift"]
SPECS = {"B3_full": list(range(10)), "B1_MAD_only": [1], "B2_SD_only": [3], "B5_logS_only": [9]}
TEST_BANK = {
    ("I", 1): "outputs/dw1_raw/hr_I1.npy", ("I", 2): "outputs/tt_expb_raw/hr_I_seed42_S2.npy",
    ("I", 4): "outputs/tt_expb_raw/hr_I_seed42_S4.npy", ("I", 8): "outputs/tt_expb_raw/hr_I_seed42_S8.npy",
    ("D", 1): "outputs/dw1_raw/hr_D1.npy", ("D", 2): "outputs/dw1_raw/hr_D2.npy",
    ("D", 4): "outputs/tt_expb_raw/hr_D_seed42_S4.npy", ("D", 8): "outputs/tt_expb_raw/hr_D_seed42_S8.npy",
    ("C", 1): "outputs/dw1_raw/hr_C1.npy", ("C", 2): "outputs/dw1_raw/hr_C2.npy",
    ("C", 4): "outputs/tt_expb_raw/hr_C_seed42_S4.npy", ("C", 8): "outputs/tt_expb_raw/hr_C_seed42_S8.npy",
}
PI, PJ = np.triu_indices(8, 1)


# ============================================================================ data
def snap(x):
    """D7: every HR value rounded to 1e-3 bpm before any feature or target."""
    return np.round(np.asarray(x, np.float64), SNAP)


def load_split(split):
    if split == "val":
        z = np.load(OUT / "val_ref.npz")
        ref, pid = z["ref_hr"], z["pid"]
        banks = {c: np.load(OUT / f"val_hr_{c[0]}_S{c[1]}.npy")[:K] for c in COND}
    elif split == "test":
        z = np.load(ROOT / "outputs/sr1_eval/arm_I_seed42.npz", allow_pickle=True)
        ref, pid = z["ref_hr"], z["pid"]
        banks = {c: np.load(ROOT / TEST_BANK[c])[:K] for c in COND}
    else:
        raise ValueError(split)
    return {c: snap(v) for c, v in banks.items()}, snap(ref), np.asarray(pid)


def sha(path):
    import hashlib
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def require_frozen_and_gate():
    """stage-order guard (amendment D4 / §5): frozen predictor exists and the informativeness gate passed."""
    assert (ART / "frozen_predictor.json").exists(), "fit stage has not frozen the predictor"
    san = ART / "sanity_informativeness.json"
    assert san.exists(), "sanity stage has not run"
    assert not json.loads(san.read_text())["UNINTERPRETABLE"], "informativeness gate failed: M1 is UNINTERPRETABLE; stop"


def partitions():
    m = json.loads((ART / "split_manifest.json").read_text())
    A = [np.array(a) for a in m["A"]]; B = [np.array(b) for b in m["B"]]
    assert len(A) == NPART and all(len(a) == len(b) == 8 and not set(a) & set(b) and set(a) | set(b) == set(range(K)) for a, b in zip(A, B))
    assert len({tuple(sorted(a)) for a in A}) == NPART
    return A, B


# ============================================================================ features / targets
def _boot_instab(args):
    a, ci, pi, win = args
    out = np.empty(a.shape[1])
    for j in range(a.shape[1]):
        rng = np.random.default_rng([SEED, ci, int(win[j]), pi])
        idx = rng.integers(0, 8, size=(RBOOT, 8))
        out[j] = np.std(np.median(a[:, j][idx], axis=1), ddof=1)
    return out


def pilot_features(a, S):
    """a: (8, n) pilot functionals in the partition's stored order -> (n, 10) raw features (column 7 filled by caller)."""
    n = a.shape[1]
    med = np.median(a, 0)
    F = np.empty((n, len(FEATS)))
    F[:, 0] = med
    F[:, 1] = np.median(np.abs(a - med), 0)
    F[:, 2] = np.percentile(a, 75, axis=0) - np.percentile(a, 25, axis=0)
    F[:, 3] = np.std(a, 0, ddof=1)
    F[:, 4] = a.max(0) - a.min(0)
    d = np.abs(a[PI] - a[PJ])
    F[:, 5] = d.mean(0)
    F[:, 6] = np.median(d, 0)
    F[:, 7] = np.nan
    F[:, 8] = np.abs(np.median(a[:4], 0) - np.median(a[4:], 0))
    F[:, 9] = np.log(S)
    return F


def depth_probe(a_S, a_2S):
    D = a_2S - a_S
    mD = np.median(D, 0)
    return np.stack([mD, np.median(np.abs(D - mD), 0), np.median(np.abs(D), 0), np.abs(np.median(a_2S, 0) - np.median(a_S, 0))], 1)


def build(split):
    cache = OUT / f"table_{split}.npz"
    prov = {"manifest_sha256": sha(ART / "split_manifest.json"), "snap": SNAP}
    if split == "test":
        require_frozen_and_gate()
        prov["frozen_predictor_sha256"] = sha(ART / "frozen_predictor.json")
        prov["sanity_sha256"] = sha(ART / "sanity_informativeness.json")
    if cache.exists():
        d = dict(np.load(cache))
        got = json.loads(str(d.pop("provenance")))
        assert got == prov, f"stale {cache.name}: {got} != {prov}"
        return d
    banks, ref, pid = load_split(split)
    A, B = partitions()
    rows = {k: [] for k in ("F", "X4", "G", "Grel", "Dwidth", "G_A", "cond", "win", "part", "pid")}
    deploy = {}
    ex = ProcessPoolExecutor(14)
    for ci, c in enumerate(COND):
        Y = banks[c]
        fin = np.isfinite(Y)
        ok = fin.all(0) & np.isfinite(ref)                                   # Omega_HR (frozen)
        idx = np.flatnonzero(ok)
        Yo, ro = Y[:, idx], ref[idx]
        S = c[1]
        c2 = (c[0], 2 * S)
        ok2 = np.isfinite(banks[c2]).all(0)[idx] if c2 in banks else np.zeros(len(idx), bool)
        # deployable view (sensitivity b): windows with A and reference finite but >= 1 future draw non-finite
        deploy[f"{NAME[c[0]]}_S{S}"] = float(np.mean([((fin[A[p]].all(0) & np.isfinite(ref)).sum() - ok.sum()) /
                                                      max((fin[A[p]].all(0) & np.isfinite(ref)).sum(), 1) for p in range(NPART)]))
        t0 = time.time()
        futs = [ex.submit(_boot_instab, (Yo[A[p]], ci, p, idx)) for p in range(NPART)]
        for p in range(NPART):
            a, b = Yo[A[p]], Yo[B[p]]
            i_b = np.abs(b - ro).mean(0)
            g_b = i_b - np.abs(np.median(b, 0) - ro)
            g_a = np.abs(a - ro).mean(0) - np.abs(np.median(a, 0) - ro)
            e_a = np.abs(np.median(a, 0) - ro)
            e_ab = np.abs(np.median(np.concatenate([a, b]), 0) - ro)
            F = pilot_features(a, S)
            F[:, 7] = futs[p].result()
            X4 = np.full((len(idx), 4), np.nan)
            if ok2.any():
                X4[ok2] = depth_probe(a[:, ok2], banks[c2][:, idx][A[p]][:, ok2])
            for k, v in (("F", F), ("X4", X4), ("G", g_b), ("Grel", g_b / np.maximum(i_b, EPS)), ("Dwidth", e_a - e_ab),
                         ("G_A", g_a), ("cond", np.full(len(idx), ci)), ("win", idx), ("part", np.full(len(idx), p)),
                         ("pid", pid[idx])):
                rows[k].append(v)
        print(f"  [{split}] {NAME[c[0]]} S={S}: {len(idx)} windows x {NPART} partitions ({time.time() - t0:.0f}s)", flush=True)
    ex.shutdown()
    d = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez(cache, provenance=json.dumps(prov), **d)
    (OUT / f"deployable_{split}.json").write_text(json.dumps(deploy, indent=1))
    return d


def transform(F):
    X = np.array(F, np.float64, copy=True)
    X[:, LOG1P] = np.log1p(np.clip(X[:, LOG1P], 0, None))
    return X


# ============================================================================ weighted ranks (bootstrap by multiplicity)
class RankPrep:
    """Average ranks of v within segments, for any non-negative integer weights w: identical to scipy rankdata on the
    array in which unit i is repeated w_i times (ties -> average ranks), without materialising it."""

    def __init__(self, v, seg=None):
        v = np.asarray(v, np.float64)
        self.n = len(v)
        self.seg = np.zeros(self.n, np.int64) if seg is None else np.asarray(seg, np.int64)
        self.order = np.lexsort((v, self.seg))
        vs, ss = v[self.order], self.seg[self.order]
        new = np.ones(self.n, bool)
        new[1:] = (vs[1:] != vs[:-1]) | (ss[1:] != ss[:-1])
        self.g = np.cumsum(new) - 1
        self.ng = int(self.g[-1]) + 1
        self.seg_of_g = ss[new]
        self.nseg = int(self.seg.max()) + 1
        self.vs = vs

    def ranks(self, w):
        ws = np.asarray(w, np.float64)[self.order]
        Wg = np.bincount(self.g, ws, self.ng)
        segtot = np.bincount(self.seg_of_g, Wg, self.nseg)
        segstart = np.cumsum(segtot) - segtot
        before = np.cumsum(Wg) - Wg - segstart[self.seg_of_g]
        r = np.empty(self.n)
        r[self.order] = (before + (Wg + 1) / 2)[self.g]
        return r


def wcorr_seg(x, y, w, seg, nseg):
    """weighted Pearson per segment -> (nseg,)"""
    sw = np.bincount(seg, w, nseg)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = np.bincount(seg, w * x, nseg) / sw; my = np.bincount(seg, w * y, nseg) / sw
        dx, dy = x - mx[seg], y - my[seg]
        cov = np.bincount(seg, w * dx * dy, nseg)
        return cov / np.sqrt(np.bincount(seg, w * dx * dx, nseg) * np.bincount(seg, w * dy * dy, nseg))


def wspearman(px, py, w):
    return wcorr_seg(px.ranks(w), py.ranks(w), w, px.seg, px.nseg)


def wquantile(prep, w, q):
    """inverted-CDF weighted quantile of the pooled values of `prep` (single segment)."""
    ws = np.asarray(w, np.float64)[prep.order]
    cw = np.cumsum(ws)
    return prep.vs[min(int(np.searchsorted(cw, q * cw[-1], side="left")), prep.n - 1)]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.ones(len(a))
    return float(wspearman(RankPrep(a), RankPrep(b), w)[0])


# ============================================================================ fitting
def make_pipe(alpha):
    return Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=alpha))])


def per_partition_spearman(pred, obs, part, cond):
    """validation OOF diagnostics: mean over partitions of pooled and within-condition Spearman."""
    pooled, within = [], []
    for p in range(NPART):
        m = part == p
        w = np.ones(m.sum())
        pooled.append(wspearman(RankPrep(pred[m]), RankPrep(obs[m]), w)[0])
        within.append(np.nanmean(wspearman(RankPrep(pred[m], cond[m]), RankPrep(obs[m], cond[m]), w)))
    return float(np.mean(pooled)), float(np.mean(within))


def cv_select(X, y, groups, part, cond, name, log_rows):
    """alpha by GroupKFold(5) over patients; criterion = row-level OOF MAE of G_B (frozen §6, D2)."""
    folds = list(GroupKFold(n_splits=5).split(X, y, groups))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])
    best, oof_best = None, None
    for al in ALPHAS:
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            oof[te] = make_pipe(al).fit(X[tr], y[tr]).predict(X[te])
        mae = float(np.mean(np.abs(oof - y)))
        log_rows.append({"model": name, "alpha": al, "cv_mae_rows": mae, "n_rows": len(y)})
        if best is None or mae < best[1] - 1e-12:
            best, oof_best = (al, mae), oof
    pipe = make_pipe(best[0]).fit(X, y)
    sp_pool, sp_within = per_partition_spearman(oof_best, y, part, cond)
    return pipe, best[0], best[1], sp_pool, sp_within


def dump(pipe, cols, alpha, mae, sp_pool, sp_within):
    sc, rg = pipe.named_steps["sc"], pipe.named_steps["rg"]
    return {"cols": [int(c) for c in cols], "alpha": alpha, "cv_mae_rows": mae, "oof_spearman_pooled_mean_partition": sp_pool,
            "oof_spearman_within_condition_mean_partition": sp_within, "scaler_mean": sc.mean_.tolist(),
            "scaler_scale": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_)}


def predict_frozen(entry, X):
    Z = (X[:, entry["cols"]] - np.array(entry["scaler_mean"])) / np.array(entry["scaler_scale"])
    return Z @ np.array(entry["coef"]) + entry["intercept"]


def fit_stage():
    assert not (OUT / "table_test.npz").exists() and not (ART / "test_metrics.json").exists(), "no refit after test tables / metrics exist"
    d = build("val")
    X, y, g, part, cond = transform(d["F"]), d["G"], d["pid"], d["part"], d["cond"]
    log, fr = [], {}
    for name, cols in SPECS.items():
        pipe, al, mae, sp, spw = cv_select(X[:, cols], y, g, part, cond, name, log)
        fr[name] = dump(pipe, cols, al, mae, sp, spw)
        print(f"  [fit] {name}: alpha={al} cv_mae_rows={mae:.4f} oof_spearman pooled={sp:.3f} within={spw:.3f}", flush=True)
    oof0 = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        oof0[te] = y[tr].mean()
    fr["B0_constant"] = {"value": float(np.mean(y)), "cv_mae_rows": float(np.mean(np.abs(oof0 - y)))}
    log.append({"model": "B0_constant", "alpha": "", "cv_mae_rows": fr["B0_constant"]["cv_mae_rows"], "n_rows": len(y)})
    m = np.isfinite(d["X4"]).all(1)
    XE = np.concatenate([X, d["X4"]], 1)
    for name, XX, cols in (("B4_extended", XE, list(range(14))), ("B3_full_on_extended_rows", X, list(range(10)))):
        pipe, al, mae, sp, spw = cv_select(XX[m][:, cols], y[m], g[m], part[m], cond[m], name, log)
        fr[name] = dump(pipe, cols, al, mae, sp, spw)
        print(f"  [fit] {name}: alpha={al} cv_mae_rows={mae:.4f} oof pooled={sp:.3f} within={spw:.3f}", flush=True)
    for arm in ARMS:
        keep = np.isin(cond, [i for i, c in enumerate(COND) if c[0] != arm])
        name = f"LOMO_without_{arm}"
        pipe, al, mae, sp, spw = cv_select(X[keep], y[keep], g[keep], part[keep], cond[keep], name, log)
        fr[name] = dump(pipe, list(range(10)), al, mae, sp, spw)
        print(f"  [fit] {name}: alpha={al} cv_mae_rows={mae:.4f}", flush=True)
    fields = ["model", "alpha", "cv_mae_rows", "n_rows"]
    with open(ART / "validation_cv.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(log)
    with open(ART / "extended_validation_cv.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        w.writerows([r for r in log if r["model"] in ("B4_extended", "B3_full_on_extended_rows")])
    frozen = {"prereg": PREREG, "target": "G_B", "features": FEATS, "log1p_indices": LOG1P, "extended_features": EXT_FEATS,
              "alpha_grid": list(ALPHAS), "cv": "GroupKFold(5) by patient on validation; criterion row-level OOF MAE of G_B",
              "hr_snap_decimals": SNAP, "n_val_rows": int(len(y)), "n_val_patients": int(len(np.unique(g))), "models": fr}
    (ART / "frozen_predictor.json").write_text(json.dumps(frozen, indent=1))
    (ART / "feature_schema.json").write_text(json.dumps({
        "primary": {f: ("log1p then StandardScaler" if i in LOG1P else "StandardScaler") for i, f in enumerate(FEATS)},
        "extended_appended": {f: "StandardScaler" for f in EXT_FEATS},
        "inputs": "generated HR functional values of the pilot draws A only (snapped to 1e-3 bpm); no reference, error, patient or model identity",
        "bootMedInstab": "SD (ddof 1) of medians of 64 bootstrap resamples of A; rng default_rng([20260924, cond_idx, window_row_index, part_idx])",
        "splitHalfInstab": "|median(A[0:4]) - median(A[4:8])| in the stored partition order",
        "baselines": {k: [FEATS[c] for c in v] for k, v in SPECS.items()}}, indent=1))



# ============================================================================ bootstrap draws
def boot_weights(pid):
    """patient-multiplicity weights for NBOOT replicates, mapping by pid (independent of load order)."""
    up, inv = np.unique(pid, return_inverse=True)
    rng = np.random.default_rng(SEED)
    return up, inv, [np.bincount(rng.integers(0, len(up), len(up)), minlength=len(up)).astype(np.float64) for _ in range(NBOOT)]


def ci(vals):
    v = np.asarray(vals, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


# ============================================================================ sanity (informativeness gate, D4)
def sanity_stage():
    assert (ART / "frozen_predictor.json").exists(), "fit stage has not frozen the predictor"
    banks, ref, pid = load_split("test")
    const = float(np.median(snap(np.load(OUT / "val_ref.npz")["ref_hr"])))
    cons, y, uc, up, uw = [], [], [], [], []
    for ci_, c in enumerate(COND):
        Y = banks[c]; ok = np.isfinite(Y).all(0) & np.isfinite(ref); idx = np.flatnonzero(ok)
        cons.append(np.median(Y[:, idx], 0)); y.append(ref[idx]); uc.append(np.full(len(idx), ci_)); up.append(pid[idx]); uw.append(idx)
    cons, y, uc, up, uw = map(np.concatenate, (cons, y, uc, up, uw))
    assert len(np.unique(up)) == N_TEST_PATIENTS
    e_c, e_k = np.abs(cons - y), np.abs(const - y)
    ups, inv, W = boot_weights(up)
    pc, py = RankPrep(cons), RankPrep(y)
    pcs, pys = RankPrep(cons, uc), RankPrep(y, uc)
    one = np.ones(len(y))

    def st(w):
        sw = np.bincount(uc, w, 12)
        return {"diff": float((w * (e_c - e_k)).sum() / w.sum()), "spearman": float(wspearman(pc, py, w)[0]),
                "diff_cond": np.bincount(uc, w * (e_c - e_k), 12) / sw, "spearman_cond": wspearman(pcs, pys, w)}
    pt = st(one)
    reps = [st(Wb[inv]) for Wb in W]
    res = {"constant_median_val_ref_hr_all_4822": const, "n_units": int(len(y)),
           "pooled": {"mae_consensus": float(e_c.mean()), "mae_constant": float(e_k.mean()), "diff": pt["diff"],
                      "diff_ci": ci([r["diff"] for r in reps]), "spearman": pt["spearman"], "spearman_ci": ci([r["spearman"] for r in reps])},
           "per_condition": {}}
    res["pooled"]["gate_A_pass"] = res["pooled"]["diff_ci"][1] < 0
    res["pooled"]["gate_B_pass"] = res["pooled"]["spearman_ci"][0] > 0
    for ci_, c in enumerate(COND):
        m = uc == ci_
        dci = ci([r["diff_cond"][ci_] for r in reps]); sci = ci([r["spearman_cond"][ci_] for r in reps])
        res["per_condition"][f"{NAME[c[0]]}_S{c[1]}"] = {"n": int(m.sum()), "mae_consensus": float(e_c[m].mean()), "mae_constant": float(e_k[m].mean()),
                                                          "diff": float(pt["diff_cond"][ci_]), "diff_ci": dci, "gate_A_pass": dci[1] < 0,
                                                          "spearman": float(pt["spearman_cond"][ci_]), "spearman_ci": sci, "gate_B_pass": sci[0] > 0}
    sys.path.insert(0, str(ROOT / "scripts"))
    import v1_evaluate as V
    import vm1_evaluate as VM
    X, _, pid2 = VM.load("test")
    assert np.array_equal(pid2, pid)
    hp = snap(np.array([V.ppg_hr(x.astype(np.float64)) for x in X]))[uw]           # on the same M1 units
    okp = np.isfinite(hp)
    e_p = np.abs(hp - y)
    dp = [float((Wb[inv] * okp * (e_c - np.nan_to_num(e_p))).sum() / (Wb[inv] * okp).sum()) for Wb in W]
    res["ppg_peak_baseline_reported_not_gating"] = {
        "units_with_finite_ppg_hr_share": float(okp.mean()), "mae_ppg": float(e_p[okp].mean()), "mae_consensus_same_units": float(e_c[okp].mean()),
        "diff_consensus_minus_ppg": float((e_c[okp] - e_p[okp]).mean()), "diff_ci": ci(dp),
        "per_condition_diff": {f"{NAME[c[0]]}_S{c[1]}": float((e_c - e_p)[(uc == i) & okp].mean()) for i, c in enumerate(COND)}}
    res["UNINTERPRETABLE"] = not (res["pooled"]["gate_A_pass"] and res["pooled"]["gate_B_pass"])
    res["conditions_failing_gate_A"] = [k for k, v in res["per_condition"].items() if not v["gate_A_pass"]]
    (ART / "sanity_informativeness.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["pooled"], indent=1), "\nfailing gate A:", res["conditions_failing_gate_A"], "\nUNINTERPRETABLE:", res["UNINTERPRETABLE"])




# ============================================================================ test
_G = {}
RK_POOLED = ("B3_full", "B1_MAD_only", "B2_SD_only", "B5_logS_only")
RK_WITHIN = ("B3_full", "B1_MAD_only", "B2_SD_only", "LOMO")


def _prepare_test():
    require_frozen_and_gate()
    fr = json.loads((ART / "frozen_predictor.json").read_text())["models"]
    d = build("test")
    X = transform(d["F"])
    P = {k: predict_frozen(fr[k], X) for k in SPECS}
    P["B0_constant"] = np.full(len(X), fr["B0_constant"]["value"])
    cond, part, pid = d["cond"], d["part"], d["pid"]
    arm = ARM_OF_COND[cond]
    lomo = np.empty(len(X))
    for ai, a in enumerate(ARMS):
        m = arm == ai
        lomo[m] = predict_frozen(fr[f"LOMO_without_{a}"], X[m])
    P["LOMO"] = lomo
    ext = np.isfinite(d["X4"]).all(1)
    XE = np.concatenate([X, np.nan_to_num(d["X4"])], 1)
    P["B4_extended"] = np.where(ext, predict_frozen(fr["B4_extended"], XE), np.nan)
    P["B3_ext_rows"] = np.where(ext, predict_frozen(fr["B3_full_on_extended_rows"], X), np.nan)
    san = json.loads((ART / "sanity_informativeness.json").read_text())
    failA = [i for i, c in enumerate(COND) if f"{NAME[c[0]]}_S{c[1]}" in san["conditions_failing_gate_A"]]
    win = d["win"]
    counts = np.zeros(win.max() + 1, int)
    for ci_ in range(12):
        counts[np.unique(win[cond == ci_])] += 1
    common = counts[win] == 12                                           # sensitivity (a): windows in all 12 conditions
    parts = []
    for p in range(NPART):
        m = np.flatnonzero(part == p)
        c, e, am = cond[m], ext[m], arm[m]
        pp = {"idx": m, "cond": c, "arm": am, "pid": pid[m], "obs": d["G"][m], "grel": d["Grel"][m], "dwid": d["Dwidth"][m],
              "pred": {k: v[m] for k, v in P.items()}, "common": common[m].astype(float), "notfailA": (~np.isin(c, failA)).astype(float)}
        pp["r_obs"] = RankPrep(pp["obs"]); pp["r_obs_c"] = RankPrep(pp["obs"], c); pp["r_obs_m"] = RankPrep(pp["obs"], am)
        pp["r"] = {k: RankPrep(pp["pred"][k]) for k in RK_POOLED}
        pp["r_c"] = {k: RankPrep(pp["pred"][k], c) for k in RK_WITHIN}
        pp["r_m"] = {k: RankPrep(pp["pred"][k], am) for k in ("B3_full", "LOMO")}
        pp["r_grel"], pp["r_dwid"] = RankPrep(pp["grel"]), RankPrep(pp["dwid"])
        ee = np.flatnonzero(e)
        pp["ext_idx"], pp["ext_cond"], pp["ext_arm"] = ee, c[ee], am[ee]
        pp["ext_noCD"] = (am[ee] != 1).astype(float)
        pp["r_ext_c"] = {k: RankPrep(pp["pred"][k][ee], c[ee]) for k in ("B4_extended", "B3_ext_rows")}
        pp["r_ext_p"] = {k: RankPrep(pp["pred"][k][ee]) for k in ("B4_extended", "B3_ext_rows")}
        pp["r_ext_m"] = {k: RankPrep(pp["pred"][k][ee], am[ee]) for k in ("B4_extended", "B3_ext_rows")}
        pp["r_ext_obs_c"], pp["r_ext_obs_p"], pp["r_ext_obs_m"] = RankPrep(pp["obs"][ee], c[ee]), RankPrep(pp["obs"][ee]), RankPrep(pp["obs"][ee], am[ee])
        parts.append(pp)
    up, inv_all, W = boot_weights(pid)
    assert len(up) == N_TEST_PATIENTS, len(up)
    return {"parts": parts, "up": up, "inv": inv_all, "W": W, "d": d, "P": P, "fr": fr, "failA": failA}


def _pstats(pp, w_unit, full=True):
    """all statistics of one partition for unit weights w_unit (bootstrap multiplicity x optional mask)."""
    o, pr = pp["obs"], pp["pred"]
    s = {}
    sw = w_unit.sum()
    z0 = np.zeros(len(o), np.int64)
    ro, roc, rom = pp["r_obs"].ranks(w_unit), pp["r_obs_c"].ranks(w_unit), pp["r_obs_m"].ranks(w_unit)
    rk = {k: pp["r"][k].ranks(w_unit) for k in RK_POOLED}
    for k in RK_POOLED:
        s[f"spearman_{k}"] = float(wcorr_seg(rk[k], ro, w_unit, z0, 1)[0])
        s[f"mae_{k}"] = float((w_unit * np.abs(pr[k] - o)).sum() / sw)
    s["mae_B0_constant"] = float((w_unit * np.abs(pr["B0_constant"] - o)).sum() / sw)
    p3 = pr["B3_full"]
    mo, mp = (w_unit * o).sum() / sw, (w_unit * p3).sum() / sw
    s["pearson"] = float((w_unit * (o - mo) * (p3 - mp)).sum() / np.sqrt((w_unit * (o - mo) ** 2).sum() * (w_unit * (p3 - mp) ** 2).sum()))
    s["r2"] = float(1 - (w_unit * (o - p3) ** 2).sum() / (w_unit * (o - mo) ** 2).sum())
    qs = [wquantile(pp["r"]["B3_full"], w_unit, q) for q in (0.25, 0.5, 0.75)]
    top, bot = p3 >= qs[2], p3 <= qs[0]
    s["quartile_diff"] = float((w_unit * o * top).sum() / (w_unit * top).sum() - (w_unit * o * bot).sum() / (w_unit * bot).sum())
    s["n_top"], s["n_bottom"] = float((w_unit * top).sum()), float((w_unit * bot).sum())
    grp = np.searchsorted(np.array(qs), p3, side="left")                # quartile groups for the figure (point estimate)
    for q in range(4):
        mq = grp == q
        s[f"q{q + 1}_mean_obs"] = float((w_unit * o * mq).sum() / max((w_unit * mq).sum(), 1e-12))
    for k in RK_WITHIN:
        per = wcorr_seg(pp["r_c"][k].ranks(w_unit), roc, w_unit, pp["cond"], 12)
        s[f"within_{k}"] = float(np.nanmean(per))
        if k in ("B3_full", "LOMO"):
            for ai, a in enumerate(ARMS):
                s[f"within_{k}_{a}"] = float(np.nanmean(per[[i for i, c in enumerate(COND) if c[0] == a]]))
        if k == "B3_full":
            for ci_ in range(12):
                s[f"within_cond_{ci_}"] = float(per[ci_])
    for k in ("B3_full", "LOMO"):
        per = wcorr_seg(pp["r_m"][k].ranks(w_unit), rom, w_unit, pp["arm"], 3)
        for ai, a in enumerate(ARMS):
            s[f"model_{k}_{a}"] = float(per[ai])
    s["delta_pooled_B3_minus_B1"] = s["spearman_B3_full"] - s["spearman_B1_MAD_only"]
    s["delta_pooled_B3_minus_B2"] = s["spearman_B3_full"] - s["spearman_B2_SD_only"]
    s["delta_pooled_B3_minus_B5"] = s["spearman_B3_full"] - s["spearman_B5_logS_only"]
    s["delta_within_B3_minus_B1"] = s["within_B3_full"] - s["within_B1_MAD_only"]
    s["delta_within_B3_minus_B2"] = s["within_B3_full"] - s["within_B2_SD_only"]
    s["delta_mae_B3_minus_B0"] = s["mae_B3_full"] - s["mae_B0_constant"]
    if not full:
        return s
    s["spearman_grel"] = float(wcorr_seg(rk["B3_full"], pp["r_grel"].ranks(w_unit), w_unit, z0, 1)[0])
    s["spearman_dwidth_not_crossfitted"] = float(wcorr_seg(rk["B3_full"], pp["r_dwid"].ranks(w_unit), w_unit, z0, 1)[0])
    wm = w_unit * pp["common"]                                           # sensitivity (a)
    s["common_spearman_B3"] = float(wcorr_seg(pp["r"]["B3_full"].ranks(wm), pp["r_obs"].ranks(wm), wm, z0, 1)[0])
    s["common_within_B3"] = float(np.nanmean(wcorr_seg(pp["r_c"]["B3_full"].ranks(wm), pp["r_obs_c"].ranks(wm), wm, pp["cond"], 12)))
    ee = pp["ext_idx"]; we = w_unit[ee]; ce, ae = pp["ext_cond"], pp["ext_arm"]; z1 = np.zeros(len(ee), np.int64)
    wnc = we * pp["ext_noCD"]
    reo_c, reo_p, reo_m, reo_nc = (pp["r_ext_obs_c"].ranks(we), pp["r_ext_obs_p"].ranks(we), pp["r_ext_obs_m"].ranks(we),
                                   pp["r_ext_obs_c"].ranks(wnc))
    for k in ("B4_extended", "B3_ext_rows"):
        s[f"ext_within_{k}"] = float(np.nanmean(wcorr_seg(pp["r_ext_c"][k].ranks(we), reo_c, we, ce, 12)))
        s[f"ext_pooled_{k}"] = float(wcorr_seg(pp["r_ext_p"][k].ranks(we), reo_p, we, z1, 1)[0])
        s[f"ext_within_noCD_{k}"] = float(np.nanmean(wcorr_seg(pp["r_ext_c"][k].ranks(wnc), reo_nc, wnc, ce, 12)))
        per = wcorr_seg(pp["r_ext_m"][k].ranks(we), reo_m, we, ae, 3)
        for ai, a in enumerate(ARMS):
            s[f"ext_model_{k}_{a}"] = float(per[ai])
    s["ext_delta_within_B4_minus_B3"] = s["ext_within_B4_extended"] - s["ext_within_B3_ext_rows"]
    s["ext_delta_pooled_B4_minus_B3"] = s["ext_pooled_B4_extended"] - s["ext_pooled_B3_ext_rows"]
    s["ext_delta_within_noCD_B4_minus_B3"] = s["ext_within_noCD_B4_extended"] - s["ext_within_noCD_B3_ext_rows"]
    for a in ARMS:
        s[f"ext_delta_model_{a}"] = s[f"ext_model_B4_extended_{a}"] - s[f"ext_model_B3_ext_rows_{a}"]
    return s


def _replicate_stats(wpat):
    """mean over partitions of every statistic; plus the Gate-A-restricted set when some condition failed Gate A."""
    G = _G["ctx"]
    acc, accA = {}, {}
    for pp in G["parts"]:
        w = wpat[G["inv"][pp["idx"]]]
        for k, v in _pstats(pp, w).items():
            acc.setdefault(k, []).append(v)
        if G["failA"]:
            for k, v in _pstats(pp, w * pp["notfailA"], full=False).items():
                accA.setdefault(k, []).append(v)
    out = {k: float(np.nanmean(v)) for k, v in acc.items()}
    out.update({f"gateAok__{k}": float(np.nanmean(v)) for k, v in accA.items()})
    return out


def _boot_chunk(bs):
    G = _G["ctx"]
    return [_replicate_stats(G["W"][b]) for b in bs]


def verdict_of(point, bci, pre=""):
    """amendment §4.5 (D3, D5, D6); `pre` selects the Gate-A-restricted statistics."""
    g = lambda k: point[pre + k]  # noqa: E731
    c = lambda k: bci[pre + k]  # noqa: E731
    models_pos = sum(g(f"model_B3_full_{a}") > 0 for a in ARMS)
    fail = {"a_pooled_nonpositive": g("spearman_B3_full") <= 0, "b_quartiles_do_not_differ": g("quartile_diff") <= 0,
            "c_collapse_not_better_than_B0": c("delta_mae_B3_minus_B0")[1] >= 0, "d_zero_models_positive": models_pos == 0}
    mad_works = c("within_B1_MAD_only")[0] > 0
    downgrade = mad_works and c("delta_within_B3_minus_B1")[0] <= 0
    strong = {"pooled_ci_lower_gt_0": c("spearman_B3_full")[0] > 0, "quartile_ci_lower_gt_0": c("quartile_diff")[0] > 0,
              "ge2_models_positive": models_pos >= 2, "within_condition_ci_lower_gt_0": c("within_B3_full")[0] > 0,
              "no_MAD_only_downgrade": not downgrade}
    cat = "FAILED" if any(fail.values()) else ("STRONG" if all(strong.values()) else "PARTIAL")
    labels = []
    if cat == "PARTIAL":
        if (not strong["within_condition_ci_lower_gt_0"] and strong["pooled_ci_lower_gt_0"] and strong["quartile_ci_lower_gt_0"]
                and strong["ge2_models_positive"]):
            labels.append("between-condition only")
        if not strong["pooled_ci_lower_gt_0"]:
            labels.append("pooled direction positive, CI crosses 0")
        if not strong["ge2_models_positive"]:
            labels.append("only one model carries it")
        if downgrade:
            labels.append("MAD-only suffices")
        if not strong["quartile_ci_lower_gt_0"]:
            labels.append("quartile contrast CI includes 0")
    return {"category": cat, "labels": labels, "failed_routes": {k: bool(v) for k, v in fail.items()},
            "strong_items": {k: bool(v) for k, v in strong.items()}, "n_models_positive": int(models_pos), "MAD_only_works": bool(mad_works)}


def test_stage():
    ctx = _prepare_test()
    _G["ctx"] = ctx
    t0 = time.time()
    point = _replicate_stats(np.ones(len(ctx["up"])))
    print(f"  [test] point estimates ({time.time() - t0:.0f}s)", flush=True)
    chunks = [list(range(i, min(i + 25, NBOOT))) for i in range(0, NBOOT, 25)]
    boots = []
    with mp.get_context("fork").Pool(NWORK) as pool:
        for i, r in enumerate(pool.imap(_boot_chunk, chunks)):
            boots += r
            if i % 8 == 0:
                print(f"  [test] bootstrap {len(boots)}/{NBOOT} ({time.time() - t0:.0f}s)", flush=True)
    bci = {k: ci([r[k] for r in boots]) for k in point}
    share_pos = {k: float(np.mean([r[k] > 0 for r in boots])) for k in point}
    n_nonfinite = {k: int(np.sum(~np.isfinite([r[k] for r in boots]))) for k in point}
    verdict = verdict_of(point, bci)
    verdict_gateA = verdict_of(point, bci, "gateAok__") if ctx["failA"] else None
    # frozen-§8 average-then-metric statistics (reuse-contaminated; sensitivity d; point + CI)
    d, P = ctx["d"], ctx["P"]
    code = d["cond"].astype(np.int64) * 10_000_000 + d["win"]
    u, inv = np.unique(code, return_inverse=True)
    n = np.bincount(inv)
    ao, ap = np.bincount(inv, d["G"]) / n, np.bincount(inv, P["B3_full"]) / n
    upid = np.zeros(len(u), np.int64); upid[inv] = d["pid"]
    _, uinv = np.unique(upid, return_inverse=True)
    ro, rp = RankPrep(ao), RankPrep(ap)

    def avg_stats(wp):
        w = wp[uinv]
        q25, q75 = wquantile(rp, w, 0.25), wquantile(rp, w, 0.75)
        top, bot = ap >= q75, ap <= q25
        return (float(wspearman(rp, ro, w)[0]), float((w * ao * top).sum() / (w * top).sum() - (w * ao * bot).sum() / (w * bot).sum()))
    ap0 = avg_stats(np.ones(len(ctx["up"])))
    apb = [avg_stats(Wb) for Wb in ctx["W"]]
    contaminated = {"spearman": ap0[0], "spearman_ci": ci([a[0] for a in apb]), "quartile_diff": ap0[1], "quartile_ci": ci([a[1] for a in apb]),
                    "label": "frozen §8 average-then-metric: reuse-contaminated, non-gating (D1)"}
    pm = []                                                              # sensitivity (c): patient-macro, point only
    for pp in ctx["parts"]:
        key = pp["pid"].astype(np.int64) * 100 + pp["cond"]
        _, ii = np.unique(key, return_inverse=True)
        nn = np.bincount(ii)
        pm.append(spearman(np.bincount(ii, pp["pred"]["B3_full"]) / nn, np.bincount(ii, pp["obs"]) / nn))
    metrics = {"prereg": PREREG, "n_units_per_partition": int(len(ctx["parts"][0]["idx"])), "n_patients": int(len(ctx["up"])),
               "point_mean_over_partitions": point, "ci95": bci, "share_boot_positive": share_pos,
               "validation_oof_pooled": {k: ctx["fr"][k].get("oof_spearman_pooled_mean_partition") for k in SPECS},
               "validation_oof_within": {k: ctx["fr"][k].get("oof_spearman_within_condition_mean_partition") for k in SPECS},
               "sensitivity_reuse_contaminated_frozen_s8": contaminated, "sensitivity_patient_macro_spearman_point": float(np.mean(pm)),
               "sensitivity_deployable_view_share_future_nonfinite": json.loads((OUT / "deployable_test.json").read_text()),
               "conditions_failing_gate_A": [f"{NAME[COND[i][0]]}_S{COND[i][1]}" for i in ctx["failA"]],
               "verdict": verdict, "verdict_without_gateA_failing_conditions_nongating": verdict_gateA}
    (ART / "test_metrics.json").write_text(json.dumps(metrics, indent=1))
    (ART / "bootstrap.json").write_text(json.dumps({"replicates": NBOOT, "seed": SEED, "unit": "patient (multiplicity weights over all units)",
                                                    "statistic": "mean over 32 partitions, recomputed in every replicate", "ci95": bci,
                                                    "share_positive": share_pos, "n_nonfinite_replicates": n_nonfinite}, indent=1))
    (ART / "extended_test_metrics.json").write_text(json.dumps({k: {"point": point[k], "ci95": bci[k]} for k in point if k.startswith("ext_")} |
                                                               {"n_units_per_partition": int(len(ctx["parts"][0]["ext_idx"])),
                                                                "rows": "S in {1,2,4}; all 16 draws finite at S and 2S; B3 refit on the same rows",
                                                                "CD_pairing": "z0-only (re-noise injected at t=j/S vs j/2S); see noCD sensitivity"}, indent=1))
    with open(ART / "model_specific.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "spearman_pooled_over_depths", "ci_lo", "ci_hi", "within_condition_mean", "w_ci_lo", "w_ci_hi",
                    "lomo_spearman", "lomo_ci_lo", "lomo_ci_hi", "lomo_within", "lomo_w_ci_lo", "lomo_w_ci_hi"])
        for a in ARMS:
            w.writerow([NAME[a], point[f"model_B3_full_{a}"], *bci[f"model_B3_full_{a}"], point[f"within_B3_full_{a}"], *bci[f"within_B3_full_{a}"],
                        point[f"model_LOMO_{a}"], *bci[f"model_LOMO_{a}"], point[f"within_LOMO_{a}"], *bci[f"within_LOMO_{a}"]])
        for ci_, c in enumerate(COND):
            w.writerow([f"condition {NAME[c[0]]}_S{c[1]}", "", "", "", point[f"within_cond_{ci_}"], *bci[f"within_cond_{ci_}"], "", "", "", "", "", ""])
    np.savez(OUT / "test_predictions_full.npz", **{k: v for k, v in d.items() if k in ("cond", "win", "part", "pid", "G")},
             **{f"pred_{k}": v for k, v in P.items()})
    with open(ART / "test_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["condition", "pred_decile_within_partition", "n_rows_all_partitions", "mean_pred_G_B", "mean_obs_G_B"])
        dec = np.zeros(len(d["G"]), int)
        for p in range(NPART):
            m = d["part"] == p
            r = RankPrep(P["B3_full"][m]).ranks(np.ones(m.sum()))
            dec[m] = np.clip(((r - 1) * 10 // m.sum()).astype(int), 0, 9)
        for ci_, c in enumerate(COND):
            for q in range(10):
                mm = (d["cond"] == ci_) & (dec == q)
                if mm.any():
                    w.writerow([f"{NAME[c[0]]}_S{c[1]}", q + 1, int(mm.sum()), float(P["B3_full"][mm].mean()), float(d["G"][mm].mean())])
    print(json.dumps(verdict, indent=1))
    for k in ("spearman_B3_full", "quartile_diff", "within_B3_full", "spearman_B1_MAD_only", "within_B1_MAD_only", "delta_within_B3_minus_B1",
              "delta_mae_B3_minus_B0", "spearman_B5_logS_only", "delta_pooled_B3_minus_B5", "model_B3_full_I", "model_B3_full_D", "model_B3_full_C"):
        print(f"  {k}: {point[k]:+.4f} {bci[k]}")
    print("  contaminated (frozen §8):", contaminated)


# ============================================================================ mechanism (D9)
def mechanism_stage():
    require_frozen_and_gate()
    out = {}
    A, B = partitions()
    tri = np.triu_indices(8, 1)
    for split in ("test", "val"):
        banks, ref, pid = load_split(split)
        up, inv_all = np.unique(pid, return_inverse=True)
        P = len(up)
        stats = []
        for c in COND:
            Y = banks[c]; ok = np.isfinite(Y).all(0) & np.isfinite(ref); idx = np.flatnonzero(ok)
            e = Y[:, idx] - ref[idx]
            io = inv_all[idx]
            n_p = np.bincount(io, None, P).astype(float)
            s1 = np.stack([np.bincount(io, e[k], P) for k in range(K)], 1)
            s2 = np.zeros((P, K, K))
            for k in range(K):
                for l in range(k, K):
                    s2[:, k, l] = s2[:, l, k] = np.bincount(io, e[k] * e[l], P)
            gB = np.stack([np.bincount(io, np.abs(e[B[p]]).mean(0) - np.abs(np.median(e[B[p]], 0)), P) for p in range(NPART)], 1)
            gA = np.stack([np.bincount(io, np.abs(e[A[p]]).mean(0) - np.abs(np.median(e[A[p]], 0)), P) for p in range(NPART)], 1)
            stats.append((n_p, s1, s2, gB, gA))

        def points(w):
            rA = np.zeros((12, NPART)); rB = np.zeros((12, NPART)); gb = np.zeros((12, NPART)); ga = np.zeros((12, NPART))
            for ci_, (n_p, s1, s2, gB, gA) in enumerate(stats):
                n = w @ n_p; m1 = (w @ s1) / n; m2 = (w @ s2.reshape(P, -1)).reshape(K, K) / n
                cov = m2 - np.outer(m1, m1); sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None)); corr = cov / np.outer(sd, sd)
                for p in range(NPART):
                    rA[ci_, p] = corr[np.ix_(A[p], A[p])][tri].mean(); rB[ci_, p] = corr[np.ix_(B[p], B[p])][tri].mean()
                gb[ci_] = (w @ gB) / n; ga[ci_] = (w @ gA) / n
            return rA, rB, gb, ga

        ls = np.log([c[1] for c in COND])

        def partial(x, y):
            rx, ry, rz = rankdata(x), rankdata(y), rankdata(ls)                 # average ranks (log S has 3-way ties)
            res = lambda v: v - np.polyval(np.polyfit(rz, v, 1), rz)  # noqa: E731
            return float(np.corrcoef(res(rx), res(ry))[0, 1])

        def summary(rA, rB, gb, ga, perm=None):
            gbp, gap = (gb, ga) if perm is None else (gb[perm], ga[perm])
            s = {"pooled_AB": float(np.mean([spearman(rA[:, p], gbp[:, p]) for p in range(NPART)])),
                 "pooled_BA": float(np.mean([spearman(rB[:, p], gap[:, p]) for p in range(NPART)]))}
            if perm is None:
                s["pooled_AB_depth_adjusted"] = float(np.mean([partial(rA[:, p], gb[:, p]) for p in range(NPART)]))
                for ai, a in enumerate(ARMS):
                    sl = slice(4 * ai, 4 * ai + 4)
                    s[f"{NAME[a]}_AB"] = float(np.mean([spearman(rA[sl, p], gb[sl, p]) for p in range(NPART)]))
                    s[f"{NAME[a]}_BA"] = float(np.mean([spearman(rB[sl, p], ga[sl, p]) for p in range(NPART)]))
            return s
        rA, rB, gb, ga = points(np.ones(P))
        res = {"conditions": [f"{NAME[c[0]]}_S{c[1]}" for c in COND], "rho_A_mean": rA.mean(1).tolist(), "rho_B_mean": rB.mean(1).tolist(),
               "G_B_mean": gb.mean(1).tolist(), "G_A_mean": ga.mean(1).tolist(), "point": summary(rA, rB, gb, ga),
               "expected_sign": "negative", "n_patients": int(P),
               "note": "descriptive; re-analysis of draws already used in B3/B3-BOOT; per-partition cross-fit (D9); window-weighted rho and G"}
        if split == "test":
            rng = np.random.default_rng(SEED)
            reps = [summary(*points(np.bincount(rng.integers(0, P, P), minlength=P).astype(float))) for _ in range(NBOOT)]
            res["bootstrap_stability_ci95"] = {k: ci([r[k] for r in reps]) for k in res["point"]}
            prng = np.random.default_rng(SEED + 1)
            obs = res["point"]["pooled_AB"]
            null = np.array([summary(rA, rB, gb, ga, perm=prng.permutation(12))["pooled_AB"] for _ in range(100_000)])
            res["permutation_pooled_AB"] = {"n": 100_000, "p_one_sided_negative": float((1 + np.sum(null <= obs)) / (1 + len(null))),
                                            "p_two_sided": float((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))}
        out[split] = res
        print(split, json.dumps(res["point"]), flush=True)
    (ART / "crossfit_mechanism.json").write_text(json.dumps(out, indent=1))


# ============================================================================ cost accounting
def cost_stage():
    rows = {f"{NAME[a]}_S{S}": {"pilot_primary_nfe": 8 * S, "future_nfe": 8 * S,
                                 "pilot_extended_nfe": 8 * (S + 2 * S) if S <= 4 else None} for a, S in COND}
    (ART / "cost_accounting.json").write_text(json.dumps({"unit": "vector-field evaluations per window", "per_condition": rows,
                                                          "note": "primary features need only the K_pilot = 8 pilot samples at depth S; the extended "
                                                                  "depth-probe needs the same 8 seeds again at 2S; no adaptive policy is evaluated"}, indent=1))


# ============================================================================ figure
def figure_stage():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tm = json.loads((ART / "test_metrics.json").read_text()); pt, cb = tm["point_mean_over_partitions"], tm["ci95"]
    mech = json.loads((ART / "crossfit_mechanism.json").read_text())["test"]
    z = np.load(OUT / "test_predictions_full.npz")
    m0 = z["part"] == 0
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    ax[0].hexbin(z["pred_B3_full"][m0], z["G"][m0], gridsize=60, bins="log", cmap="Blues", mincnt=1)
    ax[0].set_xlabel("predicted held-out gain (pilot A, partition 0)"); ax[0].set_ylabel("observed held-out gain G_B (bpm)")
    ax[0].set_title(f"A. test, pooled 12 conditions: Spearman {pt['spearman_B3_full']:+.3f} [{cb['spearman_B3_full'][0]:+.3f}, "
                    f"{cb['spearman_B3_full'][1]:+.3f}]\n(mean over 32 partitions; display = partition 0)", fontsize=8.5)
    qm = [pt[f"q{q}_mean_obs"] for q in range(1, 5)]
    ax[1].bar(range(1, 5), qm, color=["#bbb", "#999", "#777", "#1f4e79"])
    ax[1].set_xticks(range(1, 5)); ax[1].set_xticklabels(["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    ax[1].set_ylabel("mean observed G_B (bpm)")
    ax[1].set_title(f"B. by predicted-gain quartile: Q4 − Q1 {pt['quartile_diff']:+.3f} [{cb['quartile_diff'][0]:+.3f}, {cb['quartile_diff'][1]:+.3f}]", fontsize=8.5)
    labs, vals, los, his = [], [], [], []
    for key, lab in [("spearman_B3_full", "pooled"), ("within_B3_full", "within-condition")] + \
                    [(f"model_B3_full_{a}", f"{NAME[a]} (depths pooled)") for a in ARMS] + [(f"within_B3_full_{a}", f"{NAME[a]} within") for a in ARMS] + \
                    [(f"model_LOMO_{a}", f"LOMO {NAME[a]}") for a in ARMS]:
        labs.append(lab); vals.append(pt[key]); los.append(pt[key] - cb[key][0]); his.append(cb[key][1] - pt[key])
    yy = np.arange(len(labs))
    ax[2].errorbar(vals, yy, xerr=[los, his], fmt="o", ms=4, capsize=2, color="#1f4e79")
    ax[2].axvline(0, color="#555", lw=0.8); ax[2].set_yticks(yy); ax[2].set_yticklabels(labs, fontsize=7); ax[2].invert_yaxis()
    ax[2].set_xlabel("Spearman(predicted, observed G_B), 95 % patient bootstrap"); ax[2].set_title("C. pooled / within / model-specific / LOMO", fontsize=8.5)
    ra, gbm = np.array(mech["rho_A_mean"]), np.array(mech["G_B_mean"])
    for ai, (a, col) in enumerate(zip(ARMS, ("#1f4e79", "#a33", "#2e7d32"))):
        sl = slice(4 * ai, 4 * ai + 4)
        ax[3].plot(ra[sl], gbm[sl], "o-", color=col, label=NAME[a])
        for S, x_, y_ in zip(STEPS, ra[sl], gbm[sl]):
            ax[3].annotate(f"S{S}", (x_, y_), textcoords="offset points", xytext=(3, 3), fontsize=6.5, color=col)
    mp_ = mech["point"]
    ax[3].set_xlabel("ρ_A (pilot draws, uses reference)"); ax[3].set_ylabel("G_B (held-out draws)")
    ax[3].set_title(f"D. cross-fit diagnostic: mean per-partition Spearman(ρ_A, G_B) {mp_['pooled_AB']:+.2f}", fontsize=8.5)
    ax[3].legend(fontsize=7, frameon=False)
    for a_ in ax:
        a_.grid(lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(ART / "figure.png", dpi=150)
    print("wrote", ART / "figure.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["tables-val", "fit", "sanity", "tables-test", "test", "mechanism", "cost", "figure"])
    st = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    {"tables-val": lambda: build("val"), "fit": fit_stage, "sanity": sanity_stage, "tables-test": lambda: build("test"),
     "test": test_stage, "mechanism": mechanism_stage, "cost": cost_stage, "figure": figure_stage}[st]()
