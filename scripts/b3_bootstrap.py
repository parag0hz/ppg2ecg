"""B3-BOOT — patient-clustered bootstrap of the functional-error-dependence mechanism
(docs/B3_CORRELATION_BOOTSTRAP_PREREGISTRATION.md, frozen at 590d191). No training, no sampling: every quantity is
recomputed from arrays already on disk, and the full-sample point estimates are first checked against EXP-B's CSVs.

Run: .venv/bin/python scripts/b3_bootstrap.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dw2_mechanism_wave as DWV  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

OUT, RAWOUT = ROOT / "artifacts/b3_bootstrap", ROOT / "outputs/b3_bootstrap_raw"
EXPB_RAW = ROOT / "outputs/tt_expb_raw"
FS, K, N_SUB = 128, 16, 2000
N_BOOT, BOOT_SEED, CHUNK = 5000, 20260923, 250
ARMS = ("I", "D", "C")
NAME = {"I": "iMF", "D": "consistency distillation", "C": "PENGUIN (Euler)"}
DEPTHS = (1, 2, 4, 8)
CONDS = [(a, s) for a in ARMS for s in DEPTHS]
K2_RUNS = [("D", 1), ("I", 2), ("C", 4)]
K2_KS = (1, 2, 3, 4, 8, 16, 32)


# ------------------------------------------------------------------ data (identical resolution to EXP-B's Store.rows)
def seeded_rows(arm, seed, S):
    cands = [ROOT / f"outputs/dw1_raw/hr_{arm}{S}.npy"] if seed == 42 else []
    cands += list((ROOT / "outputs/dw2_raw").glob(f"hr_{arm}_seed{seed}_K*_S{S}.npy"))
    rows = [np.load(p) for p in cands if p.exists()]
    return max(rows, key=len) if rows else None


def hr_matrix(arm, seed, S, k):
    f = EXPB_RAW / f"hr_{arm}_seed{seed}_S{S}.npy"
    H = np.load(f) if f.exists() else seeded_rows(arm, seed, S)
    assert H is not None and H.shape[0] >= k, (arm, seed, S, None if H is None else H.shape)
    return H[:k].astype(np.float64)


def reference():
    d = np.load(ROOT / "outputs/rd1_detector/test_rpeaks.npz")
    peaks = np.split(d["idx"], d["off"][1:-1])
    ref = np.array([R.hr_bpm(p, FS) for p in peaks])
    pid = np.load(ROOT / "outputs/sr1_eval/arm_I_seed42.npz")["pid"]
    assert len(ref) == len(pid) == 19543
    return ref, pid


# ------------------------------------------------------------------ patient-level aggregation helpers
class Patients:
    def __init__(self, pid):
        self.subs, self.idx = np.unique(pid, return_inverse=True)
        self.P = len(self.subs)

    def per_patient_mean(self, v, idx=None):
        """nanmean of v within each patient (NaN if the patient has no finite value) — cluster_ci's first step."""
        idx = self.idx if idx is None else idx
        f = np.isfinite(v)
        s = np.bincount(idx[f], weights=v[f], minlength=self.P)
        n = np.bincount(idx[f], minlength=self.P)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / n, np.nan)


def macro(per, W):
    """Patient-macro mean for each bootstrap weight row: sum_p w_p per_p / sum_p w_p over patients with finite per_p."""
    f = np.isfinite(per)
    return (W[:, f] @ per[f]) / W[:, f].sum(1)


def corr_stats(E, idx, P):
    """Per-patient sufficient statistics of the centred signed errors E [K, n]: counts, first and second moments."""
    Ec = E - E.mean(1, keepdims=True)                       # shift invariance: improves numerical precision only
    n = np.bincount(idx, minlength=P).astype(np.float64)
    s = np.stack([np.bincount(idx, weights=Ec[k], minlength=P) for k in range(E.shape[0])], 1)          # [P, K]
    q = np.zeros((P, E.shape[0], E.shape[0]))
    for k in range(E.shape[0]):
        for l in range(k, E.shape[0]):
            q[:, k, l] = q[:, l, k] = np.bincount(idx, weights=Ec[k] * Ec[l], minlength=P)
    return n, s, q.reshape(P, -1)


def rho_bar(W, n, s, q, k=K, fisher=False):
    """Mean over pairs k<l of Pearson(e_k, e_l) on the weighted window set, for each weight row of W."""
    nt = W @ n
    mu = (W @ s) / nt[:, None]                                 # [B, K]
    cov = (W @ q).reshape(-1, k, k) / nt[:, None, None] - mu[:, :, None] * mu[:, None, :]
    sd = np.sqrt(np.einsum("bkk->bk", cov))
    r = cov / (sd[:, :, None] * sd[:, None, :])
    iu = np.triu_indices(k, 1)
    rr = r[:, iu[0], iu[1]]
    return np.tanh(np.arctanh(np.clip(rr, -0.999999, 0.999999)).mean(1)) if fisher else rr.mean(1)


def spearman_rows(A, B):
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def summ(x):
    x = np.asarray(x, float)
    return {"p2.5": float(np.percentile(x, 2.5)), "p50": float(np.percentile(x, 50)), "p97.5": float(np.percentile(x, 97.5))}


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True); RAWOUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ref, pid = reference()
    ok = np.isfinite(ref)
    pts = Patients(pid)
    sub = np.unique(np.linspace(0, len(ref) - 1, N_SUB).round().astype(int))
    sub_idx = pts.idx[sub]

    per, stats, point = {}, {}, {}
    for arm, S in CONDS:
        H = hr_matrix(arm, 42, S, K)
        with np.errstate(all="ignore"):
            m = np.nanmedian(H, 0)
            C = np.abs(m - ref); I = np.nanmean(np.abs(H - ref), 0)
            MAD = np.nanmedian(np.abs(H - m), 0); SD = np.nanstd(H, 0, ddof=1)
        G = I - C
        msk = lambda v: np.where(ok, v, np.nan)  # noqa: E731
        per[(arm, S)] = {q: pts.per_patient_mean(msk(v)) for q, v in (("C", C), ("I", I), ("G", G), ("MAD", MAD), ("SD", SD))}
        full = ok & np.isfinite(H).all(0)                   # EXP-B3's window set, unchanged
        E = (H - ref)[:, full]
        stats[(arm, S)] = corr_stats(E, pts.idx[full], pts.P)
        point[(arm, S)] = {"rho_bar_direct": float(np.corrcoef(E)[np.triu_indices(K, 1)].mean()), "n_full": int(full.sum())}
        Wf = np.load(EXPB_RAW / f"wave_{arm}_seed42_S{S}.npy").astype(np.float64)
        wave_div = DWV.pairwise_mean_dist(Wf)
        F = np.stack([PMX.default_feature_map(Wf[k], FS) for k in range(K)])
        feat_div = DWV.pairwise_mean_dist(F)
        per[(arm, S)]["RMS"] = pts.per_patient_mean(wave_div, sub_idx)
        per[(arm, S)]["FEAT"] = pts.per_patient_mean(feat_div, sub_idx)
        print(f"[b3boot] loaded {NAME[arm]:26s} S={S}  n_full={point[(arm, S)]['n_full']}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- full-sample point estimates (weight 1 per patient) and verification against EXP-B
    one = np.ones((1, pts.P))
    Q = ("rho", "rho_z", "G", "MAD", "SD", "RMS", "FEAT", "C", "I")

    def evaluate(W):
        out = {q: np.zeros((W.shape[0], len(CONDS))) for q in Q}
        for j, c in enumerate(CONDS):
            n, s, q = stats[c]
            out["rho"][:, j] = rho_bar(W, n, s, q)
            out["rho_z"][:, j] = rho_bar(W, n, s, q, fisher=True)
            for name in ("G", "MAD", "SD", "RMS", "FEAT", "C", "I"):
                out[name][:, j] = macro(per[c][name], W)
        return out
    full_pt = {q: v[0] for q, v in evaluate(one).items()}

    b1 = {(r["model"], int(float(r["S"]))): r for r in csv.DictReader(open(ROOT / "artifacts/tt_expb_mechanism/b1_fixed_K16_by_S.csv"))}
    b3 = {(r["model"], int(float(r["S"]))): r for r in csv.DictReader(open(ROOT / "artifacts/tt_expb_mechanism/b3_error_correlation.csv"))}
    verify = {}
    for j, (arm, S) in enumerate(CONDS):
        key = (NAME[arm], S)
        pairs = {"rho": (full_pt["rho"][j], float(b3[key]["pearson_mean"])),
                 "rho_direct": (point[(arm, S)]["rho_bar_direct"], float(b3[key]["pearson_mean"])),
                 "G": (full_pt["G"][j], float(b1[key]["median_gain"])), "C": (full_pt["C"][j], float(b1[key]["center_err"])),
                 "I": (full_pt["I"][j], float(b1[key]["individual_err"])), "MAD": (full_pt["MAD"][j], float(b1[key]["disp_mad"])),
                 "SD": (full_pt["SD"][j], float(b1[key]["disp_sd"])), "RMS": (full_pt["RMS"][j], float(b1[key]["wave_pairwise_rms"])),
                 "FEAT": (full_pt["FEAT"][j], float(b1[key]["feat_pairwise"]))}
        verify[f"{NAME[arm]}|S{S}"] = {q: {"here": a, "expB": b, "absdiff": abs(a - b)} for q, (a, b) in pairs.items()}
    worst = max(v["absdiff"] for d in verify.values() for v in d.values())
    print(f"[b3boot] max |difference| vs EXP-B point estimates: {worst:.3e}", flush=True)
    assert worst < 1e-6, "point estimates do not reproduce EXP-B — stop"

    # ---- bootstrap (patients with replacement; the same draw for all 12 conditions)
    rng = np.random.default_rng(BOOT_SEED)
    draws = {q: [] for q in Q}
    for b0 in range(0, N_BOOT, CHUNK):
        nb = min(CHUNK, N_BOOT - b0)
        idx = rng.integers(0, pts.P, size=(nb, pts.P))
        W = np.zeros((nb, pts.P))
        np.add.at(W, (np.repeat(np.arange(nb), pts.P), idx.ravel()), 1.0)
        res = evaluate(W)
        for q in Q:
            draws[q].append(res[q])
        print(f"[b3boot] bootstrap {b0 + nb}/{N_BOOT}  ({time.time() - t0:.0f}s)", flush=True)
    draws = {q: np.vstack(v) for q, v in draws.items()}
    np.savez_compressed(RAWOUT / "bootstrap_draws.npz", **draws, conds=np.array([f"{NAME[a]}|S{s}" for a, s in CONDS]))

    # ---- Phase 2: associations across the 12 conditions
    assoc = {}
    for name, x in (("err_corr", "rho"), ("err_corr_fisher_z", "rho_z"), ("MAD", "MAD"), ("SD", "SD"), ("waveform_RMS", "RMS"), ("feature_div", "FEAT")):
        r_b = spearman_rows(draws[x], draws["G"])
        r0 = float(spearman_rows(full_pt[x][None], full_pt["G"][None])[0])
        assoc[name] = {"point": r0, "boot": summ(r_b), "share_negative": float(np.mean(r_b < 0)), "share_positive": float(np.mean(r_b > 0))}
        assoc[name]["_draws"] = r_b
    r_err, r_wave = assoc["err_corr"]["_draws"], assoc["waveform_RMS"]["_draws"]
    diff = np.abs(r_err) - np.abs(r_wave)
    crit = {
        "C1_primary": {"ci_entirely_below_0": bool(assoc["err_corr"]["boot"]["p97.5"] < 0),
                       "sign_consistency": assoc["err_corr"]["share_negative"],
                       "holds": bool(assoc["err_corr"]["boot"]["p97.5"] < 0 and assoc["err_corr"]["share_negative"] >= 0.95)},
        "C2_comparative": {"i_sign_consistency_err_gt_wave": bool(assoc["err_corr"]["share_negative"] > assoc["waveform_RMS"]["share_positive"]),
                           "ii_abs_diff_ci": summ(diff), "ii_excludes_0": bool(np.percentile(diff, 2.5) > 0 or np.percentile(diff, 97.5) < 0),
                           "ii_direction": "err_corr stronger" if np.percentile(diff, 2.5) > 0 else ("waveform stronger" if np.percentile(diff, 97.5) < 0 else "not distinguishable")},
    }

    # ---- Phase 3: within-model ordering over the four tested depths
    within = {}
    for a in ARMS:
        cols = [j for j, c in enumerate(CONDS) if c[0] == a]
        out = {}
        for name, x in (("err_corr", "rho"), ("waveform_RMS", "RMS"), ("MAD", "MAD")):
            r_b = spearman_rows(draws[x][:, cols], draws["G"][:, cols])
            r0 = float(spearman_rows(full_pt[x][None, cols], full_pt["G"][None, cols])[0])
            out[name] = {"point": r0, "share_negative": float(np.mean(r_b < 0)), "share_positive": float(np.mean(r_b > 0)),
                         "share_perfect_negative": float(np.mean(r_b <= -1 + 1e-9)), "share_perfect_positive": float(np.mean(r_b >= 1 - 1e-9))}
        within[NAME[a]] = out
    crit["C3_within_model_share_negative_err_corr"] = {m: within[m]["err_corr"]["share_negative"] for m in within}

    # ---- Phase 4: effective sample size (mean-estimator intuition, not the actual estimator)
    thr = -1.0 / (K - 1)
    keff = lambda r: np.where(r > thr, K / (1 + (K - 1) * r), np.nan)  # noqa: E731
    keff_pt, keff_b = keff(full_pt["rho"]), keff(draws["rho"])
    valid = np.isfinite(keff_b).all(1)
    r_keff = spearman_rows(keff_b[valid], draws["G"][valid])
    ess = {"flagged_conditions": [f"{NAME[a]}|S{s}" for (a, s), v in zip(CONDS, keff_pt) if not np.isfinite(v)],
           "replicates_with_any_flag": int((~valid).sum()),
           "spearman_Keff_G": {"point": float(spearman_rows(keff_pt[None], full_pt["G"][None])[0]), "boot": summ(r_keff)},
           "note": "K_eff is a strictly decreasing function of rho_bar where defined, so Spearman(K_eff, G) = -Spearman(rho_bar, G) by construction; "
                   "it is not independent evidence."}

    # ---- per-condition table
    rows = []
    for j, (a, s) in enumerate(CONDS):
        ci = lambda q: summ(draws[q][:, j])  # noqa: E731
        rows.append({"model": NAME[a], "S": s, "K": K, "n_windows_rho": point[(a, s)]["n_full"],
                     "rho_bar": full_pt["rho"][j], "rho_bar_lo": ci("rho")["p2.5"], "rho_bar_hi": ci("rho")["p97.5"],
                     "rho_bar_fisher_z": full_pt["rho_z"][j],
                     "gain_G": full_pt["G"][j], "gain_lo": ci("G")["p2.5"], "gain_hi": ci("G")["p97.5"],
                     "center_err_C": full_pt["C"][j], "individual_err_I": full_pt["I"][j],
                     "MAD": full_pt["MAD"][j], "SD": full_pt["SD"][j], "wave_RMS": full_pt["RMS"][j], "feature_div": full_pt["FEAT"][j],
                     "K_eff": keff_pt[j], "sqrt_1_over_Keff": float(np.sqrt(1 / keff_pt[j])) if np.isfinite(keff_pt[j]) else np.nan,
                     "observed_C_over_I": full_pt["C"][j] / full_pt["I"][j]})
    with open(OUT / "conditions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- Phase 5: K = 2 anomaly — median vs mean vs 20 % trimmed mean, partition average over disjoint groups
    def pool(Hg, op):
        if op == "median":
            return np.nanmedian(Hg, 0)
        if op == "mean":
            return np.nanmean(Hg, 0)
        Xs = np.sort(Hg, 0)                                     # NaN sorted last
        n = np.isfinite(Hg).sum(0); cut = np.floor(0.2 * n).astype(int)
        pos = np.arange(Hg.shape[0])[:, None]
        keep = (pos >= cut) & (pos < n - cut)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(keep, Xs, 0.0).sum(0) / keep.sum(0)
    k2rows, k2_contrasts = [], {}
    Wone = np.ones((1, pts.P))
    rng2 = np.random.default_rng(BOOT_SEED + 1)
    Wk = np.zeros((2000, pts.P)); np.add.at(Wk, (np.repeat(np.arange(2000), pts.P), rng2.integers(0, pts.P, 2000 * pts.P)), 1.0)
    for a, S in K2_RUNS:
        for seed in (42, 1, 2):
            H = hr_matrix(a, seed, S, 32)
            perK = {}
            for Kk in K2_KS:
                g = 32 // Kk
                for op in ("median", "mean", "trimmed20"):
                    with np.errstate(all="ignore"):
                        e = np.nanmean(np.stack([np.abs(pool(H[i * Kk:(i + 1) * Kk], op) - ref) for i in range(g)]), 0)
                    pp = pts.per_patient_mean(np.where(ok, e, np.nan))
                    perK[(Kk, op)] = pp
                    k2rows.append({"model": NAME[a], "seed": seed, "S": S, "K": Kk, "groups": g, "draws_used": g * Kk,
                                   "operator": op, "hr_err": float(macro(pp, Wone)[0])})
            for Kk in (2, 3, 4, 8):
                d = perK[(Kk, "median")] - perK[(Kk, "mean")]
                k2_contrasts[f"{NAME[a]}|seed{seed}|K{Kk} median-mean"] = {"point": float(macro(d, Wone)[0]), **summ(macro(d, Wk))}
            print(f"[b3boot] K2 diagnostic {NAME[a]} seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    with open(OUT / "k2_operators.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(k2rows[0])); w.writeheader(); w.writerows(k2rows)

    for v in assoc.values():
        v.pop("_draws")
    result = {"prereg": "docs/B3_CORRELATION_BOOTSTRAP_PREREGISTRATION.md @ 590d191", "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
              "unit": "VitalDB test patient (1,156), all windows of a drawn patient, same draw for all 12 conditions",
              "runtime_s": round(time.time() - t0, 1), "verification_vs_expB_max_absdiff": worst,
              "associations_12_conditions": assoc, "criteria": crit, "within_model": within, "ess": ess,
              "k2_median_minus_mean_patient_bootstrap_2000": k2_contrasts, "verification": verify}
    (OUT / "bootstrap.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"associations": assoc, "criteria": crit, "within": within, "ess": ess}, indent=1, default=float))


if __name__ == "__main__":
    main()
