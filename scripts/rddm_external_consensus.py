"""RDDM-EXT — external consensus validation on the released RDDM checkpoint
(docs/RDDM_EXTERNAL_CONSENSUS_PREREGISTRATION.md, frozen at d294729). No training, no new sampler, no respacing,
no checkpoint modification, no upstream edit. K ∈ {1,2,3,4,8,16} nested draws at the single valid depth T = 10.

Stage `gen`      : 16 draws per window for VITALDB (primary) and WESAD / CAPNO / DALIA / BIDMC (supporting); per-sample
                   functionals computed in a process pool while the GPU generates the next draw. Resumable per corpus.
Stage `analyze`  : every preregistered metric, criterion and the verdict.

Run: PYTHONPATH=external/RDDM:outputs/rddm_env/pydeps .venv/bin/python scripts/rddm_external_consensus.py gen
     PYTHONPATH=external/RDDM:outputs/rddm_env/pydeps .venv/bin/python scripts/rddm_external_consensus.py analyze
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rddm_driver as RDR  # noqa: E402  (official loader / data / sampler; puts external/RDDM on sys.path)
import metrics as RM  # noqa: E402  (official RDDM metrics.py)
import v1_evaluate as V  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

RAW, OUT = ROOT / "outputs/rddm_ext_raw", ROOT / "artifacts/rddm_ext"
FS, NDRAW, CH = 128, 16, 256
CORPORA = ("VITALDB", "WESAD", "CAPNO", "DALIA", "BIDMC")
KS = (1, 2, 3, 4, 8, 16)
BOOT_P, BOOT_P_SEED = 2000, 20260923


# ------------------------------------------------------------------------------------------ per-sample workers
def _peaks(sig):
    return np.asarray(R.detect_rpeaks(np.asarray(sig, np.float64), FS), int)


def _sample_chunk(args):
    """generated windows [n, 512] + reference peaks -> HR (neurokit), R-peak F1 @ 50 ms, RR-MAE, MAE / RMSE vs target."""
    gen, ref_peaks, tgt = args
    out = np.full((len(gen), 5), np.nan)
    for i, (g, rp, t) in enumerate(zip(gen, ref_peaks, tgt)):
        pk = _peaks(g)
        out[i, 0] = R.hr_bpm(pk, FS)
        m, fp, fn = R.match_rpeaks(rp, pk, FS, 50.0)
        out[i, 1] = R.prf(len(m), fp, fn)[2]
        out[i, 2] = R.rr_mae_ms(rp, pk, m, FS)
        out[i, 3] = np.abs(g - t).mean()
        out[i, 4] = np.sqrt(((g - t) ** 2).mean())
    return out


def _hamilton_chunk(args):
    """Official RDDM HR extractor (metrics.ecg_bpm_array, Hamilton); generated -> filter=True, reference -> False."""
    x, filt = args
    v = np.full(len(x), np.nan)
    for i, row in enumerate(np.asarray(x, np.float64)):
        try:
            with np.errstate(all="ignore"):
                h = float(RM.ecg_bpm_array(row[None], FS, 4, filter=filt)[0])
            v[i] = h if (np.isfinite(h) and h != -1) else np.nan
        except Exception:  # noqa: BLE001  (official extractor is unguarded; a failure means "no HR")
            v[i] = np.nan
    return v


def chunks(n):
    return [(i, min(i + CH, n)) for i in range(0, n, CH)]


# ------------------------------------------------------------------------------------------ stage: gen
def gen():
    RAW.mkdir(parents=True, exist_ok=True)
    model = RDR.Model(nT=10)
    ex = ProcessPoolExecutor(16)
    for name in CORPORA:
        f = RAW / f"{name}.npz"
        if f.exists():
            print(f"[rddm-ext] {name}: cached", flush=True); continue
        t0 = time.time()
        real, P, subj, _ = RDR.load_corpus(name, 4)
        N = len(real)
        ref_peaks = list(ex.map(_peaks, list(real), chunksize=CH))
        W = np.zeros((NDRAW, N, 512), np.float16)
        futs, hfuts = [], []
        for k in range(NDRAW):
            g = model.sample(P, draw=k)
            W[k] = g.astype(np.float16)
            futs.append([ex.submit(_sample_chunk, (g[a:b], ref_peaks[a:b], real[a:b])) for a, b in chunks(N)])
            hfuts.append([ex.submit(_hamilton_chunk, (g[a:b], True)) for a, b in chunks(N)])
            print(f"[rddm-ext] {name}: draw {k} generated ({time.time() - t0:.0f}s)", flush=True)
        S = np.stack([np.concatenate([fu.result() for fu in fl]) for fl in futs])          # [16, N, 5]
        Hh = np.stack([np.concatenate([fu.result() for fu in fl]) for fl in hfuts])        # [16, N]
        Tham = np.concatenate(list(ex.map(_hamilton_chunk, [(real[a:b], False) for a, b in chunks(N)])))
        Ttgt = np.array([R.hr_bpm(p, FS) for p in ref_peaks])
        ppg = P[:, 0].numpy().astype(np.float64)
        anchor = np.array([R.hr_bpm(find_peaks(x, distance=42, prominence=0.3)[0], FS) for x in ppg])
        pairs = [(i, j) for i in range(NDRAW) for j in range(i + 1, NDRAW)]
        wrms = np.zeros(N)
        Wf = W.astype(np.float32)
        for i, j in pairs:
            wrms += np.sqrt(((Wf[i] - Wf[j]) ** 2).mean(1))
        wrms /= len(pairs)
        np.savez_compressed(f, Y=S[..., 0], F1=S[..., 1], RR=S[..., 2], MAE=S[..., 3], RMSE=S[..., 4], Yham=Hh,
                            T_target=Ttgt, T_ham=Tham, anchor=anchor, wave_rms=wrms, subject=subj)
        np.save(RAW / f"{name}_waves.npy", W)
        np.save(RAW / f"{name}_target.npy", real.astype(np.float16))
        print(f"[rddm-ext] {name}: done {N} windows x {NDRAW} draws ({time.time() - t0:.0f}s)", flush=True)
        del W, Wf


# ------------------------------------------------------------------------------------------ stage: analyze
def pool(H, op):
    if op == "median":
        return np.nanmedian(H, 0)
    if op == "mean":
        return np.nanmean(H, 0)
    Xs = np.sort(H, 0)
    n = np.isfinite(H).sum(0); cut = np.floor(0.2 * n).astype(int)
    pos = np.arange(H.shape[0])[:, None]
    keep = (pos >= cut) & (pos < n - cut)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(keep, Xs, 0.0).sum(0) / keep.sum(0)


def groups(K):
    n = 15 if K == 3 else NDRAW
    return [list(range(g * K, (g + 1) * K)) for g in range(n // K)]


def spearman_rows(A, B):
    ra, rb = rankdata(A, axis=1), rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True); rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def summ(x):
    x = np.asarray(x, float)
    return {"p2.5": float(np.nanpercentile(x, 2.5)), "p50": float(np.nanpercentile(x, 50)), "p97.5": float(np.nanpercentile(x, 97.5))}


def corpus_analysis(name, d, T, pid):
    Y = d["Y"]
    ci = lambda v: V.cluster_ci(v, pid)  # noqa: E731
    subs = np.unique(pid)
    ppat = lambda v: np.array([np.nanmean(v[pid == s]) if np.isfinite(v[pid == s]).any() else np.nan for s in subs])  # noqa: E731
    with np.errstate(all="ignore"):
        E_ind = np.abs(Y - T)                                       # [16, N]
    res = {"n_windows": int(len(T)), "n_subjects": int(len(subs)), "k_curve": {}, "operators": {}, "contrasts_vs_K1": {}}
    e1 = E_ind[0]
    for K in KS:
        with np.errstate(all="ignore"):
            est = {op: pool(Y[:K], op) for op in ("median", "mean", "trimmed20")}
            eK = {op: np.abs(v - T) for op, v in est.items()}
            I_K = np.nanmean(E_ind[:K], 0)
        C, G = eK["median"], I_K - eK["median"]
        cov = float(np.mean(np.isfinite(C) & np.isfinite(T)))
        part = {}
        for op in ("median", "mean", "trimmed20"):
            with np.errstate(all="ignore"):
                part[op] = np.nanmean(np.stack([np.abs(pool(Y[g], op) - T) for g in groups(K)]), 0)
        res["k_curve"][str(K)] = {"C_nested": ci(C), "I_nested": ci(I_K), "G_nested": ci(G), "coverage": cov,
                                  "C_partition": ci(part["median"]), "mean_nested": ci(eK["mean"]), "mean_partition": ci(part["mean"]),
                                  "trimmed20_nested": ci(eK["trimmed20"]), "trimmed20_partition": ci(part["trimmed20"]),
                                  "nfe_per_window": 20 * K}
        res["operators"][str(K)] = {"median_minus_mean_partition": ci(part["median"] - part["mean"]),
                                    "median_minus_mean_nested": ci(eK["median"] - eK["mean"])}
        if K > 1:
            common = np.isfinite(C) & np.isfinite(e1)
            dd = np.where(common, C - e1, np.nan)
            pk, p1 = ppat(np.where(common, C, np.nan)), ppat(np.where(common, e1, np.nan))
            ok = np.isfinite(pk) & np.isfinite(p1)
            res["contrasts_vs_K1"][str(K)] = {"diff": ci(dd), "common_windows": float(common.mean()), "win_rate": float(np.mean(pk[ok] < p1[ok]))}
        res.setdefault("_arrays", {})[K] = {"C": C, "G": G, "part_median": part["median"], "part_mean": part["mean"]}
    # mean-operator doubling gains (partition and nested)
    mp = {K: res["k_curve"][str(K)]["mean_partition"][0] for K in (1, 2, 4, 8, 16)}
    mn = {K: res["k_curve"][str(K)]["mean_nested"][0] for K in (1, 2, 4, 8, 16)}
    gp = [mp[K] - mp[2 * K] for K in (1, 2, 4, 8)]
    gn = [mn[K] - mn[2 * K] for K in (1, 2, 4, 8)]
    medp = {K: res["k_curve"][str(K)]["C_partition"][0] for K in (1, 2, 4, 8, 16)}
    res["doubling_gains"] = {"mean_partition": gp, "mean_nested": gn, "median_partition": [medp[K] - medp[2 * K] for K in (1, 2, 4, 8)],
                             "mean_partition_nonincreasing": bool(all(gp[i + 1] <= gp[i] for i in range(3))),
                             "mean_nested_nonincreasing": bool(all(gn[i + 1] <= gn[i] for i in range(3)))}
    # corpus-level mechanism quantities at K = 16 (EXP-B3 definitions)
    full = np.isfinite(T) & np.isfinite(Y).all(0)
    Ec = (Y - T)[:, full]
    rho = float(np.corrcoef(Ec)[np.triu_indices(NDRAW, 1)].mean())
    with np.errstate(all="ignore"):
        m = np.nanmedian(Y, 0)
        SD, MAD = np.nanstd(Y, 0, ddof=1), np.nanmedian(np.abs(Y - m), 0)
    G16 = res["_arrays"][16]["G"]
    res["mechanism_corpus"] = {"rho_bar": rho, "n_windows_rho": int(full.sum()), "G16": ci(G16)[0],
                               "SD": ci(np.where(np.isfinite(T), SD, np.nan))[0], "MAD": ci(np.where(np.isfinite(T), MAD, np.nan))[0],
                               "wave_rms": ci(d["wave_rms"])[0]}
    # secondary sample-quality metrics (K-invariant), mean over draws
    res["sample_quality"] = {q: ci(np.nanmean(d[q], 0)) for q in ("F1", "RR", "MAE", "RMSE")}
    res["sample_hr_undefined_frac"] = float(np.mean(~np.isfinite(Y[:, np.isfinite(T)])))
    # secondary functional: RDDM's own Hamilton extractor, median-pooled
    Th = d["T_ham"]
    with np.errstate(all="ignore"):
        res["hamilton_k_curve"] = {str(K): ci(np.abs(np.nanmedian(d["Yham"][:K], 0) - Th)) for K in KS}
    return res


def patient_mechanism(d, T, pid, rng_seed=BOOT_P_SEED):
    Y = d["Y"]
    elig = np.isfinite(T) & np.isfinite(Y).all(0)
    subs = np.unique(pid)
    rows = []
    for s in subs:
        w = elig & (pid == s)
        if w.sum() < 8:
            continue
        E = (Y[:, w] - T[w])
        with np.errstate(all="ignore"):
            r = np.corrcoef(E)[np.triu_indices(NDRAW, 1)]
            m = np.nanmedian(Y[:, w], 0)
            G = np.mean(np.abs(E).mean(0) - np.abs(m - T[w]))
            SD = np.mean(np.std(Y[:, w], 0, ddof=1)); MAD = np.mean(np.median(np.abs(Y[:, w] - m), 0))
        rows.append((s, np.nanmean(r) if np.isfinite(r).any() else np.nan, G, SD, MAD, float(np.mean(d["wave_rms"][w])), int(w.sum())))
    A = np.array([r[1:] for r in rows], float)
    A = A[np.isfinite(A).all(1)]
    rho, G, SD, MAD, WR = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(A), size=(BOOT_P, len(A)))
    out = {"n_patients": int(len(A)), "n_patients_with_ge8_eligible": int(len(rows)), "eligible_window_frac": float(elig.mean())}
    draws = {}
    for name, x in (("err_corr", rho), ("SD", SD), ("MAD", MAD), ("wave_rms", WR)):
        r0 = float(spearman_rows(x[None], G[None])[0])
        rb = spearman_rows(x[idx], G[idx])
        draws[name] = rb
        out[name] = {"point": r0, "boot": summ(rb), "share_negative": float(np.mean(rb < 0)), "share_positive": float(np.mean(rb > 0))}
    diff = np.abs(draws["err_corr"]) - np.abs(draws["wave_rms"])
    out["abs_err_minus_abs_wave"] = summ(diff)
    out["median_patient_rho"] = float(np.median(rho)); out["median_patient_G"] = float(np.median(G))
    return out, A


def analyze():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"prereg": "docs/RDDM_EXTERNAL_CONSENSUS_PREREGISTRATION.md @ d294729", "corpora": {}}
    for name in CORPORA:
        d = dict(np.load(RAW / f"{name}.npz", allow_pickle=True))
        pid = d["subject"]
        if name == "VITALDB":
            rp = np.load(ROOT / "outputs/rd1_detector/test_rpeaks.npz")
            T = np.array([R.hr_bpm(p, FS) for p in np.split(rp["idx"], rp["off"][1:-1])])
            v1pid = np.load(ROOT / "outputs/sr1_eval/arm_I_seed42.npz")["pid"]
            assert np.array_equal(v1pid.astype(int), pid.astype(int)), "window order differs from V1"
            pid = pid.astype(int)
        else:
            T = d["T_target"]
        r = corpus_analysis(name, d, T, pid)
        if name == "VITALDB":
            r["robustness_T_rddm_target"] = corpus_analysis(name, d, d["T_target"], pid)["contrasts_vs_K1"]["16"]
            a = np.abs(d["anchor"] - T)
            common = np.isfinite(a) & np.isfinite(r["_arrays"][1]["C"])
            r["anchor_ppg_peaks"] = {"error": V.cluster_ci(a, pid), "C1_minus_anchor": V.cluster_ci(np.where(common, r["_arrays"][1]["C"] - a, np.nan), pid),
                                     "common_windows": float(common.mean())}
            pm, A = patient_mechanism(d, T, pid)
            r["mechanism_patient"] = pm
            np.save(RAW / "vitaldb_patient_mechanism.npy", A)
        r.pop("_arrays")
        res["corpora"][name] = r
        kc = r["k_curve"]
        print(f"[rddm-ext] {name}: C1 {kc['1']['C_nested'][0]:.3f} C3 {kc['3']['C_nested'][0]:.3f} C16 {kc['16']['C_nested'][0]:.3f} "
              f"G16 {kc['16']['G_nested'][0]:.3f} cov1 {kc['1']['coverage']:.3f}  rho {r['mechanism_corpus']['rho_bar']:.3f}", flush=True)
    v = res["corpora"]["VITALDB"]
    c = v["contrasts_vs_K1"]
    crit = {
        "A-C1": bool(c["16"]["diff"][2] < 0 and v["k_curve"]["16"]["G_nested"][1] > 0),
        "A-C2": {K: bool(c[K]["diff"][2] < 0) for K in ("3", "4", "8")},
        "A-C3_i_median_eq_mean_K2": bool(abs(v["operators"]["2"]["median_minus_mean_nested"][0]) < 1e-9),
        "A-C3_ii_K3_median_better_than_mean": bool(v["operators"]["3"]["median_minus_mean_partition"][2] < 0),
        "A-C3_iii_mean_doubling_nonincreasing_partition": v["doubling_gains"]["mean_partition_nonincreasing"],
        "A-M1": bool(v["mechanism_patient"]["err_corr"]["boot"]["p97.5"] < 0),
        "instability_flag": {"a_C1_not_better_than_ppg_peaks": bool(not v["anchor_ppg_peaks"]["C1_minus_anchor"][2] < 0),
                             "b_coverage_K1_below_0.90": bool(v["k_curve"]["1"]["coverage"] < 0.90)},
    }
    crit["A-C2_all"] = all(crit["A-C2"].values())
    flag = any(crit["instability_flag"].values())
    if not crit["A-C1"]:
        verdict = "FAILED (" + ("harmful" if c["16"]["diff"][1] > 0 else "no gain") + ")"
    elif crit["A-C2_all"] and crit["A-C3_i_median_eq_mean_K2"] and crit["A-C3_ii_K3_median_better_than_mean"] and crit["A-M1"] and not flag:
        verdict = "STRONG EXTERNAL CONSENSUS REPLICATION"
    else:
        verdict = "PARTIAL"
    res["criteria"], res["verdict"] = crit, verdict
    mc = {n: res["corpora"][n]["mechanism_corpus"] for n in CORPORA}
    x = np.array([[mc[n]["rho_bar"] for n in CORPORA]]); g = np.array([[mc[n]["G16"] for n in CORPORA]])
    res["across_corpus_descriptive"] = {"spearman_rho_G": float(spearman_rows(x, g)[0]),
                                        "spearman_waveRMS_G": float(spearman_rows(np.array([[mc[n]["wave_rms"] for n in CORPORA]]), g)[0]),
                                        "spearman_SD_G": float(spearman_rows(np.array([[mc[n]["SD"] for n in CORPORA]]), g)[0]), "table": mc}
    (OUT / "result.json").write_text(json.dumps(res, indent=1, default=float))
    with open(OUT / "k_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["corpus", "K", "nfe_per_window", "coverage", "C_nested", "C_lo", "C_hi", "I_nested", "G_nested", "G_lo", "G_hi",
                                        "C_partition", "mean_partition", "trimmed20_partition", "diff_vs_K1", "diff_lo", "diff_hi", "win_rate_vs_K1"])
        for n in CORPORA:
            for K in KS:
                k = res["corpora"][n]["k_curve"][str(K)]; cc = res["corpora"][n]["contrasts_vs_K1"].get(str(K))
                w.writerow([n, K, 20 * K, k["coverage"], *k["C_nested"], k["I_nested"][0], *k["G_nested"], k["C_partition"][0], k["mean_partition"][0],
                            k["trimmed20_partition"][0], *(cc["diff"] if cc else ["", "", ""]), cc["win_rate"] if cc else ""])
    print(json.dumps({"criteria": crit, "verdict": verdict}, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["gen", "analyze"])
    {"gen": gen, "analyze": analyze}[ap.parse_args().stage]()
