"""Post-hoc diagnostics for the A0 NFE-curve predictions (no new metrics enter the pre-registered table):
 1. GT-vs-GT sanity of the metric pipeline on the real test windows (F1 must be 1.0)
 2. beat-count sanity per arm (n_pred_beats vs n_ref_beats)
 3. timing/lag analysis: per-window cross-correlation lag between prediction and GT (+-1 s), F1 at wider tolerances,
    F1 after a single global shift and after the per-window best shift (optimistic bound)
 4. per-activity breakdown (DaLiA activity labels, majority per window) of HR error / F1 for the reference arm
Writes <out>/diagnostics.json and <out>/figures/diagnostics.png"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.dalia import load_subject_raw  # noqa: E402
from ppg2ecg.data.splits import read_manifest  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 128
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}


def xcorr_lag(a, b, max_lag):
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    full = np.correlate(a, b, mode="full") / len(a)
    mid = len(a) - 1
    seg = full[mid - max_lag : mid + max_lag + 1]
    k = int(np.argmax(seg))
    return k - max_lag, float(seg[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--ref-arm", default="heun25")
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    split = read_manifest(ROOT / args.manifest)[0]
    subj = split["test"][0]
    d = np.load(ROOT / args.processed / f"{subj}.npz")
    y, starts = d["y"].astype(np.float64), d["window_start_s"]
    n, T = y.shape
    res = {"test_subject": subj, "n_windows": int(n)}

    # 1. GT vs GT
    gg = rhythm_morphology_metrics(y, y, FS)
    res["gt_vs_gt"] = {"rpeak_f1_mean": float(np.nanmean(gg["rpeak_f1"])), "hr_abs_err_mean": float(np.nanmean(gg["hr_abs_err"])), "n_ref_beats_mean": float(gg["n_ref_beats"].mean()), "windows_without_beats": int((gg["n_ref_beats"] == 0).sum())}

    # 2./3. per arm
    arms = {}
    max_lag = FS  # +-1 s
    for f in sorted((out / "predictions").glob("*.npz")):
        key = f.stem
        p = np.load(f)
        pred = p["pred"].astype(np.float64)
        nref, npred = p["pw_n_ref_beats"], p["pw_n_pred_beats"]
        lags, peaks = zip(*[xcorr_lag(pred[i], y[i], max_lag) for i in range(n)])
        lags, peaks = np.array(lags), np.array(peaks)
        # F1 at wider tolerances, and after shifts
        f1_tol = {}
        rp_ref = [R.detect_rpeaks(y[i], FS) for i in range(n)]
        rp_pred = [R.detect_rpeaks(pred[i], FS) for i in range(n)]
        def f1_shifted(i, shift, tol):
            m, fp, fn = R.match_rpeaks(rp_ref[i], rp_pred[i] - shift, FS, tol)
            return R.prf(len(m), fp, fn)[2]

        for tol in (50, 100, 150, 250):
            f1_tol[tol] = float(np.mean([f1_shifted(i, 0, tol) for i in range(n)]))
        med_lag = int(np.median(lags))
        f1_global = float(np.mean([f1_shifted(i, med_lag, 50) for i in range(n)]))
        f1_perwin = float(np.mean([f1_shifted(i, lags[i], 50) for i in range(n)]))
        arms[key] = {"solver": str(p["solver"]), "steps": int(p["steps"]), "nfe": int(p["nfe"]), "n_ref_beats_mean": float(nref.mean()), "n_pred_beats_mean": float(npred.mean()), "beat_count_ratio": float(npred.mean() / max(nref.mean(), 1e-9)),
                     "xcorr_lag_ms_median": med_lag / FS * 1000, "xcorr_lag_ms_iqr": [float(np.percentile(lags, 25) / FS * 1000), float(np.percentile(lags, 75) / FS * 1000)], "xcorr_peak_mean": float(peaks.mean()), "lag_within_100ms_frac": float(np.mean(np.abs(lags) <= 0.1 * FS)),
                     "f1_at_tol_ms": f1_tol, "f1_50ms_after_global_shift": f1_global, "f1_50ms_after_perwindow_best_shift": f1_perwin}
        arms[key]["_lags_ms"] = (lags / FS * 1000).tolist()
        print(f"{key:8s} NFE {arms[key]['nfe']:2d} | beats pred/ref {npred.mean():.1f}/{nref.mean():.1f} | lag median {arms[key]['xcorr_lag_ms_median']:.0f} ms IQR {arms[key]['xcorr_lag_ms_iqr']} | xcorr peak {peaks.mean():.2f} | F1@50/100/150/250 {f1_tol[50]:.3f}/{f1_tol[100]:.3f}/{f1_tol[150]:.3f}/{f1_tol[250]:.3f} | F1 global-shift {f1_global:.3f} per-window-shift {f1_perwin:.3f}")
    res["arms"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in arms.items()}

    # 4. per-activity for the reference arm
    raw = load_subject_raw(ROOT / "data/raw", subj)
    act = raw.activity
    act_fs = len(act) / raw.ecg_seconds  # DaLiA activity labels are sampled at 4 Hz (derived, not assumed)
    labels = []
    for s in starts:
        seg = act[int(round(s * act_fs)) : int(round((s + 8) * act_fs))]
        labels.append(int(np.bincount(seg.astype(int)).argmax()) if len(seg) else -1)
    labels = np.array(labels)
    pref = np.load(out / "predictions" / f"{args.ref_arm}.npz")
    per_act = {}
    for a in sorted(set(labels.tolist())):
        m = labels == a
        per_act[ACT.get(a, str(a))] = {"n_windows": int(m.sum()), "hr_ref_mean": float(np.nanmean(pref["pw_hr_ref"][m])), "hr_abs_err_mean": float(np.nanmean(pref["pw_hr_abs_err"][m])), "hr_abs_err_median": float(np.nanmedian(pref["pw_hr_abs_err"][m])), "rpeak_f1_mean": float(np.nanmean(pref["pw_rpeak_f1"][m])), "morph_corr_mean": float(np.nanmean(pref["pw_morph_corr"][m]))}
    res["per_activity_ref_arm"] = per_act
    res["ref_arm_hr_err_quantiles"] = {q: float(np.nanpercentile(pref["pw_hr_abs_err"], q)) for q in (10, 25, 50, 75, 90)}
    res["ref_arm_frac_hr_err_gt_10bpm"] = float(np.nanmean(pref["pw_hr_abs_err"] > 10))
    for k, v in per_act.items():
        print(f"  activity {k:10s} n={v['n_windows']:4d} HRref {v['hr_ref_mean']:.0f} | HRerr mean {v['hr_abs_err_mean']:.1f} med {v['hr_abs_err_median']:.1f} | F1 {v['rpeak_f1_mean']:.3f} | morph {v['morph_corr_mean']:.3f}")
    print("ref arm HR err quantiles:", res["ref_arm_hr_err_quantiles"], "| frac > 10 bpm:", res["ref_arm_frac_hr_err_gt_10bpm"])
    (out / "diagnostics.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for k, v in arms.items():
        axes[0].hist(v["_lags_ms"], bins=41, range=(-1000, 1000), histtype="step", label=f"{k} ({v['nfe']} NFE)")
    axes[0].set_xlabel("xcorr lag pred vs GT (ms)")
    axes[0].set_ylabel("windows")
    axes[0].legend(fontsize=7)
    axes[0].set_title("timing lag distribution")
    ks = sorted(arms, key=lambda k: arms[k]["nfe"])
    for tol in (50, 100, 150, 250):
        axes[1].plot([arms[k]["nfe"] for k in ks], [arms[k]["f1_at_tol_ms"][tol] for k in ks], "o-", label=f"tol {tol} ms")
    axes[1].plot([arms[k]["nfe"] for k in ks], [arms[k]["f1_50ms_after_perwindow_best_shift"] for k in ks], "k--", label="50 ms, per-window best shift")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("actual NFE")
    axes[1].set_ylabel("R-peak F1")
    axes[1].legend(fontsize=7)
    axes[1].set_title("F1 vs tolerance")
    names = list(per_act)
    axes[2].bar(range(len(names)), [per_act[a]["hr_abs_err_mean"] for a in names])
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("HR abs err (bpm)")
    axes[2].set_title(f"{args.ref_arm}: HR error by activity")
    fig.tight_layout()
    fig.savefig(out / "figures" / "diagnostics.png", dpi=110)
    print("wrote", out / "diagnostics.json")


if __name__ == "__main__":
    main()
