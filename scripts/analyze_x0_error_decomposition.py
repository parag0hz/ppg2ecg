"""X0: decompose one-step ECG error into timing / event / shape / amplitude / sharpness using ORACLE alignment diagnostics on the
frozen prediction arrays (docs/X0_ERROR_DECOMPOSITION_PREREGISTRATION.md). No training, no regeneration."""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation.alignment_diagnostics import FS, GLOBAL_MAX_LAG_MS, LOCAL_MAX_SHIFT_MS, QRS_HALF_MS, beat_level_analysis, event_timing, global_lag, shift_crop  # noqa: E402
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROV = json.loads((ROOT / "artifacts/x0_error_decomposition/prediction_provenance.json").read_text())
PROTO = json.loads((ROOT / "artifacts/x0_error_decomposition/protocol.json").read_text())
MAIN = ["MSE", "OT1", "iMF1", "OT50"]
COLORS = {"MSE": "tab:purple", "OT1": "tab:cyan", "iMF1": "tab:red", "OT50": "tab:blue"}
GL, LS = int(round(GLOBAL_MAX_LAG_MS / 1000 * FS)), int(round(LOCAL_MAX_SHIFT_MS / 1000 * FS))


def load(ds):
    rec = PROV["datasets"][ds]
    ti = np.load(ROOT / rec["test_inputs"], allow_pickle=True)
    preds = {m: np.load(ROOT / v["prediction_file"], allow_pickle=True)["pred"].astype(np.float64) for m, v in rec["models"].items()}
    return ti, ti["y"].astype(np.float64), preds


def qrs_mask_and_peaks(y):
    half = int(round(QRS_HALF_MS / 1000 * FS))
    mask = np.zeros(y.shape, bool)
    peaks = []
    for i in range(len(y)):
        rp = R.detect_rpeaks(y[i], FS)
        peaks.append(rp)
        for r in rp:
            mask[i, max(0, r - half) : r + half + 1] = True
    return mask, peaks


def level1(pred, y, mask, peaks):
    rm = rhythm_morphology_metrics(pred, y, FS)
    e2 = (pred - y) ** 2
    d, dg = np.diff(pred, axis=1) * FS, np.diff(y, axis=1) * FS
    return {"rmse": float(np.sqrt(e2.mean(1)).mean()), "mae": float(np.abs(pred - y).mean()), "pcc": float(np.nanmean([np.corrcoef(p, t)[0, 1] if p.std() > 1e-8 else np.nan for p, t in zip(pred, y)])),
            "morph_corr": float(np.nanmean(rm["morph_corr"])), "amp_ratio": float(np.mean(pred.std(1) / (y.std(1) + 1e-8))), "amp_ratio_median": float(np.median(pred.std(1) / (y.std(1) + 1e-8))),
            "hf_ratio_pred": float(hf_energy_ratio(pred).mean()), "hf_ratio_gt": float(hf_energy_ratio(y).mean()), "qrs_energy_retention": float(pred[mask].var() / (y[mask].var() + 1e-12)),
            "max_slope_ratio": float(np.median(np.abs(d).max(1)) / (np.median(np.abs(dg).max(1)) + 1e-12)), "rpeak_precision": float(np.nanmean(rm["rpeak_precision"])), "rpeak_recall": float(np.nanmean(rm["rpeak_recall"])),
            "rpeak_f1": float(np.nanmean(rm["rpeak_f1"])), "hr_abs_err": float(np.nanmean(rm["hr_abs_err"])), "rr_mae_ms": float(np.nanmean(rm["rr_mae_ms"])), "beats_ratio": float(rm["n_pred_beats"].mean() / max(rm["n_ref_beats"].mean(), 1e-9)),
            "rmse_qrs": float(np.sqrt(e2[mask].mean())), "rmse_nonqrs": float(np.sqrt(e2[~mask].mean())), "pw_morph": rm["morph_corr"], "pw_rmse": np.sqrt(e2.mean(1))}


def level2(pred, y):
    lags, corr_b, corr_a, rmse_b, rmse_a, pa_list, ya_list = [], [], [], [], [], [], []
    for p, t in zip(pred, y):
        L, c = global_lag(p, t, GL)
        pa, ta, _ = shift_crop(p, t, L)
        lags.append(L)
        corr_b.append(np.corrcoef(p, t)[0, 1] if p.std() > 1e-8 else np.nan)
        corr_a.append(c)
        rmse_b.append(np.sqrt(np.mean((p - t) ** 2)))
        rmse_a.append(np.sqrt(np.mean((pa - ta) ** 2)))
        pa_list.append(pa)
        ya_list.append(ta)
    # frozen morphology on the aligned/cropped signals (variable length -> per window)
    morph_a = []
    for pa, ta in zip(pa_list, ya_list):
        rm = rhythm_morphology_metrics(pa[None], ta[None], FS)
        morph_a.append(float(rm["morph_corr"][0]))
    lags = np.asarray(lags)
    return {"lag_samples": lags, "lag_ms": lags / FS * 1000, "pcc_before": np.asarray(corr_b), "pcc_after": np.asarray(corr_a), "rmse_before": np.asarray(rmse_b), "rmse_after": np.asarray(rmse_a), "morph_after": np.asarray(morph_a)}


def level3(pred, y):
    rows = [event_timing(t, p, FS) for p, t in zip(pred, y)]
    n_ref = sum(r["n_ref"] for r in rows)
    errs = np.concatenate([r["signed_err_ms"] for r in rows]) if rows else np.zeros(0)
    per_win_disp = np.asarray([r["signed_err_ms"].std() if len(r["signed_err_ms"]) > 1 else np.nan for r in rows])
    return {"n_ref": n_ref, "n_pred": sum(r["n_pred"] for r in rows), "n_matched": sum(r["n_matched"] for r in rows), "n_missing": sum(r["n_missing"] for r in rows), "n_spurious": sum(r["n_spurious"] for r in rows),
            "missing_rate": sum(r["n_missing"] for r in rows) / max(n_ref, 1), "spurious_rate": sum(r["n_spurious"] for r in rows) / max(n_ref, 1),
            "timing_bias_ms": float(errs.mean()) if len(errs) else np.nan, "timing_sd_ms": float(errs.std()) if len(errs) else np.nan, "timing_mae_ms": float(np.abs(errs).mean()) if len(errs) else np.nan,
            "per_window_dispersion_ms": float(np.nanmean(per_win_disp)), "signed_err_ms": errs, "pw_missing": np.asarray([r["n_missing"] for r in rows]), "pw_ref": np.asarray([r["n_ref"] for r in rows])}


def level4(pred, y, peaks):
    acc = {}
    win_idx = []
    for i, (p, t) in enumerate(zip(pred, y)):
        res = beat_level_analysis(p, t, peaks[i], FS, LS)
        if res["n_beats"] == 0:
            continue
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                acc.setdefault(k, []).append(v)
        win_idx.append(np.full(res["n_beats"], i))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["win_idx"] = np.concatenate(win_idx)
    out["absent"] = (np.nan_to_num(out["oracle_corr"], nan=-1) < 0.5) | (out["oracle_p2p_ratio"] < 0.2)
    return out


def cluster_bootstrap(values, clusters, n_boot=2000, seed=0):
    """Mean of `values` with a cluster bootstrap (resample clusters with replacement). values: 1-D, clusters: same length labels."""
    rng = np.random.default_rng(seed)
    labs = np.unique(clusters)
    groups = [values[clusters == c] for c in labs]
    means = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(labs), len(labs))
        means.append(np.nanmean(np.concatenate([groups[i] for i in pick])))
    return float(np.nanmean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def label(val, thr, higher_better):
    lo, hi = thr
    if higher_better:
        return "LOW" if val >= lo else ("MODERATE" if val >= hi else "HIGH")
    return "LOW" if val < lo else ("MODERATE" if val <= hi else "HIGH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["wildppg", "dalia_s2", "dalia_s1"])
    ap.add_argument("--out", default="artifacts/x0_error_decomposition")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    site_path = Path("/tmp/claude-1000/-home-kwy00-ppg2ecg-one-step/0b21629b-bc66-41bc-ad2e-49613e7563ba/scratchpad/wildppg_test_site.npy")
    tables = {k: [] for k in ("raw_metrics", "global_alignment", "event_timing", "event_failures", "beat_shape_raw", "beat_shape_oracle_aligned", "qrs_structure", "model_pair_similarity", "rmse_decomposition", "clustered_bootstrap")}
    summary = {"protocol": PROTO, "datasets": {}}
    for ds in args.datasets:
        ti, y, preds = load(ds)
        models = [m for m in MAIN if m in preds]
        mask, peaks = qrs_mask_and_peaks(y)
        n_gt_beats = sum(len(p) for p in peaks)
        if ds == "wildppg" and site_path.exists():
            site = np.load(site_path, allow_pickle=True)
            clusters = np.array([f"{s}/{t}" for s, t in zip(ti["sid"], site)])
        else:
            clusters = np.asarray(ti["sid"]).astype(str)
        L1, L2, L3, L4 = {}, {}, {}, {}
        for m in models:
            L1[m] = level1(preds[m], y, mask, peaks)
            L2[m] = level2(preds[m], y)
            L3[m] = level3(preds[m], y)
            L4[m] = level4(preds[m], y, peaks)
            l1, l2, l3, l4 = L1[m], L2[m], L3[m], L4[m]
            print(f"[{ds}] {m:5s} raw morph {l1['morph_corr']:.3f} amp {l1['amp_ratio_median']:.2f} QRSe {l1['qrs_energy_retention']:.2f} slope {l1['max_slope_ratio']:.2f} F1 {l1['rpeak_f1']:.3f} | lag med {np.median(l2['lag_ms']):+.0f} ms morph→ {np.nanmean(l2['morph_after']):.3f} | timing MAE {l3['timing_mae_ms']:.1f} bias {l3['timing_bias_ms']:+.1f} sd {l3['timing_sd_ms']:.1f} miss {l3['missing_rate']:.2f} spur {l3['spurious_rate']:.2f} | beat corr A {np.nanmean(l4['raw_corr']):.3f} → B {np.nanmean(l4['oracle_corr']):.3f} p2p {np.nanmedian(l4['oracle_p2p_ratio']):.2f} QRSe {np.nanmedian(l4['oracle_qrs_energy_ratio']):.2f} slope {np.nanmedian(l4['oracle_slope_ratio']):.2f} absent {l4['absent'].mean():.2f} shift med {np.median(np.abs(l4['shift_samples']))/FS*1000:.0f} ms", flush=True)
        # ---- tables
        for m in models:
            l1, l2, l3, l4 = L1[m], L2[m], L3[m], L4[m]
            tables["raw_metrics"].append({"dataset": ds, "model": m, **{k: v for k, v in l1.items() if not isinstance(v, np.ndarray)}})
            tables["global_alignment"].append({"dataset": ds, "model": m, "lag_median_ms": float(np.median(l2["lag_ms"])), "lag_iqr_ms": float(np.percentile(l2["lag_ms"], 75) - np.percentile(l2["lag_ms"], 25)), "abs_lag_mean_ms": float(np.abs(l2["lag_ms"]).mean()), "frac_at_range_limit": float(np.mean(np.abs(l2["lag_samples"]) >= GL)),
                                               "pcc_before": float(np.nanmean(l2["pcc_before"])), "pcc_after": float(np.nanmean(l2["pcc_after"])), "rmse_before": float(l2["rmse_before"].mean()), "rmse_after": float(l2["rmse_after"].mean()), "morph_before": l1["morph_corr"], "morph_after": float(np.nanmean(l2["morph_after"])), "delta_global_morph": float(np.nanmean(l2["morph_after"]) - l1["morph_corr"]), "delta_global_pcc": float(np.nanmean(l2["pcc_after"]) - np.nanmean(l2["pcc_before"]))})
            tables["event_timing"].append({"dataset": ds, "model": m, **{k: v for k, v in l3.items() if not isinstance(v, np.ndarray)}})
            tables["event_failures"].append({"dataset": ds, "model": m, "n_gt_beats": l3["n_ref"], "n_pred_beats": l3["n_pred"], "matched": l3["n_matched"], "missing": l3["n_missing"], "spurious": l3["n_spurious"], "missing_rate": l3["missing_rate"], "spurious_rate": l3["spurious_rate"], "oracle_absent_rate": float(l4["absent"].mean()), "n_beats_oracle": int(l4["n_beats"].sum()) if "n_beats" in l4 else int(len(l4["oracle_corr"]))})
            for ver, pre in (("beat_shape_raw", "raw_"), ("beat_shape_oracle_aligned", "oracle_")):
                tables[ver].append({"dataset": ds, "model": m, "n_beats": int(len(l4[pre + "corr"])), "corr_mean": float(np.nanmean(l4[pre + "corr"])), "corr_median": float(np.nanmedian(l4[pre + "corr"])), "p2p_ratio_median": float(np.nanmedian(l4[pre + "p2p_ratio"])), "r_amp_ratio_median": float(np.nanmedian(l4[pre + "r_amp_ratio"])), "qrs_energy_ratio_median": float(np.nanmedian(l4[pre + "qrs_energy_ratio"])), "slope_ratio_median": float(np.nanmedian(l4[pre + "slope_ratio"])), "hf_ratio_pred": float(np.nanmean(l4[pre + "hf_ratio_pred"])), "hf_ratio_gt": float(np.nanmean(l4[pre + "hf_ratio_gt"])), "seg_rmse": float(np.nanmean(l4[pre + "rmse"])), "qrs_rmse": float(np.nanmean(l4[pre + "qrs_rmse"])), "shift_abs_median_ms": float(np.median(np.abs(l4["shift_samples"])) / FS * 1000) if pre == "oracle_" else 0.0, "shift_sd_ms": float(l4["shift_samples"].std() / FS * 1000) if pre == "oracle_" else 0.0, "frac_shift_at_limit": float(np.mean(np.abs(l4["shift_samples"]) >= LS)) if pre == "oracle_" else 0.0})
            tables["qrs_structure"].append({"dataset": ds, "model": m, "qrs_energy_retention_window": l1["qrs_energy_retention"], "rmse_qrs": l1["rmse_qrs"], "rmse_nonqrs": l1["rmse_nonqrs"], "max_slope_ratio_window": l1["max_slope_ratio"], "hf_ratio": l1["hf_ratio_pred"], "oracle_qrs_energy_median": float(np.nanmedian(l4["oracle_qrs_energy_ratio"])), "oracle_slope_median": float(np.nanmedian(l4["oracle_slope_ratio"])), "oracle_qrs_rmse": float(np.nanmean(l4["oracle_qrs_rmse"]))})
            tables["rmse_decomposition"].append({"dataset": ds, "model": m, "rmse_raw": l1["rmse"], "rmse_global_aligned": float(l2["rmse_after"].mean()), "rmse_qrs_raw": l1["rmse_qrs"], "rmse_nonqrs_raw": l1["rmse_nonqrs"], "qrs_rmse_oracle_local": float(np.nanmean(l4["oracle_qrs_rmse"])), "morph_corr": l1["morph_corr"], "oracle_beat_corr": float(np.nanmean(l4["oracle_corr"])), "qrs_energy_retention": l1["qrs_energy_retention"], "amp_ratio_median": l1["amp_ratio_median"], "slope_ratio": l1["max_slope_ratio"]})
        # ---- pair similarity: raw, global-aligned-to-GT frame, oracle beat segments
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                d = preds[a] - preds[b]
                raw_rmse, raw_pcc = float(np.sqrt((d**2).mean(1)).mean()), float(np.nanmean([np.corrcoef(u, v)[0, 1] for u, v in zip(preds[a], preds[b]) if u.std() > 1e-8 and v.std() > 1e-8]))
                g_rmse, g_pcc = [], []
                for k in range(len(y)):
                    La, Lb = L2[a]["lag_samples"][k], L2[b]["lag_samples"][k]
                    pa, _, oa = shift_crop(preds[a][k], y[k], La)
                    pb, _, ob = shift_crop(preds[b][k], y[k], Lb)
                    lo, hi = max(oa, ob), min(oa + len(pa), ob + len(pb))
                    ua, ub = pa[lo - oa : hi - oa], pb[lo - ob : hi - ob]
                    g_rmse.append(np.sqrt(np.mean((ua - ub) ** 2)))
                    g_pcc.append(np.corrcoef(ua, ub)[0, 1] if ua.std() > 1e-8 and ub.std() > 1e-8 else np.nan)
                tables["model_pair_similarity"].append({"dataset": ds, "model_a": a, "model_b": b, "rmse_raw": raw_rmse, "pcc_raw": raw_pcc, "rmse_global_aligned": float(np.mean(g_rmse)), "pcc_global_aligned": float(np.nanmean(g_pcc)),
                                                        "d_amp_median": abs(L1[a]["amp_ratio_median"] - L1[b]["amp_ratio_median"]), "d_qrs_energy": abs(L1[a]["qrs_energy_retention"] - L1[b]["qrs_energy_retention"]), "d_slope": abs(L1[a]["max_slope_ratio"] - L1[b]["max_slope_ratio"]), "d_hf": abs(L1[a]["hf_ratio_pred"] - L1[b]["hf_ratio_pred"]),
                                                        "d_oracle_beat_corr": abs(float(np.nanmean(L4[a]["oracle_corr"])) - float(np.nanmean(L4[b]["oracle_corr"]))), "d_oracle_p2p": abs(float(np.nanmedian(L4[a]["oracle_p2p_ratio"])) - float(np.nanmedian(L4[b]["oracle_p2p_ratio"])))})
        # ---- recoverability, categories, verdict
        ref = L4["OT50"]
        QA, QB = {m: float(np.nanmean(L4[m]["raw_corr"])) for m in models}, {m: float(np.nanmean(L4[m]["oracle_corr"])) for m in models}
        ret_ref = {k: float(np.nanmedian(ref[f"oracle_{k}"])) for k in ("p2p_ratio", "qrs_energy_ratio", "slope_ratio")}
        cats, rgf, labels = {}, {}, {}
        for m in models:
            raw_gap, al_gap = QA["OT50"] - QA[m], QB["OT50"] - QB[m]
            f = (raw_gap - al_gap) / raw_gap if raw_gap >= 0.05 else None
            rgf[m] = {"raw_gap": raw_gap, "aligned_gap": al_gap, "recoverable_gap_fraction": f, "stable": raw_gap >= 0.05, "delta_local_Q": QB[m] - QA[m], "delta_global_morph": float(np.nanmean(L2[m]["morph_after"]) - L1[m]["morph_corr"])}
            ret = {k: float(np.nanmedian(L4[m][f"oracle_{k}"])) for k in ("p2p_ratio", "qrs_energy_ratio", "slope_ratio")}
            ok_ret = all(ret[k] >= 0.5 * ret_ref[k] for k in ret)
            bad_ret = all(ret[k] < 0.5 * ret_ref[k] for k in ret)
            absent = float(L4[m]["absent"].mean())
            if m == "OT50":
                cat = "REFERENCE"
            elif absent >= 0.5:
                cat = "EVENT-DOMINANT"
            elif f is None:
                cat = "INCONCLUSIVE(unstable gap)"
            elif f >= 0.5 and ok_ret:
                cat = "TIMING-MAJOR"
            elif f < 0.2 and bad_ret:
                cat = "SHAPE-DOMINANT"
            else:
                cat = "MIXED"
            cats[m] = cat
            tm = L3[m]["timing_mae_ms"]
            deficits = {"timing": min(1.0, tm / 50.0) if np.isfinite(tm) else 1.0, "event": absent, "shape": 1 - QB[m], "amplitude": max(0.0, 1 - ret["p2p_ratio"]), "sharpness": max(0.0, 1 - ret["slope_ratio"])}
            labels[m] = {"timing": label(tm if np.isfinite(tm) else 99, PROTO["labels"]["timing_ms"], False), "event": label(absent, PROTO["labels"]["event_absent_rate"], False), "shape": label(QB[m], PROTO["labels"]["shape_corr"], True), "amplitude": label(ret["p2p_ratio"], PROTO["labels"]["amp_p2p"], True), "sharpness": label(ret["slope_ratio"], PROTO["labels"]["sharp_slope"], True), "dominant": max(deficits, key=deficits.get), "deficits": deficits, "retention_after_alignment": ret, "category": cat}
        if cats["OT1"].startswith("INCONCLUSIVE") and cats["MSE"].startswith("INCONCLUSIVE"):
            verdict = "INCONCLUSIVE"
        elif cats["OT1"] == cats["MSE"]:
            verdict = cats["OT1"]
        else:
            verdict = f"MODEL-DEPENDENT (OT1 {cats['OT1']}; MSE {cats['MSE']})"
        # ---- cluster bootstrap on key per-window / per-beat quantities
        boot = {}
        for m in models:
            for name, vals, cl in (("global_lag_abs_ms", np.abs(L2[m]["lag_ms"]), clusters), ("delta_global_morph", L2[m]["morph_after"] - L1[m]["pw_morph"], clusters), ("oracle_beat_corr", L4[m]["oracle_corr"], clusters[L4[m]["win_idx"]]), ("delta_local_corr", L4[m]["oracle_corr"] - L4[m]["raw_corr"], clusters[L4[m]["win_idx"]]), ("oracle_p2p_ratio", L4[m]["oracle_p2p_ratio"], clusters[L4[m]["win_idx"]]), ("oracle_slope_ratio", L4[m]["oracle_slope_ratio"], clusters[L4[m]["win_idx"]]), ("oracle_absent", L4[m]["absent"].astype(float), clusters[L4[m]["win_idx"]]), ("missing_per_gt_beat", L3[m]["pw_missing"] / np.maximum(L3[m]["pw_ref"], 1), clusters)):
                mean, lo, hi = cluster_bootstrap(vals, cl)
                boot[f"{m}/{name}"] = (mean, lo, hi)
                tables["clustered_bootstrap"].append({"dataset": ds, "model": m, "quantity": name, "mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": int(np.isfinite(vals).sum()), "clusters": int(len(np.unique(cl))), "bootstrap": "cluster" if ds == "wildppg" else "window(exploratory)"})
        for a, b in (("OT1", "OT50"), ("MSE", "OT50"), ("iMF1", "OT50"), ("OT1", "MSE"), ("iMF1", "OT1")):
            if a in models and b in models:
                # paired per-window delta of oracle beat corr (window means)
                wa = np.array([np.nanmean(L4[a]["oracle_corr"][L4[a]["win_idx"] == k]) if np.any(L4[a]["win_idx"] == k) else np.nan for k in range(len(y))])
                wb = np.array([np.nanmean(L4[b]["oracle_corr"][L4[b]["win_idx"] == k]) if np.any(L4[b]["win_idx"] == k) else np.nan for k in range(len(y))])
                mean, lo, hi = cluster_bootstrap(wa - wb, clusters)
                tables["clustered_bootstrap"].append({"dataset": ds, "model": f"{a}-{b}", "quantity": "paired_delta_oracle_beat_corr", "mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": int(np.isfinite(wa - wb).sum()), "clusters": int(len(np.unique(clusters))), "bootstrap": "cluster" if ds == "wildppg" else "window(exploratory)"})
        summary["datasets"][ds] = {"n_windows": int(len(y)), "n_gt_beats": int(n_gt_beats), "n_clusters": int(len(np.unique(clusters))), "QA_raw_beat_corr": QA, "QB_oracle_beat_corr": QB, "recoverability": rgf, "categories": cats, "labels": labels, "verdict": verdict, "bootstrap": boot,
                                   "global": {m: {"lag_median_ms": float(np.median(L2[m]["lag_ms"])), "lag_iqr_ms": float(np.percentile(L2[m]["lag_ms"], 75) - np.percentile(L2[m]["lag_ms"], 25)), "morph_before": L1[m]["morph_corr"], "morph_after": float(np.nanmean(L2[m]["morph_after"])), "pcc_before": float(np.nanmean(L2[m]["pcc_before"])), "pcc_after": float(np.nanmean(L2[m]["pcc_after"])), "rmse_before": float(L2[m]["rmse_before"].mean()), "rmse_after": float(L2[m]["rmse_after"].mean())} for m in models},
                                   "timing": {m: {k: v for k, v in L3[m].items() if not isinstance(v, np.ndarray)} for m in models},
                                   "beat": {m: {"raw_corr": QA[m], "oracle_corr": QB[m], "oracle_p2p_median": float(np.nanmedian(L4[m]["oracle_p2p_ratio"])), "oracle_qrs_energy_median": float(np.nanmedian(L4[m]["oracle_qrs_energy_ratio"])), "oracle_slope_median": float(np.nanmedian(L4[m]["oracle_slope_ratio"])), "oracle_hf": float(np.nanmean(L4[m]["oracle_hf_ratio_pred"])), "shift_abs_median_ms": float(np.median(np.abs(L4[m]["shift_samples"])) / FS * 1000), "shift_sd_ms": float(L4[m]["shift_samples"].std() / FS * 1000), "absent_rate": float(L4[m]["absent"].mean()), "n_beats": int(len(L4[m]["oracle_corr"]))} for m in models},
                                   "raw": {m: {k: v for k, v in L1[m].items() if not isinstance(v, np.ndarray)} for m in models}}
        print(f"[{ds}] categories {cats} | verdict {verdict} | rgf {{m: (round(v['recoverable_gap_fraction'],3) if v['recoverable_gap_fraction'] is not None else None) for m, v in rgf.items()}}", flush=True)
        # ---- figures: frozen qualitative windows (raw + globally aligned), beat panels, distributions
        wins = PROTO["qualitative_ids"][ds]
        t = np.arange(y.shape[1]) / FS
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axes = plt.subplots(2 + 2 * len(models), len(wins), figsize=(6.2 * len(wins), 1.5 * (2 + 2 * len(models))), sharex=True, sharey=True)
            for c, w in enumerate(wins):
                axes[0, c].plot(t, ti["x"][w], color="tab:green", lw=0.9)
                axes[0, c].set_title(f"{ti['sid'][w]} window {w}", fontsize=9)
                axes[1, c].plot(t, y[w], "k", lw=0.9)
                axes[1, c].plot(peaks[w] / FS, y[w][peaks[w]], "r.", ms=5)
                row = 2
                for m in models:
                    p = preds[m][w]
                    rp = R.detect_rpeaks(p, FS)
                    axes[row, c].plot(t, y[w], color="0.8", lw=0.6)
                    axes[row, c].plot(t, p, color=COLORS[m], lw=0.9)
                    axes[row, c].plot(rp / FS, p[rp], "r.", ms=4)
                    L = L2[m]["lag_samples"][w]
                    pa, ta, off = shift_crop(p, y[w], L)
                    axes[row + 1, c].plot(t, y[w], color="0.8", lw=0.6)
                    axes[row + 1, c].plot((np.arange(len(pa)) + off) / FS, pa, color=COLORS[m], lw=0.9, ls="--")
                    axes[row + 1, c].text(0.01, 0.8, f"lag {L/FS*1000:+.0f} ms", transform=axes[row + 1, c].transAxes, fontsize=7)
                    if c == 0:
                        axes[row, c].set_ylabel(f"{m} raw", fontsize=8)
                        axes[row + 1, c].set_ylabel(f"{m} global-aligned", fontsize=8)
                    row += 2
                axes[-1, c].set_xlabel("time (s)")
            for rr in range(axes.shape[0]):
                for c in range(len(wins)):
                    axes[rr, c].grid(alpha=0.25)
                    axes[rr, c].set_ylim(-1.6, 1.6)
            fig.suptitle(f"X0 {ds}: raw vs oracle global-aligned (±{GLOBAL_MAX_LAG_MS:.0f} ms) predictions; identical y-scale (grey = GT, red = R-peaks)", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"x0_{ds}_windows_raw_vs_global.png", dpi=110)
            plt.close(fig)
            # beat panels: first 3 valid GT beats of the median-quantile window
            w = wins[1]
            segs = [(a, b, rl) for (a, b, rl) in __import__("ppg2ecg.evaluation.alignment_diagnostics", fromlist=["beat_segments_gt"]).beat_segments_gt(y[w], peaks[w], FS, margin=LS)[0]][:4]
            fig, axes = plt.subplots(len(models), len(segs), figsize=(3.2 * len(segs), 2.0 * len(models)), sharey=True)
            axes = np.atleast_2d(axes)
            for ci, (a, b, rl) in enumerate(segs):
                tt = (np.arange(a, b) - (a + rl)) / FS * 1000
                for ri, m in enumerate(models):
                    p = preds[m][w]
                    from ppg2ecg.evaluation.alignment_diagnostics import oracle_local_shift
                    d, cc = oracle_local_shift(p, y[w], a, b, LS)
                    axes[ri, ci].plot(tt, y[w][a:b], "k", lw=1.0)
                    axes[ri, ci].plot(tt, p[a:b], color=COLORS[m], lw=0.9, alpha=0.5)
                    axes[ri, ci].plot(tt, p[a + d : b + d], color=COLORS[m], lw=1.2, ls="--")
                    axes[ri, ci].set_title(f"{m}: shift {d/FS*1000:+.0f} ms, corr {cc:.2f}", fontsize=8)
                    axes[ri, ci].grid(alpha=0.25)
                    axes[ri, ci].set_ylim(-1.6, 1.6)
                axes[-1, ci].set_xlabel("ms from GT R-peak")
            fig.suptitle(f"X0 {ds} window {w}: GT beat (black), prediction at same coordinates (light), oracle-shifted ±{LOCAL_MAX_SHIFT_MS:.0f} ms (dashed)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"x0_{ds}_beats_oracle_local.png", dpi=110)
            plt.close(fig)
            # distributions
            fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
            for ax, (key, getter, lab) in zip(axes.ravel(), [("lag", lambda m: L2[m]["lag_ms"], "global lag (ms)"), ("timing", lambda m: L3[m]["signed_err_ms"], "matched R-peak timing error (ms)"), ("shift", lambda m: L4[m]["shift_samples"] / FS * 1000, "oracle local shift (ms)"), ("gain", lambda m: L4[m]["oracle_corr"] - L4[m]["raw_corr"], "raw→oracle beat corr gain"), ("qrse", lambda m: np.clip(L4[m]["oracle_qrs_energy_ratio"], 0, 3), "oracle QRS energy ratio"), ("slope", lambda m: np.clip(L4[m]["oracle_slope_ratio"], 0, 3), "oracle slope ratio"), ("p2p", lambda m: np.clip(L4[m]["oracle_p2p_ratio"], 0, 3), "oracle p2p amplitude ratio"), ("absent", None, "missing / spurious / oracle-absent rate")]):
                if getter is None:
                    xpos = np.arange(len(models))
                    ax.bar(xpos - 0.25, [L3[m]["missing_rate"] for m in models], 0.25, label="missing (detector)")
                    ax.bar(xpos, [L3[m]["spurious_rate"] for m in models], 0.25, label="spurious (detector)")
                    ax.bar(xpos + 0.25, [L4[m]["absent"].mean() for m in models], 0.25, label="oracle-absent")
                    ax.set_xticks(xpos, models)
                    ax.legend(fontsize=7)
                else:
                    for m in models:
                        v = getter(m)
                        v = v[np.isfinite(v)]
                        if len(v):
                            ax.hist(v, bins=40, histtype="step", color=COLORS[m], label=m, density=True)
                    ax.legend(fontsize=7)
                ax.set_title(lab, fontsize=9)
                ax.grid(alpha=0.25)
            fig.suptitle(f"X0 {ds}: distributions (windows/beats pooled; clustered CIs in clustered_bootstrap.csv)", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"x0_{ds}_distributions.png", dpi=110)
            plt.close(fig)
    for name, rows in tables.items():
        if rows:
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
