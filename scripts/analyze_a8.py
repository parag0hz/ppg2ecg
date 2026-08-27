"""A8 analysis (docs/A8_ABP_SCALE_SENSITIVITY_PREREGISTRATION.md §10-13): raw (A7) vs global-z (A8) controlled comparison, frozen
verdict, conditional-mean similarity, peak-region analysis, pointwise-ranking check, adaptive-weight and transport-geometry figures,
and the qualitative figure on the A7-selected example windows."""
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

from ppg2ecg.evaluation.abp_metrics import FS, PEAK_REGION_MS, evaluate_abp, gt_prominence, summarize_abp, systolic_peaks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"raw": {"ot": "outputs/a7_otcfm_mimicbp_seed42", "imf": "outputs/a7_imeanflow_mimicbp_seed42", "mse": "outputs/a7_mse_fullbackbone_mimicbp_seed42"},
        "global-z": {"ot": "outputs/a8_otcfm_mimicbp_globalz_seed42", "imf": "outputs/a8_imeanflow_mimicbp_globalz_seed42", "mse": "outputs/a8_mse_fullbackbone_mimicbp_globalz_seed42"}}
ARMS = [("O50", "ot", "heun25", 50), ("O20", "ot", "heun10", 20), ("O10", "ot", "heun5", 10), ("O4", "ot", "heun2", 4), ("O2", "ot", "heun1", 2), ("O1", "ot", "euler1", 1), ("M1", "imf", "meanflow1", 1), ("M2", "imf", "meanflow2", 2), ("M4", "imf", "meanflow4", 4), ("R", "mse", "regressor", 1)]
MAIN = ["R", "O1", "O50", "M1"]
COLORS = {"R": "tab:purple", "O50": "tab:blue", "O4": "tab:olive", "O1": "tab:cyan", "M1": "tab:red"}
KEYS = ["sbp_win_ae", "dbp_win_ae", "sbp_beat_ae", "dbp_beat_ae", "morph_corr", "pp_ae", "pp_ratio", "amp_ratio", "peak_timing_mae_ms", "pulse_interval_mae_ms", "pulse_count_ratio", "peak_f1", "peak_precision", "peak_recall", "rmse", "mae", "pcc", "slope_ratio", "hf_ratio_pr", "hf_ratio_gt", "rmse_peak", "rmse_nonpeak", "peak_region_energy_ratio", "peak_amp_ae"]


def load_scale(scale: str, y_ref=None):
    runs = RUNS[scale]
    ti = np.load(ROOT / runs["mse"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    x, y = ti["x"], ti["y"]
    for arm in ("ot", "imf"):
        o = np.load(ROOT / runs[arm] / "predictions" / "test_inputs.npz", allow_pickle=True)
        assert np.array_equal(o["x"], x) and np.array_equal(o["y"], y), (scale, arm)
    if y_ref is not None:
        assert np.array_equal(y_ref, y), "A8 test windows differ from A7"  # §7(6)
    meta = {a: json.loads((ROOT / runs[a] / "predictions_meta.json").read_text()) for a in runs}
    preds, shuf, info = {}, {}, {}
    for name, arm, fname, nfe in ARMS:
        d = np.load(ROOT / runs[arm] / "predictions" / f"{fname}.npz", allow_pickle=True)
        preds[name], shuf[name] = d["pred"].astype(np.float32), d["pred_shuffled"].astype(np.float32)
        info[name] = {"nfe": nfe, "latency_ms": meta[arm]["arms"][fname]["latency_ms_batch64"]}
        perm = d["perm"]
    tnorm = {a: meta[a].get("target_norm") for a in meta}
    return ti, x, y, preds, shuf, perm, info, tnorm


def metrics_for(preds, shuf, y, perm, info):
    out, pw_all = {}, {}
    for name in preds:
        pw = evaluate_abp(preds[name], y)
        pw_all[name] = pw
        s = summarize_abp(pw)
        sw = summarize_abp(evaluate_abp(shuf[name], y))
        out[name] = {k: s[k]["mean"] for k in KEYS} | {"nfe": info[name]["nfe"], "latency_ms": info[name]["latency_ms"],
                                                       "shuffle_sbpdbp_penalty": float(np.mean([sw["sbp_win_ae"]["mean"] - s["sbp_win_ae"]["mean"], sw["dbp_win_ae"]["mean"] - s["dbp_win_ae"]["mean"]])),
                                                       "shuffle_morph_penalty": float(s["morph_corr"]["mean"] - sw["morph_corr"]["mean"]),
                                                       "frac_windows_no_pred_peaks": float(np.mean(pw["n_peaks_pr"] == 0)), "pred_mean": float(preds[name].mean()), "pred_std": float(preds[name].std(axis=1).mean())}
    return out, pw_all


def pcc_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)


def similarity(preds, S):
    sim = []
    for i, a in enumerate(MAIN):
        for b in MAIN[i + 1 :]:
            d = preds[a] - preds[b]
            sa = np.abs(np.fft.rfft(preds[a] - preds[a].mean(1, keepdims=True), axis=1)).mean(0)
            sb = np.abs(np.fft.rfft(preds[b] - preds[b].mean(1, keepdims=True), axis=1)).mean(0)
            sim.append(dict(model_a=a, model_b=b, rmse=float(np.sqrt((d**2).mean(1)).mean()), mae=float(np.abs(d).mean(1).mean()), pcc=float(np.nanmean(pcc_rows(preds[a], preds[b]))),
                            d_morph=abs(np.nan_to_num(S[a]["morph_corr"]) - np.nan_to_num(S[b]["morph_corr"])), d_amp=abs(S[a]["amp_ratio"] - S[b]["amp_ratio"]), d_pp=abs(np.nan_to_num(S[a]["pp_ratio"]) - np.nan_to_num(S[b]["pp_ratio"])),
                            d_hf=abs(S[a]["hf_ratio_pr"] - S[b]["hf_ratio_pr"]), d_spec=float(np.mean(np.abs(sa - sb)))))
    cand = {r["model_b"]: r for r in sim if r["model_a"] == "R"}
    votes = {m: 0 for m in cand}
    for k in ("d_amp", "d_morph", "d_hf"):
        votes[min(cand, key=lambda m: cand[m][k])] += 1
    closest = {"rmse_closest": min(cand, key=lambda m: cand[m]["rmse"]), "pcc_closest": max(cand, key=lambda m: cand[m]["pcc"]), "stat_votes": votes}
    closest["closest_O1_to_R"] = bool(closest["rmse_closest"] == "O1" and votes["O1"] >= 2)
    return sim, closest


def rel_improve(new, old, higher_better):
    if not np.isfinite(new) or not np.isfinite(old) or old == 0:
        return float("nan")
    return (new - old) / abs(old) if higher_better else (old - new) / abs(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/a8_abp_scale_control")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ti_raw, x, y, preds_raw, shuf_raw, perm, info_raw, tn_raw = load_scale("raw")
    ti_n, x_n, y_n, preds_n, shuf_n, perm_n, info_n, tn_n = load_scale("global-z", y_ref=y)
    assert np.array_equal(x, x_n) and np.array_equal(perm, perm_n)
    S_raw, pw_raw = metrics_for(preds_raw, shuf_raw, y, perm, info_raw)
    S_n, pw_n = metrics_for(preds_n, shuf_n, y, perm, info_n)
    for lab, S in (("raw", S_raw), ("global-z", S_n)):
        for m in ("R", "O1", "O4", "O50", "M1"):
            s = S[m]
            print(f"[{lab:8s}] {m:3s} SBP {s['sbp_win_ae']:.2f} DBP {s['dbp_win_ae']:.2f} morph {s['morph_corr']:.3f} PP {s['pp_ratio']:.2f} slope {s['slope_ratio']:.2f} HF {s['hf_ratio_pr']:.3f} F1 {s['peak_f1']:.3f} RMSE {s['rmse']:.2f} | shuffle {s['shuffle_sbpdbp_penalty']:+.2f}/{s['shuffle_morph_penalty']:+.3f}", flush=True)
    sim_raw, close_raw = similarity(preds_raw, S_raw)
    sim_n, close_n = similarity(preds_n, S_n)
    # --- frozen verdict terms (§12)
    gt_hf = S_n["R"]["hf_ratio_gt"]
    imf_raw, imf_n = S_raw["M1"], S_n["M1"]
    rel = {"sbp_win_ae": rel_improve(imf_n["sbp_win_ae"], imf_raw["sbp_win_ae"], False), "dbp_win_ae": rel_improve(imf_n["dbp_win_ae"], imf_raw["dbp_win_ae"], False),
           "morph_corr": rel_improve(imf_n["morph_corr"], imf_raw["morph_corr"], True), "peak_f1": rel_improve(imf_n["peak_f1"], imf_raw["peak_f1"], True)}
    n_improved = sum(1 for v in rel.values() if np.isfinite(v) and v >= 0.10)
    hf_excess_raw, hf_excess_n = imf_raw["hf_ratio_pr"] / gt_hf, imf_n["hf_ratio_pr"] / gt_hf
    slope_err_raw, slope_err_n = abs(imf_raw["slope_ratio"] - 1), abs(imf_n["slope_ratio"] - 1)
    terms = {"imf_ge10pct_on_ge3_metrics": bool(n_improved >= 3), "n_metrics_improved_ge10pct": n_improved, "relative_improvements": rel,
             "hf_excess_raw": hf_excess_raw, "hf_excess_norm": hf_excess_n, "hf_excess_drop_ge50pct": bool(hf_excess_n <= 0.5 * hf_excess_raw),
             "slope_err_raw": slope_err_raw, "slope_err_norm": slope_err_n, "slope_err_drop_ge50pct": bool(slope_err_n <= 0.5 * slope_err_raw)}
    # baseline stability (§12 "materially changed")
    stability = {}
    for m in ("R", "O1", "O50"):
        d_morph = abs(np.nan_to_num(S_n[m]["morph_corr"]) - np.nan_to_num(S_raw[m]["morph_corr"]))
        d_f1 = abs(S_n[m]["peak_f1"] - S_raw[m]["peak_f1"])
        r_sbp = abs(S_n[m]["sbp_win_ae"] - S_raw[m]["sbp_win_ae"]) / S_raw[m]["sbp_win_ae"]
        r_dbp = abs(S_n[m]["dbp_win_ae"] - S_raw[m]["dbp_win_ae"]) / S_raw[m]["dbp_win_ae"]
        flags = [d_morph > 0.10, max(r_sbp, r_dbp) > 0.20, d_f1 > 0.10]
        stability[m] = {"d_morph": d_morph, "rel_sbp": r_sbp, "rel_dbp": r_dbp, "d_peak_f1": d_f1, "n_flags": int(sum(flags)), "materially_changed": bool(sum(flags) >= 2)}
    # adaptive-weight / objective diagnostics
    diag = list(csv.DictReader(open(out / "imeanflow_diagnostics.csv"))) if (out / "imeanflow_diagnostics.csv").exists() else []
    dmap = {r["label"]: r for r in diag}
    nondeg = None
    if "A7-raw-best" in dmap and "A8-globalz-best" in dmap:
        a, b = dmap["A7-raw-best"], dmap["A8-globalz-best"]
        nondeg = {"delta2_raw": float(a["delta2_mean"]), "delta2_norm": float(b["delta2_mean"]), "w_median_raw": float(a["w_median"]), "w_median_norm": float(b["w_median"]),
                  "u_norm_raw": float(a["u_norm"]), "u_norm_norm": float(b["u_norm"]), "residual_raw": float(a["residual_norm"]), "residual_norm_norm": float(b["residual_norm"]),
                  "regime_changed": bool(float(b["w_median"]) > 10 * float(a["w_median"]) and float(b["delta2_mean"]) < 0.5 * float(a["delta2_mean"]))}
        terms["diagnostics_nondegenerate_shift"] = nondeg["regime_changed"]
    confounded = sum(1 for m in ("R", "O1") if stability[m]["materially_changed"]) >= 2
    if confounded:
        verdict = "CONFOUNDED"
    elif terms["imf_ge10pct_on_ge3_metrics"] and terms["hf_excess_drop_ge50pct"] and terms["slope_err_drop_ge50pct"] and terms.get("diagnostics_nondegenerate_shift", False):
        verdict = "SCALE SENSITIVITY CONFIRMED"
    elif (imf_n["morph_corr"] > imf_raw["morph_corr"] + 0.10) or terms["hf_excess_drop_ge50pct"] or terms["imf_ge10pct_on_ge3_metrics"]:
        verdict = "PARTIAL SCALE SENSITIVITY"
    else:
        verdict = "NOT SUPPORTED"
    # rankings (§22)
    ranks = {}
    for lab, S in (("raw", S_raw), ("global-z", S_n)):
        ranks[lab] = {k: sorted(MAIN, key=lambda m: S[m][k]) for k in ("rmse", "rmse_peak", "rmse_nonpeak")}
        ranks[lab]["morph_corr"] = sorted(MAIN, key=lambda m: -np.nan_to_num(S[m]["morph_corr"]))
        ranks[lab]["peak_f1"] = sorted(MAIN, key=lambda m: -S[m]["peak_f1"])
        ranks[lab]["inversion"] = bool(ranks[lab]["rmse"][0] in ("R", "O1") and (ranks[lab]["morph_corr"][-1] == ranks[lab]["rmse"][0] or ranks[lab]["peak_f1"][-1] == ranks[lab]["rmse"][0]))
    # --- CSVs
    with open(out / "controlled_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "arm"] + list(next(iter(S_n.values())).keys()))
        w.writeheader()
        for lab, S in (("raw", S_raw), ("global-z", S_n)):
            for m in S:
                w.writerow({"scale": lab, "arm": m, **S[m]})
    with open(out / "prediction_similarity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale"] + list(sim_n[0].keys()))
        w.writeheader()
        for lab, sim in (("raw", sim_raw), ("global-z", sim_n)):
            for r in sim:
                w.writerow({"scale": lab, **r})
    with open(out / "peak_region_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scale", "arm", "rmse_all", "rmse_peak", "rmse_nonpeak", "peak_over_nonpeak", "peak_region_energy_ratio", "slope_ratio", "peak_amp_ae"])
        w.writeheader()
        for lab, S in (("raw", S_raw), ("global-z", S_n)):
            for m in S:
                w.writerow({"scale": lab, "arm": m, "rmse_all": S[m]["rmse"], "rmse_peak": S[m]["rmse_peak"], "rmse_nonpeak": S[m]["rmse_nonpeak"], "peak_over_nonpeak": S[m]["rmse_peak"] / max(S[m]["rmse_nonpeak"], 1e-9), "peak_region_energy_ratio": S[m]["peak_region_energy_ratio"], "slope_ratio": S[m]["slope_ratio"], "peak_amp_ae": S[m]["peak_amp_ae"]})
    summary = {"n_windows": int(len(y)), "target_norm": tn_n, "metrics": {"raw": S_raw, "global-z": S_n}, "similarity": {"raw": sim_raw, "global-z": sim_n}, "closest": {"raw": close_raw, "global-z": close_n},
               "ranks": ranks, "terms": terms, "baseline_stability": stability, "diagnostics": dmap, "verdict": verdict,
               "training": {lab: {a: json.loads((ROOT / RUNS[lab][a] / "training_summary.json").read_text()) for a in RUNS[lab]} for lab in RUNS}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # --- figures: qualitative (A7 example windows, reused)
    score = 0.5 * (pw_raw["O50"]["sbp_win_ae"] + pw_raw["O50"]["dbp_win_ae"])
    order = np.argsort(score)
    wins = [int(order[int(q * (len(y) - 1))]) for q in (0.10, 0.50, 0.90)]
    rows_fig = [("R", "raw", "MSE proxy (raw)"), ("R", "global-z", "MSE proxy (global-z)"), ("O1", "raw", "OT-CFM 1 (raw)"), ("O1", "global-z", "OT-CFM 1 (global-z)"),
                ("O50", "raw", "OT-CFM 50 (raw)"), ("O50", "global-z", "OT-CFM 50 (global-z)"), ("M1", "raw", "iMeanFlow 1 (raw)"), ("M1", "global-z", "iMeanFlow 1 (global-z)")]
    P = {"raw": preds_raw, "global-z": preds_n}
    t = np.arange(y.shape[1]) / FS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(2 + len(rows_fig), len(wins), figsize=(6.2 * len(wins), 1.55 * (2 + len(rows_fig))), sharex=True)
        for c, wdx in enumerate(wins):
            axes[0, c].plot(t, x[wdx], color="tab:green", lw=0.9)
            axes[0, c].set_title(f"{ti_raw['sid'][wdx]} window {wdx} · OT-50(raw) mean(SBP,DBP) AE {score[wdx]:.1f} mmHg", fontsize=9)
            prom = gt_prominence(y[wdx])
            pk = systolic_peaks(y[wdx], prom)
            lo, hi = y[wdx].min() - 20, y[wdx].max() + 20
            axes[1, c].plot(t, y[wdx], "k", lw=0.9)
            axes[1, c].plot(pk / FS, y[wdx][pk], "r.", ms=6)
            axes[1, c].set_ylim(lo, hi)
            for rr, (m, sc, _) in enumerate(rows_fig, start=2):
                p = P[sc][m][wdx]
                pp = systolic_peaks(p, prom)
                axes[rr, c].plot(t, y[wdx], color="0.8", lw=0.6)
                for r0 in pk:
                    axes[rr, c].axvspan((r0 - PEAK_REGION_MS / 1000 * FS) / FS, (r0 + PEAK_REGION_MS / 1000 * FS) / FS, color="0.93", zorder=0)
                axes[rr, c].plot(t, p, color=COLORS[m], lw=0.9, ls="-" if sc == "global-z" else "--")
                axes[rr, c].plot(pp / FS, p[pp], "r.", ms=5)
                axes[rr, c].set_ylim(lo, hi)
            for rr in range(2 + len(rows_fig)):
                axes[rr, c].grid(alpha=0.25)
            axes[-1, c].set_xlabel("time (s)")
        for rr, name in enumerate(["PPG (input, norm.)", "GT ABP (mmHg)"] + [lab for _, _, lab in rows_fig]):
            axes[rr, 0].set_ylabel(name, fontsize=7)
        fig.suptitle("MIMIC-BP test (A7 example windows): raw-mmHg (dashed) vs global-z (solid) training target; all predictions inverse-transformed to mmHg", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figures" / "a8_examples_raw_vs_norm.png", dpi=110)
        plt.close(fig)
    # --- transport geometry figure
    tg = json.loads((out / "transport_geometry.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    ts = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ax, key, lab in zip(axes[:2], ("interp_norm_t", "prior_share_of_interp_energy_t"), ("‖z_t‖ (mean per window)", "prior share of interpolant energy")):
        for sc, col in (("raw", "tab:orange"), ("global_z", "tab:blue")):
            ax.plot(ts, [tg[sc][f"{key}{t}"] for t in ts], "o-", color=col, label=sc.replace("_", "-"))
        ax.set_xlabel("t (0 = target, 1 = prior)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    bars = ["target_std", "target_l2_norm_mean", "prior_l2_norm_mean", "norm_ratio_target_over_prior", "distance_to_prior_mean"]
    xpos = np.arange(len(bars))
    axes[2].bar(xpos - 0.2, [tg["raw"][b] for b in bars], 0.4, label="raw", color="tab:orange")
    axes[2].bar(xpos + 0.2, [tg["global_z"][b] for b in bars], 0.4, label="global-z", color="tab:blue")
    axes[2].set_yscale("log")
    axes[2].set_xticks(xpos, ["target\nstd", "‖y‖", "‖e‖", "‖y‖/‖e‖", "‖y−e‖"], fontsize=8)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, axis="y")
    axes[2].set_title("scale mismatch (log scale)", fontsize=10)
    fig.suptitle("Transport geometry of the ABP target vs the standard-normal prior", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "a8_transport_geometry.png", dpi=110)
    plt.close(fig)
    # --- adaptive-weight figure
    wraw = list(csv.DictReader(open(ROOT / RUNS["raw"]["imf"] / "training_log.csv")))
    wnorm = list(csv.DictReader(open(ROOT / RUNS["global-z"]["imf"] / "training_log.csv")))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    have_w = "w_median" in wnorm[0]
    if have_w:
        ep = [int(r["epoch"]) + 1 for r in wnorm]
        for k, st in (("w_p10", ":"), ("w_median", "-"), ("w_p90", "--")):
            axes[0].plot(ep, [float(r[k]) for r in wnorm], st, color="tab:blue", label=f"global-z {k}")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("round")
        axes[0].set_title("adaptive weight w (global-z run)", fontsize=10)
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.25)
        axes[1].plot(ep, [float(r["w_saturation_frac"]) for r in wnorm], color="tab:blue", label="global-z")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_xlabel("round")
        axes[1].set_title("saturation fraction |w−1|<1e-6", fontsize=10)
        axes[1].grid(alpha=0.25)
    for lab, rows_, col in (("raw (A7)", wraw, "tab:orange"), ("global-z (A8)", wnorm, "tab:blue")):
        axes[2].plot([int(r["epoch"]) + 1 for r in rows_], [float(r["train_mse"]) for r in rows_], color=col, label=lab)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("round")
    axes[2].set_title("iMF training MSE of V vs (e−y) (per element)", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)
    if dmap:
        txt = " | ".join(f"{lab}: w_med {float(r['w_median']):.2e}, δ² {float(r['delta2_mean']):.3g}, ‖u‖ {float(r['u_norm']):.3g}" for lab, r in dmap.items())
        fig.suptitle("iMeanFlow objective diagnostics — " + txt, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "a8_adaptive_weight.png", dpi=110)
    plt.close(fig)
    print("closest(raw):", close_raw, "\nclosest(global-z):", close_n)
    print("terms:", json.dumps(terms, default=str))
    print("baseline stability:", json.dumps(stability, default=str))
    print("inversion raw:", ranks["raw"]["inversion"], "global-z:", ranks["global-z"]["inversion"])
    print("VERDICT:", verdict, "| wrote", out)


if __name__ == "__main__":
    main()
