"""A7 analysis (docs/A7_ABP_PREREGISTRATION.md §6-13): ABP metrics per arm from the saved paired predictions, shuffle penalties,
cross-model similarity, peak-region analysis, pointwise inversion, structural recovery, Pareto, frozen verdict, deterministic figures."""
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
RUNS = {"ot": "outputs/a7_otcfm_mimicbp_seed42", "imf": "outputs/a7_imeanflow_mimicbp_seed42", "mse": "outputs/a7_mse_fullbackbone_mimicbp_seed42"}
if "A7_RUNS_JSON" in __import__("os").environ:  # dry-run override only
    RUNS = json.loads(__import__("os").environ["A7_RUNS_JSON"])
ARMS = [("O50", "ot", "heun25", 50), ("O20", "ot", "heun10", 20), ("O10", "ot", "heun5", 10), ("O4", "ot", "heun2", 4), ("O2", "ot", "heun1", 2), ("O1", "ot", "euler1", 1), ("M1", "imf", "meanflow1", 1), ("M2", "imf", "meanflow2", 2), ("M4", "imf", "meanflow4", 4), ("R", "mse", "regressor", 1)]
MAIN = ["R", "O1", "O50", "M1"]
COLORS = {"R": "tab:purple", "O50": "tab:blue", "O4": "tab:olive", "O1": "tab:cyan", "M1": "tab:red"}
KEYS = ["sbp_win_ae", "dbp_win_ae", "sbp_beat_ae", "dbp_beat_ae", "morph_corr", "pp_ae", "pp_ratio", "amp_ratio", "peak_timing_mae_ms", "pulse_interval_mae_ms", "pulse_count_ratio", "peak_f1", "peak_precision", "peak_recall", "rmse", "mae", "pcc", "slope_ratio", "hf_ratio_pr", "hf_ratio_gt", "rmse_peak", "rmse_nonpeak", "peak_region_energy_ratio", "peak_amp_ae", "n_peaks_gt", "n_peaks_pr"]


def load_all():
    ti = np.load(ROOT / RUNS["mse"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    x, y = ti["x"], ti["y"]
    for arm in ("ot", "imf"):
        o = np.load(ROOT / RUNS[arm] / "predictions" / "test_inputs.npz", allow_pickle=True)
        assert np.array_equal(o["x"], x) and np.array_equal(o["y"], y), arm
    meta = {arm: json.loads((ROOT / RUNS[arm] / "predictions_meta.json").read_text()) for arm in RUNS}
    preds, shuf, info = {}, {}, {}
    for name, arm, fname, nfe in ARMS:
        d = np.load(ROOT / RUNS[arm] / "predictions" / f"{fname}.npz", allow_pickle=True)
        preds[name], shuf[name] = d["pred"].astype(np.float32), d["pred_shuffled"].astype(np.float32)
        info[name] = {"nfe": nfe, "latency_ms": meta[arm]["arms"][fname]["latency_ms_batch64"], "arm": arm, "file": fname}
        perm = d["perm"]
    return ti, x, y, preds, shuf, perm, info, meta


def pcc_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)


def recovery(m1, o1, o50, higher_is_better=True, eps=0.05):
    den = o50 - o1
    if abs(den) < eps:
        return None
    r = (m1 - o1) / den
    return float(r if higher_is_better else r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/a7_abp_generalization")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ti, x, y, preds, shuf, perm, info, meta = load_all()
    n = len(y)
    # ---- per-arm ABP metrics (correct PPG) and shuffled-PPG metrics against the *right* targets of the shuffled inputs (y[perm])
    summ, pw_all, sh_pen = {}, {}, {}
    for name in preds:
        pw = evaluate_abp(preds[name], y)
        pw_all[name] = pw
        s = summarize_abp(pw)
        pws = evaluate_abp(shuf[name], y[perm])  # control: shuffled prediction vs the target that belongs to the PPG it was driven by
        pww = evaluate_abp(shuf[name], y)  # prereg §8 "shuffled": prediction driven by the WRONG PPG, scored against the window's own target
        ss, sw = summarize_abp(pws), summarize_abp(pww)
        sh_pen[name] = {"shuffle_sbpdbp_penalty": float(np.mean([sw["sbp_win_ae"]["mean"] - s["sbp_win_ae"]["mean"], sw["dbp_win_ae"]["mean"] - s["dbp_win_ae"]["mean"]])), "shuffle_morph_penalty": float(s["morph_corr"]["mean"] - sw["morph_corr"]["mean"]), "sbp_win_ae_shuffled": sw["sbp_win_ae"]["mean"], "dbp_win_ae_shuffled": sw["dbp_win_ae"]["mean"], "morph_shuffled": sw["morph_corr"]["mean"], "sbp_win_ae_shuffled_vs_own_target": ss["sbp_win_ae"]["mean"]}
        summ[name] = {k: s[k]["mean"] for k in KEYS} | {"n_valid_morph": s["morph_corr"]["n_valid"], "frac_windows_no_pred_peaks": float(np.mean(pw["n_peaks_pr"] == 0)), "nfe": info[name]["nfe"], "latency_ms": info[name]["latency_ms"]} | sh_pen[name]
        print(f"{name:4s} NFE {info[name]['nfe']:3d} | SBP {summ[name]['sbp_win_ae']:.2f} DBP {summ[name]['dbp_win_ae']:.2f} (beat {summ[name]['sbp_beat_ae']:.2f}/{summ[name]['dbp_beat_ae']:.2f}) | morph {summ[name]['morph_corr']:.3f} PPratio {summ[name]['pp_ratio']:.2f} amp {summ[name]['amp_ratio']:.2f} slope {summ[name]['slope_ratio']:.2f} HF {summ[name]['hf_ratio_pr']:.3f} | F1 {summ[name]['peak_f1']:.3f} timing {summ[name]['peak_timing_mae_ms']:.1f} ms cnt {summ[name]['pulse_count_ratio']:.2f} | RMSE {summ[name]['rmse']:.2f} peak {summ[name]['rmse_peak']:.2f} non {summ[name]['rmse_nonpeak']:.2f} | shuffle SBPDBP +{sh_pen[name]['shuffle_sbpdbp_penalty']:.2f} morph {sh_pen[name]['shuffle_morph_penalty']:+.3f} | {info[name]['latency_ms']:.0f} ms", flush=True)
    # ---- similarity
    sim = []
    for i, a in enumerate(MAIN):
        for b in MAIN[i + 1 :]:
            d = preds[a] - preds[b]
            sim.append(dict(model_a=a, model_b=b, rmse=float(np.sqrt((d**2).mean(1)).mean()), mae=float(np.abs(d).mean(1).mean()), pcc=float(np.nanmean(pcc_rows(preds[a], preds[b]))), d_amp=abs(summ[a]["amp_ratio"] - summ[b]["amp_ratio"]), d_morph=abs(np.nan_to_num(summ[a]["morph_corr"]) - np.nan_to_num(summ[b]["morph_corr"])), d_hf=abs(summ[a]["hf_ratio_pr"] - summ[b]["hf_ratio_pr"]), d_spec=float(np.mean(np.abs(np.abs(np.fft.rfft(preds[a] - preds[a].mean(1, keepdims=True), axis=1)).mean(0) - np.abs(np.fft.rfft(preds[b] - preds[b].mean(1, keepdims=True), axis=1)).mean(0))))))
    cand = {r["model_b"]: r for r in sim if r["model_a"] == "R"}
    votes = {m: 0 for m in cand}
    for k in ("d_amp", "d_morph", "d_hf"):
        votes[min(cand, key=lambda m: cand[m][k])] += 1
    closest = {"rmse_closest": min(cand, key=lambda m: cand[m]["rmse"]), "pcc_closest": max(cand, key=lambda m: cand[m]["pcc"]), "stat_votes": votes}
    closest["closest_O1_to_R"] = bool(closest["rmse_closest"] == "O1" and votes["O1"] >= 2)
    # ---- inversion
    S = summ
    ranks = {"rmse": sorted(MAIN, key=lambda m: S[m]["rmse"]), "rmse_peak": sorted(MAIN, key=lambda m: S[m]["rmse_peak"]), "rmse_nonpeak": sorted(MAIN, key=lambda m: S[m]["rmse_nonpeak"]), "morph_corr": sorted(MAIN, key=lambda m: -np.nan_to_num(S[m]["morph_corr"])), "pp_fidelity": sorted(MAIN, key=lambda m: abs(np.nan_to_num(S[m]["pp_ratio"]) - 1)), "sbp_win_ae": sorted(MAIN, key=lambda m: S[m]["sbp_win_ae"])}
    best_rmse = ranks["rmse"][0]
    inversion = bool(best_rmse in ("R", "O1") and (ranks["morph_corr"][-1] == best_rmse or ranks["pp_fidelity"][-1] == best_rmse))
    # ---- recovery
    rec = {"morph_corr": recovery(np.nan_to_num(S["M1"]["morph_corr"]), np.nan_to_num(S["O1"]["morph_corr"]), np.nan_to_num(S["O50"]["morph_corr"])), "pp_fidelity": recovery(-abs(np.nan_to_num(S["M1"]["pp_ratio"]) - 1), -abs(np.nan_to_num(S["O1"]["pp_ratio"]) - 1), -abs(np.nan_to_num(S["O50"]["pp_ratio"]) - 1)), "sharpness": recovery(-abs(np.nan_to_num(S["M1"]["slope_ratio"]) - 1), -abs(np.nan_to_num(S["O1"]["slope_ratio"]) - 1), -abs(np.nan_to_num(S["O50"]["slope_ratio"]) - 1))}
    n_rec = sum(1 for v in rec.values() if v is not None and v >= 0.5)
    # ---- verdict terms
    def att(m):
        return bool(np.nan_to_num(S[m]["morph_corr"]) < np.nan_to_num(S["O50"]["morph_corr"]) - 0.10 and abs(np.nan_to_num(S[m]["pp_ratio"]) - 1) > abs(np.nan_to_num(S["O50"]["pp_ratio"]) - 1) + 0.15 and np.nan_to_num(S[m]["slope_ratio"]) < np.nan_to_num(S["O50"]["slope_ratio"]) - 0.15)

    terms = {"attenuation_O1": att("O1"), "attenuation_R": bool(att("R") and S["R"]["rmse"] <= S["O50"]["rmse"]), "closest_O1_to_R": closest["closest_O1_to_R"], "recovery_iMF1": bool(n_rec >= 2), "inversion": inversion, "H7_1": att("O1"), "H7_2": bool(att("R") and closest["closest_O1_to_R"]), "H7_3": bool(n_rec >= 2), "H7_4": inversion}
    others = [terms[k] for k in ("attenuation_R", "closest_O1_to_R", "recovery_iMF1", "inversion")]
    verdict = "STRONG CROSS-TARGET SUPPORT" if terms["attenuation_O1"] and all(others) else "PARTIAL CROSS-TARGET SUPPORT" if terms["attenuation_O1"] and sum(others) >= 2 else "NOT GENERALIZED"
    # ---- pareto
    par = []
    for name in preds:
        par.append({"arm": name, "nfe": info[name]["nfe"], "generative_nfe": name != "R", "latency_ms": info[name]["latency_ms"], "sbp_win_ae": S[name]["sbp_win_ae"], "dbp_win_ae": S[name]["dbp_win_ae"], "morph_corr": S[name]["morph_corr"], "pp_abs_dev": abs(np.nan_to_num(S[name]["pp_ratio"]) - 1), "slope_ratio": S[name]["slope_ratio"], "rmse": S[name]["rmse"]})
    for p in par:
        dom = [q for q in par if q is not p and q["nfe"] <= p["nfe"] and q["sbp_win_ae"] <= p["sbp_win_ae"] and np.nan_to_num(q["morph_corr"]) >= np.nan_to_num(p["morph_corr"]) and q["pp_abs_dev"] <= p["pp_abs_dev"] and (q["nfe"] < p["nfe"] or q["sbp_win_ae"] < p["sbp_win_ae"] or np.nan_to_num(q["morph_corr"]) > np.nan_to_num(p["morph_corr"]) or q["pp_abs_dev"] < p["pp_abs_dev"])]
        p["pareto_optimal"] = not dom
        p["dominated_by"] = ";".join(q["arm"] for q in dom)
    # ---- write
    with open(out / "abp_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm"] + list(next(iter(summ.values())).keys()))
        w.writeheader()
        for name in preds:
            w.writerow({"arm": name, **summ[name]})
    for fname, rows in (("cross_model_similarity.csv", sim), ("pareto.csv", par)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    peak_rows = [{"arm": m, "rmse_all": S[m]["rmse"], "rmse_peak": S[m]["rmse_peak"], "rmse_nonpeak": S[m]["rmse_nonpeak"], "peak_over_nonpeak": S[m]["rmse_peak"] / max(S[m]["rmse_nonpeak"], 1e-9), "peak_region_energy_ratio": S[m]["peak_region_energy_ratio"], "slope_ratio": S[m]["slope_ratio"], "peak_amp_ae": S[m]["peak_amp_ae"]} for m in preds]
    with open(out / "peak_region_analysis.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(peak_rows[0].keys()))
        w.writeheader()
        w.writerows(peak_rows)
    summary = {"n_windows": int(n), "peak_region_ms": PEAK_REGION_MS, "metrics": summ, "similarity": sim, "closest": closest, "ranks": ranks, "inversion": inversion, "recovery": rec, "n_recovered_ge50": n_rec, "terms": terms, "verdict": verdict, "training": {arm: json.loads((ROOT / RUNS[arm] / "training_summary.json").read_text()) for arm in RUNS}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # ---- figures: deterministic windows = 10/50/90 pct of OT-50 mean(SBP,DBP window AE)
    score = 0.5 * (pw_all["O50"]["sbp_win_ae"] + pw_all["O50"]["dbp_win_ae"])
    order = np.argsort(score)
    wins = [int(order[int(q * (n - 1))]) for q in (0.10, 0.50, 0.90)]
    rows_fig = [("R", "MSE proxy (1 fwd)"), ("O50", "OT-CFM 50 NFE"), ("O4", "OT-CFM 4 NFE"), ("O1", "OT-CFM 1 NFE"), ("M1", "iMeanFlow 1 NFE")]
    t = np.arange(y.shape[1]) / FS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(2 + len(rows_fig), len(wins), figsize=(6.2 * len(wins), 1.9 * (2 + len(rows_fig))), sharex=True)
        for c, w in enumerate(wins):
            axes[0, c].plot(t, x[w], color="tab:green", lw=0.9)
            axes[0, c].set_title(f"{ti['sid'][w]} window {w} · OT-50 mean(SBP,DBP) AE {score[w]:.1f} mmHg", fontsize=9)
            prom = gt_prominence(y[w])
            pk = systolic_peaks(y[w], prom)
            lo, hi = y[w].min() - 15, y[w].max() + 15
            axes[1, c].plot(t, y[w], "k", lw=0.9)
            axes[1, c].plot(pk / FS, y[w][pk], "r.", ms=6)
            axes[1, c].set_ylim(lo, hi)
            for rr, (m, _) in enumerate(rows_fig, start=2):
                p = preds[m][w]
                pp = systolic_peaks(p, prom)
                axes[rr, c].plot(t, y[w], color="0.8", lw=0.6)
                for r0 in pk:
                    axes[rr, c].axvspan((r0 - PEAK_REGION_MS / 1000 * FS) / FS, (r0 + PEAK_REGION_MS / 1000 * FS) / FS, color="0.93", zorder=0)
                axes[rr, c].plot(t, p, color=COLORS[m], lw=0.9)
                axes[rr, c].plot(pp / FS, p[pp], "r.", ms=6)
                axes[rr, c].set_ylim(lo, hi)
            for rr in range(2 + len(rows_fig)):
                axes[rr, c].grid(alpha=0.25)
            axes[-1, c].set_xlabel("time (s)")
        for rr, name in enumerate(["PPG (input, norm.)", "GT ABP (mmHg)"] + [lab for _, lab in rows_fig]):
            axes[rr, 0].set_ylabel(name, fontsize=8)
        fig.suptitle("MIMIC-BP test (official split, 4096-window subset): same PPG, same paired noise; y in mmHg (grey = GT, shaded = ±150 ms peak regions)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figures" / "a7_examples.png", dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k, lab in zip(axes, ("sbp_win_ae", "morph_corr", "pp_abs_dev"), ("SBP MAE (mmHg) ↓", "pulse-template correlation ↑", "|PP ratio − 1| ↓")):
        for fam, mk, col, names in (("OT-CFM", "o-", "tab:blue", ["O1", "O2", "O4", "O10", "O20", "O50"]), ("iMeanFlow", "s-", "tab:red", ["M1", "M2", "M4"]), ("MSE proxy (1 fwd)", "D", "tab:purple", ["R"])):
            ps = [p for p in par if p["arm"] in names]
            ax.plot([p["nfe"] for p in ps], [np.nan_to_num(p[k]) for p in ps], mk, color=col, label=fam, ms=6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("network evaluations")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("MIMIC-BP: quality vs compute", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "a7_pareto.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ms = MAIN
    wdt = 0.38
    ax.bar(np.arange(len(ms)) - wdt / 2, [S[m]["rmse_peak"] for m in ms], wdt, label="peak region ±150 ms", color=[COLORS[m] for m in ms])
    ax.bar(np.arange(len(ms)) + wdt / 2, [S[m]["rmse_nonpeak"] for m in ms], wdt, label="non-peak", color=[COLORS[m] for m in ms], alpha=0.45)
    ax.set_xticks(range(len(ms)), ms)
    ax.set_ylabel("RMSE (mmHg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out / "figures" / "a7_peak_region.png", dpi=110)
    plt.close(fig)
    print("closest:", closest, "| recovery:", rec, "| inversion:", inversion, "| terms:", terms)
    print("VERDICT:", verdict, "| wrote", out)


if __name__ == "__main__":
    main()
