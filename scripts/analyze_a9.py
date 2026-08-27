"""A9 analysis (docs/A9_ECG_TARGET_REPRESENTATION_PREREGISTRATION.md §9-12): window-norm (A4/A6c) vs global-z (A9) controlled
comparison on the same WildPPG test windows, conditional-mean similarity, QRS-region, timing, conditioning, inversion, verdict, figures.
Cross-representation rule: absolute RMSE/MAE are NEVER compared between representations, only rankings within each."""
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

ROOT = Path(__file__).resolve().parents[1]
FS = 128
QRS_HALF_MS = 100.0
REPS = {"window-norm": {"mse": "outputs/a6c_fullbackbone_mse_wildppg_seed42", "ot": "outputs/a4_otcfm_wildppg_seed42", "imf": "outputs/a4_imeanflow_wildppg_seed42"},
        "global-z": {"mse": "outputs/a9_mse_fullbackbone_wildppg_globalz_seed42", "ot": "outputs/a9_otcfm_wildppg_globalz_seed42", "imf": "outputs/a9_imeanflow_wildppg_globalz_seed42"}}
ARMS = [("R", "mse", "regressor", 1), ("O1", "ot", "euler1", 1), ("O2", "ot", "heun1", 2), ("O4", "ot", "heun2", 4), ("O10", "ot", "heun5", 10), ("O20", "ot", "heun10", 20), ("O50", "ot", "heun25", 50), ("M1", "imf", "meanflow1", 1), ("M2", "imf", "meanflow2", 2), ("M4", "imf", "meanflow4", 4)]
MAIN = ["R", "O1", "O50", "M1"]
COLORS = {"R": "tab:purple", "O50": "tab:blue", "O4": "tab:olive", "O1": "tab:cyan", "M1": "tab:red"}
KEYS = ["hr_abs_err_bpm", "morph_corr", "amp_ratio", "amp_ratio_median", "cond_gain_bpm", "beats_ratio", "hf_ratio_pred", "hf_ratio_target", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "qrs_width_err_ms", "rmse", "mae", "pcc", "latency_ms_batch64", "actual_NFE", "frac_windows_no_pred_beats", "hr_err_shuffled_right_target", "hr_err_shuffled_wrong_target"]


def read_rows(d):
    out = {}
    for r in csv.DictReader(open(ROOT / d / "nfe_curve.csv")):
        key = "regressor" if r["solver"] == "regressor" else f"{r['solver']}{int(float(r['solver_steps']))}"
        out[key] = {k: (float(v) if v not in ("", "None", "nan") and k != "solver" else (v if k == "solver" else float("nan"))) for k, v in r.items()}
    return out


def load_rep(rep):
    runs = REPS[rep]
    ti = np.load(ROOT / runs["mse"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    x, y = ti["x"], ti["y"]
    for arm in ("ot", "imf"):
        o = np.load(ROOT / runs[arm] / "predictions" / "test_inputs.npz", allow_pickle=True)
        assert np.array_equal(o["x"], x) and np.array_equal(o["y"], y), (rep, arm)
    preds, rows = {}, {}
    for name, arm, fname, nfe in ARMS:
        p = ROOT / runs[arm] / "predictions" / f"{fname}.npz"
        d = np.load(p, allow_pickle=True)
        preds[name] = d["pred"].astype(np.float32)
        rows[name] = read_rows(runs[arm])[fname]
        if "beats_ratio" not in rows[name] or not np.isfinite(rows[name].get("beats_ratio", float("nan"))):
            rows[name]["beats_ratio"] = float(d["pw_n_pred_beats"].mean() / max(d["pw_n_ref_beats"].mean(), 1e-9))
        rows[name]["nfe"] = nfe
    return ti, x, y, preds, rows


def pcc_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)


def beat_timing_distance(a, b):
    """mean |median RR(a) - median RR(b)| in ms plus the fraction of a's beats matched by b within 50 ms (per window)."""
    tol = int(0.05 * FS)
    dm, match = [], []
    for pa, pb in zip(a[:1024], b[:1024]):
        ra, rb = R.detect_rpeaks(pa, FS), R.detect_rpeaks(pb, FS)
        if len(ra) > 1 and len(rb) > 1:
            dm.append(abs(np.median(np.diff(ra)) - np.median(np.diff(rb))) / FS * 1000)
        if len(ra):
            match.append(np.mean([1.0 if len(rb) and np.abs(rb - r).min() <= tol else 0.0 for r in ra]))
    return float(np.mean(dm)) if dm else float("nan"), float(np.mean(match)) if match else float("nan")


def similarity(preds, S):
    sim = []
    for i, a in enumerate(MAIN):
        for b in MAIN[i + 1 :]:
            d = preds[a] - preds[b]
            tim, mt = beat_timing_distance(preds[a], preds[b])
            sim.append(dict(model_a=a, model_b=b, rmse=float(np.sqrt((d**2).mean(1)).mean()), mae=float(np.abs(d).mean(1).mean()), pcc=float(np.nanmean(pcc_rows(preds[a], preds[b]))),
                            d_morph=abs(np.nan_to_num(S[a]["morph_corr"]) - np.nan_to_num(S[b]["morph_corr"])), d_amp=abs(S[a]["amp_ratio_median"] - S[b]["amp_ratio_median"]),
                            d_hf=abs(S[a]["hf_ratio_pred"] - S[b]["hf_ratio_pred"]), rr_median_diff_ms=tim, beat_match_frac=mt))
    cand = {r["model_b"]: r for r in sim if r["model_a"] == "R"}
    votes = {m: 0 for m in cand}
    for k in ("d_amp", "d_morph", "d_hf"):
        votes[min(cand, key=lambda m: cand[m][k])] += 1
    closest = {"rmse_closest": min(cand, key=lambda m: cand[m]["rmse"]), "pcc_closest": max(cand, key=lambda m: cand[m]["pcc"]), "stat_votes": votes}
    closest["closest_O1_to_R"] = bool(closest["rmse_closest"] == "O1" and votes["O1"] >= 2)
    return sim, closest


def qrs_analysis(preds, y):
    half = int(QRS_HALF_MS / 1000 * FS)
    mask = np.zeros(y.shape, bool)
    peaks = []
    for i in range(len(y)):
        rp = R.detect_rpeaks(y[i], FS)
        peaks.append(rp)
        for r in rp:
            mask[i, max(0, r - half) : r + half + 1] = True
    rows = []
    for m, P in preds.items():
        e2 = (P - y) ** 2
        pk_amp = [abs(P[i][rp]).mean() / (abs(y[i][rp]).mean() + 1e-9) for i, rp in enumerate(peaks) if len(rp)]
        d = np.diff(P, axis=1) * FS
        dgt = np.diff(y, axis=1) * FS
        rows.append(dict(model=m, rmse_all=float(np.sqrt(e2.mean())), rmse_qrs=float(np.sqrt(e2[mask].mean())), rmse_nonqrs=float(np.sqrt(e2[~mask].mean())),
                         qrs_energy_retention=float(P[mask].var() / (y[mask].var() + 1e-9)), nonqrs_energy_retention=float(P[~mask].var() / (y[~mask].var() + 1e-9)),
                         peak_amp_ratio=float(np.median(pk_amp)) if pk_amp else float("nan"),
                         max_slope_ratio=float(np.median(np.abs(d).max(1)) / (np.median(np.abs(dgt).max(1)) + 1e-9))))
    for r in rows:
        r["qrs_over_nonqrs_rmse"] = r["rmse_qrs"] / max(r["rmse_nonqrs"], 1e-9)
    return rows, float(mask.mean())


def attenuation(S, m):
    o50 = S["O50"]
    morph_ok = np.nan_to_num(S[m]["morph_corr"]) < np.nan_to_num(o50["morph_corr"]) - 0.10
    amp_ok = abs(S[m]["amp_ratio_median"] - 1) > abs(o50["amp_ratio_median"] - 1) + 0.10
    hf_ok = S[m]["hf_ratio_pred"] < 0.5 * o50["hf_ratio_pred"]
    return {"morph_below_O50_by_0.10": bool(morph_ok), "amp_worse_by_0.10": bool(amp_ok), "hf_below_half_O50": bool(hf_ok), "attenuation": bool(morph_ok and (amp_ok or hf_ok))}


def recovery(S):
    out = {}
    for name, f, hib in (("morph", lambda s: np.nan_to_num(s["morph_corr"]), True), ("amp_fidelity", lambda s: -abs(s["amp_ratio_median"] - 1), True)):
        o1, o50, m1 = f(S["O1"]), f(S["O50"]), f(S["M1"])
        den = o50 - o1
        out[name] = {"O1": o1, "O50": o50, "M1": m1, "denominator": den, "recovery": (float((m1 - o1) / den) if abs(den) >= 0.05 else None), "improves_over_O1": bool(m1 > o1)}
    out["present"] = bool((out["morph"]["recovery"] is not None and out["morph"]["recovery"] >= 0.5 or (out["morph"]["recovery"] is None and np.nan_to_num(S["M1"]["morph_corr"]) > np.nan_to_num(S["O1"]["morph_corr"]) + 0.10)) and out["amp_fidelity"]["improves_over_O1"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/a9_ecg_representation_control")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data, summary = {}, {"qrs_half_ms": QRS_HALF_MS, "representations": {}}
    for rep in REPS:
        ti, x, y, preds, S = load_rep(rep)
        sim, closest = similarity(preds, S)
        qrs, qfrac = qrs_analysis({k: preds[k] for k in MAIN}, y)
        att = {m: attenuation(S, m) for m in ("R", "O1")}
        rec = recovery(S)
        ranks = {k: sorted(MAIN, key=lambda m: S[m][k]) for k in ("rmse", "mae", "hr_abs_err_bpm")}
        ranks["morph_corr"] = sorted(MAIN, key=lambda m: -np.nan_to_num(S[m]["morph_corr"]))
        ranks["amp_fidelity"] = sorted(MAIN, key=lambda m: abs(S[m]["amp_ratio_median"] - 1))
        ranks["rpeak_f1"] = sorted(MAIN, key=lambda m: -S[m]["rpeak_f1"])
        best = ranks["rmse"][0]
        ranks["inversion"] = bool(ranks["morph_corr"][-1] == best or ranks["amp_fidelity"][-1] == best)
        data[rep] = dict(ti=ti, x=x, y=y, preds=preds, S=S)
        summary["representations"][rep] = {"n_windows": int(len(y)), "metrics": {m: {k: S[m].get(k) for k in KEYS} for m in S}, "similarity": sim, "closest": closest,
                                           "qrs": qrs, "qrs_fraction_of_samples": qfrac, "attenuation": att, "recovery": rec, "ranks": ranks,
                                           "timing": {m: {k: S[m][k] for k in ("rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "beats_ratio")} for m in MAIN},
                                           "conditioning": {m: {"cond_gain_bpm": S[m]["cond_gain_bpm"], "hr_right": S[m]["hr_err_shuffled_right_target"], "hr_wrong": S[m]["hr_err_shuffled_wrong_target"]} for m in MAIN}}
        print(f"[{rep}]")
        for m in MAIN:
            s = S[m]
            print(f"  {m:3s} NFE {int(s['nfe']):2d} HR {s['hr_abs_err_bpm']:6.2f} morph {np.nan_to_num(s['morph_corr']):.3f} ampMed {s['amp_ratio_median']:.2f} gain {s['cond_gain_bpm']:5.2f} beats {s['beats_ratio']:.2f} HF {s['hf_ratio_pred']:.3f} F1 {s['rpeak_f1']:.3f} RR {s['rr_mae_ms']:5.1f} RMSE* {s['rmse']:.3f}")
        print("   attenuation:", {k: v["attenuation"] for k, v in att.items()}, "| closest:", closest, "| recovery:", {k: (v["recovery"] if isinstance(v, dict) else v) for k, v in rec.items() if k != "present"}, "present", rec["present"], "| inversion", ranks["inversion"])
    W, G = summary["representations"]["window-norm"], summary["representations"]["global-z"]
    terms = {"mse_attenuation_persists": G["attenuation"]["R"]["attenuation"], "ot1_attenuation_persists": G["attenuation"]["O1"]["attenuation"],
             "ot1_closest_to_mse_proxy": G["closest"]["closest_O1_to_R"], "imf_recovery_present": G["recovery"]["present"],
             "window_norm_reference": {"mse_attenuation": W["attenuation"]["R"]["attenuation"], "ot1_attenuation": W["attenuation"]["O1"]["attenuation"], "ot1_closest": W["closest"]["closest_O1_to_R"], "imf_recovery": W["recovery"]["present"], "inversion": W["ranks"]["inversion"]},
             "inversion_global_z": G["ranks"]["inversion"]}
    if all(terms[k] for k in ("mse_attenuation_persists", "ot1_attenuation_persists", "ot1_closest_to_mse_proxy", "imf_recovery_present")):
        verdict = "REPRESENTATION-ROBUST"
    elif not terms["mse_attenuation_persists"] and not terms["ot1_attenuation_persists"]:
        verdict = "REPRESENTATION-SENSITIVE"
    else:
        verdict = "MIXED"
    summary["terms"], summary["verdict"] = terms, verdict
    summary["training"] = {rep: {a: json.loads((ROOT / REPS[rep][a] / "training_summary.json").read_text()) for a in REPS[rep]} for rep in REPS}
    # CSVs
    with open(out / "controlled_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["representation", "arm"] + KEYS)
        w.writeheader()
        for rep in REPS:
            for m, v in summary["representations"][rep]["metrics"].items():
                w.writerow({"representation": rep, "arm": m, **v})
    with open(out / "representation_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "metric", "window_norm", "global_z", "delta"])
        w.writeheader()
        for m in MAIN:
            for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio_median", "cond_gain_bpm", "beats_ratio", "hf_ratio_pred", "rpeak_f1", "rr_mae_ms"):
                a, b = W["metrics"][m][k], G["metrics"][m][k]
                w.writerow({"arm": m, "metric": k, "window_norm": a, "global_z": b, "delta": (b - a) if (a is not None and b is not None) else None})
    for name, key in (("prediction_similarity.csv", "similarity"), ("qrs_region_analysis.csv", "qrs")):
        rows = [{"representation": rep, **r} for rep in REPS for r in summary["representations"][rep][key]]
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for name, key in (("timing_analysis.csv", "timing"), ("conditioning_analysis.csv", "conditioning")):
        rows = [{"representation": rep, "arm": m, **v} for rep in REPS for m, v in summary["representations"][rep][key].items()]
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # figure: the frozen A4 example windows, both representations (each panel on its own y-scale)
    met_ex = json.loads((ROOT / "outputs/a4_otcfm_wildppg_seed42" / "metrics.json").read_text())["examples"]
    wins = list(met_ex["ref_arm_hr_err_quantiles_10_50_90"])
    t = np.arange(data["window-norm"]["y"].shape[1]) / FS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(2 + 2 * len(MAIN), len(wins), figsize=(6.2 * len(wins), 1.35 * (2 + 2 * len(MAIN))), sharex=True)
        for c, wdx in enumerate(wins):
            axes[0, c].plot(t, data["window-norm"]["x"][wdx], color="tab:green", lw=0.9)
            axes[0, c].set_title(f"{data['window-norm']['ti']['sid'][wdx]} window {wdx} · t = {data['window-norm']['ti']['starts'][wdx]} s", fontsize=9)
            row = 1
            for rep in REPS:
                y = data[rep]["y"]
                rg = R.detect_rpeaks(y[wdx], FS)
                if rep == "window-norm":
                    axes[row, c].plot(t, y[wdx], "k", lw=0.9)
                    axes[row, c].plot(rg / FS, y[wdx][rg], "r.", ms=5)
                    axes[row, c].set_ylabel("GT ECG (win-norm)", fontsize=7) if c == 0 else None
                    row += 1
                for m in MAIN:
                    p = data[rep]["preds"][m][wdx]
                    rp = R.detect_rpeaks(p, FS)
                    axes[row, c].plot(t, y[wdx], color="0.8", lw=0.6)
                    axes[row, c].plot(t, p, color=COLORS[m], lw=0.9, ls="-" if rep == "global-z" else "--")
                    axes[row, c].plot(rp / FS, p[rp], "r.", ms=5)
                    axes[row, c].grid(alpha=0.25)
                    if c == 0:
                        axes[row, c].set_ylabel(f"{m} ({rep})", fontsize=7)
                    row += 1
            axes[-1, c].set_xlabel("time (s)")
        fig.suptitle("WildPPG frozen example windows — window-normalised (dashed) vs global-z (solid) ECG target; each row on its own y-scale, representations are NOT y-comparable", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "figures" / "a9_examples_both_representations.png", dpi=110)
        plt.close(fig)
    # figure: representation effect on the key metrics
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, k, lab in zip(axes, ("morph_corr", "amp_ratio_median", "hf_ratio_pred", "rpeak_f1"), ("template correlation ↑", "amplitude ratio (median)", "HF-energy ratio", "R-peak F1 ↑")):
        xpos = np.arange(len(MAIN))
        ax.bar(xpos - 0.2, [np.nan_to_num(W["metrics"][m][k]) for m in MAIN], 0.4, label="window-norm", color="tab:gray")
        ax.bar(xpos + 0.2, [np.nan_to_num(G["metrics"][m][k]) for m in MAIN], 0.4, label="global-z", color=[COLORS[m] for m in MAIN])
        ax.set_xticks(xpos, MAIN)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        if k in ("amp_ratio_median",):
            ax.axhline(1.0, ls="--", color="k", lw=0.8)
        if k == "hf_ratio_pred":
            ax.axhline(G["metrics"]["R"]["hf_ratio_target"], ls="--", color="k", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle("A9: effect of the ECG target representation (WildPPG test, same windows)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "a9_representation_effect.png", dpi=110)
    plt.close(fig)
    print("terms:", json.dumps(terms, default=str))
    print("VERDICT:", verdict, "| wrote", out)


if __name__ == "__main__":
    main()
