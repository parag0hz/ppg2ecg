"""A2 controlled comparison: OT-CFM (A0-b) 50 / 4 / 1 NFE vs iMeanFlow 1 NFE. Recovery scores, verdict (prereg Sec. 5-6),
figures (PPG / GT / OT-CFM 50, 4, 1 NFE / iMF 1 NFE on A0's deterministic example windows) and docs/A2_IMEANFLOW_REPORT.md."""
from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path

import matplotlib
import neurokit2 as nk
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.dalia import load_subject_raw  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 128
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
P_METRICS = [("hr_abs_err_bpm", "HR error (bpm)", "lower"), ("morph_corr", "morph corr", "higher"), ("amp_ratio", "amplitude ratio", "amp"), ("cond_gain_bpm", "conditioning gain (bpm)", "higher")]
OTHER = [("rmse", "RMSE", "lower"), ("mae", "MAE", "lower"), ("beats_ratio", "beats / reference", "info"), ("seed_std_mean", "seed diversity (std)", "info"), ("hf_ratio_pred", "HF-energy ratio", "info"), ("pcc", "PCC (diag.)", "info"), ("rpeak_f1", "R-peak F1 (diag.)", "info"), ("hr_err_penguin_corrected", "upstream HR err (corrected)", "info"), ("latency_ms_batch64", "latency ms / batch 64", "info"), ("actual_NFE", "actual NFE", "info")]


def load_rows(d):
    rows = {}
    for r in csv.DictReader(open(d / "nfe_curve.csv")):
        key = f"{r['solver']}{r['solver_steps']}"
        row = {k: (float(v) if k != "solver" and v != "" else v) for k, v in r.items()}
        if "beats_ratio" not in row:
            p = np.load(d / "predictions" / f"{key}.npz")
            row["beats_ratio"] = float(p["pw_n_pred_beats"].mean() / max(p["pw_n_ref_beats"].mean(), 1e-9))
        rows[key] = row
    return rows


def replication_verdict(rec, o1, i1):
    """A3/A4 rule (docs/A3_A4_REPLICATION_PREREGISTRATION.md §4)."""
    better = {"hr_abs_err_bpm": i1["hr_abs_err_bpm"] < o1["hr_abs_err_bpm"], "morph_corr": i1["morph_corr"] > o1["morph_corr"], "amp_ratio": abs(i1["amp_ratio"] - 1) < abs(o1["amp_ratio"] - 1), "cond_gain_bpm": i1["cond_gain_bpm"] > o1["cond_gain_bpm"]}
    n_better = sum(better.values())
    severe = any(np.isfinite(v) and v < -0.25 for v in rec.values())
    collapse = i1["amp_ratio"] < 0.5 and i1["beats_ratio"] < 0.7
    if n_better >= 3 and not severe:
        v = "REPLICATED"
    elif n_better <= 1 or collapse:
        v = "NOT REPLICATED"
    else:
        v = "PARTIAL"
    return v, better, severe, collapse


def recovery(m, kind, o50, o1, i1):
    if kind == "higher":
        den = o50 - o1
        return (i1 - o1) / den if abs(den) > 1e-9 else float("nan")
    if kind == "lower":
        den = o1 - o50
        return (o1 - i1) / den if abs(den) > 1e-9 else float("nan")
    if kind == "amp":
        d1 = abs(o1 - o50)
        return 1 - abs(i1 - o50) / d1 if d1 > 1e-9 else float("nan")
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--otcfm", default="outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--imf", default="outputs/a2_imeanflow_s5_ppgdalia_8s_seed42")
    ap.add_argument("--a0", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42", help="source of the deterministic example windows")
    ap.add_argument("--report", default="docs/A2_IMEANFLOW_REPORT.md")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--dataset-label", default="PPG-DaLiA")
    ap.add_argument("--title", default="A2 Improved MeanFlow Report")
    ap.add_argument("--prereg", default="docs/A2_IMEANFLOW_PREREGISTRATION.md")
    args = ap.parse_args()
    from ppg2ecg.data.splits import read_manifest

    split = read_manifest(ROOT / args.manifest)[0]
    test_subject = split["test"][0]
    d_ot, d_imf, d_a0 = ROOT / args.otcfm, ROOT / args.imf, ROOT / args.a0
    o, i = load_rows(d_ot), load_rows(d_imf)
    o50, o4, o1, i1 = o["heun25"], o["heun2"], o["euler1"], i["meanflow1"]
    arms = [("OT-CFM", "Heun 25", o50), ("OT-CFM", "Heun 2", o4), ("OT-CFM", "Euler 1", o1), ("iMeanFlow", "1 step", i1)] + [("iMeanFlow", f"{k} steps", i[f"meanflow{k}"]) for k in (2, 4) if f"meanflow{k}" in i]
    rec = {m: recovery(m, kind, o50[m], o1[m], i1[m]) for m, _, kind in P_METRICS}
    rec_other = {m: recovery(m, kind, o50[m], o1[m], i1[m]) for m, _, kind in OTHER if kind in ("lower", "higher")}
    beats_ok = i1["beats_ratio"] >= 0.7
    n_ge50 = sum(1 for v in rec.values() if np.isfinite(v) and v >= 0.5)
    all_lt25 = all((not np.isfinite(v)) or v < 0.25 for v in rec.values())
    gain_fail = (i1["cond_gain_bpm"] <= 0.25 * o50["cond_gain_bpm"]) and (rec["hr_abs_err_bpm"] < 0.25)
    summ_i = json.loads((d_imf / "training_summary.json").read_text())
    unstable = summ_i.get("early_stopped") is None
    if n_ge50 == len(rec) and beats_ok:
        verdict = "SUCCESS"
    elif all_lt25 or gain_fail or unstable:
        verdict = "FAIL"
    else:
        verdict = "PARTIAL"
    rep_verdict, better, severe, collapse = replication_verdict(rec, o1, i1)
    inversion = bool(o1["rmse"] < o50["rmse"] and o1["rmse"] < i1["rmse"] and o1["mae"] < o50["mae"])  # pointwise-error inversion
    res = {"arms": {f"{a} {b}": r for a, b, r in arms}, "recovery": rec, "recovery_other": rec_other, "beats_ratio_imf1": i1["beats_ratio"], "beats_ok": beats_ok, "n_metrics_recovered_ge50": n_ge50, "all_lt25": all_lt25, "gain_fail_rule": gain_fail, "verdict": verdict, "replication_verdict": rep_verdict, "replication_better": better, "replication_severe_negative": severe, "replication_collapse": collapse, "pointwise_error_inversion": inversion, "test_subject": test_subject, "dataset": args.dataset_label, "training_imf": summ_i, "training_otcfm": json.loads((d_ot / "training_summary.json").read_text())}
    (d_imf / "recovery.json").write_text(json.dumps(res, indent=1, default=str))

    # ---------------- figures: A0's deterministic example windows
    met_a0 = json.loads((d_a0 / "metrics.json").read_text())
    idxs = list(met_a0["examples"]["ref_arm_hr_err_quantiles_10_50_90"]) + list(met_a0["examples"]["fixed_positions"])[:3]
    ti = d_imf / "predictions" / "test_inputs.npz"
    if ti.exists():  # exact arrays the arms were evaluated on (subsampled / multi-subject)
        d = np.load(ti, allow_pickle=True)
        x, y, starts = d["x"], d["y"], d["starts"]
        sids = d["sid"]
    else:
        d = np.load(ROOT / args.processed / f"{test_subject}.npz")
        x, y, starts = d["x"], d["y"], d["window_start_s"]
        sids = np.array([test_subject] * len(x))
    try:
        raw = load_subject_raw(ROOT / "data/raw", test_subject)
        act_fs = len(raw.activity) / raw.ecg_seconds
    except Exception:  # noqa: BLE001  (non-DaLiA datasets have no activity labels)
        raw, act_fs = None, None
    preds = {"OT-CFM 50 NFE": np.load(d_ot / "predictions/heun25.npz")["pred"], "OT-CFM 4 NFE": np.load(d_ot / "predictions/heun2.npz")["pred"], "OT-CFM 1 NFE": np.load(d_ot / "predictions/euler1.npz")["pred"], "iMeanFlow 1 NFE": np.load(d_imf / "predictions/meanflow1.npz")["pred"]}
    t = np.arange(x.shape[1]) / FS
    (d_imf / "figures").mkdir(exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tag, wins in (("quantile", idxs[:3]), ("fixed", idxs[3:])):
            fig, axes = plt.subplots(2 + len(preds), len(wins), figsize=(6.2 * len(wins), 2.0 * (2 + len(preds))), sharex=True, sharey="row")
            axes = np.atleast_2d(axes).reshape(2 + len(preds), len(wins))
            for c, w in enumerate(wins):
                if raw is not None:
                    seg = raw.activity[int(round(starts[w] * act_fs)) : int(round((starts[w] + 8) * act_fs))].astype(int)
                    a = ACT.get(int(np.bincount(seg).argmax()), "?") if len(seg) else "?"
                else:
                    a = ""
                try:
                    pp = np.asarray(nk.ppg_findpeaks(nk.ppg_clean(x[w].astype(float), sampling_rate=FS), sampling_rate=FS)["PPG_Peaks"], dtype=int)
                except Exception:  # noqa: BLE001
                    pp = np.zeros(0, int)
                axes[0, c].plot(t, x[w], color="tab:green", lw=0.9)
                axes[0, c].plot(pp / FS, x[w][pp], "v", color="darkgreen", ms=5)
                axes[0, c].set_title(f"{sids[w]} window {w} · t = {starts[w]} s · {a}", fontsize=10)
                rg = R.detect_rpeaks(y[w], FS)
                axes[1, c].plot(t, y[w], "k", lw=0.9)
                axes[1, c].plot(rg / FS, y[w][rg], "r.", ms=6)
                for rr, (name, P) in enumerate(preds.items(), start=2):
                    p = P[w]
                    rp = R.detect_rpeaks(p, FS)
                    axes[rr, c].plot(t, y[w], color="0.8", lw=0.6)
                    axes[rr, c].plot(t, p, color="tab:red" if "iMean" in name else "tab:blue", lw=0.9)
                    axes[rr, c].plot(rp / FS, p[rp], "r.", ms=6)
                    axes[rr, c].set_ylim(-1.6, 1.6)
                for rr in range(2 + len(preds)):
                    axes[rr, c].grid(alpha=0.25)
                axes[-1, c].set_xlabel("time (s)")
            for rr, name in enumerate(["PPG (input)", "GT ECG"] + list(preds)):
                axes[rr, 0].set_ylabel(name)
            fig.suptitle("Same PPG, same initial noise (seed 0): OT-CFM at 50 / 4 / 1 NFE vs Improved MeanFlow at 1 NFE (grey = GT); identical y-scale", fontsize=12)
            fig.tight_layout()
            fig.savefig(d_imf / "figures" / f"controlled_examples_{tag}.png", dpi=110)
            plt.close(fig)
    # recovery bar chart
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labs = [lab for _, lab, _ in P_METRICS]
    vals = [rec[m] for m, _, _ in P_METRICS]
    ax.bar(range(len(labs)), vals, color=["tab:green" if v >= 0.5 else "tab:orange" if v >= 0.25 else "tab:red" for v in vals])
    ax.axhline(0.5, ls="--", color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel("recovery of the 50→1 NFE gap")
    ax.set_title(f"iMeanFlow 1 NFE: recovery scores (verdict {verdict})")
    fig.tight_layout()
    fig.savefig(d_imf / "figures" / "recovery.png", dpi=110)
    plt.close(fig)

    # ---------------- report
    analysis = {}
    if (d_imf / "analysis.md").exists():
        for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", (d_imf / "analysis.md").read_text(), re.S | re.M):
            analysis[m.group(1).strip()] = m.group(2).strip()
    def nar(k, default="_(to be written after results are inspected)_"):
        return analysis.get(k, default)
    L = [f"# {args.title}", "", f"Generated from `{args.imf}/` vs `{args.otcfm}/` — dataset {args.dataset_label}, test subject(s) {split['test']}, val {split['val']}. Pre-registration: `{args.prereg}`; audit: `docs/IMEANFLOW_AUDIT.md`.", ""]
    L += ["## Research question", "> Can Improved MeanFlow make the long noise→ECG transport jump in one network evaluation while preserving the physiological structure that OT-CFM needs many evaluations to generate?", ""]
    L += ["## Frozen protocol", "Identical to A0-b (data, 8 s windows, split, seed 42, backbone with 4,568,707 parameters, PPG conditioning, AdamW 1e-3 / wd 0.01 / effective batch 64, fp32, patience 20 / min_delta 1e-4 on a deterministic fixed-bank metric). Only the objective/parameterisation changed: OT-CFM → Improved MeanFlow (`V = u + (t−r)·sg(du/dt)`, v-loss with adaptive weighting, (t,r) logit-normal(−0.4,1), 50 % r=t, boundary v_θ, conditioning E(t)+E(h) via the backbone's single embedder). Gradient accumulation 2 × 32 for memory (prereg §8).", ""]
    L += ["## iMeanFlow paper/code audit", "See `docs/IMEANFLOW_AUDIT.md` (papers arXiv:2505.13447 / arXiv:2512.02012 v2; official code `Lyy-iiis/imeanflow` @ bf60cd7, submodule `external/iMeanFlow`).", ""]
    L += ["## Implementation parity tests", "`tests/test_imeanflow.py`: analytic linear-field MeanFlow identity (V ≡ v), zero loss for consistent pairs, shapes/conditioning/batch independence, backbone parity (t-only mode == upstream forward_step bit-exact), JVP vs finite differences and vs double-VJP on the backbone, stop-gradient equivalence, finite loss/grads, seed determinism, 1-NFE call count, and a JAX port of the official objective evaluated with identical weights (loss and V agree to 1e-5). Independent adversarial review: see EXPERIMENT_LOG.", ""]
    L += ["## Training", f"- iMF: {summ_i['epochs_run']} epochs, best epoch {int(summ_i['best_epoch'])+1}, early stopped {summ_i['early_stopped']}, {float(summ_i['total_train_time_s'])/3600:.2f} h, peak {float(summ_i['peak_mem_MiB'])/1024:.1f} GiB, selection metric {float(summ_i['best_selection_metric']):.5f} (fixed-bank iMF MSE)", f"- OT-CFM (A0-b): {res['training_otcfm']['epochs_run']} epochs, best {int(res['training_otcfm']['best_epoch'])+1}, {float(res['training_otcfm']['total_train_time_s'])/3600:.2f} h, peak {float(res['training_otcfm']['peak_mem_MiB'])/1024:.1f} GiB", ""]
    L += ["## Memory/runtime", "Forward-mode JVP: ≈ 0.51 GiB per sample at T = 1024 (OT-CFM 0.29) → micro-batch 32 × 2 accumulation; training step ≈ 2 × 250 ms; 1-NFE sampling latency in the table below.", ""]
    L += ["## Main controlled comparison", "| Model | Sampler | Actual NFE | HR Error (bpm) | Morph corr | Amp ratio | Cond gain (bpm) | RMSE | MAE | beats/ref | seed std | Latency (ms/batch 64) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a, b, r in arms:
        L.append(f"| {a} | {b} | {int(r['actual_NFE'])} | {r['hr_abs_err_bpm']:.2f} | {r['morph_corr']:.3f} | {r['amp_ratio']:.3f} | {r['cond_gain_bpm']:.2f} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['beats_ratio']:.2f} | {r['seed_std_mean']:.3f} | {r['latency_ms_batch64']:.0f} |")
    L += ["", "Secondary diagnostics (absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion):", "| Model | Sampler | PCC | R-peak F1 | RR MAE (ms) | QRS err (ms) | HF ratio (target 0.32) | upstream HR err corrected |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for a, b, r in arms:
        L.append(f"| {a} | {b} | {r['pcc']:.3f} | {r['rpeak_f1']:.3f} | {r['rr_mae_ms']:.1f} | {r['qrs_width_err_ms']:.1f} | {r['hf_ratio_pred']:.3f} | {r['hr_err_penguin_corrected']:.2f} |")
    L += ["", "## 1-NFE physiological recovery", "Recovery = fraction of the OT-CFM 50→1 NFE gap recovered by iMeanFlow at 1 NFE (prereg §5).", "| metric | OT-CFM 50 | OT-CFM 1 | iMeanFlow 1 | recovery |", "|---|---:|---:|---:|---:|"]
    for m, lab, _ in P_METRICS:
        L.append(f"| {lab} | {o50[m]:.3f} | {o1[m]:.3f} | {i1[m]:.3f} | **{rec[m]:+.2f}** |")
    for m, lab, kind in OTHER:
        if kind in ("lower", "higher"):
            L.append(f"| {lab} (aux.) | {o50[m]:.3f} | {o1[m]:.3f} | {i1[m]:.3f} | {rec_other[m]:+.2f} |")
    L += [f"| beats / reference | {o50['beats_ratio']:.2f} | {o1['beats_ratio']:.2f} | {i1['beats_ratio']:.2f} | {'≥ 0.7 ✔' if beats_ok else '< 0.7 ✘'} |", "", f"Figure: `{args.imf}/figures/recovery.png`.", ""]
    L += ["## Conditional fidelity", f"PPG-shuffle test (same noise, PPG replaced by a deranged window): HR error vs the *right* target / vs the *wrong* target — OT-CFM 50 NFE {o50['hr_err_shuffled_right_target']:.2f} / {o50['hr_err_shuffled_wrong_target']:.2f} (gain {o50['cond_gain_bpm']:.2f}); OT-CFM 1 NFE {o1['hr_err_shuffled_right_target']:.2f} / {o1['hr_err_shuffled_wrong_target']:.2f} (gain {o1['cond_gain_bpm']:.2f}); iMeanFlow 1 NFE {i1['hr_err_shuffled_right_target']:.2f} / {i1['hr_err_shuffled_wrong_target']:.2f} (gain **{i1['cond_gain_bpm']:.2f}**).", "", nar("Conditional fidelity"), ""]
    L += ["## Qualitative examples", f"A0's deterministic windows (HR-error quantiles 10/50/90 % of the 50-NFE arm and fixed positions): `{args.imf}/figures/controlled_examples_quantile.png`, `controlled_examples_fixed.png` — same PPG, same initial noise, identical y-scale.", "", nar("Qualitative examples"), ""]
    L += ["## Failure taxonomy", nar("Failure taxonomy"), ""]
    L += ["## Limitations", nar("Limitations", "- single seed / single test subject; iMF trained with the baseline optimiser (Adam 1e-4 + EMA in the official recipe) — a deliberate isolation choice; beat-level alignment metrics not interpretable on raw PPG-DaLiA; boundary-condition v_θ instead of the official auxiliary head (parameter-count constraint)."), ""]
    L += ["## GO / PARTIAL / FAIL", f"**{verdict}** (A2 recovery rule) — recovery ≥ 0.5 on {n_ge50}/{len(rec)} physiological metrics; beats/reference {i1['beats_ratio']:.2f} ({'ok' if beats_ok else 'below 0.7'}); all < 0.25: {all_lt25}; gain-fail rule: {gain_fail}.", f"**Replication rule (A3/A4 §4): {rep_verdict}** — iMF-1 better than OT-CFM-1 on {sum(better.values())}/4 of {list(better)}; severe negative recovery: {severe}; collapse signature: {collapse}.", f"**Pointwise-error inversion** (OT-CFM-1 has the best RMSE/MAE while physiology collapses): {'YES' if inversion else 'NO'} (RMSE OT-50 {o50['rmse']:.3f}, OT-1 {o1['rmse']:.3f}, iMF-1 {i1['rmse']:.3f}).", "", nar("Verdict rationale"), ""]
    L += ["## Recommended next research question", nar("Recommended next research question"), ""]
    (ROOT / args.report).write_text("\n".join(L))
    print(json.dumps({"recovery": rec, "recovery_other": rec_other, "beats_ratio": i1["beats_ratio"], "verdict": verdict, "replication_verdict": rep_verdict, "better": better, "pointwise_error_inversion": inversion}, indent=1))
    print("wrote", d_imf / "recovery.json", ROOT / args.report)


if __name__ == "__main__":
    main()
