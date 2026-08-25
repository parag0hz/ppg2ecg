"""A0 vs A0-b comparison, pre-registered questions (A0B prereg §5) and the mechanical iMeanFlow gate (§6).
Writes <a0b>/comparison.json, <a0b>/figures/a0_vs_a0b.png and docs/A0B_BASELINE_STABILIZATION_REPORT.md."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARGINS = {"hr": 1.0, "morph": 0.05, "amp": 0.5, "gain_frac": 0.5}


def load_curve(d: Path):
    rows = list(csv.DictReader(open(d / "nfe_curve.csv")))
    out = {}
    for r in rows:
        key = f"{r['solver']}{r['solver_steps']}"
        row = {k: (float(v) if k not in ("solver",) else v) for k, v in r.items() if v != ""}
        if "amp_ratio" not in row:  # A0: compute post hoc from saved predictions (no re-sampling)
            p = np.load(d / "predictions" / f"{key}.npz")
            y = np.load(ROOT / "data/processed/v0_8s/S2.npz")["y"][: len(p["pred"])]
            amp = p["pred"].std(axis=1) / (y.std(axis=1) + 1e-8)
            row["amp_ratio"], row["amp_ratio_median"] = float(amp.mean()), float(np.median(amp))
        out[key] = row
    return out


def ci(d: Path, key: str, metric: str):
    m = json.loads((d / "metrics.json").read_text())
    s = m["arms"][key]["summary"].get(metric)
    return tuple(round(v, 3) for v in s["ci95"]) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42")
    ap.add_argument("--a0b", default="outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42")
    args = ap.parse_args()
    A, B = ROOT / args.a0, ROOT / args.a0b
    ca, cb = load_curve(A), load_curve(B)
    sa, sb = json.loads((A / "training_summary.json").read_text()), json.loads((B / "training_summary.json").read_text())
    fa = json.loads((A / "fixed_bank_val.json").read_text())
    la, lb = list(csv.DictReader(open(A / "training_log.csv"))), list(csv.DictReader(open(B / "training_log.csv")))
    best_b = int(sb["best_epoch"])
    vb_best = float(lb[best_b]["val_cfm_fixed"])
    metrics = [("hr_abs_err_bpm", "HR error (bpm)", "lower"), ("morph_corr", "morph corr", "higher"), ("amp_ratio", "amplitude ratio", "higher"), ("cond_gain_bpm", "conditioning gain (bpm)", "higher"), ("rmse", "RMSE", "lower"), ("mae", "MAE", "lower"), ("pcc", "PCC (diag.)", "higher"), ("rpeak_f1", "R-peak F1 (diag.)", "higher"), ("latency_ms_batch64", "latency ms/batch64", "lower")]
    table = []
    for k, lab, _ in metrics:
        table.append({"metric": lab, "A0_50": ca["heun25"][k], "A0b_50": cb["heun25"][k], "A0_1": ca["euler1"][k], "A0b_1": cb["euler1"][k]})
    # --- pre-registered questions
    q1 = {"a0b_best_epoch_1based": best_b + 1, "a0_best_epoch_1based": int(sa["best_epoch"]) + 1, "val_cfm_fixed_a0_ckpt": fa["val_cfm_fixed"], "val_cfm_fixed_a0b_best": vb_best}
    q1["answer_yes"] = bool(best_b + 1 > 21 and vb_best < fa["val_cfm_fixed"] - 1e-4)
    d_hr = cb["heun25"]["hr_abs_err_bpm"] - ca["heun25"]["hr_abs_err_bpm"]
    d_mc = cb["heun25"]["morph_corr"] - ca["heun25"]["morph_corr"]
    q2 = {"delta_hr_50": d_hr, "delta_morph_50": d_mc, "delta_amp_50": cb["heun25"]["amp_ratio"] - ca["heun25"]["amp_ratio"], "delta_gain_50": cb["heun25"]["cond_gain_bpm"] - ca["heun25"]["cond_gain_bpm"], "ci_hr_a0": ci(A, "heun25", "hr_abs_err"), "ci_hr_a0b": ci(B, "heun25", "hr_abs_err"), "ci_morph_a0": ci(A, "heun25", "morph_corr"), "ci_morph_a0b": ci(B, "heun25", "morph_corr"), "changed": bool(abs(d_hr) > MARGINS["hr"] or abs(d_mc) > MARGINS["morph"])}
    r50, r1 = cb["heun25"], cb["euler1"]
    crit = {"hr_rise": r1["hr_abs_err_bpm"] - r50["hr_abs_err_bpm"] > MARGINS["hr"], "morph_drop": r1["morph_corr"] < r50["morph_corr"] - MARGINS["morph"], "amp_collapse": r1["amp_ratio"] < MARGINS["amp"], "gain_loss": r1["cond_gain_bpm"] < MARGINS["gain_frac"] * r50["cond_gain_bpm"]}
    q3 = {"criteria": crit, "collapse_persists": bool(any(crit.values())), "values_1nfe": {k: r1[k] for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio", "cond_gain_bpm", "rmse")}, "values_50nfe": {k: r50[k] for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio", "cond_gain_bpm", "rmse")}}
    q4 = "checkpoint artefact" if not q3["collapse_persists"] else "objective/sampler limitation (gap persists after stabilisation)"
    gate = "GO" if q3["collapse_persists"] else "NO-GO"
    res = {"table": table, "curves": {"A0": ca, "A0b": cb}, "training": {"A0": sa, "A0b": sb}, "q1_under_trained": q1, "q2_50nfe_change": q2, "q3_collapse_persists": q3, "q4_verdict": q4, "gate": gate, "margins": MARGINS}
    (B / "comparison.json").write_text(json.dumps(res, indent=1, default=str))
    # --- figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    order = ["euler1", "heun1", "heun2", "heun5", "heun10", "heun25"]
    for ax, (k, lab, _) in zip(axes.ravel()[:5], metrics[:5]):
        for name, c, mk in (("A0", ca, "o-"), ("A0-b", cb, "s--")):
            ax.plot([c[a]["actual_NFE"] for a in order], [c[a][k] for a in order], mk, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("actual NFE")
        ax.set_title(lab)
        ax.grid(alpha=0.3)
        ax.legend()
    ax = axes.ravel()[5]
    ax.plot([int(r["epoch"]) + 1 for r in lb], [float(r["val_cfm_fixed"]) for r in lb], "s-", ms=3, label="A0-b val_cfm_fixed")
    ax.plot([int(r["epoch"]) + 1 for r in la], [float(r["val_cfm_loss"]) for r in la], "o-", ms=3, label="A0 val CFM (per-epoch noise)")
    ax.axhline(fa["val_cfm_fixed"], color="k", ls=":", label="A0 ckpt on fixed banks")
    ax.axvline(best_b + 1, color="tab:blue", ls=":", alpha=0.6)
    ax.set_xlabel("epoch")
    ax.set_title("validation CFM loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("A0 (stochastic val-MAE selection) vs A0-b (fixed-bank val CFM selection): NFE curves and training", fontsize=12)
    fig.tight_layout()
    (B / "figures").mkdir(exist_ok=True)
    fig.savefig(B / "figures" / "a0_vs_a0b.png", dpi=110)
    plt.close(fig)
    # --- report
    L = ["# A0-b Baseline Stabilisation Report", "", f"Generated from `{args.a0b}/` (and `{args.a0}/`). Pre-registration: `docs/A0B_BASELINE_STABILIZATION_PREREGISTRATION.md`.", ""]
    L += ["## Frozen protocol", "Identical to A0 except checkpoint selection (deterministic fixed-bank validation CFM loss, 4 banks, min_delta 1e-4, patience 20). Seed 42, test S2, val S11, 8 s @ 128 Hz, upstream PENGUIN model class, OT-CFM, AdamW 1e-3 / wd 0.01 / batch 64.", ""]
    L += ["## Training", f"- A0-b: {sb['epochs_run']} epochs, best epoch **{best_b+1}** (val_cfm_fixed {vb_best:.5f}), early stopped: {sb['early_stopped']}, {float(sb['total_train_time_s'])/3600:.2f} h, peak {float(sb['peak_mem_MiB'])/1024:.1f} GiB", f"- A0 : {sa['epochs_run']} epochs, best epoch {int(sa['best_epoch'])+1}, checkpoint scored on the same fixed banks: {fa['val_cfm_fixed']:.5f}", f"- Figure `{args.a0b}/figures/a0_vs_a0b.png` (NFE curves + validation loss curves).", ""]
    L += ["## Main comparison (test S2, paired noise seed 0)", "| metric | A0 50 NFE | A0-b 50 NFE | A0 1 NFE | A0-b 1 NFE |", "|---|---:|---:|---:|---:|"]
    for t in table:
        L.append(f"| {t['metric']} | {t['A0_50']:.3f} | {t['A0b_50']:.3f} | {t['A0_1']:.3f} | {t['A0b_1']:.3f} |")
    L += ["", "Full A0-b NFE curve:", "| Solver | Steps | Actual NFE | HR err (bpm) | morph corr | amp ratio | cond gain (bpm) | RMSE | MAE | PCC* | R-F1* | latency ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ["heun25", "heun10", "heun5", "heun2", "heun1", "euler1"]:
        r = cb[a]
        L.append(f"| {r['solver'].capitalize()} | {int(r['solver_steps'])} | {int(r['actual_NFE'])} | {r['hr_abs_err_bpm']:.2f} | {r['morph_corr']:.3f} | {r['amp_ratio']:.3f} | {r['cond_gain_bpm']:.2f} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['pcc']:.3f} | {r['rpeak_f1']:.3f} | {r['latency_ms_batch64']:.0f} |")
    L += ["", "\\* secondary diagnostics: absolute beat-level temporal correspondence is unreliable under the current PPG-DaLiA protocol, especially under motion.", ""]
    L += ["## Pre-registered questions", f"1. **Was A0 under-trained?** {'YES' if q1['answer_yes'] else 'NO'} — A0-b best epoch {q1['a0b_best_epoch_1based']} (A0: {q1['a0_best_epoch_1based']}); val_cfm_fixed A0-b best {q1['val_cfm_fixed_a0b_best']:.5f} vs A0 checkpoint {q1['val_cfm_fixed_a0_ckpt']:.5f} (rule: best epoch > 21 and improvement > 1e-4).",
          f"2. **50-NFE quality change:** ΔHR {q2['delta_hr_50']:+.2f} bpm (CI A0 {q2['ci_hr_a0']}, A0-b {q2['ci_hr_a0b']}), Δmorph {q2['delta_morph_50']:+.3f} (CI A0 {q2['ci_morph_a0']}, A0-b {q2['ci_morph_a0b']}), Δamp {q2['delta_amp_50']:+.3f}, Δgain {q2['delta_gain_50']:+.2f} bpm → {'CHANGED' if q2['changed'] else 'not changed'} beyond the margins (1.0 bpm / 0.05).",
          f"3. **Does the 1-NFE collapse persist?** {'YES' if q3['collapse_persists'] else 'NO'} — criteria {json.dumps(crit)}; 1 NFE: HR {r1['hr_abs_err_bpm']:.2f} bpm, morph {r1['morph_corr']:.3f}, amp {r1['amp_ratio']:.3f}, gain {r1['cond_gain_bpm']:.2f} vs 50 NFE: HR {r50['hr_abs_err_bpm']:.2f}, morph {r50['morph_corr']:.3f}, amp {r50['amp_ratio']:.3f}, gain {r50['cond_gain_bpm']:.2f}.",
          f"4. **Checkpoint artefact or objective/sampler limitation?** {q4}.", ""]
    L += ["## iMeanFlow gate (mechanical, prereg §6)", f"**{gate}** — {'the 50→1 NFE structural gap persists after stabilisation' if gate == 'GO' else 'stabilised OT-CFM at 1 NFE is within all margins of 50 NFE'}.", ""]
    L += ["## Baseline stabilisation conclusion", "_(narrative in docs/EXPERIMENT_LOG.md and the final status report)_", ""]
    (ROOT / "docs/A0B_BASELINE_STABILIZATION_REPORT.md").write_text("\n".join(L))
    print(json.dumps({"q1": q1, "q2": {k: v for k, v in q2.items() if not k.startswith('ci')}, "q3": q3, "q4": q4, "gate": gate}, indent=1, default=str))
    print("wrote", B / "comparison.json", ROOT / "docs/A0B_BASELINE_STABILIZATION_REPORT.md")


if __name__ == "__main__":
    main()
