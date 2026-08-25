"""docs/REPLICATION_SUMMARY.md: A2 (DaLiA S2) / A3 (DaLiA S1) / A4 (WildPPG) in one table + integrated verdict (prereg Part II §8)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPS = [("A2", "PPG-DaLiA", "S2", "outputs/a2_imeanflow_s5_ppgdalia_8s_seed42", "outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42"),
        ("A3", "PPG-DaLiA", "S1", "outputs/a3_imeanflow_ppgdalia_testS1_seed42", "outputs/a3_otcfm_ppgdalia_testS1_seed42"),
        ("A4", "WildPPG", "kjd, ssx", "outputs/a4_imeanflow_wildppg_seed42", "outputs/a4_otcfm_wildppg_seed42")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/REPLICATION_SUMMARY.md")
    args = ap.parse_args()
    rows, verdicts = [], {}
    for tag, ds, subj, imf, ot in EXPS:
        rp = ROOT / imf / "recovery.json"
        if not rp.exists():
            rows.append({"tag": tag, "ds": ds, "subj": subj, "missing": True})
            continue
        r = json.loads(rp.read_text())
        a = r["arms"]
        o50, o4, o1, i1 = a["OT-CFM Heun 25"], a["OT-CFM Heun 2"], a["OT-CFM Euler 1"], a["iMeanFlow 1 step"]
        rows.append({"tag": tag, "ds": ds, "subj": subj, "o50": o50, "o4": o4, "o1": o1, "i1": i1, "rec": r["recovery"], "verdict": r["verdict"], "rep": r.get("replication_verdict", "n/a"), "inv": r.get("pointwise_error_inversion"), "tr_imf": r["training_imf"], "tr_ot": r["training_otcfm"]})
        verdicts[tag] = r.get("replication_verdict", r["verdict"])
    a3, a4 = verdicts.get("A3"), verdicts.get("A4")
    ok = lambda v: v in ("REPLICATED", "SUCCESS")  # noqa: E731
    if a3 is None or a4 is None:
        overall = "INCOMPLETE (" + ", ".join(f"{k}={v}" for k, v in verdicts.items()) + ")"
    elif ok(a3) and ok(a4):
        overall = "STRONG REPLICATION"
    elif ok(a3) and not ok(a4):
        overall = "SUBJECT-ROBUST, DATASET-UNCERTAIN"
    elif ok(a4) and not ok(a3):
        overall = "DATASET-ROBUST, SUBJECT-UNCERTAIN"
    elif any(v == "PARTIAL" for v in (a3, a4)):
        overall = "MIXED"
    else:
        overall = "NOT ROBUST"
    L = ["# Replication Summary — does the frozen one-step iMeanFlow effect replicate?", "", "Pre-registration: `docs/A3_A4_REPLICATION_PREREGISTRATION.md`. All runs: seed 42, identical backbone/objectives/recipes; only the split (A3) or the dataset (A4) changed.", ""]
    L += ["## Cross-experiment comparison (test set, paired noise seed 0)", "| Experiment | Dataset | Test subjects | OT50 HR | OT4 HR | OT1 HR | iMF1 HR | OT50 Morph | OT1 Morph | iMF1 Morph | OT50 Amp | OT1 Amp | iMF1 Amp | OT50 Gain | OT1 Gain | iMF1 Gain |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['tag']} | {r['ds']} | {r['subj']} | — | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        o50, o4, o1, i1 = r["o50"], r["o4"], r["o1"], r["i1"]
        L.append(f"| {r['tag']} | {r['ds']} | {r['subj']} | {o50['hr_abs_err_bpm']:.2f} | {o4['hr_abs_err_bpm']:.2f} | {o1['hr_abs_err_bpm']:.2f} | **{i1['hr_abs_err_bpm']:.2f}** | {o50['morph_corr']:.3f} | {o1['morph_corr']:.3f} | **{i1['morph_corr']:.3f}** | {o50['amp_ratio']:.2f} | {o1['amp_ratio']:.2f} | **{i1['amp_ratio']:.2f}** | {o50['cond_gain_bpm']:.2f} | {o1['cond_gain_bpm']:.2f} | **{i1['cond_gain_bpm']:.2f}** |")
    L += ["", "## Recovery of the 50→1 NFE gap by iMeanFlow at 1 NFE", "| Experiment | HR | Morphology | Amplitude | Conditioning | beats/ref (iMF1) | A2 rule | Replication rule | Pointwise-error inversion |", "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['tag']} | — | — | — | — | — | — | — | — |")
            continue
        rc = r["rec"]
        L.append(f"| {r['tag']} ({r['ds']} {r['subj']}) | {rc['hr_abs_err_bpm']:+.2f} | {rc['morph_corr']:+.2f} | {rc['amp_ratio']:+.2f} | {rc['cond_gain_bpm']:+.2f} | {r['i1']['beats_ratio']:.2f} | {r['verdict']} | **{r['rep']}** | {'YES' if r['inv'] else 'NO'} |")
    L += ["", "## Ordering test (the scientific claim)", "| Experiment | A: OT1 ≪ OT50 (HR) | B: iMF1 ≫ OT1 (HR) | C: iMF1 → OT50 (HR gap left, bpm) | A (morph) | B (morph) | C (morph gap left) |", "|---|---|---|---:|---|---|---:|"]
    for r in rows:
        if r.get("missing"):
            L.append(f"| {r['tag']} | — | — | — | — | — | — |")
            continue
        o50, o1, i1 = r["o50"], r["o1"], r["i1"]
        L.append(f"| {r['tag']} | {'✔' if o1['hr_abs_err_bpm'] > o50['hr_abs_err_bpm'] + 1 else '✘'} | {'✔' if i1['hr_abs_err_bpm'] < o1['hr_abs_err_bpm'] - 1 else '✘'} | {i1['hr_abs_err_bpm'] - o50['hr_abs_err_bpm']:+.2f} | {'✔' if o1['morph_corr'] < o50['morph_corr'] - 0.05 else '✘'} | {'✔' if i1['morph_corr'] > o1['morph_corr'] + 0.05 else '✘'} | {i1['morph_corr'] - o50['morph_corr']:+.3f} |")
    L += ["", "## Training", "| Experiment | OT-CFM epochs/rounds (best) | iMF epochs/rounds (best) | OT-CFM h | iMF h |", "|---|---|---|---:|---:|"]
    for r in rows:
        if r.get("missing"):
            continue
        L.append(f"| {r['tag']} | {r['tr_ot']['epochs_run']} ({int(r['tr_ot']['best_epoch'])+1}) | {r['tr_imf']['epochs_run']} ({int(r['tr_imf']['best_epoch'])+1}) | {float(r['tr_ot']['total_train_time_s'])/3600:.1f} | {float(r['tr_imf']['total_train_time_s'])/3600:.1f} |")
    L += ["", f"## Overall verdict: **{overall}**", "", "(Rule, prereg Part II §8: STRONG = A3 and A4 replicated; SUBJECT-ROBUST/DATASET-UNCERTAIN = A3 replicated, A4 not; DATASET-ROBUST/SUBJECT-UNCERTAIN = A4 replicated, A3 not; MIXED = partial somewhere; NOT ROBUST = only A2.)", ""]
    (ROOT / args.out).write_text("\n".join(L))
    print(json.dumps({"verdicts": verdicts, "overall": overall}))
    print("wrote", ROOT / args.out)


if __name__ == "__main__":
    main()
