"""D2 figures (docs/D2_BASELINE_FLOOR_PREREGISTRATION.md §6). Reads outputs/d2_baselines/*.csv only.

FIG 10  per corpus, every arm's R-peak F1@50 ms as a bar with its subject-clustered CI, our model marked.
FIG 11  matched vs mismatched PPG (B4), as the paired difference with CI, beside the arms' absolute values.
Run: .venv/bin/python scripts/d2_figures.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/d2_baselines"
OUT = ROOT / "outputs/d2_baselines/figures"
CORPORA = [("wildppg", "WildPPG"), ("dalia", "PPG-DaLiA"), ("bidmc", "BIDMC"),
           ("capnobase", "CapnoBase"), ("vitaldb", "VitalDB")]
ARMS = [("ours_nfe1", "ours (NFE 1)", "#B03A2E"),
        ("B4_mismatched_ppg", "B4 mismatched PPG", "#E59866"),
        ("B0_wrong_window", "B0 wrong-window*", "#7D6608"),
        ("B1_ppg_template", "B1 PPG+template", "#1F618D"),
        ("B2_ppg_mean_beat", "B2 PPG+mean beat", "#5499C7"),
        ("B3_train_mean", "B3 train-mean", "#7F8C8D"),
        ("B5_gt_timing", "B5 GT-timing*", "#117A65")]
LEAK_NOTE = ("*  B0 uses the test subject's own real ECG from another window and B5 places the template at the TRUE "
             "R peaks (GT-R leakage; diagnostic only). Neither is a claim about our method; they bound what is "
             "reachable when timing is known.")


def read(name: str) -> list[dict]:
    with open(SRC / name) as f:
        return list(csv.DictReader(f))


def fig10(rows: list[dict], metric: str, out: Path) -> Path:
    d = {(r["corpus"], r["arm"]): r for r in rows if r["metric"] == metric}
    fig, axes = plt.subplots(1, len(CORPORA), figsize=(3.3 * len(CORPORA), 4.4), squeeze=False, sharey=True)
    fig.subplots_adjust(left=0.13)   # the arm names live in the left margin; tight_layout alone crops them
    for c, (key, name) in enumerate(CORPORA):
        a = axes[0][c]
        ys, los, his, cols, labs = [], [], [], [], []
        for arm, lab, col in ARMS:
            r = d.get((key, arm))
            if not r:
                continue
            ys.append(float(r["subject_macro_mean"]))
            los.append(float(r["subject_macro_mean"]) - float(r["ci_lo"]))
            his.append(float(r["ci_hi"]) - float(r["subject_macro_mean"]))
            cols.append(col)
            labs.append(lab)
        pos = np.arange(len(ys))
        a.barh(pos, ys, xerr=np.vstack([los, his]), color=cols, height=0.7,
               error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2})
        a.set_yticks(pos)
        if c == 0:
            a.set_yticklabels(labs, fontsize=8)   # sharey: setting "" on a later panel wipes these too
        a.tick_params(labelleft=(c == 0))
        a.invert_yaxis()
        a.set_title(name, fontsize=10)
        a.set_xlabel("R-peak F1 @ 50 ms  ↑", fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(axis="x", alpha=0.25, lw=0.5)
        a.axvline(float(d[(key, "ours_nfe1")]["subject_macro_mean"]), color="#B03A2E", ls=":", lw=1.0, zorder=0)
    fig.suptitle("FIG 10 — where our method sits against trivial predictors, per corpus", fontsize=12)
    fig.text(0.5, 0.02,
             "Bars are the subject-macro mean with the subject-clustered bootstrap 95 % CI (2000 replicates, seed "
             "20260904) — the same estimator D1 reports, on the identical test population (asserted row-for-row "
             "against D1's own CSV). The dotted line marks our model. " + LEAK_NOTE,
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0.115, 0.09, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"d2_fig10_baseline_floor.{ext}", dpi=300)
    plt.close(fig)
    return out / "d2_fig10_baseline_floor.png"


def fig11(rows: list[dict], paired: list[dict], metric: str, out: Path) -> Path:
    d = {(r["corpus"], r["arm"]): r for r in rows if r["metric"] == metric}
    p = {(r["corpus"], r["baseline"]): r for r in paired if r["metric"] == metric}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    a = axes[0]
    w, pos = 0.36, np.arange(len(CORPORA))
    for off, (arm, lab, col) in [(-w / 2, ARMS[0]), (w / 2, ARMS[1])]:
        ys = [float(d[(k, arm)]["subject_macro_mean"]) for k, _ in CORPORA]
        lo = [float(d[(k, arm)]["subject_macro_mean"]) - float(d[(k, arm)]["ci_lo"]) for k, _ in CORPORA]
        hi = [float(d[(k, arm)]["ci_hi"]) - float(d[(k, arm)]["subject_macro_mean"]) for k, _ in CORPORA]
        a.bar(pos + off, ys, w, yerr=np.vstack([lo, hi]), label=lab, color=col,
              error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2})
    a.set_xticks(pos)
    a.set_xticklabels([n for _, n in CORPORA], fontsize=8)
    a.set_ylabel("R-peak F1 @ 50 ms  ↑", fontsize=9)
    a.set_title("same model, same noise — only the conditioning PPG differs", fontsize=10)
    a.legend(fontsize=8)
    a.grid(axis="y", alpha=0.25, lw=0.5)
    a.tick_params(labelsize=7)

    b = axes[1]
    diffs = [float(p[(k, "B4_mismatched_ppg")]["ours_minus_baseline"]) for k, _ in CORPORA]
    lo = [d_ - float(p[(k, "B4_mismatched_ppg")]["ci_lo"]) for d_, (k, _) in zip(diffs, CORPORA)]
    hi = [float(p[(k, "B4_mismatched_ppg")]["ci_hi"]) - d_ for d_, (k, _) in zip(diffs, CORPORA)]
    b.barh(pos, diffs, xerr=np.vstack([lo, hi]), color="#B03A2E", height=0.6,
           error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2})
    b.axvline(0, color="black", lw=1.0)
    b.set_yticks(pos)
    b.set_yticklabels([n for _, n in CORPORA], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("matched − mismatched  (F1 @ 50 ms)", fontsize=9)
    b.set_title("conditioning effect, paired by subject", fontsize=10)
    b.grid(axis="x", alpha=0.25, lw=0.5)
    b.tick_params(labelsize=7)
    for i, v in enumerate(diffs):
        b.text(v + 0.012, i, f"{v:+.3f}", va="center", fontsize=7.5)

    fig.suptitle("FIG 11 — does the model use the PPG it is given?", fontsize=12)
    fig.text(0.5, 0.02,
             "B4 feeds the model the PPG of the cyclically-next window OF THE SAME SUBJECT, holding the checkpoint, "
             "the NFE and the noise draw fixed, so the only difference from `ours` is which PPG goes in. A gap that "
             "excludes zero means the output depends on the conditioning signal. Right panel: the paired "
             "(ours − B4) difference, resampling subjects.",
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"d2_fig11_conditioning.{ext}", dpi=300)
    plt.close(fig)
    return out / "d2_fig11_conditioning.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="rpeak_f1_50ms")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, paired = read("summary_by_arm.csv"), read("paired_vs_ours.csv")
    for p in (fig10(rows, args.metric, out), fig11(rows, paired, args.metric, out)):
        print(f"[d2-fig] {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
