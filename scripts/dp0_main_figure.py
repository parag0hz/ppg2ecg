"""DP0 main figure (spec §54), drawn from the stored DP0 artifacts only (no model, no data access).

A architecture | B parameters | C point readout morphology / FP | D generative FD | E share-depth trade-off |
F shared-gradient cosine vs sharing depth | G combined inference cost. Point and generative metrics are never combined.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/dp0_main_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/dp0_dualreadout"
SH = ("S2", "S1", "S0")                                       # increasing share depth
LAB = {"S0": "S0 FULL", "S1": "S1 MIDDLE", "S2": "S2 STEM"}


def rd(n):
    return json.loads((ART / n).read_text())


def main():
    acc, pdev, gdev = rd("parameter_accounting.json"), rd("dev_point_metrics.json"), rd("dev_gen_metrics.json")
    bdev, gc, comp = rd("dev_bootstrap.json")["comparisons"], rd("gradient_conflict.json"), rd("compute_accounting.json")
    lm, lb, lg = rd("lock_metrics.json"), rd("lock_bootstrap.json")["comparisons"]["S1"], rd("lock_gates.json")
    lat, fl = comp["latency_ms_batch1"], comp["flops_one_window"]
    fig = plt.figure(figsize=(26, 11))
    gs = fig.add_gridspec(2, 4)
    a = fig.add_subplot(gs[0, 0])
    a.axis("off")
    a.set_title("A  DualReadout-ECG (selected: S1 MIDDLE)", loc="left")
    a.text(0.0, 0.96, "           PPG + event raster\n                   |\n   shared stem + blocks 1-6 (246,720)\n"
           "             /            \\\n     point adapter      flow adapter\n     blocks 7-8 (P)     blocks 7-8 (F)\n"
           "          |                 |   H_256, H_128 (Haar low-pass)\n     point decoder      ScaleFlow decoder (x_t, t)\n"
           "          |                 |\n         mu            ECG samples (Euler 8)\n\n"
           "  readouts never added; no residual path;\n  round-robin: 20k POINT + 20k FLOW updates",
           va="top", family="monospace", fontsize=9.5)
    a = fig.add_subplot(gs[0, 1])
    names = ["P + G\nseparate", "S0", "S1\n(selected)", "S2"]
    vals = [acc["separate_waveform_params"], acc["S0"]["total"], acc["S1"]["total"], acc["S2"]["total"]]
    a.bar(range(4), vals, color=["0.45", "#b3abe6", "#4a3aa7", "#b3abe6"])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v / 1e3:.0f}k" + ("" if i == 0 else f"\n{100 * (1 - v / vals[0]):.1f} % saved"), ha="center", va="bottom", fontsize=8)
    a.axhline(acc["e1_threshold"], color="k", ls="--", lw=0.8)
    a.set_xticks(range(4), names)
    a.set_ylim(0, 1.45e6)
    a.set_title("B  waveform parameters (dashed: E1 = 0.85 x separate)", loc="left")

    def pair(a, rows, title, ylab, margin=None):
        for i, (lab, v, col) in enumerate(rows):
            a.plot([i, i], [v[1], v[2]], color=col, lw=3)
            a.plot([i], [v[0]], "o", color=col)
        if margin is not None:
            a.axhline(margin, color="k", ls="--", lw=0.8)
        a.axhline(0, color="0.7", lw=0.6)
        a.set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
        a.set_title(title, loc="left")
        a.set_ylabel(ylab)
    dev_c, lock_c = "#7a6fd0", "#a23b52"
    pair(fig.add_subplot(gs[0, 2]), [*((f"DEV {s}", bdev[s]["corr"], dev_c) for s in SH), ("LOCK S1", lb["corr"], lock_c)],
         "C  point readout: corr - SPECIALIST P (P1 margin -0.02)", "beat corr difference", -0.02)
    pair(fig.add_subplot(gs[0, 3]), [*((f"DEV {s}", bdev[s]["fp"], dev_c) for s in SH), ("LOCK S1", lb["fp"], lock_c)],
         "C'  point readout: FP/window - SPECIALIST P (P2 margin +0.05)", "FP / window difference", 0.05)
    pair(fig.add_subplot(gs[1, 0]), [*((f"DEV {s}", bdev[s]["fd"], dev_c) for s in SH), ("LOCK S1", lb["fd"], lock_c)],
         "D  generative readout: FD - SPECIALIST G (G1 margin +1.0)", "FD difference", 1.0)
    a = fig.add_subplot(gs[1, 1])
    for s in SH:
        x, v = 100 * acc[s]["saving"], bdev[s]["fp"]
        col = "#2a7f3f" if rd("dev_gates.json")[s]["QUALIFIED"] else "#a23b52"
        a.plot([x, x], [v[1], v[2]], color=col, lw=3)
        a.plot([x], [v[0]], "o", color=col, ms=8)
        f = bdev[s]["fd"]
        a.annotate(f"{LAB[s]}\nFD - G {f[0]:+.2f} [{f[1]:+.2f}, {f[2]:+.2f}]", (x, v[0]), xytext=(8, 6), textcoords="offset points", fontsize=8)
    a.axhline(0.05, color="k", ls="--", lw=0.8)
    a.axvline(15, color="k", ls=":", lw=0.8)
    a.text(15.3, 0.97, "E1: saving >= 15 %", fontsize=8, transform=a.get_xaxis_transform(), va="top")
    a.text(0.3, 0.052, "P2 margin", fontsize=8)
    a.set_xlim(-2, 36)
    a.set_xlabel("parameter saving vs separate specialists (%)")
    a.set_ylabel("point FP/window - SPECIALIST P (DEV)")
    a.set_title("E  share-depth trade-off (DEV; green = qualified)", loc="left")
    a = fig.add_subplot(gs[1, 2])
    for j, (lab, c) in enumerate((("init", "0.6"), ("25pct", "#d59a54"), ("final", "#4a3aa7"))):
        for i, s in enumerate(SH):
            g = gc[s][lab]
            a.plot([i + (j - 1) * 0.2] * 2, [g["q25"], g["q75"]], color=c, lw=3, label=lab if i == 0 else None)
            a.plot([i + (j - 1) * 0.2], [g["median"]], "o", color=c)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(3), [f"{LAB[s]}\n({gc[s]['final']['shared_params']:,} shared)" for s in SH], fontsize=8)
    a.legend(fontsize=8)
    a.set_title("F  cos(grad L_point, grad L_FM) on shared params (median, IQR)", loc="left")
    a = fig.add_subplot(gs[1, 3])
    modes = (("point_only", "point only"), ("gen_only", "generation (NFE 8)"), ("both", "both outputs"))
    for j, (m, lab) in enumerate(modes):
        a.bar(np.arange(2) + (j - 1) * 0.27, [lat[f"cuda_separate_{m}"], lat[f"cuda_S1_{m}"]], width=0.27, label=f"GPU {lab}")
    a.set_xticks(range(2), ["P + G separate", "S1 (cached trunk)"])
    a.set_ylabel("GPU batch-1 latency (ms)")
    a.legend(fontsize=8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.88))
    a.text(0.02, 0.98, f"FLOPs both outputs: separate {fl['separate_both_nfe8'] / 1e9:.2f} G, S1 cached {fl['S1']['both_cached_nfe8'] / 1e9:.2f} G, "
           f"\nuncached {fl['S1']['both_uncached_nfe8'] / 1e9:.2f} G; CPU both: separate {lat['cpu_separate_both']:.1f} ms, S1 {lat['cpu_S1_both']:.1f} ms",
           transform=a.transAxes, va="top", fontsize=7.5)
    a.set_ylim(0, 1.6 * lat["cuda_separate_both"])
    a.set_title("G  combined inference cost", loc="left")
    fig.suptitle(f"DP0 DualReadout-ECG — development winner S1 MIDDLE; AF-LOCK (337 patients): {lg['verdict']}. "
                 "Readouts evaluated against their own specialists; no combined score.", fontsize=12)
    fig.tight_layout()
    fig.savefig(ART / "figure_main.png", dpi=105)
    print("wrote", ART / "figure_main.png")


if __name__ == "__main__":
    main()
