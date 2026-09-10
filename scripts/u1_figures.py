#!/usr/bin/env python
"""U1 — paper-style qualitative figure from the upstream PENGUIN checkpoints.

Reproduces the layout of the PENGUIN paper's qualitative figure (rows: PPG / original
vital sign / PENGUIN; columns grouped into ECG, respiration and ABP tasks) using the
checkpoints that U1's own runs produced -- upstream's model, upstream's sampler
(Heun, n_step=25 => 50 NFE), upstream's test split, at the shipped segment_len=4.

Window selection is deterministic and declared: the first N windows of the test split
in upstream's own concatenation order (i.e. the first file of `file_path["test_path"]`,
its stored window order). No window was inspected before it was selected.

The paper's figure also carries a PaPaGei-S row. PaPaGei-S was not trained in U1, so
that row is absent rather than filled from another source.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "PENGUIN" / "src"))

LOGROOT = ROOT / "outputs" / "u1_upstream"
PROC = ROOT / "data" / "processed" / "upstream_u1"
FIGDIR = ROOT / "artifacts" / "u1_upstream" / "figures"   # repo convention: committed figures live under artifacts/

GROUPS = [
    ("ECG reconstruction", ["PPG-DaLiA", "WildPPG"]),
    ("Respiratory monitoring", ["BIDMC", "WESAD"]),
    ("ABP monitoring", ["UCI-BP", "MIMIC-BP"]),
]
DATASETS = [ds for _, dss in GROUPS for ds in dss]

PPG_COLOR = "#1f4ed8"
SIG_COLOR = "#f28c28"
FS = 128
SEG_LEN = 4


def load_test_windows(ds: str, n: int) -> tuple[np.ndarray, np.ndarray, str]:
    """First n windows of the test split, in upstream's own concatenation order."""
    split = json.loads((LOGROOT / f"split_{ds}.json").read_text())
    first = split["test"][0]
    with (PROC / ds / first).open("rb") as fh:
        d = pickle.load(fh)
    x, y = np.asarray(d["x_data"]), np.asarray(d["y_data"])
    return x[:n], y[:n], first


def sample_predictions(ds: str, ppg: np.ndarray, seed: int = 42) -> tuple[np.ndarray, int]:
    from utils.help_func import fix_seed, load_checkpoint

    ckpt = LOGROOT / f"PENGUIN_{ds}_u1" / "ckpt" / "pretrain_ckpt.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fix_seed(seed)
    epoch = int(torch.load(str(ckpt), weights_only=False)["epoch"])
    model = load_checkpoint(str(ckpt), device)
    model.eval()
    with torch.no_grad():
        t = torch.tensor(ppg, dtype=torch.float32, device=device)
        pred = model(t).detach().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return pred, epoch


def style_panel(ax) -> None:
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#333333")
    for g in range(1, SEG_LEN):
        ax.axvline(g * FS, color="#cccccc", lw=0.6, zorder=0)
    ax.margins(x=0)


RESULT = {   # U1 primary metric vs the published value, from docs/U1_..._REPORT.md §1
    "PPG-DaLiA": ("HR", 16.350, 15.64, "bpm"),
    "WildPPG": ("HR", 31.006, 12.97, "bpm"),
    "BIDMC": ("RR", 3.479, 2.98, "bpm"),
    "WESAD": ("RR", 4.393, 4.45, "bpm"),
    "UCI-BP": ("SBP/DBP", None, None, "mmHg"),
    "MIMIC-BP": ("SBP/DBP", None, None, "mmHg"),
}
RESULT_BP = {"UCI-BP": (19.986, 12.61, 7.863, 7.14), "MIMIC-BP": (14.986, 17.43, 9.212, 11.34)}


def _caption(ds: str, epoch: int) -> str:
    if ds in RESULT_BP:
        s1, p1, s2, p2 = RESULT_BP[ds]
        line = f"SBP {s1:.2f} / {p1:.2f} mmHg\nDBP {s2:.2f} / {p2:.2f} mmHg"
    else:
        name, got, paper, unit = RESULT[ds]
        line = f"{name} {got:.2f} / {paper:.2f} {unit}"
    return f"{line}\nU1 / published · ckpt ep. {epoch}"


def paper_figure(data: dict, window: int, out: Path, annotate: bool = False) -> None:
    rows = ["PPG", "Original\nvital sign", "PENGUIN"]
    fig, axes = plt.subplots(3, 6, figsize=(13.4, 4.85 if annotate else 4.0))
    bottom = 0.155 if annotate else 0.02
    top = 0.872 if annotate else 0.845
    fig.subplots_adjust(left=0.085, right=0.995, top=top, bottom=bottom, wspace=0.09, hspace=0.13)

    for c, ds in enumerate(DATASETS):
        ppg, tgt, pred = data[ds]["ppg"][window], data[ds]["target"][window], data[ds]["pred"][window]
        lo = min(tgt.min(), pred.min()); hi = max(tgt.max(), pred.max())
        pad = 0.06 * (hi - lo if hi > lo else 1.0)
        for r, series in enumerate((ppg, tgt, pred)):
            ax = axes[r, c]
            ax.plot(series, color=PPG_COLOR if r == 0 else SIG_COLOR, lw=0.9)
            style_panel(ax)
            if r > 0:
                ax.set_ylim(lo - pad, hi + pad)   # target and PENGUIN share a scale per column
            if c == 0:
                ax.set_ylabel(rows[r], rotation=0, ha="right", va="center", fontsize=10, labelpad=12)
        axes[0, c].set_title(ds, fontsize=10, pad=5)
        if annotate:
            axes[2, c].set_xlabel(_caption(ds, data[ds]["epoch"]), fontsize=7.2, labelpad=4)

    # group headers with rules, above the column titles
    for title, dss in GROUPS:
        cs = [DATASETS.index(d) for d in dss]
        x0 = axes[0, cs[0]].get_position().x0
        x1 = axes[0, cs[-1]].get_position().x1
        fig.text((x0 + x1) / 2, 0.955, title, ha="center", va="bottom", fontsize=11)
        fig.add_artist(Line2D([x0, x1], [0.948, 0.948], color="#333333", lw=0.9))

    # vertical separators between task groups
    for i in (1, 3):
        xa = axes[0, i].get_position().x1
        xb = axes[0, i + 1].get_position().x0
        x = (xa + xb) / 2
        y0 = axes[2, 0].get_position().y0 - (0.105 if annotate else 0.0)
        fig.add_artist(Line2D([x, x], [max(y0, 0.01), 0.945], color="#333333", lw=0.9))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def variability_figure(data: dict, n: int, out: Path) -> None:
    """The same checkpoints on n consecutive test windows -- so one lucky window cannot
    stand in for the result. Target black, PENGUIN orange, on a shared per-cell scale."""
    h = 1.15 * n + 1.25
    fig, axes = plt.subplots(n, 6, figsize=(13.4, h))
    fig.subplots_adjust(left=0.055, right=0.995, top=1 - 0.55 / h,
                        bottom=0.40 / h, wspace=0.09, hspace=0.13)
    for c, ds in enumerate(DATASETS):
        for r in range(n):
            ax = axes[r, c]
            ax.plot(data[ds]["target"][r], color="#111111", lw=0.8)
            ax.plot(data[ds]["pred"][r], color=SIG_COLOR, lw=0.8, alpha=0.9)
            style_panel(ax)
            if c == 0:
                ax.set_ylabel(f"win {r}", rotation=0, ha="right", va="center", fontsize=8, labelpad=8)
        axes[0, c].set_title(ds, fontsize=10, pad=5)
    fig.legend(handles=[Line2D([], [], color="#111111", lw=1.2, label="original vital sign"),
                        Line2D([], [], color=SIG_COLOR, lw=1.2, label="PENGUIN (U1)")],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.004))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--paper-window", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data, manifest = {}, {}
    for ds in DATASETS:
        ppg, tgt, src = load_test_windows(ds, args.n_windows)
        pred, epoch = sample_predictions(ds, ppg, args.seed)
        data[ds] = {"ppg": ppg, "target": tgt, "pred": pred, "epoch": epoch}
        manifest[ds] = {
            "test_file": src,
            "windows": f"0..{args.n_windows - 1} (upstream test-split order, first file)",
            "checkpoint": str((LOGROOT / f'PENGUIN_{ds}_u1' / 'ckpt' / 'pretrain_ckpt.pth').relative_to(ROOT)),
            "sampler": "upstream Heun n_step=25 (50 NFE)",
            "seed": args.seed,
            "checkpoint_epoch_0based": epoch,
            "segment_len_s": SEG_LEN,
            "fs": FS,
        }
        print(f"{ds}: {src} windows {ppg.shape} pred {pred.shape}")

    paper_figure(data, args.paper_window, FIGDIR / "u1_paper_style_qualitative")
    paper_figure(data, args.paper_window, FIGDIR / "u1_paper_style_annotated", annotate=True)
    variability_figure(data, args.n_windows, FIGDIR / "u1_test_window_variability")
    (FIGDIR / "u1_figures_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {FIGDIR}/u1_paper_style_qualitative.{{png,pdf}}")
    print(f"wrote {FIGDIR}/u1_paper_style_annotated.{{png,pdf}}")
    print(f"wrote {FIGDIR}/u1_test_window_variability.{{png,pdf}}")
    print(f"wrote {FIGDIR}/u1_figures_manifest.json")


if __name__ == "__main__":
    main()
