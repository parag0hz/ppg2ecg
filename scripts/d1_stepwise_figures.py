"""D1 addendum — per-dataset visualisation of the generated ECG as a function of the sampling budget (NFE).

Reads ONLY the evaluator's already-written `outputs/d1_<corpus>_seed42/eval/waveforms_nfe<N>.npz`; the model is
never re-run, so nothing here can move a D1 number. Every budget stores the SAME (subject, window_index) rows with
byte-identical ground truth (asserted below), so a single window can be traced across budgets.

Emits, under outputs/d1_bench/figures/:
  FIG 7  grid: rows = datasets, columns = NFE budget, one representative test window each, GT vs generated.
  FIG 8  per dataset: all budgets overlaid on one window, beside the population-level divergence-from-NFE-1 and
         RMSE-to-ground-truth as functions of the budget.
  FIG 9  per dataset: |generated(NFE) - generated(NFE 1)| pooled over the whole saved population, as a heat strip,
         so "the budget changes nothing" is shown on every window rather than on a chosen one.
Run: .venv/bin/python scripts/d1_stepwise_figures.py [--corpus KEY ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import d1_common as C  # noqa: E402

FS = 128
NFES = (1, 2, 4, 10, 25, 50)
OUT = ROOT / "outputs/d1_bench/figures"
CMAP = plt.get_cmap("viridis")
NFE_COLOR = {n: CMAP(i / (len(NFES) - 1)) for i, n in enumerate(NFES)}


def load_budgets(key: str) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[tuple[str, int]]] | None:
    """{nfe: yhat}, y, x, [(subject, window_index)] — or None when the corpus has no saved waveforms."""
    d = ROOT / f"outputs/d1_{key}_seed42/eval"
    paths = {n: d / f"waveforms_nfe{n}.npz" for n in NFES}
    missing = [n for n, p in paths.items() if not p.exists()]
    if missing:
        print(f"[warn] {key}: no waveforms for NFE {missing} — corpus skipped")
        return None
    yhat, y0, x0, keys0 = {}, None, None, None
    for n, p in paths.items():
        z = np.load(p, allow_pickle=True)
        keys = list(zip(z["subject"].tolist(), z["window_index"].tolist()))
        if y0 is None:
            y0, x0, keys0 = z["y"].astype(np.float64), z["x"].astype(np.float64), keys
        else:
            assert keys == keys0, f"{key}: NFE {n} stores different windows than NFE {NFES[0]}"
            assert np.array_equal(z["y"].astype(np.float64), y0), f"{key}: NFE {n} ground truth differs"
        yhat[n] = z["yhat"].astype(np.float64)
    return yhat, y0, x0, keys0


def rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).mean(axis=1))


def pick_window(keys: list[tuple[str, int]]) -> int:
    """Deterministic and stated in the caption: the first stored row of the first test subject in natural order."""
    first = sorted({s for s, _ in keys}, key=C.natural_key)[0]
    return next(i for i, (s, _) in enumerate(keys) if s == first)


def fig7(data: dict, out: Path) -> Path:
    ks = list(data)
    t = np.arange(1024) / FS
    fig, axes = plt.subplots(len(ks), len(NFES), figsize=(3.1 * len(NFES), 2.0 * len(ks)), squeeze=False)
    for r, k in enumerate(ks):
        yhat, y, _x, keys = data[k]
        j = pick_window(keys)
        for c, n in enumerate(NFES):
            a = axes[r][c]
            a.plot(t, y[j], color="black", lw=0.8, label="ground truth" if (r, c) == (0, 0) else None)
            a.plot(t, yhat[n][j], color=NFE_COLOR[n], lw=0.8, label="generated" if (r, c) == (0, 0) else None)
            a.set_title(f"{C.corpus(k).name} · NFE {n}" if r == 0 or c == 0 else f"NFE {n}", fontsize=8)
            a.tick_params(labelsize=7)
            if c == 0:
                a.set_ylabel(f"{C.corpus(k).name}\n{keys[j][0]} w{keys[j][1]}", fontsize=8)
            if r == len(ks) - 1:
                a.set_xlabel("time (s)", fontsize=8)
            if (r, c) == (0, 0):
                a.legend(fontsize=6, loc="upper right")
    fig.suptitle("FIG 7 — generated ECG vs sampling budget, one deterministic test window per dataset", fontsize=11)
    fig.text(0.5, 0.005,
             "Rows = datasets, columns = NFE. Black = ground-truth ECG, colour = generated ECG (colour encodes the "
             "budget). Window selection is deterministic and was not inspected before selection: the FIRST stored row "
             "of the FIRST test subject in natural order. Amplitude is the frozen per-window z-score then [-1,1] "
             "normalisation, so it is dimensionless. Every budget shows the SAME window with byte-identical ground "
             "truth (asserted at load).", ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"d1_fig7_stepwise_grid.{ext}", dpi=300)
    plt.close(fig)
    return out / "d1_fig7_stepwise_grid.png"


def fig8(data: dict, out: Path) -> Path:
    ks = list(data)
    t = np.arange(1024) / FS
    fig, axes = plt.subplots(len(ks), 2, figsize=(13, 2.3 * len(ks)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.1, 1.0]})
    for r, k in enumerate(ks):
        yhat, y, _x, keys = data[k]
        j = pick_window(keys)
        a = axes[r][0]
        a.plot(t, y[j], color="black", lw=1.1, zorder=3, label="ground truth")
        for n in NFES:
            a.plot(t, yhat[n][j], color=NFE_COLOR[n], lw=0.7, alpha=0.85, label=f"NFE {n}")
        a.set_ylabel("norm. amplitude", fontsize=8)
        a.set_title(f"{C.corpus(k).name} · {keys[j][0]} window {keys[j][1]} — all budgets overlaid", fontsize=9)
        a.tick_params(labelsize=7)
        if r == len(ks) - 1:
            a.set_xlabel("time (s)", fontsize=8)
        if r == 0:
            a.legend(fontsize=6, ncol=4, loc="upper right")

        b = axes[r][1]
        div = [float(np.abs(yhat[n] - yhat[1]).mean()) for n in NFES]          # population mean |Δ| from NFE 1
        err = [float(rmse(yhat[n], y).mean()) for n in NFES]                    # population mean RMSE to GT
        b.plot(NFES, div, "o-", color="#B03A2E", lw=1.2, ms=4, label="mean |gen(NFE) − gen(NFE 1)|")
        b.set_xscale("log")
        b.set_xticks(NFES)
        b.set_xticklabels([str(n) for n in NFES], fontsize=7)
        b.set_ylabel("mean |Δ| from NFE 1", fontsize=8, color="#B03A2E")
        b.tick_params(axis="y", labelcolor="#B03A2E", labelsize=7)
        b.set_ylim(bottom=0)
        b2 = b.twinx()
        b2.plot(NFES, err, "s--", color="#1F618D", lw=1.2, ms=4, label="RMSE to ground truth")
        b2.set_ylabel("RMSE to GT", fontsize=8, color="#1F618D")
        b2.tick_params(axis="y", labelcolor="#1F618D", labelsize=7)
        b.set_title(f"{C.corpus(k).name} — whole saved population (n={len(y)})", fontsize=9)
        if r == len(ks) - 1:
            b.set_xlabel("NFE (log scale)", fontsize=8)
    fig.suptitle("FIG 8 — how much does the sampling budget actually move the output?", fontsize=11)
    fig.text(0.5, 0.005,
             "Left: every budget drawn over the same deterministic window (same selection rule as FIG 7). Right: two "
             "population quantities over ALL saved test windows of that corpus — red, the mean absolute change of the "
             "generated waveform relative to the NFE-1 output (0 would mean the budget changes nothing); blue, the "
             "mean RMSE to ground truth. Read them TOGETHER: red rising while blue stays flat or climbs means the "
             "extra steps CHANGE the answer without moving it toward the truth. The two y axes are independent and "
             "blue spans a far narrower range than red. Both are descriptive; neither is a preregistered D1 metric.",
             ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"d1_fig8_stepwise_divergence.{ext}", dpi=300)
    plt.close(fig)
    return out / "d1_fig8_stepwise_divergence.png"


def fig9(data: dict, out: Path) -> Path:
    """Per-window divergence from NFE 1, every saved window — so the claim does not rest on one chosen window."""
    ks = list(data)
    fig, axes = plt.subplots(1, len(ks), figsize=(2.9 * len(ks), 3.4), squeeze=False)
    for c, k in enumerate(ks):
        yhat, y, _x, _keys = data[k]
        m = np.stack([np.abs(yhat[n] - yhat[1]).mean(axis=1) for n in NFES])   # [n_nfe, n_windows]
        a = axes[0][c]
        im = a.imshow(m, aspect="auto", origin="lower", cmap="magma", vmin=0.0,
                      extent=(0, m.shape[1], -0.5, len(NFES) - 0.5))
        a.set_yticks(range(len(NFES)))
        a.set_yticklabels([str(n) for n in NFES], fontsize=7)
        a.set_title(f"{C.corpus(k).name}\nn={m.shape[1]} windows", fontsize=8)
        a.set_xlabel("test window (stored order)", fontsize=7)
        a.tick_params(labelsize=7)
        if c == 0:
            a.set_ylabel("NFE", fontsize=8)
        fig.colorbar(im, ax=a, fraction=0.046).ax.tick_params(labelsize=6)
    fig.suptitle("FIG 9 — per-window |generated(NFE) − generated(NFE 1)|, every saved test window", fontsize=11)
    fig.text(0.5, 0.005,
             "Each column of a panel is one test window; each row is a budget. Colour is the mean absolute change of "
             "the generated waveform relative to that window's NFE-1 output; the NFE-1 row is 0 by construction. This "
             "panel says whether the budget moves the output on EVERY window or only on the one FIG 7 and "
             "FIG 8 happen to show. It says nothing about whether the movement is an IMPROVEMENT: for that "
             "read the blue RMSE curve in FIG 8, which is flat or rising on four of the five corpora."
             , ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"d1_fig9_stepwise_population.{ext}", dpi=300)
    plt.close(fig)
    return out / "d1_fig9_stepwise_population.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", choices=list(C.BENCH_KEYS), default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data, skipped = {}, []
    for k in (args.corpus or list(C.BENCH_KEYS)):
        loaded = load_budgets(k)
        if loaded is None:
            skipped.append(k)
        else:
            data[k] = loaded
    if not data:
        print("[d1-stepwise] no corpus had saved waveforms — nothing drawn")
        return 1

    paths = [fig7(data, out), fig8(data, out), fig9(data, out)]
    summary = {
        "corpora": list(data),
        "skipped": skipped,
        "nfes": list(NFES),
        "figures": [str(p.relative_to(ROOT)) for p in paths],
        "divergence_from_nfe1": {
            k: {str(n): round(float(np.abs(d[0][n] - d[0][1]).mean()), 6) for n in NFES} for k, d in data.items()
        },
        "rmse_to_gt": {
            k: {str(n): round(float(rmse(d[0][n], d[1]).mean()), 6) for n in NFES} for k, d in data.items()
        },
        "window_selection_rule": "first stored row of the first test subject in natural order; not inspected before selection",
        "source": "evaluator waveforms_nfe*.npz only; the model was NOT re-run",
    }
    (out / "d1_stepwise_manifest.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
