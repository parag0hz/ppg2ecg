"""D1 paper figures — docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md section 8, FIG 1-6.

Drawn only from the frozen evaluator output. No generator is run and no metric is recomputed except the
beat-level matching of FIG 4, which needs per-beat information the CSVs do not carry. Window selection is
deterministic and stated in every caption; a corpus whose evaluator output is missing is skipped with a
printed warning — never fabricated, never silently dropped (prereg section 8).

The read-out layer (corpus discovery, metric identity, the subject-clustered CIs) is imported from
scripts/d1_report.py so that a figure and a table can never disagree about a number.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d1_common as C  # noqa: E402
import d1_report as R  # noqa: E402

from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402

ROOT = R.ROOT
FS, T_LEN, DPI = C.FS, C.SAMPLES_PER_WINDOW, 300
GRID_POS = (0, 4, 8)               # FIG 1: deterministic stored-order positions inside the first test subject
BEAT_WINDOWS = 200                 # FIG 4: beat-matching budget, first N stored windows
TOL_MS = 50.0                      # prereg section 6 headline matching tolerance
PALETTE = ("#2166ac", "#b2182b", "#1a9850", "#762a83", "#e08214", "#01665e", "#c51b7d", "#4d4d4d")
GT_COL, GEN_COL, PPG_COL, TICK_COL = "#111111", "#b2182b", "#2166ac", "#c8c8c8"


def colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def label(ev: dict) -> str:
    return ev["corpus"].name


def axis_label(key: str) -> str:
    """Metric name, orientation arrow and units — prereg section 8 requires every axis labelled with units."""
    u = R.units(key)
    return f"{R.SPEC[key].name} {R.ARROW[R.SPEC[key].orientation]}" + ("" if u == "dimensionless" else f" ({u})")


def save(fig, out: Path, stem: str, caption: str) -> list[str]:
    """Both formats at 300 dpi, always; the caption is drawn on the figure so it travels with the file."""
    fig.text(0.004, 0.002, caption, fontsize=6.2, va="bottom", ha="left", wrap=True, color="0.2")
    names = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
        names.append(f"{stem}.{ext}")
    plt.close(fig)
    return names


def waveforms(ev: dict, nfe: int):
    """(x, y, yhat, subject, window_index) from waveforms_nfe<N>.npz, or None with a warning."""
    p = ev["dir"] / f"waveforms_nfe{nfe}.npz"
    if not p.exists():
        R.warn(f"{ev['key']}: {p.name} missing — waveform panels skipped for this corpus")
        return None
    z = np.load(p, allow_pickle=False)
    subj = np.asarray(z["subject"]).astype(str).ravel()
    if subj.size == 1 and len(z["y"]) > 1:
        subj = np.repeat(subj, len(z["y"]))
    return (np.asarray(z["x"], np.float64), np.asarray(z["y"], np.float64), np.asarray(z["yhat"], np.float64),
            subj, np.asarray(z["window_index"]).ravel())


def select_windows(subj: np.ndarray) -> tuple[str, np.ndarray]:
    """Deterministic and stated in the caption: first test subject in natural order, its rows in stored order."""
    first = sorted(set(subj.tolist()), key=C.natural_key)[0]
    rows = np.flatnonzero(subj == first)
    pos = [p for p in GRID_POS if p < rows.size]
    if len(pos) < len(GRID_POS):
        pos = np.unique(np.linspace(0, rows.size - 1, min(len(GRID_POS), rows.size)).astype(int)).tolist()
    return first, rows[pos]


def beat_timing(ev: dict, nfe: int, budget: int) -> tuple[np.ndarray, str]:
    """Signed matched-beat timing error (ms) over the first `budget` stored windows. The only recomputation."""
    w = waveforms(ev, nfe)
    if w is None:
        return np.zeros(0), f"waveforms_nfe{nfe}.npz absent"
    _, y, yh, _, _ = w
    dt = []
    for a, b in zip(y[:budget], yh[:budget]):
        pr, pp = RP.detect_rpeaks(a, FS), RP.detect_rpeaks(b, FS)
        m, _, _ = RP.match_rpeaks(pr, pp, FS, TOL_MS)
        dt += [(pp[j] - pr[i]) / FS * 1000.0 for i, j in m]
    return np.asarray(dt), ("" if dt else f"detector matched no beat within ±{TOL_MS:.0f} ms in {budget} windows")


def hr_pairs(ev: dict, nfe: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference and generated HR per window, as the evaluator recorded them (hr_ref / hr_pred columns)."""
    a, b = R.window_values(ev, "hr_ref", "hr_pred", nfe=nfe)
    if a.size == 0 or b.size == 0:
        R.warn(f"{ev['key']}: hr_ref/hr_pred absent from per_window_metrics.csv — HR panels skipped")
        return np.zeros(0), np.zeros(0)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


# ---------------------------------------------------------------- figures
def fig1_qualitative(evs, order, out, nfe_of):
    panels = [(k, w) for k in order if (w := waveforms(evs[k], nfe_of[k])) is not None]
    if not panels:
        R.warn("FIG 1 skipped: no waveform file for any corpus")
        return []
    ncol = len(GRID_POS)
    fig = plt.figure(figsize=(4.4 * ncol, 2.7 * len(panels)))
    gs = fig.add_gridspec(len(panels), ncol, hspace=0.62, wspace=0.16)
    chosen, t = [], np.arange(T_LEN) / FS
    for r, (k, (x, y, yh, subj, widx)) in enumerate(panels):
        sub, rows = select_windows(subj)
        chosen.append(f"{label(evs[k])}: subject {sub}, windows " + "/".join(str(int(widx[i])) for i in rows))
        for c in range(ncol):
            inner = gs[r, c].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ae, ap = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            if c >= rows.size:
                ae.axis("off"); ap.axis("off")
                continue
            i = rows[c]
            for pk in RP.detect_rpeaks(y[i], FS):
                ae.axvline(pk / FS, color=TICK_COL, lw=0.8, zorder=1)
            ae.plot(t, y[i], color=GT_COL, lw=0.9, zorder=3, label="ground-truth ECG")
            ae.plot(t, yh[i], color=GEN_COL, lw=0.9, alpha=0.9, zorder=4, label="generated ECG")
            ap.plot(t, x[i], color=PPG_COL, lw=0.7)
            for a_ in (ae, ap):
                a_.set_xlim(0, T_LEN / FS); a_.tick_params(labelsize=6.5)
            ae.set_xticklabels([])
            ae.set_title(f"{label(evs[k])} · {sub} · window {int(widx[i])} · NFE {nfe_of[k]}", fontsize=7.5)
            ap.set_xlabel("time (s)", fontsize=7)
            if c == 0:
                ae.set_ylabel("ECG (normalised amplitude)", fontsize=7)
                ap.set_ylabel("PPG", fontsize=7)
            if r == 0 and c == 0:
                ae.legend(fontsize=6, loc="upper right", framealpha=0.85)
    cap = ("FIG 1 — qualitative reconstruction grid. Rows = datasets, columns = test windows. Black = ground-truth "
           "ECG, red = generated ECG on the same axes; the thin blue trace beneath each cell is the conditioning PPG. "
           "Grey vertical ticks mark R peaks detected on the GROUND-TRUTH ECG (the repo's neurokit detector). "
           "Amplitude is the frozen per-window z-score then [-1,1] normalisation, so it is dimensionless; the x axis "
           "is 8 s at 128 Hz. WINDOW SELECTION IS DETERMINISTIC AND NOT CHERRY-PICKED: inside each dataset's "
           "waveforms_nfe<N>.npz the test subjects are sorted in natural order, the FIRST subject is taken, and its "
           f"rows are taken in stored order at positions {'/'.join(str(p) for p in GRID_POS)} (evenly spaced positions "
           "if that subject has fewer windows). No window was inspected before selection and none was replaced. "
           "Selected — " + "; ".join(chosen) + ".")
    return save(fig, out, "d1_fig1_qualitative_grid", cap)


def fig2_metric_bars(evs, order, out, nfe_of, heads):
    fig, axs = plt.subplots(1, len(heads), figsize=(3.5 * len(heads), 4.3), squeeze=False)
    names = [label(evs[k]) for k in order]
    for a, m in zip(axs[0], heads):
        pts, lo, hi, miss = [], [], [], []
        for k in order:
            s = R.stat(evs[k], m, nfe_of[k])
            pts.append(s["macro"]); lo.append(max(s["macro"] - s["lo"], 0.0) if np.isfinite(s["macro"]) else 0.0)
            hi.append(max(s["hi"] - s["macro"], 0.0) if np.isfinite(s["macro"]) else 0.0)
            if not R.has(evs[k], m):
                miss.append(k)
        if miss:
            R.warn(f"FIG 2 · {m}: no column for {', '.join(miss)} — those bars are left empty")
        a.bar(range(len(order)), pts, yerr=[lo, hi], capsize=3.5,
              color=[colour(i) for i in range(len(order))], alpha=0.9)
        a.set_xticks(range(len(order))); a.set_xticklabels(names, rotation=25, ha="right", fontsize=7.5)
        a.set_title(f"{R.SPEC[m].name}  {R.ARROW[R.SPEC[m].orientation]}", fontsize=8.5)
        a.set_ylabel(axis_label(m), fontsize=7.5); a.grid(alpha=0.2, axis="y"); a.tick_params(labelsize=7)
    cap = ("FIG 2 — per-dataset headline metrics, each at that dataset's headline NFE. Bars are the SUBJECT-MACRO "
           f"mean; whiskers are the 95 % subject-clustered bootstrap CI ({C.BOOTSTRAP_N} replicates, seed "
           f"{C.BOOTSTRAP_SEED}), taken from the evaluator's own summary rows so that the figure and RESULTS.md "
           "TABLE 1 carry identical numbers. The arrow in each panel title is the orientation declared by "
           "PAPER_METRIC_SPEC (↑ higher is better, ↓ lower is better). An empty bar means the evaluator produced no "
           "such column. Corpus size differs across these datasets by two orders of magnitude (FIG 6): the ordering "
           "here is NOT evidence that one dataset is intrinsically harder than another.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, out, "d1_fig2_metric_bars", cap)


#: FIG 3 x-limits. Autoscaling a log axis whose largest point is NFE 50 stops at ~60.8, which leaves the rotated
#: NFE-50 annotation squeezed between the dashed rule and the right spine; the explicit right pad gives it a lane
#: of its own to the right of the rule. NFE_LIM must stay wider than C.NFE_GRID at both ends.
NFE_LIM = (0.8, 90.0)
NFE_ANNOT_X = 1.10                 # annotation anchor, as a multiple of the NFE-50 rule: clear of the dashes


def fig3_nfe_tradeoff(evs, order, out, heads):
    n = len(heads) + 1
    fig, axs = plt.subplots(1, n, figsize=(3.4 * n, 4.1), squeeze=False)
    for a, m in zip(axs[0], heads):
        for i, k in enumerate(order):
            xy = [(nfe, R.stat(evs[k], m, nfe)["macro"]) for nfe in evs[k]["nfes"]]
            xy = [(x, v) for x, v in xy if np.isfinite(v)]
            if xy:
                a.plot([p[0] for p in xy], [p[1] for p in xy], marker="o", ms=3.5, lw=1.2,
                       color=colour(i), label=label(evs[k]))
        a.axvline(R.PENGUIN_NFE, color="0.35", ls="--", lw=1.0)
        a.annotate("PENGUIN operating point (NFE 50)", (R.PENGUIN_NFE * NFE_ANNOT_X, 0.02),
                   xycoords=("data", "axes fraction"), fontsize=6, ha="left", va="bottom", color="0.35", rotation=90)
        a.set_xscale("log"); a.set_xlim(*NFE_LIM)
        a.set_xticks(list(C.NFE_GRID)); a.set_xticklabels([str(v) for v in C.NFE_GRID], fontsize=7)
        a.set_xlabel("NFE (log scale)", fontsize=7.5)
        a.set_ylabel(axis_label(m), fontsize=7.5)
        a.grid(alpha=0.2); a.tick_params(labelsize=7)
    a = axs[0][-1]
    for i, k in enumerate(order):
        xy = [(nfe, R.efficiency(evs[k], nfe)["gen_samples_per_s"]) for nfe in evs[k]["nfes"]]
        xy = [(x, v) for x, v in xy if np.isfinite(v)]
        if xy:
            a.plot([p[0] for p in xy], [p[1] for p in xy], marker="s", ms=3.5, lw=1.2, color=colour(i), label=label(evs[k]))
        else:
            R.warn(f"FIG 3: no generation throughput recorded for {k}")
    a.axvline(R.PENGUIN_NFE, color="0.35", ls="--", lw=1.0)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(*NFE_LIM)
    a.set_xticks(list(C.NFE_GRID)); a.set_xticklabels([str(v) for v in C.NFE_GRID], fontsize=7)
    a.set_xlabel("NFE (log scale)", fontsize=7.5); a.set_ylabel("generation throughput (windows/s) ↑", fontsize=7.5)
    a.grid(alpha=0.2); a.tick_params(labelsize=7); a.legend(fontsize=6.5)
    cap = ("FIG 3 — sampling-budget trade-off, the budget-matching figure. One line per dataset, one panel per "
           "headline metric, plus measured generation throughput on the right (log-log). Points are subject-macro "
           f"means at every evaluated NFE ({', '.join(str(v) for v in C.NFE_GRID)}, prereg section 6). The dashed rule "
           "marks NFE 50, PENGUIN's published operating point, so that a reader can compare at a matched budget "
           "instead of at our headline NFE 1; it is NOT itself a comparison — PENGUIN's published number comes from a "
           "different pipeline, split and parameter count (RESULTS.md TABLE 4). Throughput is whatever the evaluator "
           "measured on its own hardware and is not a controlled benchmark against any published system.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, out, "d1_fig3_nfe_tradeoff", cap)


def fig4_beat_failure(evs, order, out, nfe_of, budget):
    rows = []
    for k in order:
        a, b = hr_pairs(evs[k], nfe_of[k])
        dt, why = beat_timing(evs[k], nfe_of[k], budget)
        if a.size == 0 and dt.size == 0:
            R.warn(f"FIG 4: neither HR pairs nor matched beats available for {k} — row skipped")
            continue
        rows.append((k, a, b, dt, why))
    if not rows:
        R.warn("FIG 4 skipped: no beat-level data for any corpus")
        return []
    fig, axs = plt.subplots(len(rows), 3, figsize=(12.6, 3.5 * len(rows)), squeeze=False)
    for r, (k, a, b, dt, why) in enumerate(rows):
        lab = label(evs[k])
        ax = axs[r][0]
        if a.size >= 500:
            ax.hexbin(a, b, gridsize=36, mincnt=1, cmap="viridis", linewidths=0)
        elif a.size:
            ax.scatter(a, b, s=9, alpha=0.5, color=colour(r), edgecolors="none")
        lim = ([min(a.min(), b.min()) - 5, max(a.max(), b.max()) + 5] if a.size else [40, 180])
        ax.plot(lim, lim, color="0.3", ls="--", lw=1.0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("reference HR (bpm)", fontsize=7.5); ax.set_ylabel("generated HR (bpm)", fontsize=7.5)
        ax.set_title(f"{lab} — HR identity, {a.size} windows", fontsize=8.5)
        ax = axs[r][1]
        if a.size:
            diff = b - a
            bias = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
            ax.scatter((a + b) / 2.0, diff, s=9, alpha=0.5, color=colour(r), edgecolors="none")
            for v, ls, txt in ((bias, "-", f"bias {bias:+.2f}"), (bias + 1.96 * sd, "--", f"+1.96 SD {bias + 1.96 * sd:+.2f}"),
                               (bias - 1.96 * sd, "--", f"−1.96 SD {bias - 1.96 * sd:+.2f}")):
                ax.axhline(v, color="0.3", ls=ls, lw=1.0)
                ax.annotate(txt, (0.99, v), xycoords=("axes fraction", "data"), fontsize=6, ha="right", va="bottom", color="0.3")
        ax.set_xlabel("mean of reference and generated HR (bpm)", fontsize=7.5)
        ax.set_ylabel("generated − reference HR (bpm)", fontsize=7.5)
        ax.set_title(f"{lab} — Bland–Altman", fontsize=8.5)
        ax = axs[r][2]
        if dt.size:
            ax.hist(dt, bins=np.linspace(-TOL_MS, TOL_MS, 41), color=colour(r), alpha=0.85)
            ax.axvline(0.0, color="0.3", ls="--", lw=1.0)
            ax.annotate(f"n = {dt.size} matched beats\nmedian {np.median(dt):+.1f} ms\nMAE {np.mean(np.abs(dt)):.1f} ms",
                        (0.02, 0.97), xycoords="axes fraction", fontsize=6.5, va="top")
        else:
            ax.annotate(f"no matched beats\n({why})", (0.5, 0.5), xycoords="axes fraction", fontsize=8, ha="center")
        ax.set_xlabel(f"generated − reference R-peak time (ms), matched within ±{TOL_MS:.0f} ms", fontsize=7.5)
        ax.set_ylabel("matched beats", fontsize=7.5)
        ax.set_title(f"{lab} — matched-beat timing error", fontsize=8.5)
        for c in range(3):
            axs[r][c].grid(alpha=0.2); axs[r][c].tick_params(labelsize=7)
    cap = ("FIG 4 — beat-level failure analysis. THIS FIGURE EXISTS TO SHOW THAT AN HR METRIC CAN LOOK GOOD WHILE BEAT "
           "PLACEMENT FAILS. An HR scatter hugging the identity line and a narrow Bland–Altman band say only that the "
           "beat RATE was recovered over the window; the right-hand histogram, together with how few beats could be "
           f"matched at all within ±{TOL_MS:.0f} ms, says whether individual beats landed where they belong. Left and "
           "middle panels use the evaluator's per-window hr_ref / hr_pred over the WHOLE test population; the "
           f"histogram re-detects R peaks (repo neurokit path, both signals) on the first {budget} windows of "
           "waveforms_nfe<N>.npz in stored order — a deterministic subset chosen without looking at quality. Bias and "
           "limits of agreement are drawn on the Bland–Altman panel; they are descriptive, not a hypothesis test.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, out, "d1_fig4_beat_failure", cap)


def fig5_per_subject(evs, order, out, nfe_of, heads):
    fig, axs = plt.subplots(1, len(heads), figsize=(3.5 * len(heads), 4.5), squeeze=False)
    names = [label(evs[k]) for k in order]
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)                # jitter only; a fixed seed keeps the figure identical
    for a, m in zip(axs[0], heads):
        data = [R.subject_values(evs[k], m, nfe_of[k])[0] for k in order]
        data = [d[np.isfinite(d)] for d in data]
        a.boxplot([d if d.size else [np.nan] for d in data], positions=range(len(data)), widths=0.55,
                  showfliers=False, medianprops={"color": "0.15"})
        for i, d in enumerate(data):
            if d.size:
                a.scatter(i + rng.uniform(-0.16, 0.16, d.size), d, s=13, alpha=0.8, color=colour(i), edgecolors="none", zorder=3)
        a.set_xticks(range(len(data)))
        a.set_xticklabels([f"{n}\n(n={d.size})" for n, d in zip(names, data)], rotation=20, ha="right", fontsize=7)
        a.set_title(f"{R.SPEC[m].name}  {R.ARROW[R.SPEC[m].orientation]}", fontsize=8.5)
        a.set_ylabel(axis_label(m), fontsize=7.5); a.grid(alpha=0.2, axis="y"); a.tick_params(labelsize=7)
    cap = ("FIG 5 — within-dataset spread. One point per TEST SUBJECT (that subject's own window mean), overlaid on a "
           "box of the same values (box = quartiles, line = median, whiskers = 1.5 IQR; outliers are not re-drawn "
           "because every subject is already plotted, and the x-tick states n). This is the figure that stops a single "
           "mean from hiding a bimodal or single-subject-driven population. Several D1 corpora hold very few test "
           "subjects under the 70/15/15 split, so a box here can rest on a handful of points and must be read as such.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save(fig, out, "d1_fig5_per_subject", cap)


def fig6_corpus_size(evs, order, out, nfe_of, heads):
    sizes = {}
    for k in order:
        ss = R.split_sizes(evs[k]["corpus"], evs[k]["meta"])
        n = (ss or {}).get("train", {}).get("n_windows")
        if n is None:
            R.warn(f"FIG 6: no training-window count for {k} (needs the split manifest and the corpus "
                   "MANIFEST) — point omitted")
            continue
        sizes[k] = float(n)
    if not sizes:
        R.warn("FIG 6 skipped: no training-window counts available")
        return []
    fig, axs = plt.subplots(1, len(heads), figsize=(3.6 * len(heads), 4.1), squeeze=False)
    for a, m in zip(axs[0], heads):
        for k, n in sizes.items():
            s = R.stat(evs[k], m, nfe_of[k])
            if not np.isfinite(s["macro"]):
                continue
            col = colour(order.index(k))
            a.errorbar([n], [s["macro"]], yerr=[[max(s["macro"] - s["lo"], 0.0)], [max(s["hi"] - s["macro"], 0.0)]],
                       fmt="o", ms=6, color=col, capsize=3, lw=1.0, zorder=3)
            off = (6, 4) if order.index(k) % 2 == 0 else (6, -10)          # alternate so close points do not collide
            a.annotate(label(evs[k]), (n, s["macro"]), xytext=off, textcoords="offset points", fontsize=7)
        a.set_xscale("log"); a.set_xlabel("training windows (log scale)", fontsize=7.5)
        a.set_ylabel(axis_label(m), fontsize=7.5)
        a.set_title(f"{R.SPEC[m].name} vs corpus size", fontsize=8.5)
        a.grid(alpha=0.2); a.tick_params(labelsize=7)
    cap = ("FIG 6 — corpus-size confound panel. One point per dataset: x = number of TRAINING windows on a log scale, "
           "y = the headline metric at that dataset's headline NFE, whiskers = 95 % subject-clustered bootstrap CI. "
           "STATED PLAINLY: corpus size varies by roughly two orders of magnitude across D1 and is a CONFOUND BY "
           "CONSTRUCTION — it is entangled with dataset, sensor, population, native sampling rate, artifact policy and "
           "number of test subjects. NO CAUSAL CLAIM is made or may be read from this panel: it does not show that "
           "more training data causes better reconstruction, and it does not show that any dataset is intrinsically "
           "harder than another. It is here so that a size effect cannot be mistaken for a dataset effect.")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save(fig, out, "d1_fig6_corpus_size", cap)


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="D1 paper figures (prereg section 8)")
    ap.add_argument("--eval-root", default=str(ROOT / "outputs"), help="directory holding d1_<corpus>_seed42/eval/")
    ap.add_argument("--out-root", default=str(ROOT / "outputs/d1_bench"), help="figures land in <out-root>/figures/")
    ap.add_argument("--corpus", action="append", default=[], choices=list(C.BENCH_KEYS),
                    help="corpus key; repeatable. Exactly one of --corpus / --all is required")
    ap.add_argument("--all", action="store_true",
                    help=f"draw every benchmark corpus ({' '.join(C.BENCH_KEYS)}); exactly one of --corpus / --all "
                         "is required, so the corpus set is always stated explicitly")
    ap.add_argument("--beat-windows", type=int, default=BEAT_WINDOWS, help="FIG 4 beat-matching budget per corpus")
    ap.add_argument("--figures", default="1,2,3,4,5,6", help="comma-separated subset of FIG numbers")
    args = ap.parse_args()
    if bool(args.corpus) == bool(args.all):
        ap.error("give exactly one of --corpus (repeatable) or --all")

    eval_root = Path(args.eval_root)
    out = Path(args.out_root) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    want = {int(v) for v in args.figures.split(",") if v.strip()}

    evs, order, skipped = {}, [], []
    for key in (list(C.BENCH_KEYS) if args.all else args.corpus):
        c = C.corpus(key)
        ev = R.load_corpus(eval_root, c)
        if ev is None:
            skipped.append(key)
            R.warn(f"{key}: no evaluator output under {R.eval_dir(eval_root, c)} — corpus skipped in every figure")
            continue
        evs[key] = ev
        order.append(key)
    if not evs:
        raise SystemExit(f"no D1 evaluator output found under {eval_root}")
    nfe_of = {k: R.headline_nfe_of(evs[k]) for k in order}
    heads = [m for m in R.HEADLINE if m in R.SPEC]

    written = []
    if 1 in want:
        written += fig1_qualitative(evs, order, out, nfe_of)
    if 2 in want:
        written += fig2_metric_bars(evs, order, out, nfe_of, heads)
    if 3 in want:
        written += fig3_nfe_tradeoff(evs, order, out, heads)
    if 4 in want:
        written += fig4_beat_failure(evs, order, out, nfe_of, args.beat_windows)
    if 5 in want:
        written += fig5_per_subject(evs, order, out, nfe_of, heads)
    if 6 in want:
        written += fig6_corpus_size(evs, order, out, nfe_of, heads)

    (out / "figures_manifest.json").write_text(json.dumps(
        {"figures": sorted(written), "corpora": order, "skipped_corpora": skipped, "headline_nfe": nfe_of,
         "headline_metrics": heads, "eval_root": str(eval_root), "dpi": DPI,
         "window_selection": {"rule": "first test subject in natural order; stored-order positions "
                              + "/".join(str(p) for p in GRID_POS), "cherry_picked": False},
         "beat_windows": args.beat_windows, "recomputed": "FIG 4 beat matching only", "generator_run": False,
         "bootstrap": {"n": C.BOOTSTRAP_N, "seed": C.BOOTSTRAP_SEED}, "warnings": R.WARNINGS}, indent=2))
    print(json.dumps({"out": str(out), "files": sorted(written), "corpora": order, "skipped": skipped,
                      "warnings": len(R.WARNINGS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
