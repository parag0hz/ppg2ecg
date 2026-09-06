"""M2 §9 — TRAIN12-only structure-map audit, gates S0-A..D (prereg §9). NO validation, NO training, NO GPU needed.

Run: .venv/bin/python scripts/m2_structure_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.flow.structure_weight import LAMBDA_STRUCT, structure_weight
from ppg2ecg.evaluation import rpeaks as R

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/m2_structure_weighted_imeanflow"
MANIFEST = ROOT / "data/manifests/split_a4_wildppg_seed42.json"
PROCESSED = ROOT / "data/processed/wildppg_8s"
FORBIDDEN = ("kjd", "ssx")
N_AUDIT, T, FS = 8192, 1024, 128


def load_train_windows(n: int, balanced: bool) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Deterministic TRAIN12 windows, no sampling and no outcome-blind choice.

    `balanced=False` is the preregistration's literal wording — the first `n` windows in manifest order over sorted
    subjects. Because e61 alone holds 22,228 windows, that draws all 8,192 from ONE subject, which cannot serve the
    audit's stated purpose of characterising the TRAIN12 map. `balanced=True` therefore takes the first ceil(n/12)
    windows of EACH of the twelve subjects. Both are run and both are reported (deviation D-2).
    """
    split = json.loads(MANIFEST.read_text())["splits"][0]
    assert not (set(split["train"]) & set(FORBIDDEN)), "forbidden subjects in the train list"
    subs = sorted(split["train"])
    per = -(-n // len(subs)) if balanced else n
    xs, keys = [], []
    for s in subs:
        d = np.load(PROCESSED / f"{s}.npz")
        y, wi = d["y"], d["window_index"]
        take = min(len(y), per, n - len(keys))
        if take <= 0:
            break
        xs.append(y[:take].astype(np.float64))
        keys += [(s, int(w)) for w in wi[:take]]
        if len(keys) >= n:
            break
    return np.concatenate(xs), keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_AUDIT)
    ap.add_argument("--qrs-diagnostic", action="store_true", default=True)
    ap.add_argument("--balanced", action="store_true", help="equal share per TRAIN12 subject (deviation D-2)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    y, keys = load_train_windows(args.n, args.balanced)
    assert y.shape[1] == T, y.shape
    x = torch.from_numpy(y).unsqueeze(1)
    w = structure_weight(x)[:, 0].numpy()                       # [n, T], mean 1 per window
    d = np.zeros_like(y)
    d[:, 1:] = y[:, 1:] - y[:, :-1]
    ad = np.abs(d)

    finite = np.isfinite(w).all(axis=1)
    mean_err = np.abs(w.mean(axis=1) - 1.0)
    ess_ratio = (w.sum(axis=1) ** 2) / (np.maximum((w ** 2).sum(axis=1), 1e-30)) / T
    # top-20% weight positions: how much |d| energy do they capture?
    k = int(round(0.20 * T))
    top = np.argpartition(-w, k, axis=1)[:, :k]
    tot = ad.sum(axis=1)
    conc = np.take_along_axis(ad, top, axis=1).sum(axis=1) / np.maximum(tot, 1e-30)
    from ppg2ecg.flow.structure_weight import structure_map   # m recomputed from the definition, never inverted from w
    m = structure_map(x)[:, 0].numpy()
    q95 = np.percentile(np.abs(np.stack([np.convolve(np.abs(dd), np.array([1, 4, 6, 4, 1]) / 16.0, "same") for dd in d])), 95, axis=1)

    gates = {
        "S0-A_all_finite": bool(finite.all()),
        "S0-B_mean_normalisation": bool((mean_err <= 1e-6).all()),
        "S0-C_top20_derivative_energy_median_gt_0.50": bool(float(np.median(conc)) > 0.50),
        "S0-D_ess_ratio_ge_0.50_for_99.5pct": bool((ess_ratio >= 0.50).mean() >= 0.995),
    }
    summary = {
        "n_windows": int(len(y)), "n_subjects": len({s for s, _ in keys}),
        "subjects": sorted({s for s, _ in keys}), "sampling": "balanced per subject" if args.balanced else "literal prereg wording (manifest order)", "T": T,
        "lambda_struct": LAMBDA_STRUCT,
        "weight": {"min": float(w.min()), "mean": float(w.mean()), "max": float(w.max())},
        "finite_fraction": float(finite.mean()),
        "mean_normalisation_error": {"max": float(mean_err.max()), "mean": float(mean_err.mean())},
        "q95": {"min": float(q95.min()), "median": float(np.median(q95)), "max": float(q95.max())},
        "frac_m_gt_0.5": float((m > 0.5).mean()), "frac_m_gt_0.8": float((m > 0.8).mean()),
        "top20_derivative_energy_concentration": {
            "median": float(np.median(conc)), "mean": float(conc.mean()),
            "p05": float(np.percentile(conc, 5)), "p95": float(np.percentile(conc, 95))},
        "ess_over_T": {"min": float(ess_ratio.min()), "median": float(np.median(ess_ratio)),
                       "frac_ge_0.50": float((ess_ratio >= 0.50).mean())},
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
    }

    if args.qrs_diagnostic:   # prereg §9 diagnostic ONLY — no gate, no effect on the map
        excess = w - w.min(axis=1, keepdims=True)
        inside, total = 0.0, 0.0
        for i in range(len(y)):
            pk = R.detect_rpeaks(y[i], FS, "neurokit")
            mask = np.zeros(T, dtype=bool)
            for c in np.asarray(pk, dtype=int):
                mask[max(c - 10, 0):min(c + 11, T)] = True
            inside += float(excess[i][mask].sum())
            total += float(excess[i].sum())
        summary["qrs_concentration_diagnostic"] = {
            "note": "DIAGNOSTIC ONLY (prereg §9): GT R peaks describe where weight falls; no R annotation enters arm S and no gate depends on this",
            "fraction_of_excess_weight_within_R_pm10": inside / max(total, 1e-30)}

    (OUT / f"train_structure_map_summary{args.tag}.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / f"train_structure_map_audit{args.tag}.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["subject", "window_index", "w_min", "w_mean", "w_max", "mean_err", "ess_over_T", "top20_deriv_energy"])
        for (s, wi), a_, b_, c_, e_, f_, g_ in zip(keys, w.min(1), w.mean(1), w.max(1), mean_err, ess_ratio, conc):
            wr.writerow([s, wi, f"{a_:.6f}", f"{b_:.6f}", f"{c_:.6f}", f"{e_:.3e}", f"{f_:.6f}", f"{g_:.6f}"])

    print(json.dumps(summary, indent=1))
    print("\nS0 VERDICT:", "PASS — training may proceed" if summary["all_gates_pass"]
          else "FAIL -> verdict E, STRUCTURAL WEIGHT MAP INVALID, STOP")
    return 0 if summary["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
