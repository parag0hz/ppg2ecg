"""N4 — is the generator's timing variability a defect, or a posterior?
(docs/N4_TIMING_CALIBRATION_PREREGISTRATION.md)

NO TRAINING, NO WEIGHT UPDATE, NO OPTIMIZER. The frozen iMeanFlow generator X4-0/R2/R3 used,
re-run on X4-0's frozen source subset with X4-0's 32 sources, collecting the RAW per-source
offsets around each GT anchor so spread can be compared against error.

Run: .venv/bin/python scripts/n4_calibration.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/n4_timing_calibration"
PREREG = "4d14865"
VAL = ("an0", "k2s")
IMF_CKPT = "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt"
EXPECTED_STATE_SHA = "47d7ccb94e5dbf7190d777f852b18f107f3ce2628d160b5e01ff96ef2a1d0d0f"
SOURCE_SEEDS = tuple(range(32))
NFES = (1, 4)
MIN_DETECT = 16                         # X4-0's frozen >= 16/32 filter
GT_ANCHOR_MS = ER.GT_ANCHOR_MS          # 150.0, X4-0's window, unchanged
FS, T_LEN = 128, 1024
ALPHAS = (0.50, 0.20, 0.10)
TOL = 0.10                              # prereg §5
BOOT_N, BOOT_SEED = 2000, 20260911
BATCH = 256


# the repository's canonical digest (R2/R3 use it; it produced the frozen 47d7ccb9...),
# never a hand-rolled one
from ppg2ecg.flow.rhythm_transfer import state_dict_sha256 as sd_sha  # noqa: E402


def load_cohort():
    sub = json.loads((ROOT / "artifacts/x4_0_event_reliability/source_subset.json").read_text())
    ER.assert_no_test_subjects(list(sub))
    X, Y, S, W = [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = np.asarray(sub[s], dtype=int)
        assert idx.tolist() == list(sub[s]), "frozen subset order changed"
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float64))
        S.append(np.full(len(idx), s)); W.append(idx)
    return (np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(W))


def _pk(a):
    return [R.detect_rpeaks(w, FS) for w in a]


def peaks(arr, workers=12):
    ch = [arr[i:i + 64] for i in range(0, len(arr), 64)]
    with ProcessPoolExecutor(workers) as ex:
        return [p for part in ex.map(_pk, ch) for p in part]


@torch.no_grad()
def generate(net, X, seed, nfe, dev):
    g = torch.Generator().manual_seed(int(seed))
    e = torch.randn(len(X), 1, T_LEN, generator=g)
    out = []
    for i in range(0, len(X), BATCH):
        p = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, p, e[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        assert int(k) == nfe
        out.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def offsets_per_beat(gt_pk, pred_pk_per_source):
    """Raw per-source offsets (ms) around each GT anchor. gt_anchored_presence collapses these to
    mean/SD; calibration needs the raw set, so it is recomputed here with the identical rule."""
    half = GT_ANCHOR_MS / 1000.0 * FS
    per = []
    for i, g in enumerate(gt_pk):
        offs = []
        for pk in pred_pk_per_source:
            if len(pk) == 0:
                continue
            d = np.asarray(pk, float) - float(g)
            j = int(np.argmin(np.abs(d)))
            if abs(d[j]) <= half:
                offs.append(float(d[j]) / FS * 1000.0)
        per.append(np.asarray(offs))
    return per


def coverage(offs: np.ndarray, alpha: float) -> bool:
    lo, hi = np.quantile(offs, [alpha / 2, 1 - alpha / 2])
    return bool(lo <= 0.0 <= hi)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    X, Y, S, W = load_cohort()
    print(f"[N4] cohort {len(X)} windows, subjects {sorted(set(S))}", flush=True)

    ck = torch.load(ROOT / IMF_CKPT, map_location="cpu", weights_only=False)
    assert sd_sha(ck["state_dict"]) == EXPECTED_STATE_SHA, "frozen generator state differs"
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)

    gt_pk = peaks(Y)
    n_gt = sum(len(p) for p in gt_pk)
    print(f"[N4] {n_gt:,} GT beats", flush=True)

    results, rows = {}, []
    for nfe in NFES:
        pred_pk = []
        for sd_ in SOURCE_SEEDS:
            pred_pk.append(peaks(generate(net, X, sd_, nfe, dev)))
        print(f"[N4] NFE {nfe}: generated {len(SOURCE_SEEDS)} sources", flush=True)

        cov = {a: [] for a in ALPHAS}
        pit, spread, bias, subj, n_elig, n_total = [], [], [], [], 0, 0
        for i in range(len(X)):
            per = offsets_per_beat(gt_pk[i], [pred_pk[k][i] for k in range(len(SOURCE_SEEDS))])
            for offs in per:
                n_total += 1
                if len(offs) < MIN_DETECT:
                    continue
                n_elig += 1
                for a in ALPHAS:
                    cov[a].append(coverage(offs, a))
                pit.append(float((np.sum(offs < 0.0) + 0.5 * np.sum(offs == 0.0) + 0.5) / (len(offs) + 1)))
                spread.append(float(np.std(offs, ddof=1)))
                bias.append(float(np.mean(offs)))
                subj.append(S[i])
        subj = np.asarray(subj)
        macro = lambda v: float(np.mean([np.nanmean(np.asarray(v, float)[subj == s]) for s in np.unique(subj)]))  # noqa: E731
        rec = {"nfe": nfe, "n_gt_beats": int(n_total), "n_eligible": int(n_elig),
               "detection_rate": float(n_elig / max(n_total, 1)),
               "spread_ms": macro(spread), "abs_bias_ms": macro(np.abs(bias)),
               "pit_ks": float(sps.kstest(pit, "uniform").statistic),
               "spread_vs_absbias_spearman": float(sps.spearmanr(spread, np.abs(bias)).statistic),
               "pit_hist_10bins": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
               "pit_ks_pvalue": float(sps.kstest(pit, "uniform").pvalue),
               "bias_mean_ms": macro(bias), "bias_median_ms": float(np.median(bias)),
               "spread_median_ms": float(np.median(spread)),
               "absbias_over_spread_median": float(np.median(np.abs(bias)) / max(np.median(spread), 1e-9))}
        for a in ALPHAS:
            c = macro(cov[a])
            rec[f"coverage_{int((1-a)*100)}"] = c
            rec[f"delta_{int((1-a)*100)}"] = c - (1 - a)
            ci = paired_subject_bootstrap(np.full(len(cov[a]), 1 - a), np.asarray(cov[a], float),
                                          subj, "higher_better", BOOT_N, BOOT_SEED)
            rec[f"delta_{int((1-a)*100)}_ci"] = [ci["lo"], ci["hi"]]
        results[nfe] = rec
        rows.append(rec)
        print(f"[N4] NFE {nfe}: eligible {n_elig:,}/{n_total:,} ({rec['detection_rate']:.3f})  "
              f"spread {rec['spread_ms']:.1f} ms  |bias| {rec['abs_bias_ms']:.1f} ms", flush=True)
        for a in ALPHAS:
            k = int((1 - a) * 100)
            print(f"        coverage@{k}% = {rec[f'coverage_{k}']:.3f}  (nominal {1-a:.2f}, "
                  f"delta {rec[f'delta_{k}']:+.3f})", flush=True)
        print(f"        PIT KS {rec['pit_ks']:.3f}   spread-vs-|bias| Spearman {rec['spread_vs_absbias_spearman']:+.3f}", flush=True)

    def verdict(rec):
        d = [rec[f"delta_{int((1-a)*100)}"] for a in ALPHAS]
        if all(abs(x) <= TOL for x in d):
            return "TIMING POSTERIOR CALIBRATED"
        if sum(x <= -TOL for x in d) >= 2:
            return "OVERCONFIDENT"
        if sum(x >= TOL for x in d) >= 2:
            return "UNDERCONFIDENT"
        return "MISCALIBRATED (other)"

    for nfe in NFES:
        results[nfe]["verdict"] = verdict(results[nfe])
        print(f"[N4] NFE {nfe} VERDICT: {results[nfe]['verdict']}", flush=True)

    out = {"prereg": PREREG, "git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "generator": {"path": IMF_CKPT, "state_sha256": EXPECTED_STATE_SHA},
           "cohort": {"windows": int(len(X)), "subjects": sorted(set(S.tolist()))},
           "config": {"sources": len(SOURCE_SEEDS), "nfes": list(NFES), "min_detect": MIN_DETECT,
                      "gt_anchor_ms": GT_ANCHOR_MS, "alphas": list(ALPHAS), "tolerance": TOL},
           "results": results,
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n4_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N4] wrote {ART}/n4_results.json  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
