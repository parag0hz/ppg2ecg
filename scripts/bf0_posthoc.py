"""BF0 POST-HOC analyses (written after the BF0 result; not preregistered; never gate evidence).

Reads the frozen BF0 renders only (no training, no re-rendering, no new arm):
  1. G1 failure decomposition: pooled TP / FP / FN of the placed positions and of the A1 / A2 / A3 (s = 0) renders
     against the reference R peaks (±50 ms), and render-vs-placed misses / extra detections.
  2. FD on the windows that have at least one predicted position (the 438 windows without positions are flat-filled),
     with the preregistered patient-bootstrap procedure (`bf0_run.fd_diff_ci`) for A3 - A1 and A3 - A2 on that subset.
Output: artifacts/bf0_beat_first/posthoc.json
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/bf0_posthoc.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402


def counts(ref_list, hyp_list):
    tp = fp = fn = 0
    for r, h in zip(ref_list, hyp_list):
        m, f_p, f_n = RP.match_rpeaks(np.asarray(r, int), np.asarray(h, int), B.FS, 50.0)
        tp, fp, fn = tp + len(m), fp + f_p, fn + f_n
    p, r, f = RP.prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "micro_f1": f}


def main():
    _, Yv, Pv = B.load_role("val")
    R = dict(np.load(B.RUN / "render_val.npz"))
    pos = [np.asarray(p, int) for p in B.split_list(R["pos_idx"], R["pos_off"])]
    has_pos = np.array([len(p) > 0 for p in pos])
    S0 = np.load(B.ED1_VAL_CACHE, mmap_mode="r")[0].astype(np.float64)
    waves = {"A1": R["A1"], "A2": R["A2"], "A3": R["A3"][0]}
    with ProcessPoolExecutor(10) as ex:
        ref = B.val_reference_peaks(Yv, ex)
        det = {k: list(ex.map(B._peaks, list(np.asarray(w, np.float64)), chunksize=256)) for k, w in waves.items()}
    out = {"note": "POST-HOC; not preregistered; not gate evidence",
           "vs_reference_pooled": {"placed_positions": counts(ref, pos)} | {k: counts(ref, det[k]) for k in waves},
           "render_vs_placed": {k: {"placed": int(sum(len(p) for p in pos)),
                                    "placed_without_detection": counts(pos, det[k])["fn"],
                                    "detections_without_placed_beat": counts(pos, det[k])["fp"]} for k in waves},
           "fd_windows_with_positions": {"n_windows": int(has_pos.sum())}}
    for k, w in list(waves.items()) + [("imf_single", S0)]:
        out["fd_windows_with_positions"][k] = float(PMX.kanflow_fd(np.asarray(w, np.float64)[has_pos], Yv[has_pos]))
    sub = {k: np.asarray(w, np.float64)[has_pos] for k, w in waves.items()}
    for other in ("A1", "A2"):
        ci, n_min = B.fd_diff_ci(sub["A3"], sub[other], Yv[has_pos], Pv[has_pos])
        out["fd_windows_with_positions"][f"A3-{other}"] = ci
        out["fd_windows_with_positions"]["min_replicate_windows"] = n_min
    B.write_json("posthoc.json", out)
    print(out)


if __name__ == "__main__":
    main()
