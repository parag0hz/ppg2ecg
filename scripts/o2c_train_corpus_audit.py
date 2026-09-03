"""O2c step 11 — full train-corpus integer-warp audit (preregistration section 5).

Also caches the GT R schedule of every training window so that training does not re-detect them.
No model is built or trained here; validation subjects are never loaded.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.training.train_a0 import git_sha, load_arrays, read_manifest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
CACHE = ART / "_cache_train_rpeaks.npz"
MANIFEST = "data/manifests/split_a4_wildppg_seed42.json"
PROCESSED = "data/processed/wildppg_8s"
K3_BUDGET = 0.005


def _peaks(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), 128)


def audit_one(args):
    """(K, identity, status, min_int_spacing, max_core_frac, max_abs_q_shift, valid)"""
    pk = args
    w = BW.IntegerEventWarp(pk)
    if w.identity:
        return (len(pk), 1, w.status, -1, 0.0, 0.0, 1)
    qi = np.asarray(w.q, np.int64)
    return (len(pk), 0, w.status, int(np.diff(qi).min()), float(w.core_offsets().max()),
            float(np.max(np.abs(qi - np.asarray(w.q_real, float)))), int(w.valid()))


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    split = read_manifest(ROOT / MANIFEST)[0]
    ER.assert_no_test_subjects(split["train"])
    assert not (set(split["train"]) & {"an0", "k2s"}), "training corpus must not contain validation subjects"
    x, y, _ = load_arrays(ROOT / PROCESSED, split["train"], None)
    print(f"[corpus] {len(x)} windows from {len(split['train'])} subjects", flush=True)

    if CACHE.exists():
        z = np.load(CACHE)
        flat, off = z["flat"], z["offsets"]
    else:
        with ProcessPoolExecutor(max_workers=12) as ex:
            pks = list(ex.map(_peaks, list(y.astype(np.float64)), chunksize=64))
        off = np.cumsum([0] + [len(p) for p in pks]).astype(np.int64)
        flat = np.concatenate(pks).astype(np.int32) if len(pks) else np.zeros(0, np.int32)
        np.savez_compressed(CACHE, flat=flat, offsets=off)
    pks = [flat[off[i]:off[i + 1]].astype(np.int64) for i in range(len(off) - 1)]
    print(f"[peaks] {sum(len(p) for p in pks)} GT beats cached in {time.perf_counter()-t0:.0f} s", flush=True)

    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(audit_one, pks, chunksize=256))
    K = np.array([r[0] for r in res]); ident = np.array([r[1] for r in res], bool)
    status = [r[2] for r in res]; spacing = np.array([r[3] for r in res])
    core = np.array([r[4] for r in res]); qshift = np.array([r[5] for r in res]); valid = np.array([r[6] for r in res], bool)
    n_k3 = int(sum(1 for s in status if s == "K<3"))
    n_space = int(sum(1 for s in status if s == "integer spacing violated"))
    n_bad = int(ident.sum()) - n_k3 - n_space
    rows = [{"K": int(k), "count": int(c)} for k, c in sorted(Counter(K.tolist()).items())]
    with open(ART / "train_corpus_warp_audit.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=["K", "count"]); w_.writeheader(); w_.writerows(rows)
    out = {"git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
           "validation_subjects_loaded": [], "subjects": list(split["train"]), "n_windows": int(len(x)),
           "n_gt_beats": int(sum(len(p) for p in pks)),
           "K_min": int(K.min()), "K_max": int(K.max()), "K_median": float(np.median(K)),
           "identity_K_lt_3": n_k3, "identity_K_lt_3_fraction": n_k3 / len(x), "k3_budget": K3_BUDGET,
           "spacing_violations": n_space, "other_invalid": n_bad,
           "min_int_spacing": int(spacing[~ident].min()) if (~ident).any() else None,
           "min_int_spacing_required": BW.MIN_INT_SPACING,
           "max_core_fractional_coordinate": float(core.max()), "core_tolerance": BW.CORE_OFFSET_TOL,
           "max_abs_q_shift": float(qshift.max()), "all_valid": bool(valid.all()),
           "libs": {"numpy": np.__version__, "python": platform.python_version()},
           "wall_s": time.perf_counter() - t0}
    out["stop_A_invalid_warp"] = bool(n_bad > 0)
    out["stop_B_k3_fraction"] = bool(out["identity_K_lt_3_fraction"] > K3_BUDGET)
    out["stop_C_spacing_violation"] = bool(n_space > 0)
    out["passed"] = not (out["stop_A_invalid_warp"] or out["stop_B_k3_fraction"] or out["stop_C_spacing_violation"]) and out["all_valid"]
    (ART / "train_corpus_warp_audit.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[audit] K {out['K_min']}..{out['K_max']} | K<3 {n_k3} ({out['identity_K_lt_3_fraction']:.5%}) | "
          f"spacing violations {n_space} | other invalid {n_bad} | min int spacing {out['min_int_spacing']} | "
          f"max core frac {out['max_core_fractional_coordinate']:.3e} | max |q_int-q_real| {out['max_abs_q_shift']:.3f}", flush=True)
    print(f"[audit] {'PASS' if out['passed'] else 'FAIL -> TRAIN-CORPUS CANONICALIZATION INVALID'}", flush=True)
    if not out["passed"]:
        raise RuntimeError("TRAIN-CORPUS CANONICALIZATION INVALID (STOP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
