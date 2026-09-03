"""O2c step 12 — mandatory O2b Stage-0 regression guard, run before any weight update.

Re-runs the accepted integer-grid operator's round-trip on the exact frozen 2,048-window cohort with the exact O2
metric code and the exact frozen O2 Stage-0 gate, and compares against the frozen O2b reference medians.
No generator is built, loaded or trained anywhere in this script.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o2_warp as O2
from ppg2ecg.evaluation import o2b_warp as B
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
O2BART = ROOT / "artifacts/o2b_integer_grid"
O1ART = ROOT / "artifacts/o1_component_extractability"
if "o2_stage0_roundtrip" in sys.modules:
    S0 = sys.modules["o2_stage0_roundtrip"]
else:
    _s = importlib.util.spec_from_file_location("o2_stage0_roundtrip", ROOT / "scripts/o2_stage0_roundtrip.py")
    S0 = importlib.util.module_from_spec(_s); sys.modules[_s.name] = S0; _s.loader.exec_module(S0)
VAL, ALIGNED = ("an0", "k2s"), S0.ALIGNED
REF_TOL = 1e-12
VERDICT_FAIL = "OPERATOR REGRESSION DETECTED"


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t0 = time.perf_counter()
    X, Y, SUB, SITE, POS, WI = S0.load_cohort()
    Yd = Y.astype(np.float64)
    gt_pk = S0.pmap(S0._peaks, list(Yd))
    gt_tg = S0.pmap(S0._targets, list(Yd))
    n_beats = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_beats != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {n_beats} beats (STOP)")
    iqr = {t: json.loads((O1ART / "target_scaling.json").read_text())["targets"][t]["scale_train_IQR"] for t in ALIGNED}

    warps = [B.IntegerEventWarp(gt_pk[i]) for i in range(len(Y))]
    rows, med = S0.roundtrip_metrics(Y, warps, gt_pk, gt_tg, iqr, "integer_grid")
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(ART / "stage0_regression_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)

    gate = O2.roundtrip_gate(med)
    checks = {"R0-1": med["raw_rmse"] <= 0.020, "R0-2": med["T6"] <= 0.020, "R0-3": med["T7"] <= 0.020,
              "R0-4a": med["T4"] <= 0.020, "R0-4b": med["T8"] <= 0.020,
              "R0-5": med["f1_at_50"] >= 0.98, "R0-6": med["beat_count_diff"] == 0.0}
    ref = json.loads((O2BART / "decision.json").read_text())["medians"]
    diffs = {k: abs(float(med[k]) - float(ref[k])) for k in ref}
    max_diff = max(diffs.values())
    identical = max_diff <= REF_TOL
    passed = bool(gate["passed"] and all(checks.values()) and identical)
    ident = json.loads((ART / "operator_identity.json").read_text())
    out = {"verdict": "OPERATOR REGRESSION GUARD PASSED" if passed else VERDICT_FAIL,
           "gate_passed": bool(gate["passed"]), "checks": checks, "gate": gate,
           "reference_source": "artifacts/o2b_integer_grid/decision.json (frozen O2b Stage-0)",
           "reference_medians": ref, "reproduced_medians": med, "abs_diff": diffs,
           "max_abs_diff": max_diff, "tolerance": REF_TOL, "identical_to_o2b": bool(identical),
           "operator_identity": ident, "generator_trained": False,
           "git": git_sha(ROOT), "prereg": "d458895", "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "n_windows": int(len(X)), "n_gt_beats": n_beats,
           "n_identity_rows": int(sum(w.identity for w in warps)),
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "wall_s": time.perf_counter() - t0}
    (ART / "stage0_regression_guard.json").write_text(json.dumps(out, indent=2, default=float))
    print(" ".join(f"{k}:{'PASS' if v else 'FAIL'}" for k, v in checks.items()) +
          f" | max |Δ| vs O2b {max_diff:.3e} | {out['verdict']}", flush=True)
    if not passed:
        raise RuntimeError(f"{VERDICT_FAIL} (STOP): no training, no repair under O2c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
