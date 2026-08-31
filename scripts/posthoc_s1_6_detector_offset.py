"""POST-HOC (prereg section 7: labelled, additive, never substitutive).

S1.6 reports near-zero positional agreement between the two detectors despite near-equal beat counts.
That pattern is more consistent with a systematic offset in one detector than with genuine ambiguity
about where the beats are. This measures the signed offset. It ADDS to S1.6; it replaces nothing.
"""
import ppg2ecg.utils.mkl_warmup  # noqa: F401
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from ppg2ecg.evaluation import event_reliability as ER, rpeaks as R, s1_audit as S1

ROOT = Path("/home/kwy00/ppg2ecg-one-step"); FS = 128
ER.assert_no_test_subjects(["an0", "k2s"])

def pa(y): return R.detect_rpeaks(np.asarray(y, float), FS, S1.DETECTOR_A)
def pb(y): return R.detect_rpeaks(np.asarray(y, float), FS, S1.DETECTOR_B)

Y, SUB = [], []
for s in ("an0", "k2s"):
    d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
    idx = ER.select_subset("x4-event-nfe-v2", s, len(d["x"]), 1024)
    Y += [d["y"][int(i)].astype(np.float64) for i in idx]; SUB += [s] * len(idx)
SUB = np.array(SUB)
with ProcessPoolExecutor(max_workers=12) as ex:
    A = list(ex.map(pa, Y, chunksize=16)); B = list(ex.map(pb, Y, chunksize=16))

# signed nearest-partner offset B - A, no tolerance cap
off = []
for a, b in zip(A, B):
    if len(a) and len(b):
        j = np.argmin(np.abs(a[:, None] - b[None, :]), axis=1)
        off.append(b[j] - a)
off = np.concatenate(off)
med = float(np.median(off))
res = {"POST_HOC": True, "n_pairs": int(off.size),
       "median_offset_samples": med, "median_offset_ms": med / FS * 1000,
       "iqr_samples": [float(np.percentile(off, 25)), float(np.percentile(off, 75))],
       "frac_within_6_samples": float(np.mean(np.abs(off) <= 6))}

# re-score agreement after removing the single global median offset (POST-HOC, diagnostic only)
f1_shift = []
for a, b in zip(A, B):
    m, fp, fn = R.match_rpeaks(a, b - int(round(med)), FS, 50.0)
    f1_shift.append(R.prf(len(m), fp, fn)[2])
res["f1_macro_after_removing_median_offset"] = S1.macro(f1_shift, SUB)
for s in ("an0", "k2s"):
    res[f"f1_after_offset__{s}"] = float(np.mean(np.asarray(f1_shift)[SUB == s]))
print(json.dumps(res, indent=2))
Path(ROOT / "artifacts/s1_metric_validity/s1_6_posthoc_detector_offset.json").write_text(json.dumps(res, indent=2))
