"""PPGFlowECG x VitalDB 10-s evaluation windows (docs/PPGFLOWECG_VITALDB_DATA_PREREGISTRATION.md, frozen at fc99f76).

V1 test cases only, in manifest order. For every V1 test window (case, window_index k) the raw 500-Hz span
[k*2000, k*2000 + 5000) of PLETH / ECG_II is taken; an anchor is excluded only if the span runs past the end of either
channel, contains a non-finite sample, or either channel is constant over it. Kept windows go through the official
PPGFlowECG transforms, executed verbatim (official.load_prep, paper-pinned mne 1.8.0):
  PPG: ppg_clean_elgendi_mne(sr=500, 0.5, 8) -> data_resampler(500 -> 128) -> data_normalizer -> _batch_savgol(7, 2)
  ECG: ecg_clean_nk_mne(sr=500, 0.5, None)   -> data_resampler(500 -> 128) -> data_normalizer -> _batch_savgol(11, 2)
No quality selection, no ECG polarity check (stated in the preregistration).

Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=outputs/pfe_env/paper_eval:scripts/external/ppgflowecg \
     .venv/bin/python scripts/external/ppgflowecg/build_vitaldb10.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import official as O

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/raw/VitalDB/cases"
V1 = ROOT / "data/processed/v1_vitaldb"
OUT = ROOT / "outputs/ppgflowecg_external/data"
ART = ROOT / "artifacts/ppgflowecg_external"
FS_RAW, FS = 500, 128
V1_STRIDE, SPAN = 4 * FS_RAW, 10 * FS_RAW          # V1 window k starts at k*2000; the 10-s span is 5000 raw samples
_P = None


def _prep():
    global _P
    if _P is None:
        _P = O.load_prep()
    return _P


def one(case):
    P_ = _prep()
    v = np.load(V1 / f"{case}.npz")
    widx, pid = v["window_index"].astype(int), int(v["subjectid"])
    with np.load(RAW / f"{case}.npz") as z:
        P, E = z["PLETH"].astype(np.float64), z["ECG_II"].astype(np.float64)
    keep, excl, rp, re, y4 = [], [], [], [], []
    for j, k in enumerate(widx):
        s, e = k * V1_STRIDE, k * V1_STRIDE + SPAN
        if e > len(P) or e > len(E):
            excl.append((int(k), "span_past_end")); continue
        p, q = P[s:e], E[s:e]
        if not (np.isfinite(p).all() and np.isfinite(q).all()):
            excl.append((int(k), "non_finite")); continue
        if np.ptp(p) == 0 or np.ptp(q) == 0:
            excl.append((int(k), "constant")); continue
        keep.append(j); rp.append(p); re.append(q); y4.append(v["y"][j])
    if not keep:
        return case, pid, widx, keep, excl, None, None, None
    p = P_["ppg_clean_elgendi_mne"](np.array(rp), FS_RAW, 0.5, 8)
    q = P_["ecg_clean_nk_mne"](np.array(re), FS_RAW, 0.5, None)
    p = P_["data_normalizer"](None, P_["data_resampler"](None, p, FS_RAW, FS))
    q = P_["data_normalizer"](None, P_["data_resampler"](None, q, FS_RAW, FS))
    p = P_["_batch_savgol"](p.astype(np.float32), 7, 2)
    q = P_["_batch_savgol"](q.astype(np.float32), 11, 2)
    return case, pid, widx, keep, excl, p, q, np.array(y4, np.float64)


def main():
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]["test"]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(one, cases, chunksize=4))
    ppg, ecg, pat, cid, aw, order, excluded, y4 = [], [], [], [], [], [], [], []
    base, overl = 0, 0
    for case, pid, widx, keep, excl, p, q, y in res:
        for k, why in excl:
            excluded.append({"case": case, "window_index": k, "reason": why})
        if keep:
            ppg.append(p); ecg.append(q); y4.append(y)
            pat += [pid] * len(keep); cid += [int(case.split("_")[1])] * len(keep)
            aw += [int(widx[j]) for j in keep]; order += [base + j for j in keep]
            kk = np.array([widx[j] for j in keep])
            overl += int(sum(abs(kk[b] - kk[a]) * V1_STRIDE < SPAN for a in range(len(kk)) for b in range(a + 1, len(kk))))
        base += len(widx)
    ppg, ecg = np.concatenate(ppg).astype(np.float32), np.concatenate(ecg).astype(np.float32)
    y4 = np.concatenate(y4)
    assert base == 19543, base
    assert ppg.shape[1] == ecg.shape[1] == 1280 and np.isfinite(ppg).all() and np.isfinite(ecg).all()
    np.savez(OUT / "vitaldb10_test.npz", ppg=ppg, ecg=ecg, patient=np.array(pat), caseid=np.array(cid),
             anchor_window_index=np.array(aw), v1_order=np.array(order))
    # construction sanity (no model involved): the first 4 s of the processed 10-s ECG vs V1's target ECG of the anchor
    zc = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)  # noqa: E731
    corr = (zc(ecg[:, :512].astype(np.float64)) * zc(y4)).mean(1)
    reasons = {}
    for d in excluded:
        reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1
    info = {"prereg": "docs/PPGFLOWECG_VITALDB_DATA_PREREGISTRATION.md (fc99f76)",
            "v1_test_cases": len(cases), "v1_test_anchors": base, "kept_windows": int(len(ppg)),
            "kept_patients": int(len(np.unique(pat))), "kept_cases": int(len(np.unique(cid))),
            "excluded_anchors": len(excluded), "excluded_by_reason": reasons, "excluded_list": excluded,
            "overlapping_pairs_within_case": overl,
            "windows_per_patient": {"median": float(np.median(np.unique(pat, return_counts=True)[1])),
                                    "min": int(np.unique(pat, return_counts=True)[1].min()),
                                    "max": int(np.unique(pat, return_counts=True)[1].max())},
            "shape": [int(len(ppg)), 1280], "sampling_rate_hz": FS,
            "per_window_mean_std_after_savgol": {"ppg": [float(ppg.mean(1).mean()), float(ppg.std(1).mean())],
                                                 "ecg": [float(ecg.mean(1).mean()), float(ecg.std(1).mean())]},
            "sanity_corr_first4s_processed_ecg_vs_v1_target": {"median": float(np.median(corr)), "p5": float(np.percentile(corr, 5)),
                                                                "min": float(corr.min())},
            "official_function_source_sha256": O.source_hashes()}
    (ART / "data_build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k not in ("excluded_list", "official_function_source_sha256")}, indent=1))


if __name__ == "__main__":
    main()
