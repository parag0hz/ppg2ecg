"""E0 post-hoc descriptive readings (added AFTER the E0 outcomes; not preregistered, never used by the case rule).

1. Chance coincidence: for every WW-DET spontaneous detection (type C or D) the fraction of its window's samples lying
   within +-50 ms of a reference R of that window, i.e. the type-C share expected if spontaneous detections were placed
   at random times in their own windows. Compared with the observed type-C share, overall and by region.
2. Size of the detected structures: frozen feature definitions (prominence, amp_rel, qrs_corr) at WW-DET type A
   detections, for comparison with the preregistered type C / D values.
3. Where WW-DET's per-window F1 gain over the placed events comes from: windows without any placed event vs the rest
   (window-level sums divided by the number of evaluable windows; descriptive, not patient-weighted).

Run after `e0_event_audit.py evaluate`: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/e0_posthoc.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import e0_event_audit as E  # noqa: E402
from ppg2ecg.eventaudit import features as FT  # noqa: E402

TOL = E.TOL_MS / 1000.0 * E.FS


def coverage(ref) -> float:
    s = np.arange(E.T)
    r = np.asarray(ref, float)
    return float(np.mean(np.min(np.abs(s[:, None] - r[None, :]), axis=1) <= TOL)) if r.size else 0.0


def _a_feats(a):
    x, ts, template = a
    out = []
    for t in ts:
        f = FT.waveform_features(x, int(t), template)
        out.append((f["prominence"], f["amp_rel"], f["qrs_corr"]))
    return out


def main():
    import pandas as pd
    template = np.load(E.OUT / "qrs_template_train.npy")
    res = {}
    with ProcessPoolExecutor(10) as ex:
        for role in E.ROLES:
            X, Y, Pid, ref, events, waves, det = E.load_cache(role)
            df = pd.read_parquet(E.ART / f"event_level_{role}.parquet")
            ww = df[df["arm"] == "WW"]
            sp = ww[ww["type"].isin(["C", "D"])]
            cov = np.array([coverage(ref[i]) for i in sp["window"].to_numpy(int)])
            isc = (sp["type"] == "C").to_numpy()
            chance = {"all": {"n": int(len(sp)), "observed_C_share": float(isc.mean()), "expected_C_share_random_timing": float(cov.mean())}}
            for reg in ("no_placed", "before_first", "between", "after_last"):
                m = (sp["region"] == reg).to_numpy()
                if m.any():
                    chance[reg] = {"n": int(m.sum()), "observed_C_share": float(isc[m].mean()), "expected_C_share_random_timing": float(cov[m].mean())}
            a = ww[ww["type"] == "A"]
            by_w = a.groupby("window")["sample"].apply(list)
            jobs = [(waves["WW"][w], ts, template) for w, ts in by_w.items()]
            fa = np.array([v for lst in ex.map(_a_feats, jobs, chunksize=256) for v in lst])
            pa = np.concatenate([[Pid[w]] * len(ts) for w, ts in by_w.items()])
            size = {"WW_type_A": {k: _macro(fa[:, j], pa) for j, k in enumerate(("prominence", "amp_rel", "qrs_corr"))}}
            for ty in ("C", "D"):
                s = ww[ww["type"] == ty]
                size[f"WW_type_{ty}"] = {k: _macro(s[k].to_numpy(float), s["patient"].to_numpy()) for k in ("prominence", "amp_rel", "qrs_corr")}
            evaluable = np.array([len(r) > 0 for r in ref])
            m_p = E.event_metrics(Y, ref, events, evaluable)
            m_w = E.event_metrics(Y, ref, det["WW"], evaluable, waves["WW"])
            d = m_w["f1"] - m_p["f1"]
            noev = np.array([len(e) == 0 for e in events])
            n_eval = int(np.isfinite(d).sum())
            f1 = {"evaluable_windows": n_eval, "windows_without_events": int(noev.sum()),
                  "mean_window_dF1_all": float(np.nansum(d) / n_eval),
                  "contribution_from_windows_without_events": float(np.nansum(d[noev]) / n_eval),
                  "contribution_from_windows_with_events": float(np.nansum(d[~noev]) / n_eval),
                  "placed_F1_in_windows_without_events": float(np.nanmean(m_p["f1"][noev])),
                  "WW_F1_in_windows_without_events": float(np.nanmean(m_w["f1"][noev])),
                  "WW_windows_without_events_with_any_detection": int(sum(len(det["WW"][i]) > 0 for i in np.flatnonzero(noev)))}
            res[role] = {"chance_coincidence": chance, "structure_size": size, "f1_gain_location": f1}
            print(role, json.dumps(B.clean(res[role]))[:600], flush=True)
    E.write_json("posthoc_readings.json", {"status": "post-hoc descriptive readings added after the E0 outcomes; not preregistered; "
                                                     "not used by the case rule", **res})


def _macro(v, pid):
    v, pid = np.asarray(v, float), np.asarray(pid)
    per = [np.nanmedian(v[pid == p]) for p in np.unique(pid) if np.isfinite(v[pid == p]).any()]
    return {"median_of_patient_medians": float(np.median(per)) if per else float("nan"), "event_median": float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")}


if __name__ == "__main__":
    main()
