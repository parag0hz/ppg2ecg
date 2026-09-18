"""PZ2 — calibration-budget curve and the SBV difficulty axis (docs/PZ2_CALIBRATION_BUDGET_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pz1_personalisation import QRS, RI, beats_of, case_path, ci, train_beats  # noqa: E402
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.evaluation.stamping import build_template_a, stamp  # noqa: E402

FS, KS, N_CAL_MAX, N_TRAIN_CASES = 128, (1, 2, 4, 8), 8, 200
OUT = ROOT / "artifacts/pz2_calibration_budget"


def one_case(job):
    case, g = job
    d = np.load(case_path(case))
    y, wi, pid = d["y"].astype(np.float64), d["window_index"], int(d["subjectid"])
    order = np.argsort(wi)
    if len(order) <= N_CAL_MAX:
        return []
    cal_beats_by_k, beats_running = {}, []
    for j, i in enumerate(order[:N_CAL_MAX]):
        beats_running += beats_of(y[i], detect_rpeaks(y[i], FS))
        if (j + 1) in KS:
            cal_beats_by_k[j + 1] = list(beats_running)
    tmpl = {k: (build_template_a(np.stack(v))[0] if len(v) >= 5 else g) for k, v in cal_beats_by_k.items()}
    full = cal_beats_by_k[N_CAL_MAX]
    if len(full) >= 5:
        t8 = tmpl[N_CAL_MAX]
        sbv = float(np.median([1 - np.corrcoef(b, t8)[0, 1] for b in full if b.std() > 0 and t8.std() > 0]))
    else:
        sbv = np.nan
    rows = []
    for i in order[N_CAL_MAX:]:
        sig = y[i]
        pk = detect_rpeaks(sig, FS)
        tb = beats_of(sig, pk)
        if len(tb) < 2:
            continue
        tb = np.stack(tb)
        arms = {"GLOBAL": g, "ORACLE": build_template_a(tb)[0], **{f"k{k}": tmpl[k] for k in KS}}
        row = {"case": case, "pid": pid, "window": int(wi[i]), "sbv": sbv, "n_cal_beats": len(full)}
        for name, t in arms.items():
            pred = stamp(t, pk, sig.size, RI)
            pb = np.stack(beats_of(pred, pk)[: len(tb)]); tt = tb[: len(pb)]
            cc = [np.corrcoef(p, q)[0, 1] if p.std() > 0 and q.std() > 0 else np.nan for p, q in zip(pb, tt)]
            row[f"{name}.morph_corr"] = float(np.nanmean(cc))
            row[f"{name}.qrs_core_rmse"] = float(np.sqrt(np.mean((pb[:, QRS] - tt[:, QRS]) ** 2)))
            row[f"{name}.mae"] = float(np.mean(np.abs(pred - sig)))
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    g = build_template_a(train_beats(sorted(split["train"])[:N_TRAIN_CASES]))[0]
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one_case, [(c, g) for c in split["test"]], chunksize=8), 1):
            rows += r
            if i % 300 == 0:
                print(f"[pz2] {i}/{len(split['test'])} cases", flush=True)
    pid = np.array([r["pid"] for r in rows])
    arms = ["GLOBAL"] + [f"k{k}" for k in KS] + ["ORACLE"]
    curve = {a: {m: ci(np.array([r[f"{a}.{m}"] for r in rows]), pid) for m in ("morph_corr", "qrs_core_rmse", "mae")} for a in arms}
    gains = {f"k{k}": ci(np.array([r[f"k{k}.morph_corr"] - r["GLOBAL.morph_corr"] for r in rows]), pid) for k in KS}
    gap = curve["ORACLE"]["morph_corr"][0] - curve["GLOBAL"]["morph_corr"][0]
    sat = {}
    for k in KS:
        if 2 * k in KS:
            sat[f"{k}->{2 * k}"] = curve[f"k{2 * k}"]["morph_corr"][0] - curve[f"k{k}"]["morph_corr"][0]
    saturated_at = next((k for k in KS if f"{k}->{2 * k}" in sat and sat[f"{k}->{2 * k}"] < 0.01), None)
    sbv = np.array([r["sbv"] for r in rows])
    qs = np.nanpercentile(sbv, [33.33, 66.67])
    tert = {}
    for name, m in (("low", sbv <= qs[0]), ("mid", (sbv > qs[0]) & (sbv <= qs[1])), ("high", sbv > qs[1])):
        gain = np.array([r[f"k8.morph_corr"] - r["GLOBAL.morph_corr"] for r in rows])[m]
        g8 = ci(np.array([r["k8.morph_corr"] for r in rows])[m], pid[m])
        gl = ci(np.array([r["GLOBAL.morph_corr"] for r in rows])[m], pid[m])
        orc = ci(np.array([r["ORACLE.morph_corr"] for r in rows])[m], pid[m])
        tert[name] = {"n_patients": int(len(np.unique(pid[m]))), "sbv_range": [float(np.nanmin(sbv[m])), float(np.nanmax(sbv[m]))],
                      "GLOBAL": gl[0], "k8": g8[0], "ORACLE": orc[0], "gain": ci(gain, pid[m]),
                      "frac_oracle_gap_closed": (g8[0] - gl[0]) / (orc[0] - gl[0]) if orc[0] > gl[0] else None}
    out = {"n_windows": len(rows), "n_patients": int(len(np.unique(pid))), "curve": curve, "gain_vs_global": gains,
           "oracle_gap": gap, "frac_of_oracle_gap_closed": {f"k{k}": gains[f"k{k}"][0] / gap for k in KS},
           "saturation_deltas": sat, "saturated_at_windows": saturated_at,
           "seconds_per_window": 4, "sbv_tertiles": tert,
           "sbv_summary": {"median": float(np.nanmedian(sbv)), "p10": float(np.nanpercentile(sbv, 10)), "p90": float(np.nanpercentile(sbv, 90))}}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("curve", "sbv_tertiles")}, indent=1))
    for a in arms:
        print(a, {m: round(v[0], 4) for m, v in curve[a].items()})
    for t, v in tert.items():
        print("SBV", t, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "gain"}, "gain", [round(x, 4) for x in v["gain"]])


if __name__ == "__main__":
    main()
