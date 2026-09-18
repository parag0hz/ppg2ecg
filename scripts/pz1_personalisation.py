"""PZ1 — personalisation ceiling on VitalDB (docs/PZ1_PERSONALISATION_CEILING_PREREGISTRATION.md). No training."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.evaluation.stamping import build_template_a, stamp, template_geometry  # noqa: E402

FS, N_CAL, N_TRAIN_CASES, N_TRAIN_WIN, MIN_BEATS = 128, 4, 200, 2, 5
G = template_geometry(FS)
LEN, RI = G["full_len"], G["r_index_full"]
QRS = slice(RI - 10, RI + 16)
OUT = ROOT / "artifacts/pz1_personalisation"


def beats_of(sig, peaks):
    out = []
    for r in np.asarray(peaks, int):
        a, b = r - RI, r - RI + LEN
        if a >= 0 and b <= sig.size:
            out.append(sig[a:b])
    return out


def case_path(c):
    return ROOT / f"data/processed/v1_vitaldb/{c}.npz"


def train_beats(cases):
    beats = []
    for c in cases:
        y = np.load(case_path(c))["y"].astype(np.float64)
        for w in y[:N_TRAIN_WIN]:
            beats += beats_of(w, detect_rpeaks(w, FS))
    return np.stack(beats) if beats else None


def stretch(tmpl, ratio):
    """Resample the beat so its duration scales by `ratio`, keeping the R sample at index RI."""
    n = max(16, int(round(LEN * float(ratio))))
    ri = int(round(RI * float(ratio)))
    src = np.interp(np.linspace(0, LEN - 1, n), np.arange(LEN), tmpl)
    out = np.zeros(LEN)
    a, b = RI - ri, RI - ri + n                       # align the R index, clip the rest
    ta, tb = max(0, -a), n - max(0, b - LEN)
    a, b = max(0, a), min(LEN, b)
    out[a:b] = src[ta:tb]
    return out


def one_case(job):
    case, g_tmpl = job
    d = np.load(case_path(case))
    y, wi, pid = d["y"].astype(np.float64), d["window_index"], int(d["subjectid"])
    order = np.argsort(wi)
    cal, ev = order[:N_CAL], order[N_CAL:]
    cal_beats, cal_rr = [], []
    for i in cal:
        pk = detect_rpeaks(y[i], FS)
        cal_beats += beats_of(y[i], pk)
        if len(pk) > 1:
            cal_rr += list(np.diff(np.asarray(pk, float)))
    subj = build_template_a(np.stack(cal_beats))[0] if len(cal_beats) >= MIN_BEATS else g_tmpl
    fell_back = len(cal_beats) < MIN_BEATS
    rr_cal = float(np.median(cal_rr)) if cal_rr else np.nan
    rows = []
    for i in ev:
        sig = y[i]
        pk = detect_rpeaks(sig, FS)
        bts = beats_of(sig, pk)
        if len(bts) < 2:
            continue
        rr_ev = float(np.median(np.diff(np.asarray(pk, float)))) if len(pk) > 1 else np.nan
        ratio = rr_ev / rr_cal if np.isfinite(rr_cal) and np.isfinite(rr_ev) and rr_cal > 0 else 1.0
        arms = {"GLOBAL": g_tmpl, "SUBJ": subj, "SUBJ-RR": stretch(subj, ratio),
                "ORACLE": build_template_a(np.stack(bts))[0]}
        row = {"case": case, "pid": pid, "window": int(wi[i]), "fell_back": fell_back, "rr_ratio": ratio}
        tgt_beats = np.stack(bts)
        for name, tmpl in arms.items():
            pred = stamp(tmpl, pk, sig.size, RI)
            pb = np.stack([b for b in beats_of(pred, pk)][: len(tgt_beats)])
            tb = tgt_beats[: len(pb)]
            cc = [np.corrcoef(p, t)[0, 1] if p.std() > 0 and t.std() > 0 else np.nan for p, t in zip(pb, tb)]
            row[f"{name}.morph_corr"] = float(np.nanmean(cc))
            row[f"{name}.qrs_core_rmse"] = float(np.sqrt(np.mean((pb[:, QRS] - tb[:, QRS]) ** 2)))
            row[f"{name}.pcc"] = float(np.corrcoef(pred, sig)[0, 1]) if pred.std() > 0 else np.nan
            row[f"{name}.mae"] = float(np.mean(np.abs(pred - sig)))
        rows.append(row)
    return rows


def ci(v, pid, n=2000, seed=20260911):
    subs = np.unique(pid)
    per = np.array([np.nanmean(v[pid == s]) for s in subs])
    rng = np.random.default_rng(seed)
    d = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(n)])
    return float(np.nanmean(per)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]
    g = build_template_a(train_beats(sorted(split["train"])[:N_TRAIN_CASES]))[0]
    print(f"[pz1] global template from {N_TRAIN_CASES} train cases, ptp {np.ptp(g):.3f}", flush=True)
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one_case, [(c, g) for c in split["test"]], chunksize=8), 1):
            rows += r
            if i % 200 == 0:
                print(f"[pz1] {i}/{len(split['test'])} cases", flush=True)
    pid = np.array([r["pid"] for r in rows])
    res, arms = {}, ("GLOBAL", "SUBJ", "SUBJ-RR", "ORACLE")
    for m in ("morph_corr", "qrs_core_rmse", "pcc", "mae"):
        res[m] = {a: ci(np.array([r[f"{a}.{m}"] for r in rows]), pid) for a in arms}
    gains = {a: ci(np.array([r[f"{a}.morph_corr"] - r["GLOBAL.morph_corr"] for r in rows]), pid) for a in ("SUBJ", "SUBJ-RR", "ORACLE")}
    gap = res["morph_corr"]["ORACLE"][0] - res["morph_corr"]["GLOBAL"][0]
    verdict = ("PERSONALISATION WORTH IT" if gains["SUBJ"][0] >= 0.05 and gains["SUBJ"][1] > 0
               else "NOT WORTH IT AT THIS SCALE")
    out = {"verdict": verdict, "n_windows": len(rows), "n_patients": int(len(np.unique(pid))),
           "n_fallback_windows": int(sum(r["fell_back"] for r in rows)),
           "metrics": res, "gain_vs_global_morph_corr": gains,
           "oracle_gap": gap, "frac_of_oracle_gap_closed": (gains["SUBJ"][0] / gap) if gap else None,
           "frac_patients_positive_gain": float(np.mean([np.nanmean([r[f"SUBJ.morph_corr"] - r["GLOBAL.morph_corr"]
                                                                     for r in rows if r["pid"] == s]) > 0
                                                         for s in np.unique(pid)]))}
    (OUT / "result.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / "per_window.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=1))
    for m, d in out["metrics"].items():
        print(m, {a: [round(x, 4) for x in v] for a, v in d.items()})


if __name__ == "__main__":
    main()
