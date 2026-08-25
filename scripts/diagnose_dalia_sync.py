"""Data-level diagnostic: is the wrist-PPG <-> chest-ECG timing in PPG-DaLiA stable enough to learn beat-level alignment?
For each dataset reference R-peak (700 Hz), measure the delay to the next PPG pulse peak (neurokit on the raw 64 Hz BVP).
A synchronised recording gives a physiologically plausible, slowly varying pulse-arrival delay (~0.2-0.5 s);
clock drift between the two devices shows up as a delay that wanders/wraps across the RR interval over time.
Writes <out>/dalia_sync_diagnostic.json and figures/dalia_sync.png"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import neurokit2 as nk
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.dalia import BVP_FS, ECG_FS, load_subject_raw  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def analyse(subject: str):
    raw = load_subject_raw(ROOT / "data/raw", subject)
    t_r = raw.rpeaks / ECG_FS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean = nk.ppg_clean(raw.bvp.astype(float), sampling_rate=BVP_FS)
        info = nk.ppg_findpeaks(clean, sampling_rate=BVP_FS)
    t_p = np.asarray(info["PPG_Peaks"]) / BVP_FS
    # delay from each R-peak to the next PPG peak, normalised by the local RR interval (phase in [0,1))
    idx = np.searchsorted(t_p, t_r, side="right")
    ok = idx < len(t_p)
    delay = t_p[idx[ok]] - t_r[ok]
    rr = np.diff(t_r, append=np.nan)[ok]
    phase = delay / rr
    tt = t_r[ok]
    minutes = (tt // 60).astype(int)
    per_min = []
    for m in np.unique(minutes):
        sel = (minutes == m) & np.isfinite(phase) & (rr > 0.3) & (rr < 2.0)
        if sel.sum() >= 10:
            per_min.append((int(m), float(np.median(delay[sel])), float(np.median(phase[sel])), float(np.percentile(delay[sel], 75) - np.percentile(delay[sel], 25))))
    per_min = np.array(per_min)
    return {"subject": subject, "n_rpeaks": int(len(t_r)), "n_ppg_peaks": int(len(t_p)), "duration_min": float(raw.ecg_seconds / 60),
            "delay_median_s": float(np.nanmedian(delay)), "delay_iqr_s": [float(np.nanpercentile(delay, 25)), float(np.nanpercentile(delay, 75))],
            "phase_median": float(np.nanmedian(phase)), "phase_iqr": [float(np.nanpercentile(phase, 25)), float(np.nanpercentile(phase, 75))],
            "per_minute_delay_median_s": per_min[:, 1].tolist(), "per_minute_phase_median": per_min[:, 2].tolist(), "per_minute_within_iqr_s": per_min[:, 3].tolist(), "minutes": per_min[:, 0].tolist(),
            "per_minute_delay_range_s": [float(per_min[:, 1].min()), float(per_min[:, 1].max())] if len(per_min) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="S2,S11,S1,S5")
    ap.add_argument("--out-dir", default="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    res = {}
    fig, axes = plt.subplots(len(args.subjects.split(",")), 1, figsize=(14, 2.6 * len(args.subjects.split(","))), sharex=False)
    for ax, s in zip(np.atleast_1d(axes), args.subjects.split(",")):
        r = analyse(s)
        res[s] = r
        print(f"{s}: R-peaks {r['n_rpeaks']} PPG peaks {r['n_ppg_peaks']} | R->next PPG peak delay median {r['delay_median_s']*1000:.0f} ms IQR {[round(v*1000) for v in r['delay_iqr_s']]} ms | phase median {r['phase_median']:.2f} IQR {[round(v,2) for v in r['phase_iqr']]} | per-minute median delay range {[round(v*1000) for v in r['per_minute_delay_range_s']]} ms | median within-minute IQR {np.median(r['per_minute_within_iqr_s'])*1000:.0f} ms")
        ax.plot(r["minutes"], np.array(r["per_minute_delay_median_s"]) * 1000, ".-", label="median delay R -> next PPG peak")
        ax.fill_between(r["minutes"], (np.array(r["per_minute_delay_median_s"]) - np.array(r["per_minute_within_iqr_s"]) / 2) * 1000, (np.array(r["per_minute_delay_median_s"]) + np.array(r["per_minute_within_iqr_s"]) / 2) * 1000, alpha=0.2, label="within-minute IQR")
        ax.set_ylabel(f"{s}\ndelay (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("recording time (min)")
    fig.suptitle("PPG-DaLiA: R-peak -> next wrist-PPG pulse delay over time (device synchronisation / pulse-arrival diagnostic)")
    fig.tight_layout()
    fig.savefig(out / "figures" / "dalia_sync.png", dpi=110)
    (out / "dalia_sync_diagnostic.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "dalia_sync_diagnostic.json")


if __name__ == "__main__":
    main()
