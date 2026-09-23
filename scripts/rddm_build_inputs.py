"""Reconstruct RDDM's evaluation inputs from the raw corpora, following RDDM's paper (AAAI 2024, §4.1 "Data
pre-processing") as closely as its text allows, and write them under the file names RDDM's official `data.get_datasets`
expects (`<DATASET>/{ecg,ppg}_{train,test}_{4,8}sec.npy`), so the official loader and evaluator run unmodified.

Paper, verbatim: "First, we resample both signals at 128 Hz. Subsequently, we apply a high-pass Butterworth filter with a
cut-off frequency of 0.5 Hz on ECG signals. Similarly, a band-pass Butterworth filter with a cut-off frequency between
0.5 to 8 Hz is applied on PPG signals. Additionally, subject-specific z-score normalization is applied … Further, min-max
scaling is applied to re-scale the ECG and PPG signals between [−1, 1]. Finally, we segment both signals into 4-second
windows". The paper cites neurokit (Makowski et al. 2021) for both filters, so the filters are neurokit's
`ecg_clean(method="neurokit")` (Butterworth high-pass 0.5 Hz, order 5, + powerline) and `ppg_clean(method="elgendi")`
(Butterworth band-pass 0.5–8 Hz, order 2). The per-window min-max and RDDM's own in-Dataset cleaning are applied by RDDM's
code, not here.

Choices the paper does not fix (recorded in the manifest): polyphase resampling of the continuous record; windows are
non-overlapping from t = 0; non-finite or constant windows are dropped. **RDDM's train/test subject split is unpublished,
so every subject is written to the *test* files** — these files therefore contain RDDM's training subjects (the paper
trains on 80 % of the subjects of these very corpora). The *train* files are one-window placeholders, present only
because `get_datasets` loads them.

Run: .venv/bin/python scripts/rddm_build_inputs.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
from datetime import datetime
from math import gcd
from pathlib import Path

import h5py
import neurokit2 as nk
import numpy as np
from scipy.signal import resample_poly

from ppg2ecg.data import bidmc as BIDMC
from ppg2ecg.data import capnobase as CAPNO
from ppg2ecg.data.dalia import load_subject_raw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rddm_run/datasets"
FS = 128
WINDOWS = (4, 8)


def to128(x, fs):
    g = gcd(int(fs), FS)
    return resample_poly(np.asarray(x, np.float64), FS // g, int(fs) // g)


def prep(ppg, fs_p, ecg, fs_e):
    p, e = to128(ppg, fs_p), to128(ecg, fs_e)
    n = min(len(p), len(e))
    p, e = p[:n], e[:n]
    e = nk.ecg_clean(e, sampling_rate=FS, method="neurokit")
    p = nk.ppg_clean(p, sampling_rate=FS, method="elgendi")
    z = lambda v: (v - v.mean()) / v.std() if v.std() > 0 else v * np.nan  # noqa: E731
    return z(p), z(e)


def windows(p, e, sec):
    L = FS * sec
    n = len(p) // L
    P, E = p[: n * L].reshape(n, L), e[: n * L].reshape(n, L)
    ok = np.isfinite(P).all(1) & np.isfinite(E).all(1) & (P.std(1) > 0) & (E.std(1) > 0)
    return P[ok], E[ok], (np.arange(n) * sec)[ok], int((~ok).sum())


def subjects(name):
    if name == "BIDMC":
        for rec in BIDMC.participant_files(ROOT / "data/raw/BIDMC"):
            hdr, sig = BIDMC.read_signals(rec)
            if BIDMC.ECG_SIGNAL not in hdr.names:
                yield hdr.record, None, "no lead II"
                continue
            yield hdr.record, (sig[hdr.names.index(BIDMC.PPG_SIGNAL)], hdr.fs, sig[hdr.names.index(BIDMC.ECG_SIGNAL)], hdr.fs), ""
    elif name == "CAPNO":
        for f in CAPNO.participant_files(ROOT / "data/raw/CapnoBase/files"):
            with h5py.File(f, "r") as h5:
                e, p = np.asarray(h5["signal/ecg/y"]).ravel(), np.asarray(h5["signal/pleth/y"]).ravel()
            yield Path(f).stem, (p, CAPNO.FS, e, CAPNO.FS), ""
    else:
        raw = ROOT / ("data/raw/PPG-DaLiA" if name == "DALIA" else "data/raw/WESAD")
        ids = [f"S{i}" for i in range(1, 16)] if name == "DALIA" else [d.name for d in sorted(raw.iterdir()) if d.is_dir() and d.name.startswith("S")]
        for s in sorted(ids, key=lambda v: int(v[1:])):
            r = load_subject_raw(raw, s)
            yield s, (r.bvp, 64, r.ecg, 700), ""


def main():
    manifest = {"built": datetime.now().isoformat(timespec="seconds"), "fs": FS, "script": "scripts/rddm_build_inputs.py",
                "filters": {"ecg": "neurokit ecg_clean(method='neurokit'): Butterworth HP 0.5 Hz order 5 + powerline",
                            "ppg": "neurokit ppg_clean(method='elgendi'): Butterworth BP 0.5-8 Hz order 2"},
                "resampling": "scipy.signal.resample_poly on the continuous record", "normalisation": "subject z-score (per-window min-max applied later by RDDM's get_datasets)",
                "split": "ALL subjects written to *_test_*; RDDM's split is unpublished, so these include RDDM training subjects",
                "train_files": "one-window placeholders (get_datasets loads them; unused by evaluation)", "corpora": {}}
    for name in ("BIDMC", "CAPNO", "DALIA", "WESAD"):
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        acc = {s: ([], [], [], []) for s in WINDOWS}
        info = {"subjects": [], "skipped": {}, "dropped_windows": {str(s): 0 for s in WINDOWS}}
        for sid, sig, why in subjects(name):
            if sig is None:
                info["skipped"][sid] = why; continue
            p, e = prep(*sig)
            if not np.isfinite(p).all() or not np.isfinite(e).all():
                info["skipped"][sid] = "non-finite after preprocessing"; continue
            info["subjects"].append(sid)
            for sec in WINDOWS:
                P, E, t0, nd = windows(p, e, sec)
                acc[sec][0].append(P.astype(np.float32)); acc[sec][1].append(E.astype(np.float32))
                acc[sec][2].append(np.array([sid] * len(P))); acc[sec][3].append(t0)
                info["dropped_windows"][str(sec)] += nd
            print(f"[rddm-build] {name} {sid}: {len(acc[4][0][-1])} x 4 s, {len(acc[8][0][-1])} x 8 s", flush=True)
        for sec in WINDOWS:
            P, E = np.concatenate(acc[sec][0]), np.concatenate(acc[sec][1])
            np.save(d / f"ppg_test_{sec}sec.npy", P); np.save(d / f"ecg_test_{sec}sec.npy", E)
            np.save(d / f"ppg_train_{sec}sec.npy", P[:1]); np.save(d / f"ecg_train_{sec}sec.npy", E[:1])
            np.savez(d / f"meta_{sec}sec.npz", subject=np.concatenate(acc[sec][2]), window_start_s=np.concatenate(acc[sec][3]))
            info[f"n_windows_{sec}s"] = int(len(P))
            info[f"sha256_ecg_test_{sec}s"] = hashlib.sha256((d / f"ecg_test_{sec}sec.npy").read_bytes()).hexdigest()
        info["n_subjects"] = len(info["subjects"])
        manifest["corpora"][name] = info
        print(f"[rddm-build] {name}: {info['n_subjects']} subjects, {info['n_windows_4s']} x 4 s, {info['n_windows_8s']} x 8 s, skipped {info['skipped']}", flush=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
