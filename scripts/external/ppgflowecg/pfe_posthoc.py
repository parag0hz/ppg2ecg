"""PPGFlowECG external validation — POST-HOC diagnostics (not preregistered; written after the analysis stage).

1. Beat timing: cross-correlation lag of the generated ECG (S = 10, draws 0 and 1) against the processed reference within
   ±600 ms — to read the near-zero per-sample waveform correlation (as in RDDM-EXT's post-hoc lag check).
2. Patient-level association of the consensus gain G_p with waveform pairwise RMS and functional SD (the preregistered
   patient-level item covered rho_bar_p only), at every S; patients with >= 8 mechanism windows, 5,000 bootstrap.
Writes artifacts/ppgflowecg_external/posthoc.json.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/external/ppgflowecg/pfe_posthoc.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "outputs/ppgflowecg_external"
ART = ROOT / "artifacts/ppgflowecg_external"
FS, MAXLAG, SS = 128, 77, (5, 10, 15, 20, 25)          # 77 samples = 602 ms


def lagcorr(g, r):
    """normalised cross-correlation for lags -MAXLAG..MAXLAG (positive = generated later than reference)."""
    zs = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)  # noqa: E731
    g, r = zs(g.astype(np.float64)), zs(r.astype(np.float64))
    L = g.shape[1]
    out = np.zeros((len(g), 2 * MAXLAG + 1))
    for i, k in enumerate(range(-MAXLAG, MAXLAG + 1)):
        out[:, i] = (g[:, k:] * r[:, :L - k]).mean(1) if k >= 0 else (g[:, :L + k] * r[:, -k:]).mean(1)
    return out


def main():
    z = np.load(RAW / "data/vitaldb10_test.npz")
    ref = z["ecg"]
    W = np.load(RAW / "gen/S10.npy", mmap_mode="r")
    lags = np.arange(-MAXLAG, MAXLAG + 1)
    res = {"note": "POST-HOC, not preregistered", "lag": {}}
    best = []
    for d in (0, 1):
        c = lagcorr(np.asarray(W[d]), ref)
        bi = c.argmax(1); lag_ms = lags[bi] / FS * 1000; best.append(lag_ms)
        res["lag"][f"draw{d}_S10"] = {"median_ms": float(np.median(lag_ms)), "iqr_ms": [float(np.percentile(lag_ms, 25)), float(np.percentile(lag_ms, 75))],
                                      "share_within_50ms": float((np.abs(lag_ms) <= 50).mean()),
                                      "corr_at_zero_lag_median": float(np.median(c[:, MAXLAG])),
                                      "corr_at_best_lag_median": float(np.median(c.max(1)))}
    res["lag"]["draw0_vs_draw1_best_lag_abs_diff_median_ms"] = float(np.median(np.abs(best[0] - best[1])))
    res["lag"]["share_draw0_draw1_best_lag_within_50ms"] = float((np.abs(best[0] - best[1]) <= 50).mean())
    # patient-level associations with G_p
    rows = list(csv.DictReader(open(ART / "per_patient_metrics.csv")))
    fb = json.loads((ART / "fixed_k_bootstrap.json").read_text())
    res["patient_level"] = {}
    rng = np.random.default_rng(20260924)
    for S in SS:
        g = np.array([float(r[f"mech_G_S{S}"]) if r[f"mech_G_S{S}"] else np.nan for r in rows])
        out = {}
        for k in ("wRMS", "SD", "MAD"):
            v = np.array([float(r[f"mech_{k}_S{S}"]) if r[f"mech_{k}_S{S}"] else np.nan for r in rows])
            # same eligibility as the preregistered patient-level item: >= 8 mechanism windows (count re-derived below)
            out[k] = (v, g)
        res["patient_level"][S] = out
    # eligibility: patients with >= 8 Omega_mech windows (recomputed from the raw HR arrays exactly as in analyze)
    H = np.load(RAW / "hr_hamilton.npz")
    und = lambda a: np.where(np.isfinite(a) & (a != -1), a, np.nan)  # noqa: E731
    om = np.isfinite(und(H["ref"])) & np.isfinite(und(H["Y"])).all((0, 1))
    up, inv = np.unique(z["patient"], return_inverse=True)
    elig = np.bincount(inv[om], None, len(up)) >= 8
    pid = np.array([int(r["patient"]) for r in rows]); assert (pid == up).all()
    for S in SS:
        d = {}
        for k, (v, g) in res["patient_level"][S].items():
            m = elig & np.isfinite(v) & np.isfinite(g)
            vv, gg = v[m], g[m]
            bs = [spearmanr(vv[ix], gg[ix])[0] for ix in (rng.integers(0, m.sum(), m.sum()) for _ in range(5000))]
            d[f"{k}_vs_G"] = {"n_patients": int(m.sum()), "spearman": float(spearmanr(vv, gg)[0]),
                              "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        d["rho_bar_vs_G_prereg"] = fb["patient_level_spearman_rho_G"][str(S)]
        res["patient_level"][S] = d
    (ART / "posthoc.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
