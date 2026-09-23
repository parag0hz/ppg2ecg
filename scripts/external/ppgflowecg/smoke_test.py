"""PPGFlowECG external validation — step 3: smoke test on 16 VitalDB 10-s windows (np.linspace over the built set).
Technical validity only — no HR, no error against the reference is computed here.

Checks: NaN/Inf, output shape, numerical range, decoder output, seed-to-seed stochasticity at every S, same-seed
determinism, reference-ECG independence (the official path passes the reference ECG through vae_encoder_ecg only for its
latent shape), PPG-posterior noise scale.
Writes artifacts/ppgflowecg_external/smoke_test.json (+ outputs/ppgflowecg_external/smoke/ waveforms and a PNG).
Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=outputs/pfe_env/core:scripts/external/ppgflowecg .venv/bin/python \
     scripts/external/ppgflowecg/smoke_test.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pfe_model import PFE, ROOT  # noqa: E402

DATA = ROOT / "outputs/ppgflowecg_external/data/vitaldb10_test.npz"
OUT = ROOT / "artifacts/ppgflowecg_external"
RAWO = ROOT / "outputs/ppgflowecg_external/smoke"
SS = (5, 10, 15, 20, 25)


def st(x):
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "std_per_window_mean": float(x.std(1).mean())}


def main():
    RAWO.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA)
    idx = np.linspace(0, len(z["ppg"]) - 1, 16).astype(int)
    ppg, ecg = torch.from_numpy(z["ppg"][idx]), torch.from_numpy(z["ecg"][idx])
    pfe = PFE()
    res = {"windows": idx.tolist(), "patients": z["patient"][idx].tolist(), "input_shape": list(ppg.shape),
           "reference_ecg_range": st(ecg.numpy()), "input_ppg_range": st(ppg.numpy()), "per_S": {}}
    W = {}
    for S in SS:
        a = pfe.sample(ppg, ecg, S, seed=11)
        a2 = pfe.sample(ppg, ecg, S, seed=11)
        b = pfe.sample(ppg, ecg, S, seed=12)
        c = pfe.sample(ppg, torch.zeros_like(ecg), S, seed=11)
        W[S] = np.stack([a, b])
        rms_ab = np.sqrt(((a - b) ** 2).mean(1))
        res["per_S"][str(S)] = {
            "output_shape": list(a.shape), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "range": st(a),
            "same_seed_max_abs_diff": float(np.abs(a - a2).max()),
            "different_seed_waveform_rms": {"median": float(np.median(rms_ab)), "min": float(rms_ab.min()), "max": float(rms_ab.max())},
            "different_seed_waveform_corr_median": float(np.median([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)])),
            "reference_ecg_replaced_by_zeros_max_abs_diff": float(np.abs(a - c).max()),
        }
    # PPG posterior noise scale (official encoder returns (sample, mean, log_variance))
    with torch.no_grad():
        torch.manual_seed(0)
        _, mu, lv = pfe.tr.vae_encoder_ppg(ppg.unsqueeze(-1).cuda().float())
        sd = (0.5 * lv).exp()
    res["ppg_posterior"] = {"mean_abs_mu_scaled": float((mu.abs() * 0.18215).mean()), "mean_sd_scaled": float((sd * 0.18215).mean()),
                            "sd_over_abs_mu": float((sd.mean() / mu.abs().mean()))}
    ok = all(v["finite"] and v["same_seed_max_abs_diff"] == 0.0 and v["different_seed_waveform_rms"]["min"] > 0
             and v["reference_ecg_replaced_by_zeros_max_abs_diff"] == 0.0 and v["output_shape"] == [16, 1280]
             for v in res["per_S"].values())
    res["all_checks_pass"] = bool(ok)
    (OUT / "smoke_test.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(RAWO / "smoke_waves.npz", idx=idx, **{f"S{S}": W[S] for S in SS})
    fig, ax = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    t = np.arange(1280) / 128
    for r, i in enumerate((0, 5, 10, 15)):
        ax[r].plot(t, ecg[i].numpy(), color="#999", lw=0.8, label="reference (processed)")
        for S, col in ((5, "#1f4e79"), (25, "#a33")):
            ax[r].plot(t, W[S][0, i], color=col, lw=0.7, label=f"generated S={S} seed 11")
        ax[r].plot(t, W[5][1, i], color="#1f4e79", lw=0.7, ls=":", label="generated S=5 seed 12")
        ax[r].set_ylabel(f"win {idx[i]}")
    ax[0].legend(fontsize=7, ncol=4, frameon=False); ax[-1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(RAWO / "smoke.png", dpi=120)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
