"""X4-0 Fig. 5 - same-PPG x 32-source predicted R-peak raster (pre-registration sec. 38).

Frozen-inference only; reuses the X4-0 frozen subset, source seeds and detector. Windows are chosen by the SAME
deterministic hash rank as the source subset (smallest hashes first) - never by appearance.
"""
from __future__ import annotations

from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/x4_0_event_reliability/figures"
FS, T = 128, 1024
VAL = ("an0", "k2s")
SALT = "x4-event-source-v2"
N_WIN_PER_SUBJECT = 2          # frozen: the 2 smallest-hash windows of each subject's source subset
SEEDS = tuple(range(32))
NFES = (1, 8)

ER.assert_no_test_subjects(VAL)
dev = torch.device("cuda")
ck = torch.load(ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt", map_location="cpu", weights_only=False)
cfg = ck.get("imf_cfg", {})
net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                 h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
net.load_state_dict(ck["state_dict"])

sel = []
for s in VAL:
    d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
    idx = ER.select_subset(SALT, s, len(d["x"]), 256)
    order = np.argsort([ER.window_hash(SALT, s, int(i)) for i in idx], kind="stable")   # hash rank, not index order
    for w in idx[order[:N_WIN_PER_SUBJECT]]:
        sel.append((s, int(w), d["x"][w].astype(np.float32), d["y"][w].astype(np.float32)))
print("frozen raster windows (smallest hashes):", [(s, w) for s, w, _, _ in sel])

ppg = torch.from_numpy(np.stack([x for _, _, x, _ in sel])).to(dev).unsqueeze(1)
gt = np.stack([y for _, _, _, y in sel]).astype(np.float64)
gt_pk = [R.detect_rpeaks(g, FS) for g in gt]

pred_pk = {}
with torch.no_grad():
    for n in NFES:
        per_seed = []
        for sd in SEEDS:
            e = ER.source_bank(sd, len(sel)).to(dev) if hasattr(ER, "source_bank") else torch.randn(len(sel), 1, T, generator=torch.Generator().manual_seed(sd)).to(dev)
            z, nfe = ER.sample_meanflow_schedule(net, ppg, e, ER.UNIFORM[n])
            assert nfe == n
            per_seed.append([R.detect_rpeaks(w.astype(np.float64), FS) for w in z.squeeze(1).float().cpu().numpy()])
        pred_pk[n] = per_seed

t = np.arange(T) / FS
fig, axes = plt.subplots(len(sel), len(NFES) + 1, figsize=(6.0 * (len(NFES) + 1), 3.1 * len(sel)),
                         gridspec_kw={"width_ratios": [1] + [1] * len(NFES)}, sharex=True)
for r, (s, w, x, y) in enumerate(sel):
    axes[r, 0].plot(t, y, "k", lw=0.8)
    axes[r, 0].plot(gt_pk[r] / FS, y[gt_pk[r]], "r.", ms=7)
    axes[r, 0].set_ylabel(f"{s}  win {w}\nGT ECG", fontsize=9)
    axes[r, 0].grid(alpha=0.25)
    axes[r, 0].set_ylim(-1.35, 1.35)
    for c, n in enumerate(NFES, start=1):
        a = axes[r, c]
        for g in gt_pk[r]:
            a.axvline(g / FS, color="tab:red", lw=1.1, alpha=0.55)
        for k in range(len(SEEDS)):
            pk = pred_pk[n][k][r]
            if len(pk):
                a.plot(pk / FS, np.full(len(pk), k), "|", color="tab:blue", ms=7, mew=1.3)
        cnt = [len(pred_pk[n][k][r]) for k in range(len(SEEDS))]
        f1s = [ER.peak_train_agreement(pred_pk[n][i][r], pred_pk[n][j][r])["f1"]
               for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        a.set_ylim(-1, len(SEEDS))
        a.set_ylabel("source seed 0-31", fontsize=8)
        a.grid(alpha=0.2, axis="x")
        a.set_title(f"NFE {n}  |  GT {len(gt_pk[r])} beats, predicted {np.mean(cnt):.1f} ± {np.std(cnt, ddof=1):.2f}"
                    f"  |  median seed-pair F1 {np.median(f1s):.2f}", fontsize=9)
for c in range(len(NFES) + 1):
    axes[-1, c].set_xlabel("time (s)")
fig.suptitle("X4-0 Fig. 5 — same PPG, 32 Gaussian sources: each blue tick is a predicted R-peak, red lines are ground-truth beats\n"
             "deterministic hash-selected windows (no cherry-picking) — vertical scatter of ticks IS the source-sensitive event organization",
             fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig5_source_peak_raster.png", dpi=115)
plt.close(fig)
print("wrote", OUT / "fig5_source_peak_raster.png")
