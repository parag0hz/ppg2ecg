"""R1 visual atlas (prereg §4 / §21): 8 windows per validation subject x site, salt r1-visual-v1.

Visual only. Runs the frozen probes on exactly those windows and draws them. Produces NO metric; the frozen
thresholds are read from threshold_selection.json. Windows are drawn whether or not they fall inside the
1,024-per-stratum metric cohort (that membership is annotated).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events, soft_event_field

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r1_global_rhythm"
ATLAS = ART / "visual_atlas"
FS, T_LEN = 128, 1024
CK = {"global": "outputs/r1_global_tcn_seed42/checkpoint_best.pt", "local": "outputs/r1_local_tcn_seed42/checkpoint_best.pt"}


def load(v, dev):
    ck = torch.load(ROOT / CK[v], map_location="cpu", weights_only=False)
    net = RhythmTCN(ck["dilations"], n_sites=ck["n_sites"]).to(dev).eval()
    net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
    return net


@torch.no_grad()
def main() -> int:
    assert_no_test_subjects(C.VAL)
    ATLAS.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    thr = json.loads((ART / "threshold_selection.json").read_text())
    nets = {v: load(v, dev) for v in CK}
    t = np.arange(T_LEN) / FS
    for sub in C.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]
        vis = C.cohort_positions(sub, d["site"], WIs, C.N_VISUAL_PER, salt=C.VISUAL_SALT)
        met = C.cohort_positions(sub, d["site"], WIs, C.n_per_for(sub))
        for site in C.SITES:
            pos = vis[site]; in_metric = set(met[site].tolist())
            x = torch.from_numpy(Xs[pos].astype(np.float32)).to(dev).unsqueeze(1)
            pr = {v: torch.sigmoid(n(x)).squeeze(1).cpu().numpy() for v, n in nets.items()}
            fig, axes = plt.subplots(len(pos), 1, figsize=(13, 1.9 * len(pos)), sharex=True)
            for row, p in enumerate(pos):
                a = axes[row]; ppg = Xs[p].astype(np.float64); gt = R.detect_rpeaks(Ys[p].astype(np.float64), FS)
                ev_g = extract_events(pr["global"][row], thr["global"]["threshold"])
                ev_l = extract_events(pr["local"][row], thr["local"]["threshold"])
                a.fill_between(t, 0, soft_event_field(gt, T_LEN), color="0.75", alpha=0.5, lw=0, label="GT soft field (label)")
                a.plot(t, ppg / (np.abs(ppg).max() + 1e-9) * 0.9, color="k", lw=0.6, label="PPG (scaled)")
                a.plot(t, pr["global"][row], color="tab:green", lw=1.2, label="Global-TCN field")
                a.plot(t, pr["local"][row], color="tab:orange", lw=1.0, alpha=0.8, label="Local-TCN field")
                for r_ in gt:
                    a.axvline(r_ / FS, color="tab:red", ls="--", lw=0.8, alpha=0.6)
                a.plot(ev_g / FS, np.full(len(ev_g), 1.06), "v", color="tab:green", ms=6)
                a.plot(ev_l / FS, np.full(len(ev_l), -0.93), "^", color="tab:orange", ms=5, alpha=0.8)
                a.text(0.005, 0.82, f"w{int(WIs[p])}  GT {len(gt)} | Global {len(ev_g)} | Local {len(ev_l)}"
                       f"{'' if int(p) in in_metric else '   (outside metric cohort)'}", transform=a.transAxes, fontsize=7.5)
                a.set_ylim(-1.0, 1.15); a.grid(alpha=0.2); a.set_yticks([])
                if row == 0:
                    a.legend(fontsize=7, loc="upper right", ncol=4)
            axes[-1].set_xlabel("time (s)")
            fig.suptitle(f"R1 visual atlas — {sub} [validation] / {site} — PPG in, GT R (red dashed) never seen by the probe; "
                         f"Global events ▼ (thr {thr['global']['threshold']:.2f}), Local events ▲ (thr {thr['local']['threshold']:.2f})", fontsize=9.5)
            fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}.png", dpi=100); plt.close(fig)
            print(f"[atlas] {sub} {site}: {len(pos)} windows, {sum(int(p) in in_metric for p in pos)} inside metric cohort", flush=True)
            # fig D: the same 8 windows under TRUE / WINDOW-SHUFFLE (derangement within these 8) / CIRCULAR-SHIFT
            rng = np.random.default_rng(20260902)
            Xt = Xs[pos].astype(np.float32); Xsh = Xt[C.derangement(len(pos), rng)]
            off = C.circular_offsets(len(pos), rng); Xci = np.stack([np.roll(Xt[k], int(off[k])) for k in range(len(pos))])
            fields = {}
            for name, Xi in (("TRUE", Xt), ("WINDOW-SHUFFLE", Xsh), ("CIRCULAR-SHIFT", Xci)):
                fields[name] = torch.sigmoid(nets["global"](torch.from_numpy(Xi).to(dev).unsqueeze(1))).squeeze(1).cpu().numpy()
            fig, axes = plt.subplots(len(pos), 3, figsize=(19, 1.7 * len(pos)), sharex=True, sharey=True)
            for row, p in enumerate(pos):
                gt = R.detect_rpeaks(Ys[p].astype(np.float64), FS)
                for col, (name, Xi) in enumerate((("TRUE", Xt), ("WINDOW-SHUFFLE", Xsh), ("CIRCULAR-SHIFT", Xci))):
                    a = axes[row, col]; ev = extract_events(fields[name][row], thr["global"]["threshold"])
                    a.plot(t, Xi[row] / (np.abs(Xi[row]).max() + 1e-9) * 0.9, color="k", lw=0.5)
                    a.plot(t, fields[name][row], color="tab:green", lw=1.1)
                    for r_ in gt:
                        a.axvline(r_ / FS, color="tab:red", ls="--", lw=0.7, alpha=0.6)
                    a.plot(ev / FS, np.full(len(ev), 1.06), "v", color="tab:green", ms=5)
                    a.text(0.005, 0.82, f"w{int(WIs[p])} GT {len(gt)} | events {len(ev)}" + (f" | shift {int(off[row])/FS:.2f}s" if col == 2 else ""),
                           transform=a.transAxes, fontsize=7)
                    a.set_ylim(-1.0, 1.15); a.set_yticks([]); a.grid(alpha=0.2)
                    if row == 0:
                        a.set_title(name, fontsize=9)
            for col in range(3):
                axes[-1, col].set_xlabel("time (s)")
            fig.suptitle(f"R1 input-dependence — {sub} [validation] / {site} — Global-TCN field on TRUE, WINDOW-SHUFFLE (another of these 8 windows' PPG) "
                         f"and CIRCULAR-SHIFT input; GT R of the ORIGINAL window (red dashed)", fontsize=9.5)
            fig.tight_layout(); fig.savefig(ATLAS / f"controls_{sub}_{site}.png", dpi=90); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
