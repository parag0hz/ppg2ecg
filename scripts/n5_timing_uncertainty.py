"""N5 — can per-beat timing uncertainty be predicted from PPG?
(docs/N5_PER_BEAT_TIMING_UNCERTAINTY_PREREGISTRATION.md)

The R1 Global-TCN stays FROZEN (eval, requires_grad=False, state sha asserted). The only trained
object is a small heteroscedastic head reading PPG-derived features. Ground truth forms the training
target and is never an inference-time input.

Run: .venv/bin/python scripts/n5_timing_uncertainty.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppg2ecg.evaluation import event_reliability as ER          # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP                     # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT                  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import extract_events            # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

import r2_evaluate as R2E                                       # noqa: E402  (frozen scaffold batching)

ART = ROOT / "artifacts/n5_timing_uncertainty"
PREREG = "caa3034"
FS, T_LEN = 128, 1024
FIELD_HALF, PPG_HALF = 32, 96                                   # prereg §3
R1_THRESHOLD, R1_REFRACTORY = 0.35, 32
MATCH_MS = ER.GT_ANCHOR_MS                                      # 150.0
SALT, TAKE = "n2-beat-v1", 1024
STEPS, BATCH, LR, WD, SEED = 6000, 256, 1e-3, 0.01, 42
BOOT_N, BOOT_SEED = 2000, 20260911
SHARPNESS_BAR = 0.20
MIN_SIGMA_MS = 1.0


class Head(nn.Module):
    """PPG-derived features -> (mu, log sigma) of the timing residual, in ms."""

    def __init__(self, n_field: int, n_ppg: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_field + n_ppg, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, 2),
        )

    def forward(self, x):
        o = self.net(x)
        return o[:, 0], o[:, 1]


def gaussian_nll(y, mu, log_sigma):
    s = torch.clamp(log_sigma, math.log(MIN_SIGMA_MS), math.log(500.0))
    return 0.5 * math.log(2 * math.pi) + s + 0.5 * ((y - mu) / torch.exp(s)) ** 2


def crps_gaussian(y, mu, sigma):
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


@torch.no_grad()
def build(subjects, tcn, dev, tag):
    """Features and residual targets at every R1 event that matches a GT beat within +-150 ms."""
    ER.assert_no_test_subjects(subjects)
    F, P, Yr, S, n_det, n_match = [], [], [], [], 0, 0
    half = MATCH_MS / 1000.0 * FS
    for s in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        X, Yw = d["x"], d["y"]
        idx = ER.select_subset(SALT, s, len(X), TAKE)
        Xs = X[idx].astype(np.float32)
        field = R2E.scaffolds(tcn, Xs, dev)                       # PPG only; no ECG argument exists
        for k in range(len(Xs)):
            ev = extract_events(field[k], R1_THRESHOLD, R1_REFRACTORY)
            gt = RP.detect_rpeaks(Yw[idx[k]].astype(np.float64), FS)
            if len(gt) == 0:
                n_det += len(ev); continue
            for e in ev:
                n_det += 1
                if e - FIELD_HALF < 0 or e + FIELD_HALF + 1 > T_LEN:
                    continue
                if e - PPG_HALF < 0 or e + PPG_HALF + 1 > T_LEN:
                    continue
                dd = np.asarray(gt, float) - float(e)
                j = int(np.argmin(np.abs(dd)))
                if abs(dd[j]) > half:
                    continue
                n_match += 1
                F.append(field[k][e - FIELD_HALF: e + FIELD_HALF + 1])
                P.append(Xs[k][e - PPG_HALF: e + PPG_HALF + 1])
                Yr.append(float(dd[j]) / FS * 1000.0)
                S.append(s)
    F = np.asarray(F, np.float32); P = np.asarray(P, np.float32)
    Yr = np.asarray(Yr, np.float32); S = np.asarray(S)
    print(f"[N5] {tag}: {n_match:,} matched of {n_det:,} R1 detections "
          f"(unmatched {1 - n_match / max(n_det,1):.3f}), residual sd {Yr.std():.1f} ms", flush=True)
    return np.concatenate([F, P], axis=1), Yr, S, {"n_detections": int(n_det), "n_matched": int(n_match)}


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    sp = json.loads((ROOT / "artifacts/r1_global_rhythm/subject_split.json").read_text())
    TRAIN, DEV, EVAL = tuple(sp["probe_train"]), tuple(sp["internal_dev"]), tuple(sp["validation"])
    assert not (set(TRAIN) & set(EVAL)) and not (set(DEV) & set(EVAL))
    ER.assert_no_test_subjects(TRAIN + DEV + EVAL)

    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)   # asserts the frozen state sha256
    assert not any(p.requires_grad for p in tcn.parameters()), "R1 TCN must be frozen"

    Xtr, Ytr, Str, ctr = build(TRAIN, tcn, dev, "train")
    Xdv, Ydv, Sdv, cdv = build(DEV, tcn, dev, "internal-dev")
    Xev, Yev, Sev, cev = build(EVAL, tcn, dev, "eval")

    # CONST: one global (mu, sigma) from TRAIN only
    mu_c, sg_c = float(Ytr.mean()), float(Ytr.std(ddof=1))
    print(f"[N5] CONST from train: mu {mu_c:+.2f} ms, sigma {sg_c:.2f} ms", flush=True)

    seed_everything(SEED)
    head = Head(2 * FIELD_HALF + 1, 2 * PPG_HALF + 1).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xd, Yd_ = torch.from_numpy(Xdv).to(dev), torch.from_numpy(Ydv).to(dev)
    g = torch.Generator().manual_seed(SEED)
    best, best_sd = float("inf"), None
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(Xt), (BATCH,), generator=g)
        mu, ls = head(Xt[i].to(dev))
        loss = gaussian_nll(Yt[i].to(dev), mu, ls).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == STEPS:
            head.eval()
            with torch.no_grad():
                m = float(gaussian_nll(Yd_, *head(Xd)).mean())
            head.train()
            if m < best:
                best, best_sd = m, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"[N5] step {step:5d} train NLL {float(loss):.4f}  dev NLL {m:.4f}{'  *' if m == best else ''}", flush=True)
    head.load_state_dict({k: v.to(dev) for k, v in best_sd.items()}); head.eval()

    rng = np.random.default_rng(20260912)
    order = rng.permutation(len(Xev))
    bad = order == np.arange(len(Xev))
    while bad.any():
        order[bad] = rng.permutation(len(Xev))[bad]; bad = order == np.arange(len(Xev))

    with torch.no_grad():
        mu_h, ls_h = head(torch.from_numpy(Xev).to(dev))
        mu_s, ls_s = head(torch.from_numpy(Xev[order]).to(dev))
    arms = {
        "CONST": (np.full(len(Yev), mu_c), np.full(len(Yev), sg_c)),
        "HEAD": (mu_h.cpu().numpy(), np.exp(np.clip(ls_h.cpu().numpy(), math.log(MIN_SIGMA_MS), math.log(500.0)))),
        "HEAD-SHUFFLE": (mu_s.cpu().numpy(), np.exp(np.clip(ls_s.cpu().numpy(), math.log(MIN_SIGMA_MS), math.log(500.0)))),
    }
    macro = lambda v: float(np.mean([np.nanmean(np.asarray(v, float)[Sev == s]) for s in np.unique(Sev)]))  # noqa: E731

    per, table = {}, {}
    for name, (mu, sg) in arms.items():
        z = (Yev - mu) / sg
        nll = 0.5 * math.log(2 * math.pi) + np.log(sg) + 0.5 * z ** 2
        crps = crps_gaussian(Yev, mu, sg)
        absres = np.abs(Yev - mu)
        per[name] = {"nll": nll, "crps": crps, "absres": absres, "sigma": sg}
        cov = {a: np.abs(z) <= sps.norm.ppf(1 - a / 2) for a in (0.50, 0.20, 0.10)}
        table[name] = {"nll": macro(nll), "crps": macro(crps), "median_abs_res_ms": float(np.median(absres)),
                       "median_sigma_ms": float(np.median(sg)),
                       "sharpness_spearman": float(sps.spearmanr(sg, absres).statistic),
                       **{f"coverage_{int((1-a)*100)}": macro(cov[a]) for a in (0.50, 0.20, 0.10)}}
        print(f"[N5] {name:13s} NLL {table[name]['nll']:.4f}  CRPS {table[name]['crps']:.3f}  "
              f"|res| {table[name]['median_abs_res_ms']:.1f} ms  sigma {table[name]['median_sigma_ms']:.1f} ms  "
              f"sharpSpearman {table[name]['sharpness_spearman']:+.3f}", flush=True)

    pairs = {}
    for a, b, m in (("CONST", "HEAD", "nll"), ("CONST", "HEAD", "crps"),
                    ("HEAD-SHUFFLE", "HEAD", "nll"), ("HEAD-SHUFFLE", "HEAD", "crps")):
        pairs[f"{b}_vs_{a}:{m}"] = paired_subject_bootstrap(per[a][m], per[b][m], Sev, "lower_better", BOOT_N, BOOT_SEED)
    # sharpness CI by subject-clustered bootstrap of the Spearman itself
    subs = np.unique(Sev)
    rs = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(BOOT_N):
        pick = np.concatenate([np.flatnonzero(Sev == subs[j]) for j in rs.integers(0, len(subs), len(subs))])
        draws.append(sps.spearmanr(per["HEAD"]["sigma"][pick], per["HEAD"]["absres"][pick]).statistic)
    sharp_ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    vN, vS = pairs["HEAD_vs_CONST:nll"], pairs["HEAD_vs_HEAD-SHUFFLE:nll"]
    sharp = table["HEAD"]["sharpness_spearman"]
    verdict = ("PER-BEAT TIMING UNCERTAINTY IS PREDICTABLE"
               if vN["lo"] > 0 and sharp >= SHARPNESS_BAR and sharp_ci[0] > 0 and vS["lo"] > 0
               else "MARGINAL" if vN["lo"] > 0 else "NOT PREDICTABLE")

    out = {"prereg": PREREG, "git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "r1_tcn": tmeta,
           "subjects": {"train": list(TRAIN), "internal_dev": list(DEV), "eval": list(EVAL)},
           "counts": {"train": ctr, "internal_dev": cdv, "eval": cev},
           "const": {"mu_ms": mu_c, "sigma_ms": sg_c},
           "config": {"steps": STEPS, "batch": BATCH, "lr": LR, "wd": WD, "seed": SEED,
                      "field_half": FIELD_HALF, "ppg_half": PPG_HALF, "match_ms": MATCH_MS,
                      "sharpness_bar": SHARPNESS_BAR},
           "table": table, "paired": pairs, "sharpness_ci": sharp_ci, "verdict": verdict,
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n5_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N5] HEAD vs CONST  NLL {vN['point']:+.4f} [{vN['lo']:+.4f},{vN['hi']:+.4f}]")
    print(f"[N5] HEAD vs SHUFFLE NLL {vS['point']:+.4f} [{vS['lo']:+.4f},{vS['hi']:+.4f}]")
    print(f"[N5] sharpness Spearman {sharp:+.3f} [{sharp_ci[0]:+.3f},{sharp_ci[1]:+.3f}] (bar {SHARPNESS_BAR})")
    print(f"[N5] VERDICT: {verdict}   ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
