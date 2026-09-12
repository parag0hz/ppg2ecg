"""N6 — does per-beat timing uncertainty survive inside a generative sampler?
(docs/N6_GENERATIVE_TIMING_POSTERIOR_PREREGISTRATION.md)

The R1 Global-TCN stays FROZEN. Output object is the 1-D timing residual, not a waveform.
Every arm is scored from K=32 samples by ONE proper ensemble CRPS estimator.

Run: .venv/bin/python scripts/n6_generative_posterior.py
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

import diptest
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppg2ecg.evaluation import event_reliability as ER          # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT                  # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

import n5_timing_uncertainty as N5                              # noqa: E402  (population, features, Head)

ART = ROOT / "artifacts/n6_generative_posterior"
PREREG = "cd30878"
K_SAMPLES = 32
STEPS, BATCH, LR, WD, SEED = N5.STEPS, N5.BATCH, N5.LR, N5.WD, N5.SEED
BOOT_N, BOOT_SEED = N5.BOOT_N, N5.BOOT_SEED
SHARPNESS_BAR = N5.SHARPNESS_BAR
ALPHAS = (0.50, 0.20, 0.10)
DIP_ALPHA = 0.05


class FlowHead(nn.Module):
    """u(z, c, t, r) on the 1-D residual. Same conditioning features as N5's head."""

    def __init__(self, n_feat: int, h: int = 256, t_dim: int = 32):
        super().__init__()
        self.t_dim = t_dim
        self.enc = nn.Sequential(nn.Linear(n_feat, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.temb = nn.Sequential(nn.Linear(2 * t_dim, h), nn.GELU(), nn.Linear(h, h))
        self.out = nn.Sequential(nn.Linear(h + 1, h), nn.GELU(), nn.Linear(h, 1))

    @staticmethod
    def _sin(t, dim):
        half = dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.reshape(-1, 1).float() * f.reshape(1, -1)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=1)

    def u(self, z, c, t, r):
        h = self.enc(c) + self.temb(torch.cat([self._sin(t, self.t_dim), self._sin(r, self.t_dim)], dim=1))
        return self.out(torch.cat([h, z.reshape(-1, 1)], dim=1)).reshape(-1)


@torch.no_grad()
def fm_sample(net, c, k, scale, gen_seed):
    """NFE 1 endpoint form, K draws. z ~ N(0, scale) matches the training noise scale."""
    n = len(c)
    g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
    out = []
    for j in range(k):
        e = (torch.randn(n, generator=g) * scale).to(c.device)
        one = torch.ones(n, device=c.device)
        out.append((e - net.u(e, c, one, torch.zeros_like(one))).cpu().numpy())
    return np.stack(out, axis=1)                                   # [n, K]


def crps_ensemble(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(1/K) sum |x_k - y|  -  (1/2K^2) sum_j sum_k |x_j - x_k|."""
    k = samples.shape[1]
    t1 = np.abs(samples - y[:, None]).mean(1)
    s = np.sort(samples, axis=1)
    w = (2 * np.arange(1, k + 1) - k - 1).astype(np.float64)
    t2 = (s * w[None, :]).sum(1) / (k * k)                          # = (1/2K^2) sum sum |xj-xk|
    return t1 - t2


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    sp = json.loads((ROOT / "artifacts/r1_global_rhythm/subject_split.json").read_text())
    TRAIN, DEV, EVAL = tuple(sp["probe_train"]), tuple(sp["internal_dev"]), tuple(sp["validation"])
    ER.assert_no_test_subjects(TRAIN + DEV + EVAL)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    assert not any(p.requires_grad for p in tcn.parameters())

    Xtr, Ytr, Str, ctr = N5.build(TRAIN, tcn, dev, "train")
    Xdv, Ydv, Sdv, cdv = N5.build(DEV, tcn, dev, "internal-dev")
    Xev, Yev, Sev, cev = N5.build(EVAL, tcn, dev, "eval")
    mu_c, sg_c = float(Ytr.mean()), float(Ytr.std(ddof=1))

    # ---- HEAD: N5's arm, retrained here under the identical recipe so both arms are one process ----
    seed_everything(SEED)
    head = N5.Head(2 * N5.FIELD_HALF + 1, 2 * N5.PPG_HALF + 1).to(dev)
    oh = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xd, Yd_ = torch.from_numpy(Xdv).to(dev), torch.from_numpy(Ydv).to(dev)
    g = torch.Generator().manual_seed(SEED)
    bh, bh_sd = float("inf"), None
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(Xt), (BATCH,), generator=g)
        loss = N5.gaussian_nll(Yt[i].to(dev), *head(Xt[i].to(dev))).mean()
        oh.zero_grad(); loss.backward(); oh.step()
        if step % 2000 == 0 or step == STEPS:
            head.eval()
            with torch.no_grad():
                m = float(N5.gaussian_nll(Yd_, *head(Xd)).mean())
            head.train()
            if m < bh:
                bh, bh_sd = m, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            print(f"[N6] HEAD step {step:5d} dev NLL {m:.4f}{'  *' if m == bh else ''}", flush=True)
    head.load_state_dict({k: v.to(dev) for k, v in bh_sd.items()}); head.eval()

    # ---- FM: conditional MeanFlow on the 1-D residual ----
    seed_everything(SEED)
    fm = FlowHead(Xtr.shape[1]).to(dev)
    of = torch.optim.AdamW(fm.parameters(), lr=LR, weight_decay=WD)
    g2 = torch.Generator().manual_seed(SEED)
    Cd = torch.from_numpy(Xdv).to(dev)
    bf, bf_sd = float("inf"), None
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(Xt), (BATCH,), generator=g2)
        c, y = Xt[i].to(dev), Yt[i].to(dev)
        e = torch.randn(BATCH, device=dev) * sg_c
        t = torch.rand(BATCH, device=dev)
        r = torch.where(torch.rand(BATCH, device=dev) < 0.5, t, t * torch.rand(BATCH, device=dev))
        z = (1 - t) * y + t * e
        loss = torch.mean((fm.u(z, c, t, r) - (e - y)) ** 2)
        of.zero_grad(); loss.backward(); of.step()
        if step % 2000 == 0 or step == STEPS:
            fm.eval()
            s = fm_sample(fm, Cd, K_SAMPLES, sg_c, 1000)
            m = float(np.mean(crps_ensemble(s, Ydv.astype(np.float64))))
            fm.train()
            if m < bf:
                bf, bf_sd = m, {k: v.detach().cpu().clone() for k, v in fm.state_dict().items()}
            print(f"[N6] FM   step {step:5d} dev CRPS {m:.4f}{'  *' if m == bf else ''}", flush=True)
    fm.load_state_dict({k: v.to(dev) for k, v in bf_sd.items()}); fm.eval()

    # ---- samples for every arm, one estimator ----
    rng = np.random.default_rng(20260912)
    order = rng.permutation(len(Xev))
    bad = order == np.arange(len(Xev))
    while bad.any():
        order[bad] = rng.permutation(len(Xev))[bad]; bad = order == np.arange(len(Xev))
    Ce = torch.from_numpy(Xev).to(dev)
    Cs = torch.from_numpy(Xev[order]).to(dev)
    ys = Yev.astype(np.float64)
    rs = np.random.default_rng(BOOT_SEED)
    with torch.no_grad():
        mu_h, ls_h = head(Ce)
    sg_h = np.exp(np.clip(ls_h.cpu().numpy(), math.log(N5.MIN_SIGMA_MS), math.log(500.0)))
    mu_h = mu_h.cpu().numpy()

    samples = {
        "CONST": mu_c + sg_c * rs.standard_normal((len(ys), K_SAMPLES)),
        "HEAD": mu_h[:, None] + sg_h[:, None] * rs.standard_normal((len(ys), K_SAMPLES)),
        "FM": fm_sample(fm, Ce, K_SAMPLES, sg_c, 7),
        "FM-SHUFFLE": fm_sample(fm, Cs, K_SAMPLES, sg_c, 7),
    }

    macro = lambda v: float(np.mean([np.nanmean(np.asarray(v, float)[Sev == s]) for s in np.unique(Sev)]))  # noqa: E731
    per, table = {}, {}
    for name, S in samples.items():
        crps = crps_ensemble(S, ys)
        sd = S.std(axis=1, ddof=1)
        res = np.abs(ys - S.mean(axis=1))
        pit = (np.sum(S < ys[:, None], axis=1) + 0.5) / (K_SAMPLES + 1)
        cov = {a: np.array([np.quantile(S[i], a / 2) <= ys[i] <= np.quantile(S[i], 1 - a / 2)
                            for i in range(len(ys))]) for a in ALPHAS}
        per[name] = {"crps": crps, "sd": sd, "res": res}
        table[name] = {"crps": macro(crps), "median_sd_ms": float(np.median(sd)),
                       "median_abs_res_ms": float(np.median(res)),
                       "sharpness_spearman": float(sps.spearmanr(sd, res).statistic),
                       "pit_ks": float(sps.kstest(pit, "uniform").statistic),
                       **{f"coverage_{int((1-a)*100)}": macro(cov[a]) for a in ALPHAS}}
        print(f"[N6] {name:12s} CRPS {table[name]['crps']:7.3f}  sd {table[name]['median_sd_ms']:5.1f} ms  "
              f"|res| {table[name]['median_abs_res_ms']:5.1f} ms  sharp {table[name]['sharpness_spearman']:+.3f}  "
              f"cov50/80/90 {table[name]['coverage_50']:.3f}/{table[name]['coverage_80']:.3f}/{table[name]['coverage_90']:.3f}", flush=True)

    pairs = {}
    for a, b in (("CONST", "FM"), ("FM-SHUFFLE", "FM"), ("HEAD", "FM"), ("CONST", "HEAD")):
        pairs[f"{b}_vs_{a}:crps"] = paired_subject_bootstrap(per[a]["crps"], per[b]["crps"], Sev, "lower_better", BOOT_N, BOOT_SEED)
    subs = np.unique(Sev)
    rb = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(BOOT_N):
        pick = np.concatenate([np.flatnonzero(Sev == subs[j]) for j in rb.integers(0, len(subs), len(subs))])
        draws.append(sps.spearmanr(per["FM"]["sd"][pick], per["FM"]["res"][pick]).statistic)
    sharp_ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    # ---- §5 secondary 2: multimodality of FM's samples, Holm-corrected ----
    dips = np.array([diptest.diptest(samples["FM"][i])[1] for i in range(len(ys))])
    o = np.argsort(dips); m_ = len(dips)
    holm = np.empty(m_); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (m_ - rank) * dips[i]); holm[i] = min(run, 1.0)
    bimodal = holm < DIP_ALPHA
    head_crps_bi = macro(per["HEAD"]["crps"][bimodal]) if bimodal.any() else float("nan")
    head_crps_uni = macro(per["HEAD"]["crps"][~bimodal]) if (~bimodal).any() else float("nan")

    vC, vS, vH = pairs["FM_vs_CONST:crps"], pairs["FM_vs_FM-SHUFFLE:crps"], pairs["FM_vs_HEAD:crps"]
    sharp = table["FM"]["sharpness_spearman"]
    verdict = ("GENERATIVE SAMPLER PRESERVES PER-BEAT UNCERTAINTY"
               if vC["lo"] > 0 and sharp >= SHARPNESS_BAR and sharp_ci[0] > 0 and vS["lo"] > 0
               else "PARTIAL" if vC["lo"] > 0 else "DOES NOT PRESERVE")

    out = {"prereg": PREREG, "git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "r1_tcn": tmeta, "k_samples": K_SAMPLES,
           "counts": {"train": ctr, "internal_dev": cdv, "eval": cev},
           "const": {"mu_ms": mu_c, "sigma_ms": sg_c},
           "table": table, "paired": pairs, "fm_sharpness_ci": sharp_ci, "verdict": verdict,
           "multimodality": {"dip_alpha": DIP_ALPHA, "n_bimodal": int(bimodal.sum()),
                             "frac_bimodal": float(bimodal.mean()),
                             "head_crps_on_bimodal": head_crps_bi, "head_crps_on_unimodal": head_crps_uni},
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "diptest": diptest.__version__,
                    "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n6_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N6] FM vs CONST   CRPS {vC['point']:+.4f} [{vC['lo']:+.4f},{vC['hi']:+.4f}]")
    print(f"[N6] FM vs SHUFFLE CRPS {vS['point']:+.4f} [{vS['lo']:+.4f},{vS['hi']:+.4f}]")
    print(f"[N6] FM vs HEAD    CRPS {vH['point']:+.4f} [{vH['lo']:+.4f},{vH['hi']:+.4f}]  (positive = FM better)")
    print(f"[N6] FM sharpness {sharp:+.3f} [{sharp_ci[0]:+.3f},{sharp_ci[1]:+.3f}] (bar {SHARPNESS_BAR})")
    print(f"[N6] bimodal beats {bimodal.mean():.4f}  HEAD CRPS on them {head_crps_bi:.3f} vs {head_crps_uni:.3f} elsewhere")
    print(f"[N6] VERDICT: {verdict}   ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
