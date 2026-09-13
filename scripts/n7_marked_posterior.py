"""N7 — a calibrated timing posterior over CANDIDATE beats
(docs/N7_CALIBRATED_MARKED_POSTERIOR_PREREGISTRATION.md)

Fixes the five defects N6 named and changes nothing else. R1's Global-TCN stays FROZEN.
kjd/ssx are never loaded. NFE is pinned on internal dev before any evaluation number exists.

Run: .venv/bin/python scripts/n7_marked_posterior.py
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
from ppg2ecg.evaluation import rpeaks as RP                     # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT                  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import extract_events            # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

import n5_timing_uncertainty as N5                              # noqa: E402
import n6_generative_posterior as N6                            # noqa: E402
import r2_evaluate as R2E                                       # noqa: E402

ART = ROOT / "artifacts/n7_marked_posterior"
PREREG = "648838e"
FS, T_LEN = 128, 1024
FIELD_HALF, PPG_HALF = N5.FIELD_HALF, N5.PPG_HALF
MATCH_MS = N5.MATCH_MS
SALT, TAKE = N5.SALT, N5.TAKE
THRESH_PRIMARY, THRESH_SECONDARY, REFRACTORY = 0.35, 0.20, 32
NFE_GRID = (1, 2, 4, 8, 16, 32)
COV_TOL = 0.10
SEEDS = (42, 43, 44)
STEPS, BATCH, LR, WD = N5.STEPS, N5.BATCH, N5.LR, N5.WD
K_SAMPLES, BOOT_N, BOOT_SEED = N6.K_SAMPLES, N6.BOOT_N, N6.BOOT_SEED
SHARPNESS_BAR = N5.SHARPNESS_BAR
ALPHAS = (0.50, 0.20, 0.10)
DEV_SET = ("an0", "k2s")                                        # already-seen; never in a verdict


def folds():
    sp = json.loads((ROOT / "artifacts/r1_global_rhythm/subject_split.json").read_text())
    pool = sp["probe_train"] + sp["internal_dev"]
    perm = np.random.default_rng(20260913).permutation(len(pool))
    f = [[pool[i] for i in perm[k::4]] for k in range(4)]
    out = []
    for k in range(4):
        out.append({"eval": sorted(f[k]), "dev": sorted(f[(k + 1) % 4]),
                    "train": sorted(f[(k + 2) % 4] + f[(k + 3) % 4])})
    return out


@torch.no_grad()
def build_candidates(subjects, tcn, dev, thr, tag):
    """EVERY R1 candidate above `thr`, with a validity label and (where valid) a residual target."""
    ER.assert_no_test_subjects(subjects)
    F, P, valid, resid, S = [], [], [], [], []
    n_gt_total = n_gt_hit = 0
    half = MATCH_MS / 1000.0 * FS
    for s in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        X, Yw = d["x"], d["y"]
        idx = ER.select_subset(SALT, s, len(X), TAKE)
        Xs = X[idx].astype(np.float32)
        field = R2E.scaffolds(tcn, Xs, dev)
        for k in range(len(Xs)):
            ev = extract_events(field[k], thr, REFRACTORY)
            gt = np.asarray(RP.detect_rpeaks(Yw[idx[k]].astype(np.float64), FS), float)
            n_gt_total += len(gt)
            if len(gt):
                n_gt_hit += int(sum(np.any(np.abs(np.asarray(ev, float) - g) <= half) for g in gt)) if len(ev) else 0
            for e in ev:
                if e - FIELD_HALF < 0 or e + FIELD_HALF + 1 > T_LEN:
                    continue
                if e - PPG_HALF < 0 or e + PPG_HALF + 1 > T_LEN:
                    continue
                F.append(field[k][e - FIELD_HALF: e + FIELD_HALF + 1])
                P.append(Xs[k][e - PPG_HALF: e + PPG_HALF + 1])
                if len(gt):
                    dd = gt - float(e)
                    j = int(np.argmin(np.abs(dd)))
                    ok = abs(dd[j]) <= half
                else:
                    ok, j, dd = False, 0, np.zeros(1)
                valid.append(bool(ok))
                resid.append(float(dd[j]) / FS * 1000.0 if ok else np.nan)
                S.append(s)
    C = np.concatenate([np.asarray(F, np.float32), np.asarray(P, np.float32)], axis=1)
    valid = np.asarray(valid); resid = np.asarray(resid, np.float32); S = np.asarray(S)
    rec = {"n_candidates": int(len(C)), "precision": float(valid.mean()),
           "n_gt": int(n_gt_total), "gt_recall": float(n_gt_hit / max(n_gt_total, 1))}
    print(f"[N7] {tag} thr {thr}: {len(C):,} candidates, precision {rec['precision']:.3f}, "
          f"GT recall {rec['gt_recall']:.3f}", flush=True)
    return C, valid, resid, S, rec


class Validity(nn.Module):
    def __init__(self, n, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


def train_loop(model, loss_fn, C, extra, steps, seed, dev, dev_fn=None, tag=""):
    seed_everything(seed)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    Ct = torch.from_numpy(C)
    g = torch.Generator().manual_seed(seed)
    best, best_sd = float("inf"), None
    for step in range(1, steps + 1):
        i = torch.randint(0, len(Ct), (BATCH,), generator=g)
        loss = loss_fn(model, Ct[i].to(dev), i)
        opt.zero_grad(); loss.backward(); opt.step()
        if dev_fn is not None and (step % 2000 == 0 or step == steps):
            model.eval()
            with torch.no_grad():
                m = dev_fn(model)
            model.train()
            if m < best:
                best, best_sd = m, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    model.eval()
    return model


def coverage_at(S: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    lo = np.quantile(S, a / 2, axis=1); hi = np.quantile(S, 1 - a / 2, axis=1)
    return (lo <= y) & (y <= hi)


def run_one(fold, seed, tcn, dev, thr, tag_prefix=""):
    """One (fold, seed): shared validity head, CONST / HEAD / FM / FM-SHUFFLE timing arms."""
    tr, dv, ev = fold["train"], fold["dev"], fold["eval"]
    assert not (set(tr) & set(dv)) and not (set(tr) & set(ev)) and not (set(dv) & set(ev))
    Ctr, Vtr, Rtr, Str, rtr = build_candidates(tr, tcn, dev, thr, f"{tag_prefix}train")
    Cdv, Vdv, Rdv, Sdv, rdv = build_candidates(dv, tcn, dev, thr, f"{tag_prefix}dev")
    Cev, Vev, Rev, Sev, rev = build_candidates(ev, tcn, dev, thr, f"{tag_prefix}eval")

    # ---- shared validity head (one per fold+seed; every timing arm uses it) ----
    Vt = torch.from_numpy(Vtr.astype(np.float32))
    vh = train_loop(Validity(Ctr.shape[1]),
                    lambda m, x, i: nn.functional.binary_cross_entropy_with_logits(m(x), Vt[i].to(dev)),
                    Ctr, None, STEPS // 2, seed, dev,
                    dev_fn=lambda m: float(nn.functional.binary_cross_entropy_with_logits(
                        m(torch.from_numpy(Cdv).to(dev)), torch.from_numpy(Vdv.astype(np.float32)).to(dev))))
    with torch.no_grad():
        p_ev = torch.sigmoid(vh(torch.from_numpy(Cev).to(dev))).cpu().numpy()
    brier = float(np.mean((p_ev - Vev.astype(float)) ** 2))
    auc = float(sps.rankdata(p_ev)[Vev].mean() - (Vev.sum() + 1) / 2) / max((~Vev).sum(), 1)

    # ---- timing arms, trained on VALID candidates only ----
    mtr, mdv, mev = Vtr, Vdv, Vev
    Ctr_v, Rtr_v = Ctr[mtr], Rtr[mtr]
    Cdv_v, Rdv_v = Cdv[mdv], Rdv[mdv]
    Cev_v, Rev_v, Sev_v = Cev[mev], Rev[mev].astype(np.float64), Sev[mev]
    mu_c, sg_c = float(Rtr_v.mean()), float(Rtr_v.std(ddof=1))

    Yt = torch.from_numpy(Rtr_v)
    Cd_v = torch.from_numpy(Cdv_v).to(dev); Yd_v = torch.from_numpy(Rdv_v).to(dev)
    head = train_loop(N5.Head(2 * FIELD_HALF + 1, 2 * PPG_HALF + 1),
                      lambda m, x, i: N5.gaussian_nll(Yt[i].to(dev), *m(x)).mean(),
                      Ctr_v, None, STEPS, seed, dev,
                      dev_fn=lambda m: float(N5.gaussian_nll(Yd_v, *m(Cd_v)).mean()))

    def fm_loss(m, x, i):
        y = Yt[i].to(dev)
        e = torch.randn(len(y), device=dev) * sg_c
        t = torch.rand(len(y), device=dev)
        r = torch.where(torch.rand(len(y), device=dev) < 0.5, t, t * torch.rand(len(y), device=dev))
        return torch.mean((m.u((1 - t) * y + t * e, x, t, r) - (e - y)) ** 2)

    fm = train_loop(N6.FlowHead(Ctr.shape[1]), fm_loss, Ctr_v, None, STEPS, seed, dev)

    # ---- §4: pin the NFE on internal DEV ONLY, before any eval number ----
    dev_cov, chosen = {}, None
    for n in NFE_GRID:
        Sd = N6b_sample(fm, Cd_v, K_SAMPLES, sg_c, n, 1000)
        dev_cov[n] = float(coverage_at(Sd, Rdv_v.astype(np.float64), 0.20).mean())
    ok = [n for n in NFE_GRID if abs(dev_cov[n] - 0.80) <= COV_TOL]
    chosen = min(ok) if ok else max(NFE_GRID)

    # ---- evaluation ----
    rs = np.random.default_rng(BOOT_SEED)
    order = rs.permutation(len(Cev_v))
    bad = order == np.arange(len(Cev_v))
    while bad.any():
        order[bad] = rs.permutation(len(Cev_v))[bad]; bad = order == np.arange(len(Cev_v))
    Ce = torch.from_numpy(Cev_v).to(dev)
    with torch.no_grad():
        mu_h, ls_h = head(Ce)
    sg_h = np.exp(np.clip(ls_h.cpu().numpy(), math.log(N5.MIN_SIGMA_MS), math.log(500.0)))
    samples = {
        "CONST": mu_c + sg_c * rs.standard_normal((len(Rev_v), K_SAMPLES)),
        "HEAD": mu_h.cpu().numpy()[:, None] + sg_h[:, None] * rs.standard_normal((len(Rev_v), K_SAMPLES)),
        "FM": N6b_sample(fm, Ce, K_SAMPLES, sg_c, chosen, 7),
        "FM-SHUFFLE": N6b_sample(fm, torch.from_numpy(Cev_v[order]).to(dev), K_SAMPLES, sg_c, chosen, 7),
    }
    macro = lambda v: float(np.mean([np.nanmean(np.asarray(v, float)[Sev_v == s]) for s in np.unique(Sev_v)]))  # noqa: E731
    per, tab = {}, {}
    for name, Sm in samples.items():
        crps = N6.crps_ensemble(Sm, Rev_v)
        sd = Sm.std(axis=1, ddof=1); res = np.abs(Rev_v - Sm.mean(axis=1))
        per[name] = {"crps": crps, "sd": sd, "res": res}
        tab[name] = {"crps": macro(crps), "median_sd_ms": float(np.median(sd)),
                     "sharpness_spearman": float(sps.spearmanr(sd, res).statistic),
                     **{f"coverage_{int((1-a)*100)}": macro(coverage_at(Sm, Rev_v, a)) for a in ALPHAS}}
    pr = {}
    for a, b in (("CONST", "FM"), ("FM-SHUFFLE", "FM"), ("HEAD", "FM")):
        pr[f"FM_vs_{a}:crps"] = paired_subject_bootstrap(per[a]["crps"], per[b]["crps"], Sev_v,
                                                         "lower_better", BOOT_N, BOOT_SEED)
    subs = np.unique(Sev_v); rb = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(500):
        # ONE resampled index set per replicate, applied to BOTH arrays -- drawing them
        # separately silently pairs different beats and is what the first version did.
        pick = np.concatenate([np.flatnonzero(Sev_v == subs[j]) for j in rb.integers(0, len(subs), len(subs))])
        draws.append(sps.spearmanr(per["FM"]["sd"][pick], per["FM"]["res"][pick]).statistic)
    sharp_ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    c50, c80 = tab["FM"]["coverage_50"], tab["FM"]["coverage_80"]
    cal = abs(c80 - 0.80) <= COV_TOL and abs(c50 - 0.50) <= COV_TOL
    ii = pr["FM_vs_CONST:crps"]["lo"] > 0
    iii = tab["FM"]["sharpness_spearman"] >= SHARPNESS_BAR and sharp_ci[0] > 0
    iv = pr["FM_vs_FM-SHUFFLE:crps"]["lo"] > 0
    verdict = ("CALIBRATED PER-BEAT POSTERIOR" if cal and ii and iii and iv
               else "SHARP BUT OVERCONFIDENT" if ii and iii and iv else "NOT SUPPORTED")

    dips = np.array([diptest.diptest(samples["FM"][i])[1] for i in range(len(Rev_v))])
    o = np.argsort(dips); m_ = len(dips); run = 0.0; holm = np.empty(m_)
    for rank, i in enumerate(o):
        run = max(run, (m_ - rank) * dips[i]); holm[i] = min(run, 1.0)
    bim = holm < 0.05

    return {"eval_subjects": ev, "dev_subjects": dv, "seed": seed, "threshold": thr,
            "counts": {"train": rtr, "dev": rdv, "eval": rev},
            "validity": {"brier": brier, "auc": auc, "eval_precision": float(Vev.mean())},
            "nfe_dev_coverage80": dev_cov, "nfe_chosen": int(chosen),
            "table": tab, "paired": pr, "fm_sharpness_ci": sharp_ci, "verdict": verdict,
            "multimodality": {"frac_bimodal": float(bim.mean()), "n_bimodal": int(bim.sum()),
                              "head_crps_bimodal": macro(per["HEAD"]["crps"][bim]) if bim.any() else None,
                              "head_crps_unimodal": macro(per["HEAD"]["crps"][~bim]) if (~bim).any() else None}}


@torch.no_grad()
def N6b_sample(net, c, k, scale, nfe, gen_seed):
    """MeanFlow schedule, uniform steps (the N6-post sampler, imported by value)."""
    n = len(c)
    g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
    grid = torch.linspace(1.0, 0.0, nfe + 1)
    out = []
    for _ in range(k):
        z = (torch.randn(n, generator=g) * scale).to(c.device)
        for i in range(nfe):
            t = torch.full((n,), float(grid[i]), device=c.device)
            r = torch.full((n,), float(grid[i + 1]), device=c.device)
            z = z - (t - r) * net.u(z, c, t, r)
        out.append(z.cpu().numpy())
    return np.stack(out, axis=1)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    ER.assert_no_test_subjects(list(DEV_SET))
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    assert not any(p.requires_grad for p in tcn.parameters())
    FOLDS = folds()
    for k, f in enumerate(FOLDS):
        print(f"[N7] fold {k}: eval {f['eval']}  dev {f['dev']}  train {f['train']}", flush=True)

    runs = []
    for k, f in enumerate(FOLDS):
        for seed in SEEDS:
            r = run_one(f, seed, tcn, dev, THRESH_PRIMARY, f"f{k}s{seed} ")
            r["fold"] = k
            runs.append(r)
            t = r["table"]["FM"]
            print(f"[N7] fold {k} seed {seed}  NFE {r['nfe_chosen']:2d}  CRPS {t['crps']:7.3f}  "
                  f"sharp {t['sharpness_spearman']:+.3f}  cov50/80 {t['coverage_50']:.3f}/{t['coverage_80']:.3f}  "
                  f"validity Brier {r['validity']['brier']:.4f}  -> {r['verdict']}", flush=True)

    n_cal = sum(r["verdict"] == "CALIBRATED PER-BEAT POSTERIOR" for r in runs)
    stage = "SUPPORTED" if n_cal >= 9 else ("PARTIAL" if n_cal >= 5 else "NOT SUPPORTED")
    out = {"prereg": PREREG, "git": git_sha(ROOT), "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "r1_tcn": tmeta, "folds": FOLDS, "seeds": list(SEEDS),
           "threshold_primary": THRESH_PRIMARY, "nfe_grid": list(NFE_GRID),
           "runs": runs, "n_calibrated": n_cal, "n_runs": len(runs), "stage_verdict": stage,
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "diptest": diptest.__version__,
                    "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n7_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N7] STAGE VERDICT: {stage}  ({n_cal}/{len(runs)} calibrated)  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
