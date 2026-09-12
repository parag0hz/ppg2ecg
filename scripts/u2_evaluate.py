"""U2 evaluation (docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md §7-§9).

Frozen-checkpoint inference only. Both arms are scored by ONE code path on byte-identical
test windows with byte-identical noise, which is what makes the §9 pairing valid.

NFE grid, all points published (§7): arm I 1/2/4 (MeanFlow schedule);
arm C 1 (Euler 1), 2 (Heun 1), 4 (Heun 2), 50 (Heun 25).

Run: .venv/bin/python scripts/u2_evaluate.py --corpus u2_bidmc
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import paper_metrics as PMX
from ppg2ecg.evaluation import penguin_metrics as PM
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.flow.samplers import euler_sample, heun_sample, nfe_of
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/u2_paired"
BOOT_N, BOOT_SEED, BATCH = 2000, 20260911, 64
FS, SEG = 128, 4
EVAL_TARGET = 12000   # deviation U2-D5: evaluation-window budget, see `eval_subset`

CORPORA = {   # slug: (task, PENGUIN metric window in seconds)
    "u2_dalia": ("ECG", 8), "u2_wildppg": ("ECG", 8),
    "u2_bidmc": ("Resp", 60), "u2_wesad": ("Resp", 60),
    "u2_ucibp": ("ABP", 8), "u2_mimicbp": ("ABP", 8),
}
NFES = {"C": (1, 2, 4, 50), "I": (1, 2, 4)}
# orientation: +1 means lower is better
LOWER_BETTER = {"HR", "RR", "SBP", "DBP", "RR_MAE_ms", "MAE_HR", "MAE", "RMSE", "FD_fid", "FD_discrete"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_test(slug: str):
    man = ROOT / "data/manifests" / f"split_{slug}_seed42.json"
    split = json.loads(man.read_text())["splits"][0]
    if slug == "u2_wildppg":
        real = json.loads(man.read_text())["extra"]["wildppg_subject_map"]
        for s in split["test"]:
            assert not any(b in real[s] for b in ("kjd", "ssx")), f"FIREWALL: {real[s]} in test"
    X, Y, S = [], [], []
    for s in split["test"]:
        d = np.load(ROOT / "data/processed" / slug / f"{s}.npz")
        X.append(d["x"].astype(np.float32))
        Y.append(d["y"].astype(np.float64))
        S.append(np.full(len(d["x"]), s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), split


def eval_subset(subj: np.ndarray, k: int, target: int = EVAL_TARGET) -> np.ndarray:
    """Deviation U2-D5 -- deterministic evaluation-window budget.

    The preregistration costed training (§12) but not evaluation. Measured on this GPU, scoring the
    FULL test splits at 4 noise draws over the 7-point NFE grid is 26 GPU-hours, dominated by arm C
    at NFE 50 on the three large corpora. This caps the windows scored, per subject, so the cost is
    bounded without changing anything a result could depend on:

      * the cap is fixed by compute alone, before any large-corpus number exists, and BIDMC (already
        evaluated at full size) is under it, so no result informed the choice;
      * selection is exact `linspace` within each subject -- never an integer stride, the defect D1
        had to fix -- so it is deterministic and spans each recording;
      * Resp/ABP metric windows concatenate `k` CONSECUTIVE segments, so selection is over blocks of
        k, never individual windows, and contiguity inside a metric window is preserved;
      * the identical index set is used for both arms and every NFE point, so the §9 pairing holds.

    Corpora whose test split is already under the cap are untouched.
    """
    subs = np.unique(subj)
    per = max(k, int(np.ceil(target / len(subs)) // k * k))
    keep = []
    for s in subs:
        idx = np.flatnonzero(subj == s)
        n_blocks = len(idx) // k
        want = min(per // k, n_blocks)
        if want <= 0:
            continue
        blocks = np.unique(np.linspace(0, n_blocks - 1, want).round().astype(int))
        keep.append(np.concatenate([idx[b * k:(b + 1) * k] for b in blocks]))
    return np.sort(np.concatenate(keep))


def build(ckpt: Path, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_cfg" not in ck:
        # U3-B: checkpoint_last.pt carries only weights + optimiser + RNG + train_state. The
        # architecture config lives in checkpoint_best.pt of the SAME run, written by the same
        # process from the same args, so it is the config these weights were trained under.
        sib = torch.load(ckpt.with_name("checkpoint_best.pt"), map_location="cpu", weights_only=False)
        for k in ("model_cfg", "imf_cfg", "args", "target_norm"):
            if k in sib:
                ck[k] = sib[k]
    bare = not any(k.startswith("backbone.") for k in ck["state_dict"])
    backbone = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval()
    if bare:                                     # arm C: OT-CFM stores the bare backbone
        backbone.load_state_dict(ck["state_dict"])
        backbone.requires_grad_(False)
        return backbone, ck, "C"
    cfg = ck.get("imf_cfg", {}) or {}
    net = MeanFlowS5(backbone, cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    return net, ck, "I"


@torch.no_grad()
def generate(net, X, e0, nfe, dev, arm):
    outs, got = [], set()
    t0 = time.perf_counter()
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        z0 = e0[i:i + BATCH].to(dev)
        if arm == "C":
            v = lambda x, t, _p=pp: net.forward_step(x, _p, t)  # noqa: E731
            if nfe == 1:
                z, k = euler_sample(v, z0, 1)
            else:
                steps = nfe // 2
                assert nfe_of("heun", steps) == nfe, f"Heun cannot realise NFE {nfe}"
                z, k = heun_sample(v, z0, steps)
        else:
            z, k = ER.sample_meanflow_schedule(net, pp, z0, ER.UNIFORM[nfe])
        got.add(int(k))
        outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {nfe}, (arm, nfe, got)
    return np.concatenate(outs).astype(np.float64), time.perf_counter() - t0


# ---------------------------------------------------------------- per-unit metrics
def resp_wave_corr(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """U2 co-primary: 1 Hz low-pass applied to BOTH signals, then Pearson r per metric window.
    Upstream low-passes the prediction only (help_func.py:191), which is asymmetric and, being
    FFT-argmax downstream, phase-blind."""
    b, a = sps.butter(8, 1.0 / (0.5 * FS), btype="low")
    p = sps.filtfilt(b, a, pred, axis=-1)
    t = sps.filtfilt(b, a, target, axis=-1)
    p = p - p.mean(-1, keepdims=True)
    t = t - t.mean(-1, keepdims=True)
    den = np.sqrt((p ** 2).sum(-1) * (t ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (p * t).sum(-1) / den, np.nan)


def task_metrics(task: str, window_s: int, pred: np.ndarray, tgt: np.ndarray, subj: np.ndarray):
    """Returns {metric: (values, subjects)} at that metric's own unit (window or metric-window)."""
    out = {"MAE": (np.abs(pred - tgt).mean(-1), subj),
           "RMSE": (np.sqrt(((pred - tgt) ** 2).mean(-1)), subj)}
    if task in ("Resp", "ABP"):
        k = PM.segments_per_metric_window(window_s, SEG)
        P, T, S = [], [], []
        for s in np.unique(subj):
            m = subj == s
            n = (m.sum() // k) * k
            if n == 0:
                continue
            P.append(PM.concat_windows(pred[m][:n], k))
            T.append(PM.concat_windows(tgt[m][:n], k))
            S.append(np.full(n // k, s))
        P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
        if task == "Resp":
            out["RR"] = (PM.resp_rate_error(P, T), S)
            out["Resp_corr"] = (resp_wave_corr(P, T), S)
        else:
            out["SBP"] = (PM.sbp_error(P, T), S)
            out["DBP"] = (PM.dbp_error(P, T), S)
    else:  # ECG: beat-level table, one row per window
        tab = PMX.paper_metric_table(pred, tgt, fs=FS)
        out["HR"] = (tab["hr_abs_err"], subj)
        out["Rpeak_F1"] = (tab["rpeak_f1_50ms"], subj)
        out["RR_MAE_ms"] = (tab["rr_mae_ms"], subj)
        out["MAE_HR"] = (tab["hr_abs_err"], subj)          # KANFlow's name for the same quantity
        out["_counts"] = (tab, subj)
    return out


FD_MAX = 3000  # deterministic linspace subsample for the two O(T^2)/covariance metrics


def fd_subset(n: int) -> np.ndarray:
    """Exact linspace selection (the D1 rule), never an integer stride -- a stride silently halves
    small corpora, which is the defect D1 had to fix."""
    if n <= FD_MAX:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, FD_MAX).round().astype(int))


def pooled_extras(task: str, pred: np.ndarray, tgt: np.ndarray, tab) -> dict:
    """Corpus-level scalars with no per-subject unit: FD (KANFlow Eq. 23) and micro/macro F1."""
    i = fd_subset(len(pred))
    ex = {"FD_kanflow": float(PMX.kanflow_fd(pred[i], tgt[i])),
          "FD_discrete": float(np.nanmean(PMX.discrete_frechet(pred[i], tgt[i]))),
          "_fd_n": int(len(i))}
    if task == "ECG" and tab is not None:
        ex["Micro_F1"] = float(PMX.micro_f1(tab["n_tp_50ms"], tab["n_fp_50ms"], tab["n_fn_50ms"]))
        ex["Macro_F1"] = float(PMX.macro_f1(tab["rpeak_f1_50ms"], tab["n_ref_beats"])[0])
    return ex


def paired_cluster_bootstrap(c, i, subj, orient, n=BOOT_N, seed=BOOT_SEED):
    """Prereg §9: the CLUSTER (= subject) is the resampling unit, drawn once per replicate and
    applied to the per-window paired difference, so the interval is on the difference itself.

    This is the paired form of `d1_common.subject_cluster_bootstrap`, not
    `paired_stats.paired_subject_bootstrap` -- the latter holds the subject set fixed and resamples
    windows WITHIN each subject, which measures a different thing. Both are reported; only this one
    drives the §9 verdict.
    """
    d = (np.asarray(i, float) - np.asarray(c, float)) if orient == "higher_better" else \
        (np.asarray(c, float) - np.asarray(i, float))     # positive = arm I better, either way
    subs = np.unique(subj)
    per = np.array([np.nanmean(d[subj == s]) for s in subs])
    if not np.isfinite(per).any():
        return float("nan"), float("nan"), float("nan"), len(subs)
    rng = np.random.default_rng(seed)
    draws = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(n)])
    return (float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)), int(len(subs)))


def summarise(vals, subj):
    subs = np.unique(subj)
    per = np.array([np.nanmean(vals[subj == s]) for s in subs])
    if not np.isfinite(per).any():
        return float("nan"), float("nan"), float("nan"), len(vals), len(subs)
    rng = np.random.default_rng(BOOT_SEED)
    d = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(BOOT_N)])
    return (float(np.nanmean(per)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5)),
            int(len(vals)), int(len(subs)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--noise-draws", type=int, default=4, help="prereg §7: 4 draws, seeds 0..n-1")
    ap.add_argument("--checkpoint", choices=["best", "last"], default="best",
                    help="U3-B: 'last' = both arms at exactly 14,000 optimizer steps, breaking the "
                         "selection asymmetry (arm C peaks at 80%% of the budget, arm I at 37%%)")
    args = ap.parse_args()

    slug = args.corpus
    task, window_s = CORPORA[slug]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    X, Y, S, split = load_test(slug)
    n_full = len(X)
    k_seg = PM.segments_per_metric_window(window_s, SEG) if task in ("Resp", "ABP") else 1
    sel = eval_subset(S, k_seg)
    X, Y, S = X[sel], Y[sel], S[sel]
    print(f"[u2] {slug}: task={task} test windows={len(X)} of {n_full} "
          f"({'uncapped' if len(X) == n_full else f'U2-D5 cap, blocks of {k_seg}'}) "
          f"subjects={len(split['test'])}", flush=True)

    nets, cks = {}, {}
    for arm in ("C", "I"):
        ck_path = ROOT / f"outputs/{slug}_arm{arm}_seed42/checkpoint_{args.checkpoint}.pt"
        assert ck_path.exists(), f"missing checkpoint: {ck_path}"
        net, ck, kind = build(ck_path, dev)
        assert kind == arm, f"{ck_path} is a {kind} checkpoint but was loaded as arm {arm}"
        realised = int((ck.get("train_state") or {}).get("opt_steps", -1))
        if args.checkpoint == "last":
            assert realised == 14000, f"{ck_path}: opt_steps {realised}, expected 14000 (U3-B §4)"
        nets[arm], cks[arm] = net, {"path": str(ck_path.relative_to(ROOT)), "sha256": sha256(ck_path),
                                    "epoch": int(ck["epoch"]), "opt_steps": realised,
                                    "steps": int(ck.get("args", {}).get("max_steps") or -1)}

    # per-unit metric values, averaged over the noise draws; draws share e0 between arms (pairing)
    acc: dict[tuple, list] = {}
    extras: dict[tuple, list] = {}
    subj_of: dict[tuple, np.ndarray] = {}
    gen_s: dict[tuple, float] = {}
    for seed in range(args.noise_draws):
        g = torch.Generator().manual_seed(seed)
        e0 = torch.randn(len(X), 1, X.shape[1], generator=g)
        for arm in ("C", "I"):
            for nfe in NFES[arm]:
                pred, dt = generate(nets[arm], X, e0, nfe, dev, arm)
                gen_s[(arm, nfe)] = gen_s.get((arm, nfe), 0.0) + dt
                mt = task_metrics(task, window_s, pred, Y, S)
                tab = mt.pop("_counts", (None, None))[0]
                for m, (v, sub) in mt.items():
                    acc.setdefault((arm, nfe, m), []).append(np.asarray(v, float))
                    subj_of[(arm, nfe, m)] = sub
                for m, v in pooled_extras(task, pred, Y, tab).items():
                    extras.setdefault((arm, nfe, m), []).append(v)
                print(f"  seed {seed} arm {arm} NFE {nfe:2d}  gen {dt:6.1f}s", flush=True)
        del e0

    rows = []
    for (arm, nfe, m), draws in sorted(acc.items()):
        stack = np.vstack(draws)                       # [draws, units]
        mean_over_draws = np.nanmean(stack, axis=0)
        mu, lo, hi, n_u, n_s = summarise(mean_over_draws, subj_of[(arm, nfe, m)])
        per_draw = [summarise(d, subj_of[(arm, nfe, m)])[0] for d in draws]
        rows.append(dict(corpus=slug, task=task, arm=arm, nfe=nfe, metric=m, unit="per-subject-macro",
                         value=mu, ci_lo=lo, ci_hi=hi, draw_std=float(np.nanstd(per_draw)),
                         n_units=n_u, n_subjects=n_s, gen_seconds=round(gen_s[(arm, nfe)], 1)))
    for (arm, nfe, m), vals in sorted(extras.items()):
        if m == "_fd_n":
            continue
        rows.append(dict(corpus=slug, task=task, arm=arm, nfe=nfe, metric=m, unit="corpus-pooled",
                         value=float(np.nanmean(vals)), ci_lo=float("nan"), ci_hi=float("nan"),
                         draw_std=float(np.nanstd(vals)), n_units=int(extras[(arm, nfe, "_fd_n")][0]),
                         n_subjects=len(split["test"]), gen_seconds=round(gen_s[(arm, nfe)], 1)))

    # ---- §8/§9: paired non-inferiority of arm I at NFE k against arm C at NFE 50, plus the gate
    pairs = []
    metrics = sorted({m for (_a, _n, m) in acc})
    for m in metrics:
        base = acc.get(("C", 50, m))
        if base is None:
            continue
        sub = subj_of[("C", 50, m)]
        c50 = np.nanmean(np.vstack(base), axis=0)
        c1 = np.nanmean(np.vstack(acc[("C", 1, m)]), axis=0)
        orient = "lower_better" if m in LOWER_BETTER else "higher_better"
        delta_c = abs(summarise(c1, sub)[0] - summarise(c50, sub)[0])
        for k in NFES["I"]:
            ik = np.nanmean(np.vstack(acc[("I", k, m)]), axis=0)
            pt, lo, hi, ns = paired_cluster_bootstrap(c50, ik, sub, orient)
            wi = paired_subject_bootstrap(c50, ik, sub, orient, n_boot=BOOT_N, seed=BOOT_SEED)
            pairs.append(dict(corpus=slug, metric=m, k=k, orient=orient,
                              armC_nfe50=summarise(c50, sub)[0], armI_nfek=summarise(ik, sub)[0],
                              diff_positive_means_I_better=pt, ci_lo=lo, ci_hi=hi,
                              within_subject_ci_lo=wi["lo"], within_subject_ci_hi=wi["hi"],
                              armC_span_nfe1_to_50=delta_c, n_subjects=ns))

    tag = "" if args.checkpoint == "best" else "_last"
    with open(OUT / f"metrics_{slug}{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / f"paired_{slug}{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    (OUT / f"meta_{slug}{tag}.json").write_text(json.dumps(
        {"corpus": slug, "task": task, "checkpoint": args.checkpoint, "metric_window_s": window_s, "segment_len_s": SEG,
         "test_subjects": split["test"], "n_test_windows": int(len(X)), "n_test_windows_full": int(n_full),
         "eval_window_budget": EVAL_TARGET, "metric_window_segments": int(k_seg),
         "noise_draws": args.noise_draws, "checkpoints": cks,
         "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "rule": "paired, subject-clustered, equal subject weight"},
         "fd_subsample_max": FD_MAX}, indent=1))
    print(f"[u2] wrote metrics_{slug}{tag}.csv ({len(rows)} rows), paired_{slug}{tag}.csv ({len(pairs)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
