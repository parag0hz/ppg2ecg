"""M2 evaluation (docs/M2_STRUCTURE_WEIGHTED_IMEANFLOW_PREREGISTRATION.md §10-§13).

NO TRAINING. Frozen-checkpoint forward inference on the frozen VAL cohort (an0 + k2s). kjd/ssx are never loaded.
No oracle shift, no cross-correlation alignment, no DTW: nothing is ever translated.

Per window: M1's frozen region_errors / qrs_core_morphology / spectral_metrics, plus event correspondence at
tolerances 50/100/150/200 ms with the s1_audit chance floor, so raw F1 is never reported alone.
Paired effects use the E2 clustering rule: all site rows sharing one target ECG move together, subject-stratified,
equal subject weight, 2000 replicates, default_rng(20260904).

Run: .venv/bin/python scripts/m2_evaluate.py [--nfes 1,2,4] [--seeds 0] [--limit N]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/m2_structure_weighted_imeanflow"
VAL = ("an0", "k2s")
FS, T_LEN, BATCH = 128, 1024, 64
TOLS = (50.0, 100.0, 150.0, 200.0)
BOOT_N, BOOT_SEED = 2000, 20260904
ARMS = {"U": "outputs/m2_arm_U_seed42/checkpoint_best.pt",
        "S": "outputs/m2_arm_S_seed42/checkpoint_best.pt",
        "X": "outputs/m2_arm_X_seed42/checkpoint_best.pt"}


def _peaks(sig):
    return R.detect_rpeaks(sig, FS, "neurokit")


def _score(task):
    """One chunk: per-window structural + event rows. Runs in a worker process."""
    pred, gt, gpk = task
    out = []
    for i in range(len(pred)):
        row = M.region_errors(pred[i], gt[i], gpk[i])
        row |= M.qrs_core_morphology(pred[i], gt[i], gpk[i])
        row |= M.spectral_metrics(pred[i], gt[i])
        ppk = _peaks(pred[i])
        rng = np.random.default_rng(1000 + i)
        for tol in TOLS:
            m, fp, fn = R.match_rpeaks(gpk[i], ppk, FS, tol)
            pr, rc, f1 = R.prf(len(m), fp, fn)
            row[f"f1@{int(tol)}"] = f1
            if tol == 50.0:
                row |= {"precision@50": pr, "recall@50": rc, "missing@50": int(fn), "spurious@50": int(fp)}
                # chance floor: the same detector output scored against a phase-randomised beat train (s1_audit)
                fl = []
                for _ in range(8):
                    ch = S1.chance_random_phase(len(ppk), T_LEN, rng)
                    mm, ffp, ffn = R.match_rpeaks(gpk[i], np.sort(ch), FS, tol)
                    fl.append(R.prf(len(mm), ffp, ffn)[2])
                row["floor_f1@50"] = float(np.mean(fl))
                row["f1_excess@50"] = float(f1 - np.mean(fl))
        row["n_pred_beats"] = int(len(ppk))
        row["n_ref_beats"] = int(len(gpk[i]))
        row["beats_ratio_dev"] = abs(len(ppk) / max(len(gpk[i]), 1) - 1.0) if len(gpk[i]) else np.nan
        row["rr_mae_ms"] = R.rr_mae_ms(gpk[i], ppk, R.match_rpeaks(gpk[i], ppk, FS, 50.0)[0], FS)
        err = pred[i] - gt[i]
        row["rmse"] = float(np.sqrt((err ** 2).mean()))
        pc, tc = pred[i] - pred[i].mean(), gt[i] - gt[i].mean()
        row["corr"] = float((pc * tc).sum() / (np.sqrt((pc ** 2).sum() * (tc ** 2).sum()) + 1e-12))
        out.append(row)
    return out


def load_val(limit):
    X, Y, SUB, WIDX, SITE = [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        n = len(d["x"]) if not limit else min(limit, len(d["x"]))
        X.append(d["x"][:n].astype(np.float32))
        Y.append(d["y"][:n].astype(np.float32))
        SUB.append(np.full(n, s))
        WIDX.append(d["window_index"][:n])
        SITE.append(d["site"][:n])
    return (np.concatenate(X), np.concatenate(Y).astype(np.float64),
            np.concatenate(SUB), np.concatenate(WIDX), np.concatenate(SITE))


def build(ckpt, dev):
    """`checkpoint_last.pt` is a RESUME checkpoint: it carries state_dict but no model_cfg/imf_cfg. Take the
    architecture from the sibling `checkpoint_best.pt` (same run, same construction) and load the requested weights."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_cfg" not in ck:
        base = torch.load(Path(ckpt).with_name("checkpoint_best.pt"), map_location="cpu", weights_only=False)
        assert set(base["state_dict"]) == set(ck["state_dict"]), "resume checkpoint does not match the run's architecture"
        ck = {**base, **{k: v for k, v in ck.items() if k in ("state_dict", "epoch")}}
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    return net, ck


@torch.no_grad()
def generate(net, X, e0, nfe, dev):
    outs, got = [], set()
    t0 = time.perf_counter()
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        got.add(int(k))
        outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {nfe}, (nfe, got)
    return np.concatenate(outs).astype(np.float64), time.perf_counter() - t0


def clustered_paired_bootstrap(dU, dS, cluster, subject, orient, n_boot=BOOT_N, seed=BOOT_SEED):
    """effect = U-S for error metrics, S-U for higher-is-better. Resamples CLUSTERS within subject, equal subject weight."""
    diff = (dU - dS) if orient == "lower_better" else (dS - dU)
    keys, inv = np.unique(cluster, return_inverse=True)
    subs = np.unique(subject)
    csub = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        csub[i] = subject[inv == i][0]
    cmean = np.array([np.nanmean(diff[inv == i]) for i in range(len(keys))])
    per_sub = {s: np.flatnonzero(csub == s) for s in subs}
    point = float(np.nanmean([np.nanmean(cmean[per_sub[s]]) for s in subs]))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.nanmean([np.nanmean(cmean[rng.choice(per_sub[s], len(per_sub[s]), replace=True)]) for s in subs])
    return {"point": point, "lo": float(np.nanpercentile(draws, 2.5)), "hi": float(np.nanpercentile(draws, 97.5)),
            "n_clusters": int(len(keys)), "n_subjects": int(len(subs))}


def macro(vals, cluster, subject):
    keys, inv = np.unique(cluster, return_inverse=True)
    csub = np.array([subject[inv == i][0] for i in range(len(keys))])
    cmean = np.array([np.nanmean(vals[inv == i]) for i in range(len(keys))])
    return float(np.nanmean([np.nanmean(cmean[csub == s]) for s in np.unique(subject)]))


# Orientation, decided by the metric's own definition, not by any observed value. The frozen HF metric is the
# 15-64 Hz band deviation `F4__ratio_dev` (m1_structural.BANDS), which is the "HF fraction error" of prereg §10.
# Every m1_structural region_errors column is an ERROR MAGNITUDE (a1_abs = mean |err|, a2_sq = mean err^2,
# a3_dabs = mean |d1 err|, a4_amp = amplitude deviation), as is qrs_rmse_core. An earlier suffix list omitted
# "_abs", "_sq", "_dabs", "_amp" and "_core", so 13 such columns were labelled higher-is-better and their effect
# and rel_improvement carried an INVERTED SIGN. The gate and non-inferiority metrics were never affected, but the
# secondary rows were wrong; the rule below is explicit rather than suffix-guessed.
LOWER_SUFFIX = ("_dev", "_err", "_rmse", "_abs", "_sq", "_dabs", "_amp", "_core", "_ms")
LOWER_EXACT = {"rmse", "rr_mae_ms", "missing@50", "spurious@50", "qrs_curvature_err", "qrs_rmse_core"}
# `*__gt_energy` / `*__pred_energy` are raw band energies, not errors: neither direction is "better".
NEUTRAL_SUFFIX = ("__gt_energy", "__pred_energy", "_beats")
HIGHER_EXACT = {"corr", "precision@50", "recall@50", "floor_f1@50", "f1_excess@50"}
HF_METRIC = "F4__ratio_dev"


def is_lower_better(k: str) -> bool:
    if k in HIGHER_EXACT or k.startswith("f1@"):
        return False
    if k.endswith(NEUTRAL_SUFFIX):
        return False                       # reported, but no orientation is claimed for it
    return k in LOWER_EXACT or k.endswith(LOWER_SUFFIX)


def is_neutral(k: str) -> bool:
    """Raw energies and beat counts have no better/worse direction; the effects table marks them as such."""
    return k.endswith(NEUTRAL_SUFFIX) or k.endswith("__err_energy")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfes", default="1,2,4")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--arms", default="U,S")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", choices=["best", "last"], default="best",
                    help="POST-VERDICT DIAGNOSTIC ONLY (prereg §16 has no such clause; this cannot change the verdict): "
                         "'last' compares the final-epoch checkpoints instead of the fixed_imf_mse-selected ones")
    args = ap.parse_args()
    nfes = [int(x) for x in args.nfes.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    arms = args.arms.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)

    X, Y, SUB, WIDX, SITE = load_val(args.limit)
    cluster = np.array([f"{s}:{w}" for s, w in zip(SUB, WIDX)])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[m2-eval] {len(X)} val rows | {len(np.unique(cluster))} ECG clusters | {len(np.unique(SUB))} subjects", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        gt_pk = list(ex.map(_peaks, list(Y), chunksize=64))
    print(f"[m2-eval] GT beats {sum(len(p) for p in gt_pk)}", flush=True)

    rows, timings, manifest = [], [], []
    for arm in arms:
        p = ROOT / ARMS[arm].replace("checkpoint_best.pt", f"checkpoint_{args.ckpt}.pt")
        if not p.exists():
            print(f"[m2-eval] arm {arm}: checkpoint absent, skipped")
            continue
        net, ck = build(p, dev)
        manifest.append({"arm": arm, "path": str(p.relative_to(ROOT)), "ckpt_kind": args.ckpt, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "best_epoch": int(ck.get("epoch", -1)),
                         "m2_arm": ck["args"].get("m2_arm"), "seed": ck["args"].get("seed")})
        for sd in seeds:
            e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(sd))
            for nfe in nfes:
                if sd != seeds[0] and nfe != 4:
                    continue                      # §13: source stability is examined at the primary NFE only
                pred, dt = generate(net, X, e0, nfe, dev)
                ch = [(i, min(len(X), i + 256)) for i in range(0, len(X), 256)]
                out = []
                with ProcessPoolExecutor(args.workers) as ex:
                    for r in ex.map(_score, [(pred[i:j], Y[i:j], gt_pk[i:j]) for i, j in ch]):
                        out += r
                for i, r in enumerate(out):
                    rows.append({"arm": arm, "nfe": nfe, "source_seed": sd, "subject": SUB[i],
                                 "window_index": int(WIDX[i]), "site": SITE[i], "cluster": cluster[i], **r})
                timings.append({"arm": arm, "nfe": nfe, "source_seed": sd, "gen_seconds": dt,
                                "windows": int(len(X)), "windows_per_s": len(X) / dt})
                print(f"[m2-eval] {arm} NFE{nfe} seed{sd}: {dt:.0f}s gen, {len(out)} rows", flush=True)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    keys = [k for k in rows[0] if k not in ("arm", "nfe", "source_seed", "subject", "window_index", "site", "cluster")]
    with open(OUT / f"validation_metrics{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # paired effects, primary seed only
    eff = []
    if "U" in arms and "S" in arms:
        for nfe in nfes:
            iu = [r for r in rows if r["arm"] == "U" and r["nfe"] == nfe and r["source_seed"] == seeds[0]]
            is_ = [r for r in rows if r["arm"] == "S" and r["nfe"] == nfe and r["source_seed"] == seeds[0]]
            assert len(iu) == len(is_) == len(X)
            cl = np.array([r["cluster"] for r in iu])
            sb = np.array([r["subject"] for r in iu])
            for k in keys:
                u = np.array([r[k] for r in iu], dtype=np.float64)
                s = np.array([r[k] for r in is_], dtype=np.float64)
                if not (np.isfinite(u).any() and np.isfinite(s).any()):
                    continue
                orient = "neutral" if is_neutral(k) else ("lower_better" if is_lower_better(k) else "higher_better")
                b = clustered_paired_bootstrap(u, s, cl, sb, "lower_better" if orient == "neutral" else orient)
                mu, ms = macro(u, cl, sb), macro(s, cl, sb)
                eff.append({"nfe": nfe, "metric": k, "orientation": orient, "U": mu, "S": ms,
                            "effect": b["point"], "ci_lo": b["lo"], "ci_hi": b["hi"],
                            "rel_improvement": (mu - ms) / abs(mu) if orient == "lower_better" and mu else
                                               (ms - mu) / abs(mu) if mu else np.nan,
                            "ci_excludes_zero": bool(b["lo"] > 0 or b["hi"] < 0),
                            "n_clusters": b["n_clusters"], "n_subjects": b["n_subjects"]})
                # per-subject effect (§18)
                for sub in np.unique(sb):
                    m = sb == sub
                    bb = clustered_paired_bootstrap(u[m], s[m], cl[m], sb[m], orient)
                    eff[-1][f"effect_{sub}"] = bb["point"]
                    eff[-1][f"rel_{sub}"] = ((macro(u[m], cl[m], sb[m]) - macro(s[m], cl[m], sb[m]))
                                             / abs(macro(u[m], cl[m], sb[m])) if orient == "lower_better"
                                             else (macro(s[m], cl[m], sb[m]) - macro(u[m], cl[m], sb[m]))
                                             / abs(macro(u[m], cl[m], sb[m])))
        with open(OUT / f"bootstrap_effects{args.tag}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff[0]))
            w.writeheader()
            w.writerows(eff)

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / f"evaluation_meta{args.tag}.json").write_text(json.dumps(
        {"head": head, "val_subjects": list(VAL), "n_rows": int(len(X)),
         "n_clusters": int(len(np.unique(cluster))), "nfes": nfes, "source_seeds": seeds, "arms": arms,
         "tolerances_ms": list(TOLS), "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED,
         "rule": "E2 clustering: all site rows sharing one target ECG move together; subject-stratified; equal subject weight"},
         "checkpoints": manifest, "timings": timings,
         "no_alignment": "no oracle shift, no cross-correlation, no DTW — nothing is translated"}, indent=1))
    print(f"[m2-eval] wrote validation_metrics{args.tag}.csv ({len(rows)} rows), bootstrap_effects{args.tag}.csv ({len(eff)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
