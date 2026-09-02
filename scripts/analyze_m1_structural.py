"""M1 — C1 structural mechanism audit. Existing C1 checkpoints only.

Frozen protocol: docs/M1_C1_STRUCTURAL_MECHANISM_AUDIT_PREREGISTRATION.md (959eb60).
NO TRAINING. NO TEST SUBJECTS. No prediction is ever translated; no oracle statistic is computed.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.c2_cohort import SITES, atlas_cohort
from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/m1_c1_structural_audit"
FS, T_LEN, BATCH, WORKERS = 128, 1024, 64, 12
VAL, SALT, TAKE, NFES, SRC_SEED = ("an0", "k2s"), "x4-event-nfe-v2", 1024, (2, 4), 0
CKPT = {"B": "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt",
        "H25": "outputs/c1_imf_h25_seed42/checkpoint_best.pt",
        "H50": "outputs/c1_imf_h50_seed42/checkpoint_best.pt"}
ARMS = ("B", "H25", "H50")
#: lower-better unless listed here
HIGHER_BETTER = {"raw_corr", "f1_excess"}


def wcsv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _peaks(s):
    return R.detect_rpeaks(np.asarray(s, dtype=np.float64), FS)


def pmap(fn, items, chunk=16):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


def _analyse(args):
    """Per-window structural analysis for one chunk. Fixed coordinates throughout."""
    pred, gt, pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    rows, prof_a, prof_d = [], [], []
    for i in range(len(pred)):
        row = M.region_errors(pred[i], gt[i], pk[i])
        row |= M.qrs_core_morphology(pred[i], gt[i], pk[i])
        row |= M.spectral_metrics(pred[i], gt[i])
        a, d, nb = M.event_profile(pred[i], gt[i], pk[i])
        prof_a.append(a); prof_d.append(d)
        row |= {"raw_corr": float(np.corrcoef(pred[i], gt[i])[0, 1])
                if np.std(pred[i]) > 1e-12 else np.nan,
                "f1": float(rm["rpeak_f1"][i]), "profile_beats": nb}
        ev_n, ev_p = len(pk[i]), len(R.detect_rpeaks(np.asarray(pred[i], dtype=np.float64), FS))
        row["beats_ratio_dev"] = abs(ev_p / max(ev_n, 1) - 1.0)
        rows.append(row)
    return rows, np.asarray(prof_a), np.asarray(prof_d)


def _chance(args):
    gt_pk, pred_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    return [float(np.mean([R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(
        R.match_rpeaks(gt_pk[i], S1.chance_random_phase(len(pred_pk[i]), T_LEN, rng), FS, S1.MATCH_TOL_MS)))[2]
        for _ in range(S1.NULL_DRAWS)])) for i in range(len(gt_pk))]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- population + frozen cohort, verified before any prediction ----------------
    pop, X, Y, SUB, WIDX, SITE = {}, [], [], [], [], []
    cohort = {}
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(SALT, s, len(d["x"]), TAKE); pop[s] = idx
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); WIDX.append(np.asarray(idx)); SITE.append(d["site"][idx])
        cohort[s] = {k: v.tolist() for k, v in atlas_cohort(s, d["site"][idx], d["window_index"][idx]).items()}
    X, Y = np.concatenate(X), np.concatenate(Y)
    SUB, WIDX, SITE = np.concatenate(SUB), np.concatenate(WIDX), np.concatenate(SITE)
    Yd = Y.astype(np.float64)
    gt_pk = pmap(_peaks, list(Yd))
    n_gt = int(sum(len(p) for p in gt_pk))
    ncoh = sum(len(v) for c in cohort.values() for v in c.values())
    assert ncoh == 64, ncoh
    (OUT / "cohort_manifest.json").write_text(json.dumps(
        {"salt": "c2-visual-atlas-v1", "n_windows": ncoh, "strata": {s: {k: len(v) for k, v in c.items()}
         for s, c in cohort.items()}, "indices_into_frozen_subset": cohort}, indent=2))
    print(f"[P] {len(X)} windows, {n_gt} GT beats | atlas cohort {ncoh} | HEAD {head}", flush=True)

    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()

    # ---------------- generate + analyse ----------------
    manifest, per, prof = [], {}, {}
    ch = [(i, min(len(X), i + 64)) for i in range(0, len(X), 64)]
    preds_cohort = {}
    for arm in ARMS:
        p = ROOT / CKPT[arm]
        ck = torch.load(p, map_location="cpu", weights_only=False)
        manifest.append({"arm": arm, "path": CKPT[arm], "kind": "best",
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                         "md5": hashlib.md5(p.read_bytes()).hexdigest(),
                         "bytes": p.stat().st_size, "best_epoch": int(ck.get("epoch", -1)),
                         "c1_arm": ck["args"]["c1_arm"],
                         "selection_metric": float(ck["selection"]["value"])})
        cfg = ck.get("imf_cfg", {})
        net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                         h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
        net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
        with torch.no_grad():
            for n in NFES:
                outs, got = [], set()
                for i in range(0, len(X), BATCH):
                    pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
                    z, k = ER.sample_meanflow_schedule(net, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[n])
                    got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
                assert got == {n}, (arm, n, got)
                pred = np.concatenate(outs).astype(np.float64)
                rows, pa, pd_ = [], [], []
                for r, a_, d_ in pmap(_analyse, [(pred[i:j], Yd[i:j], gt_pk[i:j]) for i, j in ch], chunk=1):
                    rows += r; pa.append(a_); pd_.append(d_)
                pk = pmap(_peaks, list(pred))
                cf = []
                for i, j in ch:
                    cf += _chance((gt_pk[i:j], pk[i:j]))
                for k2, r in enumerate(rows):
                    r["chance_f1"] = cf[k2]; r["f1_excess"] = r["f1"] - cf[k2]
                per[(arm, n)] = rows
                prof[(arm, n)] = (np.concatenate(pa), np.concatenate(pd_))
                # keep only the 64 atlas windows in memory for the visual atlas
                keep = np.concatenate([np.flatnonzero((SUB == s) )[np.asarray(sum(cohort[s].values(), []), dtype=int)]
                                       for s in VAL]) if False else None
                preds_cohort[(arm, n)] = pred
                print(f"[M] {arm} NFE {n}: core sq {np.nanmean([r['qrs_core__a2_sq'] for r in rows]):.5f} "
                      f"bg sq {np.nanmean([r['background__a2_sq'] for r in rows]):.5f} "
                      f"core dabs {np.nanmean([r['qrs_core__a3_dabs'] for r in rows]):.5f} "
                      f"qrsRMSE {np.nanmean([r['qrs_rmse_core'] for r in rows]):.5f} "
                      f"curv {np.nanmean([r['qrs_curvature_err'] for r in rows]):.5f}", flush=True)
        del net; torch.cuda.empty_cache()
    wcsv(OUT / "checkpoint_manifest.csv", manifest)
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(OUT / "_cohort_preds.npz",
                        **{f"{a}_{n}": preds_cohort[(a, n)][np.concatenate(
                            [np.flatnonzero(SUB == s)[np.asarray(sum(cohort[s].values(), []), dtype=int)] for s in VAL])]
                           for a in ARMS for n in NFES},
                        gt=Yd[np.concatenate([np.flatnonzero(SUB == s)[np.asarray(sum(cohort[s].values(), []), dtype=int)] for s in VAL])],
                        ppg=X[np.concatenate([np.flatnonzero(SUB == s)[np.asarray(sum(cohort[s].values(), []), dtype=int)] for s in VAL])].astype(np.float64))
    del preds_cohort

    KEYS = [k for k in per[("B", 2)][0] if isinstance(per[("B", 2)][0][k], (int, float))]
    win = []
    for (arm, n), rows in per.items():
        for i, r in enumerate(rows):
            win.append({"arm": arm, "nfe": n, "subject": SUB[i], "window_index": int(WIDX[i]),
                        "site": SITE[i], **{k: r.get(k) for k in KEYS}})
    wcsv(OUT / "metrics_window.csv", win)

    reg = []
    for (arm, n), rows in per.items():
        g = lambda k: np.asarray([r[k] for r in rows], float)  # noqa: E731
        reg.append({"arm": arm, "nfe": n, **{k: S1.macro(g(k), SUB) for k in KEYS}})
    wcsv(OUT / "region_metrics.csv", reg)

    # ---------------- event error profiles ----------------
    prow = []
    for (arm, n), (pa, pd_) in prof.items():
        for t_i, tau in enumerate(M.PROFILE_TAU):
            prow.append({"arm": arm, "nfe": n, "tau_samples": int(tau),
                         "tau_ms": float(tau) / FS * 1000.0,
                         "abs_err": S1.macro(pa[:, t_i], SUB), "deriv_err": S1.macro(pd_[:, t_i], SUB)})
    wcsv(OUT / "event_error_profiles.csv", prow)

    # ---------------- paired bootstrap ----------------
    boot = []
    def cmp(lo_key, hi_key, label):
        for k in KEYS:
            a_ = np.asarray([r[k] for r in per[lo_key]], float)
            b_ = np.asarray([r[k] for r in per[hi_key]], float)
            orient = "higher_better" if k in HIGHER_BETTER else "lower_better"
            boot.append({"comparison": label, "metric": k, **paired_subject_bootstrap(a_, b_, SUB, orient)})
    for n in NFES:
        cmp(("H25", n), ("H50", n), f"H50-vs-H25@NFE{n}")
        cmp(("B", n), ("H50", n), f"H50-vs-B@NFE{n}")
        cmp(("B", n), ("H25", n), f"H25-vs-B@NFE{n}")
    wcsv(OUT / "paired_bootstrap.csv", boot)

    # ---------------- localisation contrast L (prereg s7) ----------------
    loc = []
    for fam, ck_, bk_ in (("waveform_sq", "qrs_core__a2_sq", "background__a2_sq"),
                          ("derivative_abs", "qrs_core__a3_dabs", "background__a3_dabs")):
        c25, c50 = np.asarray([r[ck_] for r in per[("H25", 2)]], float), np.asarray([r[ck_] for r in per[("H50", 2)]], float)
        b25, b50 = np.asarray([r[bk_] for r in per[("H25", 2)]], float), np.asarray([r[bk_] for r in per[("H50", 2)]], float)
        uniq = sorted(set(SUB.tolist())); idx = {u: np.flatnonzero(SUB == u) for u in uniq}
        rng = np.random.default_rng(S1.BOOT_SEED)
        def L_of(picks):
            def mac(a):
                return float(np.mean([np.nanmean(a[picks[u]]) for u in uniq]))
            rc = (mac(c25) - mac(c50)) / (mac(c25) + 1e-20)
            rb = (mac(b25) - mac(b50)) / (mac(b25) + 1e-20)
            return rc - rb, rc, rb
        pt, rc0, rb0 = L_of(idx)
        draws = np.array([L_of({u: rng.choice(idx[u], idx[u].size, replace=True) for u in uniq})[0]
                          for _ in range(S1.BOOT_N)])
        lo, hi = np.percentile(draws, [2.5, 97.5])
        loc.append({"family": fam, "R_core": rc0, "R_background": rb0, "L": pt,
                    "lo": float(lo), "hi": float(hi),
                    "verdict": "improves" if lo > 0 else ("worsens" if hi < 0 else "unresolved")})
        print(f"[L] {fam}: R_core {rc0:+.4f} R_bg {rb0:+.4f} L {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    wcsv(OUT / "localization.csv", loc)

    # ---------------- NFE interaction ----------------
    inter = []
    for k in ("qrs_core__a2_sq", "qrs_core__a3_dabs", "qrs_energy_dev", "qrs_ptp_dev", "background__a2_sq"):
        orient = "lower_better"
        e2 = paired_subject_bootstrap(np.asarray([r[k] for r in per[("B", 2)]], float),
                                      np.asarray([r[k] for r in per[("H50", 2)]], float), SUB, orient)
        e4 = paired_subject_bootstrap(np.asarray([r[k] for r in per[("B", 4)]], float),
                                      np.asarray([r[k] for r in per[("H50", 4)]], float), SUB, orient)
        d2v = (np.asarray([r[k] for r in per[("B", 2)]], float) - np.asarray([r[k] for r in per[("H50", 2)]], float)) - \
              (np.asarray([r[k] for r in per[("B", 4)]], float) - np.asarray([r[k] for r in per[("H50", 4)]], float))
        dd = paired_subject_bootstrap(np.zeros_like(d2v), d2v, SUB, "higher_better")
        inter.append({"metric": k, "E2": e2["point"], "E4": e4["point"], "D": dd["point"],
                      "D_lo": dd["lo"], "D_hi": dd["hi"], "D_verdict": dd["verdict"]})
        print(f"[D] {k:22s} E2 {e2['point']:+.5f} E4 {e4['point']:+.5f} D {dd['point']:+.5f} "
              f"[{dd['lo']:+.5f},{dd['hi']:+.5f}] {dd['verdict']}", flush=True)
    wcsv(OUT / "nfe_interaction.csv", inter)

    # ---------------- site-wise ----------------
    site_rows = []
    SKEYS = ["qrs_core__a2_sq", "background__a2_sq", "qrs_core__a3_dabs", "qrs_energy_dev",
             "qrs_ptp_dev", "raw_corr", "f1_excess", "beats_ratio_dev"]
    for site in SITES:
        m = SITE == site
        for arm in ARMS:
            rows = per[(arm, 2)]
            r = {"site": site, "arm": arm, "n_windows": int(m.sum())}
            for k in SKEYS:
                v = np.asarray([x[k] for x in rows], float)[m]
                r[k] = float(np.mean([np.nanmean(v[SUB[m] == s]) for s in VAL]))
            site_rows.append(r)
    wcsv(OUT / "site_metrics.csv", site_rows)

    wcsv(OUT / "spectral_metrics.csv", [{"arm": a, "nfe": n,
         **{k: S1.macro(np.asarray([r[k] for r in per[(a, n)]], float), SUB) for k in KEYS if k[:2] in ("F1", "F2", "F3", "F4")}}
        for a in ARMS for n in NFES])

    (OUT / "provenance.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "prereg": "959eb60",
        "c2_deferred": True, "c2_weight_updates": 0, "training": False,
        "checkpoints": {a: CKPT[a] for a in ARMS}, "source_seed": SRC_SEED,
        "source_bank_sha256": src_hash, "population": {s: int(pop[s].size) for s in VAL},
        "n_gt_beats": n_gt, "test_subjects_loaded": [], "oracle_metrics_used": False,
        "predictions_translated": False, "atlas_cohort_windows": ncoh}, indent=2))
    print("\n[done] M1 analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
