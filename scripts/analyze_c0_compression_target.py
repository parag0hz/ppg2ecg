"""C0 — iMeanFlow compression-target confirmation.

Frozen protocol: docs/C0_IMF_COMPRESSION_TARGET_PREREGISTRATION.md (commit 5df1a33), pushed before any
number here existed.

NO TRAINING. NO TEST SUBJECTS. NO EXTERNAL BASELINE. NO NEW METHOD.
Frozen-checkpoint forward inference only; no checkpoint written.
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

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.evaluation import alignment_diagnostics as AD  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation import s1_audit as S1  # noqa: E402
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics  # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/c0_imf_compression_target"
FIG = OUT / "figures"
FS, T_LEN = 128, 1024
VAL = ("an0", "k2s")
SALT, TAKE = "x4-event-nfe-v2", 1024
NFES = (1, 2, 4, 8)                       # frozen grid; nothing may be added post-hoc
SRC_SEED = 0
CKPT = "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
BATCH, WORKERS = 64, 12

#: the six counted primary metrics, with their frozen per-window aggregator and orientation
PRIMARY = [
    ("raw_corr",      "raw_corr",             np.nanmean,   "higher_better", False),
    ("qrs_e_dev",     "raw_qrs_energy_ratio", np.nanmedian, "lower_better",  True),
    ("slope_dev",     "raw_slope_ratio",      np.nanmedian, "lower_better",  True),
    ("p2p_dev",       "raw_p2p_ratio",        np.nanmedian, "lower_better",  True),
    ("raw_qrs_rmse",  "raw_qrs_rmse",         np.nanmean,   "lower_better",  False),
    ("raw_rmse",      "raw_rmse",             np.nanmean,   "lower_better",  False),
]
RATIO_RAW = {"qrs_e_dev": "raw_qrs_energy_ratio", "slope_dev": "raw_slope_ratio", "p2p_dev": "raw_p2p_ratio"}


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _peaks(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), FS)


def pmap(fn, items, chunk=16):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


def _score_chunk(args):
    """Per-window primary (GT-fixed, oracle-free) and secondary metrics for one chunk."""
    pred, gt, gt_pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    max_shift = int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS))
    rows = []
    for i in range(len(pred)):
        bl = AD.beat_level_analysis(pred[i], gt[i], gt_pk[i], FS, max_shift)   # only raw_* is read below
        ev = AD.event_timing(gt[i], pred[i], FS, tol_ms=S1.MATCH_TOL_MS)
        m, _, _ = R.match_rpeaks(ev["ref_rpeaks"], ev["pred_rpeaks"], FS, S1.MATCH_TOL_MS)
        n_ref = max(ev["n_ref"], 1)
        row = {"n_valid_gt_beats": int(bl["n_beats"]) if bl["n_beats"] else 0}
        for name, key, agg, _o, is_dev in PRIMARY:
            src = RATIO_RAW.get(name, key)
            v = float(agg(bl[src])) if bl["n_beats"] else np.nan
            row[name] = (abs(v - 1.0) if is_dev else v)
            if is_dev:
                row[src] = v
        row |= {
            "hf_ratio": float(hf_energy_ratio(pred[i][None])[0]),
            "hf_gt": float(hf_energy_ratio(gt[i][None])[0]),
            "f1": float(rm["rpeak_f1"][i]), "precision": float(rm["rpeak_precision"][i]),
            "recall": float(rm["rpeak_recall"][i]),
            "beats_ratio": ev["n_pred"] / n_ref, "missing": ev["n_missing"] / n_ref,
            "spurious": ev["n_spurious"] / n_ref, "n_matched": int(len(m)),
            "morph": float(rm["morph_corr"][i]), "rr_mae_ms": float(rm["rr_mae_ms"][i]),
        }
        rows.append(row)
    return rows


def _chance_chunk(args):
    gt_pk, pred_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    out = []
    for i in range(len(gt_pk)):
        n = len(pred_pk[i])
        f = [R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(
                R.match_rpeaks(gt_pk[i], S1.chance_random_phase(n, T_LEN, rng), FS, S1.MATCH_TOL_MS)))[2]
             for _ in range(S1.NULL_DRAWS)]
        out.append(float(np.mean(f)))
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_md5 = hashlib.md5((ROOT / CKPT).read_bytes()).hexdigest()
    print(f"[prov] HEAD {head} | device {dev} | ckpt md5 {ck_md5}", flush=True)

    pop, X, Y, SUB, W = {}, [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(SALT, s, len(d["x"]), TAKE)
        pop[s] = idx
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); W.append(np.asarray(idx))
    X, Y, SUB, W = np.concatenate(X), np.concatenate(Y), np.concatenate(SUB), np.concatenate(W)
    Yd = Y.astype(np.float64)
    print(f"[P] {len(X)} windows ({', '.join(f'{s}:{pop[s].size}' for s in VAL)})", flush=True)
    gt_pk = pmap(_peaks, list(Yd))
    n_gt = int(sum(len(p) for p in gt_pk))
    print(f"[P] GT beats {n_gt}", flush=True)

    ckd = torch.load(ROOT / CKPT, map_location="cpu", weights_only=False)
    cfg = ckd.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ckd["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ckd["state_dict"]); net.requires_grad_(False)

    # ONE source bank, reused for every NFE — the pairing requirement
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()
    print(f"[S] source bank seed {SRC_SEED} sha256 {src_hash[:32]}", flush=True)

    preds, nfe_seen = {}, {}
    with torch.no_grad():
        for n in NFES:
            outs, got = [], set()
            for i in range(0, len(X), BATCH):
                p = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
                z, k = ER.sample_meanflow_schedule(net, p, e0[i:i + BATCH].to(dev), ER.UNIFORM[n])
                got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
            assert got == {n}, f"NFE parity violated for {n}: realised {got}"
            nfe_seen[n] = sorted(got)[0]
            preds[n] = np.concatenate(outs).astype(np.float64)
            print(f"[A] NFE {n}: generated, realised steps {nfe_seen[n]}", flush=True)
    del net; torch.cuda.empty_cache()

    ch = [(i, min(len(X), i + 64)) for i in range(0, len(X), 64)]
    per, primary, secondary = {}, [], []
    for n in NFES:
        rows = []
        for r in pmap(_score_chunk, [(preds[n][a:b], Yd[a:b], gt_pk[a:b]) for a, b in ch], chunk=1):
            rows += r
        pk = pmap(_peaks, list(preds[n]))
        cf = []
        for a, b in ch:
            cf += _chance_chunk((gt_pk[a:b], pk[a:b]))
        for k, row in enumerate(rows):
            row["chance_f1"] = cf[k]
            row["f1_excess"] = row["f1"] - cf[k]
        per[n] = rows
        g = lambda k: np.asarray([x[k] for x in rows], float)  # noqa: E731
        cov = float(np.sum(g("n_matched")) / max(np.sum(g("n_valid_gt_beats")), 1))
        prow = {"nfe": n, "realised_steps": nfe_seen[n]}
        for name, _key, _agg, _o, is_dev in PRIMARY:
            prow[name] = S1.macro(g(name), SUB)
            if is_dev:
                prow[RATIO_RAW[name]] = S1.macro(g(RATIO_RAW[name]), SUB)
        prow |= {"hf_ratio": S1.macro(g("hf_ratio"), SUB), "hf_gt": S1.macro(g("hf_gt"), SUB)}
        primary.append(prow)
        secondary.append({"nfe": n, "f1": S1.macro(g("f1"), SUB), "chance_f1": S1.macro(g("chance_f1"), SUB),
                          "f1_excess": S1.macro(g("f1_excess"), SUB),
                          "precision": S1.macro(g("precision"), SUB), "recall": S1.macro(g("recall"), SUB),
                          "beats_ratio": S1.macro(g("beats_ratio"), SUB),
                          "beats_ratio_dev": S1.macro(np.abs(g("beats_ratio") - 1.0), SUB),
                          "missing": S1.macro(g("missing"), SUB), "spurious": S1.macro(g("spurious"), SUB),
                          "matched_coverage": cov, "matched_morph": S1.macro(g("morph"), SUB),
                          "zero_contrib_window_frac": float(np.mean(~np.isfinite(g("morph")))),
                          "matched_rr_mae_ms": S1.macro(g("rr_mae_ms"), SUB)})
        print(f"[M] NFE {n}: corr {prow['raw_corr']:.4f} qrsEdev {prow['qrs_e_dev']:.4f} "
              f"slopedev {prow['slope_dev']:.4f} p2pdev {prow['p2p_dev']:.4f} "
              f"qrsRMSE {prow['raw_qrs_rmse']:.4f} RMSE {prow['raw_rmse']:.4f} | "
              f"F1 {secondary[-1]['f1']:.4f} excess {secondary[-1]['f1_excess']:+.4f} "
              f"beats {secondary[-1]['beats_ratio']:.4f}", flush=True)
    write_csv(OUT / "primary_metrics.csv", primary)
    write_csv(OUT / "secondary_metrics.csv", secondary)

    # ---------------- paired bootstrap ----------------
    boot = []
    for lo_n, hi_n in ((1, 2), (2, 4), (4, 8)):
        for name, _k, _a, orient, _d in PRIMARY:
            a = np.asarray([r[name] for r in per[lo_n]], float)
            b = np.asarray([r[name] for r in per[hi_n]], float)
            res = paired_subject_bootstrap(a, b, SUB, orient)
            boot.append({"comparison": f"{lo_n}->{hi_n}", "metric": name, "family": "primary", **res})
        for name, orient in (("f1_excess", "higher_better"), ("beats_ratio_dev", "lower_better")):
            a = np.asarray([r["f1_excess"] if name == "f1_excess" else abs(r["beats_ratio"] - 1.0) for r in per[lo_n]], float)
            b = np.asarray([r["f1_excess"] if name == "f1_excess" else abs(r["beats_ratio"] - 1.0) for r in per[hi_n]], float)
            res = paired_subject_bootstrap(a, b, SUB, orient)
            boot.append({"comparison": f"{lo_n}->{hi_n}", "metric": name, "family": "secondary", **res})
        v = {r["metric"]: r["verdict"] for r in boot if r["comparison"] == f"{lo_n}->{hi_n}"}
        prim = [m for m, _k, _a, _o, _d in [(p[0],) + p[1:] for p in PRIMARY]]
        print(f"[B] {lo_n}->{hi_n}: improved {sum(v[m]=='improves' for m in prim)} "
              f"worsened {sum(v[m]=='worsens' for m in prim)} "
              f"unresolved {sum(v[m]=='unresolved' for m in prim)} | "
              f"F1excess {v['f1_excess']} beatsdev {v['beats_ratio_dev']}", flush=True)
    write_csv(OUT / "paired_bootstrap.csv", boot)

    # ---------------- frozen decision rule ----------------
    prim_names = [p[0] for p in PRIMARY]
    def verd(cmp_, m):
        return next(r["verdict"] for r in boot if r["comparison"] == cmp_ and r["metric"] == m)
    a_imp = [m for m in prim_names if verd("2->4", m) == "improves"]
    a_wor = [m for m in prim_names if verd("2->4", m) == "worsens"]
    a_collapse = verd("2->4", "f1_excess") == "worsens"
    gate_a = (len(a_imp) >= 2) and (len(a_wor) == 0) and (not a_collapse)
    b_imp = [m for m in prim_names if verd("4->8", m) == "improves"]
    b_wor = [m for m in prim_names if verd("4->8", m) == "worsens"]
    b_degr = (verd("4->8", "f1_excess") == "worsens") or (verd("4->8", "beats_ratio_dev") == "worsens")
    gate_b = gate_a and (len(b_imp) >= 2) and (len(b_wor) == 0) and (not b_degr)
    decision = ("COMPRESSION PREMISE NOT ESTABLISHED / INCONCLUSIVE" if not gate_a
                else ("COMPRESSION TARGET = NFE 8" if gate_b else "COMPRESSION TARGET = NFE 4"))
    dec = {"decision": decision, "gate_a_pass": bool(gate_a), "gate_b_pass": bool(gate_b) if gate_a else None,
           "gate_a": {"improved": a_imp, "worsened": a_wor, "n_improved": len(a_imp), "n_worsened": len(a_wor),
                      "event_f1_collapse": bool(a_collapse)},
           "gate_b": {"improved": b_imp, "worsened": b_wor, "n_improved": len(b_imp), "n_worsened": len(b_wor),
                      "event_or_beats_degrade": bool(b_degr)} if gate_a else "NOT REACHED",
           "one_to_two": {m: verd("1->2", m) for m in prim_names},
           "primary_metrics_counted": prim_names, "prereg": "5df1a33"}
    (OUT / "decision.json").write_text(json.dumps(dec, indent=2))
    (OUT / "provenance.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "prereg": "5df1a33",
        "checkpoint": CKPT, "checkpoint_md5": ck_md5, "source_seed": SRC_SEED,
        "source_bank_sha256": src_hash, "same_source_across_nfe": True,
        "nfe_grid": list(NFES), "realised_steps": nfe_seen,
        "population": {s: int(pop[s].size) for s in VAL}, "n_gt_beats": n_gt,
        "test_subjects_loaded": [], "training": False, "external_baseline": False,
        "oracle_metrics_used_for_decision": False,
        "note": "seed-0 values; NOT comparable to the recorded 4-seed-pooled X4-0 table"}, indent=2))

    # ---------------- figures ----------------
    labels = {"raw_corr": "raw_corr (higher)", "qrs_e_dev": "|QRS-E - 1| (lower)",
              "slope_dev": "|slope - 1| (lower)", "p2p_dev": "|p2p - 1| (lower)",
              "raw_qrs_rmse": "raw QRS RMSE (lower)", "raw_rmse": "raw RMSE (lower)"}
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, m in zip(axes.ravel(), prim_names):
        ax.plot(NFES, [p[m] for p in primary], "o-", lw=1.8)
        ax.set_xscale("log", base=2); ax.set_xticks(NFES); ax.set_xticklabels(NFES)
        ax.set_title(labels[m], fontsize=10); ax.set_xlabel("NFE"); ax.grid(alpha=0.3)
    axes.ravel()[6].plot(NFES, [s["f1_excess"] for s in secondary], "s-", color="tab:orange", lw=1.8)
    axes.ravel()[6].set_xscale("log", base=2); axes.ravel()[6].set_xticks(NFES)
    axes.ravel()[6].set_xticklabels(NFES); axes.ravel()[6].grid(alpha=0.3)
    axes.ravel()[6].set_title("F1 excess over chance (secondary)", fontsize=10); axes.ravel()[6].set_xlabel("NFE")
    axes.ravel()[7].plot(NFES, [p["hf_ratio"] for p in primary], "^-", color="tab:green", lw=1.8, label="pred")
    axes.ravel()[7].axhline(primary[0]["hf_gt"], color="k", ls="--", lw=1, label="GT")
    axes.ravel()[7].set_xscale("log", base=2); axes.ravel()[7].set_xticks(NFES)
    axes.ravel()[7].set_xticklabels(NFES); axes.ravel()[7].legend(fontsize=8); axes.ravel()[7].grid(alpha=0.3)
    axes.ravel()[7].set_title("whole-window HF ratio (no direction claimed)", fontsize=10)
    axes.ravel()[7].set_xlabel("NFE")
    fig.suptitle("C0 — frozen iMeanFlow, oracle-free GT-fixed structure vs NFE (2048 dev windows, source seed 0)")
    fig.tight_layout(); fig.savefig(FIG / "c0_nfe_oracle_free_metrics.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for p, s in zip(primary, secondary):
        ax.scatter(s["f1_excess"], p["qrs_e_dev"], s=120, zorder=3)
        ax.annotate(f"NFE {p['nfe']}", (s["f1_excess"], p["qrs_e_dev"]), fontsize=10,
                    textcoords="offset points", xytext=(9, -3))
    ax.plot([s["f1_excess"] for s in secondary], [p["qrs_e_dev"] for p in primary], "-", alpha=0.4, color="gray")
    ax.set_xlabel("event F1 excess over chance (secondary)")
    ax.set_ylabel("|raw QRS energy ratio - 1|  (lower is better)")
    ax.invert_yaxis(); ax.grid(alpha=0.3)
    ax.set_title("C0 — event vs GT-fixed QRS structure (one projection only)")
    fig.tight_layout(); fig.savefig(FIG / "c0_nfe_event_vs_structure.png", dpi=115); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, cmp_ in zip(axes, ("1->2", "2->4", "4->8")):
        rows = [r for r in boot if r["comparison"] == cmp_ and r["family"] == "primary"]
        y = np.arange(len(rows))
        ax.errorbar([r["point"] for r in rows], y,
                    xerr=[[r["point"] - r["lo"] for r in rows], [r["hi"] - r["point"] for r in rows]],
                    fmt="o", capsize=4)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y); ax.set_yticklabels([labels[r["metric"]] for r in rows], fontsize=9)
        ax.set_title(f"NFE {cmp_}   (positive = later is better)", fontsize=10); ax.grid(alpha=0.3)
    fig.suptitle("C0 — paired subject-stratified bootstrap, 2000 resamples, seed 20260901")
    fig.tight_layout(); fig.savefig(FIG / "c0_paired_improvement.png", dpi=115); plt.close(fig)

    print(f"\n[GATE A] improved {len(a_imp)} {a_imp} | worsened {len(a_wor)} {a_wor} | "
          f"collapse {a_collapse} -> {'PASS' if gate_a else 'FAIL'}", flush=True)
    if gate_a:
        print(f"[GATE B] improved {len(b_imp)} {b_imp} | worsened {len(b_wor)} {b_wor} | "
              f"degrade {b_degr} -> {'PASS' if gate_b else 'FAIL'}", flush=True)
    print(f"[DECISION] {decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
