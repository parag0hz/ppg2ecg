"""C1 evaluation: NFE 2 / 4 on the frozen C0 development population, for any set of C1 checkpoints.

Frozen protocol: docs/C1_INTERVAL_EXPOSURE_CONTROL_PREREGISTRATION.md (b32c952).
Reuses the C0 metric path verbatim: oracle-free GT-fixed raw_* primaries, S1 chance floor, C0 paired
bootstrap. NO TRAINING. NO TEST SUBJECTS. No oracle metric enters any decision.

Usage:
  python scripts/eval_c1_arms.py --arms B                # Stage 1 (baseline replay gate)
  python scripts/eval_c1_arms.py --arms B H25 H50        # Stage 2
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import alignment_diagnostics as AD
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/c1_interval_exposure"
FS, T_LEN, BATCH, WORKERS = 128, 1024, 64, 12
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
NFES, SRC_SEED = (2, 4), 0
CKPT = {"B": "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt",
        "H25": "outputs/c1_imf_h25_seed42/checkpoint_best.pt",
        "H50": "outputs/c1_imf_h50_seed42/checkpoint_best.pt"}
#: M1-M4 are counted for gap closure; M5/M6 are reported and count only as "must not worsen"
PRIMARY = [("M1_qrs_e_dev", "raw_qrs_energy_ratio", np.nanmedian, "lower_better", True),
           ("M2_p2p_dev",   "raw_p2p_ratio",        np.nanmedian, "lower_better", True),
           ("M3_qrs_rmse",  "raw_qrs_rmse",         np.nanmean,   "lower_better", False),
           ("M4_rmse",      "raw_rmse",             np.nanmean,   "lower_better", False),
           ("M5_raw_corr",  "raw_corr",             np.nanmean,   "higher_better", False),
           ("M6_slope_dev", "raw_slope_ratio",      np.nanmedian, "lower_better", True)]
GAP_METRICS = ["M1_qrs_e_dev", "M2_p2p_dev", "M3_qrs_rmse", "M4_rmse"]


def write_csv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _peaks(s):
    return R.detect_rpeaks(np.asarray(s, dtype=np.float64), FS)


def pmap(fn, items, chunk=16):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


def _score_chunk(args):
    pred, gt, gt_pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    ms = int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS))
    rows = []
    for i in range(len(pred)):
        bl = AD.beat_level_analysis(pred[i], gt[i], gt_pk[i], FS, ms)     # raw_* only
        ev = AD.event_timing(gt[i], pred[i], FS, tol_ms=S1.MATCH_TOL_MS)
        m, _, _ = R.match_rpeaks(ev["ref_rpeaks"], ev["pred_rpeaks"], FS, S1.MATCH_TOL_MS)
        n_ref = max(ev["n_ref"], 1)
        row = {"n_valid_gt_beats": int(bl["n_beats"]) if bl["n_beats"] else 0}
        for name, key, agg, _o, is_dev in PRIMARY:
            v = float(agg(bl[key])) if bl["n_beats"] else np.nan
            row[name] = abs(v - 1.0) if is_dev else v
            row[f"raw::{key}"] = v
        row |= {"f1": float(rm["rpeak_f1"][i]), "beats_ratio": ev["n_pred"] / n_ref,
                "beats_ratio_dev": abs(ev["n_pred"] / n_ref - 1.0),
                "n_matched": int(len(m)), "morph": float(rm["morph_corr"][i]),
                "hf_ratio": float(hf_energy_ratio(pred[i][None])[0])}
        rows.append(row)
    return rows


def _chance_chunk(args):
    gt_pk, pred_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    return [float(np.mean([R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(
        R.match_rpeaks(gt_pk[i], S1.chance_random_phase(len(pred_pk[i]), T_LEN, rng), FS, S1.MATCH_TOL_MS)))[2]
        for _ in range(S1.NULL_DRAWS)])) for i in range(len(gt_pk))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B"], choices=list(CKPT))
    ap.add_argument("--tag", default="stage1")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pop, X, Y, SUB = {}, [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(SALT, s, len(d["x"]), TAKE); pop[s] = idx
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float32))
        SUB.append(np.full(len(idx), s))
    X, Y, SUB = np.concatenate(X), np.concatenate(Y), np.concatenate(SUB)
    Yd = Y.astype(np.float64)
    gt_pk = pmap(_peaks, list(Yd))
    print(f"[P] {len(X)} windows, {sum(len(p) for p in gt_pk)} GT beats | HEAD {head}", flush=True)

    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()
    ch = [(i, min(len(X), i + 64)) for i in range(0, len(X), 64)]
    per, metrics, ck_md5 = {}, [], {}

    for arm in a.arms:
        p = ROOT / CKPT[arm]
        if not p.exists():
            print(f"[!] {arm}: checkpoint missing at {p} -- skipped", flush=True); continue
        ck_md5[arm] = hashlib.md5(p.read_bytes()).hexdigest()
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("imf_cfg", {})
        net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                         h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
        net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
        print(f"[A] {arm}: {CKPT[arm]} md5 {ck_md5[arm][:16]} best_epoch {ck.get('epoch')}", flush=True)
        with torch.no_grad():
            for n in NFES:
                outs, got = [], set()
                for i in range(0, len(X), BATCH):
                    pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
                    z, k = ER.sample_meanflow_schedule(net, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[n])
                    got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
                assert got == {n}, f"NFE parity violated for {arm}/{n}: {got}"
                pred = np.concatenate(outs).astype(np.float64)
                rows = []
                for r in pmap(_score_chunk, [(pred[i:j], Yd[i:j], gt_pk[i:j]) for i, j in ch], chunk=1):
                    rows += r
                pk = pmap(_peaks, list(pred))
                cf = []
                for i, j in ch:
                    cf += _chance_chunk((gt_pk[i:j], pk[i:j]))
                for k2, row in enumerate(rows):
                    row["chance_f1"] = cf[k2]; row["f1_excess"] = row["f1"] - cf[k2]
                per[(arm, n)] = rows
                g = lambda k: np.asarray([x[k] for x in rows], float)  # noqa: E731
                mrow = {"arm": arm, "nfe": n}
                for name, _k, _ag, _o, _d in PRIMARY:
                    mrow[name] = S1.macro(g(name), SUB)
                mrow |= {"f1": S1.macro(g("f1"), SUB), "chance_f1": S1.macro(g("chance_f1"), SUB),
                         "f1_excess": S1.macro(g("f1_excess"), SUB),
                         "beats_ratio": S1.macro(g("beats_ratio"), SUB),
                         "beats_ratio_dev": S1.macro(g("beats_ratio_dev"), SUB),
                         "matched_coverage": float(np.sum(g("n_matched")) / max(np.sum(g("n_valid_gt_beats")), 1)),
                         "matched_morph": S1.macro(g("morph"), SUB), "hf_ratio": S1.macro(g("hf_ratio"), SUB)}
                metrics.append(mrow)
                print(f"[M] {arm} NFE {n}: M1 {mrow['M1_qrs_e_dev']:.4f} M2 {mrow['M2_p2p_dev']:.4f} "
                      f"M3 {mrow['M3_qrs_rmse']:.4f} M4 {mrow['M4_rmse']:.4f} | M5 {mrow['M5_raw_corr']:.4f} "
                      f"M6 {mrow['M6_slope_dev']:.4f} | F1ex {mrow['f1_excess']:+.4f} "
                      f"beatsdev {mrow['beats_ratio_dev']:.4f}", flush=True)
        del net; torch.cuda.empty_cache()
    write_csv(OUT / f"{a.tag}_metrics.csv", metrics)

    boot = []
    def add(cmp_, lo_key, hi_key, names):
        for name, _k, _ag, orient, _d in PRIMARY:
            if name not in names:
                continue
            x = np.asarray([r[name] for r in per[lo_key]], float)
            y = np.asarray([r[name] for r in per[hi_key]], float)
            boot.append({"comparison": cmp_, "metric": name,
                         **paired_subject_bootstrap(x, y, SUB, orient)})
        for name, orient in (("f1_excess", "higher_better"), ("beats_ratio_dev", "lower_better")):
            x = np.asarray([r[name] for r in per[lo_key]], float)
            y = np.asarray([r[name] for r in per[hi_key]], float)
            boot.append({"comparison": cmp_, "metric": name,
                         **paired_subject_bootstrap(x, y, SUB, orient)})

    allm = [p[0] for p in PRIMARY]
    if ("B", 2) in per and ("B", 4) in per:
        add("B:2->4", ("B", 2), ("B", 4), allm)
    for arm in ("H25", "H50"):
        if (arm, 2) in per and ("B", 2) in per:
            add(f"{arm}-vs-B@NFE2", ("B", 2), (arm, 2), allm)
        if (arm, 4) in per and ("B", 4) in per:
            add(f"{arm}-vs-B@NFE4", ("B", 4), (arm, 4), allm)
    write_csv(OUT / f"{a.tag}_paired.csv", boot)
    for r in boot:
        print(f"[B] {r['comparison']:16s} {r['metric']:16s} {r['point']:+.5f} "
              f"[{r['lo']:+.5f}, {r['hi']:+.5f}] {r['verdict']}", flush=True)

    result = {"tag": a.tag, "head": head, "utc": datetime.now(timezone.utc).isoformat(),
              "prereg": "b32c952", "arms_evaluated": [k for k in a.arms if (k, NFES[0]) in per],
              "checkpoint_md5": ck_md5, "source_seed": SRC_SEED, "source_bank_sha256": src_hash,
              "nfe_grid": list(NFES), "population": {s: int(pop[s].size) for s in VAL},
              "test_subjects_loaded": [], "training": False, "oracle_metrics_used": False}

    def col(key, m):
        return np.asarray([r[m] for r in per[key]], float)

    def macro_diff(a_arr, b_arr):
        """equal-subject-weight mean of (a - b); positive = a larger"""
        return float(np.mean([np.nanmean(a_arr[SUB == s] - b_arr[SUB == s]) for s in VAL]))

    if a.tag == "stage2" and all((k, n) in per for k in ("B", "H25", "H50") for n in NFES):
        # ---- gap closure (prereg s10). All of M1-M4 are lower-better, so Q = -dev and
        #      G = Q(B,4) - Q(B,2) = dev(B,2) - dev(B,4);  I = dev(B,2) - dev(X,2).  Unclipped. ----
        closure = []
        for m in GAP_METRICS:
            g = macro_diff(col(("B", 2), m), col(("B", 4), m))
            row = {"metric": m, "G_replay": g}
            for arm in ("H25", "H50"):
                i_m = macro_diff(col(("B", 2), m), col((arm, 2), m))
                row[f"I_{arm}"] = i_m
                row[f"C_{arm}"] = (i_m / g) if g > 0 else float("nan")
            closure.append(row)
            print(f"[G] {m:16s} G={row['G_replay']:+.5f} | H25 I={row['I_H25']:+.5f} C={row['C_H25']:+.3f}"
                  f" | H50 I={row['I_H50']:+.5f} C={row['C_H50']:+.3f}", flush=True)
        write_csv(OUT / "gap_closure.csv", closure)

        # ---- specificity (s12): paired difference-of-improvement H50 - H25 at NFE 2 ----
        spec = []
        for m in GAP_METRICS:
            b2 = col(("B", 2), m)
            imp25 = b2 - col(("H25", 2), m)
            imp50 = b2 - col(("H50", 2), m)
            r = paired_subject_bootstrap(imp25, imp50, SUB, "higher_better")
            spec.append({"metric": m, **r})
            print(f"[S] {m:16s} H50-H25 {r['point']:+.5f} [{r['lo']:+.5f}, {r['hi']:+.5f}] {r['verdict']}", flush=True)
        write_csv(OUT / "specificity.csv", spec)

        def V(cmp_, m):
            return next(x["verdict"] for x in boot if x["comparison"] == cmp_ and x["metric"] == m)

        allm = [q[0] for q in PRIMARY]

        def gate(arm):
            imp = [m for m in GAP_METRICS if V(f"{arm}-vs-B@NFE2", m) == "improves"]
            wor = [m for m in allm if V(f"{arm}-vs-B@NFE2", m) == "worsens"]
            cl = [x["metric"] for x in closure if x[f"C_{arm}"] >= 0.50]
            f1w = V(f"{arm}-vs-B@NFE2", "f1_excess") == "worsens"
            bw = V(f"{arm}-vs-B@NFE2", "beats_ratio_dev") == "worsens"
            return {"arm": arm, "improved": imp, "worsened": wor, "closure_ge_0.5": cl,
                    "f1_excess_worsens": f1w, "beats_ratio_dev_worsens": bw,
                    "pass": bool(len(imp) >= 3 and not wor and len(cl) >= 2 and not f1w and not bw)}

        g50, g25 = gate("H50"), gate("H25")
        spec_ok = [x["metric"] for x in spec if x["verdict"] == "improves"]
        nfe4 = {arm: [m for m in allm if V(f"{arm}-vs-B@NFE4", m) == "worsens"] for arm in ("H25", "H50")}
        degr = {arm: bool(len(v) >= 2) for arm, v in nfe4.items()}

        if not g50["pass"]:
            verdict = "INTERVAL-EXPOSURE HYPOTHESIS NOT SUPPORTED"
        elif len(spec_ok) >= 2 and not g25["pass"]:
            verdict = "TARGET h=0.5 EXPOSURE SUPPORTED"
        else:
            verdict = "GENERIC POSITIVE-h REWEIGHTING EFFECT"
        result |= {"verdict": verdict, "h50_gate": g50, "h25_gate": g25,
                   "specificity_metrics_h50_better": spec_ok, "nfe4_clearly_worsened": nfe4,
                   "nfe4_degradation_flag": degr, "gap_closure": closure}
        for gg in (g50, g25):
            print(f"[GATE] {gg['arm']}: improved {len(gg['improved'])}{gg['improved']} "
                  f"worsened {len(gg['worsened'])}{gg['worsened']} "
                  f"closure>=0.5 {len(gg['closure_ge_0.5'])} -> {'PASS' if gg['pass'] else 'FAIL'}", flush=True)
        print(f"[SPEC] H50 > H25 on {len(spec_ok)} of 4: {spec_ok}", flush=True)
        print(f"[NFE4] H25 worsened {nfe4['H25']} flag={degr['H25']} | "
              f"H50 worsened {nfe4['H50']} flag={degr['H50']}", flush=True)
        print(f"\n[VERDICT] {verdict}", flush=True)

    if a.tag == "stage1":
        v = {r["metric"]: r["verdict"] for r in boot if r["comparison"] == "B:2->4"}
        imp = [m for m in GAP_METRICS if v.get(m) == "improves"]
        wor = [m for m in GAP_METRICS if v.get(m) == "worsens"]
        collapse = v.get("f1_excess") == "worsens"
        gate = (len(imp) >= 3) and (len(wor) == 0) and (not collapse)
        result |= {"stage1_gate_pass": bool(gate), "gap_improved": imp, "gap_worsened": wor,
                   "event_collapse": bool(collapse),
                   "verdict": None if gate else "BASELINE TARGET GAP NOT REPRODUCED"}
        print(f"\n[GATE] Stage 1: improved {len(imp)} {imp} | worsened {len(wor)} {wor} | "
              f"collapse {collapse} -> {'PASS' if gate else 'FAIL'}", flush=True)
    (OUT / f"{a.tag}_result.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
