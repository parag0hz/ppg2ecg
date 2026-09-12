"""N1 — the supervision-matched null method (docs/N1_NULL_METHOD_STAMPING_BASELINE_PREREGISTRATION.md).

R1's PPG-only event scaffold + a fixed QRS template, stamped. NO TRAINING, NO WEIGHT UPDATE,
NO OPTIMIZER: the only network touched is the frozen R1 Global-TCN, in eval() under no_grad.

Cohort, scoring and uncertainty are R2/R3's, imported rather than reimplemented.

Run: .venv/bin/python scripts/n1_evaluate.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppg2ecg.evaluation import event_reliability as ER          # noqa: E402
from ppg2ecg.evaluation import stamping as ST                   # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT                  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import extract_events            # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402

# Plain import so the module object is IDENTICAL in parent and ProcessPoolExecutor children --
# a spec-loaded copy makes `pickle` fail on r2_evaluate._peaks. Scoring imported, never reimplemented.
import r2_evaluate as R2E                                       # noqa: E402

ART = ROOT / "artifacts/n1_null_stamping"
PREREG = "f7a6b70"
VAL, SALT, TAKE = R2E.VAL, R2E.SALT, R2E.TAKE                   # ("an0","k2s"), "x4-event-nfe-v2", 1024
FS, T_LEN = 128, 1024
R1_THRESHOLD, R1_REFRACTORY = 0.35, 32                          # frozen in artifacts/r1_.../threshold_selection.json
GATE_MIN_EFFECT = 0.02                                          # R2/R3 GATE_MIN_EFFECT, reused
BOOT_N, BOOT_SEED = 2000, 20260901
TEMPLATE_A_SHA = "1a67569f8a02bc0027c0a60c4575d297dc2bc40eb0c3e285b9acf82daafd51eb"
TEMPLATE_B_SHA = "6f059015812308a9d71b4c72e08a6eeed68ee2ca29107bc801097d8a1748e595"
ORACLE_LABEL = "(GT-R leakage; diagnostic only)"
SM = ("f1_excess", "missing", "spurious", "beats_ratio_dev", "raw_qrs_rmse",
      "qrs_deriv_rmse", "qrs_curvature_err", "morph", "ww_corr", "ww_rmse")


def load_cohort():
    """The frozen R2/R3 population, asserted element-wise against x4_0's subset."""
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys = d["x"], d["y"]
        idx = ER.select_subset(SALT, s, len(Xs), TAKE)
        X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); SITE.append(np.asarray(d["site"]).astype(str)[idx])
        POS.append(idx); WI.append(d["window_index"][idx].astype(np.int64))
    X, Y, SUB, SITE, POS, WI = (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s in VAL:
        assert POS[SUB == s].tolist() == list(frozen[s]), f"frozen subset mismatch for {s}"
    return X, Y.astype(np.float64), SUB, SITE, WI, POS


def templates() -> tuple[np.ndarray, np.ndarray, dict]:
    path = ROOT / "artifacts/s1_metric_validity/template_A.npy"
    # prereg §4 quotes S1's FILE digest for template_A.npy and S1's ARRAY digest for the T-B crop,
    # exactly as S1_G1_METRIC_VALIDITY_REPORT.md labels them. Both are asserted in their own space.
    ha = hashlib.sha256(path.read_bytes()).hexdigest()
    a = np.load(path).astype(np.float64)
    b = ST.crop_qrs(a, FS)
    hb = ST.sha256_array(b)
    if ha != TEMPLATE_A_SHA or hb != TEMPLATE_B_SHA:
        raise RuntimeError(f"frozen S1 template hashes differ: A(file) {ha} B(array) {hb}")
    geo = ST.template_geometry(FS)
    return a, b, {"template_A_file_sha256": ha, "template_B_array_sha256": hb,
                  "template_A_array_sha256": ST.sha256_array(a), "geometry": geo,
                  "len_A": int(a.size), "len_B": int(b.size)}


@torch.no_grad()
def r1_events(X: np.ndarray, dev) -> tuple[list[np.ndarray], np.ndarray, dict]:
    tcn, meta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    field = R2E.scaffolds(tcn, X, dev)                                   # [N, T] sigmoid field
    ev = [extract_events(field[i], R1_THRESHOLD, R1_REFRACTORY) for i in range(len(field))]
    del tcn
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return ev, field, meta


def const_events(ev_r1: list[np.ndarray], n_time: int = T_LEN) -> list[np.ndarray]:
    """Uniform grid at the window's own median R1 RR, phase anchored on the first R1 event.
    Rhythm without placement: keeps rate, discards every individual event time."""
    out = []
    for e in ev_r1:
        if e.size < 2:
            out.append(np.asarray(e, dtype=int)); continue
        rr = float(np.median(np.diff(e)))
        if not np.isfinite(rr) or rr < 1:
            out.append(np.asarray(e, dtype=int)); continue
        first = int(e[0]) % max(int(round(rr)), 1)
        out.append(np.arange(first, n_time, int(round(rr)), dtype=int))
    return out


def render(events: list[np.ndarray], tmpl: np.ndarray, r_index: int) -> np.ndarray:
    return np.stack([ST.stamp(tmpl, ev, T_LEN, r_index) for ev in events]).astype(np.float64)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    git = git_sha(ROOT)

    X, Yd, SUB, SITE, WI, POS = load_cohort()
    gt_pk = R2E.pmap(R2E._peaks, list(Yd))
    n_gt = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_gt != 19834:
        raise RuntimeError(f"frozen population facts differ (STOP): {len(X)} windows, {n_gt} GT beats")
    print(f"[N1] {len(X)} windows, {n_gt} GT beats, subjects {sorted(set(SUB))}", flush=True)

    tA, tB, tmeta = templates()
    rA, rB = tmeta["geometry"]["r_index_full"], tmeta["geometry"]["r_index_qrs"]   # S1 convention, verbatim
    print(f"[N1] templates OK  A len {tA.size} r@{rA}   B len {tB.size} r@{rB}", flush=True)

    ev_r1, field, rmeta = r1_events(X, dev)
    partner = RT.shuffle_partner(SUB, SITE, WI)
    RT.assert_derangement(partner)
    ev_shuf = [ev_r1[int(j)] for j in partner]
    ev_const = const_events(ev_r1)
    ev_gt = [np.asarray(p, dtype=int) for p in gt_pk]

    arms = {
        "N-R1":      (ev_r1,    tB, rB),
        "N-R1-A":    (ev_r1,    tA, rA),
        "N-SHUFFLE": (ev_shuf,  tB, rB),
        "N-CONST":   (ev_const, tB, rB),
        f"N-GT {ORACLE_LABEL}": (ev_gt, tB, rB),
    }

    per, macro = {}, {}
    for name, (ev, tmpl, ri) in arms.items():
        pred = render(ev, tmpl, ri)
        rows, _pk, _errs = R2E.score(pred, Yd, gt_pk)
        per[name] = rows
        macro[name] = R2E.macro_rows(rows, SUB)
        overlap = int(sum(ST.stamp_supports_overlap(e, ri, tmpl.size) for e in ev))
        macro[name]["n_windows_overlapping_stamps"] = overlap
        macro[name]["n_events_total"] = int(sum(len(e) for e in ev))
        print(f"[N1] {name:36s} f1_excess {macro[name]['f1_excess']:+.4f}  miss {macro[name]['missing']:.3f}  "
              f"spur {macro[name]['spurious']:.3f}  S4 {macro[name]['qrs_deriv_rmse']:.4f}  "
              f"S5 {macro[name]['qrs_curvature_err']:.4f}  overlap {overlap}/{len(ev)}", flush=True)

    # ---- frozen comparators, read not recomputed ----
    import csv
    r3 = {(r["arm"], int(r["nfe"])): r for r in csv.DictReader((ROOT / "artifacts/r3_rhythm_fusion/event_metrics.csv").open())}
    comparators = {a: float(r3[(a, 4)]["f1_excess"]) for a in ("B", "ADD", "TF-TRUE", "GTF-CONST", "GTF-TRUE") if (a, 4) in r3}

    # ---- §5 primary: N-R1 vs GTF-TRUE needs paired per-window rows for both arms.
    # GTF-TRUE's per-window rows are frozen in R3; join on (subject, window_index) order.
    r3_per = ROOT / "artifacts/r3_rhythm_fusion/metrics_by_window.csv"
    def r3_arm(arm: str, nfe: int = 4) -> dict:
        """Per-window rows of a frozen R3 arm, keyed (subject, array_pos) -- an explicit join, never
        an assumption about row order."""
        out = {}
        for r in csv.DictReader(r3_per.open()):
            if r.get("arm") == arm and int(r["nfe"]) == nfe:
                out[(r["subject"], int(r["array_pos"]))] = r
        return out

    pairs, join_report = {}, {}
    for comp in ("GTF-TRUE", "B"):
        idx = r3_arm(comp)
        keys = [(SUB[i], int(POS[i])) for i in range(len(SUB))]
        hit = [k in idx for k in keys]
        join_report[comp] = {"n_rows_frozen": len(idx), "n_cohort": len(keys), "n_matched": int(sum(hit))}
        if not all(hit):
            print(f"[N1] WARNING: {comp} per-window join incomplete "
                  f"({sum(hit)}/{len(keys)}); its paired test is skipped", flush=True)
            continue
        for m in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err", "missing", "spurious"):
            a = np.asarray([float(idx[k][m]) for k in keys], float)        # earlier = comparator
            b = np.asarray([row[m] for row in per["N-R1"]], float)         # later = null method
            orient = "higher_better" if m == "f1_excess" else "lower_better"
            pairs[f"N-R1_vs_{comp}:{m}"] = paired_subject_bootstrap(a, b, SUB, orient, BOOT_N, BOOT_SEED)
    # §6 gate needs N-GT vs B on S4
    idxB = r3_arm("B")
    keys = [(SUB[i], int(POS[i])) for i in range(len(SUB))]
    if all(k in idxB for k in keys):
        for m in ("qrs_deriv_rmse", "qrs_curvature_err"):
            a = np.asarray([float(idxB[k][m]) for k in keys], float)
            b = np.asarray([row[m] for row in per[f"N-GT {ORACLE_LABEL}"]], float)
            pairs[f"N-GT_vs_B:{m}"] = paired_subject_bootstrap(a, b, SUB, "lower_better", BOOT_N, BOOT_SEED)
    joined = bool(pairs)
    for m in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err"):
        orient = "higher_better" if m == "f1_excess" else "lower_better"
        a = np.asarray([row[m] for row in per["N-SHUFFLE"]], float)
        b = np.asarray([row[m] for row in per["N-R1"]], float)
        pairs[f"N-R1_vs_N-SHUFFLE:{m}"] = paired_subject_bootstrap(a, b, SUB, orient, BOOT_N, BOOT_SEED)

    out = {"prereg": PREREG, "git": git, "utc": datetime.now(timezone.utc).isoformat(),
           "test_subjects_loaded": [], "cohort": {"windows": int(len(X)), "gt_beats": n_gt,
                                                  "subjects": sorted(set(SUB.tolist()))},
           "templates": tmeta, "r1": rmeta, "r1_threshold": R1_THRESHOLD, "r1_refractory": R1_REFRACTORY,
           "macro": macro, "frozen_comparators_nfe4": comparators, "paired": pairs,
           "gtf_per_window_joined": joined, "join_report": join_report,
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n1_results.json").write_text(json.dumps(out, indent=1, default=float))
    R2E.wcsv(ART / "macro_metrics.csv", [{"arm": k, **v} for k, v in macro.items()])
    print(f"\n[N1] wrote {ART}/n1_results.json  ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
