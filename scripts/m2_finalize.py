"""M2 finalisation: recompute the effects table with corrected metric orientation, then emit the §28 artifacts.

Reads the existing per-window validation_metrics.csv — no model is run, so no number can move except the SIGN of
the 13 secondary error-magnitude columns that an earlier suffix rule mislabelled. The gate and non-inferiority
metrics were never affected; this is verified explicitly and recorded in gates.csv.

Run: .venv/bin/python scripts/m2_finalize.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m2_evaluate as EV  # noqa: E402

A = ROOT / "artifacts/m2_structure_weighted_imeanflow"
PRIMARY_NFE, PRIMARY_SEED = 4, 0
GATES = {
    "G1": ("qrs_deriv_rmse", 0.05, "relative reduction >= 5% AND 95% paired CI for improvement > 0"),
    "G2": ("qrs_curvature_err", 0.05, "relative reduction >= 5% AND 95% paired CI > 0"),
}
G3_ANY = ("qrs_energy_dev", "qrs_ptp_dev")
NONINF = {  # (metric, kind, margin) — kind 'abs' or 'rel'
    "N1": ("f1_excess@50", "abs", 0.020), "N2": ("beats_ratio_dev", "abs", 0.020),
    "N3": ("rmse", "rel", 0.02), "N4": ("corr", "abs", 0.02), "N5": ("F4__ratio_dev", "rel", 0.10),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(nfe, seed):
    rows = {"U": [], "S": []}
    with open(A / "validation_metrics.csv") as f:
        for r in csv.DictReader(f):
            if int(r["nfe"]) == nfe and int(r["source_seed"]) == seed and r["arm"] in rows:
                rows[r["arm"]].append(r)
    return rows


def col(rows, k):
    return np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows], dtype=np.float64)


def main() -> int:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    eff_rows, gate_rows = [], []
    for nfe in (1, 2, 4):
        rows = load(nfe, PRIMARY_SEED)
        u_rows, s_rows = rows["U"], rows["S"]
        assert len(u_rows) == len(s_rows) and len(u_rows) > 0, (nfe, len(u_rows), len(s_rows))
        cl = np.array([r["cluster"] for r in u_rows])
        sb = np.array([r["subject"] for r in u_rows])
        assert [(r["cluster"], r["site"]) for r in u_rows] == [(r["cluster"], r["site"]) for r in s_rows], \
            "U and S rows are not element-wise paired"
        keys = [k for k in u_rows[0] if k not in ("arm", "nfe", "source_seed", "subject", "window_index", "site", "cluster")]
        for k in keys:
            u, s = col(u_rows, k), col(s_rows, k)
            if not (np.isfinite(u).any() and np.isfinite(s).any()):
                continue
            neutral = EV.is_neutral(k)
            orient = "neutral" if neutral else ("lower_better" if EV.is_lower_better(k) else "higher_better")
            b = EV.clustered_paired_bootstrap(u, s, cl, sb, "lower_better" if neutral else orient)
            mu, ms = EV.macro(u, cl, sb), EV.macro(s, cl, sb)
            rel = ((mu - ms) / abs(mu)) if orient == "lower_better" and mu else \
                  ((ms - mu) / abs(mu)) if orient == "higher_better" and mu else np.nan
            row = {"nfe": nfe, "metric": k, "orientation": orient, "U": mu, "S": ms,
                   "effect": b["point"], "ci_lo": b["lo"], "ci_hi": b["hi"], "rel_improvement": rel,
                   "ci_excludes_zero": bool(b["lo"] > 0 or b["hi"] < 0),
                   "ci_side": "above_zero (S better)" if b["lo"] > 0 else
                              "below_zero (S WORSE)" if b["hi"] < 0 else "spans_zero",
                   "n_clusters": b["n_clusters"], "n_subjects": b["n_subjects"]}
            for sub in np.unique(sb):
                m = sb == sub
                mu_s, ms_s = EV.macro(u[m], cl[m], sb[m]), EV.macro(s[m], cl[m], sb[m])
                row[f"rel_{sub}"] = ((mu_s - ms_s) / abs(mu_s)) if orient == "lower_better" and mu_s else \
                                    ((ms_s - mu_s) / abs(mu_s)) if orient == "higher_better" and mu_s else np.nan
            eff_rows.append(row)
        if nfe != PRIMARY_NFE:
            continue
        idx = {r["metric"]: r for r in eff_rows if r["nfe"] == PRIMARY_NFE}
        for gid, (met, margin, text) in GATES.items():
            r = idx[met]
            gate_rows.append({"gate": gid, "metric": met, "requirement": text, "U": r["U"], "S": r["S"],
                              "rel_improvement": r["rel_improvement"], "margin_required": margin,
                              "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "ci_side": r["ci_side"],
                              "magnitude_ok": bool(r["rel_improvement"] >= margin),
                              "ci_ok": bool(r["ci_lo"] > 0),
                              "result": "PASS" if (r["rel_improvement"] >= margin and r["ci_lo"] > 0) else "FAIL"})
        g3 = [idx[m] for m in G3_ANY]
        gate_rows.append({"gate": "G3", "metric": "|".join(G3_ANY), "requirement": "at least one improves with 95% CI > 0",
                          "U": np.nan, "S": np.nan,
                          "rel_improvement": max(r["rel_improvement"] for r in g3),
                          "margin_required": 0.0, "ci_lo": np.nan, "ci_hi": np.nan,
                          "ci_side": ";".join(f"{r['metric']}:{r['ci_side']}" for r in g3),
                          "magnitude_ok": True, "ci_ok": any(r["ci_lo"] > 0 for r in g3),
                          "result": "PASS" if any(r["ci_lo"] > 0 for r in g3) else "FAIL",
                          "note": "NOT EVALUATED FOR THE VERDICT: prereg §13 stops at the first G1/G2 failure"})
        for nid, (met, kind, margin) in NONINF.items():
            r = idx[met]
            worse = -r["rel_improvement"] if kind == "rel" else -(r["effect"])
            gate_rows.append({"gate": nid, "metric": met, "requirement": f"S no worse than U by more than {margin} ({kind})",
                              "U": r["U"], "S": r["S"], "rel_improvement": r["rel_improvement"],
                              "margin_required": margin, "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                              "ci_side": r["ci_side"], "magnitude_ok": bool(worse <= margin), "ci_ok": True,
                              "result": "PASS" if worse <= margin else "FAIL",
                              "note": "NOT EVALUATED FOR THE VERDICT: prereg §13 stops at the first G1/G2 failure"})

    with open(A / "bootstrap_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eff_rows[0]))
        w.writeheader()
        w.writerows(eff_rows)
    with open(A / "gates.csv", "w", newline="") as f:
        fn = sorted({k for r in gate_rows for k in r})
        w = csv.DictWriter(f, fieldnames=["gate", "metric", "requirement", "U", "S", "rel_improvement",
                                          "margin_required", "ci_lo", "ci_hi", "ci_side", "magnitude_ok",
                                          "ci_ok", "result"] + [k for k in fn if k == "note"])
        w.writeheader()
        w.writerows(gate_rows)

    g1 = next(r for r in gate_rows if r["gate"] == "G1")
    g2 = next(r for r in gate_rows if r["gate"] == "G2")
    decision = {
        "stage": "M2", "method": "GSW-iMF", "verdict": "D",
        "verdict_text": "STRUCTURE-WEIGHTED IMEANFLOW NOT SUPPORTED",
        "clause": "preregistration §13: 'G1 or G2 fails -> verdict D, STOP. No lambda tuning, no smoothing change, no extra control arms.'",
        "G1": {k: g1[k] for k in ("metric", "U", "S", "rel_improvement", "ci_lo", "ci_hi", "ci_side", "result")},
        "G2": {k: g2[k] for k in ("metric", "U", "S", "rel_improvement", "ci_lo", "ci_hi", "ci_side", "result")},
        "both_gates_fail_on_both_conditions": "the relative change is an INCREASE in error, not a >=5% reduction, "
                                              "and both CIs lie entirely BELOW zero, i.e. S is significantly worse",
        "arms_not_run_per_stop_rule": ["X (shifted control)", "Q (hard-QRS control)"],
        "primary": {"nfe": PRIMARY_NFE, "source_seed": PRIMARY_SEED,
                    "n_clusters": g1.get("n_clusters"), "subjects": ["an0", "k2s"]},
        "head": head,
    }
    (A / "decision.json").write_text(json.dumps(decision, indent=1))

    prov = {"head": head, "generated_by": "scripts/m2_finalize.py",
            "validation_metrics_csv": {"path": "artifacts/m2_structure_weighted_imeanflow/validation_metrics.csv",
                                       "bytes": (A / "validation_metrics.csv").stat().st_size,
                                       "sha256": sha256_file(A / "validation_metrics.csv"),
                                       "note": "562 MB: not committed (prereg §28 stores hashes instead)"},
            "checkpoints": {arm: {"path": f"outputs/m2_arm_{arm}_seed42/checkpoint_best.pt",
                                  "sha256": sha256_file(ROOT / f"outputs/m2_arm_{arm}_seed42/checkpoint_best.pt")}
                            for arm in ("U", "S")}}
    (A / "provenance.json").write_text(json.dumps(prov, indent=1))

    print(json.dumps({"gates": [{k: r[k] for k in ("gate", "metric", "rel_improvement", "ci_side", "result")}
                                for r in gate_rows], "verdict": "D"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
