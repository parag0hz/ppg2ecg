"""M3 finalisation: gates.csv, decision.json and a provenance record that pins the uncommitted evidence.

Reads the existing per-window CSV; no model is run and no number can move. Mirrors scripts/m2_finalize.py.
Run: .venv/bin/python scripts/m3_finalize.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts/m3_sec_imeanflow"
NFE, SEED = 4, 0
# (gate, metric, kind, margin, direction) — margins verbatim from prereg §10
GATES = [("G1", "qrs_deriv_rmse", "rel_improve_min", 0.05),
         ("G2", "qrs_curvature_err", "rel_improve_min", 0.05)]
NONINF = [("N1", "f1_excess@50", "abs_worse_max", 0.020), ("N2", "beats_ratio_dev", "abs_worse_max", 0.020),
          ("N3", "qrs_ptp_dev", "rel_worse_max", 0.05), ("N4", "qrs_energy_dev", "rel_worse_max", 0.05),
          ("N5", "corr", "abs_worse_max", 0.02), ("N6", "rmse", "rel_worse_max", 0.05),
          ("N7", "F4__ratio_dev", "rel_worse_max", 0.10)]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    eff = {r["metric"]: r for r in csv.DictReader(open(A / "bootstrap_effects.csv")) if int(r["nfe"]) == NFE}
    rows = []
    for gid, met, _kind, margin in GATES:
        r = eff[met]
        rel, lo, hi = float(r["rel_improvement"]), float(r["ci_lo"]), float(r["ci_hi"])
        mag, ci = rel >= margin, lo > 0
        rows.append({"gate": gid, "metric": met,
                     "requirement": f"relative improvement >= {margin:.0%} AND 95% paired CI entirely > 0",
                     "U": float(r["U"]), "E": float(r["S"]), "effect": float(r["effect"]),
                     "ci_lo": lo, "ci_hi": hi, "rel_improvement": rel, "margin": margin,
                     "magnitude_ok": mag, "ci_ok": ci, "result": "PASS" if (mag and ci) else "FAIL",
                     "note": "" if (mag and ci) else
                             f"CI is entirely favourable but the improvement ({rel:.2%}) is below the frozen {margin:.0%} bar"})
    # G3: neither subject may worsen by > 2% on BOTH gate metrics
    g3 = {s: {m: float(eff[m][f"rel_{s}"]) for m in ("qrs_deriv_rmse", "qrs_curvature_err")} for s in ("an0", "k2s")}
    bad = [s for s, v in g3.items() if v["qrs_deriv_rmse"] < -0.02 and v["qrs_curvature_err"] < -0.02]
    rows.append({"gate": "G3", "metric": "per-subject", "requirement": "neither subject worsens > 2% on BOTH gate metrics",
                 "U": None, "E": None, "effect": None, "ci_lo": None, "ci_hi": None,
                 "rel_improvement": None, "margin": 0.02, "magnitude_ok": not bad, "ci_ok": True,
                 "result": "PASS" if not bad else "FAIL",
                 "note": "; ".join(f"{s}: deriv {v['qrs_deriv_rmse']:+.2%}, curv {v['qrs_curvature_err']:+.2%}"
                                   for s, v in g3.items())})
    for nid, met, kind, margin in NONINF:
        r = eff[met]
        rel, ef = float(r["rel_improvement"]), float(r["effect"])
        worse = -rel if kind == "rel_worse_max" else -ef
        rows.append({"gate": nid, "metric": met, "requirement": f"E no worse than U by more than {margin} ({kind})",
                     "U": float(r["U"]), "E": float(r["S"]), "effect": ef, "ci_lo": float(r["ci_lo"]),
                     "ci_hi": float(r["ci_hi"]), "rel_improvement": rel, "margin": margin,
                     "magnitude_ok": worse <= margin, "ci_ok": True,
                     "result": "PASS" if worse <= margin else "FAIL",
                     "note": "NOT EVALUATED FOR THE VERDICT: prereg §11 stops at the G1 failure"})
    with open(A / "gates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    g1, g2 = rows[0], rows[1]
    n3 = next(r for r in rows if r["gate"] == "N3")
    n7 = next(r for r in rows if r["gate"] == "N7")
    (A / "decision.json").write_text(json.dumps({
        "stage": "M3", "method": "SEC-iMF", "arm": "E", "verdict": "D",
        "verdict_text": "SEC-IMEANFLOW NOT SUPPORTED",
        "clause": "preregistration §11: 'G1 or G2 fails -> verdict D, SEC-IMEANFLOW NOT SUPPORTED, STOP.'",
        "why": "G1 requires BOTH a relative improvement >= 5% AND a 95% CI entirely > 0. The CI condition is met "
               "(the improvement is significant) but the magnitude condition is not (+2.66% < 5%). Nothing in §10 "
               "or §11 conditions on the sign of the effect, so a favourable-but-small G1 fails exactly as a "
               "hostile one would.",
        "G1": {k: g1[k] for k in ("metric", "U", "E", "rel_improvement", "ci_lo", "ci_hi", "magnitude_ok", "ci_ok", "result")},
        "G2": {k: g2[k] for k in ("metric", "U", "E", "rel_improvement", "ci_lo", "ci_hi", "magnitude_ok", "ci_ok", "result")},
        "G3": g3,
        "counterfactual": "Had G1 passed, N3 (qrs_ptp_dev, %.4f vs a 5%% margin) also fails, so the stop tree would "
                          "have given verdict B, not A." % n3["rel_improvement"],
        "n7_does_not_fail_alone": {"N3": n3["result"], "N7": n7["result"],
                                   "consequence": "prereg §10's 'if it alone fails the verdict must explicitly say "
                                                  "spectral trade-off' clause is NOT triggered"},
        "arms_not_run_per_stop_rule": ["V (value-endpoint control)"],
        "not_run_per_stop_rule": ["NFE 1/2 sweep", "source seeds 1-3", "multi-seed", "test subjects"],
        "primary": {"nfe": NFE, "source_seed": SEED, "subjects": ["an0", "k2s"],
                    "n_rows_per_arm": 49200, "n_clusters": 12400},
    }, indent=1))

    vm = A / "validation_metrics.csv"
    (A / "provenance.json").write_text(json.dumps({
        "stage": "M3", "method": "SEC-iMF", "verdict": "D",
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "M3_START_SHA": "1cf0d361e739ca9f26d7cd9b988cbbd1e85e8e29",
        "prereg_sha": "d5db1c919a5462e18bc26a0c7b828567f71ee978",
        "erratum_sha": "88f85cea9264d4d34188d35be4a0d67194e45579",
        "implementation_sha": "c1aa8bd8f2440ee21fa511a2c2de588cd6466421",
        "numerical_audit_sha": "b71f91c891d9add4d788092da3b63cbf8265df91",
        "frozen": {"eps_struct": 1e-6, "lambda_SEC": 0.10, "w_d1": 0.5, "w_d2": 0.5, "seed": 42,
                   "optimizer_steps": 14409, "validation_rounds": 66, "primary_nfe": NFE, "primary_source_seed": SEED},
        "validation_metrics_csv": {"path": str(vm.relative_to(ROOT)), "bytes": vm.stat().st_size,
                                   "sha256": sha256_file(vm),
                                   "note": "not committed (prereg §12 / repo policy stores hashes instead)"},
        "checkpoints": {
            "U": {"path": "outputs/m2_arm_U_seed42/checkpoint_best.pt",
                  "file_sha256": sha256_file(ROOT / "outputs/m2_arm_U_seed42/checkpoint_best.pt"),
                  "state_sha256": "20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e",
                  "selected_epoch": 45, "retrained_for_M3": False},
            "E": {"path": "outputs/m3_arm_E_seed42/checkpoint_best.pt",
                  "file_sha256": sha256_file(ROOT / "outputs/m3_arm_E_seed42/checkpoint_best.pt"),
                  "selected_epoch": 60, "optimizer_steps": 14409}},
    }, indent=1))
    print(json.dumps([{k: r[k] for k in ("gate", "metric", "rel_improvement", "result")} for r in rows], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
