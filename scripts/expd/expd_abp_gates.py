"""EXP-D Part B — usability / informativeness gate (docs/EXP_D_ABP_PREREGISTRATION_AMENDMENT.md), applied after the
unchanged preregistered analysis (`expd_run.py analyze abp`, rules of eddbe60).

Per functional F in {SBP, DBP, MAP}: Gate A = (8,4) consensus MAE − training-median-constant MAE, paired subject bootstrap,
PASS if CI upper < 0; Gate B = Spearman(consensus (8,4), reference) over valid 8-s blocks, subject bootstrap, PASS if CI
lower > 0 (Pearson secondary). Usability, allocation and evidentiary verdicts per functional; cross-functional evidentiary
verdict. The same gate code is also run on BIDMC respiration (retrospective label, interpretation only).
Writes artifacts/exp_d_functional_generalization/abp/usability_gates.json.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/expd/expd_abp_gates.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import expd_run as X  # noqa: E402

GATE_CELL = (8, 4)


def corr_boot(pred, ref, sub, rank):
    ok = np.isfinite(pred) & np.isfinite(ref)
    p, r, s = pred[ok], ref[ok], sub[ok]
    f = (lambda a: rankdata(a)) if rank else (lambda a: a)
    c = lambda a, b: float(np.corrcoef(f(a), f(b))[0, 1])  # noqa: E731
    subs = np.unique(s); idx = {u: np.flatnonzero(s == u) for u in subs}
    rng = np.random.default_rng(X.BOOT_SEED)
    bs = []
    for _ in range(X.NB):
        m = np.concatenate([idx[u] for u in rng.choice(subs, len(subs))])
        bs.append(c(p[m], r[m]))
    return {"point": c(p, r), "ci": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n_blocks": int(ok.sum()),
            "n_subjects": int(len(subs))}


def gate(ds, j, const):
    z = np.load(X.RAW / f"{ds}.npz"); T = z["T"][:, j]; sub = z["block_subject"]
    K, S = GATE_CELL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cons = np.nanmedian(z[f"F_S{S}"][:K, :, j], 0)
    bt = X.Boot(sub)
    a = X.contrast(bt, np.abs(cons - T), np.abs(const - T))
    sp = corr_boot(cons, T, sub, rank=True); pe = corr_boot(cons, T, sub, rank=False)
    ga, gb = a["diff"]["ci"][1] < 0, sp["ci"][0] > 0
    use = "USABLE" if ga and gb else ("PARTIALLY INFORMATIVE" if ga or gb else "UNINFORMATIVE")
    return {"constant_b_F": float(const), "mae_consensus_8_4": a["a"], "mae_constant": a["b"], "delta_const": a["diff"],
            "gate_A_pass": bool(ga), "spearman": sp, "gate_B_pass": bool(gb), "pearson_secondary": pe, "usability": use}


def evidentiary(use, alloc):
    if use == "USABLE":
        return {"success": "SUPPORTS FUNCTIONAL GENERALISATION", "depth-clear": "GENUINE COUNTEREXAMPLE",
                "no clear difference": "NO EFFECT on a usable functional"}[alloc]
    return {"success": "NOT EVIDENCE FOR GENERALISATION", "depth-clear": "NOT EVIDENCE (not counterevidence)",
            "no clear difference": "UNINTERPRETABLE / NON-INFORMATIVE TASK"}[alloc]


def main():
    bj = json.loads((X.ART / "abp" / "bootstrap.json").read_text())
    orig = bj["verdict"]
    out = {"amendment": "docs/EXP_D_ABP_PREREGISTRATION_AMENDMENT.md", "gate_cell": list(GATE_CELL), "functionals": {},
           "original_frozen_verdict": orig}
    const = X.train_constant("MIMIC-BP", "abp")
    for j, fn in enumerate(X.FUNCS["abp"]):
        g = gate("MIMIC-BP", j, const[j])
        pf = orig["per_functional"][fn]
        alloc = "success" if pf["success"] else ("depth-clear" if pf["depth_clear"] else "no clear difference")
        c = bj["datasets"]["MIMIC-BP"]["functionals"][fn]["contrasts"]
        g.update({"allocation_verdict": alloc, "B32_width_points": {k: c[k]["diff"]["point"] for k in list(X.HEAD)[:2]},
                  "evidentiary_verdict": evidentiary(g["usability"], alloc)})
        out["functionals"][fn] = g
    F = out["functionals"]
    usable = [f for f in F if F[f]["usability"] == "USABLE"]
    ns = [f for f in usable if F[f]["allocation_verdict"] == "success"]
    nd = [f for f in usable if F[f]["allocation_verdict"] == "depth-clear"]
    ndir = [f for f in usable if F[f]["allocation_verdict"] == "no clear difference" and min(F[f]["B32_width_points"].values()) < 0]
    if len(nd) >= 2:
        v = "COUNTEREVIDENCE"
    elif len(ns) >= 2:
        v = "STRONG CROSS-FUNCTIONAL SUPPORT"
    elif len(ns) == 1 or len(ndir) >= 2:
        v = "PARTIAL CROSS-FUNCTIONAL SUPPORT"
    else:
        v = "NO SUPPORT"
    out["cross_functional_evidentiary_verdict"] = {"category": v, "usable": usable, "usable_success": ns, "usable_depth_clear": nd,
                                                   "usable_directional_only": ndir}
    # retrospective respiration label (interpretation only; frozen verdict unchanged)
    rc = X.train_constant("BIDMC", "respiration")[0]
    out["respiration_retrospective"] = {"frozen_statistical_verdict": "STRONG (unchanged)", **gate("BIDMC", 0, rc)}
    (X.ART / "abp" / "usability_gates.json").write_text(json.dumps(out, indent=1))
    for fn, g in F.items():
        print(f"{fn}: const {g['constant_b_F']:.2f} | (8,4) {X.fmt(g['mae_consensus_8_4'])} vs const {X.fmt(g['mae_constant'])} "
              f"Δ {X.fmt(g['delta_const'])} A={g['gate_A_pass']} | Spearman {X.fmt(g['spearman'])} B={g['gate_B_pass']} "
              f"(Pearson {X.fmt(g['pearson_secondary'])}) -> {g['usability']} | allocation {g['allocation_verdict']} -> {g['evidentiary_verdict']}")
    r = out["respiration_retrospective"]
    print(f"RESP retro: Δ {X.fmt(r['delta_const'])} A={r['gate_A_pass']} Spearman {X.fmt(r['spearman'])} B={r['gate_B_pass']} -> {r['usability']}")
    print("CROSS-FUNCTIONAL:", json.dumps(out["cross_functional_evidentiary_verdict"]))


if __name__ == "__main__":
    main()
