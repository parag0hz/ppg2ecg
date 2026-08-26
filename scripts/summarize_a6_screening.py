"""Collect the A6 state-constant / conditioning screening traces (outputs/gradcheck_a6_*/training_log.csv) into
artifacts/a6_capacity_control/state_constant_screening.json with the pre-registered admissibility rule (prereg §2b)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = "admissible if within 12 epochs the validation diagnostic shows beats/reference >= 0.3 or amplitude ratio >= 0.02 at any epoch"


def main():
    out = {"rule": RULE, "runs": {}}
    for d in sorted(ROOT.glob("outputs/gradcheck_a6_*")):
        meta = json.loads((d / "train_meta.json").read_text())
        rows = list(csv.DictReader(open(d / "training_log.csv")))
        ep = [{"epoch": int(r["epoch"]) + 1, "train_mse": float(r["train_mse"]), "val_mse": float(r["val_mse"]), "amp": float(r["diag_amp_ratio"]), "beats": float(r["diag_beats_ratio"]), "hr": float(r["diag_hr_abs_err"]), "morph": float(r["diag_morph_corr"])} for r in rows]
        adm = any((e["beats"] >= 0.3) or (e["amp"] >= 0.02) for e in ep)
        out["runs"][d.name] = {"model_cfg": meta.get("model_cfg"), "n_epochs": len(ep), "admissible": adm, "max_beats": max(e["beats"] for e in ep), "max_amp": max(e["amp"] for e in ep), "final_train_mse": ep[-1]["train_mse"], "best_val_mse": min(e["val_mse"] for e in ep), "epochs": ep}
        print(f"{d.name}: cfg {meta.get('model_cfg')} | {len(ep)} ep | max beats {out['runs'][d.name]['max_beats']:.2f} max amp {out['runs'][d.name]['max_amp']:.3f} | final train MSE {ep[-1]['train_mse']:.4f} | admissible {adm}")
    p = ROOT / "artifacts/a6_capacity_control/state_constant_screening.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
