#!/usr/bin/env python
"""U1 — assemble the verdict table from upstream's own logs and the realised splits.

Reads only what upstream printed (`outputs/u1_upstream/train_<DS>.log`) plus the split
manifests this stage recorded. Applies the decision rule frozen in
docs/U1_UPSTREAM_PENGUIN_ASSHIPPED_PREREGISTRATION.md §5 -- no metric is selected,
dropped or re-derived here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

LOGROOT = Path("outputs/u1_upstream")

# Published values, arXiv:2602.03858 Table 1 (as recorded in docs/PENGUIN_AUDIT.md and
# docs/D3_PENGUIN_SIX_DATASET_REPORT.md). Fixed in the preregistration before any run.
PAPER = {
    ("PPG-DaLiA", "HeartRateError"): 15.64,
    ("WildPPG", "HeartRateError"): 12.97,
    ("BIDMC", "RespRateError"): 2.98,
    ("WESAD", "RespRateError"): 4.45,
    ("UCI-BP", "SBPError"): 12.61,
    ("UCI-BP", "DBPError"): 7.14,
    ("MIMIC-BP", "SBPError"): 17.43,
    ("MIMIC-BP", "DBPError"): 11.34,
}
UNIT = {"HeartRateError": "bpm", "RespRateError": "bpm", "SBPError": "mmHg", "DBPError": "mmHg"}
ORDER = ["PPG-DaLiA", "WildPPG", "BIDMC", "WESAD", "UCI-BP", "MIMIC-BP"]
TIER = {"BIDMC": "A", "WESAD": "A", "PPG-DaLiA": "A", "MIMIC-BP": "B", "WildPPG": "B", "UCI-BP": "B"}
CAP = {"MIMIC-BP": 20, "WildPPG": 12, "UCI-BP": 12}

def _short(xs, k: int = 6) -> str:
    """Test lists run to 190 entries on MIMIC-BP; show a head and the count."""
    if not xs:
        return str(xs)
    return str(xs) if len(xs) <= k else f"{xs[:k]}… ({len(xs)} subjects)"


METRIC_RE = re.compile(r"^(HeartRateError|RespRateError|SBPError|DBPError|MAE)\s+: ([0-9.]+)", re.M)


def verdict(r: float) -> str:
    if 0.80 <= r <= 1.25:
        return "REPRODUCED"
    if 0.60 <= r < 0.80 or 1.25 < r <= 1.67:
        return "PARTIAL"
    return "NOT REPRODUCED"


def read_dataset(ds: str) -> dict | None:
    log = LOGROOT / f"train_{ds}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    # The test block is everything after the checkpoint reload line.
    if "Loaded checkpoint" not in text:
        return {"incomplete": True}
    tail = text.split("Loaded checkpoint", 1)[1]
    metrics = {m.group(1): float(m.group(2)) for m in METRIC_RE.finditer(tail)}
    epochs = text.count("\nVal, MAE:")
    best = re.findall(r"Saving checkpoint at epoch (\d+)", text)
    early = "Early stopping" in text
    out = {
        "metrics": metrics,
        "epochs_run": epochs,
        "best_epoch_1based": int(best[-1]) + 1 if best else None,
        "early_stopped": early,
        "cap": CAP.get(ds),
        "tier": TIER[ds],
    }
    sp = LOGROOT / f"split_{ds}.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        out["split"] = {k: d[k] for k in ("val", "test", "n_windows", "n_windows_total")}
        out["duplicate_groups"] = d["duplicate_groups"]
    return out


def main() -> None:
    rows, results = [], {}
    for ds in ORDER:
        r = read_dataset(ds)
        results[ds] = r
        if r is None:
            rows.append((ds, "—", "not run", "", "", "", ""))
            continue
        if r.get("incomplete"):
            rows.append((ds, "—", "still training", "", "", "", ""))
            continue
        for metric, paper in [(m, v) for (d, m), v in PAPER.items() if d == ds]:
            got = r["metrics"].get(metric)
            if got is None:
                rows.append((ds, metric, "missing", f"{paper}", "", "", ""))
                continue
            ratio = got / paper
            rows.append(
                (ds, metric, f"{got:.3f}", f"{paper:.2f}", UNIT[metric], f"{ratio:.3f}", verdict(ratio))
            )

    w = [max(len(str(row[i])) for row in rows + [("dataset", "metric", "U1", "paper", "unit", "r", "verdict")])
         for i in range(7)]
    hdr = ("dataset", "metric", "U1", "paper", "unit", "r", "verdict")
    print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for row in rows:
        print(" | ".join(str(c).ljust(w[i]) for i, c in enumerate(row)))

    counts: dict[str, int] = {}
    for row in rows:
        if row[6]:
            counts[row[6]] = counts.get(row[6], 0) + 1
    print("\nverdict counts:", counts)

    print("\nrun detail:")
    for ds in ORDER:
        r = results[ds]
        if not r or r.get("incomplete"):
            print(f"  {ds}: {'still training' if r else 'not run'}")
            continue
        s = r.get("split", {})
        print(
            f"  {ds} [tier {r['tier']}] epochs {r['epochs_run']}"
            + (f"/cap {r['cap']}" if r['cap'] else "")
            + f", best {r['best_epoch_1based']} (1-based, upstream numbering), early_stop={r['early_stopped']}"
            + f", windows {s.get('n_windows')}, test={_short(s.get('test'))}"
            + (f", DUPLICATE GROUPS {len(r['duplicate_groups'])}" if r.get("duplicate_groups") else "")
        )
    Path(LOGROOT / "u1_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {LOGROOT / 'u1_results.json'}")


if __name__ == "__main__":
    main()
