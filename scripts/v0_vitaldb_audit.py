"""V0 -- VitalDB availability audit over every downloaded case (no training, no model, no metric)."""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ppg2ecg.data.vitaldb import MIN_DURATION_S, MIN_FINITE_FRAC, is_eligible, participant_files, summarize_case

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw/VitalDB", ROOT / "artifacts/v0_vitaldb_audit"


def one(p):
    s = summarize_case(p, 4)
    return dict(caseid=s.caseid, n_samples=s.n_samples, duration_s=round(s.duration_s, 1), finite_ppg=round(s.finite_frac_ppg, 4),
                finite_ecg=round(s.finite_frac_ecg, 4), has_both=s.has_both, est_windows_4s=s.est_windows, eligible=is_eligible(s))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = participant_files(RAW)
    with ProcessPoolExecutor(24) as ex:
        rows = list(ex.map(one, files, chunksize=8))
    meta = {int(r["caseid"]): r for r in csv.DictReader(open(RAW / "cases.csv", encoding="utf-8-sig"))}
    keep = ("subjectid", "age", "sex", "department", "optype", "ane_type", "preop_ecg", "emop")
    for r in rows:
        r.update({k: meta.get(r["caseid"], {}).get(k, "") for k in keep})
    with open(OUT / "cases_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    el = [r for r in rows if r["eligible"]]
    subj = {}
    for r in el:
        subj.setdefault(r["subjectid"], []).append(r["caseid"])
    ecg = {}
    for r in el:
        ecg[r["preop_ecg"] or "(blank)"] = ecg.get(r["preop_ecg"] or "(blank)", 0) + 1
    summary = dict(
        n_case_files=len(rows), n_cases_in_cases_csv=len(meta), n_has_both=sum(r["has_both"] for r in rows),
        rule=dict(min_duration_s=MIN_DURATION_S, min_finite_frac=MIN_FINITE_FRAC),
        n_eligible_cases=len(el), n_eligible_subjects=len(subj),
        n_subjects_with_multiple_cases=sum(len(v) > 1 for v in subj.values()),
        eligible_hours=round(sum(r["duration_s"] for r in el) / 3600, 1),
        eligible_windows_4s_before_nan_drop=sum(r["est_windows_4s"] for r in el),
        median_case_duration_min=round(sorted(r["duration_s"] for r in el)[len(el) // 2] / 60, 1),
        fail_reasons=dict(missing_track=sum(not r["has_both"] for r in rows),
                          short=sum(r["has_both"] and r["duration_s"] < MIN_DURATION_S for r in rows),
                          nan=sum(r["has_both"] and r["duration_s"] >= MIN_DURATION_S and min(r["finite_ppg"], r["finite_ecg"]) < MIN_FINITE_FRAC for r in rows)),
        preop_ecg_top=dict(sorted(ecg.items(), key=lambda kv: -kv[1])[:8]),
        sex=dict((s, sum(r["sex"] == s for r in el)) for s in ("M", "F")),
    )
    ages = sorted(float(r["age"]) for r in el if r["age"] not in ("",))
    summary["age_median_iqr"] = [ages[len(ages) // 2], ages[len(ages) // 4], ages[3 * len(ages) // 4]]
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
