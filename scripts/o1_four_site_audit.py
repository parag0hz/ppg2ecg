"""O1 step 2 — four-site ECG target identity audit (preregistration section 3).

WildPPG pairs four PPG sites with ONE sternum ECG. Before any target is built we verify that
(subject, window_index) identifies the same ECG waveform, the same R-peak train and the same scalar
targets across sites. Read-only; no model, no training, no test subject.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.probes import r1_cohort as C

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o1_component_extractability"
SUBJECTS = tuple(C.TRAIN12) + tuple(C.VAL)
AUDIT_SALT = "o1-four-site-audit-v1"
N_DEEP_PER_SUBJECT = 512


def _targets(y):
    return OT.window_targets(np.asarray(y, dtype=np.float64))


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(SUBJECTS)
    rows, deep_rows, summary = [], [], []
    for s in SUBJECTS:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Y, SITE, WI = d["y"], np.asarray(d["site"]).astype(str), d["window_index"].astype(np.int64)
        groups = defaultdict(list)
        for i in range(len(WI)):
            groups[int(WI[i])].append(i)
        hashes = [hashlib.sha256(np.ascontiguousarray(Y[i], dtype=np.float32).tobytes()).hexdigest() for i in range(len(Y))]
        multi = {w: idx for w, idx in groups.items() if len(idx) >= 2}
        n_ident = sum(1 for w, idx in multi.items() if len({hashes[i] for i in idx}) == 1)
        size_hist = defaultdict(int)
        for idx in groups.values():
            size_hist[len(idx)] += 1
        # deep check: R-peaks + all scalar targets on a deterministic sample of multi-site groups
        keys = sorted(multi, key=lambda w: hashlib.sha256(f"{AUDIT_SALT}|{s}|{w}".encode()).hexdigest())[:N_DEEP_PER_SUBJECT]
        with ProcessPoolExecutor(max_workers=12) as ex:
            tg = list(ex.map(_targets, [Y[multi[w][0]].astype(np.float64) for w in keys], chunksize=8))
            tg2 = list(ex.map(_targets, [Y[multi[w][-1]].astype(np.float64) for w in keys], chunksize=8))
        n_deep_ident = 0
        for k, w in enumerate(keys):
            same = all((np.isnan(tg[k][t]) and np.isnan(tg2[k][t])) or tg[k][t] == tg2[k][t] for t in OT.TARGETS)
            same = same and np.array_equal(tg[k]["_rpeaks"], tg2[k]["_rpeaks"])
            n_deep_ident += int(same)
            deep_rows.append({"subject": s, "window_index": int(w), "n_site_rows": len(multi[w]),
                              "sites": "|".join(sorted({SITE[i] for i in multi[w]})),
                              "waveform_hash_identical": int(len({hashes[i] for i in multi[w]}) == 1),
                              "rpeaks_identical": int(np.array_equal(tg[k]["_rpeaks"], tg2[k]["_rpeaks"])),
                              "all_targets_identical": int(same)})
        rows.append({"subject": s, "n_rows": int(len(Y)), "n_unique_window_index": len(groups),
                     "n_multi_site_groups": len(multi), "n_groups_waveform_identical": n_ident,
                     "frac_waveform_identical": n_ident / max(len(multi), 1),
                     "n_deep_checked": len(keys), "n_deep_identical": n_deep_ident,
                     "group_size_hist": json.dumps(dict(sorted(size_hist.items()))),
                     "n_unique_waveforms": len(set(hashes))})
        print(f"[{s}] rows {len(Y)} | unique wi {len(groups)} | multi-site {len(multi)} | "
              f"waveform-identical {n_ident}/{len(multi)} | deep {n_deep_ident}/{len(keys)}", flush=True)
        summary.append((s, len(multi), n_ident, len(keys), n_deep_ident))
    with open(ART / "four_site_target_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(ART / "four_site_target_audit_deep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(deep_rows[0])); w.writeheader(); w.writerows(deep_rows)
    ok = all(r["n_groups_waveform_identical"] == r["n_multi_site_groups"] for r in rows) and \
         all(r["n_deep_identical"] == r["n_deep_checked"] for r in rows)
    (ART / "four_site_target_audit.json").write_text(json.dumps(
        {"salt": AUDIT_SALT, "subjects": list(SUBJECTS), "n_deep_per_subject": N_DEEP_PER_SUBJECT,
         "identity_established": bool(ok), "per_subject": rows}, indent=2))
    print(f"\n[audit] (subject, window_index) identifies one ECG across sites: {ok}", flush=True)
    if not ok:
        raise RuntimeError("four-site ECG identity NOT established (STOP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
