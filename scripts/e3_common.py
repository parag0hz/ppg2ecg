"""E3 shared helpers — frozen R1 candidates, the four schedule arms, and E2-contract evaluation.

NO TRAINING except the single permitted linear count readout, which lives in scripts/e3_fit_count.py.
Every metric comes from the frozen E2 contract; nothing is restated here.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import e3_beat_set as E3
from ppg2ecg.evaluation import event_geometry_contract as EG

sys.path.insert(0, str(Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT = C.ROOT
ART = ROOT / "artifacts/e3_beat_set_first"
E2ART = ROOT / "artifacts/e2_evaluation_contract"
E2_CONTRACT_SHA = "06e869412114e1efb9ab6624540aebc141495e2690150f4e512df7676c5a9115"
TRAIN_RPEAK_CACHE = ROOT / "artifacts/o2c_oracle_integer_grid/_cache_train_rpeaks.npz"
MANIFEST = "data/manifests/split_a4_wildppg_seed42.json"
PROCESSED = "data/processed/wildppg_8s"


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def assert_e2_contract() -> dict:
    """E3 imports the E2 contract; if its identity moved, E3 stops."""
    got = fsha(E2ART / "contract_v1.json")
    if got != E2_CONTRACT_SHA:
        raise RuntimeError(f"E2 contract changed (STOP): {got} != {E2_CONTRACT_SHA}")
    c = json.loads((E2ART / "contract_v1.json").read_text())
    assert c["contract_version"] == EG.VERSION == "e2-event-geometry-contract-v1"
    return {"contract_v1_sha256": got, "contract_version": c["contract_version"],
            "source_e1_sha": c["source"]["e1_result_sha"], "modified": False,
            "matching": c["matching"], "aggregation": c["aggregation"], "bootstrap": c["bootstrap"],
            "coverage_requirements_keys": sorted(c["coverage_requirements"]),
            "metric_ids": sorted(c["metrics"]),
            "imported_module": "ppg2ecg.evaluation.event_geometry_contract"}


@torch.no_grad()
def r1_scores(tcn, X, dev, batch: int = 64):
    """The exact frozen R1 score: sigmoid(RhythmTCN(ppg)). No recalibration, smoothing or normalisation."""
    out = []
    for i in range(0, len(X), batch):
        pp = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dev).unsqueeze(1)
        out.append(torch.sigmoid(tcn(pp)).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def r1_prelogit_features(tcn, X, dev, batch: int = 64):
    """z = concat(mean_t(H), max_t(H)) for the frozen pre-logit tensor H, the only layer before `head`."""
    out = []
    for i in range(0, len(X), batch):
        pp = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dev).unsqueeze(1)
        h = tcn.stem(pp)
        for blk in tcn.blocks:
            h = blk(h)
        out.append(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def candidates_for(P):
    """Threshold-free candidates and their scores for every window."""
    cand = [E3.candidate_events(p) for p in P]
    return [c[0] for c in cand], [c[1] for c in cand]


def arm_schedules(kind, cand_pos, cand_sc, *, threshold=None, counts=None):
    """One deterministic selection rule per arm. No GT location ever enters."""
    S, short = [], []
    for i in range(len(cand_pos)):
        if kind == "threshold":
            sel = cand_pos[i][cand_sc[i] >= threshold]
            sh = False
        elif kind == "topk":
            sel, sh = E3.topk_select(cand_pos[i], cand_sc[i], int(counts[i]))
        else:
            raise ValueError(kind)
        S.append(np.asarray(sel, dtype=np.int64)); short.append(sh)
    return S, short


def _t0_only(row: dict) -> dict:
    """B5/B6 at window level: the placement value on a T0 window, NaN elsewhere.

    A paired bootstrap on these therefore compares the INTERSECTION of the two arms' T0 windows; the number
    of dropped windows is reported with every such contrast.
    """
    t0 = row["A6_topology_class"] == EG.T0
    return {"B5_exact_set_mae_ms": row["B2_mae_ms"] if t0 else np.nan,
            "B6_exact_set_p90_ae_ms": row["B3_p90_ae_ms"] if t0 else np.nan}


def schedule_block(gt_pk, S):
    """E2 blocks A, B and the S->G half of the joint-event family. No generator is involved."""
    rows = []
    for i in range(len(S)):
        pairs, ur, up = EG.assign_schedule_to_gt(gt_pk[i], S[i])
        topo = EG.classify_topology(gt_pk[i], S[i], pairs, ur, up)
        row = {"row": i, **topo, **EG.placement_metrics(gt_pk[i], S[i], pairs),
               **EG.joint_event_fidelity(gt_pk[i], S[i], EG.TOLS_PG_MS, prefix="SG_")}
        row |= E3.topology_indicators(row)
        row |= _t0_only(row)
        rows.append(row)
    return rows


def full_block(gt_pk, S, pred, Yd, Ppk):
    """The complete frozen E2 per-window block once a generator output exists."""
    rows, beats = [], []
    for i in range(len(S)):
        r, b = EG.apply_contract(gt_pk[i], S[i], pred[i], Yd[i], Ppk[i], None)
        r["row"] = i
        r |= E3.topology_indicators(r)
        r |= _t0_only(r)
        rows.append(r); beats += [{"row": i, **x} for x in b]
    return rows, beats


def macro_block(rows, SUB, extra=()):
    num = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {k: C.macro([r[k] for r in rows], SUB) for k in num}
    out |= EG.exact_set_summary(rows, SUB)
    for k in extra:
        out[k] = C.macro([r[k] for r in rows], SUB)
    return out


HIGHER_BETTER = {"A5_exact_set", "T0_frac", "SG_F1_50", "SG_F1_100", "SG_F1_150", "SG_F1_200",
                 "PG_F1_50", "PG_F1_100", "PG_F1_150", "PG_F1_200", "SG_PREC", "SG_REC", "PG_PREC",
                 "PG_REC", "AD_F1_50", "AD_F1_100", "C8_own_local_corr", "J4_gt_local_corr"}


def effect(new_rows, ref_rows, key, SUB, CLUSTER, label=""):
    """Frozen E2 orientation: POSITIVE ALWAYS MEANS THE NEW ARM IS BETTER."""
    a = np.array([r[key] for r in new_rows], float)
    b = np.array([r[key] for r in ref_rows], float)
    d = (a - b) if key in HIGHER_BETTER else (b - a)
    res = C.O1E.cluster_bootstrap(d, SUB, CLUSTER, n_boot=EG.BOOT_N, seed=EG.BOOT_SEED)
    return {"contrast": label, "metric": key,
            "orientation": "higher_better" if key in HIGHER_BETTER else "lower_better",
            "positive_means": "NEW is better", "NEW": C.macro(a, SUB), "REF": C.macro(b, SUB),
            "point": res["point"], "lo": res["lo"], "hi": res["hi"], "verdict": res["verdict"],
            "n_dropped_windows": int(np.sum(~np.isfinite(d)))}


SCHED_EFFECT_KEYS = ("A5_exact_set", "T2_frac", "T3_frac", "A3_missing_fraction", "A4_spurious_fraction",
                     "B5_exact_set_mae_ms", "SG_F1_50", "SG_F1_150", "A1_abs_count_error", "B2_mae_ms")
GEN_EFFECT_KEYS = ("PG_F1_50", "PG_F1_150", "C1_own_T4", "C2_own_T6", "C3_own_T7", "C4_own_T8",
                   "J1_gt_local_raw_rmse", "J2_gt_local_deriv_rmse", "J3_gt_local_curvature_err",
                   "AD_F1_50")


def effects(new_rows, ref_rows, keys, SUB, CLUSTER, label):
    """B5 is defined on T0 windows only; a window without one contributes NaN and is dropped, as declared."""
    return {k: effect(new_rows, ref_rows, k, SUB, CLUSTER, label) for k in keys}
