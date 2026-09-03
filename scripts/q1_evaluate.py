"""Q1 evaluation — docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md.

FROZEN-INFERENCE ONLY. No training, no optimizer, no trainable parameter, no test subject.
Condition information is manipulated; the generator, the R1 probe and every metric are frozen.
Scoring is IMPORTED from scripts/r2_evaluate.py (the C0 pipeline verbatim).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import neurokit2
import numpy as np
import torch
from scipy.stats import spearmanr

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events  # noqa: F401
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/q1_conditional_support"
_spec = importlib.util.spec_from_file_location("r2_evaluate", ROOT / "scripts/r2_evaluate.py")
R2E = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = R2E; _spec.loader.exec_module(R2E)
FS, T_LEN, BATCH = 128, 1024, 64
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
NFE, SRC_SEED = Q.NFE_PRIMARY, Q.SRC_SEED
PREREG = "2cde60a"
R1_THRESHOLD = 0.35
R1_TOLS = (50.0, 100.0, 150.0, 200.0, 250.0)
RR_TOL_MS = 150.0

SUPPORT_M = ("r1_f1@50", "r1_f1@100", "r1_f1@150", "r1_f1@200", "r1_f1@250",
             "r1_missing", "r1_spurious", "r1_beats_ratio_dev", "r1_rr_mae_ms", "r1_rr_median_ae_ms")
FIDELITY_M = ("f1", "chance_f1", "f1_excess", "precision", "recall", "missing", "spurious", "beats_ratio_dev",
              "raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err", "qrs_e_dev", "p2p_dev", "hf_err")
PLAUS_M = ("detector_valid", "marginal_support_fraction") + tuple(f"in_support_{f}" for f in Q.PLAUS_FEATURES)
UNC_M = ("u1_pointwise_sd", "u2_pairwise_rmse", "u3_beatcount_sd", "u4_pairwise_event_f1_50", "u5_pairwise_event_f1_150", "u6_gt_beat_timing_sd_ms")

ORIENT = dict(R2E.ORIENT)
ORIENT |= {m: "higher_better" for m in ("r1_f1@50", "r1_f1@100", "r1_f1@150", "r1_f1@200", "r1_f1@250")}
ORIENT |= {m: "lower_better" for m in ("r1_missing", "r1_spurious", "r1_beats_ratio_dev", "r1_rr_mae_ms", "r1_rr_median_ae_ms",
                                       "qrs_e_dev", "p2p_dev", "hf_err")}
ORIENT |= {m: "higher_better" for m in PLAUS_M}
ORIENT |= {"f1": "higher_better", "chance_f1": "higher_better"}
UNC_ORIENT = {"u1_pointwise_sd": "higher_better", "u2_pairwise_rmse": "higher_better", "u3_beatcount_sd": "higher_better",
              "u4_pairwise_event_f1_50": "lower_better", "u5_pairwise_event_f1_150": "lower_better",
              "u6_gt_beat_timing_sd_ms": "higher_better"}


# ------------------------------------------------------------------ small helpers
def wcsv(p, rows):
    R2E.wcsv(p, rows)


def sha_block(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def macro(vals, sub):
    return S1.macro(np.asarray(vals, float), np.asarray(sub))


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    return o


def _peaks_np(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), FS)


def _feat_one(sig):
    return Q.ecg_features(sig, FS)


def _quality_one(args):
    x = np.asarray(args, dtype=np.float64)
    pk = S1.dsp_ppg_peaks(x, FS)
    return {"periodicity_score": Q.periodicity_score(x), "pulse_template_consistency": Q.pulse_template_consistency(x, pk),
            "n_pulses": int(len(pk))}


def _ppg_peaks_one(sig):
    return S1.dsp_ppg_peaks(np.asarray(sig, dtype=np.float64), FS)


# ------------------------------------------------------------------ frozen population
def load_population(subjects=VAL, salt=SALT, take=TAKE, assert_frozen=True):
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        idx = ER.select_subset(salt, s, len(Xs), take)
        X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); SITE.append(Ss[idx]); POS.append(idx); WI.append(Ws[idx].astype(np.int64))
    X, Y, SUB, SITE, POS, WI = (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))
    if assert_frozen:
        frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
        for s in subjects:
            assert POS[SUB == s].tolist() == list(frozen[s]), f"frozen subset mismatch {s}"
    return X, Y, SUB, SITE, POS, WI


def load_r1_validation_cohort():
    """The frozen R1 8,192-window validation cohort (an0/k2s, 1,024 per subject x site)."""
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in C.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        pos = C.cohort_positions(s, Ss, Ws, C.n_per_for(s))
        for site in C.SITES:
            idx = pos[site]
            X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32))
            SUB.append(np.full(len(idx), s)); SITE.append(np.full(len(idx), site)); POS.append(idx); WI.append(Ws[idx].astype(np.int64))
    return (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))


# ------------------------------------------------------------------ R1 support probe
@torch.no_grad()
def r1_probs(tcn, X, dev, batch=512):
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.ascontiguousarray(X[i:i + batch])).to(dev).unsqueeze(1)
        out.append(torch.sigmoid(tcn(x, None)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def _events_one(p):
    return extract_events(p, R1_THRESHOLD)


def support_rows(events, gt_pk):
    rows = []
    for k, (ev, g) in enumerate(zip(events, gt_pk)):
        row = {}
        for tol in R1_TOLS:
            m, fp, fn = R.match_rpeaks(g, ev, FS, tol_ms=tol)
            p, r, f = R.prf(len(m), fp, fn)
            row[f"r1_f1@{int(tol)}"] = f
            if tol == 50.0:
                row["r1_precision@50"], row["r1_recall@50"] = p, r
            if tol == RR_TOL_MS:
                n_ref = max(len(g), 1)
                row |= {"r1_missing": fn / n_ref, "r1_spurious": fp / n_ref,
                        "r1_beats_ratio_dev": abs(len(ev) / n_ref - 1.0), "r1_n_pred": int(len(ev)), "r1_n_ref": int(len(g))}
                mm = dict(m)
                ae = [abs(((g[i + 1] - g[i]) - (ev[mm[i + 1]] - ev[mm[i]])) / FS * 1000.0)
                      for i in range(len(g) - 1) if i in mm and i + 1 in mm]
                row |= {"r1_rr_mae_ms": float(np.mean(ae)) if ae else np.nan,
                        "r1_rr_median_ae_ms": float(np.median(ae)) if ae else np.nan, "r1_n_rr": len(ae)}
        rows.append(row)
    return rows


# ------------------------------------------------------------------ paired effects
def paired_rows(rows_c, rows_x, sub, metrics, orient_map, condition, axis, family):
    out = []
    for m in metrics:
        a = np.asarray([r.get(m, np.nan) for r in rows_x], float)   # corrupted
        b = np.asarray([r.get(m, np.nan) for r in rows_c], float)   # clean
        if not np.isfinite(a).any() or not np.isfinite(b).any():
            continue
        if axis == "UNCERTAINTY":
            res = paired_subject_bootstrap(b, a, sub, orient_map[m], n_boot=Q.BOOT_N, seed=Q.BOOT_SEED)
            desc = "positive = corrupted more uncertain/diverse than clean"
        else:
            res = paired_subject_bootstrap(a, b, sub, orient_map[m], n_boot=Q.BOOT_N, seed=Q.BOOT_SEED)
            desc = "positive = clean better than corrupted"
        n_fin = int(np.sum(np.isfinite(a) & np.isfinite(b)))
        out.append({"axis": axis, "family": family, "condition": condition, "metric": m,
                    "orientation": orient_map[m], "positive_means": desc,
                    "clean_macro": macro(b, sub), "corrupted_macro": macro(a, sub),
                    "n_finite_pairs": n_fin, **res})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "primary", "natural", "secondary"])
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    git = git_sha(ROOT)
    prov = {"git": git, "prereg": PREREG, "prereg_doc": "docs/Q1_CONDITIONAL_SUPPORT_DEGRADATION_PREREGISTRATION.md",
            "utc_start": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
            "libs": {"torch": torch.__version__, "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                     "neurokit2": neurokit2.__version__, "python": platform.python_version()},
            "gpu": torch.cuda.get_device_name(0), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "no_optimizer": True, "no_trainable_parameters": True,
            "conditions": list(Q.CONDITIONS), "nfe": NFE, "source_seed": SRC_SEED}
    if not args.skip_tests:
        pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_q1_conditional_support.py", "-o", "addopts=", "-q",
                             "-rs", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
        prov["tests_ran"] = {"exit": pt.returncode,
                             "summary": next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")}
        if pt.returncode != 0:
            print(pt.stdout[-4000:]); raise RuntimeError("Q1 tests fail; not evaluating")
        print(f"[tests] {prov['tests_ran']['summary']}", flush=True)

    # ---------------- population ----------------
    X, Y, SUB, SITE, POS, WI = load_population()
    Yd = Y.astype(np.float64)
    gt_pk = R2E.pmap(R2E._peaks, list(Yd))
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()
    assert src_hash == "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f", src_hash
    n_beats = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_beats != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {n_beats} GT beats (STOP)")
    prov |= {"population_windows": int(len(X)), "gt_beats": n_beats, "source_bank_sha256": src_hash}
    print(f"[P] {len(X)} windows, {n_beats} GT beats | HEAD {git['commit'][:8]}", flush=True)

    # ---------------- frozen components ----------------
    _net, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    assert not any(p.requires_grad for p in base.parameters()) and not any(p.requires_grad for p in tcn.parameters())
    (ART / "checkpoint_manifest.json").write_text(json.dumps({"generator": gmeta, "rhythm_tcn": tmeta,
        "a4_generator_md5": hashlib.md5((ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt").read_bytes()).hexdigest()}, indent=2, default=str))
    prov |= {"generator": gmeta, "rhythm_tcn": tmeta}

    # ---------------- cohorts ----------------
    partner = RT.shuffle_partner(SUB, SITE, WI, salt=Q.SHUFFLE_SALT)
    RT.assert_derangement(partner)
    unc_idx = Q.uncertainty_positions(SUB, SITE, WI)
    assert len(unc_idx) == 512 and len(set(unc_idx.tolist())) == 512
    wcsv(ART / "cohort_manifest.csv", [{"row": i, "subject": SUB[i], "site": SITE[i], "array_pos": int(POS[i]),
                                        "window_index": int(WI[i]), "shuffle_partner_row": int(partner[i]),
                                        "in_uncertainty_cohort": int(i in set(unc_idx.tolist()))} for i in range(len(X))])
    wcsv(ART / "uncertainty_cohort.csv", [{"row": int(i), "subject": SUB[i], "site": SITE[i], "array_pos": int(POS[i]),
                                           "window_index": int(WI[i])} for i in unc_idx])

    # ---------------- corruption + sanity ----------------
    t0 = time.perf_counter()
    XC = {c: (X.astype(np.float32).copy() if c == Q.CLEAN else Q.corrupt_block(X, c, SUB, SITE, WI, partner)) for c in Q.CONDITIONS}
    man, san = [], []
    clean_pulse = R2E.pmap(_ppg_peaks_one, list(X.astype(np.float64)))
    for c in Q.CONDITIONS:
        man.append({"condition": c, "family": Q.FAMILY_OF.get(c, "CONTROL" if c in (Q.SHUFFLED, Q.NULL) else "CLEAN"),
                    "sha256": sha_block(XC[c]), "n_windows": int(len(XC[c])), "renormalised": False,
                    "min": float(XC[c].min()), "max": float(XC[c].max())})
        pk_c = R2E.pmap(_ppg_peaks_one, list(XC[c].astype(np.float64)))
        for i in range(len(X)):
            s = Q.ppg_sanity(X[i], XC[c][i])
            s |= {"condition": c, "row": i, "subject": SUB[i], "site": SITE[i], "n_pulses": int(len(pk_c[i])),
                  "n_pulses_clean": int(len(clean_pulse[i])),
                  "pulse_interval_mae_ms": Q.pulse_interval_mae_ms(clean_pulse[i], pk_c[i]),
                  "achieved_snr_db": Q.achieved_snr_db(X[i], XC[c][i]) if c in Q.SNR_CONDS else np.nan}
            san.append(s)
        print(f"[corrupt] {c:<10} sha {man[-1]['sha256'][:12]} corr {np.nanmedian([r['ppg_corr'] for r in san if r['condition']==c]):+.3f}", flush=True)
    wcsv(ART / "corruption_manifest.csv", man)
    wcsv(ART / "corruption_sanity.csv", san)
    prov["corruption_wall_s"] = time.perf_counter() - t0

    # monotonicity gate
    def med(c, k):
        return float(np.nanmedian([r[k] for r in san if r["condition"] == c]))
    mono = {}
    for fam, conds in Q.FAMILIES.items():
        corrs = [med(c, "ppg_corr") for c in conds]
        nrm = [med(c, "ppg_nrmse") for c in conds]
        ok = all(corrs[i] >= corrs[i + 1] - 1e-12 for i in range(len(corrs) - 1)) and all(nrm[i] <= nrm[i + 1] + 1e-12 for i in range(len(nrm) - 1))
        mono[fam] = {"ppg_corr_by_level": corrs, "ppg_nrmse_by_level": nrm, "monotonic": bool(ok)}
    prov["monotonicity_gate"] = mono
    if not all(v["monotonic"] for v in mono.values()):
        (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        raise RuntimeError(f"corruption severity is not monotonic (STOP): {mono}")
    print("[sanity] monotonicity gate PASS " + " ".join(f"{k}:{v['monotonic']}" for k, v in mono.items()), flush=True)

    # ---------------- R1 support sweep ----------------
    t0 = time.perf_counter()
    sup, FIELD = {}, {}
    for c in Q.CONDITIONS:
        FIELD[c] = R2E.scaffolds(tcn, XC[c], dev)          # sigmoid field, R2/R3 batching (MICRO_BATCH 32)
        ev = R2E.pmap(_events_one, list(FIELD[c]))
        sup[c] = support_rows(ev, gt_pk)
        print(f"[R1] {c:<10} f1@150 {macro([r['r1_f1@150'] for r in sup[c]], SUB):.4f} rr {macro([r['r1_rr_mae_ms'] for r in sup[c]], SUB):7.2f} ms", flush=True)
    wcsv(ART / "r1_support_metrics.csv",
         [{"condition": c, "row": i, "subject": SUB[i], "site": SITE[i], **{k: v for k, v in sup[c][i].items()}}
          for c in Q.CONDITIONS for i in range(len(X))])
    prov["r1_support_wall_s"] = time.perf_counter() - t0

    # ---------------- arm-B conditional fidelity ----------------
    t0 = time.perf_counter()
    fid, preds = {}, {}
    for c in Q.CONDITIONS:
        p = R2E.gen_plain(base, XC[c], e0, NFE, dev)
        rows, _pk, _errs = R2E.score(p, Yd, gt_pk)
        fid[c] = rows; preds[c] = p
        print(f"[B ] {c:<10} f1x {macro([r['f1_excess'] for r in rows], SUB):+.4f} qrs {macro([r['raw_qrs_rmse'] for r in rows], SUB):.4f} "
              f"d {macro([r['qrs_deriv_rmse'] for r in rows], SUB):.4f}", flush=True)
    wcsv(ART / "generator_fidelity_metrics.csv",
         [{"arm": "B", "condition": c, "row": i, "subject": SUB[i], "site": SITE[i], **fid[c][i]} for c in Q.CONDITIONS for i in range(len(X))])
    prov["fidelity_wall_s"] = time.perf_counter() - t0

    # ---------------- marginal plausibility ----------------
    t0 = time.perf_counter()
    ref_rows, ref_meta = [], []
    for s in Q.TRAIN12:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Ss, Ws = np.asarray(d["site"]).astype(str), d["window_index"]
        idx = Q.reference_positions(s, Ss, Ws)
        ref_rows += R2E.pmap(_feat_one, list(d["y"][idx].astype(np.float64)))
        ref_meta.append({"subject": s, "n": int(len(idx))})
    ref = Q.reference_intervals([r for r in ref_rows if r["detector_valid"]])
    ref_json = {"cohort": {"salt": Q.REFERENCE_SALT, "subjects": list(Q.TRAIN12), "per_stratum": Q.N_REFERENCE_PER_STRATUM,
                           "n_windows": int(len(ref_rows)), "per_subject": ref_meta},
                "detector_valid_fraction_train_real": float(np.mean([r["detector_valid"] for r in ref_rows])),
                "percentiles": Q.PLAUS_PCTS, "intervals": ref}
    (ART / "marginal_plausibility_reference.json").write_text(json.dumps(ref_json, indent=2, default=str))
    print(f"[REF] {len(ref_rows)} train windows, detector-valid {ref_json['detector_valid_fraction_train_real']:.4f}, "
          + " ".join(f"{k}[{v['p_lo']:.3g},{v['p_hi']:.3g}]" for k, v in ref.items()), flush=True)
    plaus = {}
    for c in Q.CONDITIONS:
        feats = R2E.pmap(_feat_one, list(preds[c].astype(np.float64)))
        plaus[c] = [Q.support_indicators(f, ref) | {"n_peaks": f["n_peaks"]} for f in feats]
        print(f"[PL] {c:<10} det {macro([r['detector_valid'] for r in plaus[c]], SUB):.4f} "
              f"marg {macro([r['marginal_support_fraction'] for r in plaus[c]], SUB):.4f}", flush=True)
    wcsv(ART / "marginal_plausibility_metrics.csv",
         [{"arm": "B", "condition": c, "row": i, "subject": SUB[i], "site": SITE[i], **plaus[c][i]} for c in Q.CONDITIONS for i in range(len(X))])
    # real-ECG reference row for context (GT of the primary population, same features)
    gt_feats = R2E.pmap(_feat_one, list(Yd))
    gt_ind = [Q.support_indicators(f, ref) for f in gt_feats]
    prov["gt_primary_marginal_support"] = {"detector_valid": macro([r["detector_valid"] for r in gt_ind], SUB),
                                           "marginal_support_fraction": macro([r["marginal_support_fraction"] for r in gt_ind], SUB)}
    prov["plausibility_wall_s"] = time.perf_counter() - t0

    # ---------------- multi-source uncertainty ----------------
    t0 = time.perf_counter()
    Xu = {c: XC[c][unc_idx] for c in Q.CONDITIONS}
    gt_pk_u = [gt_pk[i] for i in unc_idx]
    SUBu = SUB[unc_idx]
    banks = {sd: torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(int(sd)))[unc_idx] for sd in Q.UNC_SEEDS}
    assert torch.equal(banks[0], e0[unc_idx]), "seed-0 uncertainty bank must equal the primary source bank rows"
    unc = {}
    for c in Q.CONDITIONS:
        S = np.stack([R2E.gen_plain(base, Xu[c], banks[sd], NFE, dev) for sd in Q.UNC_SEEDS])   # [8, 512, T]
        pk = R2E.pmap(_peaks_np, [S[s, i] for i in range(S.shape[1]) for s in range(S.shape[0])])
        rows = []
        for i in range(S.shape[1]):
            pks = pk[i * S.shape[0]:(i + 1) * S.shape[0]]
            rows.append(Q.uncertainty_from_samples(S[:, i], pks, gt_pk_u[i]))
        unc[c] = rows
        print(f"[U ] {c:<10} sd {macro([r['u1_pointwise_sd'] for r in rows], SUBu):.4f} "
              f"bc {macro([r['u3_beatcount_sd'] for r in rows], SUBu):.3f} f1pair {macro([r['u4_pairwise_event_f1_50'] for r in rows], SUBu):.4f}", flush=True)
    wcsv(ART / "uncertainty_metrics.csv",
         [{"arm": "B", "condition": c, "row": int(unc_idx[i]), "subject": SUBu[i], "site": SITE[unc_idx][i], **unc[c][i]}
          for c in Q.CONDITIONS for i in range(len(unc_idx))])
    prov["uncertainty_wall_s"] = time.perf_counter() - t0

    # ---------------- paired effects ----------------
    boot = []
    for c in Q.CONDITIONS:
        if c == Q.CLEAN:
            continue
        fam = Q.FAMILY_OF.get(c, "CONTROL")
        boot += paired_rows(sup[Q.CLEAN], sup[c], SUB, SUPPORT_M, ORIENT, c, "SUPPORT", fam)
        boot += paired_rows(fid[Q.CLEAN], fid[c], SUB, FIDELITY_M, ORIENT, c, "FIDELITY", fam)
        boot += paired_rows(plaus[Q.CLEAN], plaus[c], SUB, PLAUS_M, ORIENT, c, "PLAUSIBILITY", fam)
        boot += paired_rows(unc[Q.CLEAN], unc[c], SUBu, UNC_M, UNC_ORIENT, c, "UNCERTAINTY", fam)
    wcsv(ART / "paired_bootstrap.csv", boot)
    B = {(r["condition"], r["axis"], r["metric"]): r for r in boot}

    # ---------------- support-fidelity coupling ----------------
    cor = []
    pairs = (("r1_f1@150", "f1_excess"), ("r1_rr_mae_ms", "beats_ratio_dev"), ("r1_f1@150", "raw_qrs_rmse"))
    for fam, conds in list(Q.FAMILIES.items()) + [("ALL", [c for c in Q.NATURAL_CONDITIONS if c != Q.CLEAN])]:
        for xk, yk in pairs:
            xs = np.concatenate([[r[xk] for r in sup[c]] for c in conds])
            ys = np.concatenate([[r[yk] for r in fid[c]] for c in conds])
            ss = np.concatenate([SUB for _ in conds])
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys, ss = xs[ok], ys[ok], ss[ok]
            rho = float(spearmanr(xs, ys).statistic)
            uniq = sorted(set(ss.tolist())); idx = {u: np.flatnonzero(ss == u) for u in uniq}
            rng = np.random.default_rng(Q.BOOT_SEED)
            draws = np.array([float(np.mean([spearmanr(xs[j], ys[j]).statistic for j in
                                             (rng.choice(idx[u], idx[u].size, replace=True) for u in uniq)])) for _ in range(200)])
            cor.append({"family": fam, "x": xk, "y": yk, "spearman_rho": rho, "n": int(xs.size),
                        "lo": float(np.nanpercentile(draws, 2.5)), "hi": float(np.nanpercentile(draws, 97.5)),
                        "n_boot": 200, "seed": Q.BOOT_SEED, "note": "association only; not causal"})
    wcsv(ART / "support_fidelity_correlations.csv", cor)

    # ---------------- preregistered verdict ----------------
    flags = {}
    for fam, sev in Q.SEVERE.items():
        support = {"f1@150": B[(sev, "SUPPORT", "r1_f1@150")], "rr_mae_ms": B[(sev, "SUPPORT", "r1_rr_mae_ms")],
                   "missing": B[(sev, "SUPPORT", "r1_missing")], "spurious": B[(sev, "SUPPORT", "r1_spurious")]}
        fidelity = {k: B[(sev, "FIDELITY", k)] for k in ("f1_excess", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err")}
        det_drop = B[(sev, "PLAUSIBILITY", "detector_valid")]["clean_macro"] - B[(sev, "PLAUSIBILITY", "detector_valid")]["corrupted_macro"]
        mar_drop = B[(sev, "PLAUSIBILITY", "marginal_support_fraction")]["clean_macro"] - B[(sev, "PLAUSIBILITY", "marginal_support_fraction")]["corrupted_macro"]
        u1c = B[(sev, "UNCERTAINTY", "u1_pointwise_sd")]["clean_macro"]; u1x = B[(sev, "UNCERTAINTY", "u1_pointwise_sd")]["corrupted_macro"]
        unc_in = {"u1_rel_increase": (u1x - u1c) / max(abs(u1c), 1e-12),
                  "u3": B[(sev, "UNCERTAINTY", "u3_beatcount_sd")], "u4": B[(sev, "UNCERTAINTY", "u4_pairwise_event_f1_50")]}
        flags[fam] = Q.family_flags(support, fidelity, {"detector_valid_drop": det_drop, "marginal_support_drop": mar_drop}, unc_in) | {
            "severe_condition": sev, "detector_valid_drop": det_drop, "marginal_support_drop": mar_drop,
            "u1_clean": u1c, "u1_corrupted": u1x, "u1_rel_increase": unc_in["u1_rel_increase"]}
    dec = Q.decide_q1({f: v for f, v in flags.items()})
    decision = {"prereg": PREREG, "verdict": dec["verdict"], "detail": dec, "family_flags": flags,
                "thresholds": {"det_valid_drop_max": Q.DET_VALID_DROP_MAX, "marginal_drop_max": Q.MARGINAL_DROP_MAX,
                               "uncertainty_rel_increase": Q.UNC_REL_INCREASE},
                "terminology": "conditional-support / plausibility decoupling (never 'hallucination')",
                "status": "exploratory / problem-discovery; not independent confirmation",
                "controls_excluded_from_verdict": [Q.SHUFFLED, Q.NULL]}
    (ART / "decision.json").write_text(json.dumps(jsafe(decision), indent=2))
    print(f"\n[VERDICT] {dec['verdict']}  A={dec['families_A']} B={dec['families_B']} C={dec['families_C']}\n", flush=True)

    # ---------------- exploratory: natural PPG quality audit (does NOT enter the verdict) ----------------
    if args.stage in ("all", "natural"):
        t0 = time.perf_counter()
        Xn, Yn, SUBn, SITEn, POSn, WIn = load_r1_validation_cohort()
        Ynd = Yn.astype(np.float64)
        gt_pk_n = R2E.pmap(R2E._peaks, list(Ynd))
        qual = R2E.pmap(_quality_one, list(Xn.astype(np.float64)))
        fld_n = R2E.scaffolds(tcn, Xn, dev)
        ev_n = R2E.pmap(_events_one, list(fld_n))
        sup_n = support_rows(ev_n, gt_pk_n)
        e_n = torch.randn(len(Xn), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
        prov["natural_bank_sha256"] = hashlib.sha256(e_n.numpy().tobytes()).hexdigest()
        pred_n = R2E.gen_plain(base, Xn, e_n, NFE, dev)
        fid_n, _pk, _e = R2E.score(pred_n, Ynd, gt_pk_n)
        feats_n = R2E.pmap(_feat_one, list(pred_n.astype(np.float64)))
        plaus_n = [Q.support_indicators(f, ref) for f in feats_n]
        nat = [{"row": i, "subject": SUBn[i], "site": SITEn[i], "array_pos": int(POSn[i]), "window_index": int(WIn[i]),
                **qual[i], **sup_n[i], **{k: v for k, v in fid_n[i].items() if k in FIDELITY_M},
                "detector_valid": plaus_n[i]["detector_valid"], "marginal_support_fraction": plaus_n[i]["marginal_support_fraction"]}
               for i in range(len(Xn))]
        wcsv(ART / "natural_quality_metrics.csv", nat)
        REPORT_M = ("r1_f1@150", "r1_rr_mae_ms", "f1_excess", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err",
                    "marginal_support_fraction", "detector_valid")
        qrows = []
        for score_name in ("periodicity_score", "pulse_template_consistency"):
            q_of = np.full(len(nat), -1, dtype=int)
            for sub in sorted(set(SUBn.tolist())):
                for site in sorted(set(SITEn[SUBn == sub].tolist())):
                    m = np.flatnonzero((SUBn == sub) & (SITEn == site))
                    v = np.asarray([nat[i][score_name] for i in m], float)
                    ok = m[np.isfinite(v)]
                    vv = v[np.isfinite(v)]
                    if vv.size < 8:
                        continue
                    edges = np.percentile(vv, [25, 50, 75])
                    q_of[ok] = np.searchsorted(edges, vv, side="right")
            for q in range(4):
                sel = np.flatnonzero(q_of == q)
                if sel.size == 0:
                    continue
                row = {"score": score_name, "quartile": q + 1, "n": int(sel.size),
                       "score_mean": float(np.nanmean([nat[i][score_name] for i in sel])),
                       "n_undefined": int(np.sum(q_of == -1))}
                for m_ in REPORT_M:
                    row[m_] = macro([nat[i][m_] for i in sel], SUBn[sel])
                qrows.append(row)
        wcsv(ART / "natural_quality_quartiles.csv", qrows)
        prov["natural_quality"] = {"cohort": "R1 validation 8,192 (an0/k2s, 1,024 per subject x site)",
                                   "n_windows": int(len(Xn)), "wall_s": time.perf_counter() - t0,
                                   "undefined_template_fraction": float(np.mean([not np.isfinite(r["pulse_template_consistency"]) for r in nat])),
                                   "uncertainty_available": False, "status": "exploratory; not part of the verdict"}
        print(f"[NAT] 8,192-window quality audit in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    # ---------------- secondary: frozen R3 GTF-TRUE (stronger target-derived event supervision) ----------------
    if args.stage in ("all", "secondary"):
        t0 = time.perf_counter()
        h_dim = int(ck["model_cfg"]["h_dim"])
        gmod_sd = torch.load(ROOT / f"outputs/r3_gtf_true_seed42/module_step{RF.STEPS}.pt", map_location="cpu", weights_only=False)
        assert gmod_sd["step"] == RF.STEPS and gmod_sd["arm"] == "gtf_true" and gmod_sd["generator_state_sha256"] == gmeta["state_dict_sha256"]
        m = RF.build_r3_module("gtf", "adaptive", c_hidden=h_dim)
        gnet = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), m, cond_mode=cfg.get("cond_mode", "h_only"),
                                   h_scale=cfg.get("h_scale", 1.0))
        missing, unexpected = gnet.load_state_dict(ck["state_dict"], strict=False)
        assert unexpected == [] and set(missing) == {"r3." + k for k in RF.FAMILY_PARAM_NAMES[m.family]}
        gnet.r3.load_state_dict({k: v.to(dev) for k, v in gmod_sd["state_dict"].items()})
        gnet.requires_grad_(False); gnet = gnet.to(dev).eval()
        assert not any(p.requires_grad for p in gnet.parameters())
        prov["secondary_gtf"] = {"module": f"outputs/r3_gtf_true_seed42/module_step{RF.STEPS}.pt",
                                 "module_file_sha256": hashlib.sha256((ROOT / f"outputs/r3_gtf_true_seed42/module_step{RF.STEPS}.pt").read_bytes()).hexdigest(),
                                 "label": "GTF-TRUE (rhythm-augmented; scaffold from a probe trained on GT-R labels)"}
        rows_out = []
        for c in Q.CONDITIONS:
            outs, got = [], set()
            with torch.no_grad():
                for i in range(0, len(X), BATCH):
                    pp = torch.from_numpy(XC[c][i:i + BATCH]).to(dev).unsqueeze(1)
                    sf = torch.from_numpy(FIELD[c][i:i + BATCH]).to(dev).unsqueeze(1)
                    z, k = ER.sample_meanflow_schedule(gnet, RT.make_ppg2(pp, sf), e0[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
                    got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
            assert got == {NFE}
            pg = np.concatenate(outs)
            rows, _pk, _e = R2E.score(pg, Yd, gt_pk)
            fg = R2E.pmap(_feat_one, list(pg.astype(np.float64)))
            pl = [Q.support_indicators(f, ref) for f in fg]
            rows_out += [{"arm": "GTF-TRUE", "condition": c, "row": i, "subject": SUB[i], "site": SITE[i], **rows[i],
                          "detector_valid": pl[i]["detector_valid"], "marginal_support_fraction": pl[i]["marginal_support_fraction"]}
                         for i in range(len(X))]
            print(f"[G ] {c:<10} f1x {macro([r['f1_excess'] for r in rows], SUB):+.4f} d {macro([r['qrs_deriv_rmse'] for r in rows], SUB):.4f} "
                  f"marg {macro([r['marginal_support_fraction'] for r in pl], SUB):.4f}", flush=True)
        wcsv(ART / "secondary_gtf_metrics.csv", rows_out)
        # paired clean -> severe for the secondary arm (reported, never part of the verdict)
        gsec = {c: [r for r in rows_out if r["condition"] == c] for c in Q.CONDITIONS}
        gboot = []
        for c in Q.CONDITIONS:
            if c == Q.CLEAN:
                continue
            gboot += paired_rows(gsec[Q.CLEAN], gsec[c], SUB, ("f1_excess", "missing", "spurious", "qrs_deriv_rmse",
                                                               "qrs_curvature_err", "marginal_support_fraction"),
                                 ORIENT | {"marginal_support_fraction": "higher_better"}, c, "FIDELITY_SECONDARY",
                                 Q.FAMILY_OF.get(c, "CONTROL"))
        wcsv(ART / "secondary_gtf_paired.csv", gboot)
        prov["secondary_gtf"]["wall_s"] = time.perf_counter() - t0
        print(f"[G ] secondary GTF-TRUE in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    prov |= {"wall_s": time.perf_counter() - t_all, "utc_end": datetime.now(timezone.utc).isoformat(),
             "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[done] Q1 primary in {(time.perf_counter()-t_all)/60:.1f} min, peak {prov['peak_mem_MiB']:.0f} MiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
