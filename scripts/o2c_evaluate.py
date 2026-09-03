"""O2c evaluation — oracle integer-grid event-canonicalized MeanFlow against the frozen baseline B.

Frozen inference only: B, the R3 GTF-ORACLE secondary and the O2c checkpoint are loaded, never trained here.
The GT ECG R schedule is used ONLY to build the canonical coordinate and its inverse (oracle diagnostic).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

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

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import o1_targets as OT
from ppg2ecg.evaluation import o2_warp as O2W
from ppg2ecg.evaluation import o2b_warp as BW
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.evaluation import rpeaks as RP
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
O1ART = ROOT / "artifacts/o1_component_extractability"
O2BART = ROOT / "artifacts/o2b_integer_grid"
OUT = ROOT / "outputs/o2c_canon_oracle_seed42"


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m


R2E = _load("r2_evaluate", "scripts/r2_evaluate.py")
S0 = _load("o2_stage0_roundtrip", "scripts/o2_stage0_roundtrip.py")
O1E = _load("o1_evaluate", "scripts/o1_evaluate.py")

FS, T_LEN, BATCH, NFE, SRC_SEED = 128, 1024, 64, 4, 0
VAL = ("an0", "k2s")
ALIGNED = S0.ALIGNED
NAE = {t: f"nAE_{OT.TARGET_IDS[t]}" for t in ALIGNED}
SRC_BANK_SHA = "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"
BASELINE_TOL = 1e-6
FROZEN_B = {"f1": 0.4367299150733669, "chance_f1": 0.11916804674636086, "f1_excess": 0.31756186832700606,
            "missing": 0.5661580180073833, "spurious": 0.5153896464321268, "beats_ratio_dev": 0.10666361023416689}
EVENT_M = ("f1", "chance_f1", "f1_excess", "precision", "recall", "missing", "spurious", "beats_ratio_dev")
STRUCT_M = ("raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err", "qrs_e_dev", "p2p_dev", "hf_err")
HIGHER = {"f1", "f1_excess", "precision", "recall", "raw_corr"}
ARMS = ("B", "O2C-CANON-ORACLE", "GTF-ORACLE")


def orient(m):
    return "higher_better" if m in HIGHER else "lower_better"


def wcsv(p, rows):
    R2E.wcsv(p, rows)


def macro(vals, subs):
    v, s = np.asarray(vals, float), np.asarray(subs)
    return float(np.mean([np.nanmean(v[s == u]) for u in sorted(set(s.tolist()))]))


def aligned_rows(pred, gt_tg, iqr):
    tg = R2E.pmap(S0._targets, list(pred.astype(np.float64)))
    out = []
    for i in range(len(pred)):
        row = {}
        for t in ALIGNED:
            a, b = gt_tg[i][t], tg[i][t]
            row[NAE[t]] = float(abs(b - a) / iqr[t]) if np.isfinite(a) and np.isfinite(b) else np.nan
        out.append(row)
    return out


@torch.no_grad()
def gen_canonical(net, Xcan, bank, dev):
    outs, got = [], set()
    for i in range(0, len(Xcan), BATCH):
        pp = torch.from_numpy(np.ascontiguousarray(Xcan[i:i + BATCH])).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, pp, bank[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
        got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {NFE}
    return np.concatenate(outs)


def warp_block(A, warps, direction, dev, batch=512):
    out = np.empty_like(A)
    for i in range(0, len(A), batch):
        t = torch.from_numpy(np.ascontiguousarray(A[i:i + batch])).to(dev).unsqueeze(1)
        out[i:i + batch] = O2W.apply_warp(t, warps[i:i + batch], direction).squeeze(1).cpu().numpy()
    return out


def o2c_predict(net, Xcan, bank, warps, dev):
    can = gen_canonical(net, Xcan, bank, dev)
    return warp_block(can, warps, "to_raw", dev), can


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "figures").mkdir(exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    t_all = time.perf_counter()
    dev = torch.device("cuda")
    pt = subprocess.run([sys.executable, "-m", "pytest", "tests/test_o2c_oracle_integer_grid.py", "-o", "addopts=",
                         "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    if pt.returncode != 0:
        print(pt.stdout[-4000:]); raise RuntimeError("O2c tests fail; not evaluating")
    tests_summary = next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), "")
    print(f"[tests] {tests_summary}", flush=True)

    # ---------------- frozen population ----------------
    X, Y, SUB, SITE, POS, WI = S0.load_cohort()
    Yd = Y.astype(np.float64)
    gt_pk = R2E.pmap(S0._peaks, list(Yd))
    gt_tg = R2E.pmap(S0._targets, list(Yd))
    n_beats = int(sum(len(p) for p in gt_pk))
    if len(X) != 2048 or n_beats != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {n_beats} beats (STOP)")
    iqr = {t: json.loads((O1ART / "target_scaling.json").read_text())["targets"][t]["scale_train_IQR"] for t in ALIGNED}
    CLUSTER = np.array([f"{a}|{b}" for a, b in zip(SUB, WI)])
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    assert hashlib.sha256(e0.numpy().tobytes()).hexdigest() == SRC_BANK_SHA
    print(f"[P] {len(X)} windows, {n_beats} GT beats, {len(set(CLUSTER.tolist()))} ECG-window clusters", flush=True)

    # ---------------- frozen components ----------------
    _rn, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    ck_path = OUT / "checkpoint_final.pt"
    o2c_ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    o2c = MeanFlowS5(build_penguin_backbone(**o2c_ck["model_cfg"]),
                     cond_mode=o2c_ck["imf_cfg"]["cond_mode"], h_scale=o2c_ck["imf_cfg"]["h_scale"]).to(dev).eval()
    o2c.load_state_dict(o2c_ck["state_dict"]); o2c.requires_grad_(False)
    assert not any(p.requires_grad for p in o2c.parameters()) and not any(p.requires_grad for p in base.parameters())
    n_par = sum(p.numel() for p in o2c.backbone.parameters())
    assert n_par == gmeta["n_params_total"] == 4_568_707, n_par
    assert int(o2c_ck["step"]) == 10046, o2c_ck["step"]
    (ART / "frozen_component_manifest.json").write_text(json.dumps(
        {"B": {k: gmeta[k] for k in gmeta if k != "model_cfg"},
         "O2C-CANON-ORACLE": {"path": str(ck_path.relative_to(ROOT)), "step": int(o2c_ck["step"]),
                              "file_sha256": hashlib.sha256(ck_path.read_bytes()).hexdigest(),
                              "state_dict_sha256": o2c_ck.get("state_dict_sha256"), "n_params_total": int(n_par)},
         "operator": json.loads((ART / "operator_identity.json").read_text())}, indent=2, default=str))

    # ---------------- canonical coordinate ----------------
    warps = [BW.IntegerEventWarp(gt_pk[i]) for i in range(len(Y))]
    n_ident = int(sum(w.identity for w in warps))
    assert all(w.valid() or w.identity for w in warps)
    Xcan = warp_block(X, warps, "to_canonical", dev)
    Ycan = warp_block(Y, warps, "to_canonical", dev)

    # ---------------- 19. B regression gate ----------------
    p_b = R2E.gen_plain(base, X, e0, NFE, dev)
    rows_b, _a, _b = R2E.score(p_b, Yd, gt_pk)
    mac_b = R2E.macro_rows(rows_b, SUB)
    bad = {k: [mac_b[k], v] for k, v in FROZEN_B.items() if abs(mac_b[k] - v) > BASELINE_TOL}
    mx = max(abs(mac_b[k] - v) for k, v in FROZEN_B.items())
    (ART / "baseline_regression.json").write_text(json.dumps(
        {"tolerance": BASELINE_TOL, "expected_frozen": FROZEN_B, "reproduced": {k: mac_b[k] for k in FROZEN_B},
         "max_abs_diff": mx, "passed": not bad, "mismatches": bad}, indent=2))
    if bad:
        raise RuntimeError(f"BASELINE REGRESSION FAILED (STOP): {bad}")
    print(f"[B ] regression gate PASS (max |Δ| {mx:.3e})", flush=True)

    # ---------------- 20. O2c primary ----------------
    p_o2c, p_o2c_can = o2c_predict(o2c, Xcan, e0, warps, dev)
    rows_o, _c, _d = R2E.score(p_o2c, Yd, gt_pk)
    mac_o = R2E.macro_rows(rows_o, SUB)

    # ---------------- 27. GTF-ORACLE secondary (TARGET LEAKAGE DIAGNOSTIC) ----------------
    h_dim = int(ck["model_cfg"]["h_dim"])
    gpath = ROOT / f"outputs/r3_gtf_oracle_seed42/module_step{RF.STEPS}.pt"
    gmod_sd = torch.load(gpath, map_location="cpu", weights_only=False)
    assert gmod_sd["step"] == RF.STEPS and gmod_sd["arm"] == "gtf_oracle"
    assert gmod_sd["generator_state_sha256"] == gmeta["state_dict_sha256"]
    m = RF.build_r3_module("gtf", "adaptive", c_hidden=h_dim)
    gnet = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), m, cond_mode=cfg.get("cond_mode", "h_only"),
                               h_scale=cfg.get("h_scale", 1.0))
    missing, unexpected = gnet.load_state_dict(ck["state_dict"], strict=False)
    assert unexpected == [] and set(missing) == {"r3." + k for k in RF.FAMILY_PARAM_NAMES[m.family]}
    gnet.r3.load_state_dict({k: v.to(dev) for k, v in gmod_sd["state_dict"].items()})
    gnet.requires_grad_(False); gnet = gnet.to(dev).eval()
    s_oracle = RT.oracle_fields(Y, workers=12)
    outs, got = [], set()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
            sf = torch.from_numpy(s_oracle[i:i + BATCH]).to(dev).unsqueeze(1)
            z, k = ER.sample_meanflow_schedule(gnet, RT.make_ppg2(pp, sf), e0[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
            got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {NFE}
    p_g = np.concatenate(outs)
    rows_g, _e, _f = R2E.score(p_g, Yd, gt_pk)
    mac_g = R2E.macro_rows(rows_g, SUB)

    # ---------------- 21-22. metric tables ----------------
    ROWS = {"B": rows_b, "O2C-CANON-ORACLE": rows_o, "GTF-ORACLE": rows_g}
    MAC = {"B": mac_b, "O2C-CANON-ORACLE": mac_o, "GTF-ORACLE": mac_g}
    PRED = {"B": p_b, "O2C-CANON-ORACLE": p_o2c, "GTF-ORACLE": p_g}
    AL = {a: aligned_rows(PRED[a], gt_tg, iqr) for a in ARMS}
    ALM = {a: {t: macro([r[NAE[t]] for r in AL[a]], SUB) for t in ALIGNED} for a in ARMS}
    wcsv(ART / "event_metrics.csv", [{"arm": a, **{k: MAC[a][k] for k in EVENT_M}} for a in ARMS])
    wcsv(ART / "structure_metrics.csv", [{"arm": a, **{k: MAC[a][k] for k in STRUCT_M}} for a in ARMS])
    wcsv(ART / "o1_aligned_component_metrics.csv",
         [{"arm": a, **{NAE[t]: ALM[a][t] for t in ALIGNED},
           "note": "subject-macro normalised absolute error; O1 train IQR scaling"} for a in ARMS])
    for a in ARMS:
        print(f"[{a:<17}] f1x {MAC[a]['f1_excess']:+.4f} | " +
              " ".join(f"{NAE[t]} {ALM[a][t]:.4f}" for t in ALIGNED), flush=True)

    # ---------------- 23. paired clustered bootstrap (O2c - B) ----------------
    boot = []
    for m_ in EVENT_M + STRUCT_M:
        a = np.array([r[m_] for r in rows_b], float); b = np.array([r[m_] for r in rows_o], float)
        d = (b - a) if orient(m_) in ("higher_better",) else (a - b)
        boot.append({"contrast": "O2C_vs_B", "family": "event" if m_ in EVENT_M else "structure", "metric": m_,
                     "orientation": orient(m_), "positive_means": "O2c better",
                     "B": MAC["B"][m_], "O2C": MAC["O2C-CANON-ORACLE"][m_], **O1E.cluster_bootstrap(d, SUB, CLUSTER)})
    for t in ALIGNED:
        a = np.array([r[NAE[t]] for r in AL["B"]], float); b = np.array([r[NAE[t]] for r in AL["O2C-CANON-ORACLE"]], float)
        boot.append({"contrast": "O2C_vs_B", "family": "o1_aligned", "metric": NAE[t], "orientation": "lower_better",
                     "positive_means": "O2c better", "B": ALM["B"][t], "O2C": ALM["O2C-CANON-ORACLE"][t],
                     **O1E.cluster_bootstrap(a - b, SUB, CLUSTER)})
    wcsv(ART / "paired_bootstrap.csv", boot)
    BT = {r["metric"]: r for r in boot}

    # ---------------- 24. multi-source test (Q1 512-window subcohort) ----------------
    unc = Q.uncertainty_positions(SUB, SITE, WI)
    assert len(unc) == 512
    SUBu, CLu = SUB[unc], CLUSTER[unc]
    wu = [warps[i] for i in unc]
    ms_rows, U = [], {}
    for arm in ("B", "O2C-CANON-ORACLE"):
        S = []
        for sd in Q.UNC_SEEDS:
            bank = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(int(sd)))[unc]
            S.append(R2E.gen_plain(base, X[unc], bank, NFE, dev) if arm == "B"
                     else o2c_predict(o2c, Xcan[unc], bank, wu, dev)[0])
        S = np.stack(S)
        flat = R2E.pmap(R2E._peaks, [S[s, i].astype(np.float64) for i in range(S.shape[1]) for s in range(S.shape[0])])
        rows = [Q.uncertainty_from_samples(S[:, i], flat[i * S.shape[0]:(i + 1) * S.shape[0]], gt_pk[unc[i]])
                for i in range(S.shape[1])]
        U[arm] = rows
        ms_rows.append({"arm": arm, "n_windows": len(rows), "n_sources": len(Q.UNC_SEEDS),
                        "beat_count_SD": macro([r["u3_beatcount_sd"] for r in rows], SUBu),
                        "pairwise_event_F1_50": macro([r["u4_pairwise_event_f1_50"] for r in rows], SUBu),
                        "pairwise_event_F1_150": macro([r["u5_pairwise_event_f1_150"] for r in rows], SUBu),
                        "pointwise_waveform_SD": macro([r["u1_pointwise_sd"] for r in rows], SUBu),
                        "pairwise_waveform_RMSE": macro([r["u2_pairwise_rmse"] for r in rows], SUBu),
                        "gt_beat_timing_SD_ms": macro([r["u6_gt_beat_timing_sd_ms"] for r in rows], SUBu)})
        print(f"[MS {arm:<17}] beat-count SD {ms_rows[-1]['beat_count_SD']:.4f} "
              f"pairwise F1@50 {ms_rows[-1]['pairwise_event_F1_50']:.4f}", flush=True)
    wcsv(ART / "multisource_metrics.csv", ms_rows)
    s1 = O1E.cluster_bootstrap(np.array([r["u3_beatcount_sd"] for r in U["B"]], float) -
                               np.array([r["u3_beatcount_sd"] for r in U["O2C-CANON-ORACLE"]], float), SUBu, CLu)
    s2 = O1E.cluster_bootstrap(np.array([r["u4_pairwise_event_f1_50"] for r in U["O2C-CANON-ORACLE"]], float) -
                               np.array([r["u4_pairwise_event_f1_50"] for r in U["B"]], float), SUBu, CLu)
    wcsv(ART / "multisource_bootstrap.csv",
         [{"gate": "S1", "quantity": "beat-count SD (B - O2c)", "positive_means": "O2c lower (better)", **s1},
          {"gate": "S2", "quantity": "pairwise event F1@50 (O2c - B)", "positive_means": "O2c higher (better)", **s2}])

    # ---------------- 25. canonical-domain diagnostic (secondary) ----------------
    Ycd = Ycan.astype(np.float64)
    gt_pk_can = R2E.pmap(S0._peaks, list(Ycd))
    gt_tg_can = R2E.pmap(S0._targets, list(Ycd))
    rows_can, _g, _h = R2E.score(p_o2c_can, Ycd, gt_pk_can)
    mac_can = R2E.macro_rows(rows_can, SUB)
    al_can = aligned_rows(p_o2c_can, gt_tg_can, iqr)
    pk_can = R2E.pmap(S0._peaks, list(p_o2c_can.astype(np.float64)))
    f150 = []
    for i in range(len(X)):
        mm, fp, fn = RP.match_rpeaks(gt_pk_can[i], pk_can[i], FS, 150.0)
        f150.append(RP.prf(len(mm), fp, fn)[2])
    wcsv(ART / "canonical_domain_metrics.csv", [
        {"arm": "O2C canonical (before inverse warp)", "reference": "GT ECG warped to canonical coordinates",
         "f1_at_50": mac_can["f1"], "f1_excess": mac_can["f1_excess"], "f1_at_150": macro(f150, SUB),
         "raw_rmse": mac_can["raw_rmse"], "qrs_deriv_rmse": mac_can["qrs_deriv_rmse"],
         **{NAE[t]: macro([r[NAE[t]] for r in al_can], SUB) for t in ALIGNED}},
        {"arm": "O2C raw (after inverse warp)", "reference": "GT ECG in raw coordinates",
         "f1_at_50": mac_o["f1"], "f1_excess": mac_o["f1_excess"], "f1_at_150": "",
         "raw_rmse": mac_o["raw_rmse"], "qrs_deriv_rmse": mac_o["qrs_deriv_rmse"],
         **{NAE[t]: ALM["O2C-CANON-ORACLE"][t] for t in ALIGNED}}])

    # ---------------- 26. site-wise secondary ----------------
    site_rows = []
    for a in ARMS:
        for site in C.SITES:
            k = np.flatnonzero(SITE == site)
            site_rows.append({"arm": a, "site": site, "n": int(k.size),
                              "f1_excess": macro([ROWS[a][i]["f1_excess"] for i in k], SUB[k]),
                              "qrs_deriv_rmse": macro([ROWS[a][i]["qrs_deriv_rmse"] for i in k], SUB[k]),
                              **{NAE[t]: macro([AL[a][i][NAE[t]] for i in k], SUB[k]) for t in ALIGNED}})
    wcsv(ART / "site_metrics.csv", site_rows)
    wcsv(ART / "oracle_interface_comparison.csv",
         [{"arm": a, "oracle_interface": {"B": "none (deployable baseline)",
                                          "O2C-CANON-ORACLE": "GT R schedule as a time coordinate (train + inference)",
                                          "GTF-ORACLE": "GT R timing as a conditioning field (train + inference)"}[a],
           "label": "deployable baseline" if a == "B" else "TARGET LEAKAGE DIAGNOSTIC; not deployable",
           "f1_excess": MAC[a]["f1_excess"], "qrs_deriv_rmse": MAC[a]["qrs_deriv_rmse"],
           "qrs_curvature_err": MAC[a]["qrs_curvature_err"], **{NAE[t]: ALM[a][t] for t in ALIGNED}} for a in ARMS])

    # ---------------- operator floor (reported, never subtracted) ----------------
    o2b_rows = list(csv.DictReader(open(O2BART / "warp_roundtrip_metrics.csv")))
    def qs(k):
        v = np.array([float(r[k]) for r in o2b_rows if r[k] not in ("", "nan")], float)
        return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "p95": float(np.percentile(v, 95)), "max": float(v.max())}
    wcsv(ART / "operator_floor_summary.csv",
         [{"quantity": k, **qs(k), "source": "artifacts/o2b_integer_grid/warp_roundtrip_metrics.csv",
           "note": "operator round-trip floor; NEVER subtracted from any generator error"}
          for k in ("raw_rmse", "qrs_core_rmse", "nAE_T4", "nAE_T6", "nAE_T7", "nAE_T8")])

    # ---------------- 28-29. J/S gates and verdict ----------------
    def v(m_):
        return BT[m_]["verdict"]
    j = {"J1": bool(BT["f1_excess"]["lo"] > 0 and BT["f1_excess"]["point"] >= O2W.F1_EXCESS_MIN),
         "J2": bool(BT[NAE[S0.T6]]["lo"] > -O2W.NONINF_MARGIN),
         "J3": bool(BT[NAE[S0.T7]]["lo"] > -O2W.NONINF_MARGIN),
         "J4": bool(v(NAE[S0.T6]) == "improves" or v(NAE[S0.T7]) == "improves"),
         "J5": bool(v("qrs_deriv_rmse") != "worsens" and v("qrs_curvature_err") != "worsens"),
         "J6": bool(v(NAE[S0.T4]) != "worsens" and v(NAE[S0.T8]) != "worsens"),
         "J7": bool(s1["lo"] > 0 and s2["lo"] > 0),
         "morphology_improves": bool(v(NAE[S0.T6]) == "improves" and v(NAE[S0.T7]) == "improves")}
    dec = O2W.decide_o2(j)
    decision = {"verdict": dec["verdict"], "gates": dec["gates"], "morphology_improves": dec["morphology_improves"],
                "S1": s1, "S2": s2, "thresholds": {"F1_EXCESS_MIN": O2W.F1_EXCESS_MIN, "NONINF_MARGIN": O2W.NONINF_MARGIN},
                "J_detail": {k: {kk: BT[k][kk] for kk in ("point", "lo", "hi", "verdict", "B", "O2C")}
                             for k in ("f1_excess", NAE[S0.T4], NAE[S0.T6], NAE[S0.T7], NAE[S0.T8],
                                       "qrs_deriv_rmse", "qrs_curvature_err")},
                "oracle": "GT ECG R schedule builds the coordinate at training AND inference; O2c is NOT deployable",
                "status": "problem-discovery / mechanism diagnostic; one seed; two development-validation subjects",
                "secondary_analyses_cannot_alter_the_verdict": True}
    (ART / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    (ART / "provenance.json").write_text(json.dumps(
        {"git": git_sha(ROOT), "prereg": "d458895", "utc": datetime.now(timezone.utc).isoformat(),
         "test_subjects_loaded": [], "cohort": "frozen O2 2,048-window development cohort (an0/k2s, salt x4-event-nfe-v2)",
         "n_windows": int(len(X)), "n_gt_beats": n_beats, "n_clusters": int(len(set(CLUSTER.tolist()))),
         "n_identity_warps": n_ident, "nfe": NFE, "source_seed": SRC_SEED, "source_bank_sha256": SRC_BANK_SHA,
         "o1_train_iqr": {t: iqr[t] for t in ALIGNED}, "bootstrap": {"n": OT.BOOT_N, "seed": OT.BOOT_SEED},
         "tests": tests_summary, "gtf_oracle_module": str(gpath.relative_to(ROOT)),
         "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
         "gpu": torch.cuda.get_device_name(0), "wall_s": time.perf_counter() - t_all,
         "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20)}, indent=2, default=str))
    print("\n[GATES] " + " ".join(f"{k}:{'PASS' if x else 'FAIL'}" for k, x in dec["gates"].items()) +
          f" | S1 {'PASS' if s1['lo'] > 0 else 'FAIL'} S2 {'PASS' if s2['lo'] > 0 else 'FAIL'}", flush=True)
    print(f"[VERDICT] {dec['verdict']}", flush=True)
    print(f"[done] O2c evaluation in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
