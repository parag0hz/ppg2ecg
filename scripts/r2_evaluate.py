"""R2 evaluation — docs/R2_RHYTHM_SCAFFOLD_TRANSFER_PREREGISTRATION.md (f954e07) sections 11-22.

Primary population: the frozen C0/C1 2,048-window development subset, seed-0 source bank. Arms B / TRUE /
SHUFFLE / ORACLE at NFE 1, 2, 4 (+ the +256-sample phase ablation of the TRUE adapter at NFE 4). The event
pipeline is the C0 one copied verbatim; structure metrics are the frozen C0 / M1 definitions imported from
the evaluation modules. Secondary: site-wise B vs TRUE on the R1 8,192-window validation cohort.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import neurokit2
import numpy as np
import torch

from ppg2ecg.evaluation import alignment_diagnostics as AD
from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r2_rhythm_transfer"
FS, T_LEN, BATCH, WORKERS = 128, 1024, 64, 12
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
NFES, SRC_SEED = (1, 2, 4), 0
PREREG = "f954e07"
PRIMARY = [                                                       # C0 PRIMARY, verbatim
    ("raw_corr",      "raw_corr",             np.nanmean,   "higher_better", False),
    ("qrs_e_dev",     "raw_qrs_energy_ratio", np.nanmedian, "lower_better",  True),
    ("slope_dev",     "raw_slope_ratio",      np.nanmedian, "lower_better",  True),
    ("p2p_dev",       "raw_p2p_ratio",        np.nanmedian, "lower_better",  True),
    ("raw_qrs_rmse",  "raw_qrs_rmse",         np.nanmean,   "lower_better",  False),
    ("raw_rmse",      "raw_rmse",             np.nanmean,   "lower_better",  False),
]
RATIO_RAW = {"qrs_e_dev": "raw_qrs_energy_ratio", "slope_dev": "raw_slope_ratio", "p2p_dev": "raw_p2p_ratio"}
EVENT4 = ("f1_excess", "beats_ratio_dev", "missing", "spurious")
STRUCT5 = ("raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err")           # S1..S5
ORIENT = {"f1_excess": "higher_better", "precision": "higher_better", "recall": "higher_better", "raw_corr": "higher_better",
          "beats_ratio_dev": "lower_better", "missing": "lower_better", "spurious": "lower_better",
          "raw_rmse": "lower_better", "raw_qrs_rmse": "lower_better", "qrs_deriv_rmse": "lower_better", "qrs_curvature_err": "lower_better"}
PAIRED_ALL = tuple(ORIENT)
STAGE1 = ROOT / "artifacts/c1_interval_exposure/stage1_metrics.csv"
STAGE1_MAP = {"M1_qrs_e_dev": "qrs_e_dev", "M2_p2p_dev": "p2p_dev", "M3_qrs_rmse": "raw_qrs_rmse", "M4_rmse": "raw_rmse",
              "M5_raw_corr": "raw_corr", "M6_slope_dev": "slope_dev", "f1": "f1", "chance_f1": "chance_f1", "f1_excess": "f1_excess",
              "beats_ratio": "beats_ratio", "beats_ratio_dev": "beats_ratio_dev", "matched_coverage": "matched_coverage", "hf_ratio": "hf_ratio",
              "matched_morph": "morph"}
ORACLE_LABEL = "ORACLE (GT-R leakage; diagnostic only)"


# ------------------------------------------------------------------ frozen scoring (C0 verbatim + frozen extras)
def _peaks(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), FS)


def pmap(fn, items, chunk=16):
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


def _score_chunk(args):
    """C0 _score_chunk verbatim, plus the frozen M1 QRS-core morphology, whole-window RMSE/corr and the
    signed matched timing error (all GT-fixed; no new definition)."""
    pred, gt, gt_pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    max_shift = int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS))
    rows, errs = [], []
    for i in range(len(pred)):
        bl = AD.beat_level_analysis(pred[i], gt[i], gt_pk[i], FS, max_shift)   # only raw_* is read below
        ev = AD.event_timing(gt[i], pred[i], FS, tol_ms=S1.MATCH_TOL_MS)
        m, _, _ = R.match_rpeaks(ev["ref_rpeaks"], ev["pred_rpeaks"], FS, S1.MATCH_TOL_MS)
        n_ref = max(ev["n_ref"], 1)
        row = {"n_valid_gt_beats": int(bl["n_beats"]) if bl["n_beats"] else 0}
        for name, key, agg, _o, is_dev in PRIMARY:
            src = RATIO_RAW.get(name, key)
            v = float(agg(bl[src])) if bl["n_beats"] else np.nan
            row[name] = (abs(v - 1.0) if is_dev else v)
            if is_dev:
                row[src] = v
        row |= {
            "hf_ratio": float(hf_energy_ratio(pred[i][None])[0]),
            "hf_gt": float(hf_energy_ratio(gt[i][None])[0]),
            "f1": float(rm["rpeak_f1"][i]), "precision": float(rm["rpeak_precision"][i]),
            "recall": float(rm["rpeak_recall"][i]),
            "beats_ratio": ev["n_pred"] / n_ref, "missing": ev["n_missing"] / n_ref,
            "spurious": ev["n_spurious"] / n_ref, "n_matched": int(len(m)),
            "morph": float(rm["morph_corr"][i]), "rr_mae_ms": float(rm["rr_mae_ms"][i]),
        }
        row["hf_err"] = abs(row["hf_ratio"] - row["hf_gt"])
        row["beats_ratio_dev"] = abs(row["beats_ratio"] - 1.0)
        row["n_ref"], row["n_pred"] = int(ev["n_ref"]), int(ev["n_pred"])
        qm = M1.qrs_core_morphology(pred[i], gt[i], gt_pk[i])
        row |= {"qrs_deriv_rmse": float(qm["qrs_deriv_rmse"]), "qrs_curvature_err": float(qm["qrs_curvature_err"]),
                "qrs_rmse_core": float(qm["qrs_rmse_core"])}
        row["ww_rmse"] = float(np.sqrt(np.mean((pred[i] - gt[i]) ** 2)))
        row["ww_corr"] = float(np.corrcoef(pred[i], gt[i])[0, 1]) if pred[i].std() > 1e-8 and gt[i].std() > 1e-8 else np.nan
        rows.append(row); errs.append(np.asarray(ev["signed_err_ms"], float))
    return rows, errs


def _chance_chunk(args):
    gt_pk, pred_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    out = []
    for i in range(len(gt_pk)):
        n = len(pred_pk[i])
        f = [R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(
                R.match_rpeaks(gt_pk[i], S1.chance_random_phase(n, T_LEN, rng), FS, S1.MATCH_TOL_MS)))[2]
             for _ in range(S1.NULL_DRAWS)]
        out.append(float(np.mean(f)))
    return out


def score(pred: np.ndarray, Yd: np.ndarray, gt_pk):
    """Per-window rows (population order), detected prediction peaks, pooled signed timing errors."""
    ch = [(i, min(len(pred), i + 64)) for i in range(0, len(pred), 64)]
    rows, errs = [], []
    for rws, es in pmap(_score_chunk, [(pred[a:b], Yd[a:b], gt_pk[a:b]) for a, b in ch], chunk=1):
        rows += rws; errs += es
    pk = pmap(_peaks, list(pred))
    cf = []
    for a, b in ch:
        cf += _chance_chunk((gt_pk[a:b], pk[a:b]))
    for k, row in enumerate(rows):
        row["chance_f1"] = cf[k]; row["f1_excess"] = row["f1"] - cf[k]
    return rows, pk, errs


def macro_rows(rows, SUB):
    g = lambda k: np.asarray([x[k] for x in rows], float)  # noqa: E731
    out = {}
    for k in rows[0]:
        if isinstance(rows[0][k], (int, float)) and k not in ("n_ref", "n_pred", "n_matched", "n_valid_gt_beats"):
            out[k] = S1.macro(g(k), SUB)
    out["matched_coverage"] = float(np.sum(g("n_matched")) / max(np.sum(g("n_valid_gt_beats")), 1))
    out["n_windows_empty_pred"] = int(np.sum(g("n_pred") == 0))
    out["n_windows_nref0"] = int(np.sum(g("n_ref") == 0))
    return out


def wcsv(p, rows):
    """Union of keys in first-seen order; heterogeneous rows are padded with ''."""
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ generation
@torch.no_grad()
def gen_preds(net, X, e0, scaffold, adapter_sd, nfe, dev):
    """net: RhythmMeanFlowS5. scaffold: [N,T] float32 or None (zeros). adapter_sd: state_dict or None (zeros)."""
    if adapter_sd is None:
        net.rhythm_adapter.proj.weight.zero_()
    else:
        net.rhythm_adapter.load_state_dict({k: v.to(dev) for k, v in adapter_sd.items()})
    outs, got = [], set()
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        s = torch.zeros_like(pp) if scaffold is None else torch.from_numpy(scaffold[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, s), e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {nfe}, f"NFE parity violated: {got}"
    return np.concatenate(outs)


@torch.no_grad()
def gen_plain(base, X, e0, nfe, dev):
    outs = []
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(base, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe]); assert k == nfe
        outs.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def scaffolds(tcn, X, dev):
    out = []
    for i in range(0, len(X), RT.MICRO_BATCH):                            # same batch shape as training
        out.append(RT.scaffold_from_ppg(tcn, torch.from_numpy(X[i:i + RT.MICRO_BATCH]).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def rms_ratio(net, tcn_field, X, adapter_sd, dev):
    """RMS(rhythm_e) / RMS(ppg_e) on the population for one adapter (frozen stem)."""
    net.rhythm_adapter.load_state_dict({k: v.to(dev) for k, v in adapter_sd.items()})
    num = den = 0.0
    for i in range(0, len(X), 256):
        pp = torch.from_numpy(X[i:i + 256]).to(dev).unsqueeze(1)
        s = torch.from_numpy(tcn_field[i:i + 256]).to(dev).unsqueeze(1)
        num += float((net.rhythm_adapter(s) ** 2).sum()); den += float((net.backbone.pre_conv_ppg(pp) ** 2).sum())
    return float(np.sqrt(num / max(den, 1e-30)))


def paired(rows_a, rows_b, SUB, metrics, label_a, label_b, nfe):
    """Oriented paired effects, positive = rows_b (later) better. NaN counts asserted identical."""
    out = []
    for m in metrics:
        a = np.asarray([r[m] for r in rows_a], float); b = np.asarray([r[m] for r in rows_b], float)
        na, nb = int(np.isnan(a).sum()), int(np.isnan(b).sum())
        if na != nb or np.any(np.isnan(a) != np.isnan(b)):
            raise RuntimeError(f"NaN pattern differs between {label_a} and {label_b} on {m}: {na} vs {nb}")
        res = paired_subject_bootstrap(a, b, SUB, ORIENT[m], n_boot=RT.BOOT_N, seed=RT.BOOT_SEED)
        out.append({"comparison": f"{label_b}_vs_{label_a}@NFE{nfe}", "earlier": label_a, "later": label_b,
                    "earlier_label": ORACLE_LABEL if label_a == "ORACLE" else label_a, "later_label": ORACLE_LABEL if label_b == "ORACLE" else label_b,
                    "nfe": nfe, "metric": m, **res, "n_eff": int(len(a) - na), "nan_pairs": na})
    return out


def subset_paired(rows_a, rows_b, SUB, mask, metrics, label_a, label_b, tag):
    idx = np.flatnonzero(mask)
    if idx.size < 4:
        return [{"stratum": tag, "metric": m, "n": int(idx.size), "point": np.nan, "lo": np.nan, "hi": np.nan, "verdict": "n/a"} for m in metrics]
    out = []
    for m in metrics:
        a = np.asarray([rows_a[i][m] for i in idx], float); b = np.asarray([rows_b[i][m] for i in idx], float)
        res = paired_subject_bootstrap(a, b, SUB[idx], ORIENT[m], n_boot=RT.BOOT_N, seed=RT.BOOT_SEED)
        out.append({"stratum": tag, "metric": m, "n": int(idx.size), "comparison": f"{label_b}_vs_{label_a}", **res})
    return out


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    git = git_sha(ROOT); t_all = time.perf_counter()
    dev = torch.device("cuda")
    prov = {"git": git, "prereg": PREREG, "utc_start": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
            "libs": {"torch": torch.__version__, "numpy": np.__version__, "neurokit2": neurokit2.__version__, "python": platform.python_version()},
            "gpu": torch.cuda.get_device_name(0), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "scaffold_batch": RT.MICRO_BATCH}
    pt = subprocess.run(["python", "-m", "pytest", "tests/test_r2_rhythm_transfer.py", "-q", "-rs", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    prov["tests_ran"] = {"exit": pt.returncode, "summary": pt.stdout.strip().splitlines()[-1] if pt.stdout.strip() else "",
                         "skipped": [ln for ln in pt.stdout.splitlines() if ln.startswith("SKIPPED")]}
    if pt.returncode != 0:
        raise RuntimeError("R2 tests fail; not evaluating")
    pf = json.loads((ART / "runtime_preflight.json").read_text())
    tprov = {arm: json.loads((ART / f"train_provenance_{arm}.json").read_text()) for arm in RT.TRAINED_ARMS}
    hashes = {"preflight": pf["probe_hash"], **{arm: tprov[arm]["probe_hash"] for arm in RT.TRAINED_ARMS}}
    if len(set(hashes.values())) != 1 or any(tprov[arm]["opt_steps"] != RT.STEPS for arm in RT.TRAINED_ARMS):
        raise RuntimeError(f"paired-randomness probe hashes or step counts differ across processes (STOP): {hashes}")
    prov |= {"probe_hashes": hashes, "runtime_preflight": pf, "training": tprov,
             "oracle_cache_sha256": json.loads((ART / "cache_build.json").read_text())["oracle_cache_sha256"],
             "scaffold_stats_train_visited": tprov["true"]["scaffold_stats_train_visited"]}

    # ---------------- population (frozen subset, asserted) ----------------
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        idx = ER.select_subset(SALT, s, len(Xs), TAKE)
        X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32))
        SUB.append(np.full(len(idx), s)); SITE.append(Ss[idx]); POS.append(idx); WI.append(Ws[idx].astype(np.int64))
    X, Y, SUB, SITE, POS, WI = (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s in VAL:
        assert POS[SUB == s].tolist() == list(frozen[s]), f"frozen subset mismatch {s}"
    Yd = Y.astype(np.float64)
    gt_pk = pmap(_peaks, list(Yd))
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()
    assert src_hash == "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f", src_hash
    prov |= {"population_windows": int(len(X)), "gt_beats": int(sum(len(p) for p in gt_pk)), "source_bank_sha256": src_hash}
    if len(X) != 2048 or prov["gt_beats"] != 19834:
        raise RuntimeError(f"frozen population facts differ: {len(X)} windows, {prov['gt_beats']} GT beats (STOP)")
    print(f"[P] {len(X)} windows, {prov['gt_beats']} GT beats, bank {src_hash[:16]} | HEAD {git['commit'][:8]}", flush=True)

    # ---------------- frozen components, adapters, scaffolds ----------------
    net, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    net.requires_grad_(False)
    adapters = {}
    for arm in RT.TRAINED_ARMS:
        ad = torch.load(ROOT / f"outputs/r2_{arm}_adapter_seed42/adapter_step{RT.STEPS}.pt", map_location="cpu", weights_only=False)
        assert ad["step"] == RT.STEPS and ad["arm"] == arm and ad["generator_state_sha256"] == gmeta["state_dict_sha256"]
        adapters[arm.upper()] = ad["state_dict"]
    # shuffle partner on the eval population (manifest, asserted against the rule)
    man = [r for r in csv.DictReader(open(ART / "shuffle_manifest.csv")) if r["population"] == "eval"]
    partner = np.full(len(X), -1, dtype=np.int64)
    for r in man:
        partner[int(r["pop_row"])] = int(r["partner_pop_row"])
    RT.assert_derangement(partner)
    assert np.array_equal(partner, RT.shuffle_partner(SUB, SITE, WI)), "eval shuffle manifest mismatch"
    s_true = scaffolds(tcn, X, dev)
    s_shuffle = s_true[partner]
    s_oracle = RT.oracle_fields(Y, workers=WORKERS)
    s_phase = np.roll(s_true, RT.PHASE_SHIFT_SAMPLES, axis=1)
    assert np.array_equal(s_phase, RT.roll_scaffold(torch.from_numpy(s_true), RT.PHASE_SHIFT_SAMPLES).numpy())
    prov["scaffold_stats_eval"] = {"max_mean": float(s_true.max(1).mean()), "max_p10": float(np.percentile(s_true.max(1), 10)),
                                   "max_p90": float(np.percentile(s_true.max(1), 90)), "mean_mean": float(s_true.mean()),
                                   "mean_p10": float(np.percentile(s_true.mean(1), 10)), "mean_p90": float(np.percentile(s_true.mean(1), 90))}
    ARM_INPUT = {"B": (None, None), "TRUE": (s_true, adapters["TRUE"]), "SHUFFLE": (s_shuffle, adapters["SHUFFLE"]),
                 "ORACLE": (s_oracle, adapters["ORACLE"]), "PHASE": (s_phase, adapters["TRUE"])}

    # ---------------- arm-B parity (STOP) ----------------
    p_plain = gen_plain(base, X, e0, 4, dev)
    p_B4 = gen_preds(net, X, e0, None, None, 4, dev)
    if not np.array_equal(p_plain, p_B4):
        raise RuntimeError("ARM-B PARITY FAILED: zero adapter + zero scaffold != frozen MeanFlowS5 (STOP)")
    p_B4_s = gen_preds(net, X, e0, s_true, None, 4, dev)
    prov["arm_b_parity_torch_equal"] = True
    prov["zero_adapter_real_scaffold_equal"] = bool(np.array_equal(p_plain, p_B4_s))
    print(f"[B] parity OK (zero adapter: zero scaffold equal, real scaffold equal={prov['zero_adapter_real_scaffold_equal']})", flush=True)

    # ---------------- generation + scoring grid ----------------
    grid = [(arm, n) for arm in RT.ARMS for n in NFES] + [("PHASE", 4)]
    per, pks, errs = {}, {}, {}
    win_rows = []
    grid_t = {}
    for arm, n in grid:
        sc, ad = ARM_INPUT[arm]
        t_g = time.perf_counter()
        pred = (p_B4 if (arm == "B" and n == 4) else gen_preds(net, X, e0, sc, ad, n, dev)).astype(np.float64)
        rows, pk, er = score(pred, Yd, gt_pk)
        grid_t[(arm, n)] = time.perf_counter() - t_g
        per[(arm, n)], pks[(arm, n)], errs[(arm, n)] = rows, pk, er
        mr = macro_rows(rows, SUB)
        print(f"[E] {arm:8s} NFE{n} F1 {mr['f1']:.4f} excess {mr['f1_excess']:.4f} miss {mr['missing']:.3f} spur {mr['spurious']:.3f} "
              f"bdev {mr['beats_ratio_dev']:.4f} | rmse {mr['raw_rmse']:.4f} corr {mr['raw_corr']:.4f} qrs {mr['raw_qrs_rmse']:.4f} "
              f"d1 {mr['qrs_deriv_rmse']:.4f} curv {mr['qrs_curvature_err']:.4f}", flush=True)
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        for i, r in enumerate(rows):
            win_rows.append({"arm": arm, "label": ORACLE_LABEL if arm == "ORACLE" else arm, "nfe": n, "subject": SUB[i], "array_pos": int(POS[i]), "npz_window_index": int(WI[i]),
                             "site": SITE[i], **{k: r[k] for k in keys}})
        del pred
    wcsv(ART / "metrics_by_window.csv", win_rows)

    # ---------------- macro tables ----------------
    ev_rows, st_rows = [], []
    for arm, n in grid:
        mr = macro_rows(per[(arm, n)], SUB); e = np.concatenate(errs[(arm, n)]) if errs[(arm, n)] else np.zeros(0)
        label = ORACLE_LABEL if arm == "ORACLE" else arm
        ev_rows.append({"arm": label, "nfe": n, **{k: mr[k] for k in ("f1", "chance_f1", "f1_excess", "precision", "recall", "matched_coverage",
                                                                       "missing", "spurious", "beats_ratio", "beats_ratio_dev", "n_windows_empty_pred", "n_windows_nref0")},
                        "timing_median_abs_ms": float(np.median(np.abs(e))) if e.size else np.nan,
                        "timing_mean_ms": float(np.mean(e)) if e.size else np.nan,
                        "timing_frac_le25ms": float(np.mean(np.abs(e) <= 25)) if e.size else np.nan, "n_matched_beats": int(e.size)})
        st_rows.append({"arm": label, "nfe": n, **{k: mr[k] for k in ("raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err",
                                                                       "qrs_e_dev", "p2p_dev", "slope_dev", "hf_ratio", "hf_gt", "hf_err", "ww_rmse", "ww_corr", "qrs_rmse_core")}})
    wcsv(ART / "event_metrics.csv", ev_rows); wcsv(ART / "structure_metrics.csv", st_rows)

    # ---------------- historical regression diagnostic (B, NFE 4) ----------------
    reg = {}
    if STAGE1.exists():
        ref = next((r for r in csv.DictReader(open(STAGE1)) if r["arm"] == "B" and int(float(r["nfe"])) == 4), None)
        if ref:
            mr = macro_rows(per[("B", 4)], SUB)
            for k1, k2 in STAGE1_MAP.items():
                if k1 in ref:
                    reg[k2] = {"stage1": float(ref[k1]), "r2": float(mr[k2]), "delta": float(mr[k2]) - float(ref[k1]),
                               "flag_gt_1e-6": bool(abs(float(mr[k2]) - float(ref[k1])) > 1e-6)}
    prov["regression_vs_c1_stage1_B4"] = reg
    print("[R] regression flags:", [k for k, v in reg.items() if v["flag_gt_1e-6"]] or "none", flush=True)

    # ---------------- paired comparisons ----------------
    pb, og = [], []
    pb += paired(per[("B", 4)], per[("TRUE", 4)], SUB, PAIRED_ALL, "B", "TRUE", 4)
    pb += paired(per[("SHUFFLE", 4)], per[("TRUE", 4)], SUB, PAIRED_ALL, "SHUFFLE", "TRUE", 4)
    pb += paired(per[("B", 4)], per[("SHUFFLE", 4)], SUB, PAIRED_ALL, "B", "SHUFFLE", 4)
    og += paired(per[("B", 4)], per[("ORACLE", 4)], SUB, PAIRED_ALL, "B", "ORACLE", 4)
    og += paired(per[("B", 4)], per[("TRUE", 4)], SUB, PAIRED_ALL, "B", "TRUE", 4)
    og += paired(per[("TRUE", 4)], per[("ORACLE", 4)], SUB, PAIRED_ALL, "TRUE", "ORACLE", 4)
    for n in (1, 2):
        pb += paired(per[("B", n)], per[("TRUE", n)], SUB, EVENT4, "B", "TRUE", n)
        pb += paired(per[("SHUFFLE", n)], per[("TRUE", n)], SUB, EVENT4, "SHUFFLE", "TRUE", n)
    wcsv(ART / "paired_bootstrap.csv", pb); wcsv(ART / "oracle_gap.csv", og)
    get = lambda rows, comp, m: next(r for r in rows if r["comparison"] == comp and r["metric"] == m)  # noqa: E731
    for r in pb:
        if r["nfe"] == 4 and r["metric"] in EVENT4 + STRUCT5:
            print(f"[PB] {r['comparison']:22s} {r['metric']:18s} {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)
    for r in og:
        if r["metric"] == "f1_excess":
            print(f"[OG] {r['comparison']:22s} f1_excess {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)

    # ---------------- decision (section 21-22) ----------------
    tb = {m: get(pb, "TRUE_vs_B@NFE4", m) for m in PAIRED_ALL}
    ts = get(pb, "TRUE_vs_SHUFFLE@NFE4", "f1_excess"); sb = get(pb, "SHUFFLE_vs_B@NFE4", "f1_excess")
    ob = get(og, "ORACLE_vs_B@NFE4", "f1_excess"); ot = get(og, "ORACLE_vs_TRUE@NFE4", "f1_excess")
    mr_true = macro_rows(per[("TRUE", 4)], SUB)
    rec = {"item1_ci_positive": tb["f1_excess"]["verdict"] == "improves", "item1_point": float(tb["f1_excess"]["point"]),
           "item1": tb["f1_excess"]["verdict"] == "improves" and tb["f1_excess"]["point"] >= RT.GATE_MIN_EFFECT,
           "item2": ts["verdict"] == "improves",
           "item3": any(tb[m]["verdict"] == "improves" for m in ("beats_ratio_dev", "missing", "spurious")),
           "item3_carriers": [m for m in ("beats_ratio_dev", "missing", "spurious") if tb[m]["verdict"] == "improves"],
           "item3_status": {m: tb[m]["verdict"] for m in ("beats_ratio_dev", "missing", "spurious")},
           "item4_n_degraded": int(sum(tb[m]["verdict"] == "worsens" for m in STRUCT5)),
           "item4_status": {m: tb[m]["verdict"] for m in STRUCT5},
           "item5_value": float(mr_true["beats_ratio_dev"]), "item5": bool(mr_true["beats_ratio_dev"] < RT.GATE_BEATS_DEV_MAX),
           "v_OB": ob["verdict"], "v_TB": tb["f1_excess"]["verdict"], "v_OT": ot["verdict"], "v_SB": sb["verdict"]}
    rec["item4"] = rec["item4_n_degraded"] < 2
    verdict, reason = RT.decide_verdict(rec)
    quals = []
    if tb["beats_ratio_dev"]["verdict"] == "worsens" or tb["spurious"]["verdict"] == "worsens":
        quals.append("with beat-count distortion")
    fam = {"segment(S1,S2)": [m for m in ("raw_rmse", "raw_corr") if tb[m]["verdict"] == "worsens"],
           "qrs_core(S3,S4,S5)": [m for m in ("raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err") if tb[m]["verdict"] == "worsens"]}
    dec = {"verdict": verdict, "residual_reason": reason, "qualifiers": quals, "oracle_case": RT.oracle_case(rec["v_OB"], rec["v_TB"], rec["v_OT"]),
           "gate2_reading_window_specific": rec["item2"] and sb["verdict"] != "worsens", "shuffle_vs_B_f1_excess": sb["verdict"],
           "degraded_families": fam, **rec, "gate_min_effect": RT.GATE_MIN_EFFECT, "gate_beats_dev_max": RT.GATE_BEATS_DEV_MAX,
           "nfe": 4, "population": "frozen 2048 (x4-event-nfe-v2)", "prereg": PREREG, "oracle_label": ORACLE_LABEL}
    (ART / "decision.json").write_text(json.dumps(dec, indent=2, default=float))
    print(f"\n[GATE] 1:{rec['item1']} (point {rec['item1_point']:+.4f}, CI+ {rec['item1_ci_positive']}) 2:{rec['item2']} 3:{rec['item3']} {rec['item3_carriers']} "
          f"4:{rec['item4']} (degraded {rec['item4_n_degraded']}) 5:{rec['item5']} ({rec['item5_value']:.4f})\n"
          f"[ORACLE] v_OB {rec['v_OB']} v_TB {rec['v_TB']} v_OT {rec['v_OT']} -> {dec['oracle_case']} | SHUFFLE_vs_B {rec['v_SB']}\n"
          f"[VERDICT] {verdict} {('(' + reason + ')') if reason else ''} {quals}", flush=True)
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))       # primary provenance first

    # ---------------- phase ablation (section 18) ----------------
    phi = np.array([RT.phase_phi(p) for p in gt_pk]); strat = np.array([RT.phi_stratum(v) for v in phi])
    ph_rows = [{"pop_row": i, "subject": SUB[i], "site": SITE[i], "phi": phi[i], "stratum": strat[i],
                **{f"true_{m}": per[("TRUE", 4)][i][m] for m in EVENT4}, **{f"shifted_{m}": per[("PHASE", 4)][i][m] for m in EVENT4}}
               for i in range(len(X))]
    wcsv(ART / "phase_ablation.csv", ph_rows)
    ph_sum = subset_paired(per[("PHASE", 4)], per[("TRUE", 4)], SUB, np.ones(len(X), bool), EVENT4, "SHIFTED", "TRUE", "all")
    for tag in ("in_phase", "anti_phase", "rest", "undefined"):
        ph_sum += subset_paired(per[("PHASE", 4)], per[("TRUE", 4)], SUB, strat == tag, EVENT4, "SHIFTED", "TRUE", tag)
    wcsv(ART / "phase_ablation_summary.csv", ph_sum)
    for r in ph_sum:
        if r["metric"] == "f1_excess":
            print(f"[PH] {r['stratum']:10s} n={r['n']:4d} TRUE-vs-SHIFTED f1_excess {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)

    # ---------------- NFE persistence (section 19) ----------------
    pe_rows, pe_sum = [], {}
    beat_sets = {}
    for arm in ("B", "TRUE"):
        D = []
        for i in range(len(X)):
            d = RT.persistence_deltas(gt_pk[i], {n: pks[(arm, n)][i] for n in NFES})
            for b in range(len(gt_pk[i])):
                pe_rows.append({"arm": arm, "subject": SUB[i], "pop_row": i, "beat": b, "delta_1": d[b, 0], "delta_2": d[b, 1], "delta_4": d[b, 2]})
            D.append(d)
        D = np.concatenate(D) if D else np.zeros((0, 3))
        m1, m2, m4 = ~np.isnan(D[:, 0]), ~np.isnan(D[:, 1]), ~np.isnan(D[:, 2])
        all3, m14 = m1 & m2 & m4, m1 & m4
        s1, s4 = np.sign(D[all3, 0]), np.sign(D[all3, 2]); s2 = np.sign(D[all3, 1])
        cens = np.where(np.isnan(D), RT.PERSIST_TOL_MS, D)
        pe_sum[arm] = {"n_gt_beats": int(len(D)), "match_frac": {n: float(m.mean()) for n, m in zip(NFES, (m1, m2, m4))},
                       "mean_abs_delta_matched": {n: float(np.abs(D[m, j]).mean()) for j, (n, m) in enumerate(zip(NFES, (m1, m2, m4)))},
                       "mean_abs_delta_censored250": {n: float(np.abs(cens[:, j]).mean()) for j, n in enumerate(NFES)},
                       "n_all3": int(all3.sum()), "sign_consistency_1_4": float(np.mean(s1 == s4)) if all3.any() else np.nan,
                       "sign_all3_equal": float(np.mean((s1 == s2) & (s2 == s4))) if all3.any() else np.nan,
                       "n_zero_delta_all3": int(np.sum((D[all3] == 0).any(axis=1))),
                       "n_matched_1_and_4": int(m14.sum()), "mean_change_4_minus_1_ms": float(np.mean(D[m14, 2] - D[m14, 0])) if m14.any() else np.nan,
                       "mean_abs_change_ms": float(np.mean(np.abs(D[m14, 2]) - np.abs(D[m14, 0]))) if m14.any() else np.nan,
                       "frac_nfe4_closer_strict": float(np.mean(np.abs(D[m14, 2]) < np.abs(D[m14, 0]))) if m14.any() else np.nan,
                       "frac_tie": float(np.mean(np.abs(D[m14, 2]) == np.abs(D[m14, 0]))) if m14.any() else np.nan}
        beat_sets[arm] = {"all3": set(np.flatnonzero(all3).tolist()), "m14": set(np.flatnonzero(m14).tolist()), "D": D}
    inter14 = np.array(sorted(beat_sets["B"]["m14"] & beat_sets["TRUE"]["m14"]))
    inter3 = np.array(sorted(beat_sets["B"]["all3"] & beat_sets["TRUE"]["all3"]))
    pe_sum["intersection"] = {"n_m14_B_only": len(beat_sets["B"]["m14"] - beat_sets["TRUE"]["m14"]), "n_m14_TRUE_only": len(beat_sets["TRUE"]["m14"] - beat_sets["B"]["m14"]),
                              "n_m14_both": int(inter14.size), "n_all3_both": int(inter3.size),
                              "n_all3_B_only": len(beat_sets["B"]["all3"] - beat_sets["TRUE"]["all3"]), "n_all3_TRUE_only": len(beat_sets["TRUE"]["all3"] - beat_sets["B"]["all3"])}
    for arm in ("B", "TRUE"):
        D = beat_sets[arm]["D"]
        if inter14.size:
            pe_sum["intersection"][f"{arm}_frac_nfe4_closer_strict"] = float(np.mean(np.abs(D[inter14, 2]) < np.abs(D[inter14, 0])))
            pe_sum["intersection"][f"{arm}_mean_abs_delta1"] = float(np.abs(D[inter14, 0]).mean())
            pe_sum["intersection"][f"{arm}_mean_abs_delta4"] = float(np.abs(D[inter14, 2]).mean())
        if inter3.size:
            pe_sum["intersection"][f"{arm}_sign_consistency_1_4"] = float(np.mean(np.sign(D[inter3, 0]) == np.sign(D[inter3, 2])))
    wcsv(ART / "nfe_event_persistence.csv", pe_rows)
    (ART / "nfe_event_persistence_summary.json").write_text(json.dumps(pe_sum, indent=2, default=float))
    for arm in ("B", "TRUE"):
        s_ = pe_sum[arm]
        print(f"[NP] {arm:5s} match {s_['match_frac']} |d| {s_['mean_abs_delta_matched']} closer@4 {s_['frac_nfe4_closer_strict']:.3f} tie {s_['frac_tie']:.3f}", flush=True)

    # ---------------- scaffold-quality stratification (section 17, exploratory) ----------------
    sq = np.array([RT.scaffold_event_f1(s_true[i], gt_pk[i]) for i in range(len(X))])
    q = np.quantile(sq, [1 / 3, 2 / 3]); terc = (np.argsort(np.argsort(sq, kind="stable"), kind="stable") * 3) // len(sq)   # rank terciles
    sq_rows = []
    for k, tag in enumerate(("low", "mid", "high")):
        sq_rows += subset_paired(per[("B", 4)], per[("TRUE", 4)], SUB, terc == k, EVENT4, "B", "TRUE", f"scaffold_f1_{tag}")
    for r in sq_rows:
        r["tercile_edges"] = f"{q[0]:.3f},{q[1]:.3f}"
    wcsv(ART / "scaffold_quality_strata.csv", sq_rows)

    # ---------------- adapter magnitude ----------------
    prov["adapter_l2"] = {arm: float(torch.linalg.vector_norm(sd["proj.weight"])) for arm, sd in adapters.items()}
    prov["rms_rhythm_over_ppg_eval"] = {arm: rms_ratio(net, {"TRUE": s_true, "SHUFFLE": s_shuffle, "ORACLE": s_oracle}[arm], X, sd, dev) for arm, sd in adapters.items()}
    print("[A] adapter L2", prov["adapter_l2"], "| RMS ratio", prov["rms_rhythm_over_ppg_eval"], flush=True)

    # ---------------- site-wise secondary on the R1 8,192 cohort (section 20) ----------------
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    t_site = time.perf_counter()
    proj_site_s = 2 * 4 * (grid_t[("B", 4)] + grid_t[("TRUE", 4)]) / 2          # 2 arms x 4x the population
    prov["site_cohort"] = {"projected_s": proj_site_s, "budget_s": 7200.0}
    if proj_site_s > 7200.0:
        wcsv(ART / "site_metrics.csv", [{"site": "skipped", "reason": f"projected {proj_site_s:.0f} s > 1 GPU-h + 1 CPU-h budget"}])
        prov["site_cohort"]["skipped"] = True
        (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
        print("[S] site-wise secondary skipped (budget)", flush=True)
        return 0
    try:
        site_wise(X_val_paths=None, prov=prov, net=net, tcn=tcn, adapters=adapters, dev=dev)
    except Exception as ex:                                                    # secondary must never lose the primary
        wcsv(ART / "site_metrics.csv", [{"site": "skipped", "reason": f"failed: {type(ex).__name__}: {ex}"}])
        prov["site_cohort"]["skipped"] = True; prov["site_cohort"]["error"] = repr(ex)
        print(f"[S] site-wise secondary failed: {ex!r}", flush=True)
    prov["site_cohort"]["wall_s"] = time.perf_counter() - t_site
    prov |= {"generator": gmeta, "rhythm_tcn": tmeta, "adapters": {arm: str(ROOT / f"outputs/r2_{arm}_adapter_seed42/adapter_step{RT.STEPS}.pt") for arm in RT.TRAINED_ARMS},
             "wall_s": time.perf_counter() - t_all, "utc_end": datetime.now(timezone.utc).isoformat()}
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[done] R2 evaluation in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    return 0


def site_wise(X_val_paths, prov, net, tcn, adapters, dev):
    X8, Y8, SUB8, SITE8 = [], [], [], []
    for sub in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        pos = C.cohort_positions(sub, Ss, Ws, C.n_per_for(sub))
        for site in C.SITES:
            idx = pos[site]; X8.append(Xs[idx].astype(np.float32)); Y8.append(Ys[idx].astype(np.float32))
            SUB8.append(np.full(len(idx), sub)); SITE8.append(np.full(len(idx), site))
    X8, Y8, SUB8, SITE8 = (np.concatenate(v) for v in (X8, Y8, SUB8, SITE8))
    e8 = torch.randn(len(X8), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    prov["site_cohort"] |= {"windows": int(len(X8)), "source_bank_sha256": hashlib.sha256(e8.numpy().tobytes()).hexdigest()}
    Y8d = Y8.astype(np.float64); gt8 = pmap(_peaks, list(Y8d))
    s8 = scaffolds(tcn, X8, dev)
    per8 = {}
    for arm, (sc, ad) in (("B", (None, None)), ("TRUE", (s8, adapters["TRUE"]))):
        pred = gen_preds(net, X8, e8, sc, ad, 4, dev).astype(np.float64)
        per8[arm], _, _ = score(pred, Y8d, gt8); del pred
    site_rows, contrast = [], []
    SM = ("f1_excess", "beats_ratio_dev", "raw_qrs_rmse", "qrs_deriv_rmse")
    for site in C.SITES:
        m = SITE8 == site
        for arm in ("B", "TRUE"):
            mr = macro_rows([per8[arm][i] for i in np.flatnonzero(m)], SUB8[m])
            site_rows.append({"site": site, "arm": arm, "n_windows": int(m.sum()), **{k: mr[k] for k in SM}})
        for r in subset_paired(per8["B"], per8["TRUE"], SUB8, m, SM, "B", "TRUE", site):
            site_rows.append({"site": site, "arm": "TRUE_vs_B", "n_windows": r["n"], "metric": r["metric"], "point": r["point"], "lo": r["lo"], "hi": r["hi"], "verdict": r["verdict"]})
    # distal - proximal contrast: independent subject-stratified resampling within each group (windows are not paired across sites)
    distal, prox = np.isin(SITE8, ("wrist", "ankle")), np.isin(SITE8, ("sternum", "head"))
    METHOD = "independent subject-stratified two-group bootstrap (windows are unpaired across sites); adapted from the C1 difference-of-improvement idiom; rng re-seeded per metric"
    for m_ in SM:
        rng = np.random.default_rng(RT.BOOT_SEED)
        a = np.asarray([r[m_] for r in per8["B"]], float); b = np.asarray([r[m_] for r in per8["TRUE"]], float)
        dvec = (b - a) if ORIENT[m_] == "higher_better" else (a - b)

        def gmean(mask, idxs=None):
            return float(np.mean([np.nanmean(dvec[np.flatnonzero(mask & (SUB8 == s))] if idxs is None else idxs[s]) for s in VAL]))
        point = gmean(distal) - gmean(prox)
        draws = np.empty(RT.BOOT_N)
        for k in range(RT.BOOT_N):
            dd = {s: dvec[rng.choice(np.flatnonzero(distal & (SUB8 == s)), int(np.sum(distal & (SUB8 == s))), replace=True)] for s in VAL}
            pp = {s: dvec[rng.choice(np.flatnonzero(prox & (SUB8 == s)), int(np.sum(prox & (SUB8 == s))), replace=True)] for s in VAL}
            draws[k] = gmean(None, dd) - gmean(None, pp)
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        contrast.append({"metric": m_, "contrast": "(wrist+ankle) - (sternum+head) of oriented TRUE-B effect", "point": point, "lo": float(lo), "hi": float(hi),
                         "verdict": "distal_gains_more" if lo > 0 else ("proximal_gains_more" if hi < 0 else "unresolved"), "seed": RT.BOOT_SEED, "n_boot": RT.BOOT_N, "method": METHOD})
    site_rows_full = [dict(r, contrast="") for r in site_rows] + [{"site": "contrast", "arm": "TRUE_vs_B", "n_windows": int(len(X8)), "metric": c["metric"],
                                                                     "point": c["point"], "lo": c["lo"], "hi": c["hi"], "verdict": c["verdict"], "contrast": c["contrast"]} for c in contrast]
    wcsv(ART / "site_metrics.csv", site_rows_full)
    prov["site_cohort"]["contrast_method"] = METHOD
    for r in site_rows:
        if r["arm"] == "TRUE_vs_B" and r["metric"] == "f1_excess":
            print(f"[S] {r['site']:8s} TRUE-vs-B f1_excess {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)
    for c in contrast:
        print(f"[S] contrast {c['metric']:16s} {c['point']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] {c['verdict']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
