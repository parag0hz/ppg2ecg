"""R3 evaluation — docs/R3_DISENTANGLED_RHYTHM_FUSION_PREREGISTRATION.md (3d779fc) sections 10, 13-21.

Primary population: the frozen C0/C1/R2 2,048-window development subset, seed-0 bank. Nine arms at NFE 1/2/4
plus the phase-ablation and direct-route-attribution variants of TF-TRUE / GTF-TRUE at NFE 4. Scoring
functions are IMPORTED from scripts/r2_evaluate.py (which carries the C0 pipeline verbatim); everything
arm-specific is here. Secondary: site-wise on the R1 8,192-window cohort, gate diagnostics, persistence.
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

import neurokit2
import numpy as np
import torch
from scipy.stats import spearmanr

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes.rhythm_tcn import extract_events
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r3_rhythm_fusion"
R2ART = ROOT / "artifacts/r2_rhythm_transfer"
_spec = importlib.util.spec_from_file_location("r2_evaluate", ROOT / "scripts/r2_evaluate.py")
R2E = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = R2E; _spec.loader.exec_module(R2E)   # registered: pool workers pickle by module name
FS, T_LEN, BATCH = 128, 1024, 64
VAL, SALT, TAKE = ("an0", "k2s"), "x4-event-nfe-v2", 1024
NFES, SRC_SEED = (1, 2, 4), 0
PREREG = "3d779fc"
EVENT4, STRUCT5, ORIENT = R2E.EVENT4, R2E.STRUCT5, R2E.ORIENT
PAIRED_ALL = R2E.PAIRED_ALL
SM7 = ("f1_excess", "missing", "spurious", "beats_ratio_dev", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err")
CONTRASTS = [("TF-TRUE", "B"), ("TF-TRUE", "TF-SHUFFLE"), ("TF-TRUE", "ADD"), ("TF-SHUFFLE", "B"), ("GTF-TRUE", "B"), ("GTF-TRUE", "GTF-SHUFFLE"),
             ("GTF-TRUE", "GTF-CONST"), ("GTF-TRUE", "TF-TRUE"), ("GTF-TRUE", "ADD"), ("GTF-CONST", "B"), ("GTF-SHUFFLE", "B"), ("ADD", "B")]
CONTRASTS_NFE12 = [("TF-TRUE", "B"), ("TF-TRUE", "TF-SHUFFLE"), ("GTF-TRUE", "B"), ("GTF-TRUE", "GTF-SHUFFLE")]
ORACLE_CONTRASTS = [("GTF-ORACLE", "B"), ("GTF-ORACLE", "GTF-TRUE"), ("GTF-ORACLE", "ADD-ORACLE"), ("ADD-ORACLE", "B")]
PERSIST_ARMS = ("B", "ADD", "TF-TRUE", "GTF-TRUE")
SITE_ARMS = ("B", "ADD", "TF-TRUE", "GTF-TRUE", "GTF-CONST")


def label(arm: str) -> str:
    return f"{arm} {RF.ORACLE_LABEL}" if arm in RF.ORACLE_ARMS else arm


NAN_FLAGS, CI_COUNT = [], [0]


def paired3(per, first, second, nfe, metrics, SUB):
    """positive = FIRST-named arm better (earlier = second, later = first). Section 14 NaN rule: if the NaN
    pattern of a metric differs between the arms, pairwise-incomplete windows are dropped for that metric and
    the event is flagged in provenance (never aborted)."""
    rows = []
    for m in metrics:
        a = np.asarray([r[m] for r in per[(second, nfe)]], float); b = np.asarray([r[m] for r in per[(first, nfe)]], float)
        ok = ~(np.isnan(a) | np.isnan(b))
        if np.any(np.isnan(a) != np.isnan(b)):
            NAN_FLAGS.append({"comparison": f"{first}_vs_{second}@NFE{nfe}", "metric": m, "nan_first": int(np.isnan(b).sum()), "nan_second": int(np.isnan(a).sum()), "dropped": int((~ok).sum())})
            rows_a, rows_b, sub = [per[(second, nfe)][i] for i in np.flatnonzero(ok)], [per[(first, nfe)][i] for i in np.flatnonzero(ok)], SUB[ok]
        else:
            rows_a, rows_b, sub = per[(second, nfe)], per[(first, nfe)], SUB
        rows += R2E.paired(rows_a, rows_b, sub, (m,), second, first, nfe)
    for r in rows:
        r["comparison"] = f"{first}_vs_{second}@NFE{nfe}"; r["earlier_label"], r["later_label"] = label(second), label(first)
    CI_COUNT[0] += len(rows)
    return rows


def get(rows, comp, m):
    return next(r for r in rows if r["comparison"] == comp and r["metric"] == m)


def subset_paired(*a, **k):
    rows = R2E.subset_paired(*a, **k); CI_COUNT[0] += sum(1 for r in rows if r.get("verdict") != "n/a"); return rows


@torch.no_grad()
def gen_r3(net: RF.FusionMeanFlowS5, module_sd, X, e0, scaffold, nfe, dev, cancel_direct=False):
    if module_sd is not None:
        net.r3.load_state_dict({k: v.to(dev) for k, v in module_sd.items()})
    net.cancel_direct_route = bool(cancel_direct)
    outs, got = [], set()
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        s = torch.zeros_like(pp) if scaffold is None else torch.from_numpy(scaffold[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, s), e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
    net.cancel_direct_route = False
    assert got == {nfe}, f"NFE parity violated: {got}"
    return np.concatenate(outs)


@torch.no_grad()
def gen_add_local(net_add, adapter_sd, X, e0, scaffold, nfe, dev):
    """Explicit ADD loop (same class as R2's path) for the same-process parity check."""
    net_add.rhythm_adapter.load_state_dict({k: v.to(dev) for k, v in adapter_sd.items()})
    outs = []
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        s = torch.from_numpy(scaffold[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net_add, RT.make_ppg2(pp, s), e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe]); assert k == nfe
        outs.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def gate_values(module: RF.R3Fusion, S, dev):
    out = []
    for i in range(0, len(S), 256):
        out.append(module.gate_values(torch.from_numpy(S[i:i + 256]).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def regression(rows_r3, r2_rows, keys):
    """per-window |delta| against R2's stored rows (matched by (subject, array_pos)); NaN-vs-number is a flag."""
    maxd, nflag = {}, 0
    assert len(r2_rows) == len(rows_r3) == 2048
    for r in rows_r3:
        ref = r2_rows[(r["subject"], int(r["array_pos"]))]
        for k in keys:
            a, b = float(r[k]), float(ref[k])
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) != np.isnan(b):
                nflag += 1; maxd[k] = float("inf"); continue
            d = abs(a - b); maxd[k] = max(maxd.get(k, 0.0), d); nflag += int(d > 1e-6)
    return {"max_abs_delta": maxd, "n_flags_gt_1e-6": nflag}


def subject_boot(stat_fn, SUB, n_boot=RF.BOOT_N, seed=RF.BOOT_SEED):
    """Equal-subject-weight point estimate + percentile CI by resampling windows within subject; stat_fn(idx) -> float per subject-subset
    (NaN for a degenerate resample; NaN draws are counted and excluded from the percentiles)."""
    uniq = sorted(set(SUB.tolist())); idx = {u: np.flatnonzero(SUB == u) for u in uniq}
    rng = np.random.default_rng(seed)
    point = float(np.nanmean([stat_fn(idx[u]) for u in uniq]))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        draws[k] = float(np.nanmean([stat_fn(rng.choice(idx[u], idx[u].size, replace=True)) for u in uniq]))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    CI_COUNT[0] += 1
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_boot": n_boot, "seed": seed, "n_nan_draws": int(np.isnan(draws).sum())}


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(VAL)
    git = git_sha(ROOT); t_all = time.perf_counter()
    dev = torch.device("cuda")
    prov = {"git": git, "prereg": PREREG, "utc_start": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
            "libs": {"torch": torch.__version__, "numpy": np.__version__, "neurokit2": neurokit2.__version__, "python": platform.python_version()},
            "gpu": torch.cuda.get_device_name(0), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "scaffold_batch": RT.MICRO_BATCH, "scoring_imported_from": "scripts/r2_evaluate.py"}
    pt = subprocess.run(["python", "-m", "pytest", "tests/test_r3_rhythm_fusion.py", "-q", "-rs", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    prov["tests_ran"] = {"exit": pt.returncode, "summary": next((ln for ln in reversed(pt.stdout.splitlines()) if "passed" in ln or "failed" in ln), ""),
                         "skipped": [ln for ln in pt.stdout.splitlines() if ln.startswith("SKIPPED")]}
    if pt.returncode != 0:
        raise RuntimeError("R3 tests fail; not evaluating")
    pf = json.loads((ART / "runtime_preflight.json").read_text())
    tprov = {arm: json.loads((ART / f"train_provenance_{arm}.json").read_text()) for arm in RF.TRAINED_ARMS}
    hashes = {"preflight": pf["probe_hash"], **{arm: tprov[arm]["probe_hash"] for arm in RF.TRAINED_ARMS}, "r2": RF.R2_PROBE_HASH}
    if len(set(hashes.values())) != 1 or any(tprov[arm]["opt_steps"] != RF.STEPS for arm in RF.TRAINED_ARMS):
        raise RuntimeError(f"probe hashes or step counts differ across processes (STOP): {hashes}")
    prov |= {"probe_hashes": hashes, "runtime_preflight": pf, "training": tprov,
             "frozen_checkpoint_manifest": json.loads((ART / "frozen_checkpoint_manifest.json").read_text()),
             "initialization_hashes": json.loads((ART / "initialization_hashes.json").read_text())}

    # ---------------- population (frozen subset, asserted) ----------------
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
        idx = ER.select_subset(SALT, s, len(Xs), TAKE)
        X.append(Xs[idx].astype(np.float32)); Y.append(Ys[idx].astype(np.float32)); SUB.append(np.full(len(idx), s)); SITE.append(Ss[idx]); POS.append(idx); WI.append(Ws[idx].astype(np.int64))
    X, Y, SUB, SITE, POS, WI = (np.concatenate(v) for v in (X, Y, SUB, SITE, POS, WI))
    frozen = json.loads((ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json").read_text())
    for s in VAL:
        assert POS[SUB == s].tolist() == list(frozen[s]), f"frozen subset mismatch {s}"
    Yd = Y.astype(np.float64)
    gt_pk = R2E.pmap(R2E._peaks, list(Yd))
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    src_hash = hashlib.sha256(e0.numpy().tobytes()).hexdigest()
    assert src_hash == "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f", src_hash
    prov |= {"population_windows": int(len(X)), "gt_beats": int(sum(len(p) for p in gt_pk)), "source_bank_sha256": src_hash}
    if len(X) != 2048 or prov["gt_beats"] != 19834:
        raise RuntimeError("frozen population facts differ (STOP)")
    print(f"[P] {len(X)} windows, {prov['gt_beats']} GT beats, bank {src_hash[:16]} | HEAD {git['commit'][:8]}", flush=True)

    # ---------------- frozen components, R2 adapters, R3 modules, scaffolds ----------------
    ck = torch.load(ROOT / RT.GENERATOR_CKPT, map_location="cpu", weights_only=False); cfg = ck.get("imf_cfg", {})
    assert RT.state_dict_sha256(ck["state_dict"]) == RT.EXPECTED_GENERATOR_STATE_SHA
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    net_add, _ck2, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev); net_add.requires_grad_(False)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    add_sd = torch.load(ROOT / RF.R2_ADD_CKPT, map_location="cpu", weights_only=False)
    ora_sd = torch.load(ROOT / RF.R2_ORACLE_CKPT, map_location="cpu", weights_only=False)
    assert add_sd["arm"] == "true" and ora_sd["arm"] == "oracle" and add_sd["generator_state_sha256"] == gmeta["state_dict_sha256"]
    assert RT.state_dict_sha256(add_sd["state_dict"]).startswith(RF.EXPECTED_R2_ADD_STATE_SHA_PREFIX)
    assert RT.state_dict_sha256(ora_sd["state_dict"]).startswith(RF.EXPECTED_R2_ORACLE_STATE_SHA_PREFIX)
    h_dim = int(ck["model_cfg"]["h_dim"])
    nets = {}
    for fam, gm in (("tf", None), ("gtf", "adaptive"), ("gtf_const", "const")):
        m = RF.build_r3_module("gtf" if fam.startswith("gtf") else "tf", gm, c_hidden=h_dim)
        n = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), m, cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0))
        missing, unexpected = n.load_state_dict(ck["state_dict"], strict=False)
        assert unexpected == [] and set(missing) == {"r3." + k for k in RF.FAMILY_PARAM_NAMES[m.family]}
        n.requires_grad_(False); nets[fam] = n.to(dev).eval()
    modules = {}
    for arm in RF.TRAINED_ARMS:
        mk = torch.load(ROOT / f"outputs/r3_{arm}_seed42/module_step{RF.STEPS}.pt", map_location="cpu", weights_only=False)
        assert mk["step"] == RF.STEPS and mk["arm"] == arm and mk["generator_state_sha256"] == gmeta["state_dict_sha256"] and mk["probe_hash"] == RF.R2_PROBE_HASH
        assert mk["rhythm_state_sha256"] == tmeta["state_dict_sha256"] and mk["init_sha256"]["full"] == prov["initialization_hashes"]["arms"][arm]["full"]
        modules[RF.ARM_EVAL_NAME[arm]] = mk["state_dict"]
    man = [r for r in csv.DictReader(open(ART / "shuffle_manifest.csv")) if r["population"] == "eval"]
    partner = np.full(len(X), -1, dtype=np.int64)
    for r in man:
        partner[int(r["pop_row"])] = int(r["partner_pop_row"])
    RT.assert_derangement(partner); assert np.array_equal(partner, RT.shuffle_partner(SUB, SITE, WI))
    s_true = R2E.scaffolds(tcn, X, dev); s_shuffle = s_true[partner]; s_oracle = RT.oracle_fields(Y, workers=R2E.WORKERS)
    s_phase = np.roll(s_true, RF.PHASE_SHIFT_SAMPLES, axis=1)
    prov["scaffold_stats_eval"] = {"max_mean": float(s_true.max(1).mean()), "mean_mean": float(s_true.mean())}
    NET = {"TF-TRUE": "tf", "TF-SHUFFLE": "tf", "GTF-TRUE": "gtf", "GTF-SHUFFLE": "gtf", "GTF-CONST": "gtf_const", "GTF-ORACLE": "gtf"}
    SCAF = {"TF-TRUE": s_true, "TF-SHUFFLE": s_shuffle, "GTF-TRUE": s_true, "GTF-SHUFFLE": s_shuffle, "GTF-CONST": s_true, "GTF-ORACLE": s_oracle}

    # ---------------- parity STOPs (section 10) ----------------
    t_gen = {}
    t0 = time.perf_counter(); p_B4 = R2E.gen_plain(base, X, e0, 4, dev); t_gen["B"] = time.perf_counter() - t0
    parity = {}
    for fam in ("tf", "gtf"):
        fresh = RF.build_r3_module("gtf" if fam == "gtf" else "tf", "adaptive" if fam == "gtf" else None, c_hidden=h_dim).state_dict()
        sd0 = torch.load(ROOT / f"outputs/r3_{'gtf_true' if fam == 'gtf' else 'tf_true'}_seed42/module_step0.pt", map_location="cpu", weights_only=False)["state_dict"]
        if not all(torch.equal(sd0[k], fresh[k]) for k in fresh):
            raise RuntimeError(f"module_step0.pt of {fam} differs from the fresh seed-42 module (STOP)")
        attn_out = {}
        hk = nets[fam].r3.fusion.attn.register_forward_hook(lambda m, i, o: attn_out.setdefault("finite", bool(torch.isfinite(o).all())) and None)
        for name, sc in (("zero", None), ("true", s_true), ("shuffled", s_shuffle)):
            pz = gen_r3(nets[fam], fresh, X, e0, sc, 4, dev)
            parity[f"{fam}_{name}"] = bool(np.array_equal(pz, p_B4))
            if not parity[f"{fam}_{name}"]:
                raise RuntimeError(f"STEP-0 PARITY FAILED for {fam} with {name} scaffold (STOP)")
        hk.remove(); parity[f"{fam}_attn_finite"] = attn_out.get("finite", False); parity[f"{fam}_step0_ckpt_equals_fresh"] = True
        if not parity[f"{fam}_attn_finite"]:
            raise RuntimeError(f"attention output not finite for {fam} (STOP)")
    t0 = time.perf_counter(); p_add_r2path = R2E.gen_preds(net_add, X, e0, s_true, add_sd["state_dict"], 4, dev); t_gen["ADD"] = time.perf_counter() - t0
    p_add_local = gen_add_local(net_add, add_sd["state_dict"], X, e0, s_true, 4, dev)
    parity["add_same_process_torch_equal"] = bool(np.array_equal(p_add_r2path, p_add_local))
    if not parity["add_same_process_torch_equal"]:
        raise RuntimeError("ADD parity failed (STOP)")
    prov["parity"] = parity
    print(f"[B] parity OK: {parity}", flush=True)

    # ---------------- generation + scoring grid ----------------
    grid = [(arm, n) for arm in RF.ARMS for n in NFES] + [("PHASE-TF", 4), ("PHASE-GTF", 4), ("NODIRECT-TF", 4), ("NODIRECT-GTF", 4)]
    per, pks, errs, grid_t, win_rows = {}, {}, {}, {}, []
    for arm, n in grid:
        t_g = time.perf_counter()
        if arm == "B":
            pred = p_B4 if n == 4 else R2E.gen_plain(base, X, e0, n, dev)
        elif arm == "ADD":
            pred = p_add_r2path if n == 4 else R2E.gen_preds(net_add, X, e0, s_true, add_sd["state_dict"], n, dev)
        elif arm == "ADD-ORACLE":
            pred = R2E.gen_preds(net_add, X, e0, s_oracle, ora_sd["state_dict"], n, dev)
        elif arm.startswith("PHASE-"):
            src = "TF-TRUE" if arm.endswith("TF") else "GTF-TRUE"
            pred = gen_r3(nets[NET[src]], modules[src], X, e0, s_phase, 4, dev)
        elif arm.startswith("NODIRECT-"):
            src = "TF-TRUE" if arm.endswith("TF") else "GTF-TRUE"
            pred = gen_r3(nets[NET[src]], modules[src], X, e0, s_true, 4, dev, cancel_direct=True)
        else:
            pred = gen_r3(nets[NET[arm]], modules[arm], X, e0, SCAF[arm], n, dev)
        pred = pred.astype(np.float64)
        rows, pk, er = R2E.score(pred, Yd, gt_pk)
        grid_t[(arm, n)] = time.perf_counter() - t_g + (t_gen.get(arm, 0.0) if n == 4 else 0.0)
        per[(arm, n)], pks[(arm, n)], errs[(arm, n)] = rows, pk, er
        mr = R2E.macro_rows(rows, SUB)
        print(f"[E] {arm:12s} NFE{n} F1 {mr['f1']:.4f} excess {mr['f1_excess']:.4f} miss {mr['missing']:.3f} spur {mr['spurious']:.3f} bdev {mr['beats_ratio_dev']:.4f} | "
              f"rmse {mr['raw_rmse']:.4f} corr {mr['raw_corr']:.4f} qrs {mr['raw_qrs_rmse']:.4f} d1 {mr['qrs_deriv_rmse']:.4f} curv {mr['qrs_curvature_err']:.4f}", flush=True)
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        for i, r in enumerate(rows):
            win_rows.append({"arm": arm, "label": label(arm), "nfe": n, "subject": SUB[i], "array_pos": int(POS[i]), "npz_window_index": int(WI[i]), "site": SITE[i], **{k: r[k] for k in keys}})
        del pred
    R2E.wcsv(ART / "metrics_by_window.csv", win_rows)
    ev_rows, st_rows = [], []
    for arm, n in grid:
        mr = R2E.macro_rows(per[(arm, n)], SUB); e = np.concatenate(errs[(arm, n)]) if errs[(arm, n)] else np.zeros(0)
        ev_rows.append({"arm": label(arm), "nfe": n, **{k: mr[k] for k in ("f1", "chance_f1", "f1_excess", "precision", "recall", "matched_coverage", "missing", "spurious", "beats_ratio", "beats_ratio_dev", "n_windows_empty_pred")},
                        "timing_median_abs_ms": float(np.median(np.abs(e))) if e.size else np.nan, "timing_mean_ms": float(np.mean(e)) if e.size else np.nan, "n_matched_beats": int(e.size)})
        st_rows.append({"arm": label(arm), "nfe": n, **{k: mr[k] for k in ("raw_rmse", "raw_corr", "raw_qrs_rmse", "qrs_deriv_rmse", "qrs_curvature_err", "qrs_e_dev", "p2p_dev", "slope_dev", "hf_ratio", "hf_gt", "hf_err", "ww_rmse", "ww_corr", "qrs_rmse_core")}})
    R2E.wcsv(ART / "event_metrics.csv", ev_rows); R2E.wcsv(ART / "structure_metrics.csv", st_rows)

    # ---------------- regression vs R2 stored rows (B, ADD; ADD-ORACLE flagged) ----------------
    r2win = list(csv.DictReader(open(R2ART / "metrics_by_window.csv")))
    numkeys = [k for k, v in per[("B", 4)][0].items() if isinstance(v, (int, float))]
    reg = {}
    for arm_r3, arm_r2 in (("B", "B"), ("ADD", "TRUE"), ("ADD-ORACLE", "ORACLE")):
        ref = {(r["subject"], int(r["array_pos"])): r for r in r2win if r["arm"] == arm_r2 and int(float(r["nfe"])) == 4}
        rows3 = [dict(r, subject=str(SUB[i]), array_pos=int(POS[i])) for i, r in enumerate(per[(arm_r3, 4)])]
        reg[arm_r3] = regression(rows3, ref, [k for k in numkeys if k in next(iter(ref.values()))])
        mr3 = R2E.macro_rows(per[(arm_r3, 4)], SUB)
        r2ev = next(r for r in csv.DictReader(open(R2ART / "event_metrics.csv")) if r["arm"].startswith(arm_r2) and int(float(r["nfe"])) == 4)
        r2st = next(r for r in csv.DictReader(open(R2ART / "structure_metrics.csv")) if r["arm"].startswith(arm_r2) and int(float(r["nfe"])) == 4)
        reg[arm_r3]["macro_delta"] = {k: float(mr3[k]) - float(v) for src_ in (r2ev, r2st) for k, v in src_.items() if k in mr3 and v not in ("", None)}
        reg[arm_r3]["macro_flags_gt_1e-6"] = [k for k, v in reg[arm_r3]["macro_delta"].items() if abs(v) > 1e-6]
        stop_keys = ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err")
        if arm_r3 in ("B", "ADD") and any(abs(reg[arm_r3]["macro_delta"][k]) > 1e-6 for k in stop_keys):
            raise RuntimeError(f"{arm_r3} macro regression vs R2 exceeds 1e-6 (STOP): { {k: reg[arm_r3]['macro_delta'][k] for k in stop_keys} }")
    prov["regression_vs_r2"] = reg
    print("[R] regression vs R2: flags", {k: (v["n_flags_gt_1e-6"], v["macro_flags_gt_1e-6"]) for k, v in reg.items()}, flush=True)

    # ---------------- paired contrasts ----------------
    pb = []
    for first, second in CONTRASTS:
        pb += paired3(per, first, second, 4, PAIRED_ALL, SUB)
    for n in (1, 2):
        for first, second in CONTRASTS_NFE12:
            pb += paired3(per, first, second, n, EVENT4, SUB)
    og = []
    for first, second in ORACLE_CONTRASTS:
        og += paired3(per, first, second, 4, PAIRED_ALL, SUB)
    R2E.wcsv(ART / "paired_bootstrap.csv", pb); R2E.wcsv(ART / "oracle_diagnostic.csv", og)
    for r in pb:
        if r["nfe"] == 4 and r["metric"] in EVENT4 + STRUCT5:
            print(f"[PB] {r['comparison']:26s} {r['metric']:18s} {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)
    for r in og:
        if r["metric"] in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err"):
            print(f"[OG] {r['comparison']:26s} {r['metric']:18s} {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)

    # ---------------- decision (sections 15-17) ----------------
    V = lambda a, b, m: get(pb, f"{a}_vs_{b}@NFE4", m)  # noqa: E731
    tb, ts, ta = (lambda m: V("TF-TRUE", "B", m)), (lambda m: V("TF-TRUE", "TF-SHUFFLE", m)), (lambda m: V("TF-TRUE", "ADD", m))
    gb, gs, gc, gt, ga = (lambda m: V("GTF-TRUE", "B", m)), (lambda m: V("GTF-TRUE", "GTF-SHUFFLE", m)), (lambda m: V("GTF-TRUE", "GTF-CONST", m)), (lambda m: V("GTF-TRUE", "TF-TRUE", m)), (lambda m: V("GTF-TRUE", "ADD", m))
    imp = lambda r: r["verdict"] == "improves"; wor = lambda r: r["verdict"] == "worsens"  # noqa: E731
    mr_tf, mr_gtf = R2E.macro_rows(per[("TF-TRUE", 4)], SUB), R2E.macro_rows(per[("GTF-TRUE", 4)], SUB)
    rec = {"U1": imp(tb("f1_excess")) and tb("f1_excess")["point"] >= RF.GATE_MIN_EFFECT, "U1_point": float(tb("f1_excess")["point"]), "U1_ci_positive": imp(tb("f1_excess")),
           "U2": imp(ts("f1_excess")), "U3": any(imp(tb(m)) for m in ("missing", "spurious", "beats_ratio_dev")), "U3_carriers": [m for m in ("missing", "spurious", "beats_ratio_dev") if imp(tb(m))],
           "U4": not (wor(tb("qrs_deriv_rmse")) or wor(tb("qrs_curvature_err"))),
           "U5": (imp(ta("qrs_deriv_rmse")) and not wor(ta("qrs_curvature_err"))) or (imp(ta("qrs_curvature_err")) and not wor(ta("qrs_deriv_rmse"))),
           "U6": bool(mr_tf["beats_ratio_dev"] < RF.GATE_BEATS_DEV_MAX), "U6_value": float(mr_tf["beats_ratio_dev"]),
           "G1": imp(gb("f1_excess")) and gb("f1_excess")["point"] >= RF.GATE_MIN_EFFECT, "G1_point": float(gb("f1_excess")["point"]), "G1_ci_positive": imp(gb("f1_excess")),
           "G2": imp(gs("f1_excess")), "G3": not (wor(gb("qrs_deriv_rmse")) or wor(gb("qrs_curvature_err"))),
           "G4": ga("qrs_deriv_rmse")["point"] > 0 and ga("qrs_curvature_err")["point"] > 0 and (imp(ga("qrs_deriv_rmse")) or imp(ga("qrs_curvature_err"))) and not (wor(ga("qrs_deriv_rmse")) or wor(ga("qrs_curvature_err"))),
           "g5_noninferior": bool(gc("f1_excess")["lo"] > RF.NONINFERIORITY_MARGIN), "g5_structure": imp(gc("qrs_deriv_rmse")) or imp(gc("qrs_curvature_err")),
           "G6": bool(mr_gtf["beats_ratio_dev"] < RF.GATE_BEATS_DEV_MAX), "G6_value": float(mr_gtf["beats_ratio_dev"]),
           "const_superior_on_f1_within_margin": wor(gc("f1_excess")) and bool(gc("f1_excess")["lo"] > RF.NONINFERIORITY_MARGIN),
           "gc_f1_ci_halfwidth": float((gc("f1_excess")["hi"] - gc("f1_excess")["lo"]) / 2)}
    rec["G5"] = rec["g5_noninferior"] and rec["g5_structure"]
    rec["ev_tf"], rec["ev_gtf"] = rec["U1"] and rec["U2"], rec["G1"] and rec["G2"]
    rec["ev_ci_tf"], rec["ev_ci_gtf"] = rec["U1_ci_positive"] and rec["U2"], rec["G1_ci_positive"] and rec["G2"]
    rec["deg_tf"] = wor(tb("qrs_deriv_rmse")) or wor(tb("qrs_curvature_err")) or not rec["U5"]
    rec["deg_gtf"] = wor(gb("qrs_deriv_rmse")) or wor(gb("qrs_curvature_err")) or not rec["G4"]
    rec["gtf_vs_tf_structure"] = (imp(gt("qrs_deriv_rmse")) and not wor(gt("qrs_curvature_err"))) or (imp(gt("qrs_curvature_err")) and not wor(gt("qrs_deriv_rmse")))
    rec["gtf_vs_tf_protects"] = rec["gtf_vs_tf_structure"] and not any(wor(gt(m)) for m in EVENT4)
    rec["shuffle_share_tf"] = float(V("TF-SHUFFLE", "B", "f1_excess")["point"] / tb("f1_excess")["point"]) if tb("f1_excess")["point"] else np.nan
    rec["shuffle_share_gtf"] = float(V("GTF-SHUFFLE", "B", "f1_excess")["point"] / gb("f1_excess")["point"]) if gb("f1_excess")["point"] else np.nan
    rec["shuffle_vs_B_tf"], rec["shuffle_vs_B_gtf"] = V("TF-SHUFFLE", "B", "f1_excess")["verdict"], V("GTF-SHUFFLE", "B", "f1_excess")["verdict"]
    verdict = RF.decide_verdict_r3(rec)
    o = lambda a, b, m: get(og, f"{a}_vs_{b}@NFE4", m)  # noqa: E731
    oracle = {"gtf_oracle_vs_add_oracle_f1": {k: o("GTF-ORACLE", "ADD-ORACLE", "f1_excess")[k] for k in ("point", "lo", "hi", "verdict")},
              "gtf_oracle_vs_b": {m: o("GTF-ORACLE", "B", m)["verdict"] for m in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err")},
              "gtf_oracle_vs_gtf_true_f1": o("GTF-ORACLE", "GTF-TRUE", "f1_excess")["verdict"], "add_oracle_vs_b_f1": o("ADD-ORACLE", "B", "f1_excess")["point"]}
    oracle["arms"] = {a: label(a) for a in RF.ORACLE_ARMS}
    oracle["reading"] = RF.oracle_reading(oracle["gtf_oracle_vs_add_oracle_f1"]["verdict"], oracle["gtf_oracle_vs_add_oracle_f1"]["point"],
                                          oracle["gtf_oracle_vs_b"]["qrs_deriv_rmse"], oracle["gtf_oracle_vs_b"]["qrs_curvature_err"])
    dec = {**verdict, "record": rec, "oracle": oracle, "status": {"U": {f"U{i}": rec[f"U{i}"] for i in range(1, 7)}, "G": {f"G{i}": rec[f"G{i}"] for i in range(1, 7)}},
           "tb": {m: {k: tb(m)[k] for k in ("point", "lo", "hi", "verdict")} for m in EVENT4 + STRUCT5}, "gb": {m: {k: gb(m)[k] for k in ("point", "lo", "hi", "verdict")} for m in EVENT4 + STRUCT5},
           "nfe": 4, "population": "frozen 2048 (x4-event-nfe-v2)", "prereg": PREREG, "oracle_label": RF.ORACLE_LABEL}
    (ART / "decision.json").write_text(json.dumps(dec, indent=2, default=float))
    print(f"\n[U] {dec['status']['U']} (U1 point {rec['U1_point']:+.4f}, U3 {rec['U3_carriers']})\n[G] {dec['status']['G']} (G1 point {rec['G1_point']:+.4f}; g5 noninf {rec['g5_noninferior']} struct {rec['g5_structure']})\n"
          f"[EV] ev_tf {rec['ev_tf']} deg_tf {rec['deg_tf']} | ev_gtf {rec['ev_gtf']} deg_gtf {rec['deg_gtf']} | gtf protects vs tf {rec['gtf_vs_tf_protects']}\n"
          f"[ORACLE] {oracle['reading']} ({oracle['gtf_oracle_vs_add_oracle_f1']})\n[VERDICT] {verdict['verdict']} {verdict.get('codes')} necessity={verdict.get('necessity')}", flush=True)
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    # ---------------- phase ablation (18.1) and direct-route attribution (18.2) ----------------
    phi = np.array([RT.phase_phi(p) for p in gt_pk]); strat = np.array([RT.phi_stratum(v) for v in phi])
    PH6 = EVENT4 + ("qrs_deriv_rmse", "qrs_curvature_err")
    ph_rows, ph_sum = [], []
    for src, ph in (("TF-TRUE", "PHASE-TF"), ("GTF-TRUE", "PHASE-GTF")):
        for i in range(len(X)):
            ph_rows.append({"arm": src, "pop_row": i, "subject": SUB[i], "phi": phi[i], "stratum": strat[i], **{f"true_{m}": per[(src, 4)][i][m] for m in PH6}, **{f"shifted_{m}": per[(ph, 4)][i][m] for m in PH6}})
        for tag, mask in (("all", np.ones(len(X), bool)), ("in_phase", strat == "in_phase"), ("anti_phase", strat == "anti_phase"), ("rest", strat == "rest")):
            for r in subset_paired(per[(ph, 4)], per[(src, 4)], SUB, mask, PH6, "SHIFTED", src, tag):
                ph_sum.append({"arm": src, **r})
    R2E.wcsv(ART / "phase_ablation.csv", ph_rows); R2E.wcsv(ART / "phase_ablation_summary.csv", ph_sum)
    for r in ph_sum:
        if r["metric"] == "f1_excess":
            print(f"[PH] {r['arm']:8s} {r['stratum']:10s} n={r['n']:4d} TRUE-vs-SHIFTED f1_excess {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)
    dr_rows = []
    for src, nd in (("TF-TRUE", "NODIRECT-TF"), ("GTF-TRUE", "NODIRECT-GTF")):
        gain_full = tb("f1_excess")["point"] if src == "TF-TRUE" else gb("f1_excess")["point"]
        vs_b = paired3(per, nd, "B", 4, ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err"), SUB)
        vs_full = paired3(per, nd, src, 4, ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err"), SUB)
        mrn, mrf = R2E.macro_rows(per[(nd, 4)], SUB), R2E.macro_rows(per[(src, 4)], SUB)
        share = float(get(vs_b, f"{nd}_vs_B@NFE4", "f1_excess")["point"] / gain_full) if gain_full else np.nan
        dr_rows.append({"arm": src, "f1_excess_full": mrf["f1_excess"], "f1_excess_no_direct": mrn["f1_excess"], "S4_full": mrf["qrs_deriv_rmse"], "S4_no_direct": mrn["qrs_deriv_rmse"],
                        "S5_full": mrf["qrs_curvature_err"], "S5_no_direct": mrn["qrs_curvature_err"], "gain_vs_B_full": gain_full,
                        "gain_vs_B_no_direct": get(vs_b, f"{nd}_vs_B@NFE4", "f1_excess")["point"], "surviving_share": share,
                        "reading": ("n/a (no positive F1-excess gain vs B)" if not (gain_full > 0) else ("through the target stream" if share >= 0.5 else "predominantly a direct decoder-input write")),
                        **{f"nodirect_vs_full_{m}_point": get(vs_full, f"{nd}_vs_{src}@NFE4", m)["point"] for m in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err")},
                        **{f"nodirect_vs_full_{m}_verdict": get(vs_full, f"{nd}_vs_{src}@NFE4", m)["verdict"] for m in ("f1_excess", "qrs_deriv_rmse", "qrs_curvature_err")}})
        print(f"[DR] {src}: gain vs B {gain_full:+.4f} -> without direct route {dr_rows[-1]['gain_vs_B_no_direct']:+.4f} (share {share:.2f}) => {dr_rows[-1]['reading']}", flush=True)
    R2E.wcsv(ART / "direct_route_attribution.csv", dr_rows)

    # ---------------- NFE persistence (21) ----------------
    pe_rows, pe_sum, beat_sets = [], {}, {}
    for arm in PERSIST_ARMS:
        D = []
        for i in range(len(X)):
            d = RT.persistence_deltas(gt_pk[i], {n: pks[(arm, n)][i] for n in NFES})
            for b in range(len(gt_pk[i])):
                pe_rows.append({"arm": arm, "subject": SUB[i], "pop_row": i, "beat": b, "delta_1": d[b, 0], "delta_2": d[b, 1], "delta_4": d[b, 2]})
            D.append(d)
        D = np.concatenate(D); m1, m2, m4 = ~np.isnan(D[:, 0]), ~np.isnan(D[:, 1]), ~np.isnan(D[:, 2]); all3, m14 = m1 & m2 & m4, m1 & m4
        pe_sum[arm] = {"match_frac": {n: float(m.mean()) for n, m in zip(NFES, (m1, m2, m4))},
                       "mean_abs_delta": {n: float(np.abs(D[m, j]).mean()) for j, (n, m) in enumerate(zip(NFES, (m1, m2, m4)))},
                       "median_abs_delta": {n: float(np.median(np.abs(D[m, j]))) for j, (n, m) in enumerate(zip(NFES, (m1, m2, m4)))},
                       "sign_consistency_1_4": float(np.mean(np.sign(D[all3, 0]) == np.sign(D[all3, 2]))) if all3.any() else np.nan,
                       "frac_nfe4_closer_strict": float(np.mean(np.abs(D[m14, 2]) < np.abs(D[m14, 0]))) if m14.any() else np.nan,
                       "frac_tie": float(np.mean(np.abs(D[m14, 2]) == np.abs(D[m14, 0]))) if m14.any() else np.nan, "n_m14": int(m14.sum())}
        beat_sets[arm] = {"m14": np.flatnonzero(m14), "all3": np.flatnonzero(all3), "D": D}
        print(f"[NP] {arm:8s} match {pe_sum[arm]['match_frac']} |d| {pe_sum[arm]['mean_abs_delta']} closer@4 {pe_sum[arm]['frac_nfe4_closer_strict']:.3f}", flush=True)
    inter = np.array(sorted(set.intersection(*[set(v["m14"].tolist()) for v in beat_sets.values()])))
    inter3 = np.array(sorted(set.intersection(*[set(v["all3"].tolist()) for v in beat_sets.values()])))
    pe_sum["intersection"] = {"n_m14_all_arms": int(inter.size), "n_all3_all_arms": int(inter3.size)}
    for arm in PERSIST_ARMS:
        D = beat_sets[arm]["D"]
        if inter.size:
            pe_sum["intersection"][f"{arm}_frac_nfe4_closer_strict"] = float(np.mean(np.abs(D[inter, 2]) < np.abs(D[inter, 0])))
            pe_sum["intersection"][f"{arm}_mean_abs_delta1"] = float(np.abs(D[inter, 0]).mean()); pe_sum["intersection"][f"{arm}_mean_abs_delta4"] = float(np.abs(D[inter, 2]).mean())
        if inter3.size:
            pe_sum["intersection"][f"{arm}_sign_consistency_1_4"] = float(np.mean(np.sign(D[inter3, 0]) == np.sign(D[inter3, 2])))
    R2E.wcsv(ART / "nfe_event_persistence.csv", pe_rows); (ART / "nfe_event_persistence_summary.json").write_text(json.dumps(pe_sum, indent=2, default=float))

    # ---------------- gate diagnostics (20) ----------------
    gmod = RF.build_r3_module("gtf", "adaptive", c_hidden=h_dim).to(dev).eval(); gmod.load_state_dict({k: v.to(dev) for k, v in modules["GTF-TRUE"].items()})
    G = gate_values(gmod, s_true, dev)                                  # [2048, 1024]
    g_win = G.mean(1); g_std_all = float(G.std())
    sq50 = np.array([RT.scaffold_event_f1(s_true[i], gt_pk[i], 0.35, 50.0) for i in range(len(X))])
    sq150 = np.array([RT.scaffold_event_f1(s_true[i], gt_pk[i], 0.35, 150.0) for i in range(len(X))])
    def rho(x, y):
        def f(idx):
            if len(set(x[idx].tolist())) < 2 or len(set(y[idx].tolist())) < 2:
                return float("nan")
            return float(spearmanr(x[idx], y[idx]).correlation)
        return subject_boot(f, SUB)
    rA, rB = rho(g_win, sq50), rho(g_win, sq150)
    # (C) gate around scaffold peaks: matched vs unmatched at 50 ms, +-4 samples
    pk_m, pk_u = [[] for _ in range(len(X))], [[] for _ in range(len(X))]
    for i in range(len(X)):
        ev = extract_events(s_true[i], 0.35); m, _, _ = R.match_rpeaks(np.asarray(gt_pk[i]), ev, FS, 50.0); matched = {j for _, j in m}
        for j, p in enumerate(ev):
            v = float(G[i, max(0, p - 4):min(T_LEN, p + 5)].mean()); (pk_m if j in matched else pk_u)[i].append(v)
    def cdiff(idx):
        a = [v for i in idx for v in pk_m[i]]; b = [v for i in idx for v in pk_u[i]]
        return float(np.mean(a) - np.mean(b)) if a and b else float("nan")
    rC = subject_boot(cdiff, SUB)
    nC = {"n_matched_peaks": int(sum(len(v) for v in pk_m)), "n_unmatched_peaks": int(sum(len(v) for v in pk_u)), "gate_matched_mean": float(np.mean([v for l in pk_m for v in l])), "gate_unmatched_mean": float(np.mean([v for l in pk_u for v in l]))}
    med = float(np.median(sq50)); good = sq50 >= med
    def ediff(idx):
        gi, pi = [i for i in idx if good[i]], [i for i in idx if not good[i]]
        return float(g_win[gi].mean() - g_win[pi].mean()) if gi and pi else float("nan")
    rE = subject_boot(ediff, SUB)
    E = {"good_gate_mean": float(g_win[good].mean()), "poor_gate_mean": float(g_win[~good].mean()), "good_n": int(good.sum()), "poor_n": int((~good).sum()), "median_scaffold_f1_50": med, **{f"diff_{k}": v for k, v in rE.items()}}
    wording = "RELIABILITY-LIKE BEHAVIOR OBSERVED" if (rA["point"] >= 0.20 and rA["lo"] > 0 and rC["lo"] > 0 and g_std_all >= 0.01) else "GATE NOT INTERPRETABLE AS CONFIDENCE"
    gd = [{"stat": "A_spearman_gate_vs_scaffold_f1_50", **rA}, {"stat": "B_spearman_gate_vs_scaffold_f1_150", **rB}, {"stat": "C_gate_matched_minus_unmatched_peaks", **rC, **nC},
          {"stat": "E_good_vs_poor_extraction", **E}, {"stat": "gate_global", "mean": float(G.mean()), "std": g_std_all, "p10": float(np.percentile(G, 10)), "p90": float(np.percentile(G, 90)),
           "window_mean_std": float(g_win.std())}, {"stat": "wording", "value": wording}]
    for arm_name in ("GTF-CONST", "GTF-SHUFFLE"):
        mm = RF.build_r3_module("gtf", "const" if arm_name == "GTF-CONST" else "adaptive", c_hidden=h_dim).to(dev).eval(); mm.load_state_dict({k: v.to(dev) for k, v in modules[arm_name].items()})
        Gx = gate_values(mm, s_true if arm_name == "GTF-CONST" else s_shuffle, dev)
        gd.append({"stat": f"gate_global_{arm_name}", "mean": float(Gx.mean()), "std": float(Gx.std()), "window_mean_std": float(Gx.mean(1).std())})
    prov["gate_wording"] = wording
    print(f"[GATE] rhoA {rA['point']:+.3f} [{rA['lo']:+.3f},{rA['hi']:+.3f}] rhoB {rB['point']:+.3f} | matched-unmatched {rC['point']:+.4f} [{rC['lo']:+.4f},{rC['hi']:+.4f}] | g mean {G.mean():.3f} std {g_std_all:.4f} -> {wording}", flush=True)
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))

    # ---------------- site-wise secondary (19) ----------------
    t_site = time.perf_counter()
    proj = 4 * sum(grid_t[(arm, 4)] for arm in SITE_ARMS)
    prov["site_cohort"] = {"projected_s": proj, "budget_s": 7200.0}
    gd_site = []
    if proj > 7200.0:
        R2E.wcsv(ART / "site_metrics.csv", [{"site": "skipped", "reason": f"projected {proj:.0f} s > budget"}]); prov["site_cohort"]["skipped"] = True
    else:
        try:
            X8, Y8, SUB8, SITE8 = [], [], [], []
            for sub in VAL:
                d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz"); Xs, Ys, Ss, Ws = d["x"], d["y"], np.asarray(d["site"]).astype(str), d["window_index"]
                pos = C.cohort_positions(sub, Ss, Ws, C.n_per_for(sub))
                for site in C.SITES:
                    idx = pos[site]; X8.append(Xs[idx].astype(np.float32)); Y8.append(Ys[idx].astype(np.float32)); SUB8.append(np.full(len(idx), sub)); SITE8.append(np.full(len(idx), site))
            X8, Y8, SUB8, SITE8 = (np.concatenate(v) for v in (X8, Y8, SUB8, SITE8))
            e8 = torch.randn(len(X8), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
            prov["site_cohort"] |= {"windows": int(len(X8)), "source_bank_sha256": hashlib.sha256(e8.numpy().tobytes()).hexdigest()}
            Y8d = Y8.astype(np.float64); gt8 = R2E.pmap(R2E._peaks, list(Y8d)); s8 = R2E.scaffolds(tcn, X8, dev)
            per8 = {}
            for arm in SITE_ARMS:
                if arm == "B":
                    pred = R2E.gen_plain(base, X8, e8, 4, dev)
                elif arm == "ADD":
                    pred = R2E.gen_preds(net_add, X8, e8, s8, add_sd["state_dict"], 4, dev)
                else:
                    pred = gen_r3(nets[NET[arm]], modules[arm], X8, e8, s8, 4, dev)
                per8[arm], _, _ = R2E.score(pred.astype(np.float64), Y8d, gt8); del pred
            G8 = gate_values(gmod, s8, dev).mean(1)
            site_rows = []
            for site in C.SITES:
                m = SITE8 == site
                for arm in SITE_ARMS:
                    mr = R2E.macro_rows([per8[arm][i] for i in np.flatnonzero(m)], SUB8[m]); site_rows.append({"site": site, "arm": arm, "n_windows": int(m.sum()), **{k: mr[k] for k in SM7}})
                for first, second in (("GTF-TRUE", "B"), ("GTF-TRUE", "TF-TRUE"), ("TF-TRUE", "B"), ("ADD", "B"), ("GTF-CONST", "B")):
                    for r in subset_paired(per8[second], per8[first], SUB8, m, SM7, second, first, site):
                        site_rows.append({"site": site, "arm": f"{first}_vs_{second}", "n_windows": r["n"], "metric": r["metric"], "point": r["point"], "lo": r["lo"], "hi": r["hi"], "verdict": r["verdict"]})
                rD = subject_boot(lambda idx, m=m: float(G8[idx].mean()), SUB8[m]) if m.any() else {}
                gd_site.append({"stat": f"D_gate_by_site_{site}", "mean": float(G8[m].mean()), "p10": float(np.percentile(G8[m], 10)), "p90": float(np.percentile(G8[m], 90)), **{f"subject_boot_{k}": v for k, v in rD.items()}})
            R2E.wcsv(ART / "site_metrics.csv", site_rows)
            for r in site_rows:
                if r["arm"] in ("GTF-TRUE_vs_B", "GTF-TRUE_vs_TF-TRUE", "TF-TRUE_vs_B") and r.get("metric") == "f1_excess":
                    print(f"[S] {r['site']:8s} {r['arm']:20s} f1_excess {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}", flush=True)
        except Exception as ex:
            R2E.wcsv(ART / "site_metrics.csv", [{"site": "skipped", "reason": f"failed: {ex!r}"}]); prov["site_cohort"]["skipped"] = True; prov["site_cohort"]["error"] = repr(ex)
            print(f"[S] site-wise failed: {ex!r}", flush=True)
    R2E.wcsv(ART / "gate_diagnostics.csv", gd + gd_site)
    prov["site_cohort"]["wall_s"] = time.perf_counter() - t_site
    prov["nan_pattern_flags"] = NAN_FLAGS; prov["n_ci_computed"] = CI_COUNT[0]
    prov |= {"generator": gmeta, "rhythm_tcn": tmeta, "modules": {arm: str(ROOT / f"outputs/r3_{arm}_seed42/module_step{RF.STEPS}.pt") for arm in RF.TRAINED_ARMS},
             "wall_s": time.perf_counter() - t_all, "utc_end": datetime.now(timezone.utc).isoformat()}
    (ART / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    print(f"[done] R3 evaluation in {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
