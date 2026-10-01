"""E0 — WW-DET spontaneous event audit (docs/E0_WWDET_SPONTANEOUS_EVENT_AUDIT_PREREGISTRATION.md).

Analysis only. Frozen and never modified: split, timing detector, placed events, CoherentBeat-C0 (C0), WW-DET (C0-A),
metric and matching code. No model is trained; the only fitted object is the preregistered L2 logistic probe.
ARCH-VAL and ARCH-HOLDOUT were both analysed before (C0, C0-A): E0 is a post-hoc mechanistic audit of frozen outputs.

Stages: reproduce, manifest, train_infer, evaluate, figure, atlas
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/e0_event_audit.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
import c0a_ablation as CA  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import s1_audit as S1  # noqa: E402
from ppg2ecg.eventaudit import features as FT  # noqa: E402
from ppg2ecg.eventaudit import probe as PR  # noqa: E402
from ppg2ecg.eventaudit import taxonomy as TX  # noqa: E402

PREREG = "docs/E0_WWDET_SPONTANEOUS_EVENT_AUDIT_PREREGISTRATION.md"
CODE_FILES = ("scripts/e0_event_audit.py", "src/ppg2ecg/eventaudit/__init__.py", "src/ppg2ecg/eventaudit/taxonomy.py",
              "src/ppg2ecg/eventaudit/features.py", "src/ppg2ecg/eventaudit/probe.py", "tests/test_e0_event_audit.py")
ART = ROOT / "artifacts/e0_wwdet_event_audit"
OUT = ROOT / "outputs/e0_wwdet_event_audit"
FS, T = 128, 512
TOL_MS = 50.0
ARMS = ("C0", "WW")
REPRO_KEYS = ("f1", "precision", "recall", "fp", "fn", "rr_mae_ms", "hr_abs_err")
ROLES = ("val", "holdout")


# ----------------------------------------------------------------------------------------------- io
def write_json(name, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


def event_metrics(Y, ref, det, evaluable, w=None) -> dict:
    """C0 / C0-A per-window event metrics for one detection set (identical calls; `w` only sets the nan mask)."""
    w = Y if w is None else w
    prf = PMX.rpeak_prf_at(w, Y, FS, TOL_MS, peaks=(ref, det))
    bl = PMX.beat_level_metrics(w, Y, FS, TOL_MS, peaks=(ref, det))
    return {"f1": np.where(evaluable, prf["rpeak_f1"], np.nan), "precision": prf["rpeak_precision"], "recall": prf["rpeak_recall"],
            "fp": prf["n_fp"], "fn": prf["n_fn"], "rr_mae_ms": bl["rr_mae_ms"], "hr_abs_err": bl["hr_abs_err"]}


def render_role(role, ex, dev):
    """Frozen inputs and renders of one role, re-verified against C0's stored events and C0 renders."""
    X, Y, Pid = C0.load_arch(role)
    ref = C0.reference_peaks(role, Y, ex)
    if role == "train":
        events = C0.detect_events(X, dev, ex)
    else:
        st = np.load(C0.OUT / f"{role}_renders.npz")
        events = [np.asarray(e, int) for e in B.split_list(st["events_idx"], st["events_off"])]
        fresh = C0.detect_events(X, dev, ex)
        if not all(np.array_equal(a, b) for a, b in zip(events, fresh)):
            raise SystemExit(f"STOP: frozen C0 events not reproduced on {role}")
    c0, _, _ = C0.render_c0("c0", X, events, dev, C0.train_rr_median())
    if role != "train" and np.max(np.abs(c0.astype(np.float32) - st["C0"])) != 0.0:
        raise SystemExit(f"STOP: frozen C0 render not reproduced on {role}")
    ww = CA.render_ww(X, events, dev)
    return X, Y, Pid, ref, events, {"C0": c0, "WW": ww}


def save_cache(role, Pid, ref, events, waves, det):
    OUT.mkdir(parents=True, exist_ok=True)
    pk = {}
    for name, lst in (("ref", ref), ("events", events), ("det_C0", det["C0"]), ("det_WW", det["WW"])):
        pk[f"{name}_idx"], pk[f"{name}_off"] = B.pack_list(lst)
    np.savez(OUT / f"{role}_cache.npz", pid=Pid, C0=waves["C0"], WW=waves["WW"], **pk)


def load_cache(role):
    """(X, Y, Pid, ref, events, waves, det) from the frozen-output cache; X and Y are re-read from the frozen data."""
    X, Y, Pid = C0.load_arch(role)
    d = np.load(OUT / f"{role}_cache.npz")
    assert np.array_equal(d["pid"], Pid)
    lst = lambda n: [np.asarray(a, int) for a in B.split_list(d[f"{n}_idx"], d[f"{n}_off"])]  # noqa: E731
    return X, Y, Pid, lst("ref"), lst("events"), {"C0": d["C0"], "WW": d["WW"]}, {"C0": lst("det_C0"), "WW": lst("det_WW")}


# ----------------------------------------------------------------------------------------------- stage: reproduce
def input_hashes() -> dict:
    files = ["outputs/c0_coherentbeat/detector.pt", "outputs/c0_coherentbeat/c0.pt", "outputs/c0a_coherentbeat_ablation/ww.pt",
             "outputs/c0_coherentbeat/val_renders.npz", "outputs/c0_coherentbeat/holdout_renders.npz",
             "artifacts/c0_coherentbeat/split_manifest.json", "artifacts/c0_coherentbeat/checkpoint_hashes.json",
             "artifacts/c0a_coherentbeat_ablation/checkpoint_hashes.json", "artifacts/c0a_coherentbeat_ablation/val_metrics.json",
             "artifacts/c0a_coherentbeat_ablation/holdout_metrics.json", "scripts/c0_coherentbeat.py", "scripts/c0a_ablation.py",
             "src/ppg2ecg/coherentbeat/model.py", "src/ppg2ecg/coherentbeat/ablation.py", "src/ppg2ecg/evaluation/rpeaks.py",
             "src/ppg2ecg/evaluation/paper_metrics.py"]
    return {f: B.sha256_file(ROOT / f) for f in files}


def stage_reproduce(ex, dev):
    h = input_hashes()
    c0h = json.loads((C0.ART / "checkpoint_hashes.json").read_text())
    cah = json.loads((CA.ART / "checkpoint_hashes.json").read_text())
    ck = {"detector": h["outputs/c0_coherentbeat/detector.pt"] == c0h["detector"],
          "c0": h["outputs/c0_coherentbeat/c0.pt"] == c0h["c0"],
          "ww": h["outputs/c0a_coherentbeat_ablation/ww.pt"] == cah["ww"]}
    if not all(ck.values()):
        raise SystemExit(f"STOP: frozen checkpoint hash mismatch {ck}")
    write_json("input_hashes.json", {"sha256": h, "checkpoints_equal_recorded_hashes": ck})
    res = {}
    for role in ROLES:
        X, Y, Pid, ref, events, waves = render_role(role, ex, dev)
        det = {k: list(ex.map(B._peaks, list(waves[k]), chunksize=256)) for k in ARMS}
        evaluable = np.array([len(r) > 0 for r in ref])
        per = {"PLACED": event_metrics(Y, ref, events, evaluable)}
        for k in ARMS:
            per[k] = event_metrics(Y, ref, det[k], evaluable, waves[k])
        summ = {k: {m: C0.cluster_ci(v, Pid) for m, v in d.items()} for k, d in per.items()}
        stored = json.loads((CA.ART / f"{role}_metrics.json").read_text())
        dev_max = {k: max(abs(a - b) for m in REPRO_KEYS for a, b in zip(summ[k][m], stored["summary"][k][m])) for k in per}
        raw = {k: {m: int(np.nansum(per[k][m])) for m in ("fp", "fn")} for k in per}
        raw_ok = all(raw[k] == stored["raw"][k] for k in per)
        res[role] = {"windows": len(Y), "patients": int(np.unique(Pid).size), "max_abs_dev_vs_c0a": dev_max, "raw": raw,
                     "raw_equal_c0a": raw_ok, "summary": {k: {m: summ[k][m] for m in ("f1", "precision", "recall", "fp", "fn")}
                                                          for k in per}}
        if not (raw_ok and all(v <= 1e-9 for v in dev_max.values())):
            write_json("reproduction.json", res)
            raise SystemExit(f"STOP: C0-A event metrics not reproduced on {role}: {dev_max} raw_ok={raw_ok}")
        save_cache(role, Pid, ref, events, waves, det)
        print(f"[e0] {role}: reproduced (max dev {max(dev_max.values()):.2e}), raw {raw}", flush=True)
    write_json("reproduction.json", res)


# ----------------------------------------------------------------------------------------------- stage: manifest
def stage_manifest(ex, dev):
    """Hashes of the frozen E0 design, written with the preregistration commit (before any event-level outcome)."""
    write_json("prereg_manifest.json", {
        "prereg": {PREREG: B.sha256_file(ROOT / PREREG)}, "code": {f: B.sha256_file(ROOT / f) for f in CODE_FILES},
        "frozen_inputs": json.loads((ART / "input_hashes.json").read_text())["sha256"],
        "reproduction": B.sha256_file(ART / "reproduction.json"),
        "written_before_any_event_level_e0_outcome": True, "software": B.software()})


# ----------------------------------------------------------------------------------------------- taxonomy / features
def classify_role(ref, events, det) -> dict:
    """Per-window classification of every rendered detection, with accounting rows (lists indexed by window)."""
    return {i: TX.classify(ref[i], events[i], det[i]) for i in range(len(ref))}


def per_window_counts(cls, ref, events) -> dict:
    keys = ("A", "B", "C", "D", "L", "M", "tp_placed", "fp_placed", "tp_rendered", "fp_rendered")
    rows = [TX.accounting(cls[i], len(ref[i]), len(events[i])) for i in range(len(ref))]
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}


def _feat_job(a):
    """Features of every WW type-C / type-D detection of one window (worker)."""
    ww, c0, ppg, events, dets, types, template = a
    pk = S1.dsp_ppg_peaks(ppg, FS)
    out = []
    for t, ty in zip(dets, types):
        if ty not in ("C", "D"):
            out.append(None)
            continue
        w = FT.waveform_features(ww, int(t), template)
        out.append(w | FT.ppg_features(ppg, int(t), pk) | FT.c0_contrast(ww, c0, int(t), template, w))
    return out


def event_table(role, Pid, ref, events, det, cls, waves, X, template, ex, arms=ARMS) -> "pd.DataFrame":
    """One row per rendered detection (all arms): identities, type, localization; features for WW types C / D."""
    import pandas as pd
    rows = []
    for arm in arms:
        for i in range(len(ref)):
            c = cls[arm][i]
            for j, t in enumerate(det[arm][i]):
                loc = TX.localize(int(t), events[i], T, FS)
                rows.append({"role": role, "arm": arm, "window": i, "patient": int(Pid[i]), "sample": int(t), "type": str(c["type"][j]),
                             "ref_idx": int(c["ref_idx"][j]), "placed_idx": int(c["placed_idx"][j]), **loc})
    df = pd.DataFrame(rows)
    for k, v in FT.context_features({c: df[c] for c in ("dist_nearest_ms", "phase", "dist_prev_ms", "dist_next_ms")}).items():
        df[k] = v                                       # features 11-14 come from the localization columns of every row
    if "WW" in arms:
        win = [i for i in range(len(ref)) if np.isin(cls["WW"][i]["type"], ["C", "D"]).any()]
        jobs = [(waves["WW"][i], waves["C0"][i], X[i].astype(np.float64), events[i], det["WW"][i], cls["WW"][i]["type"], template) for i in win]
        feats = list(ex.map(_feat_job, jobs, chunksize=64))
        frows = []
        for i, fl in zip(win, feats):
            for t, f in zip(det["WW"][i], fl):
                if f is not None:
                    frows.append({"arm": "WW", "window": i, "sample": int(t), **f})
        if frows:
            df = df.merge(pd.DataFrame(frows), on=["arm", "window", "sample"], how="left")
    assert not [c for c in df.columns if c.endswith(("_x", "_y"))], "feature merge must not duplicate columns"
    for k in FT.ALL_FEATURES:
        if k not in df:
            df[k] = np.nan
    return df


# ----------------------------------------------------------------------------------------------- stage: train_infer
def stage_train_infer(ex, dev):
    """ARCH-TRAIN frozen outputs for the probe only: detector events, C0 and WW renders (inference, no training), the TRAIN
    QRS template, and the WW type-C / type-D feature table."""
    X, Y, Pid, ref, events, waves = render_role("train", ex, dev)
    template = FT.qrs_template(Y, ref)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "qrs_template_train.npy", template)
    write_json("qrs_template.json", {"source": "ARCH-TRAIN reference ECG at ARCH-TRAIN reference R peaks (RD1 neurokit cache)",
                                     "half_width_samples": FT.Q_HALF, "template": template})
    det = {"WW": list(ex.map(B._peaks, list(waves["WW"]), chunksize=256))}
    cls = {"WW": classify_role(ref, events, det["WW"])}
    del Y
    df = event_table("train", Pid, ref, events, det, cls, waves, X, template, ex, arms=("WW",))
    df.to_parquet(OUT / "event_level_train.parquet", compression="zstd")
    tc = {k: int((df["type"] == k).sum()) for k in TX.TYPES}
    write_json("train_population.json", {"windows": len(ref), "patients": int(np.unique(Pid).size),
                                         "windows_without_events": int(sum(len(e) == 0 for e in events)), "ww_type_counts": tc})
    print(f"[e0] train: {tc}", flush=True)


# ----------------------------------------------------------------------------------------------- stage: evaluate
def counterfactual_sets(ref, events, det_ww, cls_ww) -> dict:
    """Detection-level counterfactuals of WW-DET (diagnostics, not methods)."""
    n = len(ref)
    return {"WW": det_ww,
            "HARD_GUARD": [TX.hard_guard(det_ww[i], events[i]) for i in range(n)],
            "ORACLE1_noD": [TX.remove_types(det_ww[i], cls_ww[i]["type"], ("D",)) for i in range(n)],
            "ORACLE2_noBD": [TX.remove_types(det_ww[i], cls_ww[i]["type"], ("B", "D")) for i in range(n)],
            "WW_noC": [TX.remove_types(det_ww[i], cls_ww[i]["type"], ("C",)) for i in range(n)],
            "WW_AB_only": [TX.remove_types(det_ww[i], cls_ww[i]["type"], ("C", "D")) for i in range(n)]}


def micro(tp, fp, fn) -> dict:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else float("nan")}


def evaluate_role(role, ex, template, probes):
    import pandas as pd
    X, Y, Pid, ref, events, waves, det = load_cache(role)
    N = len(Y)
    evaluable = np.array([len(r) > 0 for r in ref])
    cls = {k: classify_role(ref, events, det[k]) for k in ARMS}
    cnt = {k: per_window_counts(cls[k], ref, events) for k in ARMS}
    ci = lambda v: C0.cluster_ci(v, Pid)  # noqa: E731
    # ---- consistency with the C0 / C0-A metric (A + C = TP, B + D = FP in every window)
    for k in ARMS:
        prf = PMX.rpeak_prf_at(waves[k], Y, FS, TOL_MS, peaks=(ref, det[k]))
        assert np.array_equal(cnt[k]["A"] + cnt[k]["C"], prf["n_tp"]) and np.array_equal(cnt[k]["B"] + cnt[k]["D"], prf["n_fp"])
    n_ref, n_placed = sum(len(r) for r in ref), sum(len(e) for e in events)
    tax = {"role": role, "windows": N, "patients": int(np.unique(Pid).size), "reference_beats": n_ref, "placed_events": n_placed,
           "windows_without_events": int(sum(len(e) == 0 for e in events)), "arms": {}}
    boot = {"per_window_rates": {}, "differences": {}, "counterfactuals": {}}
    for k in ARMS:
        tot = {t: int(cnt[k][t].sum()) for t in ("A", "B", "C", "D", "L", "M", "tp_placed", "fp_placed", "tp_rendered", "fp_rendered")}
        n_det = tot["A"] + tot["B"] + tot["C"] + tot["D"]
        tri = sum(1 for i in range(N) for j, ty in enumerate(cls[k][i]["type"]) if ty == "A"
                  and cls[k][i]["ref_placed"].get(int(cls[k][i]["ref_idx"][j]), -1) != int(cls[k][i]["placed_idx"][j]))
        tax["arms"][k] = {"raw": tot, "rendered_detections": n_det,
                          "pct_of_rendered": {t: tot[t] / n_det for t in TX.TYPES},
                          "pct_of_reference_beats": {t: tot[t] / n_ref for t in ("A", "C")},
                          "per_window_raw": {t: tot[t] / N for t in TX.TYPES},
                          "identity_TP": {"C_minus_L": tot["C"] - tot["L"], "TP_rendered_minus_TP_placed": tot["tp_rendered"] - tot["tp_placed"]},
                          "identity_FP": {"D_minus_M": tot["D"] - tot["M"], "FP_rendered_minus_FP_placed": tot["fp_rendered"] - tot["fp_placed"]},
                          "type_A_with_placed_not_the_ref_partner": tri}
        boot["per_window_rates"][k] = {t: ci(cnt[k][t]) for t in ("A", "B", "C", "D", "L", "M")}
    boot["differences"] = {"dA": ci(cnt["WW"]["A"] - cnt["C0"]["A"]), "dB": ci(cnt["WW"]["B"] - cnt["C0"]["B"]),
                           "dC": ci(cnt["WW"]["C"] - cnt["C0"]["C"]), "dD": ci(cnt["WW"]["D"] - cnt["C0"]["D"]),
                           "dFP": ci(cnt["WW"]["fp_rendered"] - cnt["C0"]["fp_rendered"]),
                           "dTP": ci(cnt["WW"]["tp_rendered"] - cnt["C0"]["tp_rendered"])}
    # ---- counterfactual event metrics (WW-DET; placed events and C0 as context)
    sets = counterfactual_sets(ref, events, det["WW"], cls["WW"])
    met = {"PLACED": event_metrics(Y, ref, events, evaluable), "C0": event_metrics(Y, ref, det["C0"], evaluable, waves["C0"])}
    for name, dl in sets.items():
        met[name] = event_metrics(Y, ref, dl, evaluable, waves["WW"])
    keys = ("precision", "recall", "f1", "fp", "fn", "rr_mae_ms", "hr_abs_err")
    cf = {name: {m: ci(met[name][m]) for m in keys} for name in met}
    for name in met:
        tp = sum(len(r) for r in ref) - int(np.nansum(met[name]["fn"]))
        cf[name]["micro"] = micro(tp, int(np.nansum(met[name]["fp"])), int(np.nansum(met[name]["fn"])))
    diffs = {}
    for name in ("HARD_GUARD", "ORACLE1_noD", "ORACLE2_noBD", "WW_noC", "WW_AB_only"):
        diffs[f"{name}-WW"] = {m: ci(met[name][m] - met["WW"][m]) for m in keys}
    for name in ("WW", "HARD_GUARD", "ORACLE1_noD", "ORACLE2_noBD"):
        diffs[f"{name}-PLACED"] = {m: ci(met[name][m] - met["PLACED"][m]) for m in ("precision", "recall", "f1", "fp", "fn")}
        diffs[f"{name}-C0"] = {m: ci(met[name][m] - met["C0"][m]) for m in ("precision", "recall", "f1", "fp", "fn")}
    boot["counterfactuals"] = diffs
    # ---- micro decomposition placed -> WW (A+B carried, + C, + D)
    t = tax["arms"]["WW"]["raw"]
    fnp = n_ref - t["tp_placed"]
    tax["arms"]["WW"]["micro_chain"] = {
        "placed": micro(t["tp_placed"], t["fp_placed"], fnp),
        "carried_A_B": micro(t["A"], t["B"], n_ref - t["A"]),
        "plus_C": micro(t["A"] + t["C"], t["B"], n_ref - t["A"] - t["C"]),
        "plus_C_D_(=WW)": micro(t["A"] + t["C"], t["B"] + t["D"], n_ref - t["A"] - t["C"])}
    # ---- event-level table, localization, features, probe scores
    df = event_table(role, Pid, ref, events, det, cls, waves, X, template, ex)
    cd = df[(df["arm"] == "WW") & df["type"].isin(["C", "D"])].copy()
    for g, pb in probes.items():
        cd[f"p_{g}"] = PR.predict(pb, cd[list(FT.GROUPS[g])].to_numpy(np.float64)) if len(cd) else []
    df = df.merge(cd[["arm", "window", "sample"] + [f"p_{g}" for g in probes]], on=["arm", "window", "sample"], how="left")
    loc = localization_summary(df)
    feats = feature_summary(cd)
    y = (cd["type"] == "C").to_numpy(int)
    subs = np.unique(cd["patient"].to_numpy())
    res = B.patient_resamples(subs.size, C0.BOOT_N, C0.BOOT_SEED)
    probe_eval = {g: PR.evaluate(pb, cd[list(FT.GROUPS[g])].to_numpy(np.float64), y, cd["patient"].to_numpy(), res) for g, pb in probes.items()}
    same = same_location(cd, X, Y, waves, events)
    pooled_share = t["C"] / (t["C"] + t["D"]) if t["C"] + t["D"] else float("nan")
    q = {"dD": boot["differences"]["dD"], "dC": boot["differences"]["dC"], "dFP": boot["differences"]["dFP"], "share_C": pooled_share,
         "hg_dF1": diffs["HARD_GUARD-WW"]["f1"], "hg_dFP": diffs["HARD_GUARD-WW"]["fp"], "hg_dRecall": diffs["HARD_GUARD-WW"]["recall"],
         "o1_dF1": diffs["ORACLE1_noD-WW"]["f1"], "o1_dFP": diffs["ORACLE1_noD-WW"]["fp"], "auroc_p3": probe_eval["P3"]["auroc"][0]}
    crit = PR.criteria(q, role)
    return {"taxonomy": tax, "bootstrap": boot, "counterfactuals": cf, "localization": loc, "features": feats, "probe": probe_eval,
            "same_location": same, "case_inputs": q, "criteria": crit, "case": PR.case_of(crit), "df": df}


def _bins(vals, edges):
    k = [TX.bin_index(v, edges) for v in vals]
    return [int(sum(1 for x in k if x == b)) for b in range(len(edges) - 1)]


def localization_summary(df) -> dict:
    out = {}
    for arm in ARMS:
        for ty in ("C", "D"):
            s = df[(df["arm"] == arm) & (df["type"] == ty)]
            n = len(s)
            bet = s[s["region"] == "between"]
            nearest = s["dist_nearest_ms"].to_numpy(np.float64)
            signed = s["signed_nearest_ms"].to_numpy(np.float64)
            out[f"{arm}_{ty}"] = {
                "n": n,
                "region": {r: int((s["region"] == r).sum()) for r in TX.REGIONS},
                "phase_bins": {"edges": TX.PHASE_EDGES, "counts": _bins(bet["phase"].to_numpy(np.float64), TX.PHASE_EDGES)},
                "abs_bins_nearest_placed_ms": {"edges": TX.ABS_EDGES_MS, "counts": _bins(nearest, TX.ABS_EDGES_MS),
                                               "after_nearest": _bins(nearest[signed > 0], TX.ABS_EDGES_MS),
                                               "before_nearest": _bins(nearest[signed < 0], TX.ABS_EDGES_MS)},
                "edge_distance_bins_ms": {"edges": TX.EDGE_EDGES_MS, "counts": _bins(s["dist_edge_ms"].to_numpy(np.float64), TX.EDGE_EDGES_MS)},
                "median_dist_nearest_ms": float(np.nanmedian(nearest)) if np.isfinite(nearest).any() else None,
                "median_phase": float(np.nanmedian(bet["phase"])) if len(bet) else None}
    return out


def feature_summary(cd) -> dict:
    out = {}
    c, d = cd[cd["type"] == "C"], cd[cd["type"] == "D"]
    subs = np.unique(cd["patient"].to_numpy())
    res = B.patient_resamples(subs.size, C0.BOOT_N, C0.BOOT_SEED)
    for f in FT.ALL_FEATURES:
        out[f] = {"C": PR.patient_macro(c[f], c["patient"]), "D": PR.patient_macro(d[f], d["patient"]),
                  "C_mean": float(np.nanmean(c[f])) if np.isfinite(c[f]).any() else None,
                  "D_mean": float(np.nanmean(d[f])) if np.isfinite(d[f]).any() else None,
                  "cohens_d_C_minus_D": PR.cohens_d_cluster(c[f], c["patient"], d[f], d["patient"], res, subs),
                  "missing_C": int((~np.isfinite(c[f].to_numpy(np.float64))).sum()), "missing_D": int((~np.isfinite(d[f].to_numpy(np.float64))).sum())}
    return out


def same_location(cd, X, Y, waves, events) -> dict:
    """Patient-macro mean +-250 ms segments (no warping) at WW type-C / type-D detections, for WW, C0, reference, PPG and
    the event raster; per-event segments are cached in outputs/."""
    out = {}
    for ty in ("C", "D"):
        s = cd[cd["type"] == ty]
        segs = {k: [] for k in ("WW", "C0", "REF", "PPG", "RASTER")}
        for i, t in zip(s["window"].to_numpy(int), s["sample"].to_numpy(int)):
            segs["WW"].append(FT.segment(waves["WW"][i], t)); segs["C0"].append(FT.segment(waves["C0"][i], t))
            segs["REF"].append(FT.segment(Y[i], t)); segs["PPG"].append(FT.segment(X[i].astype(np.float64), t))
            segs["RASTER"].append(FT.segment(AB.event_raster([events[i]])[0].astype(np.float64), t))
        pid = s["patient"].to_numpy()
        out[ty] = {"n": len(s), "lag_ms": (np.arange(-FT.H_LOCAL, FT.H_LOCAL + 1) * 1000.0 / FS).tolist()}
        for k, v in segs.items():
            a = np.array(v) if v else np.zeros((0, 2 * FT.H_LOCAL + 1))
            per = [np.nanmean(a[pid == p], axis=0) for p in np.unique(pid)] if len(a) else []
            out[ty][k] = np.nanmean(np.array(per), axis=0).tolist() if per else None
    return out


def stage_evaluate(ex, dev):
    import pandas as pd
    if not (ART / "prereg_manifest.json").exists():
        raise SystemExit("STOP: E0 outcomes are computed only after the committed preregistration manifest")
    template = np.load(OUT / "qrs_template_train.npy")
    tr = pd.read_parquet(OUT / "event_level_train.parquet")
    tr = tr[tr["type"].isin(["C", "D"])]
    ytr = (tr["type"] == "C").to_numpy(int)
    probes, cfg, cvres = {}, {}, {}
    for g, cols in FT.GROUPS.items():
        pb = PR.fit_probe(tr[list(cols)].to_numpy(np.float64), ytr, tr["patient"].to_numpy(), names=list(cols))
        probes[g] = pb
        cfg[g] = {"features": list(cols), "design_columns": pb.prep.names, "C": pb.c, "threshold": pb.threshold,
                  "coef": pb.model.coef_[0].tolist(), "intercept": float(pb.model.intercept_[0])}
        cvres[g] = pb.cv
    write_json("probe_config.json", {"model": "L2 logistic regression, class_weight=balanced, lbfgs", "C_grid": PR.C_GRID,
                                     "cv": f"GroupKFold({PR.N_FOLDS}) by patient inside ARCH-TRAIN; C by mean fold AUROC",
                                     "threshold": "max balanced accuracy on ARCH-TRAIN out-of-fold predictions at the chosen C",
                                     "positive_class": "type C (spontaneous recovery)", "train_n_C": int(ytr.sum()),
                                     "train_n_D": int((1 - ytr).sum()), "train_patients": int(tr["patient"].nunique()), "groups": cfg})
    write_json("probe_cv.json", cvres)
    out = {}
    for role in ROLES:
        r = evaluate_role(role, ex, template, probes)
        r["df"].to_parquet(ART / f"event_level_{role}.parquet", compression="zstd")
        write_json(f"event_taxonomy_{role}.json", r["taxonomy"])
        write_json(f"probe_{role}.json", r["probe"])
        out[role] = {k: v for k, v in r.items() if k != "df"}
        print(f"[e0] {role}: case {r['case']} criteria {r['criteria']}", flush=True)
    write_json("counterfactual_metrics.json", {role: out[role]["counterfactuals"] for role in ROLES})
    write_json("localization.json", {role: out[role]["localization"] for role in ROLES})
    write_json("feature_summary.json", {role: out[role]["features"] for role in ROLES})
    write_json("same_location.json", {role: out[role]["same_location"] for role in ROLES})
    write_json("bootstrap.json", {"unit": "patient", "replicates": C0.BOOT_N, "seed": C0.BOOT_SEED,
                                  **{role: out[role]["bootstrap"] for role in ROLES}})
    fc = PR.final_case(out["val"]["case"], out["holdout"]["case"])
    write_json("e0_case.json", {"val": {"inputs": out["val"]["case_inputs"], "criteria": out["val"]["criteria"], "case": out["val"]["case"]},
                                "holdout": {"inputs": out["holdout"]["case_inputs"], "criteria": out["holdout"]["criteria"],
                                            "case": out["holdout"]["case"]}, "final_case": fc})
    print(f"[e0] FINAL {fc}", flush=True)


# ----------------------------------------------------------------------------------------------- stage: figure / atlas
def stage_figure(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import precision_recall_curve, roc_curve
    tv = {r: json.loads((ART / f"event_taxonomy_{r}.json").read_text()) for r in ROLES}
    bt = json.loads((ART / "bootstrap.json").read_text())
    cf = json.loads((ART / "counterfactual_metrics.json").read_text())
    lc = json.loads((ART / "localization.json").read_text())
    sl = json.loads((ART / "same_location.json").read_text())
    case = json.loads((ART / "e0_case.json").read_text())
    fig, ax = plt.subplots(2, 4, figsize=(24, 10.5))
    a = ax[0, 0]
    a.axis("off")
    a.set_title("A  four-way taxonomy of rendered R detections", loc="left")
    cells = [["", "reference R\nyes", "reference R\nno"], ["placed R\nyes", "A\nsupported true", "B\ninherited false"],
             ["placed R\nno", "C\nspontaneous recovery", "D\nspontaneous\nhallucination"]]
    tb = a.table(cellText=cells, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 4.2)
    a = ax[0, 1]
    xs = np.arange(3)
    txt = []
    for k, (arm, role, col) in enumerate((("WW", "val", "#a23b52"), ("WW", "holdout", "#d4949f"), ("C0", "val", "#0e7c86"), ("C0", "holdout", "#86bfc4"))):
        v = [bt[role]["per_window_rates"][arm][t] for t in ("B", "C", "D")]
        a.bar(xs + (k - 1.5) * 0.2, [x[0] for x in v], 0.2, color=col, label=f"{'WW-DET' if arm == 'WW' else 'CoherentBeat-C0'} {role}")
        a.vlines(xs + (k - 1.5) * 0.2, [x[1] for x in v], [x[2] for x in v], color="k", lw=1)
        txt.append(f"A {arm} {role}: {bt[role]['per_window_rates'][arm]['A'][0]:.3f}")
    a.set_xticks(xs, ["B\ninherited false", "C\nspontaneous recovery", "D\nspontaneous hallucination"])
    a.set_ylabel("detections per window (patient mean, 95% CI)")
    a.set_title("B  type rates (type A per window in the box)", loc="left")
    a.text(0.02, 0.97, "\n".join(txt), transform=a.transAxes, va="top", fontsize=8, bbox={"fc": "w", "ec": "0.7"})
    a.legend(fontsize=8, loc="upper right")
    a = ax[0, 2]
    for k, role in enumerate(ROLES):
        ch = tv[role]["arms"]["WW"]["micro_chain"]
        steps = list(ch)
        a.plot([ch[s]["recall"] for s in steps], [ch[s]["precision"] for s in steps], "o-", label=f"WW-DET {role}")
        for s in steps:
            a.annotate(s.replace("_(=WW)", ""), (ch[s]["recall"], ch[s]["precision"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("pooled recall")
    a.set_ylabel("pooled precision")
    a.set_title("C  placed -> carried (A+B) -> +C -> +D", loc="left")
    a.legend(fontsize=8)
    a = ax[0, 3]
    mids = 0.5 * (np.array(TX.PHASE_EDGES[:-1]) + np.array(TX.PHASE_EDGES[1:]))
    wds = np.diff(TX.PHASE_EDGES)
    for ty, col, off in (("C", "#2a7f3f", -0.25), ("D", "#a23b52", 0.25)):
        c = np.array(lc["val"][f"WW_{ty}"]["phase_bins"]["counts"], float)
        a.bar(mids + off * wds / 2, c / max(c.sum(), 1) / wds, wds / 2, color=col, label=f"WW type {ty} (between placed R, ARCH-VAL)")
    a.set_xlabel("normalized phase between placed R")
    a.set_ylabel("density")
    a.set_title("D  inter-event phase", loc="left")
    a.legend(fontsize=8)
    for col_i, ty in ((0, "C"), (1, "D")):
        a = ax[1, col_i]
        s = sl["val"][ty]
        if s["n"]:
            for k, c, lab in (("REF", "k", "reference ECG"), ("WW", "#a23b52", "WW-DET"), ("C0", "#0e7c86", "CoherentBeat-C0")):
                a.plot(s["lag_ms"], s[k], color=c, label=lab)
            a2 = a.twinx()
            a2.plot(s["lag_ms"], s["RASTER"], color="0.6", ls=":", label="event raster")
            a2.set_ylim(0, 1.05)
            a2.set_yticks([])
        a.axvline(0, color="0.5", lw=0.8)
        a.set_xlabel("ms from WW detection")
        a.set_title(f"E{col_i + 1}  mean +-250 ms at WW type {ty} (n={s['n']}, ARCH-VAL)", loc="left")
        a.legend(fontsize=8)
    a = ax[1, 2]
    df = {r: pd.read_parquet(ART / f"event_level_{r}.parquet") for r in ROLES}
    for role, ls in (("val", "-"), ("holdout", "--")):
        cd = df[role][(df[role]["arm"] == "WW") & df[role]["type"].isin(["C", "D"])]
        y = (cd["type"] == "C").to_numpy(int)
        for g, col in (("P1", "#7a6fd0"), ("P2", "#d59a54"), ("P3", "#0e7c86")):
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y, cd[f"p_{g}"])
                a.plot(fpr, tpr, ls, color=col, label=f"{g} {role}")
    a.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    a.set_xlabel("false positive rate (type D called C)")
    a.set_ylabel("true positive rate (type C)")
    a.set_title("F  probe ROC (fitted on ARCH-TRAIN only)", loc="left")
    a.legend(fontsize=7, ncol=2)
    a = ax[1, 3]
    names = ("PLACED", "C0", "WW", "HARD_GUARD", "ORACLE1_noD", "ORACLE2_noBD")
    for role, mk in (("val", "o"), ("holdout", "s")):
        a.scatter([cf[role][n]["recall"][0] for n in names], [cf[role][n]["precision"][0] for n in names], marker=mk, label=role)
        for n in names:
            a.annotate(n, (cf[role][n]["recall"][0], cf[role][n]["precision"][0]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.set_xlabel("recall (patient mean)")
    a.set_ylabel("precision (patient mean)")
    a.set_title("G  detection-level counterfactuals (diagnostic only)", loc="left")
    a.legend(fontsize=8)
    fig.suptitle(f"E0 WW-DET spontaneous event audit (post-hoc audit of frozen outputs; ARCH-HOLDOUT previously opened) — "
                 f"case VAL {case['val']['case']}, HOLDOUT {case['holdout']['case']}, final {case['final_case']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure.png", dpi=110)


def stage_atlas(ex, dev):
    """Fixed-seed (salted-rank) examples from ARCH-VAL: 12 WW type C, 12 type D, 12 type B, 12 WW / C0 detection-status
    differences. No hand-picking."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    X, Y, Pid, ref, events, waves, det = load_cache("val")
    df = pd.read_parquet(ART / "event_level_val.parquet")
    ww = df[df["arm"] == "WW"]
    groups = {}
    for ty, lab in (("C", "WW type C spontaneous recovery"), ("D", "WW type D spontaneous hallucination"), ("B", "WW type B inherited false event")):
        s = ww[ww["type"] == ty]
        keys = [f"{w}:{t}" for w, t in zip(s["window"], s["sample"])]
        order = B.salted_rank(f"e0-atlas-v1-{ty}", keys)[:12]
        groups[lab] = [(int(s["window"].iloc[k]), int(s["sample"].iloc[k]), ty) for k in order]
    diff = []
    for i in range(len(ref)):
        m = {j for _, j in TX.RP.match_rpeaks(np.asarray(det["C0"][i]), np.asarray(det["WW"][i]), FS, TOL_MS)[0]}
        mc = {a for a, _ in TX.RP.match_rpeaks(np.asarray(det["C0"][i]), np.asarray(det["WW"][i]), FS, TOL_MS)[0]}
        diff += [(i, int(t), "WW only") for j, t in enumerate(det["WW"][i]) if j not in m]
        diff += [(i, int(t), "C0 only") for a, t in enumerate(det["C0"][i]) if a not in mc]
    order = B.salted_rank("e0-atlas-v1-diff", [f"{w}:{t}:{k}" for w, t, k in diff])[:12]
    groups["WW and C0 differ in detection status"] = [diff[k] for k in order]
    fig, ax = plt.subplots(16, 3, figsize=(18, 40))
    tt = np.arange(T) / FS
    for g, (lab, items) in enumerate(groups.items()):
        for k, (i, t, tag) in enumerate(items):
            a = ax[g * 4 + k // 3, k % 3]
            ppg = X[i] / (np.ptp(X[i]) + 1e-9) * np.ptp(Y[i]) * 0.6
            a.plot(tt, Y[i], "k", lw=0.9, label="reference ECG")
            a.plot(tt, waves["WW"][i] - 1.2, color="#a23b52", lw=0.9, label="WW-DET (-1.2)")
            a.plot(tt, waves["C0"][i] - 2.4, color="#0e7c86", lw=0.9, label="CoherentBeat-C0 (-2.4)")
            a.plot(tt, ppg - 3.6 - np.median(ppg), color="#2a7f3f", lw=0.8, label="PPG (scaled, -3.6)")
            for p in events[i]:
                a.axvline(p / FS, color="0.6", lw=0.8, ls="--")
            a.plot(np.asarray(ref[i]) / FS, Y[i][np.asarray(ref[i], int)], "kv", ms=4)
            a.plot(np.asarray(det["WW"][i]) / FS, waves["WW"][i][np.asarray(det["WW"][i], int)] - 1.2, "o", color="#a23b52", ms=3)
            a.plot(np.asarray(det["C0"][i]) / FS, waves["C0"][i][np.asarray(det["C0"][i], int)] - 2.4, "x", color="#0e7c86", ms=4)
            a.axvspan((t - 6) / FS, (t + 6) / FS, color="#f2c14e", alpha=0.35)
            a.set_title(f"{lab} | window {i}, t={t / FS:.2f} s {'' if tag in TX.TYPES else '(' + tag + ')'}", fontsize=8)
            a.set_yticks([])
            if g == 0 and k == 0:
                a.legend(fontsize=6, loc="upper right")
    fig.suptitle("E0 atlas (ARCH-VAL, salted-rank sampling, not hand-picked). Grey dashed = placed events; black v = reference R; "
                 "red o = WW R; blue x = C0 R; yellow band = the sampled detection +-50 ms", fontsize=11, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(ART / "atlas.png", dpi=75)
    write_json("atlas_index.json", {lab: items for lab, items in groups.items()})


STAGES = {"reproduce": stage_reproduce, "manifest": stage_manifest, "train_infer": stage_train_infer, "evaluate": stage_evaluate,
          "figure": stage_figure, "atlas": stage_atlas}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
