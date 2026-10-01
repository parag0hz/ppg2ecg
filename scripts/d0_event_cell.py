"""D0 — event-cell support diagnostic on frozen BF0 beat outputs (docs/D0_EVENT_CELL_SUPPORT_PREREGISTRATION.md).

Post-hoc with respect to BF0 (verdict frozen: Case A, G1 FAIL). Nothing is trained; BF0 checkpoints, positions and seeds
are reused; only the rendering support changes.

Stages, in order:
  reproduce   regenerate the frozen BF0 beat outputs (validation) and require the ORIGINAL renderer to reproduce every
              saved BF0 window bit for bit; input hashes
  crossfade   TRAIN only: choose the crossfade width from continuity statistics (no R detector, no F1, no FD)
  manifest    sha256 of the D0 preregistration, code and frozen design files (before any event-cell validation result)
  evaluate    event-cell rerenders, D1-D4, attribution, boundary audit, misses, waveform side effects, verdict
  figure      the D0 diagnostic figure
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/d0_event_cell.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
from ppg2ecg.beatfirst import eventcell as EC  # noqa: E402
from ppg2ecg.beatfirst import model as BM  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.beatfirst import timing as BT  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.evaluation.m1_structural import spectral_metrics  # noqa: E402

PREREG = "docs/D0_EVENT_CELL_SUPPORT_PREREGISTRATION.md"
CODE_FILES = ("scripts/d0_event_cell.py", "src/ppg2ecg/beatfirst/eventcell.py", "tests/test_d0_event_cell.py")
ART = ROOT / "artifacts/d0_event_cell_support"
OUT = ROOT / "outputs/d0_event_cell_support"
BF0_ART = ROOT / "artifacts/bf0_beat_first"
FS, T = B.FS, B.T
CANDIDATE_WIDTHS = (4, 8, 12, 16)
SENS_WIDTH = 16                                      # pre-specified sensitivity renderer: descriptive only, never in the verdict
SALT_XFADE, N_XFADE = "d0-crossfade-v1", 3000
SALT_EXAMPLE = "d0-example-v1"
TOL_SAMPLES = 50.0 / 1000.0 * FS                     # ±50 ms, the BF0 matching tolerance
SAFETY = {"flat_fill_increase_max": 0.25, "boundary_new_fp_frac_strong_max": 0.25,
          "boundary_new_fp_frac_not_supported_min": 0.5}
G1_MARGIN = B.MARGIN["g1_f1"]
ARMS = ("S", "D", "T")                               # stochastic seed 0, deterministic, template (control)


# ----------------------------------------------------------------------------------------------- pure helpers
def select_width(stats: dict, real: dict, candidates=CANDIDATE_WIDTHS, arms=("S", "D")) -> tuple[int, bool]:
    """Smallest candidate whose P95 amplitude jump AND P95 first-difference change at crossfades are <= the P95 of the
    same statistics of real TRAIN ECG at midpoints between consecutive reference R, for every arm. (16, False) if none."""
    for w in candidates:
        if all(stats[a][w]["A_p95"] <= real[w]["A_p95"] and stats[a][w]["B_p95"] <= real[w]["B_p95"] for a in arms):
            return int(w), True
    return int(max(candidates)), False


def boundary_stats(y, centre: float, width: float) -> tuple[float, float]:
    """Max |first difference| (A) and max |second difference| (B) over the crossfade plus one sample on each side."""
    lo = max(int(np.floor(centre - width / 2.0)) - 1, 0)
    hi = min(int(np.ceil(centre + width / 2.0)) + 1, len(y) - 1)
    seg = np.asarray(y[lo:hi + 1], dtype=np.float64)
    if seg.size < 3:
        return float("nan"), float("nan")
    return float(np.max(np.abs(np.diff(seg)))), float(np.max(np.abs(np.diff(seg, 2))))


def fp_detections(ref, det) -> list[int]:
    """Detected R peaks not matched one-to-one to a reference R within ±50 ms (`rpeaks.match_rpeaks`, as in BF0 G1)."""
    det = np.asarray(det, int)
    matched = {j for _, j in RP.match_rpeaks(np.asarray(ref, int), det, FS, 50.0)[0]}
    return [int(det[j]) for j in range(det.size) if j not in matched]


def d0_verdict(d: dict) -> str:
    """Frozen D0 verdict on the support-leakage hypothesis (prereg §8)."""
    if (not d["D1"]) or d["flat_fill_increase"] > SAFETY["flat_fill_increase_max"] \
            or d["boundary_new_fp_frac"] >= SAFETY["boundary_new_fp_frac_not_supported_min"]:
        return "NOT SUPPORTED"
    if d["D2"] and d["D3"] and d["D4"] and d["boundary_new_fp_frac"] < SAFETY["boundary_new_fp_frac_strong_max"] \
            and d["fd_increase"] <= d["fd_increase_max"]:
        return "STRONGLY SUPPORTED"
    return "PARTIALLY SUPPORTED"


def gap_recovery(f1_cell, f1_orig, f1_placed, pid, n_rep=B.BOOT_N):
    """(F1_cell - F1_orig) / (F1_placed - F1_orig) on equal-patient-weight means; same patient resamples as BF0."""
    pid = np.asarray(pid)
    subs = np.unique(pid)
    per = lambda a: np.array([np.nanmean(np.asarray(a, float)[pid == s]) for s in subs])  # noqa: E731
    pc, po, pp = per(f1_cell), per(f1_orig), per(f1_placed)
    ratio = lambda r: float(np.nanmean(pc[r] - po[r]) / np.nanmean(pp[r] - po[r]))  # noqa: E731
    point = ratio(np.arange(len(subs)))
    draws = np.array([ratio(r) for r in B.patient_resamples(len(subs), n_rep)])
    dens = np.array([np.nanmean(pp[r] - po[r]) for r in B.patient_resamples(len(subs), n_rep)])
    stable = bool(np.all(dens > 0))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if stable else None
    return {"point": point, "ci95": ci, "denominator_positive_in_every_replicate": stable}


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))))


# ----------------------------------------------------------------------------------------------- io / frozen inputs
def write_json(name: str, obj) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(B.clean(obj), indent=1))


def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def frozen_bf0():
    R = dict(np.load(B.RUN / "render_val.npz"))
    pos = [np.asarray(p, int) for p in B.split_list(R["pos_idx"], R["pos_off"])]
    tz = np.load(B.RUN / "template.npz")
    template, rr_median = tz["template"], float(tz["rr_median"])
    empty_fill = float(np.median(np.concatenate([template[:8], template[-8:]])))   # BF0 render stage rule
    return R, pos, template, rr_median, empty_fill


def load_nets(dev):
    nets = {}
    for kind, f in (("deterministic", "beat_deterministic.pt"), ("stochastic", "beat_stochastic.pt")):
        nets[kind] = BM.BeatFlowNet().to(dev).eval()
        nets[kind].load_state_dict(torch.load(B.RUN / f, map_location="cpu")["state_dict"])
    return nets


def regenerate(X, pos, rr_median, nets, dev):
    """BF0 render-stage conditions and beat calls, unchanged: deterministic, and stochastic realization s = 0 with the
    frozen per-beat seeds 1_000_003 n + j (window n, beat j)."""
    ppg, rr, own = B.conditions(X, pos, rr_median)
    return {"D": B.predict_deterministic(nets["deterministic"], ppg, rr, dev),
            "S": B.sample_stochastic(nets["stochastic"], ppg, rr, own, 0, dev)}


def split_beats(beats, pos):
    out, c = [], 0
    for p in pos:
        out.append(beats[c:c + len(p)])
        c += len(p)
    assert c == len(beats)
    return out


def render_cell(beats, pos, width, rr_single, empty_fill):
    ys, infos = [], []
    for b, p in zip(split_beats(beats, pos), pos):
        y, info = EC.render(b, p, T, width, rr_single, empty_fill)
        ys.append(y)
        infos.append(info)
    return np.asarray(ys, np.float32), infos


def orig_flat_fraction(beats, pos, empty_fill):
    out = []
    for b, p in zip(split_beats(beats, pos), pos):
        out.append(1.0 if len(p) == 0 else BR.assemble(b, p, T, empty_fill)[1]["n_uncovered"] / T)
    return np.asarray(out)


# ----------------------------------------------------------------------------------------------- stages
def stage_reproduce(ex, dev):
    Xv, _, _ = B.load_role("val")
    R, pos, template, rr_median, empty_fill = frozen_bf0()
    beats = regenerate(Xv, pos, rr_median, load_nets(dev), dev)
    n_beats = int(sum(len(p) for p in pos))
    beats["T"] = np.repeat(np.asarray(template, np.float64)[None], n_beats, axis=0)
    saved = {"S": R["A3"][0], "D": R["A2"], "T": R["A1"]}
    rep = {}
    for arm in ARMS:
        orig = B.assemble_all(beats[arm], pos, empty_fill)
        d = np.abs(orig.astype(np.float64) - saved[arm].astype(np.float64)).max(axis=1)
        rep[arm] = {"max_abs_diff": float(d.max()), "windows_bit_identical": int(np.sum(d == 0)), "windows": int(len(d)),
                    "saved_render_sha256": sha_array(saved[arm]), "regenerated_beats_sha256": sha_array(beats[arm])}
    ok = all(rep[a]["max_abs_diff"] == 0.0 for a in ARMS)
    write_json("original_reproduction.json", {"arms": {"S": "stochastic seed 0", "D": "deterministic", "T": "template"},
                                              "renderer": "ppg2ecg.beatfirst.render.assemble (BF0, unchanged)",
                                              "all_exact": ok} | rep)
    if not ok:
        raise SystemExit(f"STOP: original BF0 render not reproduced exactly: {rep}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "val_beats.npz", S=beats["S"], D=beats["D"])
    ck = {k: json.loads((BF0_ART / f"checkpoint_{k}.json").read_text())["sha256"] for k in ("stochastic", "deterministic")}
    write_json("input_hashes.json", {
        "bf0_prereg_commit": "504395dd967330b8ed0d80e5dad98302437632f7",
        "bf0_result_commit": "d380e3b53ebcbaf6627300345f08a92a6e794988",
        "beat_stochastic.pt": B.sha256_file(B.RUN / "beat_stochastic.pt"), "beat_deterministic.pt": B.sha256_file(B.RUN / "beat_deterministic.pt"),
        "bf0_recorded_checkpoint_sha256": ck,
        "checkpoints_match_bf0_record": B.sha256_file(B.RUN / "beat_stochastic.pt") == ck["stochastic"]
        and B.sha256_file(B.RUN / "beat_deterministic.pt") == ck["deterministic"],
        "render_val.npz": B.sha256_file(B.RUN / "render_val.npz"), "template.npz": B.sha256_file(B.RUN / "template.npz"),
        "val_rpeaks.npz": B.sha256_file(B.RUN / "val_rpeaks.npz"), "rd1_checkpoint": B.sha256_file(B.RD1_CKPT),
        "rd1_train_peaks_cache": B.sha256_file(B.RD1_TRAIN_PEAKS), "split_manifest": B.sha256_file(B.SPLIT_MANIFEST),
        "bf0_gate_metrics.json": B.sha256_file(BF0_ART / "gate_metrics.json"), "software": B.software()})
    print(json.dumps(B.clean(rep), indent=1))


def stage_crossfade(ex, dev):
    """TRAIN only. No R detector on renders, no F1, no FD, no validation data."""
    Xtr, Ytr, _ = B.load_role("train")
    ref_all = B.train_reference_peaks(Ytr, ex)
    pick = np.sort(B.salted_rank(SALT_XFADE, range(len(Xtr)))[:N_XFADE])
    X, Y = Xtr[pick], Ytr[pick]
    ref = [np.asarray(ref_all[i], int) for i in pick]
    det = B.load_detector(dev)
    field = B.detector_fields(det, X, dev)
    events = list(ex.map(B._events, list(field), chunksize=256))
    head = BT.TimingHead().to(dev).eval()
    head.load_state_dict(torch.load(B.RUN / "timing_head.pt", map_location="cpu")["state_dict"])
    pos = []
    for n in range(len(X)):
        ev = np.asarray(events[n], int)
        if ev.size == 0:
            pos.append(np.zeros(0, int))
            continue
        with torch.no_grad():
            mu, _ = head(torch.from_numpy(BT.event_features(field[n], X[n], ev)).to(dev))
        pos.append(B._positions(ev, mu.float().cpu().numpy())[0])
    _, _, template, rr_median, empty_fill = frozen_bf0()
    rr_single = rr_median * FS
    beats = regenerate(X, pos, rr_median, load_nets(dev), dev)
    stats = {a: {} for a in ("S", "D")}
    orig_ctx = {a: {} for a in ("S", "D")}
    real = {}
    for w in CANDIDATE_WIDTHS:
        for a in ("S", "D"):
            A, Bs, Ao, Bo = [], [], [], []
            for b, p in zip(split_beats(beats[a], pos), pos):
                if len(p) < 2:
                    continue
                y, info = EC.render(b, p, T, w, rr_single, empty_fill)
                yo = BR.assemble(b, p, T, empty_fill)[0]
                for c, ww in zip(info["centres"], info["widths"]):
                    sa, sb = boundary_stats(y, c, ww)
                    oa, ob = boundary_stats(yo, c, ww)
                    A.append(sa); Bs.append(sb); Ao.append(oa); Bo.append(ob)
            stats[a][w] = {"n_boundaries": len(A), "A_p95": float(np.nanpercentile(A, 95)), "B_p95": float(np.nanpercentile(Bs, 95)),
                           "A_median": float(np.nanmedian(A)), "B_median": float(np.nanmedian(Bs))}
            orig_ctx[a][w] = {"A_p95": float(np.nanpercentile(Ao, 95)), "B_p95": float(np.nanpercentile(Bo, 95))}
        RA, RB = [], []
        for y, r in zip(Y, ref):
            r = np.sort(r)
            for k in range(len(r) - 1):
                sa, sb = boundary_stats(y, 0.5 * (r[k] + r[k + 1]), w)
                RA.append(sa); RB.append(sb)
        real[w] = {"n_midpoints": len(RA), "A_p95": float(np.nanpercentile(RA, 95)), "B_p95": float(np.nanpercentile(RB, 95))}
    width, met = select_width(stats, real)
    write_json("crossfade_train_selection.json", {
        "split": "train only", "windows": int(len(X)), "salt": SALT_XFADE, "candidates_samples": list(CANDIDATE_WIDTHS),
        "statistics": {"A": "max |first difference| over the crossfade +- 1 sample",
                       "B": "max |second difference| (first-difference change) over the same samples"},
        "criterion": "smallest width with P95(A) and P95(B) at event-cell crossfades <= P95 of the same statistics of real "
                     "TRAIN ECG at midpoints between consecutive reference R (same width), for both D0 arms (S, D)",
        "rendered_positions": "BF0 event stage on TRAIN PPG (RD1 events + timing head); no R detection on any render",
        "event_cell": stats, "real_ecg": real, "original_renderer_at_same_locations_context_only": orig_ctx,
        "selected_width_samples": width, "selected_width_ms": width / FS * 1000.0, "criterion_met": met})
    write_json("renderer_config.json", {
        "crossfade": "complementary raised cosine", "crossfade_width_samples": width, "crossfade_width_ms": width / FS * 1000.0,
        "crossfade_cap": "min(width, r_{i+1} - r_i - 2) so no crossfade reaches an R sample",
        "boundary_rule": "interior: midpoint of adjacent placed events, moved the least amount that keeps the crossfade inside "
                         "both beats' output supports (R-64 .. R+101); supports overlapping by less than the width: crossfade "
                         "over the overlap; no overlap: gap, BF0 nearest-covered-value fill",
        "first_last_rule": "L_1 = r_1 - (r_2 - r_1)/2, U_N = r_N + (r_N - r_{N-1})/2, clipped to the beat's output support "
                           "and the window; no crossfade into the fill",
        "singleton_rule": "r_1 +- RR_train_median/2", "rr_single_samples": rr_single, "rr_train_median_s": rr_median,
        "rr_source": "outputs/bf0_beat_first/template.npz (BF0 train_beat stage, TRAIN reference peaks)",
        "outside_cells": "BF0 nearest-covered-value fill (ties to the left)", "empty_window_fill": empty_fill,
        "renormalization": "none (weights sum to 1 by construction)", "time_warp": "none",
        "beat_support_samples": {"before_R": 64, "after_R": 101, "length": 166}})
    print(json.dumps(B.clean({"selected_width": width, "criterion_met": met, "stats": stats, "real": real}), indent=1))


def stage_manifest(ex, dev):
    files = (PREREG,) + CODE_FILES
    write_json("prereg_manifest.json", {
        "prereg": PREREG, "sha256": {f: B.sha256_file(ROOT / f) for f in files},
        "frozen_design_files": {f: B.sha256_file(ART / f) for f in ("renderer_config.json", "crossfade_train_selection.json",
                                                                     "original_reproduction.json", "input_hashes.json")},
        "parent_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "written_before_any_event_cell_validation_result": True})


def _attrib(fps, placed, edge_mask):
    rows = []
    for d in fps:
        z, ft, dm = EC.fp_zone(d, placed)
        near = int(np.argmin(np.abs(np.asarray(placed) - d))) if len(placed) else -1
        rows.append({"d": d, "zone": z, "far": ft, "dms": dm, "edge": bool(edge_mask[near]) if near >= 0 else None})
    return rows


def stage_evaluate(ex, dev):
    cfg = json.loads((ART / "renderer_config.json").read_text())
    width, rr_single = int(cfg["crossfade_width_samples"]), float(cfg["rr_single_samples"])
    Xv, Yv, Pv = B.load_role("val")
    N = len(Yv)
    R, pos, template, rr_median, empty_fill = frozen_bf0()
    ref = B.val_reference_peaks(Yv, ex)
    evaluable = np.array([len(r) > 0 for r in ref])
    # ---- regenerate and re-verify the frozen beats (hard stop if not bit-exact)
    beats = regenerate(Xv, pos, rr_median, load_nets(dev), dev)
    beats["T"] = np.repeat(np.asarray(template, np.float64)[None], int(sum(len(p) for p in pos)), axis=0)
    orig = {"S": np.asarray(R["A3"][0], np.float64), "D": np.asarray(R["A2"], np.float64), "T": np.asarray(R["A1"], np.float64)}
    for a in ARMS:
        if max_abs_diff(B.assemble_all(beats[a], pos, empty_fill), orig[a]) != 0.0:
            raise SystemExit(f"STOP: original render of arm {a} not reproduced exactly")
    cell, info = {}, {}
    for a in ARMS:
        cell[a], info[a] = render_cell(beats[a], pos, width, rr_single, empty_fill)
        cell[a] = cell[a].astype(np.float64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cell_renders_val.npz", **{f"{a}_CELL": cell[a].astype(np.float32) for a in ARMS})
    waves = {f"{a}-ORIG": orig[a] for a in ARMS} | {f"{a}-CELL": cell[a] for a in ARMS}
    det = {k: list(ex.map(B._peaks, list(w), chunksize=256)) for k, w in waves.items()}
    edge = [EC.edge_events(p, T, rr_single) for p in pos]

    # ---- per-window event metrics (BF0 functions)
    per = {}
    for k, w in waves.items():
        prf = PMX.rpeak_prf_at(w, Yv, FS, 50.0, peaks=(ref, det[k]))
        bl = PMX.beat_level_metrics(w, Yv, FS, 50.0, peaks=(ref, det[k]))
        per[k] = {"f1": np.where(evaluable, prf["rpeak_f1"], np.nan), "fp": prf["n_fp"], "fn": prf["n_fn"], "tp": prf["n_tp"],
                  "rr_mae_ms": bl["rr_mae_ms"], "hr_abs_err": bl["hr_abs_err"]}
    pos_f1 = np.where(evaluable, PMX.rpeak_prf_at(Yv, Yv, FS, 50.0, peaks=(ref, pos))["rpeak_f1"], np.nan)
    raw = {k: {m: int(np.nansum(per[k][m])) for m in ("tp", "fp", "fn")} for k in waves}
    rates = {k: {"fp_rate": B.cluster_ci(per[k]["fp"], Pv), "fn_rate": B.cluster_ci(per[k]["fn"], Pv),
                 "f1": B.cluster_ci(per[k]["f1"], Pv), "rr_mae_ms": B.cluster_ci(per[k]["rr_mae_ms"], Pv),
                 "hr_abs_err": B.cluster_ci(per[k]["hr_abs_err"], Pv)} for k in waves}

    # ---- FP attribution (zones relative to placed events; FP = unmatched to the reference R)
    fps = {k: [fp_detections(ref[i], det[k][i]) for i in range(N)] for k in waves}
    attrib = {k: [_attrib(fps[k][i], pos[i], edge[i]) for i in range(N)] for k in waves}
    zone_counts, far_counts, interior_far = {}, {}, {}
    for k in waves:
        zc = {z: 0 for z in EC.ZONES}
        fc = {f: 0 for f in EC.FAR_TYPES}
        inter = np.zeros(N)
        for i in range(N):
            for r in attrib[k][i]:
                zc[r["zone"]] += 1
                if r["zone"] == "E_far" and r["far"] in fc:
                    fc[r["far"]] += 1
                    inter[i] += r["far"] == "interior"
        zone_counts[k], far_counts[k], interior_far[k] = zc, fc, inter
    d1 = B.cluster_ci(per["S-CELL"]["fp"] - per["S-ORIG"]["fp"], Pv)
    d4 = B.cluster_ci(per["D-CELL"]["fp"] - per["D-ORIG"]["fp"], Pv)
    d2 = B.cluster_ci(interior_far["S-CELL"] - interior_far["S-ORIG"], Pv)
    if_rate = {k: B.cluster_ci(interior_far[k], Pv) for k in ("S-ORIG", "S-CELL", "D-ORIG", "D-CELL")}
    d2_rel = 1.0 - if_rate["S-CELL"][0] / if_rate["S-ORIG"][0] if if_rate["S-ORIG"][0] else float("nan")
    dist = {}
    for k in waves:
        dm = np.array([r["dms"] for i in range(N) for r in attrib[k][i] if np.isfinite(r["dms"])])
        lag = []
        for i in range(N):
            P = pos[i]
            if len(P) < 2:
                continue
            med = float(np.median(np.diff(P)))
            for r in attrib[k][i]:
                if r["zone"] == "E_far" and (P < r["d"]).any():
                    lag.append((r["d"] - P[P < r["d"]][-1]) / med)
        dist[k] = {"dist_to_nearest_placed_ms_q10_25_50_75_90": np.percentile(np.abs(dm), [10, 25, 50, 75, 90]).tolist() if dm.size else None,
                   "far_lag_from_previous_over_local_rr_q10_25_50_75_90": np.percentile(lag, [10, 25, 50, 75, 90]).tolist() if lag else None}
    reduction = {}
    for a in ("S", "D"):
        o, c = f"{a}-ORIG", f"{a}-CELL"
        parts = {"edge_far_before_first": far_counts[o]["before_first"] - far_counts[c]["before_first"],
                 "edge_far_after_last": far_counts[o]["after_last"] - far_counts[c]["after_last"],
                 "interior_far": far_counts[o]["interior"] - far_counts[c]["interior"],
                 "gap_far": far_counts[o]["gap_missed_event"] - far_counts[c]["gap_missed_event"],
                 "P_PR_zone": zone_counts[o]["D_before_300_50ms"] - zone_counts[c]["D_before_300_50ms"],
                 "ST_zone": zone_counts[o]["B_after_50_150ms"] - zone_counts[c]["B_after_50_150ms"],
                 "T_zone": zone_counts[o]["C_after_150_450ms"] - zone_counts[c]["C_after_150_450ms"],
                 "within_50ms": zone_counts[o]["A_within_50ms"] - zone_counts[c]["A_within_50ms"]}
        tot = raw[o]["fp"] - raw[c]["fp"]
        reduction[a] = {"total_fp_reduction": tot, "by_source": parts,
                        "share_of_total": {k2: (v / tot if tot else None) for k2, v in parts.items()}}

    # ---- boundary-artifact audit: CELL-only FPs, distance to crossfade centres / owned-region limits
    boundary = {}
    for a in ("S", "D"):
        o, c = f"{a}-ORIG", f"{a}-CELL"
        new, near_x, near_l, removed = 0, 0, 0, 0
        for i in range(N):
            fo, fc_ = np.asarray(fps[o][i]), np.asarray(fps[c][i])
            for d in fc_:
                if fo.size == 0 or np.min(np.abs(fo - d)) > TOL_SAMPLES:
                    new += 1
                    kind = EC.boundary_near(int(d), info[a][i]["centres"], (info[a][i]["left_limit"], info[a][i]["right_limit"]), width)
                    near_x += kind == "crossfade"
                    near_l += kind == "edge_limit"
            for d in fo:
                if fc_.size == 0 or np.min(np.abs(fc_ - d)) > TOL_SAMPLES:
                    removed += 1
        boundary[a] = {"new_cell_only_fp": new, "boundary_near_crossfade": near_x, "boundary_near_edge_limit": near_l,
                       "boundary_near_total": near_x + near_l, "orig_fp_removed": removed,
                       "boundary_near_over_removed": (near_x + near_l) / removed if removed else (float("inf") if near_x + near_l else 0.0)}

    # ---- D3: BF0 G1 recomputed (S-ORIG must reproduce BF0 exactly)
    g_orig = B.cluster_ci(per["S-ORIG"]["f1"] - pos_f1, Pv)
    bf0_g1 = json.loads((BF0_ART / "gate_metrics.json").read_text())["detail"]["G1"]["f1_diff"]
    if max(abs(x - y) for x, y in zip(g_orig, bf0_g1)) > 1e-12:
        raise SystemExit(f"STOP: S-ORIG G1 {g_orig} does not reproduce BF0 {bf0_g1}")
    g_cell = B.cluster_ci(per["S-CELL"]["f1"] - pos_f1, Pv)
    g_dcell = B.cluster_ci(per["D-CELL"]["f1"] - pos_f1, Pv)
    g_dorig = B.cluster_ci(per["D-ORIG"]["f1"] - pos_f1, Pv)
    gap = gap_recovery(per["S-CELL"]["f1"], per["S-ORIG"]["f1"], pos_f1, Pv)

    # ---- misses
    miss = {}
    for k in waves:
        red, tot, by_cat, by_dist, t_tall = 0, 0, {"edge": [0, 0], "interior": [0, 0]}, {"<0.25s": 0, "0.25-0.5s": 0, ">=0.5s": 0}, 0
        w = waves[k]
        for i in range(N):
            P = pos[i]
            if len(P) == 0:
                continue
            m = RP.match_rpeaks(P, np.asarray(det[k][i], int), FS, 50.0)[0]
            hit = {a for a, _ in m}
            for j, p in enumerate(P):
                cat = "edge" if edge[i][j] else "interior"
                by_cat[cat][1] += 1
                tot += 1
                if j in hit:
                    red += 1
                    by_cat[cat][0] += 1
                    continue
                de = min(p, T - 1 - p)
                by_dist["<0.25s" if de < 32 else "0.25-0.5s" if de < 64 else ">=0.5s"] += 1
                tz = w[i][min(T - 1, p + 20):min(T, p + 58)]
                t_tall += bool(tz.size and tz.max() >= w[i][p])
        miss[k] = {"placed": tot, "redetected": red, "redetection_rate": red / max(tot, 1), "missed": tot - red,
                   "placed_edge": by_cat["edge"][1], "missed_edge": by_cat["edge"][1] - by_cat["edge"][0],
                   "placed_interior": by_cat["interior"][1], "missed_interior": by_cat["interior"][1] - by_cat["interior"][0],
                   "redetection_rate_edge": by_cat["edge"][0] / max(by_cat["edge"][1], 1),
                   "redetection_rate_interior": by_cat["interior"][0] / max(by_cat["interior"][1], 1),
                   "missed_by_distance_to_window_edge": by_dist, "missed_with_T_zone_ge_R": t_tall,
                   "fn_vs_reference_total": raw[k]["fn"], "fn_rate_patient_macro": rates[k]["fn_rate"]}
    edge_int = {}
    for k in waves:
        c = {"edge": 0, "interior": 0}
        for i in range(N):
            for r in attrib[k][i]:
                if r["edge"] is not None:
                    c["edge" if r["edge"] else "interior"] += 1
        edge_int[k] = {"fp_nearest_event_edge": c["edge"], "fp_nearest_event_interior": c["interior"],
                       "missed_placed_edge": miss[k]["missed_edge"], "missed_placed_interior": miss[k]["missed_interior"],
                       "redetection_rate_edge": miss[k]["redetection_rate_edge"],
                       "redetection_rate_interior": miss[k]["redetection_rate_interior"]}

    # ---- waveform side effects
    flat = {f"{a}-ORIG": orig_flat_fraction(beats[a], pos, empty_fill) for a in ARMS}
    flat |= {f"{a}-CELL": np.array([inf["flat_fill_fraction"] for inf in info[a]]) for a in ARMS}
    pou = {a: float(max(inf["pou_error"] for inf in info[a])) for a in ARMS}
    flat_summary = {}
    for k, v in flat.items():
        pp = np.array([v[Pv == s].mean() for s in np.unique(Pv)])
        flat_summary[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "iqr": np.percentile(v, [25, 75]).tolist(),
                           "p95": float(np.percentile(v, 95)), "patient_mean_q10_50_90": np.percentile(pp, [10, 50, 90]).tolist(),
                           "patient_macro": B.cluster_ci(v, Pv)}
    flat_increase = {a: B.cluster_ci(flat[f"{a}-CELL"] - flat[f"{a}-ORIG"], Pv) for a in ARMS}
    fd = {k: float(PMX.kanflow_fd(w, Yv)) for k, w in waves.items()}
    fd_diff = {a: B.fd_diff_ci(cell[a], orig[a], Yv, Pv)[0] for a in ("S", "D")}
    pairs = [BR.matched_pairs(ref[i], pos[i], T) for i in range(N)]
    corr = {k: [BR.pair_correlations(Yv[i], waves[k][i], pairs[i]) for i in range(N)] for k in waves}
    wm = B.window_pair_means(corr, tuple(waves))
    st = {k: np.array(list(ex.map(B._structure, [(waves[k][i], Yv[i], ref[i]) for i in range(N)], chunksize=128))) for k in waves}
    gt_stat = np.array([BR.boundary_statistic(Yv[i], ref[i]) for i in range(N)])
    thr = float(np.nanpercentile(gt_stat, 99.9))
    J = {k: float(np.mean(np.array([BR.boundary_statistic(waves[k][i], pos[i]) for i in range(N)]) > thr)) for k in waves}
    spec = {}
    for k in waves:
        rows = [spectral_metrics(waves[k][i], Yv[i]) for i in range(N)]
        spec[k] = {b: np.array([r[f"{b}__ratio_dev"] for r in rows]) for b in ("F1", "F2", "F3", "F4")}
        spec[k]["band_mean"] = np.mean([spec[k][b] for b in ("F1", "F2", "F3", "F4")], axis=0)
    wave_out = {}
    for k in waves:
        wave_out[k] = {"fd": fd[k], "beat_corr_matched": B.cluster_ci(wm[k], Pv), "s4_qrs_deriv_rmse": B.cluster_ci(st[k][:, 0], Pv),
                       "s5_qrs_curvature_err": B.cluster_ci(st[k][:, 1], Pv), "hr_abs_err": rates[k]["hr_abs_err"],
                       "join_artifact_rate_J": J[k],
                       "spectral_ratio_dev": {b: float(np.nanmean(spec[k][b])) for b in ("F1", "F2", "F3", "F4", "band_mean")},
                       "flat_fill_mean": flat_summary[k]["mean"]}
    wave_diff = {a: {"fd_cell_minus_orig": fd_diff[a],
                     "beat_corr_cell_minus_orig": B.cluster_ci(wm[f"{a}-CELL"] - wm[f"{a}-ORIG"], Pv),
                     "s4_cell_minus_orig": B.cluster_ci(st[f"{a}-CELL"][:, 0] - st[f"{a}-ORIG"][:, 0], Pv),
                     "s5_cell_minus_orig": B.cluster_ci(st[f"{a}-CELL"][:, 1] - st[f"{a}-ORIG"][:, 1], Pv),
                     "hr_abs_err_cell_minus_orig": B.cluster_ci(per[f"{a}-CELL"]["hr_abs_err"] - per[f"{a}-ORIG"]["hr_abs_err"], Pv),
                     "spectral_band_mean_cell_minus_orig": B.cluster_ci(spec[f"{a}-CELL"]["band_mean"] - spec[f"{a}-ORIG"]["band_mean"], Pv),
                     "flat_fill_cell_minus_orig": flat_increase[a]} for a in ("S", "D")}

    # ---- pre-specified sensitivity at the widest TRAIN candidate (descriptive only; never enters the verdict)
    sens = {}
    for a in ("S", "D"):
        ys, inf16 = render_cell(beats[a], pos, SENS_WIDTH, rr_single, empty_fill)
        ys = ys.astype(np.float64)
        dk = list(ex.map(B._peaks, list(ys), chunksize=256))
        prf16 = PMX.rpeak_prf_at(ys, Yv, FS, 50.0, peaks=(ref, dk))
        fp16 = [fp_detections(ref[i], dk[i]) for i in range(N)]
        inter16 = np.array([sum(1 for d in fp16[i] if EC.fp_zone(d, pos[i])[1] == "interior") for i in range(N)])
        new16, near16 = 0, 0
        for i in range(N):
            fo = np.asarray(fps[f"{a}-ORIG"][i])
            for d in fp16[i]:
                if fo.size == 0 or np.min(np.abs(fo - d)) > TOL_SAMPLES:
                    new16 += 1
                    near16 += EC.boundary_near(int(d), inf16[i]["centres"], (inf16[i]["left_limit"], inf16[i]["right_limit"]),
                                               SENS_WIDTH) is not None
        sens[a] = {"fp_rate_diff_vs_orig": B.cluster_ci(prf16["n_fp"] - per[f"{a}-ORIG"]["fp"], Pv),
                   "interior_far_fp_diff_vs_orig": B.cluster_ci(inter16 - interior_far[f"{a}-ORIG"], Pv),
                   "f1_minus_placed": B.cluster_ci(np.where(evaluable, prf16["rpeak_f1"], np.nan) - pos_f1, Pv),
                   "raw_fp": int(np.nansum(prf16["n_fp"])), "new_cell_only_fp": new16, "boundary_near_new_fp": near16,
                   "flat_fill_mean": float(np.mean([x["flat_fill_fraction"] for x in inf16])), "fd": float(PMX.kanflow_fd(ys, Yv))}
    write_json("sensitivity_w16.json", {"status": "pre-specified sensitivity, descriptive only; never enters the D0 verdict",
                                        "width_samples": SENS_WIDTH} | sens)

    # ---- decision (frozen)
    fd_limit = abs(json.loads((BF0_ART / "gate_metrics.json").read_text())["detail"]["G3"]["A3-A2"][0])
    flags = {"D1": bool(d1[2] < 0), "D2": bool(d2[2] < 0), "D3": bool(g_cell[1] > -G1_MARGIN), "D4": bool(d4[2] < 0),
             "flat_fill_increase": flat_increase["S"][0], "boundary_new_fp_frac": boundary["S"]["boundary_near_over_removed"],
             "fd_increase": fd_diff["S"][0], "fd_increase_max": fd_limit}
    verdict = d0_verdict(flags)

    head = {"prereg": PREREG, "post_hoc_with_respect_to_BF0": True, "same_validation_data_as_hypothesis": True,
            "bf0_verdict_unchanged": "Case A, G1 FAIL, BF1 NO-GO", "verdict": verdict}
    write_json("primary_metrics.json", head | {
        "flags": flags, "D1_fp_rate_diff_S": d1, "D2_interior_far_fp_rate_diff_S": d2, "D2_relative_reduction": d2_rel,
        "D3": {"placed_events_f1": B.cluster_ci(pos_f1, Pv), "S-ORIG_f1": rates["S-ORIG"]["f1"], "S-CELL_f1": rates["S-CELL"]["f1"],
               "S-ORIG_minus_placed": g_orig, "bf0_g1_reproduced": bf0_g1, "S-CELL_minus_placed": g_cell,
               "former_margin_met": flags["D3"], "D-ORIG_minus_placed": g_dorig, "D-CELL_minus_placed": g_dcell},
        "D4_fp_rate_diff_D": d4, "fp_rates": {k: rates[k]["fp_rate"] for k in waves},
        "interior_far_fp_rates": if_rate, "raw_counts": raw, "partition_of_unity_max_error": pou,
        "crossfade_width_samples": width, "safety_thresholds": SAFETY | {"fd_increase_max": fd_limit}})
    write_json("bootstrap.json", {"unit": "patient", "replicates": B.BOOT_N, "fd_replicates": B.BOOT_N_FD, "seed": B.BOOT_SEED,
                                  "weighting": "equal patient weight", "paired": "identical patient resamples for every arm",
                                  "rates": rates})
    write_json("fp_attribution.json", {"zones": "post-hoc zones frozen from the pre-D0 attribution", "zone_counts": zone_counts,
                                       "far_counts": far_counts, "distances": dist, "reduction": reduction})
    write_json("interior_far_fp.json", {"definition": "FP (vs reference) in zone E_far between two placed events whose gap "
                                                      "<= 1.5 x the window's median placed RR",
                                        "rates": if_rate, "D2_diff_S": d2, "D2_relative_reduction": d2_rel,
                                        "D_diff": B.cluster_ci(interior_far["D-CELL"] - interior_far["D-ORIG"], Pv)})
    write_json("boundary_artifacts.json", {"new_fp": "CELL FP with no ORIG FP within ±50 ms in the same window",
                                           "boundary_near": f"within {width} samples of a crossfade centre or an owned-region edge limit",
                                           "arms": boundary})
    write_json("edge_interior.json", edge_int)
    write_json("miss_analysis.json", miss)
    write_json("waveform_metrics.json", {"per_arm": wave_out, "cell_minus_orig": wave_diff, "J_threshold": thr})
    write_json("flat_fill.json", {"summary": flat_summary, "cell_minus_orig": flat_increase})
    write_json("gap_recovery.json", gap | {"definition": "(F1_S-CELL - F1_S-ORIG) / (F1_placed - F1_S-ORIG), equal patient weight"})
    n_shift = sum(inf["shifted"] for inf in info["S"])
    n_cent = sum(len(inf["centres"]) for inf in info["S"])
    write_json("event_cell_manifest.json", {
        "windows": N, "placed_events": int(sum(len(p) for p in pos)),
        "windows_by_n_events": {str(k): int(v) for k, v in zip(*np.unique([len(p) for p in pos], return_counts=True))},
        "edge_events": int(sum(e.sum() for e in edge)), "interior_events": int(sum((~e).sum() for e in edge)),
        "crossfades": n_cent, "crossfades_shifted_off_midpoint": n_shift,
        "gaps_without_support_overlap": int(sum(inf["gaps"] for inf in info["S"])),
        "partition_of_unity_max_error": pou, "outputs": B.rel(OUT / "cell_renders_val.npz")})
    print(json.dumps(B.clean({"verdict": verdict, "flags": flags, "D1": d1, "D2": d2, "D3": g_cell, "D4": d4}), indent=1))


def stage_figure(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pm = json.loads((ART / "primary_metrics.json").read_text())
    fa = json.loads((ART / "fp_attribution.json").read_text())
    wm = json.loads((ART / "waveform_metrics.json").read_text())
    ff = json.loads((ART / "flat_fill.json").read_text())
    cfg = json.loads((ART / "renderer_config.json").read_text())
    width, rr_single = int(cfg["crossfade_width_samples"]), float(cfg["rr_single_samples"])
    Xv, Yv, _ = B.load_role("val")
    R, pos, template, rr_median, empty_fill = frozen_bf0()
    vb = np.load(OUT / "val_beats.npz")
    cr = np.load(OUT / "cell_renders_val.npz")
    beats_S = split_beats(vb["S"], pos)
    ref = [np.asarray(p, int) for p in B.split_list(np.load(B.RUN / "val_rpeaks.npz")["idx"], np.load(B.RUN / "val_rpeaks.npz")["off"])]
    C = {"orig": "#B0621B", "cell": "#0E7A86", "ref": "#8A94A3", "D": "#4A3AA7"}
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    # A: one frozen stochastic beat output with the next R inside its support (first in salted order)
    a = fig.add_subplot(gs[0, 0])
    order = B.salted_rank(SALT_EXAMPLE, range(len(pos)))
    for n in order:
        P = pos[n]
        if len(P) >= 2 and P[1] - P[0] + 64 < 160:
            b = beats_S[n][0]
            tt = (np.arange(166) - 64) / FS * 1000
            a.plot(tt, b, color=C["cell"], lw=1.2)
            a.axvline(0, color="k", lw=0.6, ls=":")
            a.axvline((P[1] - P[0]) / FS * 1000, color=C["orig"], lw=0.8, ls="--")
            a.axvspan(-(P[1] - P[0]) / 2 / FS * 1000, (P[1] - P[0]) / 2 / FS * 1000, color=C["cell"], alpha=0.08)
            a.set_title(f"A  one frozen BF0 beat output (window {n})\nshaded: its event cell; dashed: next placed R", loc="left", fontsize=9)
            break
    a.set_xlabel("ms from its R")
    # B: schematic of event cells on a window
    b_ax = fig.add_subplot(gs[0, 1])
    P = pos[order[0]] if len(pos[order[0]]) >= 2 else next(pos[n] for n in order if len(pos[n]) >= 2)
    W, rec = EC.cell_weights(P, T, width, rr_single)
    for i in range(len(P)):
        b_ax.plot(np.arange(T) / FS, W[i], lw=1.2)
    for p in P:
        b_ax.axvline(p / FS, color="k", lw=0.5, ls=":")
    b_ax.set_title(f"B  event-cell weights (crossfade {width} samples = {width / FS * 1000:.1f} ms)", loc="left", fontsize=9)
    b_ax.set_xlabel("s")
    # C: two examples by a frozen rule: first salted window where an ORIG far FP disappears, first where CELL adds an FP
    c_ax = fig.add_subplot(gs[0, 2:])
    ex_gone, ex_new = None, None
    S_orig = np.asarray(R["A3"][0], np.float64)
    for n in order:
        if ex_gone is not None and ex_new is not None:
            break
        do = B._peaks(S_orig[n])
        dc = B._peaks(cr["S_CELL"][n].astype(np.float64))
        fo, fc_ = fp_detections(ref[n], do), fp_detections(ref[n], dc)
        if ex_gone is None and any(EC.fp_zone(d, pos[n])[0] == "E_far" and all(abs(d - e) > TOL_SAMPLES for e in fc_) for d in fo):
            ex_gone = (n, fo, fc_)
        if ex_new is None and (ex_gone is None or ex_gone[0] != n) and any(all(abs(d - e) > TOL_SAMPLES for e in fo) for d in fc_):
            ex_new = (n, fo, fc_)
    t = np.arange(T) / FS
    off = 0.0
    for lab, exm in (("FP removed", ex_gone), ("FP added (shown, not hidden)", ex_new)):
        if exm is None:
            continue
        n, fo, fc_ = exm
        c_ax.plot(t, Yv[n] + off, color=C["ref"], lw=1.2)
        c_ax.plot(t, S_orig[n] + off, color=C["orig"], lw=0.9)
        c_ax.plot(t, cr["S_CELL"][n] + off, color=C["cell"], lw=0.9)
        c_ax.plot(np.asarray(fo) / FS, S_orig[n][fo] + off, "v", color=C["orig"], ms=7)
        c_ax.plot(np.asarray(fc_) / FS, cr["S_CELL"][n][fc_] + off, "^", color=C["cell"], ms=7)
        c_ax.text(0.01, off + 1.05, f"window {n}: {lab}", fontsize=8)
        off -= 2.6
    c_ax.set_title("C  reference (grey), S-ORIG (orange, ▼ FP), S-CELL (teal, ▲ FP); same frozen beats", loc="left", fontsize=9)
    c_ax.set_xlabel("s")
    c_ax.set_yticks([])
    # D: FP rates
    d_ax = fig.add_subplot(gs[1, 0])
    ks = ["S-ORIG", "S-CELL", "D-ORIG", "D-CELL"]
    v = [pm["fp_rates"][k] for k in ks]
    d_ax.bar(range(4), [x[0] for x in v], color=[C["orig"], C["cell"], C["orig"], C["cell"]], width=0.6)
    for i, x in enumerate(v):
        d_ax.errorbar(i, x[0], yerr=[[x[0] - x[1]], [x[2] - x[0]]], color="k", capsize=4, lw=1)
    d_ax.set_xticks(range(4), ks, fontsize=8)
    d_ax.set_ylabel("false R detections per window")
    d_ax.set_title(f"D  D1 {'PASS' if pm['flags']['D1'] else 'FAIL'} / D4 {'PASS' if pm['flags']['D4'] else 'FAIL'}", loc="left", fontsize=9)
    # E: FP attribution
    e_ax = fig.add_subplot(gs[1, 1])
    cats = [("edge far", lambda k: fa["far_counts"][k]["before_first"] + fa["far_counts"][k]["after_last"]),
            ("interior far", lambda k: fa["far_counts"][k]["interior"]),
            ("gap far", lambda k: fa["far_counts"][k]["gap_missed_event"]),
            ("P/PR", lambda k: fa["zone_counts"][k]["D_before_300_50ms"]),
            ("S/ST", lambda k: fa["zone_counts"][k]["B_after_50_150ms"]),
            ("T", lambda k: fa["zone_counts"][k]["C_after_150_450ms"]),
            ("±50 ms", lambda k: fa["zone_counts"][k]["A_within_50ms"])]
    xs = np.arange(len(cats))
    for j, (k, col) in enumerate((("S-ORIG", C["orig"]), ("S-CELL", C["cell"]))):
        e_ax.bar(xs + (j - 0.5) * 0.38, [f(k) for _, f in cats], width=0.38, color=col, label=k)
    e_ax.set_xticks(xs, [c for c, _ in cats], fontsize=8, rotation=30)
    e_ax.set_ylabel("false detections (count)")
    e_ax.legend(fontsize=8)
    e_ax.set_title(f"E  attribution; D2 {'PASS' if pm['flags']['D2'] else 'FAIL'}", loc="left", fontsize=9)
    # F: G1 recomputation
    f_ax = fig.add_subplot(gs[1, 2])
    g = pm["D3"]
    vals = [g["S-ORIG_minus_placed"], g["S-CELL_minus_placed"], g["D-ORIG_minus_placed"], g["D-CELL_minus_placed"]]
    f_ax.bar(range(4), [x[0] for x in vals], color=[C["orig"], C["cell"], C["orig"], C["cell"]], width=0.6)
    for i, x in enumerate(vals):
        f_ax.errorbar(i, x[0], yerr=[[x[0] - x[1]], [x[2] - x[0]]], color="k", capsize=4, lw=1)
    f_ax.axhline(-G1_MARGIN, color="k", ls="--", lw=0.8)
    f_ax.axhline(0, color="k", lw=0.5)
    f_ax.set_xticks(range(4), ["S-ORIG", "S-CELL", "D-ORIG", "D-CELL"], fontsize=8)
    f_ax.set_ylabel("F1 render − F1 placed events")
    f_ax.set_title(f"F  former BF0 G1 margin (−0.02): {'met' if pm['flags']['D3'] else 'not met'} post hoc\nBF0 verdict stays FAIL", loc="left", fontsize=9)
    # inset table
    t_ax = fig.add_subplot(gs[1, 3])
    t_ax.axis("off")
    rows = [["", "S-ORIG", "S-CELL", "D-ORIG", "D-CELL"],
            ["FD"] + [f"{wm['per_arm'][k]['fd']:.1f}" for k in ks],
            ["beat corr"] + [f"{wm['per_arm'][k]['beat_corr_matched'][0]:.3f}" for k in ks],
            ["flat-fill"] + [f"{ff['summary'][k]['mean']:.3f}" for k in ks],
            ["J"] + [f"{wm['per_arm'][k]['join_artifact_rate_J']:.3f}" for k in ks]]
    tb = t_ax.table(cellText=rows, loc="center", cellLoc="center")
    tb.scale(1, 1.6)
    t_ax.set_title("secondary: waveform side effects", loc="left", fontsize=9)
    fig.suptitle(f"D0 (post hoc on BF0 validation; BF0 verdict unchanged) — {pm['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure.png", dpi=150)


STAGES = {"reproduce": stage_reproduce, "crossfade": stage_crossfade, "manifest": stage_manifest,
          "evaluate": stage_evaluate, "figure": stage_figure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
