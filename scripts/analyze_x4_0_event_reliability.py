"""X4-0 - iMeanFlow event reliability / source-condition / interval diagnostic.

Frozen protocol: docs/X4_0_EVENT_RELIABILITY_PREREGISTRATION.md (prereg 14a248e, pushed before any real-data metric).
NO TRAINING. Frozen-checkpoint inference only. WildPPG test kjd/ssx never loaded (fail-loud firewall).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.evaluation import alignment_diagnostics as AD  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.flow.samplers import heun_sample, nfe_of  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/x4_0_event_reliability"
FS, T_LEN = 128, 1024
VAL = ("an0", "k2s")
NFES = (1, 2, 4, 8, 16, 25, 50)
NFE_SEEDS = (0, 1, 2, 3)
SOURCE_SEEDS = tuple(range(32))
SOURCE_PAIR = {0: 1, 2: 3}          # frozen baseline -> alternative source mapping (prereg sec. 10)
COND_NFES = (1, 8)
BOOT, BOOT_SEED = 2000, 20260830
IMF_CKPT = "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
OT_CKPT = "outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt"
SUBSETS = {"nfe": ("x4-event-nfe-v2", 1024), "source": ("x4-event-source-v2", 256), "schedule": ("x4-event-schedule-v2", 512)}


def load_val(indices_by_subject):
    ER.assert_no_test_subjects(list(indices_by_subject))
    X, Y, S, W = [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = indices_by_subject[s]
        X.append(d["x"][idx].astype(np.float32))
        Y.append(d["y"][idx].astype(np.float32))
        S.append(np.full(len(idx), s))
        W.append(np.asarray(idx))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(W)


def build_subsets():
    out = {}
    for name, (salt, n) in SUBSETS.items():
        per = {}
        for s in VAL:
            d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
            per[s] = ER.select_subset(salt, s, len(d["x"]), n)
        out[name] = per
    return out


def load_imf(dev):
    ck = torch.load(ROOT / IMF_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    return net, ck


def load_ot(dev):
    ck = torch.load(ROOT / OT_CKPT, map_location="cpu", weights_only=False)
    m = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


def source_bank(seed, n):
    g = torch.Generator().manual_seed(int(seed))
    return torch.randn(n, 1, T_LEN, generator=g)


@torch.no_grad()
def gen_imf(net, ppg_np, e, h_list, batch, dev, nfe_log):
    out = []
    for i in range(0, len(ppg_np), batch):
        p = torch.from_numpy(ppg_np[i:i + batch]).to(dev).unsqueeze(1)
        z, nfe = ER.sample_meanflow_schedule(net, p, e[i:i + batch].to(dev), h_list)
        nfe_log.add(int(nfe))
        out.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def gen_ot50(model, ppg_np, e, batch, dev, nfe_log):
    out = []
    for i in range(0, len(ppg_np), batch):
        p = torch.from_numpy(ppg_np[i:i + batch]).to(dev).unsqueeze(1)
        v = lambda x, t: model.forward_step(x, p, t)  # noqa: E731
        z, nfe = heun_sample(v, e[i:i + batch].to(dev), 25)
        assert nfe == nfe_of("heun", 25) == 50, f"OT-CFM reference must be 50 NFE, got {nfe}"
        nfe_log.add(int(nfe))
        out.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------------------------------
# Metric blocks A / B / C, all reusing X0 semantics
# ----------------------------------------------------------------------------------------------------------------------
def _score_chunk(args):
    pred, gt, gt_pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    rows = []
    for i in range(len(pred)):
        ev = AD.event_timing(gt[i], pred[i], FS, tol_ms=ER.MATCH_TOL_MS)
        bl = AD.beat_level_analysis(pred[i], gt[i], gt_pk[i], FS, int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS)))
        n_ref = max(ev["n_ref"], 1)
        half = int(round(AD.QRS_HALF_MS / 1000 * FS))
        mask = np.zeros(len(gt[i]), bool)
        for r in gt_pk[i]:
            mask[max(0, r - half):r + half + 1] = True
        d, dg = np.diff(pred[i]) * FS, np.diff(gt[i]) * FS
        err = ev["signed_err_ms"]
        absent = ((np.nan_to_num(bl["oracle_corr"], nan=-1) < 0.5) | (bl["oracle_p2p_ratio"] < 0.2)) if bl["n_beats"] else np.array([])
        rows.append({
            "precision": float(rm["rpeak_precision"][i]), "recall": float(rm["rpeak_recall"][i]), "f1": float(rm["rpeak_f1"][i]),
            "n_pred_beats": int(ev["n_pred"]), "n_gt_beats": int(ev["n_ref"]), "beats_ratio": ev["n_pred"] / n_ref,
            "missing": ev["n_missing"] / n_ref, "spurious": ev["n_spurious"] / n_ref,
            "rr_mae_ms": float(rm["rr_mae_ms"][i]), "timing_mae_ms": float(np.abs(err).mean()) if len(err) else np.nan,
            "timing_bias_ms": float(err.mean()) if len(err) else np.nan, "timing_sd_ms": float(err.std()) if len(err) > 1 else np.nan,
            "oracle_absent": float(absent.mean()) if absent.size else np.nan,
            "same_coord_corr": float(np.nanmean(bl["raw_corr"])) if bl["n_beats"] else np.nan,
            "oracle_corr": float(np.nanmean(bl["oracle_corr"])) if bl["n_beats"] else np.nan,
            "oracle_p2p": float(np.nanmedian(bl["oracle_p2p_ratio"])) if bl["n_beats"] else np.nan,
            "oracle_qrs_e": float(np.nanmedian(bl["oracle_qrs_energy_ratio"])) if bl["n_beats"] else np.nan,
            "oracle_slope": float(np.nanmedian(bl["oracle_slope_ratio"])) if bl["n_beats"] else np.nan,
            "morph": float(rm["morph_corr"][i]), "rmse": float(np.sqrt(((pred[i] - gt[i]) ** 2).mean())),
            "mae": float(np.abs(pred[i] - gt[i]).mean()), "amp_ratio": float(pred[i].std() / (gt[i].std() + 1e-8)),
            "qrs_energy": float(pred[i][mask].var() / (gt[i][mask].var() + 1e-12)) if mask.any() else np.nan,
            "slope_ratio": float(np.abs(d).max() / (np.abs(dg).max() + 1e-12)),
            "hf_ratio": float(hf_energy_ratio(pred[i][None])[0]), "hf_gt": float(hf_energy_ratio(gt[i][None])[0]),
        })
    return rows


def score(pred, gt, gt_pk, workers):
    ch = [c for c in np.array_split(np.arange(len(pred)), max(1, workers)) if len(c)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_score_chunk, [(pred[c], gt[c], [gt_pk[i] for i in c]) for c in ch]))
    return [r for part in res for r in part]


def _peaks_chunk(a):
    y, = a
    return [R.detect_rpeaks(w, FS) for w in y]


def gt_peaks(gt, workers):
    ch = [c for c in np.array_split(np.arange(len(gt)), workers * 2) if len(c)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_peaks_chunk, [(gt[c],) for c in ch]))
    return [p for part in res for p in part]


def agg(rows, subjects, keys, boot=True):
    out = {}
    subj = np.asarray(subjects)
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        if boot:
            p, lo, hi = ER.subject_stratified_bootstrap(v, subj, BOOT, BOOT_SEED)
            out[k], out[k + "_lo"], out[k + "_hi"] = p, lo, hi
        else:
            out[k] = float(np.nanmean(v))
        for s in VAL:
            m = subj == s
            out[f"{k}__{s}"] = float(np.nanmean(v[m])) if m.any() else np.nan
    return out


MKEYS = ["f1", "precision", "recall", "spurious", "missing", "beats_ratio", "rr_mae_ms", "timing_mae_ms", "timing_bias_ms",
         "timing_sd_ms", "oracle_absent", "same_coord_corr", "oracle_corr", "oracle_p2p", "oracle_qrs_e", "oracle_slope",
         "morph", "rmse", "mae", "amp_ratio", "qrs_energy", "slope_ratio", "hf_ratio", "hf_gt"]


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    # debug-only shrink flags; defaults are the FROZEN protocol values
    ap.add_argument("--smoke", action="store_true", help="debug only: tiny subsets/seeds to exercise the code path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global SUBSETS, NFES, NFE_SEEDS, SOURCE_SEEDS, SOURCE_PAIR, COND_NFES, BOOT, OUT
    if args.out:
        OUT = ROOT / args.out
    if args.smoke:
        SUBSETS = {"nfe": ("x4-event-nfe-v2", 24), "source": ("x4-event-source-v2", 12), "schedule": ("x4-event-schedule-v2", 12)}
        NFES, NFE_SEEDS, SOURCE_SEEDS, SOURCE_PAIR, COND_NFES, BOOT = (1, 4), (0,), tuple(range(4)), {0: 1}, (1,), 50
        print("[DEBUG] SMOKE MODE - not a protocol change; frozen values are NFE", (1, 2, 4, 8, 16, 25, 50), "seeds 0-3, 32 sources")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ER.assert_no_test_subjects(VAL)
    subs = build_subsets()
    for name, per in subs.items():
        (OUT / f"{name}_subset.json").write_text(json.dumps({s: per[s].tolist() for s in VAL}, indent=1))
    imf, imf_ck = load_imf(dev)
    ot, ot_ck = load_ot(dev)
    nfe_log = set()
    gate = {}

    # ---------- X4-0A: training interval exposure (no waveform data) ----------
    g = torch.Generator().manual_seed(BOOT_SEED)
    from ppg2ecg.flow.imeanflow import sample_tr
    t_, r_, _ = sample_tr(2_000_000, generator=g, p_mean=-0.4, p_std=1.0, data_proportion=0.5)
    hh = (t_ - r_).squeeze(1).numpy()
    nz = hh[hh > 0]
    hdist = {"n": int(len(hh)), "seed": BOOT_SEED, "frac_h_zero": float((hh == 0).mean()), "mean": float(hh.mean()),
             "median": float(np.median(hh)), "median_given_positive": float(np.median(nz)), "max_observed": float(hh.max()),
             **{f"q{q}": float(np.percentile(hh, q)) for q in (75, 90, 95, 99, 99.9)},
             "tail": {str(x): float((hh >= x).mean()) for x in (0.0625, 0.125, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95)},
             "inference_markers": {str(n): {"h": 1.0 / n, "P_train_ge": float((hh >= 1.0 / n).mean())} for n in (1, 2, 4, 8, 16)}}
    (OUT / "h_distribution.json").write_text(json.dumps(hdist, indent=1))
    write_csv(OUT / "h_distribution.csv", [{"threshold": x, "P_train_h_ge": float((hh >= x).mean())} for x in (0.0625, 0.125, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)])
    print(f"[A] h: zero-frac {hdist['frac_h_zero']:.4f}  max {hdist['max_observed']:.4f}  P(h>=.5) {hdist['tail']['0.5']:.5f}  P(h>=.7) {hdist['tail']['0.7']:.5f}  P(h>=1) {(hh>=1).mean():.6f}", flush=True)

    # ---------- X4-0B + OT50: NFE frontier ----------
    Xn, Yn, Sn, Wn = load_val(subs["nfe"])
    pk_n = gt_peaks(Yn.astype(np.float64), args.workers)
    print(f"[B] NFE subset {len(Xn)} windows ({(Sn=='an0').sum()} an0 / {(Sn=='k2s').sum()} k2s), {sum(len(p) for p in pk_n)} GT beats", flush=True)
    nfe_rows, by_seed = [], []
    for seed in NFE_SEEDS:
        e = source_bank(seed, len(Xn))
        for n in NFES:
            pred = gen_imf(imf, Xn, e, ER.UNIFORM[n], args.batch, dev, nfe_log)
            rows = score(pred.astype(np.float64), Yn.astype(np.float64), pk_n, args.workers)
            by_seed.append({"model": "iMF", "nfe": n, "seed": seed, **agg(rows, Sn, MKEYS, boot=False)})
            nfe_rows.append((("iMF", n, seed), rows))
        pred = gen_ot50(ot, Xn, e, args.batch, dev, nfe_log)
        rows = score(pred.astype(np.float64), Yn.astype(np.float64), pk_n, args.workers)
        by_seed.append({"model": "OT-CFM", "nfe": 50, "seed": seed, **agg(rows, Sn, MKEYS, boot=False)})
        nfe_rows.append((("OT-CFM", 50, seed), rows))
        print(f"[B] seed {seed} done", flush=True)
    pooled = []
    for model, n in [("iMF", n) for n in NFES] + [("OT-CFM", 50)]:
        rr = [r for (m, nn, _), rs in nfe_rows if m == model and nn == n for r in rs]
        ss = np.concatenate([Sn for (m, nn, _), _ in nfe_rows if m == model and nn == n])
        pooled.append({"model": model, "nfe": n, "n_scored": len(rr), **agg(rr, ss, MKEYS)})
        p = pooled[-1]
        print(f"[B] {model:7s} NFE {n:>3}: morph {p['morph']:.4f} F1 {p['f1']:.4f} spur {p['spurious']:.4f} RR {p['rr_mae_ms']:.1f} absent {p['oracle_absent']:.4f} oracorr {p['oracle_corr']:.4f} qrsE {p['qrs_energy']:.3f} slope {p['slope_ratio']:.3f} hf {p['hf_ratio']:.3f}", flush=True)
    write_csv(OUT / "nfe_metrics.csv", pooled)
    write_csv(OUT / "nfe_metrics_by_seed.csv", by_seed)
    write_csv(OUT / "nfe_metrics_by_subject.csv", [{"model": p["model"], "nfe": p["nfe"], "subject": s, **{k: p[f"{k}__{s}"] for k in MKEYS}} for p in pooled for s in VAL])
    write_csv(OUT / "ot50_reference.csv", [p for p in pooled if p["model"] == "OT-CFM"])
    gate["nfe"] = pooled

    # ---------- X4-0C: 32-source event diagnostic ----------
    Xs, Ys, Ss, Ws = load_val(subs["source"])
    pk_s = gt_peaks(Ys.astype(np.float64), args.workers)
    src_rows, anchor_rows = [], []
    store = {}
    for n in (1, 4, 8, 16):
        preds = np.stack([gen_imf(imf, Xs, source_bank(sd, len(Xs)), ER.UNIFORM[n], args.batch, dev, nfe_log) for sd in SOURCE_SEEDS])
        store[n] = preds
        flat = [gt_peaks(preds[k].astype(np.float64), args.workers) for k in range(len(SOURCE_SEEDS))]
        pk = [[flat[k][i] for k in range(len(SOURCE_SEEDS))] for i in range(len(Xs))]
        per_win = []
        for i in range(len(Xs)):
            counts = np.array([len(p) for p in pk[i]])
            f1s = [ER.peak_train_agreement(pk[i][a], pk[i][b])["f1"] for a in range(len(SOURCE_SEEDS)) for b in range(a + 1, len(SOURCE_SEEDS))]
            an = ER.gt_anchored_presence(pk_s[i], pk[i])
            elig = an["n_detected"] >= (len(SOURCE_SEEDS) // 2)
            per_win.append({"seed_pair_f1": float(np.median(f1s)), "beat_count_sd": float(counts.std(ddof=1)),
                            "beat_count_mean": float(counts.mean()), "beat_count_min": int(counts.min()), "beat_count_max": int(counts.max()),
                            "gt_detection_prob": float(np.median(an["detection_probability"])) if len(an["detection_probability"]) else np.nan,
                            "frac_unstable_beats": float(np.mean((an["detection_probability"] >= 0.25) & (an["detection_probability"] <= 0.75))) if len(an["detection_probability"]) else np.nan,
                            "cond_timing_sd_ms": float(np.nanmedian(an["timing_sd_ms"][elig])) if elig.any() else np.nan})
            anchor_rows.append({"nfe": n, "subject": Ss[i], "window": int(Ws[i]), **per_win[-1]})
        task = {k: [] for k in ("f1", "rr_mae_ms", "morph", "oracle_corr")}
        for k in range(len(SOURCE_SEEDS)):
            rs = score(preds[k].astype(np.float64), Ys.astype(np.float64), pk_s, args.workers)
            for key in task:
                task[key].append([r[key] for r in rs])
        sds = {f"{k}_sd": float(np.nanmedian(np.nanstd(np.array(v), axis=0))) for k, v in task.items()}
        row = {"nfe": n, **{k: float(np.nanmedian([w[k] for w in per_win])) for k in per_win[0]}, **sds}
        for s in VAL:
            m = Ss == s
            row.update({f"seed_pair_f1__{s}": float(np.nanmedian([per_win[i]["seed_pair_f1"] for i in np.where(m)[0]]))})
        src_rows.append(row)
        print(f"[C] NFE {n:>2}: seed-pair F1 {row['seed_pair_f1']:.4f} beatSD {row['beat_count_sd']:.3f} detP {row['gt_detection_prob']:.3f} condTimingSD {row['cond_timing_sd_ms']:.1f} ms F1sd {row['f1_sd']:.4f}", flush=True)
    write_csv(OUT / "source_event_variability.csv", src_rows)
    write_csv(OUT / "source_gt_anchor_variability.csv", anchor_rows)
    gate["source"] = src_rows

    # ---------- X4-0C2: mandatory PPG-derangement condition perturbation ----------
    rng = np.random.default_rng(1)
    while True:
        perm = rng.permutation(len(Xs))
        if not np.any(perm == np.arange(len(Xs))):
            break
    cond_rows = []
    for n in COND_NFES:
        for base, alt in SOURCE_PAIR.items():
            A = store[n][SOURCE_SEEDS.index(base)] if n in store else gen_imf(imf, Xs, source_bank(base, len(Xs)), ER.UNIFORM[n], args.batch, dev, nfe_log)
            B = store[n][SOURCE_SEEDS.index(alt)] if n in store else gen_imf(imf, Xs, source_bank(alt, len(Xs)), ER.UNIFORM[n], args.batch, dev, nfe_log)
            C = gen_imf(imf, Xs[perm], source_bank(base, len(Xs)), ER.UNIFORM[n], args.batch, dev, nfe_log)
            pA = gt_peaks(A.astype(np.float64), args.workers)
            pB = gt_peaks(B.astype(np.float64), args.workers)
            pC = gt_peaks(C.astype(np.float64), args.workers)
            for lab, P, Q in (("source", pA, pB), ("ppg_shuffle", pA, pC)):
                Wv = A if lab == "source" else A
                Ov = B if lab == "source" else C
                f1 = [ER.peak_train_agreement(P[i], Q[i])["f1"] for i in range(len(Xs))]
                dcount = [abs(len(P[i]) - len(Q[i])) for i in range(len(Xs))]
                pcc = [float(np.corrcoef(Wv[i], Ov[i])[0, 1]) if Wv[i].std() > 1e-8 and Ov[i].std() > 1e-8 else np.nan for i in range(len(Xs))]
                tim = []
                for i in range(len(Xs)):
                    pr, _, _ = R.match_rpeaks(P[i], Q[i], FS, tol_ms=ER.MATCH_TOL_MS) if len(P[i]) and len(Q[i]) else ([], 0, 0)
                    tim.append(float(np.mean([abs(P[i][a] - Q[i][b]) for a, b in pr]) / FS * 1000) if pr else np.nan)
                cond_rows.append({"nfe": n, "base_seed": base, "alt_seed": alt, "perturbation": lab,
                                  "event_f1": float(np.nanmean(f1)), "event_disagreement": float(1 - np.nanmean(f1)),
                                  "beat_count_diff": float(np.mean(dcount)), "timing_disagreement_ms": float(np.nanmean(tim)),
                                  "waveform_pcc": float(np.nanmean(pcc)), "waveform_disagreement": float(1 - np.nanmean(pcc))})
                print(f"[C2] NFE {n} seed {base}->{alt} {lab:12s}: event F1 {np.nanmean(f1):.4f}  dBeats {np.mean(dcount):.2f}  timing {np.nanmean(tim):.1f} ms  wavePCC {np.nanmean(pcc):.3f}", flush=True)
    write_csv(OUT / "condition_perturbation.csv", cond_rows)
    write_csv(OUT / "source_vs_condition.csv", [{"nfe": n, "perturbation": p,
        **{k: float(np.mean([r[k] for r in cond_rows if r["nfe"] == n and r["perturbation"] == p]))
           for k in ("event_disagreement", "beat_count_diff", "timing_disagreement_ms", "waveform_disagreement")}}
        for n in COND_NFES for p in ("source", "ppg_shuffle")])
    gate["condition"] = cond_rows
    del store

    # ---------- X4-0D: fixed-NFE interval stress ----------
    Xc, Yc, Sc, Wc = load_val(subs["schedule"])
    pk_c = gt_peaks(Yc.astype(np.float64), args.workers)
    stress = []
    for name in ("U4", "LN4", "LD4", "U8", "LN8", "LD8"):
        allr, alls = [], []
        for seed in NFE_SEEDS:
            pred = gen_imf(imf, Xc, source_bank(seed, len(Xc)), ER.SCHEDULES[name], args.batch, dev, nfe_log)
            rs = score(pred.astype(np.float64), Yc.astype(np.float64), pk_c, args.workers)
            allr += rs
            alls.append(Sc)
        row = {"schedule": name, "nfe": len(ER.SCHEDULES[name]), "max_h": max(ER.SCHEDULES[name]),
               **agg(allr, np.concatenate(alls), ["f1", "precision", "recall", "spurious", "oracle_absent", "oracle_corr", "morph", "rr_mae_ms"])}
        stress.append(row)
        print(f"[D] {name:4s} (NFE {row['nfe']}, max h {row['max_h']:.2f}): F1 {row['f1']:.4f} spur {row['spurious']:.4f} absent {row['oracle_absent']:.4f} oracorr {row['oracle_corr']:.4f} morph {row['morph']:.4f} RR {row['rr_mae_ms']:.1f}", flush=True)
    write_csv(OUT / "interval_stress.csv", stress)
    gate["stress"] = stress

    # ---------- X4-0E: event-matching tolerance calibration ----------
    cal = []
    for sd in (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        rg = np.random.default_rng(BOOT_SEED)
        f1 = [ER.peak_train_agreement(p, ER.jitter_peaks(p, sd, rng=rg, n_time=T_LEN)) for p in pk_n if len(p)]
        cal.append({"kind": "jitter_sd_ms", "value": sd, "f1": float(np.mean([r["f1"] for r in f1])),
                    "precision": float(np.mean([r["precision"] for r in f1])), "recall": float(np.mean([r["recall"] for r in f1]))})
    for sh in (10.0, 20.0, 30.0, 40.0, 50.0):
        f1 = [ER.peak_train_agreement(p, ER.jitter_peaks(p, 0.0, shift_ms=sh, n_time=T_LEN)) for p in pk_n if len(p)]
        cal.append({"kind": "fixed_shift_ms", "value": sh, "f1": float(np.mean([r["f1"] for r in f1])),
                    "precision": float(np.mean([r["precision"] for r in f1])), "recall": float(np.mean([r["recall"] for r in f1]))})
    write_csv(OUT / "event_matching_calibration.csv", cal)
    print("[E] calibration:", " ".join(f"{c['kind'][:6]}{c['value']:.0f}={c['f1']:.3f}" for c in cal), flush=True)
    gate["calibration"] = cal

    # ---------- latency ----------
    lat = []
    nb = min(64, len(Xn))                      # protocol batch is 64; smoke subsets may be smaller
    xb = torch.from_numpy(Xn[:nb]).to(dev).unsqueeze(1)
    eb = source_bank(0, nb).to(dev)
    for tag, fn, n in [("iMF", lambda k=n: ER.sample_meanflow_schedule(imf, xb, eb, ER.UNIFORM[k]), n) for n in (1, 4, 8, 16, 25, 50)] + \
                      [("OT-CFM", lambda: heun_sample(lambda x, t: ot.forward_step(x, xb, t), eb, 25), 50)]:
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(100):
            t0 = torch.cuda.Event(True), torch.cuda.Event(True)
            t0[0].record()
            fn()
            t0[1].record()
            torch.cuda.synchronize()
            ts.append(t0[0].elapsed_time(t0[1]))
        ts = np.array(ts)
        lat.append({"model": tag, "nfe": n, "median_ms": float(np.median(ts)), "p10_ms": float(np.percentile(ts, 10)),
                    "p90_ms": float(np.percentile(ts, 90)), "batch": int(nb), "samples_per_s": float(nb / (np.median(ts) / 1000))})
        print(f"[L] {tag:7s} NFE {n:>3}: {np.median(ts):.1f} ms  {lat[-1]['samples_per_s']:.0f} samples/s", flush=True)
    write_csv(OUT / "latency.csv", lat)
    gate["latency"] = lat

    # ---------- flags ----------
    imf_pool = [p for p in pooled if p["model"] == "iMF"]
    imf50 = max(imf_pool, key=lambda p: p["nfe"])          # internal iMF reference = highest evaluated NFE (50 in the frozen protocol)
    sat = {}
    for n in (8, 16):
        cand = [x for x in imf_pool if x["nfe"] == n]
        if not cand:
            continue
        p = cand[0]
        sat[n] = bool(abs(p["morph"] - imf50["morph"]) <= 0.03 and abs(p["oracle_corr"] - imf50["oracle_corr"]) <= 0.03
                      and (abs(p["f1"] - imf50["f1"]) <= 0.03 or p["oracle_absent"] <= imf50["oracle_absent"] + 0.05)
                      and p["spurious"] <= imf50["spurious"] + 0.05 and p["rr_mae_ms"] <= imf50["rr_mae_ms"] + 3)
    s1 = next(r for r in src_rows if r["nfe"] == 1)
    crit = {"seed_pair_f1<0.80": s1["seed_pair_f1"] < 0.80, "beat_count_sd>=0.75": s1["beat_count_sd"] >= 0.75,
            "cond_timing_sd>=15ms": s1["cond_timing_sd_ms"] >= 15.0, "f1_sd>=0.05": s1["f1_sd"] >= 0.05}
    material = sum(crit.values()) >= 2
    key = {"seed_pair_f1<0.80": ("seed_pair_f1", -1), "beat_count_sd>=0.75": ("beat_count_sd", 1),
           "cond_timing_sd>=15ms": ("cond_timing_sd_ms", 1), "f1_sd>=0.05": ("f1_sd", 1)}
    resp = {}
    for c, ok in crit.items():
        if not ok:
            continue
        k, sgn = key[c]
        best = max((s1[k] - next(r for r in src_rows if r["nfe"] == n)[k]) * sgn / max(abs(s1[k]), 1e-9) for n in (8, 16))
        resp[c] = float(best)
    nfe_responsive = any(v >= 0.30 for v in resp.values())
    stress_flag = {}
    for base, alts in (("U4", ("LN4", "LD4")), ("U8", ("LN8", "LD8"))):
        b = next(r for r in stress if r["schedule"] == base)
        for a in alts:
            x = next(r for r in stress if r["schedule"] == a)
            stress_flag[f"{a}-{base}"] = {"dF1": x["f1"] - b["f1"], "d_oracle_corr": x["oracle_corr"] - b["oracle_corr"],
                                          "d_spurious": x["spurious"] - b["spurious"], "d_oracle_absent": x["oracle_absent"] - b["oracle_absent"],
                                          "material": bool(x["f1"] - b["f1"] <= -0.05 or x["oracle_corr"] - b["oracle_corr"] <= -0.05
                                                           or x["spurious"] - b["spurious"] >= 0.10 or x["oracle_absent"] - b["oracle_absent"] >= 0.10)}
    gate["flags"] = {"few_step_saturation": sat, "source_material": material, "source_criteria": crit,
                     "source_nfe_response": resp, "source_class": ("NFE-RESPONSIVE SOURCE SENSITIVITY" if (material and nfe_responsive)
                                                                   else "PERSISTENT SOURCE SENSITIVITY" if material else "NOT MATERIAL"),
                     "interval_stress": stress_flag, "observed_nfe": sorted(nfe_log)}
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prereg = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/X4_0_EVENT_RELIABILITY_PREREGISTRATION.md"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "gate_summary.json").write_text(json.dumps(gate, indent=1, default=str))
    (OUT / "provenance.json").write_text(json.dumps({
        "repo_sha": git_sha, "preregistration_sha": prereg, "created": datetime.now().isoformat(timespec="seconds"),
        "script": "scripts/analyze_x4_0_event_reliability.py", "subjects_loaded": sorted(VAL), "test_subjects_loaded": [],
        "firewall": "enforced (assert_no_test_subjects)", "imf_checkpoint": IMF_CKPT, "imf_round": imf_ck.get("epoch"),
        "imf_cfg": imf_ck.get("imf_cfg"), "ot_checkpoint": OT_CKPT, "ot_round": ot_ck.get("epoch"),
        "nfe_grid": list(NFES), "nfe_seeds": list(NFE_SEEDS), "source_seeds": list(SOURCE_SEEDS), "source_pair": SOURCE_PAIR,
        "schedules": {k: v for k, v in ER.SCHEDULES.items()}, "subset_sizes": {k: {s: len(v[s]) for s in VAL} for k, v in subs.items()},
        "excluded_previewed_windows": list(ER.PREVIEWED_WINDOWS), "bootstrap": {"n": BOOT, "seed": BOOT_SEED},
        "observed_nfe": sorted(nfe_log), "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
    }, indent=1, default=str))
    _figures(pooled, src_rows, cond_rows, stress, cal, lat, hdist, hh)
    print(f"\n===== FLAGS: saturation {sat} | source {gate['flags']['source_class']} (material {material}) | stress {[k for k,v in stress_flag.items() if v['material']] or 'none material'} =====")
    print("wrote", OUT)


def _figures(pooled, src, cond, stress, cal, lat, hdist, hh):
    F = OUT / "figures"
    imf = sorted([p for p in pooled if p["model"] == "iMF"], key=lambda p: p["nfe"])
    ot = next(p for p in pooled if p["model"] == "OT-CFM")
    x = [p["nfe"] for p in imf]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, [p["morph"] for p in imf], yerr=[[p["morph"] - p["morph_lo"] for p in imf], [p["morph_hi"] - p["morph"] for p in imf]], fmt="o-", color="tab:red", label="iMF")
    ax.axhline(ot["morph"], color="tab:cyan", ls="--", label=f"OT-CFM-50 ({ot['morph']:.3f})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("NFE")
    ax.set_ylabel("morphology correlation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("X4-0 Fig. 1 - iMF morphology vs NFE (development validation an0/k2s)", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "fig1_morph_vs_nfe.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for a, k, lab in zip(axes.ravel(), ["f1", "precision", "recall", "spurious", "oracle_absent", "oracle_corr"],
                         ["R-peak F1", "precision", "recall", "spurious fraction", "oracle-absent fraction", "oracle beat correlation"]):
        a.errorbar(x, [p[k] for p in imf], yerr=[[p[k] - p[k + "_lo"] for p in imf], [p[k + "_hi"] - p[k] for p in imf]], fmt="o-", color="tab:red", label="iMF")
        a.axhline(ot[k], color="tab:cyan", ls="--", label="OT-CFM-50")
        a.set_xscale("log", base=2)
        a.set_xlabel("NFE")
        a.set_title(lab, fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.suptitle("X4-0 Fig. 2 - event reliability vs NFE, with the OT-CFM-50 contextual reference (not an oracle)", fontsize=12)
    fig.tight_layout()
    fig.savefig(F / "fig2_event_vs_nfe.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    li = [r for r in lat if r["model"] == "iMF"]
    ax.plot([r["nfe"] for r in li], [r["median_ms"] for r in li], "o-", color="tab:red", label="iMF")
    lo = [r for r in lat if r["model"] == "OT-CFM"][0]
    ax.plot([lo["nfe"]], [lo["median_ms"]], "s", color="tab:cyan", label="OT-CFM-50")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("latency, batch 64 (ms)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("X4-0 Fig. 3 - latency vs NFE", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "fig3_latency.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(hh[hh > 0], bins=200, color="0.6", density=True)
    for n, c in ((1, "tab:red"), (2, "tab:orange"), (4, "tab:green"), (8, "tab:blue"), (16, "tab:purple")):
        ax.axvline(1.0 / n, color=c, ls="--", lw=1.2, label=f"NFE {n}: h={1.0/n:.4g} (P={hdist['inference_markers'][str(n)]['P_train_ge']:.4f})")
    ax.set_xlabel("training interval h = t - r  (the 50 % mass at h = 0 exactly is not shown)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(f"X4-0 Fig. 4 - training interval exposure; max observed h = {hdist['max_observed']:.4f}\nexact h = 1 has zero training probability (extreme boundary query)", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "fig4_h_distribution.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = [r["nfe"] for r in src]
    for a, k, lab in zip(axes, ["seed_pair_f1", "beat_count_sd", "cond_timing_sd_ms"],
                         ["median seed-pair event F1", "median beat-count SD across 32 sources", "conditional timing SD (ms), >=16/32 detections"]):
        a.plot(xs, [r[k] for r in src], "o-", color="tab:red")
        a.set_xscale("log", base=2)
        a.set_xlabel("NFE")
        a.set_title(lab, fontsize=9)
        a.grid(alpha=0.3)
    axes[0].axhline(0.80, color="k", ls="--", lw=1, label="flag 0.80")
    axes[1].axhline(0.75, color="k", ls="--", lw=1, label="flag 0.75")
    axes[2].axhline(15.0, color="k", ls="--", lw=1, label="flag 15 ms")
    for a in axes:
        a.legend(fontsize=7)
    fig.suptitle("X4-0 Fig. 6 - source-seed event consistency vs NFE (same PPG, 32 Gaussian sources)", fontsize=11)
    fig.tight_layout()
    fig.savefig(F / "fig6_source_consistency.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.35
    for i, p in enumerate(("source", "ppg_shuffle")):
        vals = [np.mean([r["event_disagreement"] for r in cond if r["nfe"] == n and r["perturbation"] == p]) for n in COND_NFES]
        ax.bar(np.arange(len(COND_NFES)) + (i - 0.5) * w, vals, w, label="Gaussian source" if p == "source" else "PPG shuffle (strong condition anchor)")
    ax.set_xticks(range(len(COND_NFES)), [f"NFE {n}" for n in COND_NFES])
    ax.set_ylabel("event disagreement  (1 - predicted-vs-predicted F1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("X4-0 Fig. 7 - source perturbation vs strong PPG perturbation\n(descriptive context only; the two are NOT on a common causal scale)", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "fig7_source_vs_condition.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, grp in zip(axes, (("U4", "LN4", "LD4"), ("U8", "LN8", "LD8"))):
        rs = [next(r for r in stress if r["schedule"] == g) for g in grp]
        xx = np.arange(3)
        for j, (k, c) in enumerate((("f1", "tab:blue"), ("spurious", "tab:orange"), ("oracle_absent", "tab:green"), ("morph", "tab:red"))):
            a.bar(xx + (j - 1.5) * 0.2, [r[k] for r in rs], 0.2, color=c, label=k)
        a.set_xticks(xx, [f"{g}\nmax h={max(ER.SCHEDULES[g]):.2f}" for g in grp])
        a.grid(alpha=0.3, axis="y")
        a.legend(fontsize=7)
        a.set_title(f"NFE {len(ER.SCHEDULES[grp[0]])}", fontsize=10)
    fig.suptitle("X4-0 Fig. 8 - fixed-NFE interval stress. LIMITATION: max tested h = 0.70 / 0.50, whereas 1-NFE uses h = 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(F / "fig8_interval_stress.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    j = [c for c in cal if c["kind"] == "jitter_sd_ms"]
    ax.plot([c["value"] for c in j], [c["f1"] for c in j], "o-", color="tab:purple")
    ax.set_xlabel("GT timing jitter SD (ms)")
    ax.set_ylabel("F1 of jittered GT vs GT (50 ms matcher)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    ax.set_title("X4-0 Fig. 9 - event-matching tolerance calibration\ntiming-only degradation; NOT an upper bound on achievable F1", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "fig9_matching_calibration.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
