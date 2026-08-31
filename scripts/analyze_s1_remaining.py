"""S1.2-S1.6 — the remaining preregistered metric-validity items.

Frozen protocol: docs/S1_METRIC_VALIDITY_PREREGISTRATION.md (b749339), Amendment 1 (dc75079).
G1 passed (f27234f, T-B 0.9993 at 50 ms), which unblocks these items.

NO TRAINING. Frozen-checkpoint forward inference only. No checkpoint written. WildPPG test subjects are
never loaded (fail-loud firewall). No new threshold, metric, tolerance or subset is introduced.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.evaluation import alignment_diagnostics as AD  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation import s1_audit as S1  # noqa: E402
from ppg2ecg.evaluation import stamping as ST  # noqa: E402
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.flow.samplers import heun_sample, nfe_of  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.models.regressor import REGRESSOR_MODELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s1_metric_validity"
FIG = OUT / "figures"
FS, T_LEN = 128, 1024
VAL = ("an0", "k2s")
NFE_SALT, NFE_TAKE = "x4-event-nfe-v2", 1024
SRC_SALT, SRC_TAKE = "x4-event-source-v2", 256
IMF_NFES = (1, 4, 8, 50)
COND_NFES = (1, 8)
SOURCE_PAIR = {0: 1, 2: 3}
IMF_CKPT = "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
OT_CKPT = "outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt"
#: Selection rule fixed BEFORE any S1.2-S1.6 number: the MSE arm must be capacity-matched to the
#: generative arms (the same unmodified PENGUIN backbone, 4,568,707 params) and trained on the same A4
#: split with the same window-normalised ECG target. a5c is reduced-capacity and a9 uses a different
#: target representation; neither is comparable and neither is used.
MSE_CKPT = "outputs/a6c_fullbackbone_mse_wildppg_seed42/checkpoint_best.pt"
BATCH, WORKERS = 64, 12


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_val(idx_by_subject):
    ER.assert_no_test_subjects(list(idx_by_subject))
    X, Y, S, W = [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        i = idx_by_subject[s]
        X.append(d["x"][i].astype(np.float32))
        Y.append(d["y"][i].astype(np.float32))
        S.append(np.full(len(i), s))
        W.append(np.asarray(i))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(W)


def subsets(salt, take):
    ER.assert_no_test_subjects(VAL)
    out = {}
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        out[s] = ER.select_subset(salt, s, len(d["x"]), take)
    return out


def source_bank(seed, n):
    return torch.randn(n, 1, T_LEN, generator=torch.Generator().manual_seed(int(seed)))


def load_imf(dev):
    ck = torch.load(ROOT / IMF_CKPT, map_location="cpu", weights_only=False)
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    return net


def load_ot(dev):
    ck = torch.load(ROOT / OT_CKPT, map_location="cpu", weights_only=False)
    m = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval()
    m.load_state_dict(ck["state_dict"])
    m.requires_grad_(False)
    return m


def load_mse(dev):
    """Loaded exactly as the frozen X3-G0 precedent does (analyze_x3_g0_coupling_geometry.py:98-100)."""
    ck = torch.load(ROOT / MSE_CKPT, map_location="cpu", weights_only=False)
    cls, _ = REGRESSOR_MODELS[ck.get("model_key", "state_token")]
    m = cls(**ck["model_cfg"]).to(dev).eval()
    m.load_state_dict(ck["state_dict"])
    m.requires_grad_(False)
    return m, ck


@torch.no_grad()
def gen_imf(net, ppg, e, nfe, dev):
    out = []
    for i in range(0, len(ppg), BATCH):
        p = torch.from_numpy(ppg[i:i + BATCH]).to(dev).unsqueeze(1)
        z, got = ER.sample_meanflow_schedule(net, p, e[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        assert got == nfe, (got, nfe)
        out.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def gen_ot50(model, ppg, e, dev):
    out = []
    for i in range(0, len(ppg), BATCH):
        p = torch.from_numpy(ppg[i:i + BATCH]).to(dev).unsqueeze(1)
        z, nfe = heun_sample(lambda x, t: model.forward_step(x, p, t), e[i:i + BATCH].to(dev), 25)
        assert nfe == nfe_of("heun", 25) == 50, nfe
        out.append(z.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def gen_mse(model, ppg, dev):
    out = []
    for i in range(0, len(ppg), BATCH):
        p = torch.from_numpy(ppg[i:i + BATCH]).to(dev).unsqueeze(1)
        out.append(model(p).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def _peaks(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), FS)


def _peaks_b(sig):
    return R.detect_rpeaks(np.asarray(sig, dtype=np.float64), FS, S1.DETECTOR_B)


def _ppg_peaks(sig):
    return S1.dsp_ppg_peaks(sig, FS)


def pmap(fn, items, workers=WORKERS, chunk=16):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=chunk))


# ---------------------------------------------------------------------------------- scoring workers
def _score_chunk(args):
    """The frozen X0/X4-0 metric set, unmodified, for one chunk of (pred, gt, gt_peaks)."""
    pred, gt, gt_pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    rows = []
    for i in range(len(pred)):
        ev = AD.event_timing(gt[i], pred[i], FS, tol_ms=S1.MATCH_TOL_MS)
        bl = AD.beat_level_analysis(pred[i], gt[i], gt_pk[i], FS, int(round(AD.LOCAL_MAX_SHIFT_MS / 1000 * FS)))
        n_ref = max(ev["n_ref"], 1)
        absent = ((np.nan_to_num(bl["oracle_corr"], nan=-1) < 0.5) | (bl["oracle_p2p_ratio"] < 0.2)) if bl["n_beats"] else np.array([])
        m, _, _ = R.match_rpeaks(ev["ref_rpeaks"], ev["pred_rpeaks"], FS, S1.MATCH_TOL_MS)
        rows.append({
            "f1": float(rm["rpeak_f1"][i]), "precision": float(rm["rpeak_precision"][i]),
            "recall": float(rm["rpeak_recall"][i]), "beats_ratio": ev["n_pred"] / n_ref,
            "missing": ev["n_missing"] / n_ref, "spurious": ev["n_spurious"] / n_ref,
            "n_gt_beats": int(ev["n_ref"]), "n_pred_beats": int(ev["n_pred"]), "n_matched": int(len(m)),
            "morph": float(rm["morph_corr"][i]), "rr_mae_ms": float(rm["rr_mae_ms"][i]),
            "qrs_width_err_ms": float(rm["qrs_width_err_ms"][i]),
            "n_valid_gt_beats": int(bl["n_beats"]) if bl["n_beats"] else 0,
            "same_coord_corr": float(np.nanmean(bl["raw_corr"])) if bl["n_beats"] else np.nan,
            "oracle_corr": float(np.nanmean(bl["oracle_corr"])) if bl["n_beats"] else np.nan,
            "oracle_absent": float(absent.mean()) if absent.size else np.nan,
            "oracle_qrs_e": float(np.nanmedian(bl["oracle_qrs_energy_ratio"])) if bl["n_beats"] else np.nan,
            "hf_ratio": float(hf_energy_ratio(pred[i][None])[0]),
        })
    return rows


def _classify_chunk(args):
    """S1.4a: threshold is the 5th percentile of GT matched-beat amp_rel, computed on this arm's matched set."""
    pred, gt, gt_pk, thr = args
    tot = {"displaced": 0, "weak": 0, "absent": 0, "contested": 0, "n_unmatched": 0}
    per = []
    for i in range(len(pred)):
        pp = R.detect_rpeaks(np.asarray(pred[i], dtype=np.float64), FS)
        m, _, _ = R.match_rpeaks(gt_pk[i], pp, FS, S1.MATCH_TOL_MS)
        c = S1.classify_unmatched(gt_pk[i], pp, pred[i], m, thr, FS)
        for k in tot:
            tot[k] += c[k]
        per.append(c)
    return tot, per


def _gt_amp_rel_chunk(args):
    """amp_rel of the GT signal at MATCHED beat positions, per the preregistered threshold definition."""
    pred, gt, gt_pk = args
    vals = []
    for i in range(len(pred)):
        pp = R.detect_rpeaks(np.asarray(pred[i], dtype=np.float64), FS)
        m, _, _ = R.match_rpeaks(gt_pk[i], pp, FS, S1.MATCH_TOL_MS)
        for j, _ in m:
            vals.append(S1.amp_rel(gt[i], int(gt_pk[i][j]), FS))
    return vals


def _null_chunk(args):
    pred, gt, gt_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    same, orac, nb = [], [], 0
    for i in range(len(pred)):
        r = S1.oracle_null_gain(pred[i], gt[i], gt_pk[i], rng, FS, S1.NULL_DRAWS)
        if r["n_pairs"]:
            same.append(r["null_same"])
            orac.append(r["null_oracle"])
            nb += r["n_pairs"]
    return same, orac, nb


def _chance_chunk(args):
    """S1.4c: count-matched random-phase and circular-shift floors for one chunk."""
    gt_pk, pred_pk, n_time = args
    rng = np.random.default_rng(S1.NULL_SEED)
    rp, cs = [], []
    for i in range(len(gt_pk)):
        n = len(pred_pk[i])
        a = [R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(R.match_rpeaks(gt_pk[i], S1.chance_random_phase(n, n_time, rng), FS, S1.MATCH_TOL_MS)))[2] for _ in range(S1.NULL_DRAWS)]
        b = [R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(R.match_rpeaks(gt_pk[i], S1.chance_circular_shift(pred_pk[i], n_time, rng), FS, S1.MATCH_TOL_MS)))[2] for _ in range(S1.NULL_DRAWS)]
        rp.append(float(np.mean(a)))
        cs.append(float(np.mean(b)))
    return rp, cs


def chunks(n, k=64):
    return [(i, min(n, i + k)) for i in range(0, n, k)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prov] HEAD {head} | device {dev}", flush=True)

    # ------------------------------------------------------------------ population
    pop = subsets(NFE_SALT, NFE_TAKE)
    X, Y, SUB, WIDX = load_val(pop)
    print(f"[P] {len(X)} windows ({', '.join(f'{s}:{pop[s].size}' for s in VAL)})", flush=True)
    Yd = Y.astype(np.float64)
    gt_pk = pmap(_peaks, list(Yd))
    n_gt = int(sum(len(p) for p in gt_pk))
    print(f"[P] GT beats {n_gt}", flush=True)
    ch = chunks(len(X))

    # ------------------------------------------------------------------ frozen arms, seed 0
    imf, ot, (mse, mse_ck) = load_imf(dev), load_ot(dev), load_mse(dev)
    e0 = source_bank(0, len(X))
    arms: dict[str, np.ndarray] = {}
    for n in IMF_NFES:
        arms[f"iMF-{n}"] = gen_imf(imf, X, e0, n, dev).astype(np.float64)
        print(f"[A] generated iMF-{n}", flush=True)
    arms["OT-CFM-50"] = gen_ot50(ot, X, e0, dev).astype(np.float64)
    arms["MSE"] = gen_mse(mse, X, dev).astype(np.float64)
    print(f"[A] generated OT-CFM-50 and MSE ({MSE_CKPT})", flush=True)
    del imf, ot, mse
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ S1.4c chance floors (needed first)
    print("[S1.4c] chance floors", flush=True)
    chance = {}
    for name, P in arms.items():
        pk = pmap(_peaks, list(P))
        rp, cs = [], []
        for a, b in ch:
            r, c = _chance_chunk((gt_pk[a:b], pk[a:b], T_LEN))
            rp += r
            cs += c
        chance[name] = {"random_phase": np.asarray(rp), "circular_shift": np.asarray(cs), "peaks": pk}
        print(f"[S1.4c] {name}: RP {S1.macro(rp, SUB):.4f}  CS {S1.macro(cs, SUB):.4f}", flush=True)

    # ------------------------------------------------------------------ S1.2 DSP floor
    print("[S1.2] PPG peak detection (library defaults, no tuning)", flush=True)
    ppg_pk = pmap(_ppg_peaks, list(X.astype(np.float64)))

    tr_subs = [s for s in ST.TEMPLATE_SUBJECTS]        # training subjects only, as preregistered
    tr_delays = []
    for s in tr_subs:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        idx = ER.select_subset(ST.TEMPLATE_SALT, s, len(d["x"]), ST.TEMPLATE_N_TAKE, exclude=())
        yy = [d["y"][int(i)].astype(np.float64) for i in idx]
        xx = [d["x"][int(i)].astype(np.float64) for i in idx]
        gp, pp = pmap(_peaks, yy), pmap(_ppg_peaks, xx)
        for a, b in zip(gp, pp):
            tr_delays.append(S1.pat_delays_ms(a, b, FS))
    tr_delays = np.concatenate([d for d in tr_delays if d.size]) if tr_delays else np.zeros(0)
    delta_train = float(np.median(tr_delays))
    print(f"[S1.2] train-only delay: median {delta_train:.1f} ms over {tr_delays.size} pairs "
          f"({len(tr_subs)} train subjects)", flush=True)

    val_delta = {}
    for s in VAL:
        m = SUB == s
        dd = [S1.pat_delays_ms(gt_pk[i], ppg_pk[i], FS) for i in np.flatnonzero(m)]
        dd = np.concatenate([d for d in dd if d.size]) if dd else np.zeros(0)
        val_delta[s] = float(np.median(dd)) if dd.size else np.nan
    loso = {VAL[0]: val_delta[VAL[1]], VAL[1]: val_delta[VAL[0]]}
    print(f"[S1.2] per-val-subject medians {val_delta} -> LOSO {loso}", flush=True)

    template_a = np.load(OUT / "template_A.npy")
    r_full = ST.template_geometry(FS)["r_index_full"]
    dsp_rows = []
    for regime in ("delta_0", "delta_train_global", "delta_loso"):
        sigs = []
        for i in range(len(X)):
            d = 0.0 if regime == "delta_0" else (delta_train if regime == "delta_train_global" else loso[SUB[i]])
            p = S1.shift_peaks(ppg_pk[i], d, FS, T_LEN)
            sigs.append(ST.stamp(template_a, p, T_LEN, r_full))
        det = pmap(_peaks, sigs)
        for tol in (50.0, 75.0, 150.0):
            f1, pr, rc, br = [], [], [], []
            for i in range(len(X)):
                m, fp, fn = R.match_rpeaks(gt_pk[i], det[i], FS, tol)
                a, b, c = R.prf(len(m), fp, fn)
                f1.append(c); pr.append(a); rc.append(b)
                br.append(len(det[i]) / max(len(gt_pk[i]), 1))
            bs = S1.subject_bootstrap(f1, SUB)
            row = {"regime": regime, "delta_ms": 0.0 if regime == "delta_0" else (delta_train if regime == "delta_train_global" else np.nan),
                   "tol_ms": tol, "f1_macro": S1.macro(f1, SUB), "f1_lo": bs["lo"], "f1_hi": bs["hi"],
                   "precision_macro": S1.macro(pr, SUB), "recall_macro": S1.macro(rc, SUB),
                   "beats_ratio_macro": S1.macro(br, SUB),
                   "chance_random_phase": np.nan, "f1_excess_chance": np.nan}
            if tol == 50.0:
                cf = []
                rng = np.random.default_rng(S1.NULL_SEED)
                for i in range(len(X)):
                    cf.append(float(np.mean([R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(R.match_rpeaks(gt_pk[i], S1.chance_random_phase(len(det[i]), T_LEN, rng), FS, 50.0)))[2] for _ in range(S1.NULL_DRAWS)])))
                row["chance_random_phase"] = S1.macro(cf, SUB)
                row["f1_excess_chance"] = row["f1_macro"] - row["chance_random_phase"]
            for s in VAL:
                row[f"f1__{s}"] = S1.macro(np.asarray(f1)[SUB == s], SUB[SUB == s])
            dsp_rows.append(row)
            print(f"[S1.2] {regime:20s} @{tol:>5.0f} ms: F1 {row['f1_macro']:.4f} "
                  f"[{row['f1_lo']:.4f},{row['f1_hi']:.4f}] P {row['precision_macro']:.4f} "
                  f"R {row['recall_macro']:.4f} beats {row['beats_ratio_macro']:.4f}", flush=True)
    write_csv(OUT / "s1_2_dsp_floor.csv", dsp_rows)

    # ------------------------------------------------------------------ S1.3 joint fidelity census
    print("[S1.3] joint-fidelity census", flush=True)
    scored, census = {}, []
    for name, P in arms.items():
        rows = []
        for r in pmap(_score_chunk, [(P[a:b], Yd[a:b], gt_pk[a:b]) for a, b in ch], chunk=1):
            rows += r
        scored[name] = rows
        g = lambda k: np.asarray([x[k] for x in rows], float)  # noqa: E731
        cov = float(np.sum(g("n_matched")) / max(np.sum(g("n_valid_gt_beats")), 1))
        bs = S1.subject_bootstrap(g("f1"), SUB)
        row = {"arm": name, "f1": S1.macro(g("f1"), SUB), "f1_lo": bs["lo"], "f1_hi": bs["hi"],
               "chance_random_phase": S1.macro(chance[name]["random_phase"], SUB),
               "chance_circular_shift": S1.macro(chance[name]["circular_shift"], SUB),
               "precision": S1.macro(g("precision"), SUB), "recall": S1.macro(g("recall"), SUB),
               "missing": S1.macro(g("missing"), SUB), "spurious": S1.macro(g("spurious"), SUB),
               "matched_coverage": cov,
               "zero_contrib_window_frac": float(np.mean(~np.isfinite(g("morph")))),
               "morph_matched": S1.macro(g("morph"), SUB), "rr_mae_ms_matched": S1.macro(g("rr_mae_ms"), SUB),
               "same_coord_corr": S1.macro(g("same_coord_corr"), SUB),
               "oracle_corr": S1.macro(g("oracle_corr"), SUB),
               "oracle_absent": S1.macro(g("oracle_absent"), SUB),
               "oracle_qrs_energy_median": float(np.nanmedian(g("oracle_qrs_e"))),
               "qrs_width_err_ms": S1.macro(g("qrs_width_err_ms"), SUB),
               "beats_ratio": S1.macro(g("beats_ratio"), SUB), "hf_ratio": S1.macro(g("hf_ratio"), SUB)}
        row["f1_excess_chance"] = row["f1"] - row["chance_random_phase"]
        census.append(row)
        print(f"[S1.3] {name:11s} F1 {row['f1']:.4f} (chance {row['chance_random_phase']:.4f}, "
              f"excess {row['f1_excess_chance']:+.4f}) cov {cov:.4f} morph {row['morph_matched']:.4f} "
              f"same {row['same_coord_corr']:.4f} oracle {row['oracle_corr']:.4f} "
              f"qrsE {row['oracle_qrs_energy_median']:.4f} width {row['qrs_width_err_ms']:.2f} "
              f"beats {row['beats_ratio']:.4f} hf {row['hf_ratio']:.4f}", flush=True)

    # preregistered gate RULE, applied exactly as frozen (reported, never used to select a method)
    Rref = next(r for r in census if r["arm"] == "OT-CFM-50")
    for r in census:
        r["gate_morph"] = bool(r["morph_matched"] >= 0.80 * Rref["morph_matched"])
        r["gate_coverage"] = bool(r["matched_coverage"] >= 0.80 * Rref["matched_coverage"])
        r["gate_qrs_energy"] = bool(r["oracle_qrs_energy_median"] >= 0.60 * Rref["oracle_qrs_energy_median"])
        r["gate_qrs_width"] = bool(r["qrs_width_err_ms"] <= 1.50 * Rref["qrs_width_err_ms"])
        r["gate_beats_ratio"] = bool(0.90 <= r["beats_ratio"] <= 1.10)
        r["gate_all"] = bool(r["gate_morph"] and r["gate_coverage"] and r["gate_qrs_energy"]
                             and r["gate_qrs_width"] and r["gate_beats_ratio"])
    write_csv(OUT / "s1_3_joint_fidelity.csv", census)

    # ------------------------------------------------------------------ S1.4a/b
    print("[S1.4a] displacement-aware classification", flush=True)
    cls_rows = []
    for name, P in arms.items():
        gv = []
        for v in pmap(_gt_amp_rel_chunk, [(P[a:b], Yd[a:b], gt_pk[a:b]) for a, b in ch], chunk=1):
            gv += v
        thr = float(np.percentile(gv, 5)) if gv else np.nan
        tot = {"displaced": 0, "weak": 0, "absent": 0, "contested": 0, "n_unmatched": 0}
        per_sub = {s: {"displaced": 0, "weak": 0, "absent": 0, "n_unmatched": 0} for s in VAL}
        flags = []
        for (a, b), (t, per) in zip(ch, pmap(_classify_chunk, [(P[a:b], Yd[a:b], gt_pk[a:b], thr) for a, b in ch], chunk=1)):
            for k in tot:
                tot[k] += t[k]
            for off, c in enumerate(per):
                s = SUB[a + off]
                for k in per_sub[s]:
                    per_sub[s][k] += c[k]
                u = max(c["n_unmatched"], 1)
                flags.append((c["displaced"] / u, c["weak"] / u, c["absent"] / u) if c["n_unmatched"] else (np.nan,) * 3)
        u = max(tot["n_unmatched"], 1)
        fl = np.asarray(flags, float)
        bd = S1.subject_bootstrap(fl[:, 0], SUB)
        bw = S1.subject_bootstrap(fl[:, 1], SUB)
        ba = S1.subject_bootstrap(fl[:, 2], SUB)
        row = {"arm": name, "amp_rel_threshold_p5": thr, "n_unmatched": tot["n_unmatched"],
               "n_displaced": tot["displaced"], "n_weak": tot["weak"], "n_absent": tot["absent"],
               "contested": tot["contested"],
               "displaced_frac": tot["displaced"] / u, "weak_frac": tot["weak"] / u, "absent_frac": tot["absent"] / u,
               "displaced_lo": bd["lo"], "displaced_hi": bd["hi"], "weak_lo": bw["lo"], "weak_hi": bw["hi"],
               "absent_lo": ba["lo"], "absent_hi": ba["hi"]}
        for s in VAL:
            uu = max(per_sub[s]["n_unmatched"], 1)
            row[f"displaced__{s}"] = per_sub[s]["displaced"] / uu
            row[f"weak__{s}"] = per_sub[s]["weak"] / uu
            row[f"absent__{s}"] = per_sub[s]["absent"] / uu
        cls_rows.append(row)
        print(f"[S1.4a] {name:11s} thr {thr:.3f} | unmatched {tot['n_unmatched']:6d} | "
              f"DISPLACED {row['displaced_frac']:.4f} WEAK {row['weak_frac']:.4f} "
              f"ABSENT {row['absent_frac']:.4f} | contested {tot['contested']}", flush=True)
    write_csv(OUT / "s1_4_event_failure_classes.csv", cls_rows)

    print("[S1.4b] oracle null calibration", flush=True)
    null_rows = []
    for name, P in arms.items():
        same, orac, npairs = [], [], 0
        for s_, o_, n_ in pmap(_null_chunk, [(P[a:b], Yd[a:b], gt_pk[a:b]) for a, b in ch], chunk=1):
            same += s_
            orac += o_
            npairs += n_
        cen = next(r for r in census if r["arm"] == name)
        true_gain = cen["oracle_corr"] - cen["same_coord_corr"]
        null_gain = float(np.nanmean(orac)) - float(np.nanmean(same))
        row = {"arm": name, "same_coord_corr": cen["same_coord_corr"], "oracle_corr": cen["oracle_corr"],
               "true_oracle_gain": true_gain, "null_same_corr": float(np.nanmean(same)),
               "null_oracle_corr": float(np.nanmean(orac)), "null_oracle_gain": null_gain,
               "oracle_excess_over_null": true_gain - null_gain, "n_null_pairs": npairs,
               "n_draws": S1.NULL_DRAWS, "rng_seed": S1.NULL_SEED}
        null_rows.append(row)
        print(f"[S1.4b] {name:11s} same {row['same_coord_corr']:.4f} oracle {row['oracle_corr']:.4f} "
              f"| true gain {true_gain:+.4f} null gain {null_gain:+.4f} "
              f"excess {row['oracle_excess_over_null']:+.4f}", flush=True)
    write_csv(OUT / "s1_4_oracle_null.csv", null_rows)
    write_csv(OUT / "s1_4_chance_floor.csv", [
        {"arm": n, "random_phase_f1": S1.macro(chance[n]["random_phase"], SUB),
         "circular_shift_f1": S1.macro(chance[n]["circular_shift"], SUB),
         "observed_f1": next(r for r in census if r["arm"] == n)["f1"],
         "excess_random_phase": next(r for r in census if r["arm"] == n)["f1"] - S1.macro(chance[n]["random_phase"], SUB),
         "excess_circular_shift": next(r for r in census if r["arm"] == n)["f1"] - S1.macro(chance[n]["circular_shift"], SUB),
         "n_draws": S1.NULL_DRAWS, "rng_seed": S1.NULL_SEED} for n in arms])
    del arms
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ S1.5 C2 re-analysis
    print("[S1.5] C2 perturbation re-analysis", flush=True)
    spop = subsets(SRC_SALT, SRC_TAKE)
    Xs, Ys, SUBs, _ = load_val(spop)
    imf = load_imf(dev)
    rng = np.random.default_rng(1)
    while True:
        perm = rng.permutation(len(Xs))
        if not np.any(perm == np.arange(len(Xs))):
            break
    c2_rows = []
    for n in COND_NFES:
        for base, alt in SOURCE_PAIR.items():
            A = gen_imf(imf, Xs, source_bank(base, len(Xs)), n, dev).astype(np.float64)
            B = gen_imf(imf, Xs, source_bank(alt, len(Xs)), n, dev).astype(np.float64)
            C = gen_imf(imf, Xs[perm], source_bank(base, len(Xs)), n, dev).astype(np.float64)
            pA, pB, pC = pmap(_peaks, list(A)), pmap(_peaks, list(B)), pmap(_peaks, list(C))
            for lab, Q, qk in (("source", B, pB), ("ppg_shuffle", C, pC)):
                f1, pcc, dt_all, dt_nz, nz, nden = [], [], [], [], 0, 0
                unc = []
                for i in range(len(Xs)):
                    pr, _, _ = R.match_rpeaks(pA[i], qk[i], FS, S1.MATCH_TOL_MS) if len(pA[i]) and len(qk[i]) else ([], 0, 0)
                    d = np.asarray([abs(pA[i][a] - qk[i][b]) / FS * 1000.0 for a, b in pr], float)
                    nden += d.size
                    nz += int(np.sum(d == 0.0))
                    dt_all.append(float(d.mean()) if d.size else np.nan)
                    dz = d[d != 0.0]
                    dt_nz.append(float(dz.mean()) if dz.size else np.nan)
                    # uncensored matcher: nearest partner with no tolerance cap
                    if len(pA[i]) and len(qk[i]):
                        du = np.min(np.abs(pA[i][:, None] - np.asarray(qk[i])[None, :]), axis=1) / FS * 1000.0
                        unc.append(float(du.mean()))
                    else:
                        unc.append(np.nan)
                    f1.append(ER.peak_train_agreement(pA[i], qk[i])["f1"])
                    pcc.append(float(np.corrcoef(A[i], Q[i])[0, 1]) if A[i].std() > 1e-8 and Q[i].std() > 1e-8 else np.nan)
                # permutation chance floor: pair window i's A with a different window's Q
                pr_rng = np.random.default_rng(S1.NULL_SEED)
                pf = []
                for _ in range(S1.NULL_DRAWS):
                    q = pr_rng.permutation(len(Xs))
                    pf.append(float(np.nanmean([ER.peak_train_agreement(pA[i], qk[q[i]])["f1"] for i in range(len(Xs))])))
                c2_rows.append({
                    "nfe": n, "base_seed": base, "alt_seed": alt, "perturbation": lab,
                    "event_f1": S1.macro(f1, SUBs), "event_f1_permutation_floor": float(np.mean(pf)),
                    "event_f1_excess": S1.macro(f1, SUBs) - float(np.mean(pf)),
                    "wave_pcc": S1.macro(pcc, SUBs),
                    "timing_ms_all_matched": S1.macro(dt_all, SUBs),
                    "timing_ms_excl_zero": S1.macro(dt_nz, SUBs),
                    "timing_ms_uncensored": S1.macro(unc, SUBs),
                    "n_matched_pairs": nden, "n_exact_zero_pairs": nz,
                    "frac_exact_zero": nz / max(nden, 1), "n_pairs_after_exclusion": nden - nz,
                    "n_draws": S1.NULL_DRAWS, "rng_seed": S1.NULL_SEED})
                print(f"[S1.5] NFE {n} {base}->{alt} {lab:12s}: F1 {c2_rows[-1]['event_f1']:.4f} "
                      f"(floor {c2_rows[-1]['event_f1_permutation_floor']:.4f}) PCC {c2_rows[-1]['wave_pcc']:.4f} "
                      f"| timing all {c2_rows[-1]['timing_ms_all_matched']:.2f} excl-0 {c2_rows[-1]['timing_ms_excl_zero']:.2f} "
                      f"uncens {c2_rows[-1]['timing_ms_uncensored']:.1f} | zeros {nz}/{nden} "
                      f"({c2_rows[-1]['frac_exact_zero']:.3f})", flush=True)
            del A, B, C
    write_csv(OUT / "s1_5_c2_reanalysis.csv", c2_rows)
    del imf
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------ S1.6 GT reliability
    print("[S1.6] development GT reliability (two detectors)", flush=True)
    pk_b = pmap(_peaks_b, list(Yd))
    gt_rows = []
    agr = [S1.detector_agreement(Yd[i], FS) for i in range(len(Yd))]
    f1 = [a["f1"] for a in agr]
    bs = S1.subject_bootstrap(f1, SUB)
    for s in list(VAL) + ["MACRO"]:
        m = np.ones(len(Yd), bool) if s == "MACRO" else (SUB == s)
        sel = [agr[i] for i in np.flatnonzero(m)]
        gt_rows.append({
            "subject": s, "n_windows": int(m.sum()),
            "detector_a": S1.DETECTOR_A, "detector_b": S1.DETECTOR_B,
            "f1": float(np.mean([x["f1"] for x in sel])) if s != "MACRO" else S1.macro(f1, SUB),
            "f1_lo": bs["lo"] if s == "MACRO" else np.nan, "f1_hi": bs["hi"] if s == "MACRO" else np.nan,
            "precision_b_vs_a": float(np.mean([x["precision_b_vs_a"] for x in sel])),
            "recall_b_vs_a": float(np.mean([x["recall_b_vs_a"] for x in sel])),
            "n_beats_a": int(sum(x["n_a"] for x in sel)), "n_beats_b": int(sum(x["n_b"] for x in sel)),
            "mean_beat_count_diff": float(np.mean([x["beat_count_diff"] for x in sel])),
            "frac_windows_f1_lt_0.9": float(np.mean([x["f1"] < 0.9 for x in sel])),
            "frac_rr_out_a": float(np.mean([S1.rr_plausibility(gt_pk[i])["frac_rr_out"] for i in np.flatnonzero(m) if len(gt_pk[i]) >= 2])),
            "frac_rr_out_b": float(np.mean([S1.rr_plausibility(pk_b[i])["frac_rr_out"] for i in np.flatnonzero(m) if len(pk_b[i]) >= 2])),
            "frac_count_implausible_a": float(np.mean([not S1.rr_plausibility(gt_pk[i])["count_plausible"] for i in np.flatnonzero(m)])),
            "frac_count_implausible_b": float(np.mean([not S1.rr_plausibility(pk_b[i])["count_plausible"] for i in np.flatnonzero(m)])),
        })
        print(f"[S1.6] {s:6s}: A/B F1 {gt_rows[-1]['f1']:.4f} | beats A {gt_rows[-1]['n_beats_a']} "
              f"B {gt_rows[-1]['n_beats_b']} | windows F1<0.9 {gt_rows[-1]['frac_windows_f1_lt_0.9']:.4f}", flush=True)
    write_csv(OUT / "s1_6_gt_reliability.csv", gt_rows)

    # ------------------------------------------------------------------ bookkeeping
    (OUT / "bootstrap_summary.json").write_text(json.dumps(
        {"n_boot": S1.BOOT_N, "seed": S1.BOOT_SEED, "null_draws": S1.NULL_DRAWS, "null_seed": S1.NULL_SEED,
         "dsp_f1_ci": {r["regime"] + f"@{r['tol_ms']:.0f}": [r["f1_lo"], r["f1_hi"]] for r in dsp_rows},
         "arm_f1_ci": {r["arm"]: [r["f1_lo"], r["f1_hi"]] for r in census},
         "gt_detector_f1_ci": [bs["lo"], bs["hi"]]}, indent=2))
    (OUT / "provenance_s1_remaining.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(),
        "prereg": "b749339", "amendment_1": "dc75079", "semantics_correction": "2e0c20b", "g1": "f27234f",
        "test_subjects_loaded": [], "training": False, "checkpoints_written": [],
        "checkpoints_read": [IMF_CKPT, OT_CKPT, MSE_CKPT],
        "mse_arm_selection_rule": "capacity-matched (unmodified PENGUIN backbone) on the A4 split with the "
                                  "same window-normalised ECG target; a5c (reduced capacity) and a9 (global-z "
                                  "target) are not comparable and were not used",
        "population": {s: int(pop[s].size) for s in VAL}, "n_gt_beats": n_gt,
        "source_population": {s: int(spop[s].size) for s in VAL},
        "items_run": ["S1.2", "S1.3", "S1.4a", "S1.4b", "S1.4c", "S1.5", "S1.6"],
        "seed0_only": True,
        "note": "seed-0 values; NOT comparable to the recorded 4-seed-pooled X4-0 table"}, indent=2))

    # ------------------------------------------------------------------ figures
    names = [r["arm"] for r in census]
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for r in census:
        ax.scatter(r["f1_excess_chance"], r["oracle_corr"], s=110,
                   c=[[0.85, 0.3, 0.1]] if r["arm"] == "MSE" else None, zorder=3)
        ax.annotate(f"{r['arm']}\nqrsE {r['oracle_qrs_energy_median']:.2f} | HF {r['hf_ratio']:.3f}\n"
                    f"width {r['qrs_width_err_ms']:.0f} ms | beats {r['beats_ratio']:.2f}",
                    (r["f1_excess_chance"], r["oracle_corr"]), fontsize=7,
                    textcoords="offset points", xytext=(8, -4))
    ax.set_xlabel("event fidelity — F1 excess over count-matched chance @50 ms")
    ax.set_ylabel("waveform fidelity — oracle_corr (all-GT-beat, detector-independent)")
    ax.set_title("S1.3 joint-fidelity frontier (2048 dev windows, seed 0)\n"
                 "structural metrics annotated separately, never collapsed into one scalar")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "joint_fidelity_frontier.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    b = np.arange(len(cls_rows))
    d = np.array([r["displaced_frac"] for r in cls_rows])
    w = np.array([r["weak_frac"] for r in cls_rows])
    a_ = np.array([r["absent_frac"] for r in cls_rows])
    ax.bar(b, d, label="DISPLACED (50-150 ms)")
    ax.bar(b, w, bottom=d, label="WEAK")
    ax.bar(b, a_, bottom=d + w, label="ABSENT")
    ax.set_xticks(b); ax.set_xticklabels([r["arm"] for r in cls_rows], rotation=15)
    ax.set_ylabel("fraction of UNMATCHED GT beats"); ax.legend(fontsize=8)
    ax.set_title("S1.4a event-failure decomposition (±150 ms search, preregistered threshold)")
    fig.tight_layout(); fig.savefig(FIG / "event_failure_decomposition.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(census))
    ax.plot(x, [r["morph_matched"] for r in census], "o-", label="matched-beat morph (~coverage-limited)")
    ax.plot(x, [r["oracle_corr"] for r in census], "s-", label="oracle_corr (all-GT-beat)")
    ax.plot(x, [r["same_coord_corr"] for r in census], "^-", label="same_coord_corr (all-GT-beat, unaligned)")
    ax.plot(x, [r["matched_coverage"] for r in census], "d--", label="matched coverage", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("S1.3 matched-beat vs all-GT-beat morphology")
    fig.tight_layout(); fig.savefig(FIG / "matched_vs_all_gt_morphology.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(null_rows))
    ax.bar(x - 0.2, [r["true_oracle_gain"] for r in null_rows], 0.4, label="true oracle gain")
    ax.bar(x + 0.2, [r["null_oracle_gain"] for r in null_rows], 0.4, label="null oracle gain (mismatched beat)")
    ax.set_xticks(x); ax.set_xticklabels([r["arm"] for r in null_rows], rotation=15)
    ax.set_ylabel("oracle_corr − same_coord_corr"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("S1.4b oracle-shift gain against its own null (20 draws, seed 20260901)")
    fig.tight_layout(); fig.savefig(FIG / "oracle_true_vs_null.png", dpi=115); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    sel = [r for r in gt_rows if r["subject"] != "MACRO"]
    x = np.arange(len(sel))
    ax.bar(x - 0.2, [r["f1"] for r in sel], 0.4, label=f"{S1.DETECTOR_A} vs {S1.DETECTOR_B} F1")
    ax.bar(x + 0.2, [r["frac_windows_f1_lt_0.9"] for r in sel], 0.4, label="fraction of windows F1 < 0.9")
    ax.set_xticks(x); ax.set_xticklabels([r["subject"] for r in sel]); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("S1.6 GT detector agreement on the reference ECG (development only)")
    fig.tight_layout(); fig.savefig(FIG / "gt_detector_agreement.png", dpi=115); plt.close(fig)

    print("\n[done] S1.2-S1.6 complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
