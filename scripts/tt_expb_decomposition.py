"""EXP-B — why width reduces functional error: fixed-K depth sweep with error decomposition (B1), fixed-S width sweep (B2),
and functional-error correlation across samples (B3). docs/TOP_TIER_COMPLETION_PREREGISTRATION.md, EXP-B. No training.

Samplers are DW1's (`dw1_depth_width.make_sampler`), noise seed k → sample k, VitalDB V1 test. Per-window functional
Y_{S,k} = HR of sample k at depth S (neurokit peaks, as the standard pipeline); reference T* = HR of the target ECG.

Run: .venv/bin/python scripts/tt_expb_decomposition.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dw1_depth_width as D  # noqa: E402
import dw2_mechanism_wave as DWV  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

OUT, RAW = ROOT / "artifacts/tt_expb_mechanism", ROOT / "outputs/tt_expb_raw"
CK = {("I", 42): "outputs/v1_vitaldb_armI_seed42", ("I", 1): "outputs/sr1_I_seed1", ("I", 2): "outputs/sr1_I_seed2",
      ("C", 42): "outputs/v1_vitaldb_armC_seed42", ("C", 1): "outputs/sr1_C_seed1", ("C", 2): "outputs/sr1_C_seed2",
      ("D", 42): "outputs/cd1_vitaldb_armD_seed42", ("D", 1): "outputs/dw2_D_seed1", ("D", 2): "outputs/dw2_D_seed2"}
ARMS, FS = ("I", "D", "C"), 128
D.BS = 128                                          # smaller generation batch: EXP-A shares the GPU
B1_K, B1_S = 16, (1, 2, 4, 8)
B2_S, B2_K, B2_SEEDS = {"D": 1, "I": 2, "C": 4}, (1, 2, 4, 8, 16, 32), (42, 1, 2)
CHOSEN_S = {"D": 2, "I": 2, "C": 4}                  # preregistered reading pairs: S = 1 → chosen S
N_SUB = 2000


def _peaks(sig):
    return np.asarray(R.detect_rpeaks(sig, FS), int)


def seeded_rows(arm, seed, S):
    """HR rows already computed with the identical sampler / noise seeds (DW1 seed 42, DW2-A seeds 1 / 2)."""
    cands = [ROOT / f"outputs/dw1_raw/hr_{arm}{S}.npy"] if seed == 42 else []
    cands += list((ROOT / "outputs/dw2_raw").glob(f"hr_{arm}_seed{seed}_K*_S{S}.npy"))
    rows = [np.load(p) for p in cands if p.exists()]
    return max(rows, key=len) if rows else None


class Store:
    """Per (arm, seed, S): HR rows [k, N] (all windows), F1 rows [k, N], waveform subset [k, n_sub, T] (float16)."""

    def __init__(self, ex, X, Y, ref_peaks, sub):
        self.ex, self.X, self.Y, self.ref_peaks, self.sub = ex, X, Y, ref_peaks, sub
        self.samplers = {}

    def sampler(self, arm, seed, dev):
        if (arm, seed) not in self.samplers:
            self.samplers.clear(); torch.cuda.empty_cache()
            D.CK[arm] = CK[(arm, seed)]
            self.samplers[(arm, seed)] = D.make_sampler(arm, dev)
        return self.samplers[(arm, seed)]

    def rows(self, arm, seed, S, K, dev, need_f1=False, need_wave=False):
        fh, ff, fw = (RAW / f"{n}_{arm}_seed{seed}_S{S}.npy" for n in ("hr", "f1", "wave"))
        H = np.load(fh) if fh.exists() else seeded_rows(arm, seed, S)
        H = np.zeros((0, len(self.X))) if H is None else H
        F1 = np.load(ff) if ff.exists() else np.zeros((0, len(self.X)))
        W = np.load(fw) if fw.exists() else np.zeros((0, len(self.sub), self.X.shape[1]), np.float16)
        while H.shape[0] < K or (need_f1 and F1.shape[0] < K) or (need_wave and W.shape[0] < K):
            k = min(H.shape[0], F1.shape[0] if need_f1 else 10 ** 9, W.shape[0] if need_wave else 10 ** 9)
            w = self.sampler(arm, seed, dev)(self.X, k, S)
            pk = list(self.ex.map(_peaks, list(w), chunksize=256))
            hr = np.array([R.hr_bpm(p, FS) for p in pk])
            f1 = np.array([R.prf(len(m), fp, fn)[2] for m, fp, fn in (R.match_rpeaks(r, h, FS, 50.0) for r, h in zip(self.ref_peaks, pk))])
            if H.shape[0] == k:
                H = np.vstack([H, hr[None]]); np.save(fh, H)
            if need_f1 and F1.shape[0] == k:
                F1 = np.vstack([F1, f1[None]]); np.save(ff, F1)
            if need_wave and W.shape[0] == k:
                W = np.concatenate([W, w[self.sub][None].astype(np.float16)]); np.save(fw, W)
        return H[:K], (F1[:K] if need_f1 else None), (W[:K] if need_wave else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); ex = ProcessPoolExecutor(8)
    X, Y, pid = VM.load("test")
    ref_peaks = list(ex.map(_peaks, list(Y), chunksize=256))
    ref = np.array([R.hr_bpm(p, FS) for p in ref_peaks])
    ok = np.isfinite(ref)                                     # Ω_HR: windows with a defined reference HR
    sub = np.unique(np.linspace(0, len(X) - 1, N_SUB).round().astype(int))
    ref_feat = PMX.default_feature_map(Y[sub], FS)
    fd_ref = Y[sub]
    ci = lambda v: V.cluster_ci(np.where(ok, v, np.nan), pid)  # noqa: E731
    store = Store(ex, X, Y, ref_peaks, sub)

    # ---------------- B1: K fixed = 16, S varied ----------------
    b1, per_window = [], {}
    for arm in ARMS:
        for S in B1_S:
            H, F1, W = store.rows(arm, 42, S, B1_K, dev, need_f1=True, need_wave=True)
            with np.errstate(all="ignore"):
                m = np.nanmedian(H, 0)
                C = np.abs(m - ref)
                I = np.nanmean(np.abs(H - ref), 0)
                Dmad = np.nanmedian(np.abs(H - m), 0)
                Dsd = np.nanstd(H, 0, ddof=1)
            G = I - C
            per_window[(arm, S)] = dict(C=C, I=I, G=G, Dmad=Dmad, Dsd=Dsd, F1=np.nanmean(F1, 0), H=H)
            Wf = W.astype(np.float64)
            wave_div = DWV.pairwise_mean_dist(Wf)
            F = np.stack([PMX.default_feature_map(Wf[k], FS) for k in range(B1_K)])
            feat_div = DWV.pairwise_mean_dist(F)
            fd = float(np.mean([PMX.kanflow_fd(Wf[k], fd_ref) for k in range(B1_K)]))
            row = dict(model=D.NAME[arm], S=S, K=B1_K, total_nfe=B1_K * S,
                       center_err=ci(C), individual_err=ci(I), median_gain=ci(G), disp_mad=ci(Dmad), disp_sd=ci(Dsd),
                       f1_per_sample=ci(np.nanmean(F1, 0)), nan_sample_frac=float(np.mean(~np.isfinite(H[:, ok]))),
                       wave_pairwise_rms=V.cluster_ci(wave_div, pid[sub]), feat_pairwise=V.cluster_ci(feat_div, pid[sub]), fd_per_sample_mean=fd)
            b1.append(row)
            print(f"[expB1] {D.NAME[arm]:26s} S={S}  C {row['center_err'][0]:.3f}  I {row['individual_err'][0]:.3f}  G {row['median_gain'][0]:.3f}  "
                  f"MAD {row['disp_mad'][0]:.3f} SD {row['disp_sd'][0]:.3f}  F1 {row['f1_per_sample'][0]:.4f}  waveRMS {row['wave_pairwise_rms'][0]:.3f}  FD {fd:.2f}", flush=True)
    b1_contrasts = {}
    for arm in ARMS:
        d = {}
        for S in B1_S[1:]:
            d[f"S{S}-S1"] = {q: ci(per_window[(arm, S)][q] - per_window[(arm, 1)][q]) for q in ("C", "I", "G", "Dmad", "Dsd", "F1")}
        b1_contrasts[D.NAME[arm]] = d
    crit = {"B1-1 PENGUIN C(S4)<C(S1)": b1_contrasts[D.NAME["C"]]["S4-S1"]["C"][2] < 0,
            "B1-2 iMF C(S2)<C(S1)": b1_contrasts[D.NAME["I"]]["S2-S1"]["C"][2] < 0,
            "B1-3 CD no improvement S1->S2": not (b1_contrasts[D.NAME["D"]]["S2-S1"]["C"][2] < 0)}
    reading = {D.NAME[arm]: {"pair": f"S1->S{CHOSEN_S[arm]}", **{f"delta_{q}": b1_contrasts[D.NAME[arm]][f"S{CHOSEN_S[arm]}-S1"][q] for q in ("C", "I", "G")}} for arm in ARMS}

    # ---------------- B3: error correlation across samples (from B1 matrices) ----------------
    b3 = []
    for arm in ARMS:
        for S in B1_S:
            E = per_window[(arm, S)]["H"] - ref                 # signed errors [16, N]
            full = ok & np.isfinite(E).all(0)
            Ef = E[:, full]
            pear = np.corrcoef(Ef)[np.triu_indices(B1_K, 1)]
            rk = np.argsort(np.argsort(Ef, axis=1), axis=1).astype(np.float64)
            spear = np.corrcoef(rk)[np.triu_indices(B1_K, 1)]
            msb = Ef.shape[0] * np.var(Ef.mean(0), ddof=1); msw = np.mean(np.var(Ef, axis=0, ddof=1))
            icc = (msb - msw) / (msb + (B1_K - 1) * msw)
            b3.append(dict(model=D.NAME[arm], S=S, n_windows=int(full.sum()), pearson_mean=float(pear.mean()), pearson_p5=float(np.percentile(pear, 5)),
                           pearson_p95=float(np.percentile(pear, 95)), spearman_mean=float(spear.mean()), icc1=float(icc),
                           median_gain=float(ci(per_window[(arm, S)]["G"])[0]), gain_ci_lower=float(ci(per_window[(arm, S)]["G"])[1])))
            print(f"[expB3] {D.NAME[arm]:26s} S={S}  pearson mean {pear.mean():.3f} [{np.percentile(pear, 5):.3f}, {np.percentile(pear, 95):.3f}]  spearman {spear.mean():.3f}  ICC {icc:.3f}  gain {b3[-1]['median_gain']:.3f}", flush=True)
    pos = [r for r in b3 if r["gain_ci_lower"] > 0]
    b3_1 = all(r["pearson_mean"] < 0.9 for r in pos)
    rho_g = spearmanr([r["pearson_mean"] for r in b3], [r["median_gain"] for r in b3])
    b3_2 = bool(rho_g.statistic < 0)
    crit.update({"B3-1 gain>0 conditions have mean pearson<0.9": bool(b3_1), "B3-2 spearman(rho_bar, gain)<0": b3_2})
    b3_summary = {"spearman_rho_bar_vs_gain": float(rho_g.statistic), "p": float(rho_g.pvalue), "n_conditions": len(b3), "n_gain_positive": len(pos)}

    # ---------------- B2: S fixed, K varied ----------------
    b2, b2_crit = [], {}
    for arm in ARMS:
        S = B2_S[arm]
        for seed in B2_SEEDS:
            H, _, _ = store.rows(arm, seed, S, 32, dev)
            R_part, R_nest = {}, {}
            for K in B2_K:
                with np.errstate(all="ignore"):
                    groups = [np.abs(np.nanmedian(H[g * K:(g + 1) * K], 0) - ref) for g in range(32 // K)]
                    R_part[K] = np.nanmean(np.stack(groups), 0)            # per-window, averaged over disjoint groups
                    R_nest[K] = np.abs(np.nanmedian(H[:K], 0) - ref)
            gains = {K: ci(R_part[K] - R_part[2 * K]) for K in B2_K[:-1]}
            gvals = [gains[K][0] for K in B2_K[:-1]]
            nonincreasing = all(gvals[i + 1] <= gvals[i] + 1e-9 for i in range(len(gvals) - 1))
            sat = next((K for K in B2_K[:-1] if abs(gains[K][0]) < 0.1), None)  # AB1 rule: doubling K changes HR by < 0.1 bpm
            b2_crit[f"{D.NAME[arm]}|seed{seed}"] = {"B2-1 marginal gains non-increasing (point estimates)": bool(nonincreasing), "saturation_K": sat}
            for K in B2_K:
                b2.append(dict(model=D.NAME[arm], seed=seed, S=S, K=K, total_nfe=K * S, R_partition=ci(R_part[K]), R_nested=ci(R_nest[K]),
                               marginal_gain_to_2K=gains.get(K, (np.nan, np.nan, np.nan)), R_minus_R32=ci(R_part[K] - R_part[32])))
                print(f"[expB2] {D.NAME[arm]:26s} seed {seed:2d} S={S} K={K:2d}  R_part {b2[-1]['R_partition'][0]:.3f}  R_nest {b2[-1]['R_nested'][0]:.3f}  gain→2K {gains[K][0] if K in gains else float('nan'):.3f}", flush=True)

    def dump(rows, name):
        flat = []
        for r in rows:
            fr = {}
            for k, v in r.items():
                if isinstance(v, (tuple, list)) and len(v) == 3:
                    fr[k], fr[k + "_lo"], fr[k + "_hi"] = v
                else:
                    fr[k] = v
            flat.append(fr)
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    dump(b1, "b1_fixed_K16_by_S.csv"); dump(b2, "b2_fixed_S_by_K.csv"); dump(b3, "b3_error_correlation.csv")
    (OUT / "result.json").write_text(json.dumps({"criteria": crit, "b1_contrasts_vs_S1": b1_contrasts, "b1_preregistered_reading": reading,
                                                  "b2_criteria": b2_crit, "b3_summary": b3_summary}, indent=1, default=float))
    print(json.dumps(crit, indent=1)); print(json.dumps(b2_crit, indent=1)); print(json.dumps(b3_summary, indent=1))


if __name__ == "__main__":
    main()
