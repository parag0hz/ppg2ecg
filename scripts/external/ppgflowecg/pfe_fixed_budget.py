"""PPGFlowECG external fixed-budget validation (docs/PPGFLOWECG_FIXED_BUDGET_PREREGISTRATION.md, frozen at 3dee94f).

Stages (each resumable; large arrays under outputs/ppgflowecg_external/, gitignored):
  gen      GPU. 16 draws x S in {5,10,15,20,25}, one full official sampling call each; batch 512 in data order,
           seed 1_000_003*d + b for every S (common noise across S). + per-cell latency. Env: core.
  ham      Official Hamilton HR (verbatim calculate_metric.ecg_bpm_array; generated filter=True, reference filter=False)
           and the official ppg_bpm_array baseline. Env: paper_eval (biosppy 2.2.3, neurokit2 0.1.7).
  proj     Project-standard HR, R-peak F1 @ 50 ms, RR-MAE (ppg2ecg.evaluation.rpeaks) + find_peaks PPG baseline. Env: .venv.
  wave     Per-sample MAE / RMSE / Pearson r, waveform pairwise RMS over the 16 draws, official calculate_fd (draw 0).
  analyze  Every preregistered estimate, contrast, bootstrap, mechanism statistic and the verdict.

Run (from the project root):
  C=outputs/pfe_env/core; PE=outputs/pfe_env/paper_eval; A=scripts/external/ppgflowecg
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$C:$A  .venv/bin/python $A/pfe_fixed_budget.py gen
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PE:$A .venv/bin/python $A/pfe_fixed_budget.py ham
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A     .venv/bin/python $A/pfe_fixed_budget.py proj | wave | analyze
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "outputs/ppgflowecg_external/data/vitaldb10_test.npz"
RAW = ROOT / "outputs/ppgflowecg_external"
GEN = RAW / "gen"
ART = ROOT / "artifacts/ppgflowecg_external"
FS, ND, BS = 128, 16, 512
SS = (5, 10, 15, 20, 25)
NB, BOOT_SEED = 5000, 20260924
CH = 256
PRIMARY = {"B10": ((2, 5), (1, 10)), "B15": ((3, 5), (1, 15)), "B20": ((4, 5), (1, 20))}
SECONDARY = {"B25": ((5, 5), (1, 25))}
CELLS = [(1, 5), (1, 10), (2, 5), (1, 15), (3, 5), (1, 20), (2, 10), (4, 5), (1, 25), (5, 5)] + [(16, S) for S in SS]


def seed_of(d, b):
    return 1_000_003 * d + b


# ================================================================================================ gen
def gen():
    import torch
    from pfe_model import PFE
    GEN.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA)
    ppg, ecg = torch.from_numpy(z["ppg"]), torch.from_numpy(z["ecg"])
    N = len(ppg); nb = -(-N // BS)
    man = {"batch_size": BS, "n_windows": N, "n_batches": nb, "draws": ND, "S": list(SS),
           "seed_formula": "torch.manual_seed(1_000_003 * d + b) before the sampling call of batch b, draw d; identical for every S",
           "batch_map": "batch b = windows [512 b, 512 b + 512) of outputs/ppgflowecg_external/data/vitaldb10_test.npz",
           "seeds": {str(d): [seed_of(d, b) for b in range(nb)] for d in range(ND)}}
    (ART / "seed_manifest.json").write_text(json.dumps(man, indent=1))
    pfe = PFE()
    nonfin = {}
    for S in SS:
        f = GEN / f"S{S}.npy"
        if f.exists():
            print(f"[gen] S={S} cached", flush=True); continue
        t0 = time.time()
        tmp = GEN / f"S{S}.tmp.npy"
        W = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(ND, N, 1280))
        bad = 0
        for d in range(ND):
            for b in range(nb):
                sl = slice(b * BS, min(N, (b + 1) * BS))
                x = pfe.sample(ppg[sl], ecg[sl], S, seed=seed_of(d, b))
                bad += int((~np.isfinite(x).all(1)).sum())
                W[d, sl] = x.astype(np.float16)
            print(f"[gen] S={S} draw {d} ({time.time() - t0:.0f}s)", flush=True)
        W.flush(); del W
        os.replace(tmp, f)
        nonfin[str(S)] = bad
        (GEN / f"S{S}.nonfinite.json").write_text(json.dumps({"nonfinite_samples": bad}))
    # per-cell wall-clock (GPU; content-independent, first window): K sequential batch-1 calls vs one call of batch K
    lat = {}
    p1, e1 = ppg[:1], ecg[:1]
    for K, S in CELLS:
        seq = pfe.timed(p1, e1, S, seed=5, reps=10) * K
        bat = pfe.timed(p1.repeat(K, 1), e1.repeat(K, 1), S, seed=5, reps=10)
        lat[f"K{K}_S{S}"] = {"sequential_batch1_ms": seq, "batched_K_ms": bat}
    (RAW / "latency_cells.json").write_text(json.dumps(lat, indent=1))
    print("[gen] done", flush=True)


# ================================================================================================ ham
_E = None


def _ham(args):
    """Official Hamilton HR per row; raw values kept (-1 = official 'no HR'; NaN = NaN output or extractor exception)."""
    global _E
    x, filt = args
    if _E is None:
        warnings.filterwarnings("ignore")
        import official as O
        _E = O.load_eval()
    v = np.full(len(x), np.nan)
    for i, row in enumerate(np.asarray(x, np.float64)):
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v[i] = float(_E["ecg_bpm_array"](row[None], FS, 10, filter=filt)[0])
        except Exception:  # noqa: BLE001  (official extractor is unguarded)
            v[i] = np.nan
    return v


def _ppg_off(x):
    global _E
    if _E is None:
        warnings.filterwarnings("ignore")
        import official as O
        _E = O.load_eval()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_E["ppg_bpm_array"](np.asarray(x, np.float64), FS, 10), float)


def chunks(n):
    return [(a, min(a + CH, n)) for a in range(0, n, CH)]


def ham():
    f = RAW / "hr_hamilton.npz"
    z = np.load(DATA); N = len(z["ecg"])
    Y = np.full((len(SS), ND, N), np.nan)
    with ProcessPoolExecutor(18) as ex:
        t0 = time.time()
        ref = np.concatenate(list(ex.map(_ham, [(z["ecg"][a:b], False) for a, b in chunks(N)])))
        ppg_off = np.concatenate(list(ex.map(_ppg_off, [z["ppg"][a:b] for a, b in chunks(N)])))
        print(f"[ham] reference + PPG done ({time.time() - t0:.0f}s)", flush=True)
        for si, S in enumerate(SS):
            W = np.load(GEN / f"S{S}.npy", mmap_mode="r")
            for d in range(ND):
                Y[si, d] = np.concatenate(list(ex.map(_ham, [(np.asarray(W[d, a:b]), True) for a, b in chunks(N)])))
            print(f"[ham] S={S} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f, Y=Y, ref=ref, ppg_official=ppg_off)


# ================================================================================================ proj
def _proj_ref(x):
    from ppg2ecg.evaluation import rpeaks as R
    return [np.asarray(R.detect_rpeaks(r, FS), int) for r in np.asarray(x, np.float64)]


def _proj(args):
    from ppg2ecg.evaluation import rpeaks as R
    x, refpk = args
    out = np.full((len(x), 3), np.nan)
    for i, (g, rp) in enumerate(zip(np.asarray(x, np.float64), refpk)):
        pk = np.asarray(R.detect_rpeaks(g, FS), int)
        out[i, 0] = R.hr_bpm(pk, FS)
        m, fp, fn = R.match_rpeaks(rp, pk, FS, 50.0)
        out[i, 1] = R.prf(len(m), fp, fn)[2]
        out[i, 2] = R.rr_mae_ms(rp, pk, m, FS)
    return out


def proj():
    from ppg2ecg.evaluation import rpeaks as R
    from scipy.signal import find_peaks
    z = np.load(DATA); N = len(z["ecg"])
    Y = np.full((len(SS), ND, N, 3), np.nan)
    with ProcessPoolExecutor(18) as ex:
        t0 = time.time()
        refpk = sum(ex.map(_proj_ref, [z["ecg"][a:b] for a, b in chunks(N)]), [])
        ref = np.array([R.hr_bpm(p, FS) for p in refpk])
        ppg_fp = np.array([R.hr_bpm(find_peaks(x, distance=42, prominence=0.3)[0], FS) for x in z["ppg"].astype(np.float64)])
        for si, S in enumerate(SS):
            W = np.load(GEN / f"S{S}.npy", mmap_mode="r")
            for d in range(ND):
                Y[si, d] = np.concatenate(list(ex.map(_proj, [(np.asarray(W[d, a:b]), refpk[a:b]) for a, b in chunks(N)])))
            print(f"[proj] S={S} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(RAW / "hr_project.npz", Y=Y[..., 0], F1=Y[..., 1], RR=Y[..., 2], ref=ref, ppg_findpeaks=ppg_fp,
             n_ref_peaks=np.array([len(p) for p in refpk]))


# ================================================================================================ wave
def wave():
    sys.path.insert(0, str(Path(__file__).parent))
    import official as O
    ns = {"np": np}
    from scipy.linalg import sqrtm
    ns["sqrtm"] = sqrtm
    for src in O._extract(O.UP / "evaluation/calculate_metric.py", ("calculate_fd",)).values():
        exec(compile(src, "calculate_metric.py", "exec"), ns)
    z = np.load(DATA); ref = z["ecg"].astype(np.float32); N = len(ref)
    rc = ref - ref.mean(1, keepdims=True); rn = np.sqrt((rc ** 2).sum(1))
    out = {"MAE": np.zeros((len(SS), ND, N), np.float32), "RMSE": np.zeros((len(SS), ND, N), np.float32),
           "R": np.zeros((len(SS), ND, N), np.float32), "wRMS": np.zeros((len(SS), N), np.float32), "FD": np.zeros(len(SS))}
    for si, S in enumerate(SS):
        t0 = time.time()
        W = np.load(GEN / f"S{S}.npy").astype(np.float32)
        for d in range(ND):
            e = W[d] - ref
            out["MAE"][si, d] = np.abs(e).mean(1)
            out["RMSE"][si, d] = np.sqrt((e ** 2).mean(1))
            gc = W[d] - W[d].mean(1, keepdims=True)
            out["R"][si, d] = (gc * rc).sum(1) / (np.sqrt((gc ** 2).sum(1)) * rn)
        acc = np.zeros(N, np.float64)
        for i in range(ND):
            for j in range(i + 1, ND):
                acc += np.sqrt(((W[i] - W[j]) ** 2).mean(1))
        out["wRMS"][si] = acc / (ND * (ND - 1) / 2)
        out["FD"][si] = ns["calculate_fd"](ref[..., None], W[0][..., None])
        print(f"[wave] S={S} FD={out['FD'][si]:.2f} ({time.time() - t0:.0f}s)", flush=True)
        del W
    np.savez(RAW / "wave_metrics.npz", **out)


# ================================================================================================ analyze helpers
def groups(K, scheme="partition"):
    if scheme == "nested" or K == ND:
        return [list(range(K))]
    n = (ND // K) * K
    return [list(range(g * K, (g + 1) * K)) for g in range(n // K)]


def cell_err(Yd, ystar, K, scheme="partition"):
    """Yd [16, N] HR with NaN = undefined. Returns window error [N] (NaN if undefined) and coverage share."""
    errs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in groups(K, scheme):
            errs.append(np.abs(np.nanmedian(Yd[g], 0) - ystar))
        E = np.stack(errs)
        w = np.nanmean(E, 0)
    ok = np.isfinite(ystar)
    return w, float(np.isfinite(E[:, ok]).mean())


class Boot:
    def __init__(self, patient):
        self.up, self.inv = np.unique(patient, return_inverse=True)
        self.P = len(self.up)
        rng = np.random.default_rng(BOOT_SEED)
        self.C = rng.multinomial(self.P, np.full(self.P, 1 / self.P), size=NB).astype(np.float64)   # [NB, P]

    def pmean(self, v, mask=None):
        """per-patient mean of window values (finite & mask) -> [P] (NaN if the patient has none)."""
        ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
        s = np.bincount(self.inv[ok], v[ok], self.P); n = np.bincount(self.inv[ok], None, self.P)
        with np.errstate(invalid="ignore"):
            return s / n

    def ci(self, pv):
        m = np.isfinite(pv)
        pt = float(pv[m].mean())
        b = (self.C[:, m] @ pv[m]) / self.C[:, m].sum(1)
        return {"point": pt, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n_patients": int(m.sum()),
                "share_boot_below_0": float((b < 0).mean())}


def contrast(bt, ea, eb):
    """paired patient-clustered contrast a - b on windows where both are defined."""
    m = np.isfinite(ea) & np.isfinite(eb)
    pa, pb = bt.pmean(ea, m), bt.pmean(eb, m)
    d = pa - pb
    fin = np.isfinite(d)
    r = {"a": bt.ci(pa), "b": bt.ci(pb), "diff": bt.ci(d), "n_windows": int(m.sum()),
         "win_rate_a_better": float((d[fin] < 0).mean()), "tie_rate": float((d[fin] == 0).mean())}
    r["significant"] = "a_better" if r["diff"]["ci"][1] < 0 else ("b_better" if r["diff"]["ci"][0] > 0 else "ns")
    return r


def rank5(x):
    return np.argsort(np.argsort(x, axis=-1), axis=-1).astype(float)


def spearman_rows(a, b):
    ra, rb = rank5(a), rank5(b)
    ra -= ra.mean(-1, keepdims=True); rb -= rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def rho_from_stats(n, s1, s2):
    """n [..], s1 [.., 16], s2 [.., 16, 16] -> mean pairwise Pearson correlation of the 16 error series."""
    mu = s1 / n[..., None]
    cov = s2 / n[..., None, None] - mu[..., :, None] * mu[..., None, :]
    sd = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    corr = cov / (sd[..., :, None] * sd[..., None, :])
    iu = np.triu_indices(ND, 1)
    return corr[..., iu[0], iu[1]].mean(-1)


def fmt(r):
    return f"{r['point']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"


# ================================================================================================ analyze
def analyze():
    from scipy.stats import spearmanr
    z = np.load(DATA); patient = z["patient"]; N = len(patient)
    H = np.load(RAW / "hr_hamilton.npz"); Pz = np.load(RAW / "hr_project.npz"); Wz = np.load(RAW / "wave_metrics.npz")
    und = lambda a: np.where(np.isfinite(a) & (a != -1), a, np.nan)  # noqa: E731
    Yh, ystar = und(H["Y"]), und(H["ref"])                   # [5,16,N], [N]
    Yp, ystar_p = Pz["Y"], Pz["ref"]
    bt = Boot(patient)
    si = {S: i for i, S in enumerate(SS)}
    res = {"prereg": "docs/PPGFLOWECG_FIXED_BUDGET_PREREGISTRATION.md (3dee94f)", "n_windows": N, "n_patients": bt.P}
    nonfin = {S: json.loads((GEN / f"S{S}.nonfinite.json").read_text())["nonfinite_samples"] for S in SS}

    # ---------------------------------------------------------------- coverage / technical flags
    raw = H["Y"]
    res["coverage"] = {
        "Ystar_defined": float(np.isfinite(ystar).mean()), "Ystar_minus1": float((H["ref"] == -1).mean()),
        "Ystar_nan": float(np.isnan(H["ref"]).mean()),
        "sample_defined_by_S": {S: float(np.isfinite(Yh[si[S]]).mean()) for S in SS},
        "sample_minus1_by_S": {S: float((raw[si[S]] == -1).mean()) for S in SS},
        "sample_nan_by_S": {S: float(np.isnan(raw[si[S]]).mean()) for S in SS},
        "project_Ystar_defined": float(np.isfinite(ystar_p).mean()),
        "project_sample_defined_by_S": {S: float(np.isfinite(Yp[si[S]]).mean()) for S in SS},
        "nonfinite_generated_samples_by_S": nonfin}
    tech = {"nonfinite_gt_1pct": any(nonfin[S] / (ND * N) > 0.01 for S in SS),
            "hamilton_undefined_gt_50pct_at_1_10": 1 - res["coverage"]["sample_defined_by_S"][10] > 0.5,
            "ystar_undefined_gt_50pct": 1 - res["coverage"]["Ystar_defined"] > 0.5}

    # ---------------------------------------------------------------- cells (primary Hamilton; secondary project)
    def all_cells(Y, ys, scheme="partition"):
        return {(K, S): cell_err(Y[si[S]], ys, K, scheme) for K, S in CELLS}
    EH, EHn, EP = all_cells(Yh, ystar), all_cells(Yh, ystar, "nested"), all_cells(Yp, ystar_p)
    res["cells"] = {}
    for K, S in CELLS:
        res["cells"][f"K{K}_S{S}"] = {"B": K * S, "hamilton": bt.ci(bt.pmean(EH[(K, S)][0])), "hamilton_coverage": EH[(K, S)][1],
                                      "hamilton_nested": bt.ci(bt.pmean(EHn[(K, S)][0])),
                                      "project": bt.ci(bt.pmean(EP[(K, S)][0])), "project_coverage": EP[(K, S)][1]}
    # ---------------------------------------------------------------- headline contrasts
    res["contrasts"] = {}
    for tag, (a, b) in {**PRIMARY, **SECONDARY, "B20_intermediate_(2,10)-(1,20)": ((2, 10), (1, 20)),
                        "B20_(4,5)-(2,10)_descriptive": ((4, 5), (2, 10)), "consensus_(16,5)-(1,5)": ((16, 5), (1, 5))}.items():
        res["contrasts"][tag] = {"a": list(a), "b": list(b), "hamilton": contrast(bt, EH[a][0], EH[b][0]),
                                 "hamilton_nested": contrast(bt, EHn[a][0], EHn[b][0]), "project": contrast(bt, EP[a][0], EP[b][0])}
    # ---------------------------------------------------------------- depth curve
    res["depth_curve"] = {S: {"hamilton_vs_S10": contrast(bt, EH[(1, S)][0], EH[(1, 10)][0]) if S != 10 else None,
                              "project_vs_S10": contrast(bt, EP[(1, S)][0], EP[(1, 10)][0]) if S != 10 else None} for S in SS}
    hpts = [res["cells"][f"K1_S{S}"]["hamilton"]["point"] for S in SS]
    res["depth_curve_ordering_hamilton"] = [SS[i] for i in np.argsort(hpts)]
    ppts = [res["cells"][f"K1_S{S}"]["project"]["point"] for S in SS]
    res["depth_curve_ordering_project"] = [SS[i] for i in np.argsort(ppts)]
    # paper-literal MAE_hr (official semantics: -1 kept, NaN dropped, window-pooled), K = 1, mean over draws
    lit = {}
    for S in SS:
        v = []
        for d in range(ND):
            f, r = raw[si[S], d], H["ref"]
            ok = ~np.isnan(f) & ~np.isnan(r)
            v.append(float(np.mean(np.abs(r[ok] - f[ok]))))
        lit[S] = float(np.mean(v))
    res["paper_literal_MAE_hr_K1"] = lit
    # ---------------------------------------------------------------- baselines
    e_fp = np.abs(Pz["ppg_findpeaks"] - ystar)
    e_off = np.abs(und(H["ppg_official"]) - ystar)
    const = float(np.nanmedian(ystar)); e_c = np.abs(const - ystar)
    res["baselines"] = {"ppg_findpeaks": bt.ci(bt.pmean(e_fp)), "ppg_findpeaks_coverage": float(np.isfinite(e_fp[np.isfinite(ystar)]).mean()),
                        "ppg_official": bt.ci(bt.pmean(e_off)), "ppg_official_coverage": float(np.isfinite(e_off[np.isfinite(ystar)]).mean()),
                        "constant_median": const, "constant": bt.ci(bt.pmean(e_c)),
                        "K1S10_minus_ppg_findpeaks": contrast(bt, EH[(1, 10)][0], e_fp),
                        "K1S10_minus_constant": contrast(bt, EH[(1, 10)][0], e_c),
                        "project_K1S10_minus_ppg_findpeaks_projref": contrast(bt, EP[(1, 10)][0], np.abs(Pz["ppg_findpeaks"] - ystar_p))}
    # ---------------------------------------------------------------- waveform / event metrics (per sample, draw mean)
    wv = {}
    for S in SS:
        i = si[S]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = {"MAE": Wz["MAE"][i].mean(0), "RMSE": Wz["RMSE"][i].mean(0), "pearson_r": Wz["R"][i].mean(0),
                 "rpeak_F1": Pz["F1"][i].mean(0), "RR_MAE_ms": np.nanmean(Pz["RR"][i], 0), "waveform_pairwise_RMS": Wz["wRMS"][i]}
        wv[S] = {k: bt.ci(bt.pmean(v.astype(np.float64))) for k, v in d.items()}
        wv[S]["FD_official_draw0"] = float(Wz["FD"][i])
    res["waveform"] = wv
    deep = (10, 15, 20, 25)
    f1_5, rm_5 = wv[5]["rpeak_F1"]["point"], wv[5]["RMSE"]["point"]
    f1_best = max(wv[S]["rpeak_F1"]["point"] for S in deep); rm_best = min(wv[S]["RMSE"]["point"] for S in deep)
    collapse = ((f1_best - f1_5 > 0.05 and (f1_best - f1_5) / max(f1_best, 1e-12) > 0.20) or (rm_5 - rm_best) / rm_best > 0.20
                or nonfin[5] / (ND * N) > 0.01)
    res["collapse_check"] = {"F1_S5": f1_5, "F1_best_deep": f1_best, "RMSE_S5": rm_5, "RMSE_best_deep": rm_best, "collapse": bool(collapse)}

    # ---------------------------------------------------------------- fixed-K mechanism
    om = np.isfinite(ystar) & np.isfinite(Yh).all((0, 1))
    res["mechanism"] = {"omega_mech_windows": int(om.sum()), "omega_mech_patients": int(len(np.unique(patient[om])))}
    inv = bt.inv
    mech_rows, pp_rows = {}, {}
    stats_pat = {}
    for S in SS:
        Y = Yh[si[S]][:, om]; ys = ystar[om]; e = Y - ys
        I = np.abs(e).mean(0); C = np.abs(np.median(Y, 0) - ys); G = I - C
        SD = Y.std(0, ddof=1); MAD = np.median(np.abs(Y - np.median(Y, 0)), 0)
        wr = Wz["wRMS"][si[S]][om].astype(np.float64)
        vals = {}
        for k, v in (("I", I), ("C", C), ("G", G), ("SD", SD), ("MAD", MAD), ("wRMS", wr)):
            w = np.full(N, np.nan); w[om] = v; vals[k] = bt.pmean(w)
        rho = float(np.mean(np.corrcoef(e)[np.triu_indices(ND, 1)]))
        # per-patient sufficient statistics for the bootstrap of rho
        io = inv[om]
        n_p = np.bincount(io, None, bt.P).astype(float)
        s1 = np.stack([np.bincount(io, e[k], bt.P) for k in range(ND)], 1)
        s2 = np.zeros((bt.P, ND, ND))
        for k in range(ND):
            for l in range(k, ND):
                s2[:, k, l] = s2[:, l, k] = np.bincount(io, e[k] * e[l], bt.P)
        stats_pat[S] = (n_p, s1, s2, vals)
        mech_rows[S] = {"n_windows": int(om.sum()), "I": bt.ci(vals["I"]), "C": bt.ci(vals["C"]), "G": bt.ci(vals["G"]),
                        "G_over_I": float(vals["G"][np.isfinite(vals["G"])].mean() / vals["I"][np.isfinite(vals["I"])].mean()),
                        "SD": bt.ci(vals["SD"]), "MAD": bt.ci(vals["MAD"]), "wRMS": bt.ci(vals["wRMS"]),
                        "rho_bar": rho, "K_eff": 16 / (1 + 15 * rho)}
        # secondary: patient-level Spearman(rho_p, G_p), patients with >= 8 mech windows
        rp, gp = [], []
        for p in np.flatnonzero(n_p >= 8):
            sel = io == p
            c = np.corrcoef(e[:, sel])
            if np.isfinite(c).all():
                rp.append(c[np.triu_indices(ND, 1)].mean()); gp.append(vals["G"][p])
        rp, gp = np.array(rp), np.array(gp)
        rng = np.random.default_rng(BOOT_SEED)
        bs = [spearmanr(rp[ix], gp[ix])[0] for ix in (rng.integers(0, len(rp), len(rp)) for _ in range(NB))]
        pp_rows[S] = {"n_patients": int(len(rp)), "spearman": float(spearmanr(rp, gp)[0]),
                      "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "share_negative": float(np.mean(np.array(bs) < 0))}
    res["mechanism"]["per_S"] = mech_rows
    res["mechanism"]["patient_level_spearman_rho_G"] = pp_rows
    # across-S Spearman with patient bootstrap
    Cb = bt.C
    rb = np.stack([rho_from_stats(Cb @ stats_pat[S][0], np.einsum("bp,pk->bk", Cb, stats_pat[S][1]),
                                  (Cb @ stats_pat[S][2].reshape(bt.P, -1)).reshape(NB, ND, ND)) for S in SS], 1)   # [NB, 5]

    def bmean(S, k):
        v = stats_pat[S][3][k]; m = np.isfinite(v)
        return (Cb[:, m] @ v[m]) / Cb[:, m].sum(1)
    Gb = np.stack([bmean(S, "G") for S in SS], 1)
    Ib = np.stack([bmean(S, "I") for S in SS], 1)
    pts = {k: np.array([mech_rows[S][k]["point"] if isinstance(mech_rows[S][k], dict) else mech_rows[S][k] for S in SS])
           for k in ("G", "I", "SD", "MAD", "wRMS", "rho_bar")}
    sp = {}
    for name, (xa, xb, ba, bb) in {"rho_bar_vs_G": (pts["rho_bar"], pts["G"], rb, Gb),
                                   "rho_bar_vs_G_over_I": (pts["rho_bar"], pts["G"] / pts["I"], rb, Gb / Ib),
                                   "wRMS_vs_G": (pts["wRMS"], pts["G"], np.stack([bmean(S, "wRMS") for S in SS], 1), Gb),
                                   "SD_vs_G": (pts["SD"], pts["G"], np.stack([bmean(S, "SD") for S in SS], 1), Gb),
                                   "MAD_vs_G": (pts["MAD"], pts["G"], np.stack([bmean(S, "MAD") for S in SS], 1), Gb)}.items():
        b = spearman_rows(ba, bb)
        sp[name] = {"point": float(spearmanr(xa, xb)[0]), "ci": [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))],
                    "share_negative": float(np.mean(b < 0))}
    sp["rho_bar_vs_G"]["reproduces"] = bool(sp["rho_bar_vs_G"]["point"] < 0 and sp["rho_bar_vs_G"]["share_negative"] >= 0.95)
    res["mechanism"]["across_S_spearman"] = sp
    res["mechanism"]["ordering"] = {k: [SS[i] for i in np.argsort(pts[k])] for k in pts}
    res["mechanism"]["rho_bar_boot_ci"] = {S: [float(np.percentile(rb[:, i], 2.5)), float(np.percentile(rb[:, i], 97.5))] for i, S in enumerate(SS)}

    # ---------------------------------------------------------------- latency
    lat = json.loads((ART / "latency.json").read_text())
    lat["per_cell"] = json.loads((RAW / "latency_cells.json").read_text())
    (ART / "latency.json").write_text(json.dumps(lat, indent=1))

    # ---------------------------------------------------------------- verdict
    prim = {t: res["contrasts"][t]["hamilton"] for t in PRIMARY}
    s = [t for t, r in prim.items() if r["significant"] == "a_better"]
    dsig = [t for t, r in prim.items() if r["significant"] == "b_better"]
    nneg = sum(r["diff"]["point"] < 0 for r in prim.values())
    no_gain = res["contrasts"]["consensus_(16,5)-(1,5)"]["hamilton"]["diff"]["ci"][1] >= 0
    not_usable = res["cells"]["K1_S10"]["hamilton"]["point"] >= res["baselines"]["constant"]["point"]
    if any(tech.values()):
        verdict = "UNINTERPRETABLE"
    elif not_usable or no_gain or nneg == 0 or (not s and nneg <= 1):
        verdict = "FAILED"
    elif len(s) >= 2 and ({"B15", "B20"} & set(s)) and not dsig and not collapse:
        verdict = "STRONG"
    else:
        verdict = "PARTIAL"
    ppg_flag = res["baselines"]["K1S10_minus_ppg_findpeaks"]["diff"]["point"] > 0
    res["verdict"] = {"category": verdict, "succeeded": s, "depth_significant": dsig, "n_negative_points": int(nneg),
                      "flags": {**tech, "not_usable_vs_constant": bool(not_usable), "no_useful_consensus_gain": bool(no_gain),
                                "waveform_collapse": bool(collapse), "instability_K1_worse_than_ppg_peaks": bool(ppg_flag)},
                      "B25_secondary": res["contrasts"]["B25"]["hamilton"]["significant"]}

    # ---------------------------------------------------------------- write artifacts
    jd = lambda o: json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))  # noqa: E731
    (ART / "fixed_budget_bootstrap.json").write_text(json.dumps(jd({k: res[k] for k in ("prereg", "n_windows", "n_patients", "coverage",
        "cells", "contrasts", "depth_curve", "depth_curve_ordering_hamilton", "depth_curve_ordering_project", "paper_literal_MAE_hr_K1",
        "baselines", "waveform", "collapse_check", "verdict")}), indent=1))
    (ART / "fixed_k_bootstrap.json").write_text(json.dumps(jd(res["mechanism"]), indent=1))
    with open(ART / "fixed_budget.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "S", "B_vector_field_NFE", "encoder_decoder_calls", "hamilton_err", "hamilton_lo", "hamilton_hi", "hamilton_coverage",
                    "hamilton_nested_err", "project_err", "project_lo", "project_hi", "latency_seq_batch1_ms", "latency_batched_ms",
                    "per_sample_rpeak_F1", "per_sample_RMSE"])
        for K, S in CELLS:
            c = res["cells"][f"K{K}_S{S}"]; L = lat["per_cell"][f"K{K}_S{S}"]
            w.writerow([K, S, K * S, 3 * K, c["hamilton"]["point"], *c["hamilton"]["ci"], c["hamilton_coverage"], c["hamilton_nested"]["point"],
                        c["project"]["point"], *c["project"]["ci"], L["sequential_batch1_ms"], L["batched_K_ms"],
                        wv[S]["rpeak_F1"]["point"], wv[S]["RMSE"]["point"]])
    with open(ART / "fixed_k_mechanism.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["S", "n_windows", "I", "C", "G", "G_lo", "G_hi", "G_over_I", "SD", "MAD", "waveform_pairwise_RMS", "rho_bar",
                    "rho_bar_lo", "rho_bar_hi", "K_eff"])
        for S in SS:
            m = mech_rows[S]
            w.writerow([S, m["n_windows"], m["I"]["point"], m["C"]["point"], m["G"]["point"], *m["G"]["ci"], m["G_over_I"], m["SD"]["point"],
                        m["MAD"]["point"], m["wRMS"]["point"], m["rho_bar"], *res["mechanism"]["rho_bar_boot_ci"][S], m["K_eff"]])
    with open(ART / "per_patient_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["patient", "n_windows", "n_omega_hr"] + [f"ham_K{K}_S{S}" for K, S in CELLS] + [f"proj_K{K}_S{S}" for K, S in CELLS] + \
              ["ppg_findpeaks", "ppg_official"] + [f"mech_{k}_S{S}" for S in SS for k in ("I", "C", "G", "SD", "MAD", "wRMS")]
        w.writerow(hdr)
        cols = [np.bincount(bt.inv, None, bt.P), np.bincount(bt.inv, np.isfinite(ystar).astype(float), bt.P)]
        cols += [bt.pmean(EH[c][0]) for c in CELLS] + [bt.pmean(EP[c][0]) for c in CELLS] + [bt.pmean(e_fp), bt.pmean(e_off)]
        cols += [stats_pat[S][3][k] for S in SS for k in ("I", "C", "G", "SD", "MAD", "wRMS")]
        for p in range(bt.P):
            w.writerow([int(bt.up[p])] + [("" if not np.isfinite(c[p]) else round(float(c[p]), 6)) for c in cols])
    # console summary
    print(json.dumps(res["coverage"], indent=1))
    for t in list(PRIMARY) + list(SECONDARY) + ["B20_intermediate_(2,10)-(1,20)", "consensus_(16,5)-(1,5)"]:
        r = res["contrasts"][t]
        print(f"{t:34s} ham a {fmt(r['hamilton']['a'])} b {fmt(r['hamilton']['b'])} diff {fmt(r['hamilton']['diff'])} "
              f"win {r['hamilton']['win_rate_a_better']:.2f} | nested diff {fmt(r['hamilton_nested']['diff'])} | proj diff {fmt(r['project']['diff'])}")
    for S in SS:
        print(f"K1 S{S}: ham {fmt(res['cells'][f'K1_S{S}']['hamilton'])} proj {fmt(res['cells'][f'K1_S{S}']['project'])} "
              f"F1 {wv[S]['rpeak_F1']['point']:.3f} RMSE {wv[S]['RMSE']['point']:.3f} r {wv[S]['pearson_r']['point']:.3f} FD {wv[S]['FD_official_draw0']:.1f} "
              f"| mech G {fmt(mech_rows[S]['G'])} rho {mech_rows[S]['rho_bar']:.3f} wRMS {mech_rows[S]['wRMS']['point']:.3f}")
    print("baselines", fmt(res["baselines"]["ppg_findpeaks"]), fmt(res["baselines"]["ppg_official"]), fmt(res["baselines"]["constant"]))
    print("spearman", json.dumps(sp)); print("verdict", json.dumps(res["verdict"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["gen", "ham", "proj", "wave", "analyze"])
    {"gen": gen, "ham": ham, "proj": proj, "wave": wave, "analyze": analyze}[ap.parse_args().stage]()
