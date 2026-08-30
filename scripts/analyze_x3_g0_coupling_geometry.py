"""X3-G0 - coupling cost-geometry / feasibility gate.

Frozen protocol: docs/X3_G0_COUPLING_GEOMETRY_PREREGISTRATION.md (prereg commit a9707e7, pushed before any G0 metric).
ZERO DEEP TRAINING. Frozen-checkpoint inference only (A6 / iMF-1 / OT-CFM-50). Writes only to new X3 paths.
WildPPG TEST subjects (kjd, ssx) are firewalled: any attempt to load them raises.

  --stage cache    frozen A6 predictions + GT R-peak QRS masks for the pooled train/val windows -> outputs/x3_*/
  --stage analyze  assignment geometry, cross-objective regret, source->residual dependence, linear endpoint proxy
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (must precede torch)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.evaluation import coupling_geometry as CG  # noqa: E402
from ppg2ecg.evaluation.metrics import hf_energy_ratio, rhythm_morphology_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/x3_g0_coupling_geometry"
OUT = ROOT / "artifacts/x3_g0_coupling_geometry"

# ---- frozen protocol constants (pre-registration sections 5-12)
POOL_CAP = 4096
B_VALUES = (1, 8, 32, 64, 128, 256, 512)
PAIR_BUDGET = 32768
ASSIGN_SEED = 20260830
N_RANDOM_REGRET = 16
N_PERM = 100
RIDGE_LAM = 1e-3
PCA_VAR, PCA_CAP = 0.95, 128
K_PROXY = 32
PROXY_PER_FOLD = 64
DOMAINS = ("FULL", "QRS", "HF")
PRIMARY_ARMS = ("RAW", "WHITE", "HF")
RESID_B = (64, 256)
A6_CKPT = "outputs/a6c_fullbackbone_mse_wildppg_seed42/checkpoint_best.pt"
IMF_CKPT = "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
OT_CKPT = "outputs/a4_otcfm_wildppg_seed42/checkpoint_best.pt"
T_LEN = 1024


def n_batches_for(b: int) -> int:
    return max(32, -(-PAIR_BUDGET // b))


def split_subjects():
    s = json.loads((ROOT / "data/manifests/split_a4_wildppg_seed42.json").read_text())["splits"][0]
    return s["train"], s["val"], s["test"]


def fold_mapping(train_subjects):
    """Frozen: 4 folds, 3 held-out subjects each, by manifest order."""
    return {f: list(train_subjects[3 * f : 3 * f + 3]) for f in range(4)}


def stride_pool(n: int, cap: int = POOL_CAP) -> np.ndarray:
    stride = -(-n // cap) if n > cap else 1
    return np.arange(0, n, stride)[:cap]


# ----------------------------------------------------------------------------------------------------------------------
# Stage 1: cache
# ----------------------------------------------------------------------------------------------------------------------
def _rpeaks_chunk(args):
    from ppg2ecg.evaluation import rpeaks as R
    y, = args
    return [R.detect_rpeaks(w, CG.FS) for w in y]


def stage_cache(args):
    import torch
    from ppg2ecg.flow.imeanflow import MeanFlowS5, sample_meanflow
    from ppg2ecg.flow.samplers import heun_sample, nfe_of
    from ppg2ecg.models import build_penguin_backbone
    from ppg2ecg.models.regressor import REGRESSOR_MODELS
    from ppg2ecg.utils.upstream import assert_upstream_pinned

    train, val, test = split_subjects()
    subjects = list(train) + list(val)
    CG.assert_no_test_subjects(subjects)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    up = assert_upstream_pinned()

    ck = torch.load(ROOT / A6_CKPT, map_location="cpu", weights_only=False)
    cls, _ = REGRESSOR_MODELS[ck.get("model_key", "state_token")]
    a6 = cls(**ck["model_cfg"]).to(device).eval()
    a6.load_state_dict(ck["state_dict"])

    meta = {"subjects": {}, "created": datetime.now().isoformat(timespec="seconds"), "upstream": up,
            "a6_checkpoint": A6_CKPT, "a6_epoch": ck.get("epoch"), "pool_cap": POOL_CAP, "fs": CG.FS}
    for sub in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        idx = stride_pool(len(d["x"]))
        x, y = d["x"][idx].astype(np.float32), d["y"][idx].astype(np.float32)
        site = np.asarray(d["site"])[idx].astype(str)
        with torch.no_grad():
            m = np.concatenate([a6(torch.from_numpy(x[i : i + 256]).to(device).unsqueeze(1)).squeeze(1).float().cpu().numpy()
                                for i in range(0, len(x), 256)])
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            chunks = np.array_split(np.arange(len(y)), args.workers * 4)
            rp = [p for part in ex.map(_rpeaks_chunk, [(y[c],) for c in chunks if len(c)]) for p in part]
        qmask = CG.qrs_mask_from_rpeaks(rp, T_LEN).astype(np.uint8)
        np.savez_compressed(CACHE / f"pool_{sub}.npz", x=x, y=y, m=m.astype(np.float32), qmask=qmask,
                            site=site, window_index=idx.astype(np.int32),
                            n_beats=np.array([len(p) for p in rp], dtype=np.int32))
        meta["subjects"][sub] = {"n_pool": int(len(idx)), "n_source": int(len(d["x"])), "stride": int(idx[1] - idx[0]) if len(idx) > 1 else 1,
                                 "n_beats_total": int(sum(len(p) for p in rp)), "sites": sorted(set(site.tolist()))}
        print(f"  cached {sub}: {len(idx)} windows, {sum(len(p) for p in rp)} GT beats", flush=True)

    # ---- proxy windows (frozen) and reference-model predictions on them
    folds = fold_mapping(train)
    proxy = {}
    for f, ho in folds.items():
        n_ho = sum(meta["subjects"][s]["n_pool"] for s in ho)
        sel = np.round(np.linspace(0, n_ho - 1, PROXY_PER_FOLD)).astype(int)
        proxy[str(f)] = {"held_out": ho, "n_held_out": int(n_ho), "sel": sel.tolist()}
    meta["proxy"] = proxy
    meta["folds"] = {str(k): v for k, v in folds.items()}

    imf_ck = torch.load(ROOT / IMF_CKPT, map_location="cpu", weights_only=False)
    imf = MeanFlowS5(build_penguin_backbone(**imf_ck["model_cfg"]), **{k: imf_ck.get("imf_cfg", {}).get(k, d)
                     for k, d in (("cond_mode", "h_only"), ("h_scale", 1.0))}).to(device).eval()
    imf.load_state_dict(imf_ck["state_dict"])
    ot_ck = torch.load(ROOT / OT_CKPT, map_location="cpu", weights_only=False)
    otm = build_penguin_backbone(**ot_ck["model_cfg"]).to(device).eval()
    otm.load_state_dict(ot_ck["state_dict"])

    for f, ho in folds.items():
        xs = np.concatenate([np.load(CACHE / f"pool_{s}.npz")["x"] for s in ho])
        sel = np.array(proxy[str(f)]["sel"])
        ppg = xs[sel]
        g = __import__("torch").Generator().manual_seed(0)
        e = __import__("torch").randn(len(ppg), 1, T_LEN, generator=g)
        with torch.no_grad():
            p = torch.from_numpy(ppg).to(device).unsqueeze(1)
            imf1 = sample_meanflow(imf, p, e.to(device), 1)[0].squeeze(1).float().cpu().numpy()
            v = lambda xx, tt: otm.forward_step(xx, p, tt)  # noqa: E731
            ot50, nfe = heun_sample(v, e.to(device), 25)
            ot50 = ot50.squeeze(1).float().cpu().numpy()
        assert nfe == nfe_of("heun", 25) == 50
        np.savez_compressed(CACHE / f"refs_fold{f}.npz", imf1=imf1.astype(np.float32), ot50=ot50.astype(np.float32), sel=sel)
        print(f"  fold {f} refs: iMF1 + OT50 (NFE {nfe}) on {len(sel)} proxy windows", flush=True)

    meta["ref_checkpoints"] = {"iMF1": IMF_CKPT, "OT50": OT_CKPT, "ot50_nfe": 50, "ref_noise_seed": 0}
    (CACHE / "cache_meta.json").write_text(json.dumps(meta, indent=1))
    print("cache written to", CACHE)


# ----------------------------------------------------------------------------------------------------------------------
# Stage 2: analyze
# ----------------------------------------------------------------------------------------------------------------------
def load_pool(subs):
    CG.assert_no_test_subjects(subs)
    parts = [np.load(CACHE / f"pool_{s}.npz") for s in subs]
    return {"x": np.concatenate([p["x"] for p in parts]), "y": np.concatenate([p["y"] for p in parts]),
            "m": np.concatenate([p["m"] for p in parts]), "qmask": np.concatenate([p["qmask"] for p in parts]),
            "site": np.concatenate([np.asarray(p["site"]).astype(str) for p in parts]),
            "subject": np.concatenate([np.full(len(p["x"]), s) for s, p in zip(subs, parts)])}


def make_phis(pool, w):
    y = pool["y"].astype(np.float64)
    return {"RAW": CG.phi_raw(y).astype(np.float32), "WHITE": CG.phi_white(y, w).astype(np.float32),
            "HF": CG.phi_hf(y).astype(np.float32), "RESID": CG.phi_resid(y, pool["m"]).astype(np.float32)}


def make_pairs(pool_phis, n_pool, b, seed, arms, n_batches, collect_regret):
    """Shared batches across arms: draw window indices and x0 once, then assign under each cost geometry."""
    rng = np.random.default_rng(seed)
    x0_all = np.empty((n_batches * b, T_LEN), dtype=np.float32)
    tgt = {a: np.empty(n_batches * b, dtype=np.int32) for a in arms}
    regret_acc, overlap_acc, cost_acc = [], [], []
    for k in range(n_batches):
        idx = rng.choice(n_pool, b, replace=False)
        x0 = rng.standard_normal((b, T_LEN)).astype(np.float32)
        x0_all[k * b : (k + 1) * b] = x0
        if b == 1:
            for a in arms:
                tgt[a][k] = idx[0]
            continue
        phis = {a: pool_phis[a][idx] for a in arms}
        if collect_regret:
            res = CG.cross_objective_regret(x0, phis, N_RANDOM_REGRET, seed + k)
            regret_acc.append(res["regret"])
            overlap_acc.append(res["overlap"])
            cost_acc.append(res["cost"])
            for a in arms:
                tgt[a][k * b : (k + 1) * b] = idx[res["perms"][a]]
        else:
            for a in arms:
                tgt[a][k * b : (k + 1) * b] = idx[CG.solve_assignment(x0, phis[a])]
    agg = None
    if collect_regret and regret_acc:
        keys = regret_acc[0].keys()
        agg = {"regret": {k: float(np.mean([r[k] for r in regret_acc])) for k in keys},
               "overlap": {k: float(np.mean([o[k] for o in overlap_acc])) for k in keys},
               "cost": {a: {m: float(np.mean([c[a][m] for c in cost_acc])) for m in ("assigned", "random", "reduction")} for a in arms}}
    return x0_all, tgt, agg


def _metric_chunk(args):
    pred, gt, qmask = args
    rm = rhythm_morphology_metrics(pred, gt, CG.FS)
    d, dg = np.diff(pred, axis=1) * CG.FS, np.diff(gt, axis=1) * CG.FS
    mask = qmask.astype(bool)
    # NaN-safe accumulation: morphology/F1 are undefined on windows where no beats are detected, so the
    # combiner needs (sum, count) of FINITE values rather than a per-chunk mean.
    def sc(v):
        v = np.asarray(v, dtype=np.float64)
        f = np.isfinite(v)
        return float(v[f].sum()), int(f.sum())

    ms, mc = sc(rm["morph_corr"])
    fs_, fc = sc(rm["rpeak_f1"])
    return {"morph_sum": ms, "morph_n": mc, "f1_sum": fs_, "f1_n": fc,
            "amp_sum": float((pred.std(1) / (gt.std(1) + 1e-8)).sum()),
            "qrs_pred_var": float(pred[mask].var()) * float(mask.sum()), "qrs_gt_var": float(gt[mask].var()) * float(mask.sum()),
            "qrs_n": float(mask.sum()),
            "slope_pred_sum": float(np.abs(d).max(1).sum()), "slope_gt_sum": float(np.abs(dg).max(1).sum()),
            "hf_sum": float(hf_energy_ratio(pred).sum()), "rmse_sum": float(np.sqrt(((pred - gt) ** 2).mean(1)).sum()),
            "n": len(pred)}


def structural_metrics(pred, gt, qmask, pool_workers):
    chunks = [c for c in np.array_split(np.arange(len(pred)), max(1, pool_workers)) if len(c)]
    with ProcessPoolExecutor(max_workers=pool_workers) as ex:
        res = list(ex.map(_metric_chunk, [(pred[c], gt[c], qmask[c]) for c in chunks]))
    n = sum(r["n"] for r in res)
    tot = {k: sum(r[k] for r in res) for k in res[0]}
    return {"morph": (tot["morph_sum"] / tot["morph_n"]) if tot["morph_n"] else float("nan"),
            "morph_n": float(tot["morph_n"]),
            "f1": (tot["f1_sum"] / tot["f1_n"]) if tot["f1_n"] else float("nan"),
            "amp": tot["amp_sum"] / n,
            "qrs_energy": tot["qrs_pred_var"] / max(tot["qrs_gt_var"], 1e-12),
            "slope": tot["slope_pred_sum"] / max(tot["slope_gt_sum"], 1e-12),
            "hf": tot["hf_sum"] / n, "rmse": tot["rmse_sum"] / n, "n_windows": float(n)}


def run_setting(name, subs_fit_by_fold, folds, args, secondary=False):
    """Primary (train cross-fit) or secondary (validation) run under the identical frozen protocol."""
    rows = {k: [] for k in ("dose", "overlap", "regret", "r2", "null", "proxy", "recovery", "dim")}
    per_fold = {}
    for f, ho in folds.items():
        fit_subs = subs_fit_by_fold[f]
        pool_fit, pool_ho = load_pool(fit_subs), load_pool(ho)
        w = CG.fit_whitener(pool_fit["y"])                       # FIT subjects only
        phi_fit, phi_ho = make_phis(pool_fit, w), make_phis(pool_ho, w)
        r_fit = (pool_fit["y"].astype(np.float64) - pool_fit["m"])
        r_ho = (pool_ho["y"].astype(np.float64) - pool_ho["m"])
        dom_fit = CG.residual_domains(r_fit, pool_fit["qmask"])
        dom_ho = CG.residual_domains(r_ho, pool_ho["qmask"])
        pca = {d: CG.PCABasis(dom_fit[d], PCA_VAR, PCA_CAP) for d in DOMAINS}   # FIT subjects only
        for d in DOMAINS:
            pr = CG.participation_ratio(dom_fit[d])
            rows["dim"].append({"setting": name, "fold": f, "representation": d, **pr, "pca_k": pca[d].k_,
                                "pca_explained": round(pca[d].explained_, 4)})
        rows["dim"].append({"setting": name, "fold": f, "representation": "GT", **CG.participation_ratio(pool_fit["y"].astype(np.float64)),
                            "pca_k": -1, "pca_explained": -1})
        z_fit = {d: pca[d].transform(dom_fit[d]) for d in DOMAINS}
        z_ho = {d: pca[d].transform(dom_ho[d]) for d in DOMAINS}

        # proxy windows / references for this fold
        sel = np.array(json.loads((CACHE / "cache_meta.json").read_text())["proxy"][str(f)]["sel"]) if not secondary \
            else np.round(np.linspace(0, len(pool_ho["y"]) - 1, PROXY_PER_FOLD)).astype(int)
        gt_px, m_px, q_px = pool_ho["y"][sel].astype(np.float64), pool_ho["m"][sel].astype(np.float64), pool_ho["qmask"][sel]
        proxy_x0 = np.stack([np.random.default_rng(s).standard_normal((len(sel), T_LEN)) for s in range(K_PROXY)])

        for b in B_VALUES:
            arms = list(PRIMARY_ARMS) + (["RESID"] if b in RESID_B else [])
            nb = n_batches_for(b)
            seed = ASSIGN_SEED + 1000 * f + b
            xf, tf, agg = make_pairs(phi_fit, len(pool_fit["y"]), b, seed, arms, nb, collect_regret=(b > 1))
            xh, th, _ = make_pairs(phi_ho, len(pool_ho["y"]), b, seed + 500000, arms, nb, collect_regret=False)
            ridge_x = xf.astype(np.float64)
            if agg:
                for (q, p), v in agg["regret"].items():
                    if q != p:
                        rows["regret"].append({"setting": name, "fold": f, "B": b, "cost_q": q, "assignment_p": p,
                                               "regret": round(v, 6), "overlap": round(agg["overlap"][(q, p)], 6)})
                for a in arms:
                    rows["overlap"].append({"setting": name, "fold": f, "B": b, "arm": a, **{k: round(v, 6) for k, v in agg["cost"][a].items()}})
            for a in arms:
                red = agg["cost"][a]["reduction"] if agg else 0.0
                dose = {"setting": name, "fold": f, "B": b, "arm": a if b > 1 else "independent", "cost_reduction": round(red, 6)}
                for d in DOMAINS:
                    res = CG.dependence_with_null(ridge_x, z_fit[d][tf[a]], xh.astype(np.float64), z_ho[d][th[a]],
                                               lam=RIDGE_LAM, n_perm=args.n_perm, seed=seed)
                    dose[f"dR2_{d}"] = round(res["delta_r2"], 6)
                    rows["r2"].append({"setting": name, "fold": f, "B": b, "arm": a, "domain": d,
                                       **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()}})
                dose["QRS_relative"] = round(max(dose["dR2_QRS"], 0) / max(dose["dR2_FULL"], 1e-9), 4)
                dose["HF_relative"] = round(max(dose["dR2_HF"], 0) / max(dose["dR2_FULL"], 1e-9), 4)
                rows["dose"].append(dose)
                # ---- linear endpoint proxy from the FULL-residual map
                a_full = CG.RidgeDiagnostic(ridge_x, RIDGE_LAM).fit(z_fit["FULL"][tf[a]])
                preds = np.concatenate([m_px + pca["FULL"].inverse_transform(proxy_x0[s] @ a_full) for s in range(K_PROXY)])
                sm = structural_metrics(preds, np.tile(gt_px, (K_PROXY, 1)), np.tile(q_px, (K_PROXY, 1)), args.workers)
                rows["proxy"].append({"setting": name, "fold": f, "B": b, "arm": a if b > 1 else "independent", **{k: round(v, 6) for k, v in sm.items()}})
            print(f"  [{name}] fold {f} B={b:4d} " + " | ".join(
                f"{r['arm']}: dFULL {r['dR2_FULL']:+.4f} dQRS {r['dR2_QRS']:+.4f} dHF {r['dR2_HF']:+.4f} red {r['cost_reduction']:.3f}"
                for r in rows["dose"] if r["fold"] == f and r["B"] == b), flush=True)
        # references on the same proxy windows
        refs = {"A6": m_px}
        if not secondary:
            rf = np.load(CACHE / f"refs_fold{f}.npz")
            refs["iMF1"], refs["OT50"] = rf["imf1"].astype(np.float64), rf["ot50"].astype(np.float64)
        for rn, rp in refs.items():
            rows["proxy"].append({"setting": name, "fold": f, "B": -1, "arm": f"ref:{rn}",
                                  **{k: round(v, 6) for k, v in structural_metrics(rp, gt_px, q_px, args.workers).items()}})
        per_fold[f] = {"n_fit": len(pool_fit["y"]), "n_ho": len(pool_ho["y"]), "pca_k": {d: pca[d].k_ for d in DOMAINS}}
    return rows, per_fold


def pool_rows(rows, keys, value_keys):
    """Average per-fold rows into the primary pooled estimate."""
    out = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        out.setdefault(k, []).append(r)
    res = []
    for k, rs in out.items():
        d = dict(zip(keys, k))
        for v in value_keys:
            vals = [r[v] for r in rs if isinstance(r.get(v), (int, float)) and np.isfinite(r[v])]
            d[v] = float(np.mean(vals)) if vals else float("nan")
            d[v + "_sd"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        d["n_folds"] = len(rs)
        res.append(d)
    return res


def gate_verdict(dose, proxy):
    """Frozen resource-gate rules (pre-registration sec. 14)."""
    def dmax(arm, dom):
        v = [r[f"dR2_{dom}"] for r in dose if r["arm"] == arm and r["B"] > 1]
        return max(v) if v else float("nan")

    ref = {r["arm"].split(":")[1]: r for r in proxy if str(r["arm"]).startswith("ref:")}
    def recovery(arm_rows, metric):
        if "A6" not in ref or "iMF1" not in ref:
            return None
        den = ref["iMF1"][metric] - ref["A6"][metric]
        if den <= 1e-6:
            return None
        return max((r[metric] - ref["A6"][metric]) / den for r in arm_rows) if arm_rows else None

    def struct_ok(arm, thr):
        rs = [r for r in proxy if r["arm"] == arm]
        mo = recovery(rs, "morph")
        if mo is None or mo < thr:
            return False, mo, []
        others = [m for m in ("qrs_energy", "slope", "hf") if (recovery(rs, m) or -9) >= thr]
        return len(others) >= 1, mo, others

    raw_strong = max(dmax("RAW", "QRS"), dmax("RAW", "HF")) >= 0.05
    raw_weak = dmax("RAW", "QRS") < 0.02 and dmax("RAW", "HF") < 0.02
    spec_strong = any(max(dmax(a, "QRS"), dmax(a, "HF")) >= 0.05 for a in ("WHITE", "HF"))
    all_weak = all(dmax(a, "QRS") < 0.02 and dmax(a, "HF") < 0.02 for a in PRIMARY_ARMS)
    raw_ok, raw_mo, raw_oth = struct_ok("RAW", 0.20)
    spec_ok = any(struct_ok(a, 0.20)[0] for a in ("WHITE", "HF"))
    weak_struct = all((struct_ok(a, 0.10)[1] or 0) < 0.10 for a in PRIMARY_ARMS)

    if raw_strong and raw_ok:
        v = "RAW-COUPLING GO"
    elif raw_weak and spec_strong and spec_ok:
        v = "SPECTRAL-COUPLING CANDIDATE"
    elif raw_weak and spec_strong:
        v = "COST-GEOMETRY LIMITED"
    elif all_weak and weak_struct:
        v = "WEAK FINITE-BATCH LEVER UNDER TESTED RANGE"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "max_dR2": {a: {d: dmax(a, d) for d in DOMAINS} for a in PRIMARY_ARMS},
            "raw_strong": bool(raw_strong), "raw_weak": bool(raw_weak), "spectral_strong": bool(spec_strong),
            "all_weak": bool(all_weak), "raw_structural_ok": bool(raw_ok), "spectral_structural_ok": bool(spec_ok),
            "raw_morph_recovery": raw_mo, "references": {k: {m: ref[k][m] for m in ("morph", "amp", "qrs_energy", "slope", "hf", "f1", "rmse")} for k in ref}}


def stage_analyze(args):
    train, val, test = split_subjects()
    CG.assert_no_test_subjects(train + val)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    folds = fold_mapping(train)
    if args.only_folds:
        keep = {int(v) for v in args.only_folds.split(",")}
        folds = {f: v for f, v in folds.items() if f in keep}
        print(f"[DEBUG] folds restricted to {sorted(folds)}")
    fit_by_fold = {f: [s for s in train if s not in ho] for f, ho in folds.items()}
    print(("=== PRIMARY: 12 train subjects, 4 folds x 3 held-out ==="), flush=True)
    rows, per_fold = run_setting("train_crossfit", fit_by_fold, folds, args)

    sec_rows = None
    if not args.skip_secondary:
        print("=== SECONDARY (design-informed): validation an0/k2s, whitener from the 12 train subjects ===", flush=True)
        sec_rows, _ = run_setting("val_secondary", {0: list(train)}, {0: list(val)}, args, secondary=True)

    dose_p = pool_rows(rows["dose"], ["B", "arm"], ["cost_reduction", "dR2_FULL", "dR2_QRS", "dR2_HF", "QRS_relative", "HF_relative"])
    proxy_p = pool_rows(rows["proxy"], ["B", "arm"], ["morph", "amp", "qrs_energy", "slope", "hf", "f1", "rmse", "morph_n", "n_windows"])
    regret_p = pool_rows(rows["regret"], ["B", "cost_q", "assignment_p"], ["regret", "overlap"])
    dim_p = pool_rows(rows["dim"], ["representation"], ["d_PR", "d90", "d95", "pca_k"])
    gate = gate_verdict(dose_p, proxy_p)

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prereg = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/X3_G0_COUPLING_GEOMETRY_PREREGISTRATION.md"],
                            cwd=ROOT, capture_output=True, text=True).stdout.strip()
    meta = json.loads((CACHE / "cache_meta.json").read_text())
    prov = {"repo_sha": git_sha, "preregistration_sha": prereg, "created": datetime.now().isoformat(timespec="seconds"),
            "script": "scripts/analyze_x3_g0_coupling_geometry.py", "subjects_loaded": sorted(train + val),
            "test_subjects_loaded": [], "wildppg_test_firewall": "enforced (assert_no_test_subjects)",
            "folds": {str(k): v for k, v in folds.items()}, "per_fold": {str(k): v for k, v in per_fold.items()},
            "cache_meta": meta, "protocol": {"B_values": list(B_VALUES), "pair_budget": PAIR_BUDGET,
            "assignment_seed": ASSIGN_SEED, "n_random_regret": N_RANDOM_REGRET, "n_perm": args.n_perm,
            "ridge_lambda": RIDGE_LAM, "pca_var": PCA_VAR, "pca_cap": PCA_CAP, "K_proxy": K_PROXY,
            "proxy_per_fold": PROXY_PER_FOLD, "pool_cap": POOL_CAP, "solver": "scipy exact linear_sum_assignment"}}
    for name, data in (("assignment_dose_response", dose_p), ("structural_proxy_metrics", proxy_p),
                       ("cross_objective_regret", regret_p), ("residual_dimension", dim_p),
                       ("source_residual_r2", rows["r2"]), ("assignment_overlap", rows["overlap"]),
                       ("permutation_null", rows["r2"]), ("secondary_validation", (sec_rows or {}).get("dose", []))):
        if data:
            keys = sorted({k for r in data for k in r})
            with open(OUT / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(data)
    (OUT / "gate_summary.json").write_text(json.dumps({"gate": gate, "dose_pooled": dose_p, "proxy_pooled": proxy_p,
                                                       "dimension_pooled": dim_p, "regret_pooled": regret_p,
                                                       "secondary": (sec_rows or {}).get("dose", [])}, indent=1, default=str))
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    (OUT / "protocol.json").write_text(json.dumps(prov["protocol"], indent=1))
    (OUT / "fold_mapping.json").write_text(json.dumps({str(k): {"held_out": v, "fit": fit_by_fold[k]} for k, v in folds.items()}, indent=1))
    _figures(dose_p, regret_p, proxy_p, dim_p, gate)
    print(f"\n===== X3-G0 VERDICT: {gate['verdict']} =====")
    print(json.dumps(gate["max_dR2"], indent=1))
    print("wrote", OUT)


def _figures(dose, regret, proxy, dim, gate):
    C = {"RAW": "tab:blue", "WHITE": "tab:orange", "HF": "tab:green", "RESID": "tab:red", "independent": "0.5"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, d in enumerate(DOMAINS):
        rr = [r for r in dim if r["representation"] == (d if d != "FULL" else "FULL")]
        ax[i].bar(["d_PR", "d90", "d95"], [rr[0]["d_PR"], rr[0]["d90"], rr[0]["d95"]] if rr else [0, 0, 0], color="tab:purple")
        ax[i].set_title(f"{d} residual effective dimension", fontsize=10)
        ax[i].grid(alpha=0.3)
    fig.suptitle("X3-G0 Fig. 1 - descriptive residual dimensionality (no feasibility threshold follows from this)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures/x3g0_fig1_residual_dimension.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, dom in enumerate(("QRS", "HF")):
        for a in PRIMARY_ARMS:
            rr = sorted([r for r in dose if r["arm"] == a and r["B"] > 1], key=lambda r: r["B"])
            if rr:
                ax[j].plot([r["B"] for r in rr], [r[f"dR2_{dom}"] for r in rr], "o-", color=C[a], label=a)
        ax[j].axhline(0.05, color="k", ls="--", lw=1, label="gate 0.05")
        ax[j].axhline(0.02, color="k", ls=":", lw=1, label="weak 0.02")
        ax[j].set_xscale("log", base=2)
        ax[j].set_xlabel("minibatch size B")
        ax[j].set_ylabel(f"delta R2 {dom}")
        ax[j].set_title(f"source->residual dependence, {dom}", fontsize=10)
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.suptitle("X3-G0 Fig. 2 - batch-size dose response by cost geometry (train cross-fit, permutation-null corrected)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures/x3g0_fig2_dose_response.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pair, st in (("RAW|WHITE", "o-"), ("RAW|HF", "s-"), ("WHITE|HF", "^-")):
        q, p = pair.split("|")
        rr = sorted([r for r in regret if r["cost_q"] == q and r["assignment_p"] == p], key=lambda r: r["B"])
        if rr:
            ax.plot([r["B"] for r in rr], [r["regret"] for r in rr], st, label=f"regret({q} <- {p})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("minibatch size B")
    ax.set_ylabel("cross-objective regret")
    ax.axhline(0, color="k", lw=0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("X3-G0 Fig. 3 - is the geometry really different, or just near-tie churn?", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "figures/x3g0_fig3_regret.png", dpi=120)
    plt.close(fig)

    ref = gate["references"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, metric in enumerate(("morph", "qrs_energy")):
        den = ref.get("iMF1", {}).get(metric, np.nan) - ref.get("A6", {}).get(metric, np.nan)
        for a in PRIMARY_ARMS:
            rr = sorted([r for r in proxy if r["arm"] == a], key=lambda r: r["B"])
            if rr and np.isfinite(den) and den > 1e-6:
                ax[j].plot([r["B"] for r in rr], [(r[metric] - ref["A6"][metric]) / den for r in rr], "o-", color=C[a], label=a)
        ax[j].axhline(0.20, color="k", ls="--", lw=1, label="gate 0.20")
        ax[j].axhline(0, color="k", lw=0.8)
        ax[j].set_xscale("log", base=2)
        ax[j].set_xlabel("minibatch size B")
        ax[j].set_ylabel(f"recovery fraction of A6->iMF1 gap ({metric})")
        ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.suptitle("X3-G0 Fig. 4 - structural recovery of the cross-fitted linear endpoint proxy (NOT an upper bound)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "figures/x3g0_fig4_recovery.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cache", "analyze"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--skip-secondary", action="store_true")
    # debug-only restrictions; defaults are the FROZEN protocol values
    ap.add_argument("--b-values", default=None, help="debug only: comma-separated subset of the frozen B grid")
    ap.add_argument("--only-folds", default=None, help="debug only: comma-separated subset of folds")
    args = ap.parse_args()
    if args.b_values:
        global B_VALUES
        B_VALUES = tuple(int(v) for v in args.b_values.split(","))
        print(f"[DEBUG] B restricted to {B_VALUES} (not a protocol change; frozen grid is {(1, 8, 32, 64, 128, 256, 512)})")
    (stage_cache if args.stage == "cache" else stage_analyze)(args)


if __name__ == "__main__":
    main()
