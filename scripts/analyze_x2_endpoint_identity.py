"""X2 — Endpoint barycenter identity and source-noise cancellation in one-step conditional flow matching.

Frozen protocol: docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md (commit d2b0cff, pushed before any X2 real-data metric).
ANALYSIS ONLY: loads frozen checkpoints, runs new inference under torch.no_grad, writes exclusively to new X2 paths.
No training, no checkpoint or historical prediction is written or modified.

    F_k(c) = x0_k + v_theta(x0_k, c, t = 0)          (OT-CFM, exactly the frozen Euler-1 map, 1 NFE)
    F_k(c) = e_k  - u_theta(e_k, c, t = 1, h = 1)    (iMF-1, its own one-step rule and time convention)

Run: PYTHONPATH=src .venv/bin/python scripts/analyze_x2_endpoint_identity.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`)
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ppg2ecg.data.splits import read_manifest  # noqa: E402
from ppg2ecg.data.wildppg_sites import wildppg_clusters, wildppg_test_site_labels  # noqa: E402
from ppg2ecg.evaluation.source_sensitivity import cluster_bootstrap, jvp_sensitivity, pcc, rmse, source_bank, source_stats, unit_rms_directions  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5, sample_meanflow  # noqa: E402
from ppg2ecg.flow.samplers import euler_sample, heun_sample, nfe_of  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.models.regressor import REGRESSOR_MODELS  # noqa: E402
from ppg2ecg.utils.upstream import assert_upstream_pinned  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS, T_LEN = 128, 1024
K_SOURCE = 32                       # frozen source-bank size (prereg sec. 6)
K50 = 8                             # frozen multistep barycenter samples (prereg sec. 9)
N_SUBSET, N_DIR, DIR_SEED = 64, 4, 20260301   # frozen Jacobian / t-profile design (prereg sec. 8)
T_PROFILE = (0.00, 0.01, 0.05, 0.10)          # frozen exploratory oracle times (prereg sec. 12)
N_BOOT, BOOT_SEED = 2000, 0
QUALITATIVE = {"wildppg": [2439, 297, 415], "dalia_s2": [880, 482, 824], "dalia_s1": [1067, 726, 221]}
PROCESSED = {"wildppg": "data/processed/wildppg_8s", "dalia_s2": "data/processed/v0_8s", "dalia_s1": "data/processed/v0_8s"}
NFE_HEUN_STEPS = 25                 # canonical OT50 reference sampler: Heun 25 steps = 50 NFE


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            if limit and f.tell() > limit:
                break
    return h.hexdigest()


def load_ot(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = build_penguin_backbone(**ck["model_cfg"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


def load_imf(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0)).to(device).eval()
    net.load_state_dict(ck["state_dict"])
    return net, ck


def load_reg(ckpt: Path, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cls, _ = REGRESSOR_MODELS[ck.get("model_key", "state_token")]
    m = cls(**ck["model_cfg"]).to(device).eval()
    m.load_state_dict(ck["state_dict"])
    return m, ck


@torch.no_grad()
def endpoints(kind, model, ppg_np, x0, batch, device, nfe_log):
    """One-step endpoint map for every window. ppg_np [N,T] float32, x0 [N,1,T] -> [N,T] float32."""
    out = []
    for i in range(0, len(ppg_np), batch):
        ppg = torch.from_numpy(ppg_np[i : i + batch]).to(device)
        z = x0[i : i + batch].to(device)
        if kind == "ot":
            v = lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t)  # noqa: E731
            y, nfe = euler_sample(v, z, 1)  # exactly x0 + v(x0, c, t=0); t = float32 0.0
        else:
            y, nfe = sample_meanflow(model, ppg.unsqueeze(1), z, 1)  # e - u(e, c, t=1, h=1)
        nfe_log.add(int(nfe))
        out.append(y.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def ot50(model, ppg_np, x0, batch, device, nfe_log):
    out = []
    for i in range(0, len(ppg_np), batch):
        ppg = torch.from_numpy(ppg_np[i : i + batch]).to(device)
        v = lambda x, t: model.forward_step(x, ppg.unsqueeze(1), t)  # noqa: E731
        y, nfe = heun_sample(v, x0[i : i + batch].to(device), NFE_HEUN_STEPS)
        nfe_log.add(int(nfe))
        out.append(y.squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def regress(model, ppg_np, batch, device):
    out = []
    for i in range(0, len(ppg_np), batch):
        p = torch.from_numpy(ppg_np[i : i + batch]).to(device).unsqueeze(1)
        out.append(model(p).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def subset_indices(clusters, n, k=N_SUBSET):
    """Deterministic cluster-stratified subset (prereg sec. 8): equal share per cluster at evenly spaced positions."""
    labs = sorted(set(clusters.tolist()))
    per = k // len(labs)
    idx = []
    for c in labs:
        pos = np.where(clusters == c)[0]
        take = per if c != labs[-1] else k - per * (len(labs) - 1)
        sel = np.round(np.linspace(0, len(pos) - 1, take)).astype(int)
        idx += pos[sel].tolist()
    return np.array(sorted(set(idx)))


def summarize(v, clusters, name, ds, rows, boot=True):
    v = np.asarray(v, dtype=np.float64)
    fin = v[np.isfinite(v)]
    med, lo, hi = (cluster_bootstrap(v, clusters, N_BOOT, BOOT_SEED, np.nanmedian) if boot else (float(np.nanmedian(fin)), np.nan, np.nan))
    mean, mlo, mhi = (cluster_bootstrap(v, clusters, N_BOOT, BOOT_SEED, np.nanmean) if boot else (float(np.nanmean(fin)), np.nan, np.nan))
    row = {"dataset": ds, "quantity": name, "median": med, "median_ci_lo": lo, "median_ci_hi": hi, "mean": mean, "mean_ci_lo": mlo, "mean_ci_hi": mhi,
           "q25": float(np.nanpercentile(fin, 25)) if fin.size else np.nan, "q75": float(np.nanpercentile(fin, 75)) if fin.size else np.nan, "n": int(fin.size),
           "n_clusters": int(len(np.unique(clusters))), "bootstrap": "cluster(subject,site)" if ds == "wildppg" else "window(within-subject, descriptive)"}
    rows.append(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["wildppg", "dalia_s2", "dalia_s1"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--jvp-batch", type=int, default=16)
    ap.add_argument("--out", default="artifacts/x2_endpoint_identity")
    ap.add_argument("--tensor-out", default="outputs/x2_endpoint_identity")
    args = ap.parse_args()
    out, tout = ROOT / args.out, ROOT / args.tensor_out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    tout.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    up = assert_upstream_pinned()
    prov0 = json.loads((ROOT / "artifacts/x0_error_decomposition/prediction_provenance.json").read_text())
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    prereg_sha = subprocess.run(["git", "log", "-1", "--format=%H", "--", "docs/X2_ENDPOINT_IDENTITY_PREREGISTRATION.md"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    tables = {k: [] for k in ("source_sensitivity", "jacobian_sensitivity", "conditional_center_similarity", "same_model_barycenter", "t_profile_oracle", "clustered_bootstrap")}
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "repo_sha": git_sha, "preregistration_sha": prereg_sha, "upstream": up,
               "protocol": {"K_source": K_SOURCE, "source_seeds": list(range(K_SOURCE)), "K50": K50, "ot50_solver": f"heun:{NFE_HEUN_STEPS}", "ot50_nfe": nfe_of("heun", NFE_HEUN_STEPS),
                            "n_subset": N_SUBSET, "n_directions": N_DIR, "direction_seed": DIR_SEED, "t_profile": list(T_PROFILE), "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                            "qualitative_windows": QUALITATIVE, "batch_size": args.batch_size, "jvp_batch": args.jvp_batch, "dtype": "float32", "device": torch.cuda.get_device_name(0),
                            "torch": torch.__version__, "cuda": torch.version.cuda},
               "datasets": {}}
    provenance = {"repo_sha": git_sha, "preregistration_sha": prereg_sha, "script": "scripts/analyze_x2_endpoint_identity.py", "script_sha256": sha256(Path(__file__)),
                  "upstream": up, "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "created": datetime.now().isoformat(timespec="seconds"),
                  "note": "ANALYSIS ONLY: frozen checkpoints, new inference, nothing overwritten. Source bank seed 0 reproduces the historical evaluation draw.", "datasets": {}}

    for ds in args.datasets:
        rec = prov0["datasets"][ds]
        ti = np.load(ROOT / rec["test_inputs"], allow_pickle=True)
        ppg, gt, sid = ti["x"].astype(np.float32), ti["y"].astype(np.float64), np.asarray(ti["sid"]).astype(str)
        N = len(ppg)
        # ---- clusters (fail loudly; never silently fall back)
        if ds == "wildppg":
            split = read_manifest(ROOT / rec["split_manifest"])[0]
            site = wildppg_test_site_labels(ROOT / PROCESSED[ds], split["test"], N, ti["starts"])
            clusters = wildppg_clusters(sid, site)
        else:
            clusters = sid.copy()
        # prereg sec. 10: WildPPG resamples the 8 (subject, site) clusters; DaLiA has ONE held-out subject, so the
        # resampling unit there is the WINDOW (a descriptive within-subject interval, not cross-subject generalisation).
        boot_unit = clusters if ds == "wildppg" else np.arange(N).astype(str)
        sub = subset_indices(clusters, N)
        print(f"\n===== {ds}: N={N}, clusters={len(np.unique(clusters))}, subset={len(sub)} =====", flush=True)

        ck_paths = {m: ROOT / rec["models"][m]["checkpoint"] for m in ("OT1", "iMF1", "MSE")}
        ot_model, ot_ck = load_ot(ck_paths["OT1"], device)
        imf_model, imf_ck = load_imf(ck_paths["iMF1"], device)
        reg_model, reg_ck = load_reg(ck_paths["MSE"], device)
        nfe_log = set()

        # ---- frozen references (existing arrays; read-only)
        M_A6 = np.load(ROOT / rec["models"]["MSE"]["prediction_file"], allow_pickle=True)["pred"].astype(np.float64)
        OT50_0 = np.load(ROOT / rec["models"]["OT50"]["prediction_file"], allow_pickle=True)["pred"].astype(np.float64)
        iMF_0 = np.load(ROOT / rec["models"]["iMF1"]["prediction_file"], allow_pickle=True)["pred"].astype(np.float64)
        OT1_frozen = np.load(ROOT / rec["models"]["OT1"]["prediction_file"], allow_pickle=True)["pred"].astype(np.float64)
        # sanity: our regressor re-inference must reproduce the frozen MSE arm
        M_re = regress(reg_model, ppg, args.batch_size, device).astype(np.float64)
        mse_parity = float(np.abs(M_re - M_A6).max())

        # ---- source bank + endpoints for OT1 and iMF1 (K = 32); statistics computed in window chunks
        bank = [source_bank(k, N, T_LEN) for k in range(K_SOURCE)]
        res = {}
        for tag, kind, model in (("OT1", "ot", ot_model), ("iMF1", "imf", imf_model)):
            F = np.empty((K_SOURCE, N, T_LEN), dtype=np.float32)
            for k in range(K_SOURCE):
                F[k] = endpoints(kind, model, ppg, bank[k], args.batch_size, device, nfe_log)
                if k == 0:
                    print(f"  {tag} seed 0 done (nfe={sorted(nfe_log)})", flush=True)
            parts = []
            for i in range(0, N, 512):
                x0c = np.stack([bank[k][i : i + 512].squeeze(1).numpy() for k in range(K_SOURCE)])
                parts.append(source_stats(F[:, i : i + 512], x0c))
            stats = {q: np.concatenate([p[q] for p in parts]) for q in ("v_endpoint", "v_source", "r_source", "std_retention", "beta", "pair_rmse_endpoint", "pair_rmse_source", "d_pair")}
            stats["n_pairs"], stats["n_sources"] = parts[0]["n_pairs"], K_SOURCE
            res[tag] = {"F": F, "stats": stats, "Fbar": F.mean(0, dtype=np.float64)}
            s = res[tag]["stats"]
            print(f"  {tag}: R_source med {np.median(s['r_source']):.5f} (std-ret {np.median(s['std_retention']):.4f}) beta med {np.median(s['beta']):+.4f} D_pair med {np.median(s['d_pair']):.4f}", flush=True)
        parity = float(np.abs(res["OT1"]["F"][0] - OT1_frozen).max())
        parity_imf = float(np.abs(res["iMF1"]["F"][0] - iMF_0).max())
        print(f"  seed-0 parity vs frozen arrays: OT1 max|d| {parity:.2e}, iMF1 {parity_imf:.2e}, MSE {mse_parity:.2e}", flush=True)

        for tag in ("OT1", "iMF1"):
            s = res[tag]["stats"]
            tables["source_sensitivity"].append({"dataset": ds, "model": tag, "N": N, "K": K_SOURCE,
                "r_source_median": float(np.median(s["r_source"])), "r_source_q25": float(np.percentile(s["r_source"], 25)), "r_source_q75": float(np.percentile(s["r_source"], 75)), "r_source_mean": float(np.mean(s["r_source"])),
                "std_retention_median": float(np.median(s["std_retention"])), "std_retention_mean": float(np.mean(s["std_retention"])),
                "beta_median": float(np.median(s["beta"])), "beta_q25": float(np.percentile(s["beta"], 25)), "beta_q75": float(np.percentile(s["beta"], 75)), "beta_mean": float(np.mean(s["beta"])),
                "d_pair_median": float(np.median(s["d_pair"])), "d_pair_mean": float(np.mean(s["d_pair"])), "n_pairs": s["n_pairs"],
                "v_endpoint_median": float(np.median(s["v_endpoint"])), "v_source_median": float(np.median(s["v_source"])),
                "pair_rmse_endpoint_median": float(np.median(s["pair_rmse_endpoint"])), "pair_rmse_source_median": float(np.median(s["pair_rmse_source"]))})
            for q in ("r_source", "std_retention", "beta", "d_pair"):
                summarize(s[q], boot_unit, f"{tag}/{q}", ds, tables["clustered_bootstrap"])
        tables["source_sensitivity"].append({"dataset": ds, "model": "MSE(A6)", "N": N, "K": 0, "r_source_median": "N/A - deterministic model has no source latent",
                                             "std_retention_median": "N/A - deterministic model has no source latent", "beta_median": "N/A - deterministic model has no source latent",
                                             "d_pair_median": "N/A - deterministic model has no source latent"})

        # ---- H-X2-MEAN: conditional-center comparisons
        Fbar, F = res["OT1"]["Fbar"], res["OT1"]["F"]
        d_bar_M, d_bar_gt = rmse(Fbar, M_A6), rmse(Fbar, gt)
        d_bar_ot50, d_bar_imf = rmse(Fbar, OT50_0), rmse(Fbar, iMF_0)
        d_k_M = np.stack([rmse(F[k], M_A6) for k in range(K_SOURCE)])       # [K, N]
        q_mse = d_bar_M / np.maximum(np.median(d_k_M, axis=0), 1e-30)
        p_bar_M = pcc(Fbar, M_A6)
        cmp_all = {"M_A6": float(np.mean(d_bar_M)), "OT50_seed0": float(np.mean(d_bar_ot50)), "iMF1_seed0": float(np.mean(d_bar_imf)), "GT": float(np.mean(d_bar_gt))}
        m1 = all(cmp_all["M_A6"] < cmp_all[x] for x in ("OT50_seed0", "iMF1_seed0", "GT"))
        tables["conditional_center_similarity"].append({"dataset": ds, "N": N, "rmse_Fbar_M_A6": cmp_all["M_A6"], "rmse_Fbar_OT50seed0": cmp_all["OT50_seed0"], "rmse_Fbar_iMF1seed0": cmp_all["iMF1_seed0"], "rmse_Fbar_GT": cmp_all["GT"],
            "rmse_F0_M_A6": float(np.mean(rmse(F[0], M_A6))), "rmse_Fk_M_A6_median_over_k_mean": float(np.mean(np.median(d_k_M, axis=0))), "rmse_Fk_M_A6_spread_over_k": float(np.mean(d_k_M.std(0))),
            "pcc_Fbar_M_A6_mean": float(np.nanmean(p_bar_M)), "pcc_Fbar_M_A6_median": float(np.nanmedian(p_bar_M)), "pcc_F0_M_A6_mean": float(np.nanmean(pcc(F[0], M_A6))),
            "Q_MSE_median": float(np.median(q_mse)), "Q_MSE_mean": float(np.mean(q_mse)), "M1_closest_to_M_A6": bool(m1), "mse_reinference_max_abs_diff": mse_parity,
            "ot1_seed0_parity_max_abs_diff": parity, "imf1_seed0_parity_max_abs_diff": parity_imf})
        for nm, vv in (("Fbar_vs_M_A6_rmse", d_bar_M), ("Fbar_vs_M_A6_pcc", p_bar_M), ("Q_MSE", q_mse), ("Fbar_vs_GT_rmse", d_bar_gt)):
            summarize(vv, boot_unit, f"OT1/{nm}", ds, tables["clustered_bootstrap"])
        print(f"  H-X2-MEAN: RMSE(Fbar,·) M_A6 {cmp_all['M_A6']:.4f} | OT50 {cmp_all['OT50_seed0']:.4f} | iMF1 {cmp_all['iMF1_seed0']:.4f} | GT {cmp_all['GT']:.4f} -> M1 {m1}; PCC(Fbar,M_A6) {np.nanmean(p_bar_M):.3f}; Q_MSE med {np.median(q_mse):.4f}", flush=True)

        # ---- same-model multistep barycenter proxy B50 (subset only)
        ppg_s = ppg[sub]
        b50_samples = np.stack([ot50(ot_model, ppg_s, bank[k][sub], args.batch_size, device, nfe_log).astype(np.float64) for k in range(K50)])
        B50 = b50_samples.mean(0)
        V50 = float(np.mean(b50_samples.var(0, ddof=1)))
        def deb(a):
            return float(np.mean(((a - B50) ** 2).mean(-1)) - V50 / K50)
        tables["same_model_barycenter"].append({"dataset": ds, "n_subset": len(sub), "K50": K50, "ot50_nfe": nfe_of("heun", NFE_HEUN_STEPS), "V50_within_window_variance": V50,
            "rmse_Fbar_B50": float(np.mean(rmse(Fbar[sub], B50))), "rmse_M_A6_B50": float(np.mean(rmse(M_A6[sub], B50))), "rmse_F0_B50": float(np.mean(rmse(F[0][sub], B50))), "rmse_GT_B50": float(np.mean(rmse(gt[sub], B50))),
            "debiased_sqdist_Fbar_B50": deb(Fbar[sub]), "debiased_sqdist_M_A6_B50": deb(M_A6[sub]), "debiased_sqdist_F0_B50": deb(F[0][sub]), "debiased_sqdist_GT_B50": deb(gt[sub]),
            "pcc_Fbar_B50_mean": float(np.nanmean(pcc(Fbar[sub], B50))), "pcc_M_A6_B50_mean": float(np.nanmean(pcc(M_A6[sub], B50))),
            "rmse_Fbar_M_A6_on_subset": float(np.mean(rmse(Fbar[sub], M_A6[sub])))})
        print(f"  B50 (same-model multistep barycenter proxy, K50={K50}): RMSE Fbar {np.mean(rmse(Fbar[sub], B50)):.4f} | M_A6 {np.mean(rmse(M_A6[sub], B50)):.4f} | debiased sq-dist {deb(Fbar[sub]):+.5f} / {deb(M_A6[sub]):+.5f}", flush=True)

        # ---- local Jacobian rho_J (subset, seed-0 source, 4 unit-RMS directions)
        dirs = unit_rms_directions(len(sub), T_LEN, DIR_SEED, N_DIR)
        jvp_res = {}
        for tag, kind, model in (("OT1", "ot", ot_model), ("iMF1", "imf", imf_model)):
            rho, cos = [], []
            for i in range(0, len(sub), args.jvp_batch):
                idx = sub[i : i + args.jvp_batch]
                p = torch.from_numpy(ppg[idx]).to(device).unsqueeze(1)
                x = bank[0][idx].to(device)
                tz = torch.zeros(len(idx), 1, device=device)
                on = torch.ones(len(idx), 1, device=device)
                fn = (lambda z: z + model.forward_step(z, p, tz)) if kind == "ot" else (lambda z: z - model.u(z, p, on, on))
                for j in range(N_DIR):
                    r = jvp_sensitivity(fn, x, dirs[i : i + len(idx), j].to(device))
                    rho.append(r["rho_J"])
                    cos.append(r["cos_resp_negd"])
            rho = np.concatenate(rho)
            cos = np.concatenate(cos)
            unit_sub = boot_unit[sub]
            cl_dir = np.concatenate([unit_sub[i : i + min(args.jvp_batch, len(sub) - i)] for i in range(0, len(sub), args.jvp_batch) for _ in range(N_DIR)])
            jvp_res[tag] = {"rho": rho, "cos": cos, "clusters": cl_dir}
            tables["jacobian_sensitivity"].append({"dataset": ds, "model": tag, "n_windows": len(sub), "n_directions": N_DIR, "direction_seed": DIR_SEED, "source_seed": 0, "n_samples": int(rho.size),
                "rho_J_median": float(np.median(rho)), "rho_J_mean": float(np.mean(rho)), "rho_J_q25": float(np.percentile(rho, 25)), "rho_J_q75": float(np.percentile(rho, 75)),
                "cos_resp_negd_median": float(np.median(cos)), "cos_resp_negd_mean": float(np.mean(cos))})
            summarize(rho, cl_dir, f"{tag}/rho_J", ds, tables["clustered_bootstrap"])
            print(f"  rho_J {tag}: median {np.median(rho):.4f} [q25 {np.percentile(rho,25):.4f}, q75 {np.percentile(rho,75):.4f}] cos(Jv d, -d) med {np.median(cos):+.3f}", flush=True)

        # ---- exploratory ORACLE path-state t-profile (OT-CFM only; x_t uses the ground-truth target)
        gt_s = torch.from_numpy(gt[sub].astype(np.float32)).unsqueeze(1)
        prof = {}
        for t in T_PROFILE:
            G = np.empty((K_SOURCE, len(sub), T_LEN))
            for k in range(K_SOURCE):
                xk = bank[k][sub]
                xt = (1 - t) * xk + t * gt_s
                gg = []
                with torch.no_grad():
                    for i in range(0, len(sub), args.batch_size):
                        p = torch.from_numpy(ppg_s[i : i + args.batch_size]).to(device).unsqueeze(1)
                        z = xt[i : i + args.batch_size].to(device)
                        tv = torch.full((z.shape[0], 1), float(t), device=device)
                        gg.append((z + (1 - t) * ot_model.forward_step(z, p, tv)).squeeze(1).float().cpu().numpy())
                G[k] = np.concatenate(gg)
            S = (1 - t) * np.stack([bank[k][sub].squeeze(1).numpy().astype(np.float64) for k in range(K_SOURCE)])
            st = source_stats(G, S, pairwise=False) if t < 1 else None
            rho_t = []
            with torch.no_grad():
                for i in range(0, len(sub), args.jvp_batch):
                    idx = sub[i : i + args.jvp_batch]
                    p = torch.from_numpy(ppg[idx]).to(device).unsqueeze(1)
                    xt_b = ((1 - t) * bank[0][idx] + t * torch.from_numpy(gt[idx].astype(np.float32)).unsqueeze(1)).to(device)
                    tv = torch.full((len(idx), 1), float(t), device=device)
                    fn = lambda z: z + (1 - t) * ot_model.forward_step(z, p, tv)  # noqa: E731,B023
                    for j in range(N_DIR):
                        rho_t.append(jvp_sensitivity(fn, xt_b, dirs[i : i + len(idx), j].to(device))["rho_J"])
            rho_t = np.concatenate(rho_t)
            prof[t] = {"r_source": st["r_source"], "beta": st["beta"], "rho": rho_t}
            tables["t_profile_oracle"].append({"dataset": ds, "diagnostic": "ORACLE PATH-STATE DIAGNOSTIC (x_t uses the ground-truth target; not a realisable generation state)",
                "t": t, "n_windows": len(sub), "K": K_SOURCE, "r_source_median": float(np.median(st["r_source"])), "std_retention_median": float(np.median(np.sqrt(st["r_source"]))),
                "beta_median": float(np.median(st["beta"])), "rho_J_median": float(np.median(rho_t)), "rho_J_mean": float(np.mean(rho_t))})
            print(f"  [ORACLE t-profile] t={t:.2f}: R_source med {np.median(st['r_source']):.5f} beta med {np.median(st['beta']):+.4f} rho_J med {np.median(rho_t):.4f}", flush=True)

        # ---- verdict terms
        s_ot = res["OT1"]["stats"]
        c1, c2, c3 = float(np.median(s_ot["r_source"])) < 0.05, float(np.median(s_ot["beta"])) < 0.25, float(np.median(jvp_res["OT1"]["rho"])) < 0.25
        summary["datasets"][ds] = {
            "N": N, "n_clusters": int(len(np.unique(clusters))), "n_subset": int(len(sub)), "K_source": K_SOURCE,
            "parity": {"ot1_seed0_vs_frozen_euler1_max_abs": parity, "imf1_seed0_vs_frozen_meanflow1_max_abs": parity_imf, "mse_reinference_vs_frozen_max_abs": mse_parity, "nfe_observed": sorted(nfe_log)},
            "source_sensitivity": {tag: {"r_source_median": float(np.median(res[tag]["stats"]["r_source"])), "std_retention_median": float(np.median(res[tag]["stats"]["std_retention"])),
                                          "beta_median": float(np.median(res[tag]["stats"]["beta"])), "d_pair_median": float(np.median(res[tag]["stats"]["d_pair"]))} for tag in ("OT1", "iMF1")},
            "mse_proxy_source_sensitivity": "N/A - deterministic model has no source latent",
            "jacobian": {tag: {"rho_J_median": float(np.median(jvp_res[tag]["rho"])), "cos_median": float(np.median(jvp_res[tag]["cos"]))} for tag in ("OT1", "iMF1")},
            "conditional_center": {"rmse_Fbar": cmp_all, "M1_closest_to_M_A6": bool(m1), "pcc_Fbar_M_A6_mean": float(np.nanmean(p_bar_M)), "Q_MSE_median": float(np.median(q_mse))},
            "same_model_barycenter": tables["same_model_barycenter"][-1],
            "t_profile_oracle": {str(t): {"r_source_median": float(np.median(prof[t]["r_source"])), "beta_median": float(np.median(prof[t]["beta"])), "rho_J_median": float(np.median(prof[t]["rho"]))} for t in T_PROFILE},
            "H_X2_CANCEL": {"C1_r_source_lt_0.05": bool(c1), "C2_beta_lt_0.25": bool(c2), "C3_rho_J_lt_0.25": bool(c3), "pass": bool(c1 and c2 and c3)},
            "H_X2_MEAN_terms": {"M1": bool(m1), "M2_wildppg_pcc_ge_0.60": bool(np.nanmean(p_bar_M) >= 0.60) if ds == "wildppg" else None},
        }
        provenance["datasets"][ds] = {"N": N, "split_manifest": rec["split_manifest"], "test_inputs": rec["test_inputs"], "clusters": sorted(np.unique(clusters).tolist()),
            "subset_indices": sub.tolist(), "source_seeds": list(range(K_SOURCE)), "b50_seeds": list(range(K50)), "direction_seed": DIR_SEED,
            "checkpoints": {m: {"path": rec["models"][m]["checkpoint"], "sha256": rec["models"][m].get("checkpoint_sha256"), "epoch": rec["models"][m].get("checkpoint_epoch")} for m in ("OT1", "iMF1", "MSE")},
            "frozen_reference_predictions": {m: rec["models"][m]["prediction_file"] for m in ("OT1", "OT50", "iMF1", "MSE")},
            "nfe": {"OT1": 1, "iMF1": 1, "OT50": nfe_of("heun", NFE_HEUN_STEPS), "MSE": 1}, "observed_nfe": sorted(nfe_log)}

        np.savez_compressed(tout / f"{ds}_x2_tensors.npz", Fbar_ot1=res["OT1"]["Fbar"].astype(np.float32), Fbar_imf1=res["iMF1"]["Fbar"].astype(np.float32),
                            F_ot1_subset=res["OT1"]["F"][:, sub].astype(np.float32), F_imf1_subset=res["iMF1"]["F"][:, sub].astype(np.float32),
                            B50=B50.astype(np.float32), b50_samples=b50_samples.astype(np.float32), subset=sub, clusters=clusters)

        # ---- figures
        _figures(out, ds, ppg, gt, res, M_A6, B50, sub, jvp_res, prof, clusters, s_ot)
        del res, bank, F
        torch.cuda.empty_cache()

    # ---- verdict
    cancel = {ds: summary["datasets"][ds]["H_X2_CANCEL"]["pass"] for ds in args.datasets}
    m1_pass = sum(summary["datasets"][ds]["H_X2_MEAN_terms"]["M1"] for ds in args.datasets)
    m2 = summary["datasets"].get("wildppg", {}).get("H_X2_MEAN_terms", {}).get("M2_wildppg_pcc_ge_0.60")
    mean_pass = (m1_pass >= 2) and bool(m2)
    n_cancel = sum(cancel.values())
    verdict = "STRONG SUPPORT" if (n_cancel == len(args.datasets) and mean_pass) else ("NOT SUPPORTED" if n_cancel <= len(args.datasets) - 2 else "PARTIAL SUPPORT")
    summary["verdict"] = {"H_X2_CANCEL_per_condition": cancel, "H_X2_CANCEL_n_pass": n_cancel, "H_X2_MEAN_M1_n_pass": int(m1_pass), "H_X2_MEAN_M2_wildppg": m2, "H_X2_MEAN_pass": mean_pass, "overall": verdict}
    print(f"\n===== VERDICT: {verdict} | CANCEL {cancel} | MEAN M1 {m1_pass}/3, M2 {m2} =====", flush=True)

    for name, rows in tables.items():
        if rows:
            keys = sorted({k for r in rows for k in r})
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1, default=str))
    print("wrote", out)


def _figures(out, ds, ppg, gt, res, M_A6, B50, sub, jvp_res, prof, clusters, s_ot):
    fig_dir = out / "figures"
    C = {"OT1": "tab:cyan", "iMF1": "tab:red"}
    # Fig 1: source-noise cancellation at t = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, lab, logy) in zip(axes, [("r_source", "R_source = Var_k[F] / Var_k[x0]", True), ("std_retention", "std retention = sqrt(R_source)", True), ("beta", "source-retention slope beta", False)]):
        data = [res[t]["stats"][key] for t in ("OT1", "iMF1")]
        ax.boxplot(data, tick_labels=["OT-CFM-1", "iMF-1"], showfliers=False)
        for i, t in enumerate(("OT1", "iMF1")):
            ax.scatter(np.full(min(400, len(data[i])), i + 1) + np.linspace(-0.12, 0.12, min(400, len(data[i]))), data[i][:: max(1, len(data[i]) // 400)][: min(400, len(data[i]))], s=3, alpha=0.3, color=C[t])
        if logy:
            ax.set_yscale("log")
        ax.axhline(1.0, color="0.6", ls=":", lw=1)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].axhline(0.05, color="tab:green", ls="--", lw=1, label="frozen threshold 0.05")
    axes[2].axhline(0.25, color="tab:green", ls="--", lw=1, label="frozen threshold 0.25")
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle(f"X2 Fig. 1 — {ds}: source-noise dependence of the one-step endpoint map at t = 0\n(K = {K_SOURCE} sources; A6 MSE proxy: N/A - deterministic model has no source latent)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"x2_fig1_source_cancellation_{ds}.png", dpi=120)
    plt.close(fig)
    # Fig 2: frozen qualitative windows
    wins = [w for w in QUALITATIVE[ds] if w < len(gt)]
    tt = np.arange(gt.shape[1]) / FS
    fig, axes = plt.subplots(len(wins), 1, figsize=(14, 3.0 * len(wins)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, wins):
        ax.plot(tt, gt[w], color="0.75", lw=0.7, label="GT ECG")
        for k in range(0, 32, 4):
            ax.plot(tt, res["OT1"]["F"][k, w], color="tab:cyan", lw=0.5, alpha=0.55, label="OT-CFM-1 F_k (8 of 32 sources)" if k == 0 else None)
        ax.plot(tt, res["OT1"]["Fbar"][w], color="tab:blue", lw=1.4, label="Fbar (source-averaged)")
        ax.plot(tt, M_A6[w], color="tab:purple", lw=1.1, ls="--", label="A6 MSE proxy")
        if w in sub:
            ax.plot(tt, B50[list(sub).index(w)], color="tab:green", lw=1.0, ls=":", label="B50 same-model multistep barycenter proxy")
        ax.set_title(f"{ds} window {w}", fontsize=9)
        ax.set_ylabel("a.u.")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=5)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"X2 Fig. 2 — {ds}: do one-step endpoints from different sources collapse onto a common conditional center?\n(pre-registered frozen window IDs; no post-hoc example selection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / f"x2_fig2_traces_{ds}.png", dpi=120)
    plt.close(fig)
    # Fig 3: local Jacobian
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for t in ("OT1", "iMF1"):
        ax.hist(jvp_res[t]["rho"], bins=40, histtype="step", density=True, color=C[t], label=f"{t} (median {np.median(jvp_res[t]['rho']):.3f})")
    ax.axvline(0.25, color="tab:green", ls="--", lw=1, label="frozen threshold 0.25")
    ax.axvline(1.0, color="0.6", ls=":", lw=1, label="identity map (no cancellation)")
    ax.set_xlabel(r"$\rho_J = \|J_x F\,d\| / \|d\|$")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title(f"X2 Fig. 3 — {ds}: local Jacobian source sensitivity\n({N_SUBSET} windows x {N_DIR} unit-RMS directions, direction seed {DIR_SEED}, source seed 0)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / f"x2_fig3_jacobian_{ds}.png", dpi=120)
    plt.close(fig)
    # Fig 4: exploratory oracle t-profile
    ts = sorted(prof)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, lab in zip(axes, ("r_source", "beta", "rho"), ("R_source(t)", "beta(t)", r"$\rho_J(t)$")):
        med = [np.median(prof[t][key]) for t in ts]
        q1 = [np.percentile(prof[t][key], 25) for t in ts]
        q3 = [np.percentile(prof[t][key], 75) for t in ts]
        ax.plot(ts, med, "o-", color="tab:orange")
        ax.fill_between(ts, q1, q3, color="tab:orange", alpha=0.2)
        ax.set_xlabel("t (path-state time; t = 0 is the source endpoint)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
        if key == "r_source":
            ax.set_yscale("log")
    fig.suptitle(f"X2 Fig. 4 — {ds}: EXPLORATORY ORACLE PATH-STATE DIAGNOSTIC\n(x_t is built from the ground-truth target; these are NOT realisable one-step generation states)", fontsize=10, color="darkred")
    fig.tight_layout()
    fig.savefig(fig_dir / f"x2_fig4_t_profile_oracle_{ds}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
