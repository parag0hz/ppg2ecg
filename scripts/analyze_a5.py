"""A5 analysis (pre-registered in docs/A5_CONDITIONAL_MEAN_CONTROL_PREREGISTRATION.md §8-12):
prediction-to-prediction similarity, residual / QRS-region analysis, RMSE-inversion ranking, quality-compute Pareto, verdict, figures.
Reads the frozen paired predictions of the generative arms and the A5 regressor predictions on identical test windows."""
from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FS = 128
QRS_HALF_MS = 100.0  # frozen: +/-100 ms around GT R-peaks
LF_HZ, HF_HZ = 5.0, 15.0
DATASETS = {
    "a5a": dict(label="PPG-DaLiA test S2", short="DaLiA-S2", aligned=False, reg="outputs/a5a_mse_regressor_dalia_testS2_seed42", ot="outputs/a0b_penguin_otcfm_ppgdalia_8s_seed42", imf="outputs/a2_imeanflow_s5_ppgdalia_8s_seed42", examples="outputs/a0_penguin_otcfm_ppgdalia_8s_seed42", processed="data/processed/v0_8s", test=["S2"]),
    "a5b": dict(label="PPG-DaLiA test S1", short="DaLiA-S1", aligned=False, reg="outputs/a5b_mse_regressor_dalia_testS1_seed42", ot="outputs/a3_otcfm_ppgdalia_testS1_seed42", imf="outputs/a3_imeanflow_ppgdalia_testS1_seed42", examples="outputs/a3_otcfm_ppgdalia_testS1_seed42", processed="data/processed/v0_8s", test=["S1"]),
    "a5c": dict(label="WildPPG test kjd, ssx (4096-window subset)", short="WildPPG", aligned=True, reg="outputs/a5c_mse_regressor_wildppg_seed42", ot="outputs/a4_otcfm_wildppg_seed42", imf="outputs/a4_imeanflow_wildppg_seed42", examples="outputs/a4_otcfm_wildppg_seed42", processed="data/processed/wildppg_8s", test=["kjd", "ssx"]),
}
MODELS = {"R": ("regressor", "reg", "regressor"), "O1": ("OT-CFM 1 NFE", "ot", "euler1"), "O50": ("OT-CFM 50 NFE", "ot", "heun25"), "M1": ("iMeanFlow 1 NFE", "imf", "meanflow1")}
COLORS = {"R": "tab:purple", "O50": "tab:blue", "O1": "tab:cyan", "M1": "tab:red"}
PARETO_ARMS = [("OT-CFM", "ot", "heun25", 50), ("OT-CFM", "ot", "heun10", 20), ("OT-CFM", "ot", "heun5", 10), ("OT-CFM", "ot", "heun2", 4), ("OT-CFM", "ot", "heun1", 2), ("OT-CFM", "ot", "euler1", 1), ("iMeanFlow", "imf", "meanflow1", 1), ("iMeanFlow", "imf", "meanflow2", 2), ("iMeanFlow", "imf", "meanflow4", 4), ("MSE regressor", "reg", "regressor", 1)]


def read_rows(d):
    with open(ROOT / d / "nfe_curve.csv") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        key = "regressor" if r["solver"] == "regressor" else f"{r['solver']}{int(float(r['solver_steps']))}"
        out[key] = {k: (float(v) if v not in ("", "None", "nan") and k not in ("solver",) else v) for k, v in r.items()}
    return out


def load_preds(cfg):
    ti = np.load(ROOT / cfg["reg"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    x, y, sid, starts = ti["x"], ti["y"], ti["sid"], ti["starts"]
    for arm in ("ot", "imf"):  # identical test windows (bit-exact)
        p = ROOT / cfg[arm] / "predictions" / "test_inputs.npz"
        if p.exists():
            o = np.load(p, allow_pickle=True)
            assert np.array_equal(o["x"], x) and np.array_equal(o["y"], y), f"test windows differ: {cfg[arm]}"
        else:  # A0-b / A2: whole test subject in processed order
            d = np.load(ROOT / cfg["processed"] / f"{cfg['test'][0]}.npz")
            assert np.array_equal(d["x"][: len(x)], x) and len(d["x"]) == len(x) and np.array_equal(d["y"], y), f"test windows differ from processed {cfg['test']}"
    preds = {}
    for m, (_, arm, fname) in MODELS.items():
        d = np.load(ROOT / cfg[arm] / "predictions" / f"{fname}.npz", allow_pickle=True)
        preds[m] = d["pred"].astype(np.float32)
        assert preds[m].shape == y.shape, (m, preds[m].shape, y.shape)
    rows = {m: read_rows(cfg[arm])[fname] for m, (_, arm, fname) in MODELS.items()}
    for m, (_, arm, fname) in MODELS.items():  # beats/reference: from the saved per-window beat counts (generative arms) or the eval row (regressor)
        if "beats_ratio" not in rows[m]:
            d = np.load(ROOT / cfg[arm] / "predictions" / f"{fname}.npz", allow_pickle=True)
            if "pw_n_pred_beats" in d and "pw_n_ref_beats" in d:
                rows[m]["beats_ratio"] = float(d["pw_n_pred_beats"].mean() / max(d["pw_n_ref_beats"].mean(), 1e-9))
            else:
                n_pred = np.array([len(R.detect_rpeaks(p, FS)) for p in preds[m]])
                n_ref = np.array([len(R.detect_rpeaks(t, FS)) for t in y])
                rows[m]["beats_ratio"] = float(n_pred.mean() / max(n_ref.mean(), 1e-9))
    for m in MODELS:
        r = rows[m]
        r["nan_convention"] = []
        for k, sub in (("morph_corr", 0.0), ("cond_gain_bpm", 0.0), ("rpeak_f1", 0.0), ("rpeak_precision", 0.0), ("rpeak_recall", 0.0)):
            if not np.isfinite(r[k]):
                r["nan_convention"].append(f"{k}: undefined (no detectable predicted beats) -> {sub}")
                r[k] = sub
    return x, y, sid, starts, preds, rows


def nankey(v):
    return (not np.isfinite(v), v)


def pcc_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)


def band_fractions(res):
    spec = np.abs(np.fft.rfft(res, axis=1)) ** 2
    f = np.fft.rfftfreq(res.shape[1], 1 / FS)
    tot = spec.sum(1) + 1e-12
    return float((spec[:, f < LF_HZ].sum(1) / tot).mean()), float((spec[:, f > HF_HZ].sum(1) / tot).mean())


def qrs_mask(y):
    half = int(round(QRS_HALF_MS / 1000 * FS))
    mask = np.zeros(y.shape, bool)
    n_peaks = 0
    for i in range(len(y)):
        for r in R.detect_rpeaks(y[i], FS):
            mask[i, max(0, r - half) : r + half + 1] = True
            n_peaks += 1
    return mask, n_peaks


def similarity(preds, rows, y):
    out = []
    keys = list(MODELS)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = preds[a] - preds[b]
            out.append(dict(model_a=a, model_b=b, rmse=float(np.sqrt((d**2).mean(1)).mean()), mae=float(np.abs(d).mean(1).mean()), pcc=float(np.nanmean(pcc_rows(preds[a], preds[b]))), d_amp=abs(rows[a]["amp_ratio"] - rows[b]["amp_ratio"]), d_morph=abs(rows[a]["morph_corr"] - rows[b]["morph_corr"]), d_hf=abs(rows[a]["hf_ratio_pred"] - rows[b]["hf_ratio_pred"])))
    return out


def closest_to_R(sim):
    cand = {r["model_b"]: r for r in sim if r["model_a"] == "R"}
    res = {"rmse_closest": min(cand, key=lambda m: cand[m]["rmse"]), "mae_closest": min(cand, key=lambda m: cand[m]["mae"]), "pcc_closest": max(cand, key=lambda m: cand[m]["pcc"])}
    votes = {m: 0 for m in cand}
    for k in ("d_amp", "d_morph", "d_hf"):
        w = min(cand, key=lambda m: cand[m][k])
        res[f"{k}_closest"] = w
        votes[w] += 1
    res["stat_votes"] = votes
    res["closest_O1_to_R"] = bool(res["rmse_closest"] == "O1" and votes["O1"] >= 2)
    return res


def residual_qrs(preds, y, mask):
    out = []
    for m, P in preds.items():
        res = P - y
        lf, hf = band_fractions(res)
        e2 = res**2
        out.append(dict(model=m, rmse_all=float(np.sqrt(e2.mean())), rmse_qrs=float(np.sqrt(e2[mask].mean())), rmse_nonqrs=float(np.sqrt(e2[~mask].mean())), resid_std=float(res.std(1).mean()), resid_lf_frac=lf, resid_hf_frac=hf, pred_std_qrs=float(P[mask].std()), pred_std_nonqrs=float(P[~mask].std()), gt_std_qrs=float(y[mask].std()), gt_std_nonqrs=float(y[~mask].std())))
    for r in out:
        r["qrs_over_nonqrs_rmse"] = r["rmse_qrs"] / max(r["rmse_nonqrs"], 1e-9)
        r["qrs_amp_ratio"] = r["pred_std_qrs"] / max(r["gt_std_qrs"], 1e-9)
        r["nonqrs_amp_ratio"] = r["pred_std_nonqrs"] / max(r["gt_std_nonqrs"], 1e-9)
    return out


def inversion_test(rows):
    keys = list(MODELS)
    rank = {k: sorted(keys, key=lambda m: nankey(rows[m][k])) for k in ("rmse", "mae")}
    rank["amp_dist_from_1"] = sorted(keys, key=lambda m: nankey(abs(rows[m]["amp_ratio"] - 1)))
    for k in ("morph_corr", "rpeak_f1", "cond_gain_bpm"):
        rank[k] = sorted(keys, key=lambda m: nankey(-rows[m][k]))
    rank["hr_abs_err_bpm"] = sorted(keys, key=lambda m: nankey(rows[m]["hr_abs_err_bpm"]))
    r_pos = rank["rmse"].index("R")
    yes = bool(r_pos <= 1 and rows["R"]["amp_ratio"] < rows["O50"]["amp_ratio"] - 0.1 and rows["R"]["morph_corr"] < rows["O50"]["morph_corr"] - 0.05)
    return {"ranks": rank, "regressor_rmse_rank": r_pos + 1, "regressor_inversion": yes, "ot1_inversion": bool(rows["O1"]["rmse"] < rows["O50"]["rmse"] and rows["O1"]["rmse"] < rows["M1"]["rmse"] and rows["O1"]["mae"] < rows["O50"]["mae"])}


def pareto(cfg):
    rows = {arm: read_rows(cfg[arm]) for arm in ("ot", "imf", "reg")}
    pts = []
    for fam, arm, key, nfe in PARETO_ARMS:
        r = rows[arm][key]
        pts.append(dict(family=fam, arm=key, nfe=nfe, generative_nfe=arm != "reg", latency_ms=r["latency_ms_batch64"], hr_abs_err_bpm=r["hr_abs_err_bpm"], morph_corr=r["morph_corr"], amp_abs_dev=abs(r["amp_ratio"] - 1), rmse=r["rmse"], rpeak_f1=r["rpeak_f1"], cond_gain_bpm=r["cond_gain_bpm"]))
    for p in pts:  # dominated if another point has <= NFE and better on all three quality axes (strictly better on one)
        dom = [q for q in pts if q is not p and q["nfe"] <= p["nfe"] and q["hr_abs_err_bpm"] <= p["hr_abs_err_bpm"] and q["morph_corr"] >= p["morph_corr"] and q["amp_abs_dev"] <= p["amp_abs_dev"] and (q["hr_abs_err_bpm"] < p["hr_abs_err_bpm"] or q["morph_corr"] > p["morph_corr"] or q["amp_abs_dev"] < p["amp_abs_dev"] or q["nfe"] < p["nfe"])]
        p["dominated_by"] = ";".join(f"{q['family']}:{q['arm']}" for q in dom)
        p["pareto_optimal"] = not dom
    return pts


def verdict_terms(rows, sim_close, aligned):
    att = bool(rows["R"]["amp_ratio"] < rows["O50"]["amp_ratio"] - 0.25 and rows["R"]["morph_corr"] < rows["O50"]["morph_corr"] - 0.10 and rows["R"]["rmse"] <= rows["O50"]["rmse"])
    t = {"attenuation_R": att, "closest_O1_to_R": sim_close["closest_O1_to_R"]}
    if aligned:
        t["wildppg_preserved_R"] = bool(rows["R"]["rpeak_f1"] >= 0.8 * rows["O50"]["rpeak_f1"] and rows["R"]["cond_gain_bpm"] >= 0.5 * rows["O50"]["cond_gain_bpm"])
    # hypotheses (directional readings, see report)
    t["H1"] = t["closest_O1_to_R"]
    if not aligned:
        t["H2"] = bool(rows["R"]["amp_ratio"] < 0.5 and (rows["R"]["beats_ratio"] < 0.5 or rows["R"]["hr_abs_err_bpm"] > 25) and rows["O1"]["amp_ratio"] < 0.5 and (rows["O1"]["beats_ratio"] < 0.5 or rows["O1"]["hr_abs_err_bpm"] > 25))
    else:
        t["H3"] = bool(att and t["wildppg_preserved_R"])
        t["H4"] = bool(rows["M1"]["morph_corr"] > rows["R"]["morph_corr"] + 0.1 and abs(rows["M1"]["amp_ratio"] - 1) < abs(rows["R"]["amp_ratio"] - 1) and ((np.isfinite(rows["R"]["rr_mae_ms"]) and rows["M1"]["rr_mae_ms"] > rows["R"]["rr_mae_ms"]) or rows["M1"]["rpeak_f1"] < rows["R"]["rpeak_f1"]))
    return t


def overall_verdict(terms):
    att = [t["attenuation_R"] for t in terms.values()]
    clo = [t["closest_O1_to_R"] for t in terms.values()]
    both = [a and c for a, c in zip(att, clo)]
    pres = terms.get("a5c", {}).get("wildppg_preserved_R", False)
    if all(both) and pres:
        return "STRONG SUPPORT"
    if clo.count(False) >= 2 or att.count(False) >= 2:
        return "NOT SUPPORTED"
    if sum(both) >= 2 or (all(both) and not pres):
        return "PARTIAL SUPPORT"
    return "NOT SUPPORTED"


def example_figure(cfg, x, y, sid, starts, preds, out_png, wins, title):
    order = ["R", "O50", "O1", "M1"]
    names = {"R": "MSE regressor (1 fwd)", "O50": "OT-CFM 50 NFE", "O1": "OT-CFM 1 NFE", "M1": "iMeanFlow 1 NFE"}
    t = np.arange(x.shape[1]) / FS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, axes = plt.subplots(2 + len(order), len(wins), figsize=(6.2 * len(wins), 2.0 * (2 + len(order))), sharex=True, sharey="row")
        axes = np.atleast_2d(axes).reshape(2 + len(order), len(wins))
        for c, w in enumerate(wins):
            axes[0, c].plot(t, x[w], color="tab:green", lw=0.9)
            axes[0, c].set_title(f"{sid[w]} window {w} · t = {starts[w]} s", fontsize=10)
            rg = R.detect_rpeaks(y[w], FS)
            axes[1, c].plot(t, y[w], "k", lw=0.9)
            axes[1, c].plot(rg / FS, y[w][rg], "r.", ms=6)
            for rr, m in enumerate(order, start=2):
                p = preds[m][w]
                rp = R.detect_rpeaks(p, FS)
                axes[rr, c].plot(t, y[w], color="0.8", lw=0.6)
                for r0 in rg:
                    axes[rr, c].axvspan((r0 - QRS_HALF_MS / 1000 * FS) / FS, (r0 + QRS_HALF_MS / 1000 * FS) / FS, color="0.92", zorder=0)
                axes[rr, c].plot(t, p, color=COLORS[m], lw=0.9)
                axes[rr, c].plot(rp / FS, p[rp], "r.", ms=6)
                axes[rr, c].set_ylim(-1.6, 1.6)
            for rr in range(2 + len(order)):
                axes[rr, c].grid(alpha=0.25)
            axes[-1, c].set_xlabel("time (s)")
        for rr, name in enumerate(["PPG (input)", "GT ECG"] + [names[m] for m in order]):
            axes[rr, 0].set_ylabel(name, fontsize=9)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default="artifacts/a5_conditional_mean_control")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sim_rows, qrs_rows, par_rows, summary = [], [], [], {"qrs_half_ms": QRS_HALF_MS, "lf_hz": LF_HZ, "hf_hz": HF_HZ, "datasets": {}}
    for key in args.datasets:
        cfg = DATASETS[key]
        x, y, sid, starts, preds, rows = load_preds(cfg)
        sim = similarity(preds, rows, y)
        close = closest_to_R(sim)
        mask, n_peaks = qrs_mask(y)
        rq = residual_qrs(preds, y, mask)
        inv = inversion_test(rows)
        par = pareto(cfg)
        terms = verdict_terms(rows, close, cfg["aligned"])
        for r in sim:
            sim_rows.append({"dataset": key, **r})
        for r in rq:
            qrs_rows.append({"dataset": key, **r})
        for r in par:
            par_rows.append({"dataset": key, **r})
        summary["datasets"][key] = {"label": cfg["label"], "aligned": cfg["aligned"], "n_windows": int(len(y)), "n_gt_rpeaks": n_peaks, "qrs_fraction_of_samples": float(mask.mean()), "metrics": {m: {"nan_convention": rows[m]["nan_convention"], **{k: rows[m][k] for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio", "cond_gain_bpm", "beats_ratio", "rmse", "mae", "pcc", "hf_ratio_pred", "hf_ratio_target", "rpeak_f1", "rpeak_precision", "rpeak_recall", "rr_mae_ms", "latency_ms_batch64", "actual_NFE")}} for m in MODELS}, "similarity": sim, "closest": close, "residual_qrs": rq, "inversion": inv, "terms": terms}
        # figures
        met_ex = json.loads((ROOT / cfg["examples"] / "metrics.json").read_text())["examples"]
        q_idx, f_idx = list(met_ex["ref_arm_hr_err_quantiles_10_50_90"]), list(met_ex["fixed_positions"])[:3]
        ttl = f"{cfg['label']}: MSE regressor vs OT-CFM 50/1 NFE vs iMeanFlow 1 NFE (grey = GT, shaded = ±{QRS_HALF_MS:.0f} ms QRS windows); identical y-scale"
        example_figure(cfg, x, y, sid, starts, preds, out / "figures" / f"{key}_examples_quantile.png", q_idx, ttl)
        example_figure(cfg, x, y, sid, starts, preds, out / "figures" / f"{key}_examples_fixed.png", f_idx, ttl)
        # similarity bars
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        cand = [r for r in sim if r["model_a"] == "R"]
        for ax, k, lab in zip(axes, ("rmse", "pcc", "d_amp"), ("waveform RMSE(R, ·)", "prediction PCC(R, ·)", "|amp(R) − amp(·)|")):
            ax.bar([r["model_b"] for r in cand], [r[k] for r in cand], color=[COLORS[r["model_b"]] for r in cand])
            ax.set_title(lab, fontsize=10)
            ax.grid(alpha=0.25, axis="y")
        fig.suptitle(f"{cfg['label']}: distance from the MSE regressor (R)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{key}_similarity.png", dpi=110)
        plt.close(fig)
        # QRS region bars
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ms = [r["model"] for r in rq]
        w = 0.38
        ax.bar(np.arange(len(ms)) - w / 2, [r["rmse_qrs"] for r in rq], w, label=f"QRS ±{QRS_HALF_MS:.0f} ms", color=[COLORS[m] for m in ms])
        ax.bar(np.arange(len(ms)) + w / 2, [r["rmse_nonqrs"] for r in rq], w, label="non-QRS", color=[COLORS[m] for m in ms], alpha=0.45)
        ax.set_xticks(range(len(ms)), ms)
        ax.set_ylabel("RMSE vs GT")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        ax.set_title(f"{cfg['label']}: QRS-region vs non-QRS RMSE" + ("" if cfg["aligned"] else " (DaLiA: not beat-synchronised → secondary)"), fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{key}_qrs_region.png", dpi=110)
        plt.close(fig)
        # Pareto
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, k, lab in zip(axes, ("hr_abs_err_bpm", "morph_corr", "amp_abs_dev"), ("HR abs error (bpm) ↓", "template correlation ↑", "|amplitude ratio − 1| ↓")):
            for fam, mk, col in (("OT-CFM", "o-", "tab:blue"), ("iMeanFlow", "s-", "tab:red"), ("MSE regressor", "D", "tab:purple")):
                ps = sorted([p for p in par if p["family"] == fam], key=lambda p: p["nfe"])
                ax.plot([p["nfe"] for p in ps], [p[k] for p in ps], mk, color=col, label=fam + ("" if fam != "MSE regressor" else " (1 fwd, not a generative NFE)"), ms=6)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("network evaluations")
            ax.set_title(lab, fontsize=10)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{cfg['label']}: quality vs compute", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{key}_pareto.png", dpi=110)
        plt.close(fig)
        print(f"[{key}] R: HR {rows['R']['hr_abs_err_bpm']:.2f} morph {rows['R']['morph_corr']:.3f} amp {rows['R']['amp_ratio']:.2f} gain {rows['R']['cond_gain_bpm']:.2f} RMSE {rows['R']['rmse']:.3f} F1 {rows['R']['rpeak_f1']:.3f} | closest-to-R: rmse {close['rmse_closest']} pcc {close['pcc_closest']} votes {close['stat_votes']} | terms {terms} | inversion(R) {inv['regressor_inversion']}")
    if len(args.datasets) == len(DATASETS):
        summary["overall_verdict"] = overall_verdict({k: v["terms"] for k, v in summary["datasets"].items()})
    for name, rowsx in (("cross_model_similarity.csv", sim_rows), ("qrs_region_analysis.csv", qrs_rows), ("pareto.csv", par_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsx[0].keys()))
            w.writeheader()
            w.writerows(rowsx)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("verdict:", summary.get("overall_verdict", "(partial run)"), "| wrote", out)


if __name__ == "__main__":
    main()
