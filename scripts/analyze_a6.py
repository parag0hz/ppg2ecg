"""A6 analysis (docs/A6_CAPACITY_MATCHED_MEAN_CONTROL_PREREGISTRATION.md §7-9): Rfull (A6) vs Rsmall (A5), O1, O50, M1 on identical
test windows — GT metrics, prediction distances, QRS-region RMSE, parameter-parity table, frozen verdict. Reuses analyze_a5 helpers."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_a5 as A5  # noqa: E402

A6 = {"a6a": dict(a5="a5a", label="PPG-DaLiA test S2", aligned=False, reg="outputs/a6a_fullbackbone_mse_dalia_testS2_seed42"), "a6b": dict(a5="a5b", label="PPG-DaLiA test S1", aligned=False, reg="outputs/a6b_fullbackbone_mse_dalia_testS1_seed42"), "a6c": dict(a5="a5c", label="WildPPG test kjd, ssx (4096-window subset)", aligned=True, reg="outputs/a6c_fullbackbone_mse_wildppg_seed42")}
MODELS6 = {"Rfull": ("A6 full-backbone MSE", "reg6", "regressor"), "Rsmall": ("A5 regressor", "reg", "regressor"), "O1": ("OT-CFM 1 NFE", "ot", "euler1"), "O50": ("OT-CFM 50 NFE", "ot", "heun25"), "M1": ("iMeanFlow 1 NFE", "imf", "meanflow1")}
COLORS = {"Rfull": "tab:orange", "Rsmall": "tab:purple", "O50": "tab:blue", "O1": "tab:cyan", "M1": "tab:red"}


def load(key):
    cfg = {**A5.DATASETS[A6[key]["a5"]], "reg6": A6[key]["reg"]}
    ti = np.load(ROOT / cfg["reg6"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    x, y, sid, starts = ti["x"], ti["y"], ti["sid"], ti["starts"]
    o = np.load(ROOT / cfg["reg"] / "predictions" / "test_inputs.npz", allow_pickle=True)
    assert np.array_equal(o["x"], x) and np.array_equal(o["y"], y)
    preds, rows = {}, {}
    for m, (_, arm, fname) in MODELS6.items():
        d = np.load(ROOT / cfg[arm] / "predictions" / f"{fname}.npz", allow_pickle=True)
        preds[m] = d["pred"].astype(np.float32)
        assert preds[m].shape == y.shape
        rows[m] = A5.read_rows(cfg[arm])[fname]
        if "beats_ratio" not in rows[m]:
            rows[m]["beats_ratio"] = float(d["pw_n_pred_beats"].mean() / max(d["pw_n_ref_beats"].mean(), 1e-9))
        rows[m]["nan_convention"] = []
        for k, sub in (("morph_corr", 0.0), ("cond_gain_bpm", 0.0), ("rpeak_f1", 0.0), ("rpeak_precision", 0.0), ("rpeak_recall", 0.0)):
            if not np.isfinite(rows[m][k]):
                rows[m]["nan_convention"].append(f"{k}: undefined -> {sub}")
                rows[m][k] = sub
    return cfg, x, y, sid, starts, preds, rows


def similarity(preds, rows, ref="Rfull"):
    out = []
    for b in [m for m in MODELS6 if m != ref]:
        d = preds[ref] - preds[b]
        out.append(dict(model_a=ref, model_b=b, rmse=float(np.sqrt((d**2).mean(1)).mean()), mae=float(np.abs(d).mean(1).mean()), pcc=float(np.nanmean(A5.pcc_rows(preds[ref], preds[b]))), d_amp=abs(rows[ref]["amp_ratio"] - rows[b]["amp_ratio"]), d_morph=abs(rows[ref]["morph_corr"] - rows[b]["morph_corr"]), d_hf=abs(rows[ref]["hf_ratio_pred"] - rows[b]["hf_ratio_pred"])))
    return out


def closest(sim):
    cand = {r["model_b"]: r for r in sim if r["model_b"] in ("O1", "O50", "M1")}  # generative candidates only (Rsmall reported separately)
    votes = {m: 0 for m in cand}
    res = {"rmse_closest": min(cand, key=lambda m: cand[m]["rmse"]), "pcc_closest": max(cand, key=lambda m: cand[m]["pcc"])}
    for k in ("d_amp", "d_morph", "d_hf"):
        votes[min(cand, key=lambda m: cand[m][k])] += 1
    res["stat_votes"] = votes
    res["closest_O1"] = bool(res["rmse_closest"] == "O1" and votes["O1"] >= 2)
    return res


def terms(rows, close, aligned):
    R, O50 = rows["Rfull"], rows["O50"]
    t = {"attenuation_Rfull": bool(R["amp_ratio"] < O50["amp_ratio"] - 0.25 and R["morph_corr"] < O50["morph_corr"] - 0.10 and R["rmse"] <= O50["rmse"]), "closest_O1_to_Rfull": close["closest_O1"]}
    t["morph_recovered_vs_O50"] = bool(R["morph_corr"] >= O50["morph_corr"] - 0.10 or R["amp_ratio"] >= O50["amp_ratio"] - 0.25)
    if aligned:
        t["wildppg_preserved_Rfull"] = bool(R["rpeak_f1"] >= 0.8 * O50["rpeak_f1"] and R["cond_gain_bpm"] >= 0.5 * O50["cond_gain_bpm"])
    t["H6_1"] = bool(t["attenuation_Rfull"] and R["hf_ratio_pred"] < 0.5 * rows["O50"]["hf_ratio_pred"])
    t["H6_2"] = t["closest_O1_to_Rfull"]
    if aligned:
        t["H6_3"] = t["wildppg_preserved_Rfull"]
    return t


def verdict(all_terms):
    att = [t["attenuation_Rfull"] for t in all_terms.values()]
    clo = [t["closest_O1_to_Rfull"] for t in all_terms.values()]
    rec = [t["morph_recovered_vs_O50"] for t in all_terms.values()]
    pres = all_terms.get("a6c", {}).get("wildppg_preserved_Rfull", True)
    if all(att) and all(clo) and pres:
        return "CAPACITY OBJECTION RESOLVED"
    if sum(rec) >= 2:
        return "CAPACITY SENSITIVE"
    return "MIXED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(A6))
    ap.add_argument("--out", default="artifacts/a6_capacity_control")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ppg2ecg.models import build_penguin_backbone, count_params
    from ppg2ecg.models.regressor import S5ConditionalMeanRegressor, S5FullBackboneRegressor, count_full_backbone_params, count_regressor_params

    cg = count_params(build_penguin_backbone(n_step=25), exclude_prefixes=("cross_attn", "revin"))
    parity = [{"model": "PENGUIN OT-CFM", "total": cg["total"], "effective": cg["effective"]}, {"model": "iMeanFlow (same backbone)", "total": cg["total"], "effective": cg["effective"]}]
    c5, c6 = count_regressor_params(S5ConditionalMeanRegressor()), count_full_backbone_params(S5FullBackboneRegressor())
    parity += [{"model": "A5 regressor (state token)", "total": c5["total"], "effective": c5["effective"]}, {"model": "A6 full-backbone MSE", "total": c6["total"], "effective": c6["effective"]}]
    sim_rows, qrs_rows, summary = [], [], {"parameter_parity": parity, "datasets": {}}
    for key in args.datasets:
        cfg, x, y, sid, starts, preds, rows = load(key)
        sim = similarity(preds, rows)
        close = closest(sim)
        mask, n_peaks = A5.qrs_mask(y)
        rq = A5.residual_qrs(preds, y, mask)
        t = terms(rows, close, A6[key]["aligned"])
        rank = {k: sorted(MODELS6, key=lambda m: A5.nankey(rows[m][k])) for k in ("rmse", "mae")}
        rank["morph_corr"] = sorted(MODELS6, key=lambda m: A5.nankey(-rows[m]["morph_corr"]))
        rank["amp_dist_from_1"] = sorted(MODELS6, key=lambda m: A5.nankey(abs(rows[m]["amp_ratio"] - 1)))
        r6, r5 = rows["Rfull"], rows["Rsmall"]
        delta = {k: r6[k] - r5[k] for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio", "cond_gain_bpm", "rmse", "mae", "hf_ratio_pred", "rpeak_f1", "beats_ratio")}
        sim_rows += [{"dataset": key, **r} for r in sim]
        qrs_rows += [{"dataset": key, **r} for r in rq]
        tr6 = json.loads((ROOT / cfg["reg6"] / "training_summary.json").read_text())
        summary["datasets"][key] = {"label": A6[key]["label"], "aligned": A6[key]["aligned"], "n_windows": int(len(y)), "training": {k: tr6[k] for k in ("epochs_run", "best_epoch", "best_selection_metric", "total_train_time_s", "peak_mem_MiB")}, "model_cfg": json.loads((ROOT / cfg["reg6"] / "train_meta.json").read_text()).get("model_cfg"), "metrics": {m: {k: rows[m][k] for k in ("hr_abs_err_bpm", "morph_corr", "amp_ratio", "cond_gain_bpm", "beats_ratio", "rmse", "mae", "pcc", "hf_ratio_pred", "rpeak_f1", "rr_mae_ms", "latency_ms_batch64")} | {"nan_convention": rows[m]["nan_convention"]} for m in MODELS6}, "similarity_from_Rfull": sim, "closest": close, "residual_qrs": rq, "ranks": rank, "Rfull_minus_Rsmall": delta, "terms": t}
        # figure: examples (same pre-registered windows), rows PPG/GT/Rfull/Rsmall/O50/O1/M1
        met_ex = json.loads((ROOT / cfg["examples"] / "metrics.json").read_text())["examples"]
        wins = list(met_ex["ref_arm_hr_err_quantiles_10_50_90"])
        order = ["Rfull", "Rsmall", "O50", "O1", "M1"]
        tgrid = np.arange(x.shape[1]) / A5.FS
        import warnings

        from ppg2ecg.evaluation import rpeaks as R

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, axes = plt.subplots(2 + len(order), len(wins), figsize=(6.2 * len(wins), 1.9 * (2 + len(order))), sharex=True, sharey="row")
            for c, w in enumerate(wins):
                axes[0, c].plot(tgrid, x[w], color="tab:green", lw=0.9)
                axes[0, c].set_title(f"{sid[w]} window {w} · t = {starts[w]} s", fontsize=10)
                rg = R.detect_rpeaks(y[w], A5.FS)
                axes[1, c].plot(tgrid, y[w], "k", lw=0.9)
                axes[1, c].plot(rg / A5.FS, y[w][rg], "r.", ms=6)
                for rr, m in enumerate(order, start=2):
                    p = preds[m][w]
                    rp = R.detect_rpeaks(p, A5.FS)
                    axes[rr, c].plot(tgrid, y[w], color="0.8", lw=0.6)
                    axes[rr, c].plot(tgrid, p, color=COLORS[m], lw=0.9)
                    axes[rr, c].plot(rp / A5.FS, p[rp], "r.", ms=6)
                    axes[rr, c].set_ylim(-1.6, 1.6)
                for rr in range(2 + len(order)):
                    axes[rr, c].grid(alpha=0.25)
                axes[-1, c].set_xlabel("time (s)")
            for rr, name in enumerate(["PPG (input)", "GT ECG"] + [MODELS6[m][0] for m in order]):
                axes[rr, 0].set_ylabel(name, fontsize=8)
            fig.suptitle(f"{A6[key]['label']}: capacity-matched MSE control (A6) vs A5 regressor vs OT-CFM 50/1 vs iMF-1 (grey = GT)", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / "figures" / f"{key}_examples.png", dpi=110)
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, k, lab in zip(axes, ("rmse", "pcc", "d_amp"), ("waveform RMSE(Rfull, ·)", "PCC(Rfull, ·)", "|amp(Rfull) − amp(·)|")):
            ax.bar([r["model_b"] for r in sim], [r[k] for r in sim], color=[COLORS[r["model_b"]] for r in sim])
            ax.set_title(lab, fontsize=10)
            ax.grid(alpha=0.25, axis="y")
        fig.suptitle(f"{A6[key]['label']}: distance from the capacity-matched MSE control", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"{key}_similarity.png", dpi=110)
        plt.close(fig)
        print(f"[{key}] Rfull: HR {r6['hr_abs_err_bpm']:.2f} morph {r6['morph_corr']:.3f} amp {r6['amp_ratio']:.3f} gain {r6['cond_gain_bpm']:.2f} RMSE {r6['rmse']:.3f} HF {r6['hf_ratio_pred']:.3f} F1 {r6['rpeak_f1']:.3f} | Rsmall: morph {r5['morph_corr']:.3f} amp {r5['amp_ratio']:.3f} RMSE {r5['rmse']:.3f} | closest {close} | terms {t}")
    if len(args.datasets) == len(A6):
        summary["verdict"] = verdict({k: v["terms"] for k, v in summary["datasets"].items()})
    for name, rowsx in (("cross_model_similarity.csv", sim_rows), ("qrs_region_analysis.csv", qrs_rows), ("parameter_parity.csv", parity)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsx[0].keys()))
            w.writeheader()
            w.writerows(rowsx)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("verdict:", summary.get("verdict", "(partial)"), "| wrote", out)


if __name__ == "__main__":
    main()
