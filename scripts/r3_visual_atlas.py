"""R3 visual atlas — prereg 3d779fc section 22. V1 validation VIZ cohort (64 windows), V1 32-row batch
construction, nine rows (PPG, R1 scaffold, GT ECG, B, ADD, TF-TRUE, GTF-TRUE, GTF-CONST, GTF-ORACLE at NFE 4),
gate g(t) overlaid on the GTF rows, GT-R-centred zoom. GT R is a vertical reference only; nothing is shifted.
Counts only (per-arm missing / spurious, per-window S4/S5 vs B, gate in weak vs strong scaffold segments).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M1
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import v1_timing as V
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.flow.imeanflow import MeanFlowS5

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r3_rhythm_fusion"
ATLAS = ART / "visual_atlas"
FS, T_LEN, SRC_SEED, NFE = 128, 1024, 0, 4
ROW_ARMS = ("B", "ADD", "TF-TRUE", "GTF-TRUE", "GTF-CONST", "GTF-ORACLE")
COL = {"B": "tab:blue", "ADD": "tab:brown", "TF-TRUE": "tab:green", "GTF-TRUE": "tab:red", "GTF-CONST": "tab:orange", "GTF-ORACLE": "tab:purple"}
ZOOM_LO, ZOOM_HI = 38, 64


def events_annot(gt_pk, pred):
    pk = R.detect_rpeaks(pred.astype(np.float64), FS)
    m, _, _ = R.match_rpeaks(gt_pk, pk, FS, 50.0)
    mg, mp = {i for i, _ in m}, {j for _, j in m}
    return pk, [gt_pk[i] for i in range(len(gt_pk)) if i not in mg], [pk[j] for j in range(len(pk)) if j not in mp]


@torch.no_grad()
def main() -> int:
    ER.assert_no_test_subjects(V.VAL)
    ATLAS.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ck = torch.load(ROOT / RT.GENERATOR_CKPT, map_location="cpu", weights_only=False); cfg = ck.get("imf_cfg", {})
    assert RT.state_dict_sha256(ck["state_dict"]) == RT.EXPECTED_GENERATOR_STATE_SHA
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0)).to(dev).eval(); base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    net_add, _c, _g = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev); net_add.requires_grad_(False)
    add_sd = torch.load(ROOT / RF.R2_ADD_CKPT, map_location="cpu", weights_only=False)["state_dict"]; net_add.rhythm_adapter.load_state_dict({k: v.to(dev) for k, v in add_sd.items()})
    tcn, _t = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    h_dim = int(ck["model_cfg"]["h_dim"])
    nets = {}
    for arm, fam, gm in (("TF-TRUE", "tf", None), ("GTF-TRUE", "gtf", "adaptive"), ("GTF-CONST", "gtf", "const"), ("GTF-ORACLE", "gtf", "adaptive")):
        m = RF.build_r3_module(fam, gm, c_hidden=h_dim); n = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), m, cond_mode=cfg.get("cond_mode", "h_only"), h_scale=cfg.get("h_scale", 1.0))
        missing, unexpected = n.load_state_dict(ck["state_dict"], strict=False); assert unexpected == [] and set(missing) == {"r3." + k for k in RF.FAMILY_PARAM_NAMES[fam]}
        n = n.to(dev).eval(); n.requires_grad_(False)
        mk = torch.load(ROOT / f"outputs/r3_{arm.lower().replace('-', '_')}_seed42/module_step{RF.STEPS}.pt", map_location="cpu", weights_only=False)
        assert mk["step"] == RF.STEPS and mk["arm"] == arm.lower().replace("-", "_") and mk["generator_state_sha256"] == RT.EXPECTED_GENERATOR_STATE_SHA and mk["probe_hash"] == RF.R2_PROBE_HASH
        n.r3.load_state_dict({k: v.to(dev) for k, v in mk["state_dict"].items()}); nets[arm] = n
    t = np.arange(T_LEN) / FS
    def lab(arm):
        return f"{arm} {RF.ORACLE_LABEL}" if arm in RF.ORACLE_ARMS else arm
    summary = {lab(arm): {"n_gt": 0, "n_missing": 0, "n_spurious": 0, "windows_with_spurious": 0, "windows_with_missing": 0, "windows_S4_lower_than_B": 0, "windows_S5_lower_than_B": 0,
                     "windows_f1_higher_than_B": 0, "windows_f1_lower_than_B": 0, "windows_spurious_more_than_B": 0, "windows_missing_fewer_than_B": 0} for arm in ROW_ARMS}
    gate_weak, gate_strong, index_rows = [], [], []
    by_site = {site: {lab(arm): {"n_missing": 0, "n_spurious": 0, "windows_f1_higher_than_B": 0, "windows_f1_lower_than_B": 0, "windows_S4_lower_than_B": 0} for arm in ROW_ARMS} for site in V.SITES}

    def run(arm, X32, s32, e0):
        pp = torch.from_numpy(X32).to(dev).unsqueeze(1); e = e0.to(dev)
        if arm == "B":
            z, _ = ER.sample_meanflow_schedule(base, pp, e, ER.UNIFORM[NFE])
        elif arm == "ADD":
            z, _ = ER.sample_meanflow_schedule(net_add, RT.make_ppg2(pp, torch.from_numpy(s32).to(dev).unsqueeze(1)), e, ER.UNIFORM[NFE])
        else:
            z, _ = ER.sample_meanflow_schedule(nets[arm], RT.make_ppg2(pp, torch.from_numpy(s32).to(dev).unsqueeze(1)), e, ER.UNIFORM[NFE])
        return z.squeeze(1).float().cpu().numpy()

    for sub in V.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz"); Xs, Ys, WIs = d["x"], d["y"], d["window_index"]
        c = V.cohorts(sub, d["site"], WIs)
        for site in V.SITES:
            idx = c[site]["metrics"]; X32, Y32 = Xs[idx].astype(np.float32), Ys[idx].astype(np.float64)
            e0 = torch.randn(len(X32), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
            vizpos = [int(np.flatnonzero(idx == v)[0]) for v in c[site]["viz"]]
            s_true = RT.scaffold_from_ppg(tcn, torch.from_numpy(X32).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy(); s_orc = RT.oracle_fields(Y32, workers=1)
            preds = {arm: run(arm, X32, s_orc if arm == "GTF-ORACLE" else s_true, e0) for arm in ROW_ARMS}
            g_true = nets["GTF-TRUE"].r3.gate_values(torch.from_numpy(s_true).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy()
            g_const = nets["GTF-CONST"].r3.gate_values(torch.from_numpy(s_true).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy()
            g_orc = nets["GTF-ORACLE"].r3.gate_values(torch.from_numpy(s_orc).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy()
            for vp in vizpos:
                wi = int(WIs[idx[vp]]); gt = Y32[vp]; gt_pk = R.detect_rpeaks(gt, FS); ppg = X32[vp]
                weak = s_true[vp] < 0.35; gate_weak += g_true[vp][weak].tolist(); gate_strong += g_true[vp][~weak].tolist()
                fig, axes = plt.subplots(9, 1, figsize=(13, 16), sharex=True)
                axes[0].plot(t, ppg, "k", lw=0.8); axes[0].set_ylabel("PPG", fontsize=9)
                axes[1].plot(t, s_true[vp], color="tab:green", lw=1.2); axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("R1 scaffold", fontsize=9)
                for r_ in gt_pk:
                    axes[1].text(r_ / FS, 1.02, f"{float(s_true[vp][int(r_)]):.2f}", fontsize=6, ha="center", color="darkgreen")
                axes[2].plot(t, gt, "k", lw=0.8); axes[2].plot(gt_pk / FS, gt[gt_pk], "r.", ms=5); axes[2].set_ylabel("GT ECG", fontsize=9)
                qmB = M1.qrs_core_morphology(preds["B"][vp].astype(np.float64), gt, gt_pk); _, missB, spurB = events_annot(gt_pk, preds["B"][vp])
                for k_, arm in enumerate(ROW_ARMS):
                    ax = axes[3 + k_]; pred = preds[arm][vp]; pk, miss, spur = events_annot(gt_pk, pred)
                    ax.plot(t, pred, color=COL[arm], lw=0.8)
                    if len(pk):
                        ax.plot(pk / FS, pred[pk], "o", color="darkorange", ms=3.5)
                    for m_ in miss:
                        ax.plot(m_ / FS, 1.25, "x", color="red", ms=6)
                    for s_ in spur:
                        ax.plot(s_ / FS, -1.25, "^", color="magenta", ms=5)
                    if arm.startswith("GTF"):
                        g = {"GTF-TRUE": g_true, "GTF-CONST": g_const, "GTF-ORACLE": g_orc}[arm][vp]
                        ax.plot(t, g * 2.4 - 1.3, color="gray", ls=":", lw=1.0)
                    ax.set_ylabel(f"{lab(arm)}\nNFE {NFE}", fontsize=7); ax.set_ylim(-1.4, 1.4)
                    qm = M1.qrs_core_morphology(pred.astype(np.float64), gt, gt_pk)
                    ax.text(0.005, 0.85, f"beats {len(pk)}/{len(gt_pk)} missing {len(miss)} spurious {len(spur)} | S4 {qm['qrs_deriv_rmse']:.3f} S5 {qm['qrs_curvature_err']:.3f}" + ("  (gate dotted)" if arm.startswith("GTF") else ""), transform=ax.transAxes, fontsize=7)
                    S_ = summary[lab(arm)]; B_ = by_site[site][lab(arm)]
                    S_["n_gt"] += len(gt_pk); S_["n_missing"] += len(miss); S_["n_spurious"] += len(spur); S_["windows_with_spurious"] += int(len(spur) > 0); S_["windows_with_missing"] += int(len(miss) > 0)
                    B_["n_missing"] += len(miss); B_["n_spurious"] += len(spur)
                    if arm != "B":
                        s4 = int(qm["qrs_deriv_rmse"] < qmB["qrs_deriv_rmse"]); S_["windows_S4_lower_than_B"] += s4; B_["windows_S4_lower_than_B"] += s4
                        S_["windows_S5_lower_than_B"] += int(qm["qrs_curvature_err"] < qmB["qrs_curvature_err"])
                        fB, fA = R.prf(len(gt_pk) - len(missB), len(spurB), len(missB))[2], R.prf(len(gt_pk) - len(miss), len(spur), len(miss))[2]
                        S_["windows_f1_higher_than_B"] += int(fA > fB); S_["windows_f1_lower_than_B"] += int(fA < fB); B_["windows_f1_higher_than_B"] += int(fA > fB); B_["windows_f1_lower_than_B"] += int(fA < fB)
                        S_["windows_spurious_more_than_B"] += int(len(spur) > len(spurB)); S_["windows_missing_fewer_than_B"] += int(len(miss) < len(missB))
                for ax in axes:
                    for r_ in gt_pk:
                        ax.axvline(r_ / FS, color="tab:red", ls="--", lw=0.7, alpha=0.35)
                    ax.grid(alpha=0.2)
                axes[-1].set_xlabel("time (s)")
                fig.suptitle(f"R3 atlas — {sub} [val] / {site} / w{wi} — GT R red dashed (reference only); predictions unshifted; x = missing, ^ = spurious (50 ms one-to-one); GTF rows: gate g(t) dotted", fontsize=9.5)
                fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}_w{wi}.png", dpi=100); plt.close(fig)
                rz = next((int(r_) for r_ in gt_pk if r_ - ZOOM_LO >= 0 and r_ + ZOOM_HI <= T_LEN), None)
                if rz is not None:
                    sl = slice(rz - ZOOM_LO, rz + ZOOM_HI); tz = (np.arange(rz - ZOOM_LO, rz + ZOOM_HI) - rz) / FS * 1000
                    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.plot(tz, gt[sl], "k", lw=1.6, label="GT ECG")
                    for arm in ROW_ARMS:
                        ax.plot(tz, preds[arm][vp][sl], color=COL[arm], lw=1.0, label=lab(arm))
                    ax.plot(tz, g_true[vp][sl] * 2 - 1, color="gray", ls=":", lw=1.0, label="GTF-TRUE gate (scaled)")
                    ax.axvline(0, color="tab:red", ls="--", lw=0.8); ax.set_xlabel("ms from GT R"); ax.grid(alpha=0.3); ax.legend(fontsize=6.5, ncol=2)
                    ax.set_title(f"{sub}/{site}/w{wi} — GT-R-centred [-300, +500] ms, NFE {NFE}", fontsize=9)
                    fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}_w{wi}_zoom.png", dpi=100); plt.close(fig)
                index_rows.append({"subject": sub, "site": site, "window_index": wi, "array_pos": int(idx[vp]), "n_gt": len(gt_pk), "zoom_r": rz if rz is not None else ""})
            print(f"[atlas] {sub} {site}: 8 windows", flush=True)
    with open(ATLAS / "atlas_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0])); w.writeheader(); w.writerows(index_rows)
    (ATLAS / "atlas_summary.json").write_text(json.dumps({"n_windows": len(index_rows), "per_arm": summary, "per_arm_by_site": by_site,
                                                          "gate_true_in_weak_scaffold_segments(s<0.35)": {"mean": float(np.mean(gate_weak)), "n": len(gate_weak)},
                                                          "gate_true_in_strong_segments": {"mean": float(np.mean(gate_strong)), "n": len(gate_strong)},
                                                          "cudnn_deterministic": bool(torch.backends.cudnn.deterministic)}, indent=2))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
