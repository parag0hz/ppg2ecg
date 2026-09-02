"""R2 visual atlas — prereg f954e07 section 24. V1 validation VIZ cohort (64 windows), V1 source construction,
rows: PPG / TRUE scaffold / GT ECG / B / TRUE / SHUFFLE / ORACLE at NFE 4, plus a GT-R-centred zoom.
Deterministic; no window selected or removed; counts only in atlas_summary.json.
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
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import v1_timing as V
from ppg2ecg.flow import rhythm_transfer as RT

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r2_rhythm_transfer"
ATLAS = ART / "visual_atlas"
FS, T_LEN, SRC_SEED, NFE = 128, 1024, 0, 4
ARMS = ("B", "TRUE", "SHUFFLE", "ORACLE")
COL = {"B": "tab:blue", "TRUE": "tab:green", "SHUFFLE": "tab:orange", "ORACLE": "tab:purple"}
ZOOM_LO, ZOOM_HI = 38, 64                                        # -300 ms ... +500 ms


def events_annot(gt_pk, pred):
    pk = R.detect_rpeaks(pred.astype(np.float64), FS)
    m, fp, fn = R.match_rpeaks(gt_pk, pk, FS, 50.0)
    matched_gt = {i for i, _ in m}; matched_pred = {j for _, j in m}
    missing = [gt_pk[i] for i in range(len(gt_pk)) if i not in matched_gt]
    spurious = [pk[j] for j in range(len(pk)) if j not in matched_pred]
    return pk, missing, spurious


@torch.no_grad()
def main() -> int:
    ER.assert_no_test_subjects(V.VAL)
    ATLAS.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    net, _ck, _g = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev); net.requires_grad_(False)
    tcn, _t = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    adapters = {arm.upper(): torch.load(ROOT / f"outputs/r2_{arm}_adapter_seed42/adapter_step{RT.STEPS}.pt", map_location="cpu", weights_only=False)["state_dict"]
                for arm in RT.TRAINED_ARMS}
    viz_man = [r for r in csv.DictReader(open(ART / "shuffle_manifest.csv")) if r["population"] == "viz"]
    p_rule = RT.shuffle_partner([r["subject"] for r in viz_man], [r["site"] for r in viz_man], [int(r["window_index"]) for r in viz_man])
    RT.assert_derangement(p_rule)
    if any(int(viz_man[int(p_rule[i])]["array_pos"]) != int(r["partner_array_pos"]) for i, r in enumerate(viz_man)) or len(viz_man) != 64:
        raise RuntimeError("viz shuffle manifest does not match the frozen derangement rule")
    partner_of = {(r["subject"], r["site"], int(r["array_pos"])): int(r["partner_array_pos"]) for r in viz_man}
    t = np.arange(T_LEN) / FS
    summary = {arm: {"n_gt": 0, "n_missing": 0, "n_spurious": 0, "windows_with_spurious": 0, "windows_with_missing": 0} for arm in ARMS}
    scaffold_at_r, index_rows = [], []

    def run_arm(arm, X32, s32, e0):
        sd = None if arm == "B" else adapters[arm]
        if sd is None:
            net.rhythm_adapter.proj.weight.zero_()
        else:
            net.rhythm_adapter.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        pp = torch.from_numpy(X32).to(dev).unsqueeze(1)
        s = torch.zeros_like(pp) if s32 is None else torch.from_numpy(s32).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, s), e0.to(dev), ER.UNIFORM[NFE]); assert k == NFE
        return z.squeeze(1).float().cpu().numpy()

    for sub in V.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]
        c = V.cohorts(sub, d["site"], WIs)
        for site in V.SITES:
            idx = c[site]["metrics"]                                   # 32 rows, one batch, as V1
            X32, Y32 = Xs[idx].astype(np.float32), Ys[idx].astype(np.float64)
            e0 = torch.randn(len(X32), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
            vizpos = [int(np.flatnonzero(idx == v)[0]) for v in c[site]["viz"]]
            s_true = RT.scaffold_from_ppg(tcn, torch.from_numpy(X32).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy()
            s_shuf = s_true.copy()
            for vp in vizpos:
                partner_pos = partner_of[(sub, site, int(idx[vp]))]
                s_shuf[vp] = s_true[int(np.flatnonzero(idx == partner_pos)[0])]
            s_orc = RT.oracle_fields(Y32, workers=1)
            preds = {"B": run_arm("B", X32, None, e0), "TRUE": run_arm("TRUE", X32, s_true, e0),
                     "SHUFFLE": run_arm("SHUFFLE", X32, s_shuf, e0), "ORACLE": run_arm("ORACLE", X32, s_orc, e0)}
            for vp in vizpos:
                wi = int(WIs[idx[vp]]); gt = Y32[vp]; gt_pk = R.detect_rpeaks(gt, FS); ppg = X32[vp]
                fig, axes = plt.subplots(7, 1, figsize=(13, 12.5), sharex=True)
                axes[0].plot(t, ppg, "k", lw=0.8); axes[0].set_ylabel("PPG", fontsize=9)
                axes[1].plot(t, s_true[vp], color="tab:green", lw=1.2); axes[1].set_ylim(0, 1.1); axes[1].set_ylabel("scaffold\nTRUE", fontsize=9)
                for r_ in gt_pk:
                    val = float(s_true[vp][int(r_)]); scaffold_at_r.append(val)
                    axes[1].text(r_ / FS, 1.02, f"{val:.2f}", fontsize=6, ha="center", color="darkgreen")
                axes[2].plot(t, gt, "k", lw=0.8); axes[2].set_ylabel("GT ECG", fontsize=9)
                axes[2].plot(gt_pk / FS, gt[gt_pk], "r.", ms=5)
                for k_, arm in enumerate(ARMS):
                    ax = axes[3 + k_]; pred = preds[arm][vp]
                    pk, miss, spur = events_annot(gt_pk, pred)
                    ax.plot(t, pred, color=COL[arm], lw=0.8)
                    if len(pk):
                        ax.plot(pk / FS, pred[pk], "o", color="darkorange", ms=3.5)
                    for m_ in miss:
                        ax.plot(m_ / FS, 1.25, "x", color="red", ms=6)
                    for s_ in spur:
                        ax.plot(s_ / FS, -1.25, "^", color="magenta", ms=5)
                    label = "ORACLE (GT-R leak)" if arm == "ORACLE" else arm
                    ax.set_ylabel(f"{label}\nNFE {NFE}", fontsize=9); ax.set_ylim(-1.4, 1.4)
                    ax.text(0.005, 0.85, f"beats {len(pk)}/{len(gt_pk)}  missing {len(miss)}  spurious {len(spur)}", transform=ax.transAxes, fontsize=7.5)
                    S_ = summary[arm]; S_["n_gt"] += len(gt_pk); S_["n_missing"] += len(miss); S_["n_spurious"] += len(spur)
                    S_["windows_with_spurious"] += int(len(spur) > 0); S_["windows_with_missing"] += int(len(miss) > 0)
                for ax in axes:
                    for r_ in gt_pk:
                        ax.axvline(r_ / FS, color="tab:red", ls="--", lw=0.7, alpha=0.35)
                    ax.grid(alpha=0.2)
                axes[-1].set_xlabel("time (s)")
                fig.suptitle(f"R2 atlas — {sub} [val] / {site} / w{wi} — GT R red dashed (reference only); predictions unshifted; "
                             f"x = missing, ^ = spurious (50 ms one-to-one)", fontsize=10)
                fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}_w{wi}.png", dpi=100); plt.close(fig)
                # zoom: first GT R with r-38 >= 0 and r+64 <= 1024 (V1 rule)
                rz = next((int(r_) for r_ in gt_pk if r_ - ZOOM_LO >= 0 and r_ + ZOOM_HI <= T_LEN), None)
                if rz is not None:
                    sl = slice(rz - ZOOM_LO, rz + ZOOM_HI); tz = (np.arange(rz - ZOOM_LO, rz + ZOOM_HI) - rz) / FS * 1000
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(tz, gt[sl], "k", lw=1.6, label="GT ECG")
                    for arm in ARMS:
                        ax.plot(tz, preds[arm][vp][sl], color=COL[arm], lw=1.0, label=("ORACLE (GT-R leak)" if arm == "ORACLE" else arm))
                    ax.plot(tz, s_true[vp][sl] * 2 - 1, color="tab:green", ls=":", lw=1.0, label="TRUE scaffold (scaled)")
                    ax.axvline(0, color="tab:red", ls="--", lw=0.8); ax.set_xlabel("ms from GT R"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
                    ax.set_title(f"{sub}/{site}/w{wi} — GT-R-centred [-300, +500] ms, NFE {NFE}", fontsize=9)
                    fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}_w{wi}_zoom.png", dpi=100); plt.close(fig)
                index_rows.append({"subject": sub, "site": site, "window_index": wi, "array_pos": int(idx[vp]), "n_gt": len(gt_pk),
                                   "partner_array_pos": partner_of[(sub, site, int(idx[vp]))], "zoom_r": rz if rz is not None else ""})
            print(f"[atlas] {sub} {site}: 8 windows", flush=True)
    with open(ATLAS / "atlas_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0])); w.writeheader(); w.writerows(index_rows)
    sa = np.asarray(scaffold_at_r)
    (ATLAS / "atlas_summary.json").write_text(json.dumps({"n_windows": len(index_rows), "per_arm": summary,
                                                          "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                                                          "true_scaffold_value_at_gt_r": {"n": int(sa.size), "median": float(np.median(sa)), "p10": float(np.percentile(sa, 10)),
                                                                                          "p90": float(np.percentile(sa, 90)), "frac_lt_0.35": float(np.mean(sa < 0.35))}}, indent=2))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
