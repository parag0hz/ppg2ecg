"""O2c visualisation — preregistration section 18.

The exact frozen V1 64 validation windows, eight deterministic rows per window plus R-centred overlays, and a
forest plot of the already-computed primary contrasts. No new example is selected, no prediction is shifted.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import o2_warp as O2W  # noqa: E402
from ppg2ecg.evaluation import o2b_warp as BW  # noqa: E402
from ppg2ecg.flow import rhythm_fusion as RF  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/o2c_oracle_integer_grid"
OUT = ART / "figures"
ATLAS = OUT / "atlas"
V1_MANIFEST = ROOT / "artifacts/v1_stepwise_visualization/cohort_manifest.csv"
VAL = ("an0", "k2s")
FS, T_LEN, NFE, SRC_SEED, BATCH = 128, 1024, 4, 0, 64
PRE_MS, POST_MS = 300.0, 500.0


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    sp = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m


R2E = _load("r2_evaluate", "scripts/r2_evaluate.py")
t_ax = np.arange(T_LEN) / FS


def atlas_rows():
    rows = [r for r in csv.DictReader(open(V1_MANIFEST)) if r["cohort"] == "viz" and r["subject"] in VAL]
    ER.assert_no_test_subjects([r["subject"] for r in rows])
    return sorted(rows, key=lambda r: (r["subject"], r["site"], int(r["array_pos"])))


def warp_block(A, warps, direction, dev):
    x = torch.from_numpy(np.ascontiguousarray(A)).to(dev).unsqueeze(1)
    return O2W.apply_warp(x, warps, direction).squeeze(1).cpu().numpy()


def main() -> int:
    ATLAS.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rows = atlas_rows()
    if len(rows) != 64:
        raise RuntimeError(f"expected the 64 frozen V1 validation viz windows, got {len(rows)}")
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ws = d["x"], d["y"], d["window_index"]
        for r in [r for r in rows if r["subject"] == s]:
            i = int(r["array_pos"])
            X.append(Xs[i].astype(np.float32)); Y.append(Ys[i].astype(np.float32))
            SUB.append(s); SITE.append(r["site"]); POS.append(i); WI.append(int(Ws[i]))
    X, Y = np.stack(X), np.stack(Y)
    SUB, SITE, POS, WI = (np.asarray(v) for v in (SUB, SITE, POS, WI))
    gt_pk = R2E.pmap(R2E._peaks, list(Y.astype(np.float64)))

    _n, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    o2c_ck = torch.load(ROOT / "outputs/o2c_canon_oracle_seed42/checkpoint_final.pt", map_location="cpu", weights_only=False)
    o2c = MeanFlowS5(build_penguin_backbone(**o2c_ck["model_cfg"]), cond_mode=o2c_ck["imf_cfg"]["cond_mode"],
                     h_scale=o2c_ck["imf_cfg"]["h_scale"]).to(dev).eval()
    o2c.load_state_dict(o2c_ck["state_dict"]); o2c.requires_grad_(False)

    warps = [BW.IntegerEventWarp(p) for p in gt_pk]
    Xcan = warp_block(X, warps, "to_canonical", dev)
    Ycan = warp_block(Y, warps, "to_canonical", dev)
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
    P_B = R2E.gen_plain(base, X, e0, NFE, dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            pp = torch.from_numpy(np.ascontiguousarray(Xcan[i:i + BATCH])).to(dev).unsqueeze(1)
            z, k = ER.sample_meanflow_schedule(o2c, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
            assert k == NFE
            outs.append(z.squeeze(1).float().cpu().numpy())
    P_CAN = np.concatenate(outs)
    P_O2C = warp_block(P_CAN, warps, "to_raw", dev)

    h_dim = int(ck["model_cfg"]["h_dim"])
    gmod = torch.load(ROOT / f"outputs/r3_gtf_oracle_seed42/module_step{RF.STEPS}.pt", map_location="cpu", weights_only=False)
    m = RF.build_r3_module("gtf", "adaptive", c_hidden=h_dim)
    gnet = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), m, cond_mode=cfg.get("cond_mode", "h_only"),
                               h_scale=cfg.get("h_scale", 1.0))
    gnet.load_state_dict(ck["state_dict"], strict=False)
    gnet.r3.load_state_dict({k: v.to(dev) for k, v in gmod["state_dict"].items()})
    gnet.requires_grad_(False); gnet = gnet.to(dev).eval()
    sfields = RT.oracle_fields(Y, workers=8)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
            sf = torch.from_numpy(sfields[i:i + BATCH]).to(dev).unsqueeze(1)
            z, _k = ER.sample_meanflow_schedule(gnet, RT.make_ppg2(pp, sf), e0[i:i + BATCH].to(dev), ER.UNIFORM[NFE])
            outs.append(z.squeeze(1).float().cpu().numpy())
    P_G = np.concatenate(outs)

    ROWS = [("raw PPG", X, "raw"), ("canonical PPG", Xcan, "can"), ("raw GT ECG", Y, "raw"),
            ("canonical GT ECG", Ycan, "can"), ("B  NFE 4", P_B, "raw"), ("O2c canonical", P_CAN, "can"),
            ("O2c inverse-warped", P_O2C, "raw"), ("GTF-ORACLE", P_G, "raw")]
    COL = ("#444444", "#2166ac", "#111111", "#1a9850", "#b2182b", "#762a83", "#d6604d", "#8c6d31")
    idx = []
    for k in range(len(X)):
        r = np.asarray(gt_pk[k], int)
        q = np.asarray(warps[k].q, int) if not warps[k].identity else r
        fig, ax = plt.subplots(len(ROWS), 1, figsize=(11, 13), sharex=True)
        for a, (name, arr, dom), c in zip(ax, ROWS, COL):
            a.plot(t_ax, arr[k], lw=0.7, color=c)
            for p in (r if dom == "raw" else q):
                a.axvline(p / FS, color="0.7" if dom == "raw" else "#2166ac", lw=0.5, alpha=0.6)
            a.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center"); a.grid(alpha=0.15)
        ax[-1].set_xlabel("time (s)")
        ax[0].set_title(f"{SUB[k]} {SITE[k]} pos {POS[k]} window {WI[k]} | K={len(r)} | "
                        f"{'IDENTITY (K<3)' if warps[k].identity else 'integer-grid canonical'} | "
                        "grey = GT R (raw), blue = q_int (canonical) | ORACLE DIAGNOSTIC", fontsize=8)
        fig.tight_layout(); fig.savefig(ATLAS / f"v1_{k:02d}_{SUB[k]}_{SITE[k]}_{POS[k]}.png", dpi=110); plt.close(fig)
        idx.append({"file": f"v1_{k:02d}_{SUB[k]}_{SITE[k]}_{POS[k]}.png", "subject": SUB[k], "site": SITE[k],
                    "array_pos": int(POS[k]), "window_index": int(WI[k]), "K": int(len(r)),
                    "identity": int(warps[k].identity)})
    with open(OUT / "atlas_index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(idx[0])); w.writeheader(); w.writerows(idx)

    # R-centred overlays (raw coordinates, GT R as the centre; no alignment of any prediction)
    a0, b0 = int(round(PRE_MS * FS / 1000)), int(round(POST_MS * FS / 1000))
    seg = {n: [] for n, _a, _d in ROWS if _d == "raw"}
    for k in range(len(X)):
        for p in np.asarray(gt_pk[k], int):
            if p - a0 < 0 or p + b0 >= T_LEN:
                continue
            for name, arr, dom in ROWS:
                if dom == "raw":
                    seg[name].append(arr[k][p - a0:p + b0])
    tt = (np.arange(-a0, b0) / FS) * 1000.0
    fig, ax = plt.subplots(1, len(seg), figsize=(4 * len(seg), 3.2), sharey=True)
    for a, (name, S) in zip(np.atleast_1d(ax), seg.items()):
        S = np.stack(S)
        a.plot(tt, S.mean(0), lw=1.2, color="#b2182b")
        a.fill_between(tt, np.percentile(S, 25, axis=0), np.percentile(S, 75, axis=0), color="#b2182b", alpha=0.2)
        a.axvline(0, color="0.4", lw=0.6); a.grid(alpha=0.2)
        a.set_title(f"{name}  (n={len(S)} beats)", fontsize=8); a.set_xlabel("ms from GT R")
    np.atleast_1d(ax)[0].set_ylabel("amplitude")
    fig.suptitle("R-centred −300…+500 ms, median IQR band; GT R centres only, no prediction is shifted", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "r_centred_overlays.png", dpi=130); plt.close(fig)

    # forest plot of the frozen primary contrasts (already computed; nothing is recomputed here)
    boot = list(csv.DictReader(open(ART / "paired_bootstrap.csv")))
    keys = ["f1_excess", "nAE_T4", "nAE_T6", "nAE_T7", "nAE_T8", "qrs_deriv_rmse", "qrs_curvature_err"]
    sel = [r for k in keys for r in boot if r["metric"] == k]
    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(sel) + 1.6))
    for i, r in enumerate(sel):
        lo, hi, pt = float(r["lo"]), float(r["hi"]), float(r["point"])
        ax.plot([lo, hi], [i, i], color="#333333", lw=1.4)
        ax.plot([pt], [i], "o", color="#b2182b" if lo > 0 else ("#2166ac" if hi < 0 else "#777777"), ms=5)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(sel))); ax.set_yticklabels([r["metric"] for r in sel], fontsize=8)
    ax.set_xlabel("paired effect, positive = O2c better (ECG-window clustered, subject-stratified, 2,000 replicates)")
    ax.set_title("O2c vs B — frozen primary contrasts (ORACLE DIAGNOSTIC)", fontsize=9)
    ax.grid(alpha=0.2, axis="x"); fig.tight_layout()
    fig.savefig(OUT / "primary_contrasts_forest.png", dpi=130); plt.close(fig)
    (OUT / "figures_manifest.json").write_text(json.dumps(
        {"atlas_windows": len(idx), "atlas_rows": [n for n, _a, _d in ROWS], "overlay_window_ms": [-PRE_MS, POST_MS],
         "source_seed": SRC_SEED, "nfe": NFE, "shifted_predictions": False, "cherry_picked": False,
         "cohort": "frozen V1 visualisation windows of an0/k2s (64)"}, indent=2))
    print(f"[fig] {len(idx)} atlas windows + overlays + forest", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
