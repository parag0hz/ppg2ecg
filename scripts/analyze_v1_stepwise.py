"""V1 part 2 — stepwise NFE inference, structural metrics, per-window figures and the dashboard.

Frozen protocol: docs/V1_STEPWISE_VISUALIZATION_PREREGISTRATION.md (a73cafa).
NO TRAINING. Frozen checkpoint forward inference only. kjd/ssx never loaded. Nothing is ever translated.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv, hashlib, json, subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation import s1_audit as S1
from ppg2ecg.evaluation import v1_timing as V
from ppg2ecg.evaluation.metrics import rhythm_morphology_metrics
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/v1_stepwise_visualization"
FIG, ZOOM, DASH = OUT / "figures", OUT / "beat_zooms", OUT / "dashboard"
FS, T_LEN, BATCH, WORKERS = 128, 1024, 64, 12
NFES = (1, 2, 4, 8, 50)
SRC_SEED = 0
CKPT = "outputs/c1_imf_baseline_replay_seed42/checkpoint_best.pt"
SUBJECTS = V.TRAIN + V.VAL
ZOOM_LO_MS, ZOOM_HI_MS = -300.0, 500.0
COL = {1: "tab:red", 2: "tab:orange", 4: "tab:olive", 8: "tab:green", 50: "tab:blue"}


def wcsv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _score(args):
    pred, gt, pk = args
    rm = rhythm_morphology_metrics(pred, gt, FS)
    out = []
    for i in range(len(pred)):
        m = M.qrs_core_morphology(pred[i], gt[i], pk[i])
        reg = M.region_errors(pred[i], gt[i], pk[i])
        ev = R.detect_rpeaks(np.asarray(pred[i], dtype=np.float64), FS)
        n_ref = max(len(pk[i]), 1)
        out.append({"raw_rmse": float(np.sqrt(np.mean((pred[i] - gt[i]) ** 2))),
                    "raw_corr": float(np.corrcoef(pred[i], gt[i])[0, 1]) if np.std(pred[i]) > 1e-12 else np.nan,
                    "qrs_rmse_core": m["qrs_rmse_core"], "qrs_energy_dev": m["qrs_energy_dev"],
                    "qrs_ptp_dev": m["qrs_ptp_dev"], "qrs_deriv_rmse": m["qrs_deriv_rmse"],
                    "qrs_curvature_err": m["qrs_curvature_err"],
                    "background_sq": reg["background__a2_sq"],
                    "f1": float(rm["rpeak_f1"][i]), "beats_ratio_dev": abs(len(ev) / n_ref - 1.0)})
    return out


def _chance(args):
    gt_pk, pred_pk = args
    rng = np.random.default_rng(S1.NULL_SEED)
    return [float(np.mean([R.prf(*(lambda t: (len(t[0]), t[1], t[2]))(
        R.match_rpeaks(gt_pk[i], S1.chance_random_phase(len(pred_pk[i]), T_LEN, rng), FS, S1.MATCH_TOL_MS)))[2]
        for _ in range(S1.NULL_DRAWS)])) for i in range(len(gt_pk))]


def _peaks(s):
    return R.detect_rpeaks(np.asarray(s, dtype=np.float64), FS)


def main() -> int:
    for d in (OUT, FIG, ZOOM, DASH):
        d.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(SUBJECTS)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = ROOT / CKPT
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sdh = hashlib.sha256(b"".join(np.ascontiguousarray(ck["state_dict"][k].numpy()).tobytes()
                                  for k in sorted(ck["state_dict"]))).hexdigest()
    (OUT / "checkpoint_manifest.json").write_text(json.dumps(
        {"path": CKPT, "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "state_dict_sha256": sdh,
         "round": int(ck["epoch"]), "c1_arm": ck["args"]["c1_arm"],
         "selection_metric": float(ck["selection"]["value"])}, indent=2))
    cfg = ck.get("imf_cfg", {})
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
    print(f"[prov] HEAD {head} | ckpt round {ck['epoch']} sd {sdh[:16]}", flush=True)

    wrows, pman, viz_store = [], [], {}
    for sub in SUBJECTS:
        split = "train" if sub in V.TRAIN else "val"
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        c = V.cohorts(sub, d["site"], d["window_index"])
        for site in V.SITES:
            idx = c[site]["metrics"]
            if idx.size == 0:
                continue
            X = d["x"][idx].astype(np.float32); Y = d["y"][idx].astype(np.float64)
            wi = d["window_index"][idx]
            gt_pk = [_peaks(Y[i]) for i in range(len(Y))]
            e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(SRC_SEED))
            vizpos = [int(np.flatnonzero(idx == v)[0]) for v in c[site]["viz"]]
            store = {}
            with torch.no_grad():
                for n in NFES:
                    outs, got = [], set()
                    for i in range(0, len(X), BATCH):
                        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
                        z, k = ER.sample_meanflow_schedule(net, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[n])
                        got.add(int(k)); outs.append(z.squeeze(1).float().cpu().numpy())
                    assert got == {n}, (sub, site, n, got)
                    pred = np.concatenate(outs).astype(np.float64)
                    store[n] = pred[vizpos].copy()
                    rows = _score((pred, Y, gt_pk))
                    pk = [_peaks(pred[i]) for i in range(len(pred))]
                    cf = _chance((gt_pk, pk))
                    for i, r_ in enumerate(rows):
                        r_["f1_excess"] = r_["f1"] - cf[i]
                        wrows.append({"subject": sub, "split": split, "site": site,
                                      "window_index": int(wi[i]), "nfe": n, **r_})
                    pman.append({"subject": sub, "split": split, "site": site, "nfe": n,
                                 "n_windows": int(len(pred)), "realised_steps": n,
                                 "source_seed": SRC_SEED})
            viz_store[(sub, site)] = {"ppg": X[vizpos].astype(np.float64), "gt": Y[vizpos],
                                      "wi": wi[vizpos], "pred": store,
                                      "gt_pk": [gt_pk[i] for i in vizpos],
                                      "ppg_pk": [S1.dsp_ppg_peaks(X[i].astype(np.float64), FS) for i in vizpos]}
        print(f"[G] {sub:4s} ({split}) done", flush=True)
    del net; torch.cuda.empty_cache()
    wcsv(OUT / "metrics_by_window.csv", wrows)
    wcsv(OUT / "predictions_manifest.csv", pman)

    KEYS = ["raw_rmse", "raw_corr", "qrs_rmse_core", "qrs_energy_dev", "qrs_ptp_dev",
            "qrs_deriv_rmse", "qrs_curvature_err", "background_sq", "f1", "f1_excess", "beats_ratio_dev"]
    def agg(sel, **tag):
        v = [r for r in wrows if sel(r)]
        if not v:
            return None
        return {**tag, "n_windows": len(v), **{k: float(np.nanmean([r[k] for r in v])) for k in KEYS}}
    by_sub = [a for sub in SUBJECTS for n in NFES
              if (a := agg(lambda r, s=sub, m=n: r["subject"] == s and r["nfe"] == m,
                           subject=sub, split="train" if sub in V.TRAIN else "val", nfe=n))]
    by_site = [a for sp in ("train", "val") for site in V.SITES for n in NFES
               if (a := agg(lambda r, p=sp, t=site, m=n: r["split"] == p and r["site"] == t and r["nfe"] == m,
                            split=sp, site=site, nfe=n))]
    by_split = [a for sp in ("train", "val") for n in NFES
                if (a := agg(lambda r, p=sp, m=n: r["split"] == p and r["nfe"] == m, split=sp, site="ALL", nfe=n))]
    wcsv(OUT / "metrics_by_subject.csv", by_sub)
    wcsv(OUT / "metrics_by_site.csv", by_site + by_split)
    for r_ in by_split:
        print(f"[N] {r_['split']:5s} NFE {r_['nfe']:2d}: rmse {r_['raw_rmse']:.4f} corr {r_['raw_corr']:.4f} "
              f"qrsRMSE {r_['qrs_rmse_core']:.4f} energyDev {r_['qrs_energy_dev']:.4f} "
              f"derivRMSE {r_['qrs_deriv_rmse']:.4f} F1ex {r_['f1_excess']:+.4f} "
              f"beatsDev {r_['beats_ratio_dev']:.4f}", flush=True)

    # ---------------- per-window figures ----------------
    t = np.arange(T_LEN) / FS
    ROWS = [("PPG input", None), ("GT ECG", "gt")] + [(f"iMF NFE {n}", n) for n in NFES]
    nfig = 0
    for (sub, site), S in viz_store.items():
        split = "train" if sub in V.TRAIN else "val"
        for k in range(len(S["wi"])):
            ymin = min(S["gt"][k].min(), *[S["pred"][n][k].min() for n in NFES])
            ymax = max(S["gt"][k].max(), *[S["pred"][n][k].max() for n in NFES])
            pad = 0.08 * (ymax - ymin + 1e-9)
            fig, ax = plt.subplots(len(ROWS), 1, figsize=(13, 1.35 * len(ROWS)), sharex=True)
            for r_i, (name, key) in enumerate(ROWS):
                a_ = ax[r_i]
                y = S["ppg"][k] if key is None else (S["gt"][k] if key == "gt" else S["pred"][key][k])
                a_.plot(t, y, lw=0.75, color="k" if key in (None, "gt") else COL[key])
                for rp in S["gt_pk"][k]:
                    a_.axvline(rp / FS, color="tab:red", ls="--", lw=0.7, alpha=0.45)
                if key is None:
                    a_.plot(S["ppg_pk"][k] / FS, S["ppg"][k][S["ppg_pk"][k]], "v", color="tab:purple", ms=5)
                else:
                    a_.set_ylim(ymin - pad, ymax + pad)     # common scale: NOT per-panel normalised
                a_.set_ylabel(name, fontsize=8); a_.grid(alpha=0.2)
            ax[-1].set_xlabel("time (s)")
            fig.suptitle(f"V1 stepwise — {sub} [{split}] / {site} / window {int(S['wi'][k])}   "
                         f"source seed 0, identical across NFE; red dashed = GT R-peak, purple = PPG systolic peak",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(FIG / f"{sub}_{site}_w{int(S['wi'][k])}.png", dpi=100); plt.close(fig)
            nfig += 1

            # R-centred zoom, first fully contained GT beat
            lo_s, hi_s = int(round(ZOOM_LO_MS / 1000 * FS)), int(round(ZOOM_HI_MS / 1000 * FS))
            cand = [rp for rp in S["gt_pk"][k] if rp + lo_s >= 0 and rp + hi_s < T_LEN]
            if not cand:
                continue
            rp = int(cand[0])
            tt = np.arange(lo_s, hi_s) / FS * 1000.0
            fig, ax = plt.subplots(len(ROWS), 1, figsize=(9, 1.25 * len(ROWS)), sharex=True)
            for r_i, (name, key) in enumerate(ROWS):
                a_ = ax[r_i]
                y = S["ppg"][k] if key is None else (S["gt"][k] if key == "gt" else S["pred"][key][k])
                a_.plot(tt, y[rp + lo_s:rp + hi_s], lw=1.0, color="k" if key in (None, "gt") else COL[key])
                a_.axvline(0, color="tab:red", ls="--", lw=1.0)
                if key is None:
                    inw = [q for q in S["ppg_pk"][k] if rp + lo_s <= q < rp + hi_s]
                    for q in inw:
                        dms = (q - rp) / FS * 1000.0
                        a_.plot(dms, S["ppg"][k][q], "v", color="tab:purple", ms=7)
                        a_.annotate(f"{dms:+.0f} ms", (dms, S["ppg"][k][q]), fontsize=7,
                                    textcoords="offset points", xytext=(4, 4), color="tab:purple")
                else:
                    a_.set_ylim(ymin - pad, ymax + pad)
                a_.set_ylabel(name, fontsize=8); a_.grid(alpha=0.2)
            ax[-1].set_xlabel("time relative to GT R-peak (ms)   [t_R = 0]")
            fig.suptitle(f"V1 R-centred — {sub} [{split}] / {site} / window {int(S['wi'][k])}  (no alignment)",
                         fontsize=10)
            fig.tight_layout()
            fig.savefig(ZOOM / f"{sub}_{site}_w{int(S['wi'][k])}_zoom.png", dpi=100); plt.close(fig)
    print(f"[P] wrote {nfig} stepwise figures and their R-centred zooms", flush=True)

    (OUT / "provenance.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "protocol": "a73cafa",
        "checkpoint": CKPT, "state_dict_sha256": sdh, "source_seed": SRC_SEED, "nfe_grid": list(NFES),
        "subjects": list(SUBJECTS), "sites": list(V.SITES), "test_subjects_loaded": [],
        "training": False, "predictions_translated": False, "oracle_metrics_used": False,
        "viz_windows": nfig, "metric_windows": len(wrows) // len(NFES)}, indent=2))
    print("\n[done] stepwise analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
