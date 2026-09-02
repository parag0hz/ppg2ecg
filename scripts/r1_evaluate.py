"""R1 steps 9-12 — internal-dev threshold freeze, an0/k2s evaluation, controls, site, bootstrap, figures.

Frozen protocol c7481f9. Thresholds chosen on INTERNAL_DEV only, before any validation number.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv, json, subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, THRESH_GRID, TOL_MS, extract_events, n_trainable

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r1_global_rhythm"
FIG, ATLAS = ART / "figures", ART / "visual_atlas"
FS, T_LEN, SEED_CTRL = 128, 1024, 20260902
CK = {"global": "outputs/r1_global_tcn_seed42/checkpoint_best.pt",
      "local": "outputs/r1_local_tcn_seed42/checkpoint_best.pt",
      "global_site": "outputs/r1_global_site_tcn_seed42/checkpoint_best.pt"}
RR_TOL_MS = 150.0


def wcsv(p, rows):
    if rows:
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load(variant, dev):
    ck = torch.load(ROOT / CK[variant], map_location="cpu", weights_only=False)
    net = RhythmTCN(ck["dilations"], n_sites=ck["n_sites"]).to(dev).eval()
    net.load_state_dict(ck["state_dict"]); net.requires_grad_(False)
    return net, ck


@torch.no_grad()
def probs(net, X, S, dev):
    out = []
    for i in range(0, len(X), 512):
        x = torch.from_numpy(X[i:i + 512]).to(dev).unsqueeze(1)
        s = torch.from_numpy(S[i:i + 512]).to(dev) if net.n_sites else None
        out.append(torch.sigmoid(net(x, s)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def score_window(gt, pred, tol_ms):
    m, fp, fn = R.match_rpeaks(gt, pred, FS, tol_ms)
    p, r, f = R.prf(len(m), fp, fn)
    err = [abs(int(pred[j]) - int(gt[i])) / FS * 1000.0 for i, j in m]
    return {"precision": p, "recall": r, "f1": f, "n_matched": len(m), "missing": fn, "spurious": fp,
            "n_gt": int(len(gt)), "n_pred": int(len(pred)),
            "timing_med_ms": float(np.median(err)) if err else np.nan,
            "timing_mean_ms": float(np.mean(err)) if err else np.nan, "matches": m}


def rr_window(gt, pred, matches):
    """RR errors only where consecutive GT beats BOTH have one-to-one matches."""
    mm = dict(matches); out = []
    for i in range(len(gt) - 1):
        if i in mm and i + 1 in mm:
            rr_g = (gt[i + 1] - gt[i]) / FS * 1000.0
            rr_p = (pred[mm[i + 1]] - pred[mm[i]]) / FS * 1000.0
            out.append((rr_g, rr_p))
    return out


def evaluate(gt_list, ev_list, subjects, label):
    """Per-window event metrics at every tolerance + RR metrics; returns (rows_per_window, summary_rows)."""
    per, summ = [], []
    for tol in TOL_MS:
        for k, (g, e) in enumerate(zip(gt_list, ev_list)):
            s = score_window(g, e, tol)
            per.append({"model": label, "tol_ms": tol, "window": k, "subject": subjects[k],
                        **{kk: v for kk, v in s.items() if kk != "matches"},
                        "beats_ratio_dev": abs(s["n_pred"] / max(s["n_gt"], 1) - 1.0)})
    for tol in TOL_MS:
        sel = [r for r in per if r["tol_ms"] == tol]
        def mac(k):
            return float(np.mean([np.nanmean([r[k] for r in sel if r["subject"] == s]) for s in C.VAL]))
        summ.append({"model": label, "tol_ms": tol, "precision": mac("precision"), "recall": mac("recall"),
                     "f1": mac("f1"), "coverage": float(sum(r["n_matched"] for r in sel) / max(sum(r["n_gt"] for r in sel), 1)),
                     "missing": int(sum(r["missing"] for r in sel)), "spurious": int(sum(r["spurious"] for r in sel)),
                     "beats_ratio": float(sum(r["n_pred"] for r in sel) / max(sum(r["n_gt"] for r in sel), 1)),
                     "beats_ratio_dev": mac("beats_ratio_dev"),
                     "timing_med_ms": float(np.nanmedian([r["timing_med_ms"] for r in sel])),
                     "timing_mean_ms": float(np.nanmean([r["timing_mean_ms"] for r in sel]))})
    return per, summ


def rr_metrics(gt_list, ev_list, subjects, label):
    pairs, perwin = [], []
    for k, (g, e) in enumerate(zip(gt_list, ev_list)):
        s = score_window(g, e, RR_TOL_MS)
        rr = rr_window(g, e, s["matches"])
        pairs += [(subjects[k], a, b) for a, b in rr]
        ae = [abs(a - b) for a, b in rr]
        perwin.append({"model": label, "window": k, "subject": subjects[k], "n_rr": len(rr),
                       "rr_mae_ms": float(np.mean(ae)) if ae else np.nan,
                       "rr_rel_med": float(np.median([abs(a - b) / max(a, 1e-9) for a, b in rr])) if rr else np.nan})
    g = np.array([a for _, a, _ in pairs]); p = np.array([b for _, _, b in pairs])
    e = np.abs(g - p); rel = e / np.maximum(g, 1e-9)
    summ = {"model": label, "n_rr_pairs": int(g.size),
            "rr_mae_ms": float(e.mean()) if g.size else np.nan, "rr_median_ae_ms": float(np.median(e)) if g.size else np.nan,
            "rr_rmse_ms": float(np.sqrt(np.mean(e ** 2))) if g.size else np.nan,
            "rr_corr": float(np.corrcoef(g, p)[0, 1]) if g.size > 2 else np.nan,
            "rr_rel_median": float(np.median(rel)) if g.size else np.nan,
            **{f"rr_within_{t}ms": float(np.mean(e <= t)) if g.size else np.nan for t in (25, 50, 100)},
            "rr_within_10pct": float(np.mean(rel <= 0.10)) if g.size else np.nan,
            "rr_within_20pct": float(np.mean(rel <= 0.20)) if g.size else np.nan}
    return pairs, perwin, summ


def main() -> int:
    for d in (ART, FIG, ATLAS):
        d.mkdir(parents=True, exist_ok=True)
    assert_no_test_subjects(C.VAL)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dev = torch.device("cuda")
    split = C.internal_dev_split()

    # ---------------- internal-dev arrays (cached by training) + threshold freeze ----------------
    z = np.load(ART / "_cache_internal_dev.npz"); Xdv, Sdv = z["x"], z["site"]
    gt_dv = []
    for sub in split["internal_dev"]:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]   # decompress ONCE: every d[key] access re-reads the npz
        pos = C.cohort_positions(sub, d["site"], d["window_index"], C.n_per_for(sub))
        for site in C.SITES:
            gt_dv += [R.detect_rpeaks(Ys[int(i)].astype(np.float64), FS) for i in pos[site]]
    assert len(gt_dv) == len(Xdv)
    variants = [v for v in ("global", "local", "global_site") if (ROOT / CK[v]).exists()]
    nets = {v: load(v, dev) for v in variants}
    thr = {}
    for v, (net, ck) in nets.items():
        pr = probs(net, Xdv, Sdv, dev)
        best = None
        for t in THRESH_GRID:
            f1 = np.mean([score_window(g, extract_events(pr[k], t), 150.0)["f1"] for k, g in enumerate(gt_dv)])
            if best is None or f1 > best[1]:
                best = (t, float(f1))
        thr[v] = {"threshold": best[0], "internal_dev_f1_150ms": best[1], "grid": list(THRESH_GRID),
                  "selection_metric": "F1@150ms on INTERNAL_DEV", "nms_refractory_samples": 32}
        print(f"[T] {v:12s} threshold {best[0]:.2f}  internal-dev F1@150 {best[1]:.4f}", flush=True)
    (ART / "threshold_selection.json").write_text(json.dumps(thr, indent=2))
    del Xdv, Sdv, gt_dv

    # ---------------- validation arrays (built here; ECG used only for labels) ----------------
    Xv, Sv, SUB, WI, gt_v = [], [], [], [], []
    for sub in C.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]   # decompress ONCE: every d[key] access re-reads the npz
        pos = C.cohort_positions(sub, d["site"], d["window_index"], C.n_per_for(sub))
        for si, site in enumerate(C.SITES):
            idx = pos[site]
            Xv.append(Xs[idx].astype(np.float32)); Sv.append(np.full(len(idx), si))
            SUB += [sub] * len(idx); WI += [(sub, site, int(WIs[i])) for i in idx]
            gt_v += [R.detect_rpeaks(Ys[int(i)].astype(np.float64), FS) for i in idx]
    Xv, Sv, SUB = np.concatenate(Xv), np.concatenate(Sv), np.array(SUB)
    SITE = np.array([w[1] for w in WI])
    print(f"[V] validation {len(Xv)} windows, {sum(len(g) for g in gt_v)} GT beats", flush=True)

    ev_rows, ev_sum, rr_sum, rr_per_all, ev_per_all, events = [], [], [], {}, {}, {}
    for v, (net, ck) in nets.items():
        pr = probs(net, Xv, Sv, dev)
        ev = [extract_events(pr[k], thr[v]["threshold"]) for k in range(len(pr))]
        events[v] = (pr, ev)
        per, summ = evaluate(gt_v, ev, SUB, v); ev_rows += per; ev_sum += summ; ev_per_all[v] = per
        pairs, rrp, rs = rr_metrics(gt_v, ev, SUB, v); rr_sum.append(rs); rr_per_all[v] = rrp
        for s in summ:
            print(f"[E] {v:12s} @{s['tol_ms']:>3.0f}ms F1 {s['f1']:.4f} P {s['precision']:.4f} R {s['recall']:.4f} "
                  f"cov {s['coverage']:.3f} beats {s['beats_ratio']:.3f} tmed {s['timing_med_ms']:.1f}", flush=True)
        print(f"[RR] {v:12s} medAE {rs['rr_median_ae_ms']:.1f} MAE {rs['rr_mae_ms']:.1f} rel {rs['rr_rel_median']:.3f} "
              f"corr {rs['rr_corr']:.3f} <=100 {rs['rr_within_100ms']:.3f} n={rs['n_rr_pairs']}", flush=True)
        if v == "global":
            wcsv(ART / "rr_pairs_global.csv", [{"subject": s, "rr_gt_ms": a, "rr_pred_ms": b} for s, a, b in pairs])
    wcsv(ART / "event_metrics.csv", ev_sum); wcsv(ART / "event_metrics_per_window.csv", ev_rows)
    wcsv(ART / "rr_metrics.csv", rr_sum)

    # ---------------- Global vs Local paired bootstrap ----------------
    boot = []
    if "global" in nets and "local" in nets:
        def col(v, tol, k):
            return np.asarray([r[k] for r in ev_per_all[v] if r["tol_ms"] == tol], float)
        for tol in TOL_MS:
            r = paired_subject_bootstrap(col("local", tol, "f1"), col("global", tol, "f1"), SUB, "higher_better",
                                         seed=SEED_CTRL)
            boot.append({"metric": f"f1@{tol:.0f}ms", **r})
        r = paired_subject_bootstrap(np.asarray([x["rr_mae_ms"] for x in rr_per_all["local"]], float),
                                     np.asarray([x["rr_mae_ms"] for x in rr_per_all["global"]], float),
                                     SUB, "lower_better", seed=SEED_CTRL)
        boot.append({"metric": "rr_mae_ms", **r})
        r = paired_subject_bootstrap(col("local", 150.0, "beats_ratio_dev"), col("global", 150.0, "beats_ratio_dev"),
                                     SUB, "lower_better", seed=SEED_CTRL)
        boot.append({"metric": "beats_ratio_dev", **r})
        for b in boot:
            print(f"[B] {b['metric']:16s} Global-Local {b['point']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] {b['verdict']}", flush=True)
    wcsv(ART / "paired_bootstrap.csv", boot)

    # ---------------- input-dependence controls (Global, no retraining) ----------------
    ctrl = []
    if "global" in nets:
        net, _ = nets["global"]; rng = np.random.default_rng(SEED_CTRL)
        Xs = Xv.copy()
        for sub in C.VAL:
            for site in C.SITES:
                m = np.flatnonzero((SUB == sub) & (SITE == site))
                Xs[m] = Xv[m][C.derangement(m.size, rng)]
        off = C.circular_offsets(len(Xv), rng)
        Xc = np.stack([np.roll(Xv[k], int(off[k])) for k in range(len(Xv))])
        for cname, Xi in (("TRUE", Xv), ("WINDOW-SHUFFLE", Xs), ("CIRCULAR-SHIFT", Xc)):
            pr = probs(net, Xi, Sv, dev)
            ev = [extract_events(pr[k], thr["global"]["threshold"]) for k in range(len(pr))]
            _, summ = evaluate(gt_v, ev, SUB, f"global/{cname}")
            _, _, rs = rr_metrics(gt_v, ev, SUB, f"global/{cname}")
            row = {"input": cname, **{f"f1@{s['tol_ms']:.0f}": s["f1"] for s in summ},
                   "rr_mae_ms": rs["rr_mae_ms"], "rr_median_ae_ms": rs["rr_median_ae_ms"],
                   "beats_ratio_dev": summ[2]["beats_ratio_dev"]}
            ctrl.append(row)
            print(f"[C] {cname:15s} F1@50 {row['f1@50']:.4f} F1@200 {row['f1@200']:.4f} RR MAE {row['rr_mae_ms']:.1f}", flush=True)
            if cname != "TRUE":
                events[f"global/{cname}"] = (pr, ev)
    wcsv(ART / "input_control_metrics.csv", ctrl)

    # ---------------- site-wise (Global) ----------------
    site_rows = []
    if "global" in nets:
        _, ev = events["global"]
        for site in C.SITES:
            m = np.flatnonzero(SITE == site)
            _, summ = evaluate([gt_v[k] for k in m], [ev[k] for k in m], SUB[m], "global")
            _, _, rs = rr_metrics([gt_v[k] for k in m], [ev[k] for k in m], SUB[m], "global")
            site_rows.append({"site": site, "n_windows": int(m.size),
                              **{f"f1@{s['tol_ms']:.0f}": s["f1"] for s in summ},
                              "rr_mae_ms": rs["rr_mae_ms"], "rr_median_ae_ms": rs["rr_median_ae_ms"],
                              "beats_ratio_dev": summ[2]["beats_ratio_dev"]})
            print(f"[S] {site:8s} F1@50 {site_rows[-1]['f1@50']:.3f} @150 {site_rows[-1]['f1@150']:.3f} "
                  f"@200 {site_rows[-1]['f1@200']:.3f} RR MAE {rs['rr_mae_ms']:.1f} beatsdev {site_rows[-1]['beats_ratio_dev']:.3f}", flush=True)
    wcsv(ART / "site_metrics.csv", site_rows)

    # ---------------- decision (frozen gates, prereg s17) ----------------
    g_sum = {s["tol_ms"]: s for s in ev_sum if s["model"] == "global"}
    g_rr = next(r for r in rr_sum if r["model"] == "global")
    B = {b["metric"]: b for b in boot}
    c_true = next((c for c in ctrl if c["input"] == "TRUE"), None)
    c_shuf = next((c for c in ctrl if c["input"] == "WINDOW-SHUFFLE"), None)
    wins = [m for m in ("f1@150ms", "f1@200ms", "rr_mae_ms", "beats_ratio_dev") if B.get(m, {}).get("verdict") == "improves"]
    g1 = len(wins) >= 2
    g2 = bool(c_true and c_shuf and c_true["f1@200"] > c_shuf["f1@200"] and c_true["rr_mae_ms"] < c_shuf["rr_mae_ms"])
    g3 = bool(g_rr["rr_median_ae_ms"] < 100.0 or g_rr["rr_rel_median"] < 0.10)
    g4 = bool(g_sum[150.0]["beats_ratio_dev"] < 0.20)
    global_beats_local = g1
    if g1 and g2 and g3 and g4:
        coarse = "GLOBAL RHYTHM SCAFFOLD SUPPORTED"
    elif global_beats_local:
        coarse = "GLOBAL CONTEXT HELPS BUT RHYTHM SCAFFOLD REMAINS WEAK"
    else:
        coarse = "GLOBAL RHYTHM SCAFFOLD NOT SUPPORTED"
    dec = {"coarse_rhythm_verdict": coarse,
           "gate1_global_beats_local_on": wins, "gate1": g1, "gate2_true_beats_shuffle": g2,
           "gate3_rr_informative": g3, "gate4_beats_ratio_dev_lt_0.20": g4,
           "global_f1_50": g_sum[50.0]["f1"], "global_timing_med_ms_50": g_sum[50.0]["timing_med_ms"],
           "global_missing_50": g_sum[50.0]["missing"], "global_spurious_50": g_sum[50.0]["spurious"],
           "v1_fixed_delay_prior_cov50": 0.218, "v1_fixed_delay_prior_median_ae_ms": 172.3,
           "rr": g_rr, "thresholds": thr, "prereg": "c7481f9"}
    (ART / "decision.json").write_text(json.dumps(dec, indent=2, default=float))
    print(f"\n[GATES] 1:{g1} ({wins}) 2:{g2} 3:{g3} 4:{g4}\n[COARSE VERDICT] {coarse}", flush=True)

    # ---------------- figures ----------------
    t = np.arange(T_LEN) / FS
    if "global" in nets:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for v in variants:
            ax.plot(TOL_MS, [next(s["f1"] for s in ev_sum if s["model"] == v and s["tol_ms"] == tol) for tol in TOL_MS], "o-", lw=2, label=v)
        for c in ctrl:
            if c["input"] != "TRUE":
                ax.plot(TOL_MS, [c[f"f1@{tol:.0f}"] for tol in TOL_MS], "s--", lw=1.4, alpha=0.7, label=f"global / {c['input']}")
        ax.axhline(0.218, color="gray", ls=":", label="V1 fixed-delay prior cov@50 (site)")
        ax.set_xlabel("one-to-one tolerance (ms)"); ax.set_ylabel("F1 (equal-subject macro)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("R1 — F1 vs tolerance on an0/k2s   (>=200 ms = coarse event localization, not R-peak accuracy)")
        fig.tight_layout(); fig.savefig(FIG / "f1_vs_tolerance.png", dpi=115); plt.close(fig)

        pr = [r for r in csv.DictReader(open(ART / "rr_pairs_global.csv"))]
        g = np.array([float(r["rr_gt_ms"]) for r in pr]); p = np.array([float(r["rr_pred_ms"]) for r in pr])
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        ax.hexbin(g, p, gridsize=50, cmap="magma", bins="log"); ax.plot([300, 1500], [300, 1500], "w--", lw=1)
        ax.set_xlabel("RR_gt (ms)"); ax.set_ylabel("RR_pred (ms), Global-TCN"); ax.set_title(f"R1 — RR scatter (n={g.size}, corr {g_rr['rr_corr']:.3f})")
        fig.tight_layout(); fig.savefig(FIG / "rr_scatter_global.png", dpi=115); plt.close(fig)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        x = np.arange(4)
        for i, tol in enumerate((50.0, 100.0, 150.0, 200.0, 250.0)):
            ax[0].bar(x + (i - 2) * 0.15, [r[f"f1@{tol:.0f}"] for r in site_rows], 0.15, label=f"@{tol:.0f}")
        ax[0].set_xticks(x); ax[0].set_xticklabels(C.SITES); ax[0].set_title("Global-TCN F1 by site"); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, axis="y")
        ax[1].bar(x, [r["rr_mae_ms"] for r in site_rows]); ax[1].set_xticks(x); ax[1].set_xticklabels(C.SITES)
        ax[1].set_title("Global-TCN RR MAE by site (ms)"); ax[1].grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(FIG / "site_coarse_f1.png", dpi=115); plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for sub in C.VAL:
            v = [x["rr_mae_ms"] for x in rr_per_all["global"] if x["subject"] == sub and np.isfinite(x["rr_mae_ms"])]
            ax.hist(v, bins=40, histtype="step", lw=1.8, label=f"{sub} (median {np.median(v):.0f} ms)")
        ax.set_xlabel("per-window RR MAE (ms), Global-TCN"); ax.set_ylabel("windows"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(FIG / "subject_rr_error.png", dpi=115); plt.close(fig)

        # visual atlas: metadata-only cohort r1-visual-v1, 8 per val subject x site
        pr_g, ev_g = events["global"]; pr_l, ev_l = events["local"] if "local" in events else (None, None)
        for sub in C.VAL:
            d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
            Xs, Ys, WIs = d["x"], d["y"], d["window_index"]   # decompress ONCE: every d[key] access re-reads the npz
            vis = C.cohort_positions(sub, d["site"], d["window_index"], C.N_VISUAL_PER, salt=C.VISUAL_SALT)
            for site in C.SITES:
                fig, axes = plt.subplots(8, 1, figsize=(13, 15), sharex=True)
                for row, pos in enumerate(vis[site]):
                    wi = int(WIs[pos])
                    k = next((kk for kk, w in enumerate(WI) if w == (sub, site, wi)), None)
                    a_ = axes[row]
                    if k is None:
                        a_.text(0.5, 0.5, f"window {wi}: not in R1 validation cohort", transform=a_.transAxes, ha="center"); continue
                    a_.plot(t, Xv[k] / (np.abs(Xv[k]).max() + 1e-9) * 0.9, color="k", lw=0.6, label="PPG (scaled)")
                    a_.plot(t, pr_g[k], color="tab:green", lw=1.2, label="Global field")
                    if pr_l is not None:
                        a_.plot(t, pr_l[k], color="tab:orange", lw=1.0, alpha=0.8, label="Local field")
                    for r_ in gt_v[k]:
                        a_.axvline(r_ / FS, color="tab:red", ls="--", lw=0.8, alpha=0.6)
                    a_.plot(ev_g[k] / FS, np.full(len(ev_g[k]), 1.05), "v", color="tab:green", ms=6)
                    a_.set_ylim(-1.0, 1.15); a_.set_ylabel(f"w{wi}", fontsize=8); a_.grid(alpha=0.2)
                    if row == 0:
                        a_.legend(fontsize=7, loc="upper right", ncol=3)
                axes[-1].set_xlabel("time (s)")
                fig.suptitle(f"R1 atlas — {sub} [val] / {site}: PPG, GT R (red dashed), Global/Local soft field, Global events (▼)")
                fig.tight_layout(); fig.savefig(ATLAS / f"{sub}_{site}.png", dpi=100); plt.close(fig)

    (ART / "model_manifest.json").write_text(json.dumps({v: {"path": CK[v], "params": n_trainable(nets[v][0]),
        "rf_samples": nets[v][0].rf, "rf_ms": nets[v][0].rf / FS * 1000, "best_epoch": nets[v][1]["epoch"],
        "internal_dev_bce": nets[v][1]["internal_dev_bce"]} for v in variants}, indent=2))
    (ART / "provenance.json").write_text(json.dumps({
        "head": head, "utc": datetime.now(timezone.utc).isoformat(), "prereg": "c7481f9",
        "validation_windows": int(len(Xv)), "validation_gt_beats": int(sum(len(g) for g in gt_v)),
        "test_subjects_loaded": [], "ecg_input_to_probe": False, "r_available_at_inference": False,
        "thresholds_selected_on": "internal_dev (u7y, e61) only", "control_rng_seed": SEED_CTRL}, indent=2))
    print("[done] R1 evaluation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
