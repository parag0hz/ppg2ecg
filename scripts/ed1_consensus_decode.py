"""ED1 — event-level consensus decoding (docs/ED1_EVENT_CONSENSUS_DECODING_PREREGISTRATION.md). No training.

  dev : choose (w, theta, b) per generator on VALIDATION patients -> artifacts/ed1_consensus_decoding/params.json
  test: evaluate the frozen params once on the test patients      -> artifacts/ed1_consensus_decoding/result.json
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cd1_evaluate as CDE  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from pz3_compare import morph_at  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

FS, K, LEN, RI, REFRACT = 128, 16, 83, 32, int(0.250 * 128)
GRID_W_MS, GRID_THETA = (25, 40, 50), (0.25, 0.375, 0.5)
RUN = {"I": "outputs/v1_vitaldb_armI_seed42", "D": "outputs/cd1_vitaldb_armD_seed42"}
OUT, CACHE = ROOT / "artifacts/ed1_consensus_decoding", ROOT / "outputs/ed1_cache"


def _peaks(sig):
    return np.asarray(detect_rpeaks(sig, FS), int)


def decode_window(args):
    """samples [K,T], peaks (list of K int arrays), w (samples), theta, b (samples) -> wave, pos, vote_frac, vote_sd."""
    S, P, w, theta, b = args
    Kk, T = S.shape
    allp = np.concatenate([p for p in P if len(p)] or [np.zeros(0, int)])
    wave = np.median(S, 0).astype(np.float64)
    if allp.size == 0:
        return wave, np.zeros(0), np.zeros(0), np.zeros(0)
    votes = np.bincount(np.clip(allp, 0, T - 1), minlength=T).astype(float)
    cnt = np.convolve(votes, np.ones(2 * w + 1), mode="same")
    cand = [i for i in range(T) if cnt[i] >= theta * Kk and (i == 0 or cnt[i] >= cnt[i - 1]) and (i == T - 1 or cnt[i] > cnt[i + 1])]
    cand.sort(key=lambda i: -cnt[i])
    events = []
    for i in cand:
        box = allp[np.abs(allp - i) <= w]
        c = float(np.median(box))
        if all(abs(c - e[0]) >= REFRACT for e in events):
            events.append((c, len(box) / Kk, float(np.std(box))))
    events.sort()
    acc, n = np.zeros(T), np.zeros(T)
    for c, _, _ in events:
        beats = []
        for k in range(Kk):
            if len(P[k]) == 0:
                continue
            q = P[k][np.argmin(np.abs(P[k] - c))]
            if abs(q - c) <= w and q - RI >= 0 and q - RI + LEN <= T:
                beats.append(S[k, q - RI:q - RI + LEN])
        if not beats:
            continue
        beat = np.mean(beats, 0)
        a = int(round(c + b)) - RI
        ta, tb = max(0, -a), LEN - max(0, a + LEN - T)
        a0, b0 = max(0, a), min(T, a + LEN)
        if b0 > a0:
            acc[a0:b0] += beat[ta:tb]; n[a0:b0] += 1
    wave = np.where(n > 0, acc / np.maximum(n, 1), wave)
    ev = np.array(events) if events else np.zeros((0, 3))
    return wave, ev[:, 0] + b, ev[:, 1], ev[:, 2]


def samples_and_peaks(arm, split, X, ex, dev):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"samples_{arm}_{split}.npy"
    if f.exists():
        S = np.load(f).astype(np.float32)
    else:
        if arm == "D":
            ck = torch.load(ROOT / RUN[arm] / "checkpoint_last.pt", map_location="cpu", weights_only=False)
            net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
            gen = lambda s: CDE.sample(net, X, s, 1, dev)[0]  # noqa: E731
        else:
            net = U2.build(ROOT / RUN[arm] / "checkpoint_last.pt", dev)[0]
            def gen(s, _n=net):
                e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
                return U2.generate(_n, X, e0, 1, dev, "I")[0]
        S = np.stack([gen(s).astype(np.float32) for s in range(K)])
        np.save(f, S.astype(np.float16))
        del net; torch.cuda.empty_cache()
    flat = S.reshape(-1, S.shape[-1])
    pk = list(ex.map(_peaks, list(flat.astype(np.float64)), chunksize=256))
    n = S.shape[1]
    return S, [[pk[k * n + i] for k in range(K)] for i in range(n)]


def decode_all(S, P, w, theta, b, ex):
    n = S.shape[1]
    res = list(ex.map(decode_window, [(S[:, i, :], P[i], w, theta, b) for i in range(n)], chunksize=128))
    return np.stack([r[0] for r in res]), [r[1] for r in res], [r[2] for r in res], [r[3] for r in res]


def timing_bias(pos, gt):
    errs = []
    for p, g in zip(pos, gt):
        m, _, _ = R.match_rpeaks(g, np.round(p).astype(int), FS, 50.0)
        errs += [p[j] - g[i] for i, j in m]
    return float(np.median(errs)) if errs else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dev", "test"])
    mode = ap.parse_args().mode
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ex = ProcessPoolExecutor(10)
    split = "val" if mode == "dev" else "test"
    X, Y, pid = VM.load(split)
    gt = list(ex.map(_peaks, list(Y), chunksize=256))

    if mode == "dev":
        params = {}
        for arm in ("I", "D"):
            S, P = samples_and_peaks(arm, split, X, ex, dev)
            table = []
            for w_ms in GRID_W_MS:
                for th in GRID_THETA:
                    w = int(round(w_ms / 1000 * FS))
                    _, pos0, _, _ = decode_all(S, P, w, th, 0.0, ex)
                    b = -timing_bias(pos0, gt)
                    wave, _, _, _ = decode_all(S, P, w, th, b, ex)
                    mt = U2.task_metrics("ECG", 8, wave, Y, pid)
                    f1, hr = V.cluster_ci(mt["Rpeak_F1"][0], pid)[0], V.cluster_ci(mt["HR"][0], pid)[0]
                    table.append(dict(w_ms=w_ms, theta=th, b=b, f1=f1, hr=hr))
                    print(f"[ed1-dev] {arm} w {w_ms} theta {th} b {b:+.2f}  F1 {f1:.4f} HR {hr:.3f}", flush=True)
            best = max(t["f1"] for t in table)
            ch = min((t for t in table if t["f1"] >= best - 0.002), key=lambda t: t["hr"])
            params[arm] = {"chosen": ch, "grid": table}
            print(f"[ed1-dev] {arm} CHOSEN {ch}", flush=True)
        (OUT / "params.json").write_text(json.dumps(params, indent=1))
        return

    params = json.loads((OUT / "params.json").read_text())
    sr1 = {"I": dict(np.load(ROOT / "outputs/sr1_eval/arm_I_seed42.npz")), "D": dict(np.load(ROOT / "outputs/sr1_eval/arm_D_seed42.npz")),
           "C": dict(np.load(ROOT / "outputs/sr1_eval/arm_C_seed42.npz"))}
    assert all(np.array_equal(d["pid"], pid) for d in sr1.values())
    result = {}
    for arm in ("I", "D"):
        ch = params[arm]["chosen"]
        w = int(round(ch["w_ms"] / 1000 * FS))
        S, P = samples_and_peaks(arm, split, X, ex, dev)
        wave, pos, vfrac, vsd = decode_all(S, P, w, ch["theta"], ch["b"], ex)
        mt = U2.task_metrics("ECG", 8, wave, Y, pid); tab = mt.pop("_counts")[0]
        fd = U2.pooled_extras("ECG", wave, Y, tab)["FD_kanflow"]
        morph_c = np.array(list(ex.map(morph_at, zip(wave, Y, gt), chunksize=128)))
        morph_s = np.nanmean(np.vstack([np.array(list(ex.map(morph_at, zip(S[k].astype(np.float64), Y, gt), chunksize=128))) for k in range(4)]), 0)
        one = sr1[arm]
        ci = lambda v: V.cluster_ci(v, pid)  # noqa: E731
        d_f1, d_m = ci(mt["Rpeak_F1"][0] - one["nfe1_Rpeak_F1"]), ci(morph_c - morph_s)
        ok_f1, ok_m = d_f1[0] >= 0.05 and d_f1[1] > 0, d_m[0] >= 0.15 and d_m[1] > 0
        # reliability over matched consensus events
        sds, errs, tp, fp, fn, tp_h, fp_h, n_gt = [], [], 0, 0, 0, 0, 0, 0
        for p, vf, vs, g in zip(pos, vfrac, vsd, gt):
            pi = np.round(p).astype(int)
            m, f_p, f_n = R.match_rpeaks(g, pi, FS, 50.0)
            tp += len(m); fp += f_p; fn += f_n; n_gt += len(g)
            for i, j in m:
                sds.append(vs[j]); errs.append(abs(p[j] - g[i]))
            hi = vf >= 0.75
            mh, fph, _ = R.match_rpeaks(g, pi[hi], FS, 50.0)
            tp_h += len(mh); fp_h += fph
        rho = float(spearmanr(sds, errs).statistic) if len(sds) > 10 else float("nan")
        prf = lambda a, b_, c: (a / max(a + b_, 1), a / max(a + c, 1))  # noqa: E731
        result[arm] = {
            "params": ch, "verdict": "WORKS" if ok_f1 and ok_m else "PARTIAL" if ok_f1 or ok_m else "FAILS",
            "consensus": {m: ci(mt[m][0]) for m in ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")} | {"FD": fd, "morph_corr@GT": ci(morph_c)},
            "single_sample": {"HR": ci(one["nfe1_HR"]), "Rpeak_F1": ci(one["nfe1_Rpeak_F1"]), "RR_MAE_ms": ci(one["nfe1_RR_MAE_ms"]),
                              "FD": float(one["nfe1_pooled_FD_kanflow"]), "morph_corr@GT": ci(morph_s)},
            "gain_vs_single": {"Rpeak_F1": d_f1, "morph_corr@GT": d_m, "HR": ci(mt["HR"][0] - one["nfe1_HR"]),
                               "RR_MAE_ms": ci(mt["RR_MAE_ms"][0] - one["nfe1_RR_MAE_ms"])},
            "vs_PENGUIN50": {"HR": ci(mt["HR"][0] - sr1["C"]["nfe50_HR"]), "Rpeak_F1": ci(mt["Rpeak_F1"][0] - sr1["C"]["nfe50_Rpeak_F1"])},
            "vs_HR_median_consensus": ci(mt["HR"][0] - one["cons16_nfe1_hr_err"]),
            "reliability": {"spearman_voteSD_vs_abs_timing_err": rho, "n_matched": len(sds),
                            "all_events_precision_recall": prf(tp, fp, fn),
                            "votefrac>=0.75_precision_recall": (tp_h / max(tp_h + fp_h, 1), tp_h / max(n_gt, 1))}}
        print(f"[ed1-test] {arm} {result[arm]['verdict']}  F1 gain {d_f1}  morph gain {d_m}", flush=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
