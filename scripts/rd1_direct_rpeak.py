"""RD1 — direct PPG -> R-peak detector on VitalDB (docs/RD1_DIRECT_RPEAK_DETECTOR_PREREGISTRATION.md).

The repository's R1 Global-TCN, unchanged; Gaussian targets (sigma = 20 ms) at the reference R peaks; BCE; peaks
extracted with R1's frozen rule (threshold 0.35, refractory 32 samples). Scored with the standard per-window pipeline
(`paper_metrics.rpeak_prf_at` / `beat_level_metrics` fed the detector's peak sequence) and compared, paired per window,
with iMF single, iMF / CD consensus-decoded (ED1 frozen params, recomputed here from the ED1 sample cache) and
PENGUIN 50 NFE (SR1 seed-42 arrays).
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ed1_consensus_decode as ED  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation.rpeaks import detect_rpeaks  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events, n_trainable, soft_event_field  # noqa: E402
from ppg2ecg.utils.seed import seed_everything  # noqa: E402

OUT, RUN = ROOT / "artifacts/rd1_direct_rpeak", ROOT / "outputs/rd1_detector"
FS, T = 128, 512                                     # V1 VitalDB windows are 4 s
STEPS, BATCH, LR, WD, SEED = 14000, 64, 1e-3, 0.01, 42
SIGMA_MS, THRESHOLD, REFRACTORY = 20.0, 0.35, 32
SIGMA = SIGMA_MS / 1000.0 * FS                       # 2.56 samples
N_REP, THREADS = 50, 4                               # LAT1 protocol for the CPU batch-1 latency


def _peaks(sig):
    return np.asarray(detect_rpeaks(sig, FS), int)


def _field(r):
    return soft_event_field(r, T, SIGMA)


def peaks_cached(name, Y, ex):
    f = RUN / f"{name}_rpeaks.npz"
    if f.exists():
        d = np.load(f)
        return np.split(d["idx"], d["off"][1:-1])
    P = list(ex.map(_peaks, list(Y), chunksize=256))
    off = np.cumsum([0] + [len(p) for p in P])
    np.savez(f, idx=np.concatenate(P) if P else np.zeros(0, int), off=off)
    return P


def train(ex, dev):
    Xtr, Ytr, _ = VM.load("train"); assert Xtr.shape[1] == T
    P = peaks_cached("train", Ytr, ex)
    F = np.stack(list(ex.map(_field, P, chunksize=512))).astype(np.float16)
    Xt = torch.from_numpy(Xtr.astype(np.float32)).to(dev)
    Ft = torch.from_numpy(F).to(dev)
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    perm, pos, t0, acc = torch.randperm(len(Xt), generator=g), 0, time.time(), []
    net.train()
    for step in range(1, STEPS + 1):
        if pos + BATCH > len(perm):
            perm, pos = torch.randperm(len(Xt), generator=g), 0
        b = perm[pos:pos + BATCH].to(dev); pos += BATCH
        loss = lossf(net(Xt[b][:, None])[:, 0], Ft[b].float())
        opt.zero_grad(); loss.backward(); opt.step(); acc.append(loss.item())
        if step % 1000 == 0:
            print(f"[rd1] step {step} BCE {np.mean(acc):.5f}", flush=True); acc = []
    torch.save({"state_dict": net.state_dict(), "opt_steps": STEPS, "sigma_ms": SIGMA_MS, "seed": SEED}, RUN / "checkpoint_last.pt")
    return net, time.time() - t0, len(Xt)


@torch.no_grad()
def predict(net, X, dev, bs=1024):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + bs].astype(np.float32)).to(dev)[:, None])[:, 0]).float().cpu().numpy())
    return np.concatenate(out)


def _extract(p):
    return extract_events(p, THRESHOLD, REFRACTORY)


def detector_metrics(Y, ref, hyp):
    """The standard per-window columns, computed from the detector's peak sequence (Y as a finite placeholder)."""
    prf = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ref, hyp))
    beat = PMX.beat_level_metrics(Y, Y, FS, 50.0, peaks=(ref, hyp))
    return {"Rpeak_F1": prf["rpeak_f1"], "Rpeak_precision": prf["rpeak_precision"], "Rpeak_recall": prf["rpeak_recall"],
            "n_tp": prf["n_tp"], "n_fp": prf["n_fp"], "n_fn": prf["n_fn"], "n_ref_beats": beat["n_ref_beats"],
            "HR": beat["hr_abs_err"], "RR_MAE_ms": beat["rr_mae_ms"]}


def decoded_arm(arm, X, Y, pid, ex, dev):
    """ED1's consensus decoding at the frozen params, scored with the standard pipeline; per-window arrays cached."""
    f = RUN / f"decoded_{arm}_test.npz"
    if f.exists():
        return dict(np.load(f))
    ch = json.loads((ED.OUT / "params.json").read_text())[arm]["chosen"]
    S, P = ED.samples_and_peaks(arm, "test", X, ex, dev)
    wave = ED.decode_all(S, P, int(round(ch["w_ms"] / 1000 * FS)), ch["theta"], ch["b"], ex)[0]
    tab = PMX.paper_metric_table(wave, Y, fs=FS, with_quadratic=False)   # = task_metrics' ECG columns minus the O(T^2) ones
    d = {"HR": tab["hr_abs_err"], "Rpeak_F1": tab["rpeak_f1_50ms"], "RR_MAE_ms": tab["rr_mae_ms"]}
    np.savez(f, **d)
    del S; torch.cuda.empty_cache()
    return d


def latency(net, x):
    res = {}
    torch.set_num_threads(THREADS)
    for dev_name in ("cpu", "cuda"):
        dev = torch.device(dev_name)
        n = net.to(dev).eval()
        xt = torch.from_numpy(x).to(dev)[:, None]
        with torch.no_grad():
            for _ in range(3):
                n(xt)
            ts = []
            for _ in range(N_REP):
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter(); n(xt)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
        res[dev_name] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90))}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True); RUN.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED, deterministic=False)
    dev = torch.device("cuda")
    ex = ProcessPoolExecutor(10)
    ck = RUN / "checkpoint_last.pt"
    if ck.exists():
        net = RhythmTCN().to(dev); net.load_state_dict(torch.load(ck, map_location="cpu")["state_dict"])
        train_s, n_train = float("nan"), int(json.loads((OUT / "train_meta.json").read_text())["n_train"])
    else:
        net, train_s, n_train = train(ex, dev)
        (OUT / "train_meta.json").write_text(json.dumps({"train_seconds": round(train_s, 1), "n_train": n_train, "steps": STEPS}))
    n_par = n_trainable(net)

    X, Y, pid = VM.load("test")
    ref = peaks_cached("test", Y, ex)
    t1 = time.time(); prob = predict(net, X, dev); torch.cuda.synchronize(); infer_ms = 1000 * (time.time() - t1) / len(X)
    hyp = list(ex.map(_extract, list(prob), chunksize=256))
    det = detector_metrics(Y, ref, hyp)
    np.savez(RUN / "test_metrics.npz", pid=pid, **det)

    sr1 = {a: dict(np.load(ROOT / f"outputs/sr1_eval/arm_{a}_seed42.npz")) for a in ("I", "C")}
    assert all(np.array_equal(d["pid"], pid) for d in sr1.values())
    dec = {a: decoded_arm(a, X, Y, pid, ex, dev) for a in ("I", "D")}
    arms = {"iMF single (1 NFE)": {"HR": sr1["I"]["nfe1_HR"], "Rpeak_F1": sr1["I"]["nfe1_Rpeak_F1"], "RR_MAE_ms": sr1["I"]["nfe1_RR_MAE_ms"]},
            "iMF consensus-decoded (16 NFE)": dec["I"], "CD consensus-decoded (16 NFE)": dec["D"],
            "PENGUIN 50 NFE (one sample)": {"HR": sr1["C"]["nfe50_HR"], "Rpeak_F1": sr1["C"]["nfe50_Rpeak_F1"], "RR_MAE_ms": sr1["C"]["nfe50_RR_MAE_ms"]}}
    ci = lambda v: V.cluster_ci(v, pid)  # noqa: E731
    M = ("Rpeak_F1", "RR_MAE_ms", "HR")
    per = lambda d: {m: ci(d[m]) for m in M}  # noqa: E731
    win = lambda a, b, better_low: float(np.mean([np.nanmean((a - b)[pid == s]) < 0 if better_low else np.nanmean((a - b)[pid == s]) > 0  # noqa: E731
                                                  for s in np.unique(pid)]))
    res = {"detector": per(det) | {"Rpeak_precision": ci(det["Rpeak_precision"]), "Rpeak_recall": ci(det["Rpeak_recall"]),
                                   "Micro_F1": float(PMX.micro_f1(det["n_tp"], det["n_fp"], det["n_fn"])),
                                   "Macro_F1": float(PMX.macro_f1(det["Rpeak_F1"], det["n_ref_beats"])[0]),
                                   "params": n_par, "train_seconds": train_s, "n_train_windows": n_train,
                                   "ms_per_window_gpu_batch1024": infer_ms, "threshold": THRESHOLD, "refractory_samples": REFRACTORY, "sigma_ms": SIGMA_MS},
           "others": {k: per(v) for k, v in arms.items()},
           "detector_minus_other": {k: {m: ci(det[m] - v[m]) for m in M} for k, v in arms.items()},
           "detector_patient_win_rate": {k: {m: win(det[m], v[m], m != "Rpeak_F1") for m in M} for k, v in arms.items()}}
    res["latency_batch1"] = latency(net, X[:1].astype(np.float32))
    res["verdict"] = {k: {m: ("detector better" if (d[m][2] < 0 if m != "Rpeak_F1" else d[m][1] > 0) else
                              "other better" if (d[m][1] > 0 if m != "Rpeak_F1" else d[m][2] < 0) else "no significant difference")
                          for m in M} for k, d in res["detector_minus_other"].items()}
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
