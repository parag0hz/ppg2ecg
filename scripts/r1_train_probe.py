"""R1 step 8 — train one probe (Global-TCN or Local-TCN). PPG in, soft R-event field out.

Frozen protocol c7481f9. Seed 42 shared; identical init (identical shapes), identical example order.
Early stopping on INTERNAL_DEV BCE only. an0/k2s are never touched here.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse, csv, json, subprocess, time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.probes import r1_cohort as C
from ppg2ecg.probes.rhythm_tcn import GLOBAL_DILATIONS, LOCAL_DILATIONS, RhythmTCN, n_trainable, soft_event_field
from ppg2ecg.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/r1_global_rhythm"
FS, T_LEN = 128, 1024
SEED, BATCH, LR, WD, MAX_EPOCHS, PATIENCE = 42, 128, 1e-3, 1e-4, 30, 5
BUDGET_H = 6.0


def _peaks(y):
    return R.detect_rpeaks(np.asarray(y, dtype=np.float64), FS)


def build_arrays(subjects, tag):
    """PPG inputs + soft targets from GT R-peaks. Cached; the ECG itself is discarded after labelling."""
    cache = ART / f"_cache_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["y"], z["site"]
    X, Y, S = [], [], []
    for sub in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{sub}.npz")
        Xs, Ys, WIs = d["x"], d["y"], d["window_index"]   # decompress ONCE: every d[key] access re-reads the npz
        pos = C.cohort_positions(sub, d["site"], d["window_index"], C.n_per_for(sub))
        for si, site in enumerate(C.SITES):
            idx = pos[site]
            ecg = [Ys[int(i)] for i in idx]
            with ProcessPoolExecutor(max_workers=12) as ex:
                pk = list(ex.map(_peaks, ecg, chunksize=32))
            X.append(Xs[idx].astype(np.float32))
            Y.append(np.stack([soft_event_field(p, T_LEN) for p in pk]))
            S.append(np.full(len(idx), si, dtype=np.int64))
        print(f"[L] {tag}: {sub} labelled", flush=True)
    X, Y, S = np.concatenate(X), np.concatenate(Y), np.concatenate(S)
    np.savez_compressed(cache, x=X, y=Y, site=S)
    return X, Y, S


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["global", "local", "global_site"], required=True)
    a = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    split = C.internal_dev_split()
    assert_no_test_subjects(split["probe_train"] + split["internal_dev"])
    out = ROOT / f"outputs/r1_{a.variant}_tcn_seed{SEED}"
    assert not (out / "checkpoint_best.pt").exists(), f"refusing to overwrite {out}"
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    Xtr, Ytr, Str = build_arrays(split["probe_train"], "probe_train")
    Xdv, Ydv, Sdv = build_arrays(split["internal_dev"], "internal_dev")
    print(f"[D] train {len(Xtr)} | internal-dev {len(Xdv)} | ecg arrays discarded after labelling", flush=True)

    seed_everything(SEED, deterministic=True)
    dev = torch.device("cuda")
    dil = LOCAL_DILATIONS if a.variant == "local" else GLOBAL_DILATIONS
    torch.manual_seed(SEED)                                    # identical init across variants (same shapes)
    net = RhythmTCN(dil, n_sites=4 if a.variant == "global_site" else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED)                  # identical example order across variants
    xt, yt, st = torch.from_numpy(Xtr), torch.from_numpy(Ytr), torch.from_numpy(Str)
    xd, yd, sd = torch.from_numpy(Xdv).to(dev), torch.from_numpy(Ydv).to(dev), torch.from_numpy(Sdv).to(dev)

    def dev_loss():
        net.eval(); tot = 0.0
        with torch.no_grad():
            for i in range(0, len(xd), 512):
                lg = net(xd[i:i + 512].unsqueeze(1), sd[i:i + 512] if net.n_sites else None).squeeze(1)
                tot += lossf(lg, yd[i:i + 512]).item() * (min(i + 512, len(xd)) - i)
        net.train(); return tot / len(xd)

    log, best, bad, step, t0 = [], float("inf"), 0, 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(f"[M] {a.variant}: rf={net.rf} params={n_trainable(net):,}", flush=True)
    for ep in range(MAX_EPOCHS):
        perm = torch.randperm(len(xt), generator=gen)
        run = 0.0; n = 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x = xt[b].to(dev, non_blocking=True).unsqueeze(1); y = yt[b].to(dev, non_blocking=True)
            s = st[b].to(dev) if net.n_sites else None
            lg = net(x, s).squeeze(1)
            loss = lossf(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            run += loss.item() * len(b); n += len(b); step += 1
            if step == 100:
                per = (time.perf_counter() - t0) / 100
                proj = per * (len(perm) // BATCH + 1) * MAX_EPOCHS / 3600
                print(f"[P] {per*1000:.1f} ms/step -> projected max {proj:.2f} h for this probe", flush=True)
                if proj * 2 > BUDGET_H:
                    print(f"[P] STOP: projection for both probes {proj*2:.2f} h > {BUDGET_H} h budget", flush=True)
                    return 2
        dl = dev_loss()
        row = {"epoch": ep, "train_bce": run / n, "internal_dev_bce": dl, "steps": step,
               "elapsed_s": time.perf_counter() - t0, "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2 ** 20}
        log.append(row)
        improved = dl < best - 1e-6
        if improved:
            best, bad = dl, 0
            torch.save({"state_dict": net.state_dict(), "epoch": ep, "variant": a.variant, "dilations": list(dil),
                        "n_sites": net.n_sites, "internal_dev_bce": dl, "seed": SEED, "git": head,
                        "params": n_trainable(net), "rf": net.rf}, out / "checkpoint_best.pt")
        else:
            bad += 1
        print(f"[E] ep {ep:2d} train {row['train_bce']:.5f} dev {dl:.5f} {'*' if improved else ''} "
              f"{row['elapsed_s']:.0f}s peak {row['peak_mem_MiB']:.0f}MiB", flush=True)
        if bad >= PATIENCE:
            print(f"[E] early stop (patience {PATIENCE})", flush=True); break
    with open(ART / f"training_log_{a.variant}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    (out / "training_summary.json").write_text(json.dumps({
        "variant": a.variant, "epochs_run": len(log), "best_internal_dev_bce": best,
        "best_epoch": int(np.argmin([r["internal_dev_bce"] for r in log])), "steps": step,
        "wall_s": time.perf_counter() - t0, "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2 ** 20,
        "params": n_trainable(net), "rf": net.rf, "seed": SEED, "git": head,
        "finished": datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f"[done] {a.variant}: best dev BCE {best:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
