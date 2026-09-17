"""P1 -- consensus sampling on U2 validation subjects (docs/P1_CONSENSUS_SAMPLING_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402

FS, PER_SUBJECT, TOL_MS, REFRACT_MS = 128, 2048, 50.0, 250.0
PLAN = {("C", 50): 4, ("C", 1): 16, ("I", 1): 16, ("I", 2): 16}   # (arm, nfe): number of seeds
OUT_A, OUT_O = ROOT / "artifacts/p1_consensus", ROOT / "outputs/p1_consensus"


def load_val(slug):
    man = json.loads((ROOT / "data/manifests" / f"split_{slug}_seed42.json").read_text())
    split = man["splits"][0]
    if slug == "u2_wildppg":
        real = man["extra"]["wildppg_subject_map"]
        for s in split["val"] + split["test"]:
            if s in split["val"]:
                assert not any(b in real[s] for b in ("kjd", "ssx")), f"FIREWALL: {real[s]}"
    X, Y, S = [], [], []
    for s in split["val"]:                      # test subjects are never opened
        d = np.load(ROOT / "data/processed" / slug / f"{s}.npz")
        idx = np.unique(np.linspace(0, len(d["x"]) - 1, min(PER_SUBJECT, len(d["x"]))).round().astype(int))
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float64)); S.append(np.full(len(idx), s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), split["val"]


def consensus_peaks(peak_lists, n_samples, K):
    """Amendment 1: count samples voting within +-TOL_MS -> local maxima with >= K/2 votes, >= REFRACT_MS apart;
    position = median of the votes inside that +-TOL_MS box."""
    votes = np.zeros(n_samples)
    allp = np.concatenate([np.asarray(p, int) for p in peak_lists if len(p)] or [np.zeros(0, int)])
    np.add.at(votes, np.clip(allp, 0, n_samples - 1), 1.0)
    h = int(round(TOL_MS / 1000 * FS))
    cnt = np.convolve(votes, np.ones(2 * h + 1), mode="same")
    cand = [i for i in range(n_samples) if cnt[i] >= K / 2 and (i == 0 or cnt[i] >= cnt[i - 1])
            and (i == n_samples - 1 or cnt[i] > cnt[i + 1])]
    cand.sort(key=lambda i: -cnt[i]); keep, gap = [], REFRACT_MS / 1000 * FS
    for i in cand:
        c = float(np.median(allp[np.abs(allp - i) <= h]))
        if all(abs(c - j) >= gap for j in keep):
            keep.append(c)
    return np.array(sorted(keep))


def f1_of(ref, hyp):
    m, fp, fn = R.match_rpeaks(ref, hyp, FS, TOL_MS)
    return R.prf(len(m), fp, fn)[2]


def macro(vals, S):
    return float(np.mean([np.nanmean(vals[S == s]) for s in np.unique(S)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True, choices=["u2_dalia", "u2_wildppg"])
    slug = ap.parse_args().corpus
    dev = torch.device("cuda")
    X, Y, S, subs = load_val(slug)
    n, T = X.shape
    ref_peaks = PMX.detect_batch(Y, FS)
    ref_hr = np.array([R.hr_bpm(p, FS) for p in ref_peaks])
    print(f"[p1] {slug} val={subs} windows={n}", flush=True)
    nets = {a: U2.build(ROOT / f"outputs/{slug}_arm{a}_seed42/checkpoint_last.pt", dev)[0] for a in ("C", "I")}
    OUT_O.mkdir(parents=True, exist_ok=True); OUT_A.mkdir(parents=True, exist_ok=True)

    hr_pred, peaks, gen_s = {}, {}, {}
    for (arm, nfe), n_seeds in PLAN.items():
        H, P, secs = np.full((n_seeds, n), np.nan), [], 0.0
        samples = np.zeros((n_seeds, n, T), np.float16)
        for seed in range(n_seeds):
            e0 = torch.randn(n, 1, T, generator=torch.Generator().manual_seed(seed))
            z, s = U2.generate(nets[arm], X, e0, nfe, dev, arm); secs += s
            samples[seed] = z
            pk = PMX.detect_batch(z, FS); P.append(pk)
            H[seed] = [R.hr_bpm(p, FS) for p in pk]
        np.savez_compressed(OUT_O / f"{slug}_{arm}{nfe}.npz", samples=samples, subjects=S)
        hr_pred[(arm, nfe)], peaks[(arm, nfe)], gen_s[(arm, nfe)] = H, P, secs / n_seeds
        print(f"[p1] {slug} {arm}{nfe}: {n_seeds} seeds, {secs / n_seeds:.1f} s/seed", flush=True)

    rows = []
    for (arm, nfe), H in hr_pred.items():
        P = peaks[(arm, nfe)]
        single_hr = np.nanmean(np.abs(H[:4] - ref_hr), axis=0)
        single_f1 = np.mean([[f1_of(ref_peaks[w], P[s][w]) for w in range(n)] for s in range(4)], axis=0)
        rows.append(dict(arm=arm, nfe=nfe, K=1, nfe_cost=nfe, hr_err=macro(single_hr, S), f1=macro(single_f1, S)))
        for K in (4, 16):
            if K > len(H) or (arm, nfe, K) == ("C", 50, 16):
                continue
            with np.errstate(all="ignore"):
                cons_hr = np.abs(np.nanmedian(H[:K], axis=0) - ref_hr)
            cp = [consensus_peaks([P[s][w] for s in range(K)], T, K) for w in range(n)]
            cons_f1 = np.array([f1_of(ref_peaks[w], cp[w]) for w in range(n)])
            cons_hr_peaks = np.abs(np.array([R.hr_bpm(c, FS) for c in cp]) - ref_hr)
            rows.append(dict(arm=arm, nfe=nfe, K=K, nfe_cost=nfe * K, hr_err=macro(cons_hr, S), f1=macro(cons_f1, S),
                             hr_err_from_consensus_peaks=macro(cons_hr_peaks, S)))
    for r in rows:
        r["label"] = f"{'PENGUIN' if r['arm'] == 'C' else 'iMF'} NFE{r['nfe']} K={r['K']}"
        r["gen_s_per_seed"] = round(gen_s[(r['arm'], r['nfe'])], 1)
        print(f"[p1] {slug} {r['label']:<22} cost={r['nfe_cost']:>4}  HR={r['hr_err']:.3f}  F1={r['f1']:.4f}", flush=True)
    json.dump(dict(corpus=slug, val_subjects=subs, windows=int(n), rows=rows,
                   ref_hr_nan=int(np.isnan(ref_hr).sum())), open(OUT_A / f"{slug}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
