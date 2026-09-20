"""MC1: one corpus, all arms, every metric (docs/MC1_MULTI_CORPUS_PREREGISTRATION.md)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ed1_consensus_decode as E1  # noqa: E402
import ed2_run as E2  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
from pz3_compare import morph_at  # noqa: E402

CAP, FS = 3000, 128
OUT = ROOT / "artifacts/mc1_multicorpus"
METRICS = ("HR", "Rpeak_F1", "RR_MAE_ms", "MAE", "RMSE")


def load_test(corpus):
    man = json.loads((ROOT / f"data/manifests/split_mc1_{corpus}.json").read_text())
    proc = ROOT / man["extra"]["processed"]
    X, Y, S = [], [], []
    for s in man["splits"][0]["test"]:
        d = np.load(proc / f"{s}.npz")
        idx = np.unique(np.linspace(0, len(d["x"]) - 1, min(CAP, len(d["x"]))).round().astype(int))
        X.append(d["x"][idx].astype(np.float32)); Y.append(d["y"][idx].astype(np.float64)); S.append(np.full(len(idx), s))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)


def full_metrics(preds, Y, subj, gt, ex):
    """preds: list of [n,T] waveforms (draws) -> per-window arrays averaged over draws + pooled scalars."""
    acc, pooled, morph = {}, {}, []
    for p in preds:
        mt = U2.task_metrics("ECG", 8, p, Y, subj); tab = mt.pop("_counts")[0]
        for m in METRICS:
            acc.setdefault(m, []).append(np.asarray(mt[m][0], float))
        for m, v in U2.pooled_extras("ECG", p, Y, tab).items():
            pooled.setdefault(m, []).append(v)
        morph.append(np.array(list(ex.map(morph_at, zip(p, Y, gt), chunksize=128))))
    per = {m: np.nanmean(np.vstack(v), 0) for m, v in acc.items()}
    per["morph_corr@GT"] = np.nanmean(np.vstack(morph), 0)
    return per, {m: float(np.mean(v)) for m, v in pooled.items() if m != "_fd_n"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True)
    corpus = ap.parse_args().corpus
    OUT.mkdir(parents=True, exist_ok=True)
    ex, dev = ProcessPoolExecutor(10), torch.device("cuda")
    X, Y, subj = load_test(corpus)
    gt = list(ex.map(E1._peaks, list(Y), chunksize=256))
    ref_hr = V.hr_batch(ex, Y)
    ci = lambda v: V.cluster_ci(v, subj)  # noqa: E731
    arms, per_all = {}, {}
    gC = E2.make_gen(ROOT / f"outputs/mc1_{corpus}_armC/checkpoint_last.pt", "C", dev)
    for nfe in (50, 1):
        per, pooled = full_metrics([gC(X, s, nfe) for s in range(4)], Y, subj, gt, ex)
        per_all[f"PENGUIN-{nfe}"] = per
        arms[f"PENGUIN-{nfe}"] = {m: ci(v) for m, v in per.items()} | pooled
        print(f"[mc1] {corpus} PENGUIN-{nfe} done", flush=True)
    for arm, name in (("I", "iMF"), ("D", "CD")):
        S, P = E2.sample_k(E2.make_gen(ROOT / f"outputs/mc1_{corpus}_arm{arm}/checkpoint_last.pt", arm, dev), X, ex)
        per, pooled = full_metrics([S[j].astype(np.float64) for j in range(4)], Y, subj, gt, ex)
        per_all[f"{name}-1"] = per
        arms[f"{name}-1"] = {m: ci(v) for m, v in per.items()} | pooled
        hr = np.array([[E1.R.hr_bpm(p, FS) for p in P[i]] for i in range(len(X))]).T          # [K, n]
        with np.errstate(all="ignore"):
            cons = np.abs(np.nanmedian(hr, 0) - ref_hr)
        per_all[f"{name} K16 HR"] = {"HR": cons}
        arms[f"{name}-1 K16 HR-median"] = {"HR": ci(cons)}
        wave, _, _, _ = E2.decode(S, P, arm, ex)
        perd, pooledd = full_metrics([wave], Y, subj, gt, ex)
        per_all[f"{name} decoded"] = perd
        arms[f"{name} consensus-decoded"] = {m: ci(v) for m, v in perd.items()} | pooledd
        print(f"[mc1] {corpus} {name} done", flush=True)
    c50 = per_all["PENGUIN-50"]
    claims = {"H1_iMF1_vs_P50": {"HR": ci(per_all["iMF-1"]["HR"] - c50["HR"]), "Rpeak_F1": ci(per_all["iMF-1"]["Rpeak_F1"] - c50["Rpeak_F1"])},
              "H2_iMF_K16_vs_P50_HR": ci(per_all["iMF K16 HR"]["HR"] - c50["HR"]),
              "H2b_CD_K16_vs_P50_HR": ci(per_all["CD K16 HR"]["HR"] - c50["HR"]),
              "H-ED_iMF_decoded_vs_single_F1": ci(per_all["iMF decoded"]["Rpeak_F1"] - per_all["iMF-1"]["Rpeak_F1"]),
              "H-ED_CD_decoded_vs_single_F1": ci(per_all["CD decoded"]["Rpeak_F1"] - per_all["CD-1"]["Rpeak_F1"]),
              "decoded_iMF_vs_P50_F1": ci(per_all["iMF decoded"]["Rpeak_F1"] - c50["Rpeak_F1"])}
    res = {"corpus": corpus, "n_test_subjects": int(len(np.unique(subj))), "n_windows": int(len(X)), "arms": arms, "claims": claims,
           "verdicts": {"H2": bool(claims["H2_iMF_K16_vs_P50_HR"][2] < 0), "H2b": bool(claims["H2b_CD_K16_vs_P50_HR"][2] < 0),
                        "H-ED_iMF": bool(claims["H-ED_iMF_decoded_vs_single_F1"][0] >= 0.05 and claims["H-ED_iMF_decoded_vs_single_F1"][1] > 0),
                        "H-ED_CD": bool(claims["H-ED_CD_decoded_vs_single_F1"][0] >= 0.05 and claims["H-ED_CD_decoded_vs_single_F1"][1] > 0)}}
    (OUT / f"{corpus}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"corpus": corpus, "verdicts": res["verdicts"], "claims": claims}, indent=1), flush=True)


if __name__ == "__main__":
    main()
