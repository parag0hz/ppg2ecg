"""ED2 — consensus decoding on WildPPG (part A) and on VitalDB short-term HRV (part B). No training."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cd1_evaluate as CDE  # noqa: E402
import ed1_consensus_decode as E1  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import wp1_evaluate as WP  # noqa: E402
from pz3_compare import morph_at  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

FS, K = 128, 16
OUT = ROOT / "artifacts/ed2_decoding_transfer"
PAR = json.loads((ROOT / "artifacts/ed1_consensus_decoding/params.json").read_text())


def make_gen(ck_path, arm, dev):
    if arm == "D":
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
        return lambda X, s, nfe=1: CDE.sample(net, X, s, nfe, dev)[0]
    net, _, kind = U2.build(ck_path, dev)
    def gen(X, s, nfe=1, _n=net, _k=kind):
        e0 = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(s))
        return U2.generate(_n, X, e0, nfe, dev, _k)[0]
    return gen


def sample_k(gen, X, ex):
    S = np.stack([gen(X, s).astype(np.float32) for s in range(K)])
    pk = list(ex.map(E1._peaks, list(S.reshape(-1, S.shape[-1]).astype(np.float64)), chunksize=256))
    n = S.shape[1]
    return S, [[pk[k * n + i] for k in range(K)] for i in range(n)]


def decode(S, P, arm, ex):
    ch = PAR[arm]["chosen"]
    return E1.decode_all(S, P, int(round(ch["w_ms"] / 1000 * FS)), ch["theta"], ch["b"], ex)


def part_a(ex, dev):
    res = {}
    for arm in ("I", "D"):
        rows = {k: [] for k in ("subj", "f1_c", "f1_s", "hr_c", "hr_s", "rr_c", "rr_s", "m_c", "m_s", "f1_p50", "hr_p50")}
        fds = []
        for k in range(4):
            X, Y, subj = WP.load_test(k)
            gt = list(ex.map(E1._peaks, list(Y), chunksize=256))
            S, P = sample_k(make_gen(ROOT / f"outputs/wp1_f{k}_arm{arm}/checkpoint_last.pt", arm, dev), X, ex)
            wave, _, _, _ = decode(S, P, arm, ex)
            mt = U2.task_metrics("ECG", 8, wave, Y, subj); tab = mt.pop("_counts")[0]
            fds.append(U2.pooled_extras("ECG", wave, Y, tab)["FD_kanflow"])
            one = dict(np.load(ROOT / f"outputs/wp1_eval/fold{k}_arm{arm}.npz")); c50 = dict(np.load(ROOT / f"outputs/wp1_eval/fold{k}_armC.npz"))
            assert np.array_equal(one["subj"], subj)
            rows["subj"].append(subj)
            rows["f1_c"].append(mt["Rpeak_F1"][0]); rows["f1_s"].append(one["nfe1_Rpeak_F1"])
            rows["hr_c"].append(mt["HR"][0]); rows["hr_s"].append(one["nfe1_HR"])
            rows["rr_c"].append(mt["RR_MAE_ms"][0]); rows["rr_s"].append(one["nfe1_RR_MAE_ms"])
            rows["m_c"].append(np.array(list(ex.map(morph_at, zip(wave, Y, gt), chunksize=128))))
            rows["m_s"].append(np.nanmean(np.vstack([np.array(list(ex.map(morph_at, zip(S[j].astype(np.float64), Y, gt), chunksize=128))) for j in range(4)]), 0))
            rows["f1_p50"].append(c50["nfe50_Rpeak_F1"]); rows["hr_p50"].append(c50["nfe50_HR"])
            print(f"[ed2-A] {arm} fold {k} done", flush=True)
        r = {k: np.concatenate(v) for k, v in rows.items()}
        ci = lambda v: V.cluster_ci(v, r["subj"])  # noqa: E731
        d_f1, d_m = ci(r["f1_c"] - r["f1_s"]), ci(r["m_c"] - r["m_s"])
        ok1, ok2 = d_f1[0] >= 0.05 and d_f1[1] > 0, d_m[0] >= 0.15 and d_m[1] > 0
        res[arm] = {"verdict": "WORKS" if ok1 and ok2 else "PARTIAL" if ok1 or ok2 else "FAILS",
                    "consensus": {"Rpeak_F1": ci(r["f1_c"]), "HR": ci(r["hr_c"]), "RR_MAE_ms": ci(r["rr_c"]), "morph_corr@GT": ci(r["m_c"]), "FD_per_fold": fds},
                    "single": {"Rpeak_F1": ci(r["f1_s"]), "HR": ci(r["hr_s"]), "RR_MAE_ms": ci(r["rr_s"]), "morph_corr@GT": ci(r["m_s"])},
                    "gain": {"Rpeak_F1": d_f1, "morph_corr@GT": d_m, "RR_MAE_ms": ci(r["rr_c"] - r["rr_s"]), "HR": ci(r["hr_c"] - r["hr_s"])},
                    "vs_PENGUIN50": {"Rpeak_F1": ci(r["f1_c"] - r["f1_p50"]), "HR": ci(r["hr_c"] - r["hr_p50"])}}
        print(f"[ed2-A] {arm} {res[arm]['verdict']} F1 gain {d_f1} morph gain {d_m}", flush=True)
    return res


def rr_of(peaks):
    rr = np.diff(np.asarray(peaks, float)) / FS * 1000.0
    return rr[(rr >= 300) & (rr <= 2000)]


def hrv_block(peak_lists):
    """peak_lists: one array per window of the block -> (mean RR, SDNN, RMSSD) or NaNs."""
    rrs, dd = [], []
    for p in peak_lists:
        rr = rr_of(p)
        rrs += list(rr)
        if len(rr) > 1:
            dd += list(np.diff(rr))
    if len(rrs) < 30 or len(dd) < 5:
        return np.nan, np.nan, np.nan
    return float(np.mean(rrs)), float(np.std(rrs)), float(np.sqrt(np.mean(np.square(dd))))


def part_b(ex, dev):
    d = np.load(ROOT / "outputs/ed2_hrv_blocks.npz")
    X, Y, case, pid = d["x"], d["y"].astype(np.float64), d["case"], d["pid"]
    blocks = [np.flatnonzero(case == c) for c in dict.fromkeys(case)]
    bpid = np.array([pid[b[0]] for b in blocks])
    peaks = {"REF": list(ex.map(E1._peaks, list(Y), chunksize=256)),
             "PRV (PPG peaks, no model)": [find_peaks(x, distance=42, prominence=0.3)[0] for x in X.astype(np.float64)]}
    gC = make_gen(ROOT / "outputs/v1_vitaldb_armC_seed42/checkpoint_last.pt", "C", dev)
    peaks["PENGUIN-50"] = list(ex.map(E1._peaks, list(gC(X, 0, 50)), chunksize=256)); print("[ed2-B] PENGUIN-50 done", flush=True)
    for arm, name, path in (("I", "iMF", "outputs/v1_vitaldb_armI_seed42"), ("D", "CD", "outputs/cd1_vitaldb_armD_seed42")):
        S, P = sample_k(make_gen(ROOT / path / "checkpoint_last.pt", arm, dev), X, ex)
        peaks[f"{name}-1 single"] = [P[i][0] for i in range(len(X))]
        _, pos, _, _ = decode(S, P, arm, ex)
        peaks[f"{name} consensus-decoded"] = [np.round(p).astype(int) for p in pos]
        print(f"[ed2-B] {name} done", flush=True)
    H = {a: np.array([hrv_block([pk[i] for i in b]) for b in blocks]) for a, pk in peaks.items()}
    ref = H.pop("REF")
    out = {"n_blocks": len(blocks), "n_patients": int(len(np.unique(bpid))), "n_scored_ref": int(np.isfinite(ref[:, 1]).sum()),
           "skipped_cases": int(len(d["skipped"])), "arms": {}}
    err = {}
    for a, h in H.items():
        ok = np.isfinite(h[:, 1]) & np.isfinite(ref[:, 1])
        err[a] = np.where(ok[:, None], np.abs(h - ref), np.nan)
        out["arms"][a] = {"n_scored": int(ok.sum()),
                          "meanRR_err_ms": V.cluster_ci(err[a][:, 0], bpid), "SDNN_err_ms": V.cluster_ci(err[a][:, 1], bpid),
                          "RMSSD_err_ms": V.cluster_ci(err[a][:, 2], bpid),
                          "SDNN_spearman": float(spearmanr(h[ok, 1], ref[ok, 1]).statistic),
                          "RMSSD_spearman": float(spearmanr(h[ok, 2], ref[ok, 2]).statistic)}
    dci = lambda a, b: V.cluster_ci(err[a][:, 1] - err[b][:, 1], bpid)  # noqa: E731
    out["claims"] = {"H-HRV1_iMF_decoded_vs_single": dci("iMF consensus-decoded", "iMF-1 single"),
                     "H-HRV1_CD_decoded_vs_single": dci("CD consensus-decoded", "CD-1 single"),
                     "H-HRV2_iMF_decoded_vs_PENGUIN50": dci("iMF consensus-decoded", "PENGUIN-50"),
                     "iMF_decoded_vs_PRV": dci("iMF consensus-decoded", "PRV (PPG peaks, no model)")}
    out["reference"] = {"SDNN_median_ms": float(np.nanmedian(ref[:, 1])), "RMSSD_median_ms": float(np.nanmedian(ref[:, 2]))}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("part", choices=["A", "B"])
    part = ap.parse_args().part
    OUT.mkdir(parents=True, exist_ok=True)
    ex, dev = ProcessPoolExecutor(10), torch.device("cuda")
    res = part_a(ex, dev) if part == "A" else part_b(ex, dev)
    (OUT / f"part_{part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
