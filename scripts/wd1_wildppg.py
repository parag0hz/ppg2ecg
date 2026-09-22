"""WD1 — WildPPG: direct HR regression (DB1's model, unchanged) and width-vs-depth at B = 32, over the WP1 folds."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import db1_discriminative_hr as DB  # noqa: E402
import dw1_depth_width as D  # noqa: E402
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import wp1_evaluate as WP  # noqa: E402
from cd1_train import f_consistency  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import rpeaks as R  # noqa: E402
from ppg2ecg.flow.samplers import euler_sample  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.utils.seed import seed_everything  # noqa: E402

OUT, RAW, FS = ROOT / "artifacts/wd1_wildppg", ROOT / "outputs/wd1_raw", 128
CONDS = [(32, 1), (16, 2), (1, 32)]
NAME = {"I": "iMF", "D": "consistency distillation", "C": "PENGUIN (Euler)"}


def load_train(k):
    split = json.loads((ROOT / f"data/manifests/split_wp1_fold{k}.json").read_text())["splits"][0]
    X, Y = [], []
    for s in split["train"]:
        d = np.load(ROOT / f"data/processed/u2_wildppg/{s}.npz")
        X.append(d["x"].astype(np.float32)); Y.append(d["y"].astype(np.float64))
    return np.concatenate(X), np.concatenate(Y)


def make_sampler(ck_path, arm, dev):
    if arm == "D":
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
    else:
        net = U2.build(ck_path, dev)[0]

    @torch.no_grad()
    def sample(X, seed, S, BS=512):
        g = torch.Generator().manual_seed(seed)
        z0 = torch.randn(len(X), 1, X.shape[1], generator=g)
        extra = [torch.randn(len(X), 1, X.shape[1], generator=g) for _ in range(S - 1)] if arm == "D" else []
        out = []
        for i in range(0, len(X), BS):
            ppg = torch.from_numpy(X[i:i + BS]).to(dev).unsqueeze(1); z = z0[i:i + BS].to(dev)
            if arm == "I":
                x, _ = ER.sample_meanflow_schedule(net, ppg, z, [1.0 / S] * S)
            elif arm == "C":
                x, _ = euler_sample(lambda a, t, _p=ppg: net.forward_step(a, _p, t), z, S)
            else:
                B = z.shape[0]; x = f_consistency(net, z, ppg, torch.zeros(B, 1, device=dev))
                for j, e in enumerate(extra, 1):
                    tv = j / S; x = f_consistency(net, (1 - tv) * e[i:i + BS].to(dev) + tv * x, ppg, torch.full((B, 1), tv, device=dev))
            out.append(x[:, 0].float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    return sample


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("part", choices=["A", "B"])
    part = ap.parse_args().part
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda"); ex = ProcessPoolExecutor(8)
    allrows, per_subj = [], {}
    for k in range(4):
        X, Y, subj = WP.load_test(k)
        ref = V.hr_batch(ex, Y)
        if part == "A":
            seed_everything(42, deterministic=False)
            Xtr, Ytr = load_train(k)
            hr_tr = V.hr_batch(ex, Ytr); ok = np.isfinite(hr_tr)
            Xt = torch.from_numpy(Xtr[ok]).to(dev); Ht = torch.from_numpy((hr_tr[ok] / 100).astype(np.float32)).to(dev)
            net = DB.HRNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
            g = torch.Generator().manual_seed(42); perm, pos = torch.randperm(len(Xt), generator=g), 0
            net.train(); t0 = time.time()
            for step in range(DB.STEPS):
                if pos + DB.BATCH > len(perm):
                    perm, pos = torch.randperm(len(Xt), generator=g), 0
                b = perm[pos:pos + DB.BATCH].to(dev); pos += DB.BATCH
                loss = (net(Xt[b]) - Ht[b]).abs().mean(); opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pred = np.concatenate([100 * net(torch.from_numpy(X[i:i + 1024]).to(dev)).float().cpu().numpy() for i in range(0, len(X), 1024)])
            torch.save({"state_dict": net.state_dict()}, RAW / f"hrnet_fold{k}.pt")
            e = {"direct HR regression": np.abs(pred - ref),
                 "PPG peak counting": np.abs(np.array([R.hr_bpm(find_peaks(x, distance=42, prominence=0.3)[0], FS) for x in X.astype(np.float64)]) - ref)}
            C, I, Dd = (dict(np.load(ROOT / f"outputs/wp1_eval/fold{k}_arm{a}.npz")) for a in "CID")
            assert np.array_equal(C["subj"], subj)
            e.update({"PENGUIN 50 NFE (one sample)": C["nfe50_HR"], "iMF K=16 consensus": I["cons16_nfe1_hr_err"], "CD K=16 consensus": Dd["cons16_nfe1_hr_err"]})
            np.savez(RAW / f"partA_fold{k}.npz", subj=subj, ref=ref, **{kk.replace(" ", "_"): v for kk, v in e.items()})
            for name, v in e.items():
                allrows.append(dict(fold=k, arm=name, subj=subj, err=v))
            print(f"[wd1-A] fold {k} regressor {np.nanmean(e['direct HR regression']):.3f} train {time.time() - t0:.0f}s", flush=True)
        else:
            for arm in ("I", "D", "C"):
                sampler = make_sampler(ROOT / f"outputs/wp1_f{k}_arm{arm}/checkpoint_last.pt", arm, dev)
                for K, S in CONDS:
                    H = np.stack([V.hr_batch(ex, sampler(X, s, S)) for s in range(K)])
                    with np.errstate(all="ignore"):
                        err = np.abs(np.nanmedian(H, 0) - ref)
                    np.save(RAW / f"partB_fold{k}_{arm}_K{K}_S{S}.npy", err)
                    allrows.append(dict(fold=k, arm=f"{NAME[arm]} (K {K}, S {S})", subj=subj, err=err))
                    print(f"[wd1-B] fold {k} {NAME[arm]} (K {K}, S {S}) HR {np.nanmean(err):.3f}", flush=True)
                del sampler; torch.cuda.empty_cache()
    # pool over folds: every subject exactly once
    arms = sorted({r["arm"] for r in allrows})
    subj_all = np.concatenate([r["subj"] for r in allrows if r["arm"] == arms[0]])
    E = {a: np.concatenate([r["err"] for r in sorted((x for x in allrows if x["arm"] == a), key=lambda x: x["fold"])]) for a in arms}
    ci = lambda v: V.cluster_ci(v, subj_all)  # noqa: E731
    subs = np.unique(subj_all)
    pp = lambda v: np.array([np.nanmean(v[subj_all == s]) for s in subs])  # noqa: E731
    res = {"arms": {a: {"hr_err": ci(v), "per_subject": dict(zip(subs.tolist(), pp(v).round(3).tolist())),
                        "per_fold": {int(r["fold"]): float(np.nanmean(r["err"])) for r in allrows if r["arm"] == a}} for a, v in E.items()}}
    if part == "A":
        reg = E["direct HR regression"]
        res["regressor_minus"] = {a: ci(reg - v) for a, v in E.items() if a != "direct HR regression"}
        res["outcome_vs_iMF_consensus"] = ("A regressor better" if res["regressor_minus"]["iMF K=16 consensus"][2] < 0 else
                                           "B consensus better" if res["regressor_minus"]["iMF K=16 consensus"][1] > 0 else "C comparable")
        res["win_rate_regressor_vs_iMF_consensus"] = float(np.mean(pp(reg) < pp(E["iMF K=16 consensus"])))
    else:
        res["width_minus_depth"] = {}
        for arm in ("I", "D", "C"):
            w, b, d = (E[f"{NAME[arm]} (K {K}, S {S})"] for K, S in CONDS)
            res["width_minus_depth"][NAME[arm]] = {"(32,1)-(1,32)": ci(w - d), "(16,2)-(1,32)": ci(b - d), "(16,2)-(32,1)": ci(b - w),
                                                   "width_beats_depth": bool(ci(w - d)[2] < 0), "win_rate_width_vs_depth": float(np.mean(pp(w) < pp(d)))}
    (OUT / f"part_{part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
    for a, v in res["arms"].items():
        print(f"{a:40s} HR {v['hr_err'][0]:.3f} [{v['hr_err'][1]:.3f}, {v['hr_err'][2]:.3f}]  folds {[round(x, 2) for x in v['per_fold'].values()]}")


if __name__ == "__main__":
    main()
