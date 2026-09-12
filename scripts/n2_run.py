"""N2 — oracle-timing beat-shape probe (docs/N2_ORACLE_TIMING_BEAT_SHAPE_PREREGISTRATION.md).

Every arm receives GROUND-TRUTH R positions at inference. This is an oracle coordinate probe:
no number here is a deployable result. Labelled "(GT-R anchor; oracle coordinate -- diagnostic only)".

Run: .venv/bin/python scripts/n2_run.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppg2ecg.evaluation import event_reliability as ER          # noqa: E402
from ppg2ecg.evaluation import m1_structural as M1              # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP                     # noqa: E402
from ppg2ecg.evaluation import stamping as ST                   # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.probes import beat_shape as BS                     # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

ART = ROOT / "artifacts/n2_beat_shape"
PREREG = "d552eb4"
ORACLE = "(GT-R anchor; oracle coordinate -- diagnostic only)"
FS, T_LEN = BS.FS, 1024
BEAT_LEN, R_IDX, CTX_HALF, CTX_LEN = BS.BEAT_LEN, BS.BEAT_R_INDEX, BS.CTX_HALF, BS.CTX_LEN
BEAT_SALT, BEAT_TAKE = "n2-beat-v1", 1024
EVAL_SALT, EVAL_TAKE = "x4-event-nfe-v2", 1024
STEPS, BATCH, LR, WD, SEED = 6000, 256, 1e-3, 0.01, 42
BOOT_N, BOOT_SEED = 2000, 20260911
MARGIN_VS_TEMPLATE, MARGIN_VS_SHUFFLE = 0.05, 0.025
TEMPLATE_A_FILE_SHA = "1a67569f8a02bc0027c0a60c4575d297dc2bc40eb0c3e285b9acf82daafd51eb"


def split() -> dict:
    return json.loads((ROOT / "artifacts/r1_global_rhythm/subject_split.json").read_text())


def extract_beats(subjects, salt, take, tag):
    """GT R anchors with a complete beat window AND a complete PPG context. Deterministic."""
    ER.assert_no_test_subjects(subjects)
    P, Y, S = [], [], []
    for s in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        X, Yw = d["x"], d["y"]
        idx = ER.select_subset(salt, s, len(X), take)
        for i in idx:
            ppg, ecg = X[int(i)].astype(np.float64), Yw[int(i)].astype(np.float64)
            for r in RP.detect_rpeaks(ecg, FS):
                r = int(r)
                if r - R_IDX < 0 or r - R_IDX + BEAT_LEN > T_LEN:
                    continue
                if r - CTX_HALF < 0 or r + CTX_HALF + 1 > T_LEN:
                    continue
                Y.append(ecg[r - R_IDX: r - R_IDX + BEAT_LEN])
                P.append(ppg[r - CTX_HALF: r + CTX_HALF + 1])
                S.append(s)
    P, Y, S = np.asarray(P, np.float32), np.asarray(Y, np.float32), np.asarray(S)
    print(f"[N2] {tag}: {len(P):,} beats from {len(subjects)} subjects", flush=True)
    return P, Y, S


def train_arm(arm: str, Ptr, Ytr, Pdev, Ydev, dev):
    seed_everything(SEED)
    net = (BS.BeatRegressor() if arm == "REG" else BS.BeatMeanFlow()).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    Pt = torch.from_numpy(Ptr).unsqueeze(1)
    Yt = torch.from_numpy(Ytr)
    g = torch.Generator().manual_seed(SEED)
    # deterministic fixed validation bank for IMF; dev MSE for REG
    dev_p = torch.from_numpy(Pdev).unsqueeze(1).to(dev)
    dev_y = torch.from_numpy(Ydev).to(dev)
    dev_e = torch.randn(len(Pdev), BEAT_LEN, generator=torch.Generator().manual_seed(1000)).to(dev)
    best, best_sd, hist = float("inf"), None, []
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(Pt), (BATCH,), generator=g)
        p, y = Pt[i].to(dev), Yt[i].to(dev)
        if arm == "REG":
            loss = torch.mean((net(p) - y) ** 2)
        else:
            e = torch.randn(BATCH, BEAT_LEN, device=dev)
            t = torch.rand(BATCH, device=dev)
            r = torch.where(torch.rand(BATCH, device=dev) < 0.5, t, t * torch.rand(BATCH, device=dev))
            z = (1 - t).unsqueeze(1) * y + t.unsqueeze(1) * e          # z_t between data (t=0) and noise (t=1)
            v = e - y                                                   # target velocity
            loss = torch.mean((net.u(z, p, t, r) - v) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == STEPS:
            net.eval()
            with torch.no_grad():
                if arm == "REG":
                    m = float(torch.mean((net(dev_p) - dev_y) ** 2))
                else:
                    m = float(torch.mean((BS.sample_one_step(net, dev_p, dev_e) - dev_y) ** 2))
            net.train()
            hist.append({"step": step, "train_loss": float(loss), "dev_metric": m})
            if m < best:
                best, best_sd = m, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"[N2] {arm} step {step:5d} loss {float(loss):.5f} dev {m:.5f}{'  *' if m == best else ''}", flush=True)
    final_sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return net, best_sd, final_sd, hist, best


@torch.no_grad()
def predict(arm, net, sd, P, dev, bs=4096):
    net.load_state_dict({k: v.to(dev) for k, v in sd.items()}); net.eval()
    out = []
    for i in range(0, len(P), bs):
        p = torch.from_numpy(P[i:i + bs]).unsqueeze(1).to(dev)
        if arm == "REG":
            out.append(net(p).cpu().numpy())
        else:
            e = torch.randn(len(p), BEAT_LEN, generator=torch.Generator().manual_seed(i), device="cpu").to(dev)
            out.append(BS.sample_one_step(net, p, e).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def beat_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    p = pred - pred.mean(1, keepdims=True)
    t = true - true.mean(1, keepdims=True)
    den = np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(den > 0, (p * t).sum(1) / den, np.nan)
    rmse = np.sqrt(((pred - true) ** 2).mean(1))
    # qrs_core_morphology is per-WINDOW (1-D pred/gt + peak list -> scalars); at beat scale each beat
    # is its own window with a single peak at R_IDX.
    d4 = np.empty(len(pred)); d5 = np.empty(len(pred))
    for i in range(len(pred)):
        q = M1.qrs_core_morphology(pred[i], true[i], [R_IDX])
        d4[i], d5[i] = q["qrs_deriv_rmse"], q["qrs_curvature_err"]
    ptp_p, ptp_t = np.ptp(pred, axis=1), np.ptp(true, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ptp_dev = np.abs(np.where(ptp_t > 0, ptp_p / ptp_t, np.nan) - 1.0)
    return {"beat_corr": corr, "beat_rmse": rmse, "ptp_dev": ptp_dev,
            "qrs_deriv_rmse": d4, "qrs_curvature_err": d5}


def macro(v: np.ndarray, subj: np.ndarray) -> float:
    return float(np.mean([np.nanmean(v[subj == s]) for s in np.unique(subj)]))


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    sp = split()
    TRAIN, DEV, EVAL = tuple(sp["probe_train"]), tuple(sp["internal_dev"]), tuple(sp["validation"])
    assert not (set(TRAIN) & set(EVAL)) and not (set(DEV) & set(EVAL)), "train/dev/eval subjects overlap"
    ER.assert_no_test_subjects(TRAIN + DEV + EVAL)

    tpath = ROOT / "artifacts/s1_metric_validity/template_A.npy"
    assert hashlib.sha256(tpath.read_bytes()).hexdigest() == TEMPLATE_A_FILE_SHA, "S1 template hash differs"
    tmpl = np.load(tpath).astype(np.float64)
    assert tmpl.size == BEAT_LEN

    Ptr, Ytr, _ = extract_beats(TRAIN, BEAT_SALT, BEAT_TAKE, "train")
    Pdv, Ydv, _ = extract_beats(DEV, BEAT_SALT, 256, "internal-dev")
    Pev, Yev, Sev = extract_beats(EVAL, EVAL_SALT, EVAL_TAKE, "eval (frozen N1 cohort)")

    # derangement of eval beats for the shuffle controls: a partner beat from a DIFFERENT subject
    rng = np.random.default_rng(20260912)
    order = rng.permutation(len(Pev))
    bad = order == np.arange(len(Pev))
    while bad.any():                                   # force a derangement
        order[bad] = rng.permutation(len(Pev))[bad]
        bad = order == np.arange(len(Pev))
    assert not np.any(order == np.arange(len(Pev))), "shuffle is not a derangement"

    arms, nets, sds, hists = {}, {}, {}, {}
    for arm in ("REG", "IMF"):
        net, best_sd, final_sd, hist, best = train_arm(arm, Ptr, Ytr, Pdv, Ydv, dev)
        nets[arm], sds[arm], hists[arm] = net, {"best": best_sd, "final": final_sd}, hist
        print(f"[N2] {arm} trained: {BS.n_params(net):,} params, best dev {best:.5f}", flush=True)

    preds = {"T-FIXED": np.tile(tmpl, (len(Pev), 1))}
    for arm in ("REG", "IMF"):
        for ck in ("best", "final"):
            preds[f"{arm}[{ck}]"] = predict(arm, nets[arm], sds[arm][ck], Pev, dev)
            preds[f"{arm}-SHUFFLE[{ck}]"] = predict(arm, nets[arm], sds[arm][ck], Pev[order], dev)

    Yd = Yev.astype(np.float64)
    per = {k: beat_metrics(v, Yd) for k, v in preds.items()}
    table = {k: {m: macro(v[m], Sev) for m in v} for k, v in per.items()}
    print()
    for k, v in table.items():
        print(f"[N2] {k:22s} corr {v['beat_corr']:+.4f}  rmse {v['beat_rmse']:.4f}  "
              f"S4 {v['qrs_deriv_rmse']:.4f}  S5 {v['qrs_curvature_err']:.4f}  ptp_dev {v['ptp_dev']:.4f}", flush=True)

    def pair(a, b, m, orient="higher_better"):
        return paired_subject_bootstrap(per[a][m], per[b][m], Sev, orient, BOOT_N, BOOT_SEED)

    pairs = {}
    for ck in ("best", "final"):
        for arm in ("REG", "IMF"):
            pairs[f"{arm}[{ck}]_vs_T-FIXED:beat_corr"] = pair("T-FIXED", f"{arm}[{ck}]", "beat_corr")
            pairs[f"{arm}[{ck}]_vs_SHUFFLE:beat_corr"] = pair(f"{arm}-SHUFFLE[{ck}]", f"{arm}[{ck}]", "beat_corr")
            for m in ("qrs_deriv_rmse", "qrs_curvature_err", "beat_rmse"):
                pairs[f"{arm}[{ck}]_vs_T-FIXED:{m}"] = pair("T-FIXED", f"{arm}[{ck}]", m, "lower_better")
        pairs[f"IMF[{ck}]_vs_REG[{ck}]:beat_corr"] = pair(f"REG[{ck}]", f"IMF[{ck}]", "beat_corr")

    # ---- §6 verdict, on the own-best checkpoint (the preregistered primary reading) ----
    cand = {a: table[f"{a}[best]"]["beat_corr"] for a in ("REG", "IMF")}
    best_arm = max(cand, key=cand.get)
    vT = pairs[f"{best_arm}[best]_vs_T-FIXED:beat_corr"]
    vS = pairs[f"{best_arm}[best]_vs_SHUFFLE:beat_corr"]
    ci_pos = vT["lo"] > 0
    verdict = ("PPG CARRIES BEAT SHAPE" if ci_pos and vT["point"] >= MARGIN_VS_TEMPLATE
               and vS["lo"] > 0 and vS["point"] >= MARGIN_VS_SHUFFLE
               else "MARGINAL" if ci_pos else "PPG DOES NOT CARRY BEAT SHAPE")

    out = {"prereg": PREREG, "oracle_label": ORACLE, "git": git_sha(ROOT),
           "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
           "subjects": {"train": list(TRAIN), "internal_dev": list(DEV), "eval": list(EVAL)},
           "n_beats": {"train": int(len(Ptr)), "dev": int(len(Pdv)), "eval": int(len(Pev))},
           "params": {a: int(BS.n_params(nets[a])) for a in nets},
           "config": {"steps": STEPS, "batch": BATCH, "lr": LR, "wd": WD, "seed": SEED,
                      "beat_len": BEAT_LEN, "r_index": R_IDX, "ctx_len": CTX_LEN},
           "table": table, "paired": pairs, "training_history": hists,
           "verdict": {"best_arm": best_arm, "verdict": verdict,
                       "vs_template": vT, "vs_shuffle": vS,
                       "margins": {"vs_template": MARGIN_VS_TEMPLATE, "vs_shuffle": MARGIN_VS_SHUFFLE}},
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n2_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N2] {ORACLE}")
    print(f"[N2] best arm {best_arm}:  vs T-FIXED {vT['point']:+.4f} [{vT['lo']:+.4f},{vT['hi']:+.4f}] (bar +{MARGIN_VS_TEMPLATE})")
    print(f"[N2]              vs SHUFFLE {vS['point']:+.4f} [{vS['lo']:+.4f},{vS['hi']:+.4f}] (bar +{MARGIN_VS_SHUFFLE})")
    print(f"[N2] VERDICT: {verdict}   ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
