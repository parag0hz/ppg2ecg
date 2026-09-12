"""N3 — four PPG views of one heartbeat (docs/N3_MULTIVIEW_BEAT_SHAPE_PREREGISTRATION.md).

Every arm receives GROUND-TRUTH R positions at inference: an oracle coordinate probe.
Labelled "(GT-R anchor; oracle coordinate -- diagnostic only)". Nothing here is deployable.

Run: .venv/bin/python scripts/n3_run.py
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
from ppg2ecg.evaluation import rpeaks as RP                     # noqa: E402
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap  # noqa: E402
from ppg2ecg.probes import beat_shape as BS                     # noqa: E402
from ppg2ecg.training.train_a0 import git_sha                   # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

import n2_run as N2                                             # noqa: E402  (beat_metrics, macro, train loop shapes)

ART = ROOT / "artifacts/n3_multiview"
PREREG = "76a406f"
ORACLE = "(GT-R anchor; oracle coordinate -- diagnostic only)"
FS, T_LEN = BS.FS, 1024
BEAT_LEN, R_IDX, CTX_HALF, CTX_LEN = BS.BEAT_LEN, BS.BEAT_R_INDEX, BS.CTX_HALF, BS.CTX_LEN
SITES = ("ankle", "head", "sternum", "wrist")                   # fixed order, prereg §3
BEAT_SALT, BEAT_TAKE = "n2-beat-v1", 1024
EVAL_SALT, EVAL_TAKE = "x4-event-nfe-v2", 1024
STEPS, BATCH, LR, WD, SEED = N2.STEPS, N2.BATCH, N2.LR, N2.WD, N2.SEED
BOOT_N, BOOT_SEED = N2.BOOT_N, N2.BOOT_SEED
MARGIN_T, MARGIN_S = N2.MARGIN_VS_TEMPLATE, N2.MARGIN_VS_SHUFFLE
TEMPLATE_A_FILE_SHA = N2.TEMPLATE_A_FILE_SHA


def extract_multiview(subjects, salt, take, tag):
    """Beats whose window has ALL FOUR sites and a complete context in each.
    Returns P [n, 4, CTX_LEN] in fixed site order, Y [n, BEAT_LEN], S [n], plus the site axis."""
    ER.assert_no_test_subjects(subjects)
    P, Y, S = [], [], []
    n_win_all4 = n_win_seen = 0
    for s in subjects:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        X, Yw = d["x"], d["y"]
        site = np.asarray(d["site"]).astype(str)
        wi = np.asarray(d["window_index"]).astype(np.int64)
        # Select WINDOWS, not rows. select_subset over row space almost never picks all four
        # site-rows of the same window (measured: 1 of 1,922), so the subset is taken in the
        # unique-window space and every site-row of a chosen window is then gathered.
        uw = np.unique(wi)
        sel = ER.select_subset(salt, s, len(uw), take)
        chosen = set(uw[np.asarray(sel, dtype=int)].tolist())
        by_win: dict[int, dict[str, int]] = {}
        for i in range(len(wi)):
            w = int(wi[i])
            if w in chosen:
                by_win.setdefault(w, {})[site[i]] = i
        for w, m in by_win.items():
            n_win_seen += 1
            if set(m) != set(SITES):
                continue
            n_win_all4 += 1
            rows = [m[k] for k in SITES]
            ecg = Yw[rows[0]].astype(np.float64)
            # the four rows must share the ECG target -- asserted, not assumed
            for r_ in rows[1:]:
                if not np.array_equal(Yw[r_], Yw[rows[0]]):
                    raise RuntimeError(f"{s} window {w}: sites disagree on the ECG target")
            ppg4 = np.stack([X[r_].astype(np.float64) for r_ in rows])      # [4, T]
            for r in RP.detect_rpeaks(ecg, FS):
                r = int(r)
                if r - R_IDX < 0 or r - R_IDX + BEAT_LEN > T_LEN:
                    continue
                if r - CTX_HALF < 0 or r + CTX_HALF + 1 > T_LEN:
                    continue
                Y.append(ecg[r - R_IDX: r - R_IDX + BEAT_LEN])
                P.append(ppg4[:, r - CTX_HALF: r + CTX_HALF + 1])
                S.append(s)
    P, Y, S = np.asarray(P, np.float32), np.asarray(Y, np.float32), np.asarray(S)
    print(f"[N3] {tag}: {len(P):,} beats, windows with all 4 sites {n_win_all4:,}/{n_win_seen:,}", flush=True)
    return P, Y, S


def make_net(arm: str, in_ch: int):
    net = BS.BeatRegressor() if arm.startswith("REG") else BS.BeatMeanFlow()
    if in_ch != 1:                                   # only the first conv changes
        old = net.enc.net[0]
        net.enc.net[0] = torch.nn.Conv1d(in_ch, old.out_channels, old.kernel_size[0],
                                         stride=old.stride[0], padding=old.padding[0])
    return net


def train_arm(arm: str, in_ch: int, Ptr, Ytr, Pdv, Ydv, dev):
    seed_everything(SEED)
    net = make_net(arm, in_ch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    Pt, Yt = torch.from_numpy(Ptr), torch.from_numpy(Ytr)
    g = torch.Generator().manual_seed(SEED)
    dev_p, dev_y = torch.from_numpy(Pdv).to(dev), torch.from_numpy(Ydv).to(dev)
    dev_e = torch.randn(len(Pdv), BEAT_LEN, generator=torch.Generator().manual_seed(1000)).to(dev)
    best, best_sd, hist = float("inf"), None, []
    for step in range(1, STEPS + 1):
        i = torch.randint(0, len(Pt), (BATCH,), generator=g)
        p, y = Pt[i].to(dev), Yt[i].to(dev)
        if arm.startswith("REG"):
            loss = torch.mean((net(p) - y) ** 2)
        else:
            e = torch.randn(BATCH, BEAT_LEN, device=dev)
            t = torch.rand(BATCH, device=dev)
            r = torch.where(torch.rand(BATCH, device=dev) < 0.5, t, t * torch.rand(BATCH, device=dev))
            z = (1 - t).unsqueeze(1) * y + t.unsqueeze(1) * e
            loss = torch.mean((net.u(z, p, t, r) - (e - y)) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == STEPS:
            net.eval()
            with torch.no_grad():
                m = float(torch.mean(((net(dev_p) if arm.startswith("REG")
                                       else BS.sample_one_step(net, dev_p, dev_e)) - dev_y) ** 2))
            net.train()
            hist.append({"step": step, "train_loss": float(loss), "dev_metric": m})
            if m < best:
                best, best_sd = m, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            print(f"[N3] {arm:8s} step {step:5d} loss {float(loss):.5f} dev {m:.5f}{'  *' if m == best else ''}", flush=True)
    return net, best_sd, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, hist, best


@torch.no_grad()
def predict(arm, net, sd, P, dev, bs=4096):
    net.load_state_dict({k: v.to(dev) for k, v in sd.items()}); net.eval()
    out = []
    for i in range(0, len(P), bs):
        p = torch.from_numpy(P[i:i + bs]).to(dev)
        if arm.startswith("REG"):
            out.append(net(p).cpu().numpy())
        else:
            e = torch.randn(len(p), BEAT_LEN, generator=torch.Generator().manual_seed(i)).to(dev)
            out.append(BS.sample_one_step(net, p, e).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    sp = N2.split()
    TRAIN, DEV, EVAL = tuple(sp["probe_train"]), tuple(sp["internal_dev"]), tuple(sp["validation"])
    assert not (set(TRAIN) & set(EVAL)) and not (set(DEV) & set(EVAL))
    ER.assert_no_test_subjects(TRAIN + DEV + EVAL)

    tpath = ROOT / "artifacts/s1_metric_validity/template_A.npy"
    assert hashlib.sha256(tpath.read_bytes()).hexdigest() == TEMPLATE_A_FILE_SHA
    tmpl = np.load(tpath).astype(np.float64)

    Ptr, Ytr, _ = extract_multiview(TRAIN, BEAT_SALT, BEAT_TAKE, "train")
    Pdv, Ydv, _ = extract_multiview(DEV, BEAT_SALT, 256, "internal-dev")
    Pev, Yev, Sev = extract_multiview(EVAL, EVAL_SALT, EVAL_TAKE, "eval")

    rng = np.random.default_rng(20260912)
    order = rng.permutation(len(Pev))
    bad = order == np.arange(len(Pev))
    while bad.any():
        order[bad] = rng.permutation(len(Pev))[bad]; bad = order == np.arange(len(Pev))
    assert not np.any(order == np.arange(len(Pev)))

    # REG-1: the matched single-view control -- same beats, one channel, sites pooled
    def pool1(P):
        n = len(P)
        pick = np.arange(n) % len(SITES)                      # deterministic site rotation
        return P[np.arange(n), pick][:, None, :]

    specs = [("REG-1", 1, pool1(Ptr), pool1(Pdv), pool1(Pev)),
             ("REG-4", 4, Ptr, Pdv, Pev),
             ("IMF-4", 4, Ptr, Pdv, Pev)]
    nets, sds, hists, preds = {}, {}, {}, {"T-FIXED": np.tile(tmpl, (len(Pev), 1))}
    for arm, ch, tr, dv, ev in specs:
        net, bsd, fsd, hist, best = train_arm(arm, ch, tr, Ytr, dv, Ydv, dev)
        nets[arm], sds[arm], hists[arm] = net, {"best": bsd, "final": fsd}, hist
        print(f"[N3] {arm} trained: {BS.n_params(net):,} params, best dev {best:.5f}", flush=True)
        for ck in ("best", "final"):
            preds[f"{arm}[{ck}]"] = predict(arm, net, sds[arm][ck], ev, dev)
            if arm.endswith("-4"):
                preds[f"{arm}-SHUFFLE[{ck}]"] = predict(arm, net, sds[arm][ck], ev[order], dev)

    # per-site single-view numbers (prereg §7): the already-trained REG-1 evaluated on each site alone
    for k, s in enumerate(SITES):
        preds[f"REG-1@{s}"] = predict("REG-1", nets["REG-1"], sds["REG-1"]["best"], Pev[:, k][:, None, :], dev)

    Yd = Yev.astype(np.float64)
    per = {k: N2.beat_metrics(v, Yd) for k, v in preds.items()}
    table = {k: {m: N2.macro(v[m], Sev) for m in v} for k, v in per.items()}
    print()
    for k, v in table.items():
        print(f"[N3] {k:22s} corr {v['beat_corr']:+.4f}  rmse {v['beat_rmse']:.4f}  "
              f"S4 {v['qrs_deriv_rmse']:.4f}  S5 {v['qrs_curvature_err']:.4f}", flush=True)

    def pair(a, b, m, orient="higher_better"):
        return paired_subject_bootstrap(per[a][m], per[b][m], Sev, orient, BOOT_N, BOOT_SEED)

    pairs = {}
    for ck in ("best", "final"):
        for arm in ("REG-4", "IMF-4"):
            pairs[f"{arm}[{ck}]_vs_T-FIXED:beat_corr"] = pair("T-FIXED", f"{arm}[{ck}]", "beat_corr")
            pairs[f"{arm}[{ck}]_vs_SHUFFLE:beat_corr"] = pair(f"{arm}-SHUFFLE[{ck}]", f"{arm}[{ck}]", "beat_corr")
        pairs[f"REG-4[{ck}]_vs_REG-1[{ck}]:beat_corr"] = pair(f"REG-1[{ck}]", f"REG-4[{ck}]", "beat_corr")
        pairs[f"REG-1[{ck}]_vs_T-FIXED:beat_corr"] = pair("T-FIXED", f"REG-1[{ck}]", "beat_corr")

    cand = {a: table[f"{a}[best]"]["beat_corr"] for a in ("REG-4", "IMF-4")}
    best_arm = max(cand, key=cand.get)
    vT, vS = pairs[f"{best_arm}[best]_vs_T-FIXED:beat_corr"], pairs[f"{best_arm}[best]_vs_SHUFFLE:beat_corr"]
    verdict = ("MULTI-VIEW CARRIES BEAT SHAPE" if vT["lo"] > 0 and vT["point"] >= MARGIN_T
               and vS["lo"] > 0 and vS["point"] >= MARGIN_S
               else "MARGINAL" if vT["lo"] > 0 else "MULTI-VIEW DOES NOT CARRY BEAT SHAPE")

    out = {"prereg": PREREG, "oracle_label": ORACLE, "git": git_sha(ROOT),
           "utc": datetime.now(timezone.utc).isoformat(), "test_subjects_loaded": [],
           "sites": list(SITES), "subjects": {"train": list(TRAIN), "internal_dev": list(DEV), "eval": list(EVAL)},
           "n_beats": {"train": int(len(Ptr)), "dev": int(len(Pdv)), "eval": int(len(Pev))},
           "params": {a: int(BS.n_params(nets[a])) for a in nets},
           "config": {"steps": STEPS, "batch": BATCH, "lr": LR, "wd": WD, "seed": SEED},
           "table": table, "paired": pairs, "training_history": hists,
           "verdict": {"best_arm": best_arm, "verdict": verdict, "vs_template": vT, "vs_shuffle": vS,
                       "multiview_gain_REG4_minus_REG1": pairs["REG-4[best]_vs_REG-1[best]:beat_corr"],
                       "margins": {"vs_template": MARGIN_T, "vs_shuffle": MARGIN_S}},
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n3_results.json").write_text(json.dumps(out, indent=1, default=float))
    g4 = pairs["REG-4[best]_vs_REG-1[best]:beat_corr"]
    print(f"\n[N3] {ORACLE}")
    print(f"[N3] best arm {best_arm}:  vs T-FIXED {vT['point']:+.4f} [{vT['lo']:+.4f},{vT['hi']:+.4f}] (bar +{MARGIN_T})")
    print(f"[N3]              vs SHUFFLE {vS['point']:+.4f} [{vS['lo']:+.4f},{vS['hi']:+.4f}] (bar +{MARGIN_S})")
    print(f"[N3] multi-view gain REG-4 - REG-1 {g4['point']:+.4f} [{g4['lo']:+.4f},{g4['hi']:+.4f}] {g4['verdict']}")
    print(f"[N3] VERDICT: {verdict}   ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
