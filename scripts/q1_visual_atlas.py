"""Q1 visual atlas — preregistration section 12 (task section 21).

The frozen V1 visualisation windows belonging to the two development-validation subjects (64 rows).
No new example is selected, no prediction is shifted, every annotation is deterministic.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.evaluation import q1_corruption as Q  # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT  # noqa: E402
from ppg2ecg.flow.imeanflow import MeanFlowS5  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import extract_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/q1_conditional_support"
OUT = ART / "visual_atlas"
if "r2_evaluate" in sys.modules:                                    # never exec twice: pool workers pickle by module name
    R2E = sys.modules["r2_evaluate"]
else:
    _spec = importlib.util.spec_from_file_location("r2_evaluate", ROOT / "scripts/r2_evaluate.py")
    R2E = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = R2E; _spec.loader.exec_module(R2E)
if "q1_evaluate" in sys.modules:
    Q1 = sys.modules["q1_evaluate"]
else:
    _q1 = importlib.util.spec_from_file_location("q1_evaluate", ROOT / "scripts/q1_evaluate.py")
    Q1 = importlib.util.module_from_spec(_q1); sys.modules[_q1.name] = Q1; _q1.loader.exec_module(Q1)
FS, T_LEN, NFE = 128, 1024, Q.NFE_PRIMARY
V1_MANIFEST = ROOT / "artifacts/v1_stepwise_visualization/cohort_manifest.csv"
SEVERE = ("LP_1.25Hz", "SNR_0dB", "DROP_2.0s")
t = np.arange(T_LEN) / FS


def atlas_rows():
    rows = [r for r in csv.DictReader(open(V1_MANIFEST)) if r["cohort"] == "viz" and r["subject"] in Q1.VAL]
    ER.assert_no_test_subjects([r["subject"] for r in rows])
    return sorted(rows, key=lambda r: (r["subject"], r["site"], int(r["array_pos"])))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    rows = atlas_rows()
    if len(rows) != 64:
        raise RuntimeError(f"expected the 64 frozen V1 validation viz windows, got {len(rows)}")
    X, Y, SUB, SITE, POS, WI = [], [], [], [], [], []
    for s in Q1.VAL:
        d = np.load(ROOT / f"data/processed/wildppg_8s/{s}.npz")
        Xs, Ys, Ws = d["x"], d["y"], d["window_index"]
        for r in [r for r in rows if r["subject"] == s]:
            i = int(r["array_pos"])
            X.append(Xs[i].astype(np.float32)); Y.append(Ys[i].astype(np.float32))
            SUB.append(s); SITE.append(r["site"]); POS.append(i); WI.append(int(Ws[i]))
    X, Y = np.stack(X), np.stack(Y)
    SUB, SITE, POS, WI = (np.asarray(v) for v in (SUB, SITE, POS, WI))
    Yd = Y.astype(np.float64)
    gt_pk = R2E.pmap(R2E._peaks, list(Yd))

    _net, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    tcn, _tm = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    partner = RT.shuffle_partner(SUB, SITE, WI, salt=Q.SHUFFLE_SALT)
    RT.assert_derangement(partner)
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(Q.SRC_SEED))

    conds = ("CLEAN",) + SEVERE + (Q.SHUFFLED, Q.NULL)
    XC = {c: (X.copy() if c == "CLEAN" else Q.corrupt_block(X, c, SUB, SITE, WI, partner)) for c in conds}
    P = {c: R2E.gen_plain(base, XC[c], e0, NFE, dev) for c in conds}
    FLD = {c: R2E.scaffolds(tcn, XC[c], dev) for c in conds}
    EV = {c: R2E.pmap(Q1._events_one, list(FLD[c])) for c in conds}
    sup = {c: Q1.support_rows(EV[c], gt_pk) for c in conds}
    fid = {c: R2E.score(P[c], Yd, gt_pk)[0] for c in conds}
    ref = json.loads((ART / "marginal_plausibility_reference.json").read_text())["intervals"]
    plaus = {c: [Q.support_indicators(f, ref) for f in R2E.pmap(Q1._feat_one, list(P[c].astype(np.float64)))] for c in conds}
    banks = {sd: torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(int(sd))) for sd in Q.UNC_SEEDS}
    ENV = {c: np.stack([R2E.gen_plain(base, XC[c], banks[sd], NFE, dev) for sd in Q.UNC_SEEDS]) for c in ("CLEAN", "SNR_0dB")}

    idx_rows = []
    for k in range(len(X)):
        fig, ax = plt.subplots(12, 1, figsize=(11, 20), sharex=True)
        def line(a, y, ttl, color):
            a.plot(t, y, lw=0.7, color=color); a.set_ylabel(ttl, fontsize=7, rotation=0, ha="right", va="center"); a.grid(alpha=0.15)
        line(ax[0], Yd[k], "GT ECG", "k")
        for r in gt_pk[k]:
            ax[0].axvline(r / FS, color="tab:red", lw=0.4, alpha=0.5)
        line(ax[1], XC["CLEAN"][k], "PPG clean", "tab:blue")
        for j, c in enumerate(SEVERE):
            line(ax[2 + j], XC[c][k], f"PPG {c}", "tab:cyan")
        line(ax[5], P["CLEAN"][k], "gen clean", "tab:green")
        for j, c in enumerate(SEVERE):
            line(ax[6 + j], P[c][k], f"gen {c}", "tab:olive")
        line(ax[9], P[Q.SHUFFLED][k], "gen SHUFFLED", "tab:orange")
        line(ax[10], P[Q.NULL][k], "gen NULL (OOD)", "tab:gray")
        m0, s0 = ENV["CLEAN"][:, k].mean(0), ENV["CLEAN"][:, k].std(0)
        m1, s1 = ENV["SNR_0dB"][:, k].mean(0), ENV["SNR_0dB"][:, k].std(0)
        ax[11].plot(t, m0, lw=0.7, color="tab:green"); ax[11].fill_between(t, m0 - s0, m0 + s0, color="tab:green", alpha=0.25)
        ax[11].plot(t, m1, lw=0.7, color="tab:olive"); ax[11].fill_between(t, m1 - s1, m1 + s1, color="tab:olive", alpha=0.25)
        ax[11].set_ylabel("8-source mean ± 1 SD\nclean / SNR_0dB", fontsize=7, rotation=0, ha="right", va="center"); ax[11].grid(alpha=0.15)
        ax[11].set_xlabel("time (s)")
        ann = (f"{SUB[k]}/{SITE[k]} pos {POS[k]} wi {WI[k]} | R1 f1@150 clean {sup['CLEAN'][k]['r1_f1@150']:.2f} "
               f"-> LP {sup['LP_1.25Hz'][k]['r1_f1@150']:.2f} / SNR0 {sup['SNR_0dB'][k]['r1_f1@150']:.2f} / DROP {sup['DROP_2.0s'][k]['r1_f1@150']:.2f} | "
               f"B f1excess {fid['CLEAN'][k]['f1_excess']:+.2f} -> {fid['SNR_0dB'][k]['f1_excess']:+.2f} | "
               f"marg.support {plaus['CLEAN'][k]['marginal_support_fraction']:.2f} -> {plaus['SNR_0dB'][k]['marginal_support_fraction']:.2f} | "
               f"sample SD {float(s0.mean()):.3f} -> {float(s1.mean()):.3f}")
        fig.suptitle("Q1 conditional-support atlas — frozen arm B, NFE 4, source seed 0 (no shifting)\n" + ann, fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
        fn = OUT / f"q1_atlas_{SUB[k]}_{SITE[k]}_{WI[k]}.png"
        fig.savefig(fn, dpi=110); plt.close(fig)
        idx_rows.append({"file": fn.name, "subject": SUB[k], "site": SITE[k], "array_pos": int(POS[k]), "window_index": int(WI[k]),
                         **{f"r1_f1@150_{c}": sup[c][k]["r1_f1@150"] for c in conds},
                         **{f"f1_excess_{c}": fid[c][k]["f1_excess"] for c in conds},
                         **{f"marginal_support_{c}": plaus[c][k]["marginal_support_fraction"] for c in conds},
                         "sample_sd_clean": float(s0.mean()), "sample_sd_snr0": float(s1.mean())})
    R2E.wcsv(OUT / "atlas_index.csv", idx_rows)
    print(f"[atlas] {len(idx_rows)} figures -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
