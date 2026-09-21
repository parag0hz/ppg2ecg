"""DW1 — depth vs width at a fixed inference budget (docs/DW1_DEPTH_VS_WIDTH_PREREGISTRATION.md). No training."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import u2_evaluate as U2  # noqa: E402
import v1_evaluate as V  # noqa: E402
import vm1_evaluate as VM  # noqa: E402
from cd1_train import f_consistency  # noqa: E402
from ppg2ecg.evaluation import event_reliability as ER  # noqa: E402
from ppg2ecg.flow.samplers import euler_sample  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402

STEPS, BUDGETS, BMAX, BS = (1, 2, 4, 8, 16, 32), (1, 2, 4, 8, 16, 32), 32, 512
RAW, OUT = ROOT / "outputs/dw1_raw", ROOT / "artifacts/dw1_depth_width"
CK = {"I": "outputs/v1_vitaldb_armI_seed42", "D": "outputs/cd1_vitaldb_armD_seed42", "C": "outputs/v1_vitaldb_armC_seed42"}
NAME = {"I": "iMF", "D": "consistency distillation", "C": "PENGUIN (Euler)"}
AB1 = {("I", 1): "hr_I1.npy", ("I", 2): "hr_I2.npy", ("D", 1): "hr_D1.npy", ("C", 1): "hr_C1.npy"}


@torch.no_grad()
def make_sampler(arm, dev):
    p = ROOT / CK[arm] / "checkpoint_last.pt"
    if arm == "D":
        ck = torch.load(p, map_location="cpu", weights_only=False)
        net = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval(); net.load_state_dict(ck["state_dict"])
    else:
        net = U2.build(p, dev)[0]

    @torch.no_grad()
    def sample(X, seed, S):
        g = torch.Generator().manual_seed(seed)
        z0 = torch.randn(len(X), 1, X.shape[1], generator=g)
        extra = [torch.randn(len(X), 1, X.shape[1], generator=g) for _ in range(S - 1)] if arm == "D" else []
        out = []
        for i in range(0, len(X), BS):
            ppg = torch.from_numpy(X[i:i + BS]).to(dev).unsqueeze(1)
            z = z0[i:i + BS].to(dev)
            if arm == "I":
                x, k = ER.sample_meanflow_schedule(net, ppg, z, [1.0 / S] * S)
            elif arm == "C":
                x, k = euler_sample(lambda a, t, _p=ppg: net.forward_step(a, _p, t), z, S)
            else:
                B = z.shape[0]
                x, k = f_consistency(net, z, ppg, torch.zeros(B, 1, device=dev)), 1
                for j, e in enumerate(extra, 1):
                    tv = j / S
                    x = f_consistency(net, (1 - tv) * e[i:i + BS].to(dev) + tv * x, ppg, torch.full((B, 1), tv, device=dev)); k += 1
            assert int(k) == S
            out.append(x[:, 0].float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)
    return sample


def main():
    RAW.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    X, Y, pid = VM.load("test")
    ex = ProcessPoolExecutor(8)
    ref_hr = V.hr_batch(ex, Y)
    rows, quality, H = [], [], {}
    for arm in ("I", "D", "C"):
        sampler = make_sampler(arm, dev)
        for S in STEPS:
            kmax = BMAX // S
            f = RAW / f"hr_{arm}{S}.npy"
            if not f.exists() and (arm, S) in AB1 and (ROOT / "outputs/ab1_raw" / AB1[(arm, S)]).exists():
                shutil.copy(ROOT / "outputs/ab1_raw" / AB1[(arm, S)], f)
            first = sampler(X, 0, S)                                  # seed-0 sample: single-sample quality at this depth
            mt = U2.task_metrics("ECG", 8, first, Y, pid); tab = mt.pop("_counts")[0]
            quality.append(dict(model=NAME[arm], S=S, f1=V.cluster_ci(mt["Rpeak_F1"][0], pid)[0],
                                fd=float(U2.pooled_extras("ECG", first, Y, tab)["FD_kanflow"])))
            if f.exists() and np.load(f).shape[0] >= kmax:
                hr = np.load(f)[:kmax]
            else:
                hr = np.full((kmax, len(X)), np.nan, np.float32)
                hr[0] = V.hr_batch(ex, first)
                for s in range(1, kmax):
                    hr[s] = V.hr_batch(ex, sampler(X, s, S))
                np.save(f, hr)
            H[(arm, S)] = hr
            print(f"[dw1] {NAME[arm]} S={S} K<={kmax} F1 {quality[-1]['f1']:.4f} FD {quality[-1]['fd']:.2f}", flush=True)
        del sampler; torch.cuda.empty_cache()

    err = {}
    for arm in ("I", "D", "C"):
        for B in BUDGETS:
            for S in [s for s in STEPS if s <= B]:
                K = B // S
                with np.errstate(all="ignore"):
                    e = np.abs(np.nanmedian(H[(arm, S)][:K], 0) - ref_hr)
                err[(arm, B, S)] = e
                mu, lo, hi = V.cluster_ci(e, pid)
                rows.append(dict(model=NAME[arm], B=B, K=K, S=S, hr_err=mu, ci_lo=lo, ci_hi=hi))
    claims = {}
    for arm in ("I", "D", "C"):
        for B in (16, 32):
            d = V.cluster_ci(err[(arm, B, 1)] - err[(arm, B, B)], pid)          # width extreme minus depth extreme
            cells = {S: V.cluster_ci(err[(arm, B, S)], pid)[0] for S in STEPS if S <= B}
            best = min(cells, key=cells.get)
            claims[f"{NAME[arm]}|B={B}"] = {"width_minus_depth": d, "width_better": bool(d[2] < 0), "depth_better": bool(d[1] > 0),
                                           "best_S": best, "best_K": B // best, "best_err": cells[best],
                                           "interior_beats_both_extremes": bool(best not in (1, B)), "cells_by_S": cells}
    verdict = {"H-W_iMF": all(claims[f"{NAME['I']}|B={B}"]["width_better"] for B in (16, 32)),
               "H-W_CD": all(claims[f"{NAME['D']}|B={B}"]["width_better"] for B in (16, 32)),
               "H-D_PENGUIN": all(claims[f"{NAME['C']}|B={B}"]["depth_better"] for B in (16, 32))}
    with open(OUT / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OUT / "result.json").write_text(json.dumps({"verdict": verdict, "claims": claims, "single_sample_quality": quality}, indent=1))
    print(json.dumps(verdict, indent=1))
    for k, c in claims.items():
        print(k, "width-depth", [round(x, 3) for x in c["width_minus_depth"]], "best (K,S)=", (c["best_K"], c["best_S"]), round(c["best_err"], 3),
              "cells", {s: round(v, 3) for s, v in c["cells_by_S"].items()})


if __name__ == "__main__":
    main()
