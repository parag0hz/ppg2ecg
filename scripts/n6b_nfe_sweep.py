"""N6-post — multi-step sampling of the SAME frozen FM head.

DECLARED POST-HOC DIAGNOSTIC. The N6 preregistration did not pin the sampler's NFE; the
implementation used NFE 1 (the endpoint form), consistent with this program's one-step focus,
and returned DOES NOT PRESERVE. This sweep was written AFTER seeing that result and therefore
CANNOT change the frozen 5 verdict. It exists to say whether the collapse is a property of
one-step sampling or of the generative form, so a successor preregistration can pin the NFE
in advance.

Run: .venv/bin/python scripts/n6b_nfe_sweep.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ppg2ecg.evaluation import event_reliability as ER          # noqa: E402
from ppg2ecg.flow import rhythm_transfer as RT                  # noqa: E402
from ppg2ecg.utils.seed import seed_everything                  # noqa: E402

import n5_timing_uncertainty as N5                              # noqa: E402
import n6_generative_posterior as N6                            # noqa: E402

ART = ROOT / "artifacts/n6_generative_posterior"
NFES = (1, 2, 4, 8, 16)
K = N6.K_SAMPLES
ALPHAS = N6.ALPHAS


@torch.no_grad()
def sample_nfe(net, c, k, scale, nfe, gen_seed):
    """MeanFlow schedule, uniform steps: z_r = z_t - (t-r) * u(z_t, c, t, t-r)."""
    n = len(c)
    g = torch.Generator(device="cpu").manual_seed(int(gen_seed))
    grid = torch.linspace(1.0, 0.0, nfe + 1)
    out = []
    for _ in range(k):
        z = (torch.randn(n, generator=g) * scale).to(c.device)
        for i in range(nfe):
            t = torch.full((n,), float(grid[i]), device=c.device)
            r = torch.full((n,), float(grid[i + 1]), device=c.device)
            z = z - (t - r) * net.u(z, c, t, r)
        out.append(z.cpu().numpy())
    return np.stack(out, axis=1)


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    sp = json.loads((ROOT / "artifacts/r1_global_rhythm/subject_split.json").read_text())
    TRAIN, DEV, EVAL = tuple(sp["probe_train"]), tuple(sp["internal_dev"]), tuple(sp["validation"])
    ER.assert_no_test_subjects(TRAIN + DEV + EVAL)
    tcn, _ = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)

    Xtr, Ytr, _, _ = N5.build(TRAIN, tcn, dev, "train")
    Xev, Yev, Sev, _ = N5.build(EVAL, tcn, dev, "eval")
    sg_c = float(Ytr.std(ddof=1))

    # retrain FM under the identical recipe (same seed/steps) so the sweep uses the same weights
    seed_everything(N6.SEED)
    fm = N6.FlowHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(fm.parameters(), lr=N6.LR, weight_decay=N6.WD)
    Xt, Yt = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    g2 = torch.Generator().manual_seed(N6.SEED)
    for step in range(1, N6.STEPS + 1):
        i = torch.randint(0, len(Xt), (N6.BATCH,), generator=g2)
        c, y = Xt[i].to(dev), Yt[i].to(dev)
        e = torch.randn(N6.BATCH, device=dev) * sg_c
        t = torch.rand(N6.BATCH, device=dev)
        r = torch.where(torch.rand(N6.BATCH, device=dev) < 0.5, t, t * torch.rand(N6.BATCH, device=dev))
        z = (1 - t) * y + t * e
        loss = torch.mean((fm.u(z, c, t, r) - (e - y)) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    fm.eval()

    Ce = torch.from_numpy(Xev).to(dev)
    ys = Yev.astype(np.float64)
    macro = lambda v: float(np.mean([np.nanmean(np.asarray(v, float)[Sev == s]) for s in np.unique(Sev)]))  # noqa: E731
    rows = []
    print(f"\n[N6-post] DECLARED POST-HOC. CONST bar: CRPS 31.100, sd 54.7 ms; HEAD: CRPS 28.704, sharp +0.455\n")
    for nfe in NFES:
        S = sample_nfe(fm, Ce, K, sg_c, nfe, 7)
        crps = N6.crps_ensemble(S, ys)
        sd = S.std(axis=1, ddof=1); res = np.abs(ys - S.mean(axis=1))
        cov = {a: np.array([np.quantile(S[i], a / 2) <= ys[i] <= np.quantile(S[i], 1 - a / 2)
                            for i in range(len(ys))]) for a in ALPHAS}
        row = {"nfe": nfe, "crps": macro(crps), "median_sd_ms": float(np.median(sd)),
               "median_abs_res_ms": float(np.median(res)),
               "sharpness_spearman": float(sps.spearmanr(sd, res).statistic),
               **{f"coverage_{int((1-a)*100)}": macro(cov[a]) for a in ALPHAS}}
        rows.append(row)
        print(f"[N6-post] NFE {nfe:2d}  CRPS {row['crps']:7.3f}  sd {row['median_sd_ms']:6.1f} ms  "
              f"|res| {row['median_abs_res_ms']:5.1f} ms  sharp {row['sharpness_spearman']:+.3f}  "
              f"cov50/80/90 {row['coverage_50']:.3f}/{row['coverage_80']:.3f}/{row['coverage_90']:.3f}", flush=True)
    out = {"post_hoc": True, "cannot_change_verdict": True, "nfes": list(NFES), "k_samples": K,
           "rows": rows, "const_bar": {"crps": 31.100, "median_sd_ms": 54.7},
           "head_reference": {"crps": 28.704, "sharpness_spearman": 0.455},
           "seconds": round(time.perf_counter() - t0, 1)}
    (ART / "n6b_nfe_sweep.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[N6-post] wrote n6b_nfe_sweep.json ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
