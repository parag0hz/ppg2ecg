"""O3 runtime preflight (preregistration section 25) — 100 windows through every planned synthetic condition.

NO TRAINING. Nothing produced here enters any O3 result; only wall time and VRAM are measured.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ppg2ecg.evaluation import o3_schedule as O3
from ppg2ecg.training.train_a0 import git_sha

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import o3_common as C  # noqa: E402

ROOT, ART = C.ROOT, C.ART
N = O3.PREFLIGHT_WINDOWS
CONDS = [("JITTER", j) for j in (1, 2, 4, 6, 8)] + [("MISS", n) for n in (1, 2)] + [("EXTRA", n) for n in (1, 2)]
N_REPS, N_MS_SYNTH_ARMS, N_SOURCES = len(O3.REPS), 5, 8      # multi-source: 5 non-B synthetic arms + B


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    coh = C.load_cohort()
    idx = np.arange(N)
    X, Yd, SUB = coh["X"][idx], coh["Yd"][idx], coh["SUB"][idx]
    gt_pk = [coh["gt_pk"][i] for i in idx]
    gt_tg = [coh["gt_tg"][i] for i in idx]
    iqr = coh["iqr"]
    base, o2c, _meta = C.load_models(dev)
    e0 = C.source_bank(len(coh["X"]))[idx]

    t0 = time.perf_counter()
    p_b = C.R2E.gen_plain(base, X, e0, C.NFE, dev)
    rows_b, _a, _b = C.R2E.score(p_b, Yd, gt_pk)
    al_b = C.aligned_rows(p_b, gt_tg, iqr)
    t_b = time.perf_counter() - t0

    t1 = time.perf_counter()
    for fam, lv in CONDS:
        S, RP = [], []
        for j, i in enumerate(idx):
            r = np.asarray(coh["gt_pk"][i], dtype=np.int64)
            s = O3.supplied_schedule(fam, lv, 0, r, coh["SUB"][i], coh["SITE"][i], int(coh["WI"][i]))
            S.append(s); RP.append(O3.retained_pairs(fam, lv, 0, r, s, coh["SUB"][i], coh["SITE"][i], int(coh["WI"][i])))
        [O3.precheck_schedule(s) for s in S]
        [O3.schedule_quality(gt_pk[j], S[j]) for j in range(len(S))]
        warps = C.build_warps(S)
        C.S0.roundtrip_metrics(coh["Y"][idx], warps, gt_pk, gt_tg, iqr, f"{fam}{lv}")
        p_c, _cn = C.o2c_predict(o2c, X, warps, e0, dev)
        rows_c, _c, _d = C.R2E.score(p_c, Yd, gt_pk)
        al_c = C.aligned_rows(p_c, gt_tg, iqr)
        gen_pk = C.R2E.pmap(C.S0._peaks, list(p_c.astype(np.float64)))
        [O3.adherence(S[j], gen_pk[j]) for j in range(len(S))]
        C.shape_only(p_c, Yd, S, gt_pk, RP, iqr)
    t_sweep = time.perf_counter() - t1

    # one real clustered-bootstrap call on the full cohort (zero difference vector: no result is produced)
    z = np.zeros(len(coh["X"]))
    t2 = time.perf_counter()
    C.O1E.cluster_bootstrap(z, coh["SUB"], coh["cluster"], n_boot=O3.BOOT_N, seed=O3.BOOT_SEED)
    t_boot = time.perf_counter() - t2

    scale = len(coh["X"]) / N
    per_cond = t_sweep / len(CONDS) * scale
    n_cond_runs = len(CONDS) * N_REPS + 1                                   # + ORACLE
    sweep_h = (per_cond * n_cond_runs + t_b * scale) / 3600.0
    boot_h = t_boot * len(C.BOOT_M) * (n_cond_runs + 1) / 3600.0
    ms_h = per_cond * (N_MS_SYNTH_ARMS + 1) * N_SOURCES * (512 / len(coh["X"])) / 3600.0
    r1_h = (per_cond * 2 + t_b * scale) / 3600.0 + \
           (per_cond + t_b * scale) * N_SOURCES * (512 / len(coh["X"])) / 3600.0     # BOTH stage-B arms
    total = sweep_h + boot_h + ms_h + r1_h
    out = {"utc": datetime.now(timezone.utc).isoformat(), "git": git_sha(ROOT), "prereg": "60d1810",
           "n_preflight_windows": N, "conditions_measured": len(CONDS), "training_performed": False,
           "t_baseline_s": t_b, "t_sweep_s": t_sweep, "t_one_bootstrap_s": t_boot,
           "projected_sweep_h": sweep_h, "projected_bootstrap_h": boot_h, "projected_multisource_h": ms_h,
           "projected_r1_h": r1_h, "projected_total_gpu_hours": total, "budget_gpu_hours": O3.BUDGET_GPU_HOURS,
           "stop": bool(total > O3.BUDGET_GPU_HOURS),
           "peak_mem_MiB": float(torch.cuda.max_memory_allocated() / 2 ** 20),
           "libs": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
           "gpu": torch.cuda.get_device_name(0)}
    (ART / "runtime_preflight.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("t_sweep_s", "t_one_bootstrap_s", "projected_sweep_h",
                                          "projected_bootstrap_h", "projected_multisource_h", "projected_r1_h",
                                          "projected_total_gpu_hours", "peak_mem_MiB", "stop")}, indent=2))
    if out["stop"]:
        raise RuntimeError(f"projected {total:.2f} GPU-h exceeds the frozen {O3.BUDGET_GPU_HOURS} h budget (STOP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
