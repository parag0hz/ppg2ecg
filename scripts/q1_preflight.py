"""Q1 runtime preflight — preregistration section 13. 100 windows x all conditions x arm B x NFE 4.

Measures wall time and peak VRAM, projects the full Q1 cost and STOPS if the projection exceeds 4 GPU hours.
No result of any kind is produced; nothing is trained.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import q1_corruption as Q
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.training.train_a0 import git_sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/q1_conditional_support"
_spec = importlib.util.spec_from_file_location("r2_evaluate", ROOT / "scripts/r2_evaluate.py")
R2E = importlib.util.module_from_spec(_spec); sys.modules[_spec.name] = R2E; _spec.loader.exec_module(R2E)
_q1 = importlib.util.spec_from_file_location("q1_evaluate", ROOT / "scripts/q1_evaluate.py")
Q1 = importlib.util.module_from_spec(_q1); sys.modules[_q1.name] = Q1; _q1.loader.exec_module(Q1)
T_LEN, NFE = 1024, Q.NFE_PRIMARY


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    ER.assert_no_test_subjects(Q1.VAL)
    dev = torch.device("cuda")
    n = Q.PREFLIGHT_WINDOWS
    X, Y, SUB, SITE, POS, WI = Q1.load_population()
    idx = np.arange(n)
    _net, ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    cfg = ck.get("imf_cfg", {})
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                      h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    e0 = torch.randn(len(X), 1, T_LEN, generator=torch.Generator().manual_seed(Q.SRC_SEED))
    partner = RT.shuffle_partner(SUB, SITE, WI, salt=Q.SHUFFLE_SALT)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    per_cond = {}
    for c in Q.CONDITIONS:
        t1 = time.perf_counter()
        Xc = Q.corrupt_block(X[idx], c, SUB[idx], SITE[idx], WI[idx], partner=np.arange(n)[::-1].copy() if c == Q.SHUFFLED else None)
        _ = R2E.scaffolds(tcn, Xc, dev)
        _ = R2E.gen_plain(base, Xc, e0[idx], NFE, dev)
        per_cond[c] = time.perf_counter() - t1
    gen_s = time.perf_counter() - t0
    peak = float(torch.cuda.max_memory_allocated() / 2 ** 20)

    t0 = time.perf_counter()
    p = R2E.gen_plain(base, X[idx], e0[idx], NFE, dev)
    Yd = Y[idx].astype(np.float64)
    gt_pk = R2E.pmap(R2E._peaks, list(Yd))
    _rows, _pk, _e = R2E.score(p, Yd, gt_pk)
    _ = R2E.pmap(Q1._feat_one, list(p.astype(np.float64)))
    score_s = time.perf_counter() - t0

    per_win_gen = gen_s / (n * len(Q.CONDITIONS))
    per_win_score = score_s / n
    n_cond = len(Q.CONDITIONS)
    proj = {
        "primary_generation_and_field_s": per_win_gen * 2048 * n_cond,
        "primary_scoring_s": per_win_score * 2048 * n_cond,
        "uncertainty_s": per_win_gen * 512 * n_cond * len(Q.UNC_SEEDS) + 0.35 * per_win_score * 512 * n_cond * len(Q.UNC_SEEDS),
        "natural_quality_s": per_win_gen * 8192 + per_win_score * 8192,
        "secondary_gtf_s": (per_win_gen * 2048 * n_cond) + per_win_score * 2048 * n_cond,
        "marginal_reference_s": 0.5 * per_win_score * 12288,
    }
    total_h = sum(proj.values()) / 3600.0
    out = {"utc": datetime.now(timezone.utc).isoformat(), "git": git_sha(ROOT), "n_windows": n, "conditions": list(Q.CONDITIONS),
           "nfe": NFE, "gen_wall_s": gen_s, "score_wall_s": score_s, "per_window_gen_s": per_win_gen,
           "per_window_score_s": per_win_score, "peak_mem_MiB": peak, "per_condition_s": per_cond,
           "projection_s": proj, "projected_total_gpu_hours": total_h, "budget_gpu_hours": Q.BUDGET_GPU_HOURS,
           "stop": bool(total_h > Q.BUDGET_GPU_HOURS), "generator": gmeta["state_dict_sha256"][:16], "tcn": tmeta["state_dict_sha256"][:16]}
    (ART / "runtime_preflight.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("gen_wall_s", "score_wall_s", "peak_mem_MiB", "projected_total_gpu_hours", "stop")}, indent=2))
    if out["stop"]:
        raise RuntimeError(f"projected {total_h:.2f} GPU-h exceeds the frozen {Q.BUDGET_GPU_HOURS} h budget (STOP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
