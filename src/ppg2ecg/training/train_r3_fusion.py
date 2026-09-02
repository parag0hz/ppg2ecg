"""R3 fusion training driver — docs/R3_DISENTANGLED_RHYTHM_FUSION_PREREGISTRATION.md (3d779fc) sections 4-12.

One process per arm: --arm preflight | tf_true | tf_shuffle | gtf_true | gtf_shuffle | gtf_const | gtf_oracle.
Generator and Global-TCN frozen; only the R3 module trains. The A4/C1/R2 loader, (t, r) and source streams
are reproduced verbatim (probe hash asserted equal to the preflight's and to R2's); only the scaffold and
the gate mode differ between arms. No validation window is read.
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401  (must precede torch)

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.evaluation.event_reliability import assert_no_test_subjects
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import imeanflow_loss
from ppg2ecg.flow.interval_exposure import sample_tr_c1
from ppg2ecg.training.train_a0 import git_sha, load_arrays
from ppg2ecg.training.train_a2 import WSTAT_KEYS
from ppg2ecg.training.train_r2_adapter import load_train, _stats, MANIFEST, PROCESSED, TR_KW, IMF_KW
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/r3_rhythm_fusion"
R2ART = ROOT / "artifacts/r2_rhythm_transfer"
PREREG = "3d779fc"
NOISY_ECG_SUBJECTS = ("fex", "p5d")
LOG_FIELDS = ["step", "loss_weighted", "mse", "loss_before_weighting", *WSTAT_KEYS, "u_abs", "dudt_abs",
              "fusion_l2", "out_proj_l2", "gate_mean", "gate_std", "step_s", "elapsed_s", "peak_mem_MiB", "probe_hash"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["preflight", *RF.TRAINED_ARMS], required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: 100 for preflight; trained arms are fixed at 2200")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--probe-micro-batches", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    trained = a.arm != "preflight"
    arm = "gtf_true" if a.arm == "preflight" else a.arm                 # preflight = the GTF-TRUE model
    family, gate_mode, scaffold_kind = RF.ARM_FAMILY[arm], RF.ARM_GATE_MODE[arm], RF.ARM_SCAFFOLD[arm]
    steps = a.steps if a.steps is not None else (RF.PREFLIGHT_STEPS if not trained else RF.STEPS)
    if trained and steps != RF.STEPS:
        raise RuntimeError(f"trained arms run exactly {RF.STEPS} steps (got {steps})")
    out = Path(a.out_dir) if a.out_dir else ROOT / f"outputs/r3_{a.arm}_seed42"
    ART.mkdir(parents=True, exist_ok=True)
    if trained:
        if out.exists() and any(out.glob("module_step*.pt")):
            raise RuntimeError(f"refusing to overwrite module checkpoints in {out}")
        if (ART / f"training_log_{a.arm}.csv").exists() or (ART / f"train_provenance_{a.arm}.json").exists():
            raise RuntimeError(f"refusing to overwrite training artifacts for arm {a.arm}")
        out.mkdir(parents=True, exist_ok=True)
        pf_path = ART / "runtime_preflight.json"
        if not pf_path.exists():
            raise RuntimeError("runtime_preflight.json missing: run --arm preflight first")
        preflight = json.loads(pf_path.read_text())
        if preflight.get("stop"):
            raise RuntimeError("preflight STOP flag is set: budget exceeded")
    else:
        preflight = None

    seed_everything(RF.SEED, deterministic=True)
    dev = torch.device("cuda")
    up = assert_upstream_pinned()
    git = git_sha(ROOT)
    split = read_manifest(ROOT / MANIFEST)[0]
    assert_no_test_subjects(split["train"])
    processed = ROOT / PROCESSED
    x, y, sub, site, wi, offsets = load_train(processed, split["train"])
    x_ref, y_ref, _ = load_arrays(processed, split["train"], None)
    if not (np.array_equal(x, x_ref) and np.array_equal(y, y_ref)):
        raise RuntimeError("training arrays differ from train_a0.load_arrays")
    del x_ref, y_ref
    N, T = x.shape
    print(f"[D] train {N} windows x {T} | arm {a.arm} family {family} gate {gate_mode} scaffold {scaffold_kind} | validation never read", flush=True)

    # ---- arm inputs fixed before any optimizer step (R2 manifests / caches, re-verified) ----
    partner_t = oracle_t = None
    cache_sha = None
    checks = {}
    if scaffold_kind == "partner":
        rows = [r for r in csv.DictReader(open(ART / "shuffle_manifest.csv")) if r["population"] == "train"]
        partner = np.full(N, -1, dtype=np.int64)
        for r in rows:
            partner[int(r["train_row"])] = int(r["partner_train_row"])
            if int(r["train_row"]) - offsets[r["subject"]] != int(r["array_pos"]):
                raise RuntimeError("shuffle manifest array_pos/train_row inconsistent")
        RT.assert_derangement(partner)
        if not np.array_equal(partner, RT.shuffle_partner(sub, site, wi)):
            raise RuntimeError("shuffle manifest does not match the frozen derangement rule")
        rng_np = np.random.default_rng(0)                                  # numpy RNG only; no torch stream touched
        n_ok = 0
        for s_ in split["train"]:
            Xs = np.load(processed / f"{s_}.npz")["x"]
            for r in rng_np.choice([r for r in rows if r["subject"] == s_], 8, replace=False):
                if not np.array_equal(x[int(r["train_row"])], Xs[int(r["array_pos"])].astype(np.float32)):
                    raise RuntimeError("x_tr[train_row] != npz x[array_pos]")
                n_ok += 1
        partner_t = torch.from_numpy(partner).to(dev)
        checks["shuffle_rows"] = int(len(rows)); checks["shuffle_array_pos_rows_checked"] = n_ok
        print(f"[S] SHUFFLE partner tensor: {N} rows, bijective, no fixed point", flush=True)
    if scaffold_kind == "oracle":
        z = np.load(R2ART / "_cache_oracle_train.npz")
        field = z["field"]
        if field.shape != (N, T) or field.dtype != np.float32:
            raise RuntimeError(f"oracle cache shape/dtype {field.shape} {field.dtype}")
        cache_sha = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
        cb = json.loads((R2ART / "cache_build.json").read_text())
        if cache_sha != cb["oracle_cache_sha256"]:
            raise RuntimeError("oracle cache sha256 != R2 cache_build.json")
        rng_np = np.random.default_rng(0)
        for i in rng_np.choice(N, 32, replace=False):
            if not np.array_equal(field[i], RT._oracle_one(y[i])):
                raise RuntimeError(f"oracle cache row {i} != recomputed GT-R field")
        checks["oracle_cache_rows_spot_checked"] = 32
        oracle_t = torch.from_numpy(field).to(dev)
        print(f"[O] ORACLE field cache (R2) verified: sha256 {cache_sha[:16]} (GT-R leakage by design)", flush=True)

    # ---- frozen components + R3 module (fixed construction order, seed 42; no CUDA draw before step 1) ----
    ck = torch.load(ROOT / RT.GENERATOR_CKPT, map_location="cpu", weights_only=False)
    gsha = RT.state_dict_sha256(ck["state_dict"])
    if gsha != RT.EXPECTED_GENERATOR_STATE_SHA:
        raise RuntimeError("generator state_dict sha mismatch")
    from ppg2ecg.models import build_penguin_backbone
    module = RF.build_r3_module(family, gate_mode, c_hidden=int(ck["model_cfg"]["h_dim"]), seed=RF.SEED)
    init_sha = {"full": RF.params_sha256(module), "fusion_subset": RF.params_sha256(module, only_prefix="fusion.")}
    cfg = ck.get("imf_cfg", {})
    net = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), module, cond_mode=cfg.get("cond_mode", "h_only"),
                              h_scale=cfg.get("h_scale", 1.0))
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or set(missing) != {"r3." + n for n in RF.FAMILY_PARAM_NAMES[family]}:
        raise RuntimeError(f"unexpected checkpoint layout: missing={missing} unexpected={unexpected}")
    net.backbone.requires_grad_(False); net.r3.requires_grad_(True)
    net = net.to(dev).eval()
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    RF.assert_only_r3_trainable(net)
    exp_n = RF.EXPECTED_TF_PARAMS if family == "tf" else RF.EXPECTED_GTF_PARAMS
    if RF.n_params(net.r3) != exp_n:
        raise RuntimeError(f"R3 module has {RF.n_params(net.r3)} params, expected {exp_n}")
    if not (torch.all(net.r3.fusion.out.weight == 0) and torch.all(net.r3.fusion.out.bias == 0)):
        raise RuntimeError("output projection must start at zero")
    if not (ART / "initialization_hashes.json").exists():
        raise RuntimeError("initialization_hashes.json missing: run scripts/r3_prepare.py first")
    ih = json.loads((ART / "initialization_hashes.json").read_text())
    if ih["arms"][arm]["full"] != init_sha["full"] or ih["arms"][arm]["fusion_subset"] != init_sha["fusion_subset"]:
        raise RuntimeError("module initialisation hash differs from initialization_hashes.json")
    if a.probe_micro_batches != 4:
        raise RuntimeError("the R2 probe hash is defined over exactly 4 micro-batches")
    opt = torch.optim.AdamW(net.r3.parameters(), lr=RF.LR, weight_decay=RF.WEIGHT_DECAY)
    x_t, y_t, idx_t = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.arange(N, device=dev)
    gen = torch.Generator(); gen.manual_seed(RF.SEED)
    tr_gen = torch.Generator(); tr_gen.manual_seed(RF.SEED + 1)
    loader = DataLoader(TensorDataset(x_t, y_t, idx_t), batch_size=RF.BATCH, shuffle=True, generator=gen)
    lop = json.loads((R2ART / "loader_order_provenance.json").read_text()) if (R2ART / "loader_order_provenance.json").exists() else None

    s_max, s_mean, g_mean, g_std = [], [], [], []

    def scaffold(ppg_c: torch.Tensor, idx_c: torch.Tensor) -> torch.Tensor:
        if scaffold_kind == "own":
            s = RT.scaffold_from_ppg(tcn, ppg_c.unsqueeze(1))
        elif scaffold_kind == "partner":
            s = RT.scaffold_from_ppg(tcn, x_t[partner_t[idx_c]].unsqueeze(1))
        else:
            s = oracle_t[idx_c].unsqueeze(1)
        s_max.append(s.amax(dim=(1, 2)).cpu().numpy()); s_mean.append(s.mean(dim=(1, 2)).cpu().numpy())
        return s

    probe = hashlib.sha256(); n_probe = 0; probe_hash = None

    def save_module(step: int):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.r3.state_dict().items()},
                    "param_names": list(RF.param_names(net.r3)), "step": step, "arm": a.arm, "family": family, "gate_mode": gate_mode,
                    "scaffold": scaffold_kind, "generator_state_sha256": gsha, "rhythm_state_sha256": tmeta["state_dict_sha256"],
                    "init_sha256": init_sha, "git": git, "seed": RF.SEED, "prereg": PREREG, "probe_hash": probe_hash,
                    "h_dim": int(ck["model_cfg"]["h_dim"])}, out / f"module_step{step}.pt")

    if trained:
        save_module(0)
    visited = np.zeros(N, dtype=bool)
    log_rows, step_times = [], []
    opt_steps = 0
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t_start = time.perf_counter()
    it = iter(loader)
    net.train()
    for step in range(1, steps + 1):
        ts = time.perf_counter()
        ppg, ecg, idx = next(it)
        if step == 1 and lop is not None:
            if hashlib.sha256(idx.cpu().numpy().astype(np.int64).tobytes()).hexdigest() != lop["first_batch_sha256"]:
                raise RuntimeError("first batch differs from the R2 loader_order_provenance.json (STOP)")
        B = len(ppg)
        opt.zero_grad()
        acc = {k: 0.0 for k in ("loss", "mse", "lbw", "u", "d", *WSTAT_KEYS)}
        for i0 in range(0, B, RF.MICRO_BATCH):
            ppg_c, ecg_c, idx_c = ppg[i0:i0 + RF.MICRO_BATCH], ecg[i0:i0 + RF.MICRO_BATCH], idx[i0:i0 + RF.MICRO_BATCH]
            Bc = len(ppg_c)
            t, r, _ = sample_tr_c1(Bc, tr_gen, arm="B", **TR_KW)
            t, r = t.to(dev), r.to(dev)
            e = torch.randn(Bc, 1, T, device=dev)
            if n_probe < a.probe_micro_batches:
                RT.probe_update(probe, idx_c, t, r, e); n_probe += 1
                if n_probe == a.probe_micro_batches:
                    probe_hash = probe.hexdigest()
                    if probe_hash != RF.R2_PROBE_HASH:
                        raise RuntimeError("paired-randomness probe differs from R2's stream hash (STOP)")
                    if trained and probe_hash != preflight["probe_hash"]:
                        raise RuntimeError("paired-randomness probe mismatch vs runtime_preflight.json (STOP)")
            s = scaffold(ppg_c, idx_c)
            if net.r3.gate is not None:
                with torch.no_grad():
                    gv = net.r3.gate_values(s); g_mean.append(float(gv.mean())); g_std.append(float(gv.std()))
            loss, info = imeanflow_loss(net, ecg_c.unsqueeze(1), RT.make_ppg2(ppg_c.unsqueeze(1), s), e, t, r, **IMF_KW)
            if not (torch.isfinite(loss) and torch.isfinite(info["mse"]) and torch.isfinite(info["dudt_abs_mean"])):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            (loss * (Bc / B)).backward()
            acc["loss"] += loss.item() * Bc / B; acc["mse"] += float(info["mse"]) * Bc / B
            acc["lbw"] += float(info["loss_before_weighting"]) * Bc / B
            acc["u"] += float(info["u_abs_mean"]) * Bc / B; acc["d"] += float(info["dudt_abs_mean"]) * Bc / B
            for k in WSTAT_KEYS:
                acc[k] += float(info[k]) * Bc / B
            visited[idx_c.cpu().numpy()] = True
        opt.step()
        opt_steps += 1
        if step == 1:
            RF.assert_frozen_have_no_grad(net, tcn)
        torch.cuda.synchronize()
        dt = time.perf_counter() - ts; step_times.append(dt)
        row = {"step": step, "loss_weighted": acc["loss"], "mse": acc["mse"], "loss_before_weighting": acc["lbw"],
               **{k: acc[k] for k in WSTAT_KEYS}, "u_abs": acc["u"], "dudt_abs": acc["d"],
               "fusion_l2": float(torch.sqrt(sum((p.detach() ** 2).sum() for p in net.r3.fusion.parameters()))),
               "out_proj_l2": float(net.r3.fusion.out.weight.detach().norm()),
               "gate_mean": float(np.mean(g_mean[-2:])) if g_mean else np.nan, "gate_std": float(np.mean(g_std[-2:])) if g_std else np.nan,
               "step_s": dt, "elapsed_s": time.perf_counter() - t_start, "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2 ** 20}
        log_rows.append(row)
        if step % 50 == 0 or step == 1:
            print(f"[T] {a.arm} step {step:4d} mse {row['mse']:.5f} |fusion| {row['fusion_l2']:.3f} |out| {row['out_proj_l2']:.3f} "
                  f"gate {row['gate_mean']:.3f}±{row['gate_std']:.3f} {dt*1000:.0f} ms peak {row['peak_mem_MiB']:.0f} MiB", flush=True)
        if trained and step in RF.CKPT_STEPS:
            save_module(step)
    wall = time.perf_counter() - t_start
    s_per_step = float(np.mean(step_times[1:])) if len(step_times) > 1 else float(step_times[0])
    peak_alloc, peak_res = torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20
    visits = {f"{s_}/{st}": int(np.sum(visited & (sub == s_) & (site == st))) for s_ in split["train"] for st in ("sternum", "head", "wrist", "ankle")}
    n_vis = int(visited.sum())
    prov = {"arm": a.arm, "model_arm": arm, "family": family, "gate_mode": gate_mode, "scaffold": scaffold_kind, "opt_steps": opt_steps,
            "steps_requested": steps, "wall_s": wall, "s_per_step_mean_2_to_end": s_per_step, "step1_s": float(step_times[0]),
            "peak_mem_alloc_MiB": peak_alloc, "peak_mem_reserved_MiB": peak_res, "probe_hash": probe_hash, "probe_micro_batches": a.probe_micro_batches,
            "probe_hash_matches_preflight": (probe_hash == preflight["probe_hash"]) if trained else None, "probe_hash_matches_r2": probe_hash == RF.R2_PROBE_HASH,
            "init_sha256": init_sha, "final_fusion_l2": log_rows[-1]["fusion_l2"], "final_out_proj_l2": log_rows[-1]["out_proj_l2"],
            "final_gate_mean": log_rows[-1]["gate_mean"], "final_gate_std": log_rows[-1]["gate_std"],
            "windows_visited": n_vis, "no_window_twice": bool(n_vis == opt_steps * RF.BATCH), "visits_per_subject_site": visits,
            "noisy_ecg_subject_share": float(np.sum(visited & np.isin(sub, NOISY_ECG_SUBJECTS)) / max(n_vis, 1)),
            "scaffold_stats_train_visited": {"max": _stats(np.concatenate(s_max)), "mean": _stats(np.concatenate(s_mean)), "field": scaffold_kind},
            "checks": checks, "generator_state_sha256": gsha, "rhythm_tcn": tmeta, "oracle_cache_sha256": cache_sha, "git": git, "prereg": PREREG,
            "upstream_commit": UPSTREAM_COMMIT, "upstream_state": up, "seed": RF.SEED, "lr": RF.LR, "weight_decay": RF.WEIGHT_DECAY,
            "batch": RF.BATCH, "micro_batch": RF.MICRO_BATCH, "tr_kw": TR_KW, "imf_kw": IMF_KW, "trainable": list(RF.trainable_names(net)),
            "n_trainable": RF.n_params(net.r3), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "finished": datetime.now(timezone.utc).isoformat()}
    if not trained:
        proj_h = 6 * RF.STEPS * s_per_step / 3600.0
        prov |= {"projected_gpu_hours_6_arms": proj_h, "budget_gpu_hours": RF.BUDGET_GPU_HOURS, "stop": bool(proj_h > RF.BUDGET_GPU_HOURS),
                 "module_state_discarded": True}
        (ART / "runtime_preflight.json").write_text(json.dumps(prov, indent=2, default=float))
        print(f"[P] {s_per_step*1000:.0f} ms/step (step 1 {step_times[0]*1000:.0f} ms) peak {peak_alloc:.0f} MiB -> 6 x {RF.STEPS} = {proj_h:.2f} GPU-h "
              f"(budget {RF.BUDGET_GPU_HOURS}) | probe {probe_hash[:16]} (== R2)", flush=True)
        if proj_h > RF.BUDGET_GPU_HOURS:
            print("[P] STOP: projected cost exceeds the budget", flush=True)
            return 2
        return 0
    if opt_steps != RF.STEPS:
        raise RuntimeError(f"realised steps {opt_steps} != {RF.STEPS}")
    for row in log_rows:
        row["probe_hash"] = probe_hash
    with open(ART / f"training_log_{a.arm}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS); w.writeheader(); w.writerows(log_rows)
    (ART / f"train_provenance_{a.arm}.json").write_text(json.dumps(prov, indent=2, default=float))
    (out / "training_summary.json").write_text(json.dumps({k: prov[k] for k in ("arm", "family", "gate_mode", "opt_steps", "wall_s", "s_per_step_mean_2_to_end",
                                                                                 "peak_mem_alloc_MiB", "final_fusion_l2", "final_gate_mean", "probe_hash", "git", "prereg")}, indent=2))
    print(f"[done] {a.arm}: {opt_steps} steps, {wall/60:.1f} min, |fusion| {prov['final_fusion_l2']:.3f}, probe {probe_hash[:16]} (matches preflight and R2)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
