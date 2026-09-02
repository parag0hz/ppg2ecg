"""R2 adapter training driver — docs/R2_RHYTHM_SCAFFOLD_TRANSFER_PREREGISTRATION.md (f954e07) sections 3-10.

One process per arm: --arm preflight | true | shuffle | oracle. The generator and the Global-TCN are frozen;
the only trainable tensor is rhythm_adapter.proj.weight (128 floats). The A4/C1 loader, (t, r) and source
streams are reproduced verbatim; only the scaffold differs between arms. No validation window is read.
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
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import imeanflow_loss
from ppg2ecg.flow.interval_exposure import sample_tr_c1
from ppg2ecg.training.train_a0 import git_sha, load_arrays
from ppg2ecg.training.train_a2 import WSTAT_KEYS
from ppg2ecg.utils.seed import seed_everything
from ppg2ecg.utils.upstream import UPSTREAM_COMMIT, assert_upstream_pinned

ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/r2_rhythm_transfer"
MANIFEST, PROCESSED = "data/manifests/split_a4_wildppg_seed42.json", "data/processed/wildppg_8s"
TR_KW = dict(p_mean=-0.4, p_std=1.0, data_proportion=0.5)
IMF_KW = dict(norm_p=1.0, norm_eps=0.01, jvp_mode="forward")
PREREG = "f954e07"
NOISY_ECG_SUBJECTS = ("fex", "p5d")
LOG_FIELDS = ["step", "loss_weighted", "mse", "loss_before_weighting", *WSTAT_KEYS, "u_abs", "dudt_abs",
              "adapter_l2", "step_s", "elapsed_s", "peak_mem_MiB", "probe_hash"]


def load_train(processed: Path, subjects):
    """x, y exactly as train_a0.load_arrays (asserted by the caller) plus subject / site / npz window_index
    and the per-subject row offsets into the concatenated tensor."""
    xs, ys, sub, site, wi, offsets, off = [], [], [], [], [], {}, 0
    for s in subjects:
        d = np.load(processed / f"{s}.npz")
        X, Y, S, W = d["x"], d["y"], d["site"], d["window_index"]        # bind once (npz re-reads per access)
        offsets[s] = off; off += len(X)
        xs.append(X); ys.append(Y); sub += [s] * len(X); site.append(np.asarray(S).astype(str)); wi.append(W)
    return (np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32), np.array(sub),
            np.concatenate(site), np.concatenate(wi).astype(np.int64), offsets)


def _stats(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(v.size)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["preflight", *RT.TRAINED_ARMS], required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: 100 for preflight; trained arms are fixed at 2200")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--probe-micro-batches", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    trained = a.arm != "preflight"
    steps = a.steps if a.steps is not None else (RT.PREFLIGHT_STEPS if a.arm == "preflight" else RT.STEPS)
    if trained and steps != RT.STEPS:
        raise RuntimeError(f"trained arms run exactly {RT.STEPS} steps (got {steps})")
    out = Path(a.out_dir) if a.out_dir else ROOT / f"outputs/r2_{a.arm}_adapter_seed42"
    ART.mkdir(parents=True, exist_ok=True)
    if trained:
        if out.exists() and any(out.glob("adapter_step*.pt")):
            raise RuntimeError(f"refusing to overwrite adapter checkpoints in {out}")
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

    seed_everything(RT.SEED, deterministic=True)
    dev = torch.device("cuda")
    up = assert_upstream_pinned()
    git = git_sha(ROOT)
    split = read_manifest(ROOT / MANIFEST)[0]
    assert_no_test_subjects(split["train"])
    processed = ROOT / PROCESSED
    x, y, sub, site, wi, offsets = load_train(processed, split["train"])
    x_ref, y_ref, _ = load_arrays(processed, split["train"], None)          # frozen A4 loader, asserted equal
    if not (np.array_equal(x, x_ref) and np.array_equal(y, y_ref)):
        raise RuntimeError("training arrays differ from train_a0.load_arrays")
    del x_ref, y_ref
    N, T = x.shape
    print(f"[D] train {N} windows x {T} | subjects {list(split['train'])} | validation never read", flush=True)
    tp = json.loads((ART / "trainable_parameters.json").read_text()) if (ART / "trainable_parameters.json").exists() else None

    # ---- arm-specific inputs, fixed before any optimizer step ----
    partner_t = oracle_t = None
    cache_sha = None
    checks = {}
    if a.arm == "shuffle":
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
        checks["shuffle_array_pos_rows_checked"] = n_ok
        partner_t = torch.from_numpy(partner).to(dev)
        print(f"[S] SHUFFLE partner tensor: {N} rows, bijective, no fixed point; {n_ok} manifest rows checked against npz", flush=True)
    if a.arm == "oracle":
        z = np.load(ART / "_cache_oracle_train.npz")
        field = z["field"]
        if field.shape != (N, T) or field.dtype != np.float32:
            raise RuntimeError(f"oracle cache shape/dtype {field.shape} {field.dtype}")
        cache_sha = hashlib.sha256(np.ascontiguousarray(field).tobytes()).hexdigest()
        cb = json.loads((ART / "cache_build.json").read_text())
        if cache_sha != cb["oracle_cache_sha256"]:
            raise RuntimeError("oracle cache sha256 != cache_build.json")
        rng_np = np.random.default_rng(0)
        for i in rng_np.choice(N, 32, replace=False):
            if not np.array_equal(field[i], RT._oracle_one(y[i])):
                raise RuntimeError(f"oracle cache row {i} != recomputed GT-R field")
        checks["oracle_cache_rows_spot_checked"] = 32
        oracle_t = torch.from_numpy(field).to(dev)
        print(f"[O] ORACLE field cache loaded: {field.shape} sha256 {cache_sha[:16]} verified (GT-R leakage by design)", flush=True)

    # ---- frozen components ----
    net, _ck, gmeta = RT.load_generator(ROOT / RT.GENERATOR_CKPT, dev)
    tcn, tmeta = RT.load_rhythm_tcn(ROOT / RT.RHYTHM_CKPT, dev)
    RT.assert_only_adapter_trainable(net)
    if RT.n_trainable(net) != gmeta["h_dim"]:
        raise RuntimeError("adapter parameter count != h_dim")
    if tp is not None and (tp["trainable"] != RT.trainable_names(net) or tp["n_trainable"] != RT.n_trainable(net)):
        raise RuntimeError("trainable_parameters.json disagrees with the constructed model")
    if not torch.all(net.rhythm_adapter.proj.weight == 0):
        raise RuntimeError("adapter must start at zero")
    opt = torch.optim.AdamW(net.rhythm_adapter.parameters(), lr=RT.LR, weight_decay=RT.WEIGHT_DECAY)
    x_t, y_t, idx_t = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev), torch.arange(N, device=dev)
    gen = torch.Generator(); gen.manual_seed(RT.SEED)
    tr_gen = torch.Generator(); tr_gen.manual_seed(RT.SEED + 1)
    loader = DataLoader(TensorDataset(x_t, y_t, idx_t), batch_size=RT.BATCH, shuffle=True, generator=gen)
    lop = json.loads((ART / "loader_order_provenance.json").read_text()) if (ART / "loader_order_provenance.json").exists() else None

    s_max, s_mean = [], []                                                  # scaffold statistics on visited windows

    def scaffold(ppg_c: torch.Tensor, idx_c: torch.Tensor) -> torch.Tensor:
        if a.arm in ("preflight", "true"):
            s = RT.scaffold_from_ppg(tcn, ppg_c.unsqueeze(1))
        elif a.arm == "shuffle":
            s = RT.scaffold_from_ppg(tcn, x_t[partner_t[idx_c]].unsqueeze(1))
        else:
            s = oracle_t[idx_c].unsqueeze(1)                                # oracle
        s_max.append(s.amax(dim=(1, 2)).cpu().numpy()); s_mean.append(s.mean(dim=(1, 2)).cpu().numpy())
        return s

    probe = hashlib.sha256(); n_probe = 0; probe_hash = None

    def save_adapter(step: int):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.rhythm_adapter.state_dict().items()},
                    "step": step, "arm": a.arm, "generator_state_sha256": gmeta["state_dict_sha256"],
                    "rhythm_state_sha256": tmeta["state_dict_sha256"], "git": git, "seed": RT.SEED,
                    "prereg": PREREG, "h_dim": gmeta["h_dim"], "probe_hash": probe_hash}, out / f"adapter_step{step}.pt")

    if trained:
        save_adapter(0)
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
                raise RuntimeError("first batch differs from loader_order_provenance.json (STOP)")
        B = len(ppg)
        opt.zero_grad()
        acc = {k: 0.0 for k in ("loss", "mse", "lbw", "u", "d", *WSTAT_KEYS)}
        for i0 in range(0, B, RT.MICRO_BATCH):
            ppg_c, ecg_c, idx_c = ppg[i0:i0 + RT.MICRO_BATCH], ecg[i0:i0 + RT.MICRO_BATCH], idx[i0:i0 + RT.MICRO_BATCH]
            Bc = len(ppg_c)
            t, r, _ = sample_tr_c1(Bc, tr_gen, arm="B", **TR_KW)          # bit-identical historical sampler
            t, r = t.to(dev), r.to(dev)
            e = torch.randn(Bc, 1, T, device=dev)                            # CUDA global stream, as in A4
            if n_probe < a.probe_micro_batches:
                RT.probe_update(probe, idx_c, t, r, e); n_probe += 1
                if n_probe == a.probe_micro_batches:
                    probe_hash = probe.hexdigest()
                    if trained and probe_hash != preflight["probe_hash"]:
                        raise RuntimeError("paired-randomness probe mismatch vs runtime_preflight.json (STOP)")
            s = scaffold(ppg_c, idx_c)
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
            RT.assert_frozen_have_no_grad(net, tcn)
        torch.cuda.synchronize()
        dt = time.perf_counter() - ts; step_times.append(dt)
        row = {"step": step, "loss_weighted": acc["loss"], "mse": acc["mse"], "loss_before_weighting": acc["lbw"],
               **{k: acc[k] for k in WSTAT_KEYS}, "u_abs": acc["u"], "dudt_abs": acc["d"],
               "adapter_l2": float(net.rhythm_adapter.proj.weight.detach().norm()), "step_s": dt,
               "elapsed_s": time.perf_counter() - t_start, "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2 ** 20}
        log_rows.append(row)
        if step % 50 == 0 or step == 1:
            print(f"[T] {a.arm} step {step:4d} mse {row['mse']:.5f} lw {row['loss_weighted']:.4f} |W| {row['adapter_l2']:.4f} "
                  f"{dt*1000:.0f} ms peak {row['peak_mem_MiB']:.0f} MiB", flush=True)
        if trained and step in RT.CKPT_STEPS:
            save_adapter(step)
    wall = time.perf_counter() - t_start
    s_per_step = float(np.mean(step_times[1:])) if len(step_times) > 1 else float(step_times[0])
    peak_alloc, peak_res = torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20

    # realised visits from the loader order alone
    visits = {f"{s_}/{st}": int(np.sum(visited & (sub == s_) & (site == st))) for s_ in split["train"] for st in ("sternum", "head", "wrist", "ankle")}
    n_vis = int(visited.sum())
    noisy_share = float(np.sum(visited & np.isin(sub, NOISY_ECG_SUBJECTS)) / max(n_vis, 1))
    prov = {"arm": a.arm, "opt_steps": opt_steps, "steps_requested": steps, "wall_s": wall, "s_per_step_mean_2_to_end": s_per_step,
            "step1_s": float(step_times[0]), "peak_mem_alloc_MiB": peak_alloc, "peak_mem_reserved_MiB": peak_res,
            "probe_hash": probe_hash, "probe_micro_batches": a.probe_micro_batches,
            "probe_hash_matches_preflight": (probe_hash == preflight["probe_hash"]) if trained else None,
            "final_adapter_l2": float(net.rhythm_adapter.proj.weight.detach().norm()),
            "windows_visited": n_vis, "no_window_twice": bool(n_vis == opt_steps * RT.BATCH),
            "visits_per_subject_site": visits, "noisy_ecg_subject_share": noisy_share,
            "scaffold_stats_train_visited": {"max": _stats(np.concatenate(s_max)), "mean": _stats(np.concatenate(s_mean)),
                                             "field": {"preflight": "s_pred(own)", "true": "s_pred(own)", "shuffle": "s_pred(partner)", "oracle": "s_oracle(GT-R)"}[a.arm]},
            "checks": checks, "generator": gmeta, "rhythm_tcn": tmeta, "oracle_cache_sha256": cache_sha, "git": git, "prereg": PREREG,
            "upstream_commit": UPSTREAM_COMMIT, "upstream_state": up, "seed": RT.SEED, "lr": RT.LR, "weight_decay": RT.WEIGHT_DECAY,
            "batch": RT.BATCH, "micro_batch": RT.MICRO_BATCH, "tr_kw": TR_KW, "imf_kw": IMF_KW, "trainable": RT.trainable_names(net),
            "n_trainable": RT.n_trainable(net), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "finished": datetime.now(timezone.utc).isoformat()}
    if a.arm == "preflight":
        proj_h = 3 * RT.STEPS * s_per_step / 3600.0
        prov |= {"projected_gpu_hours_3_arms": proj_h, "budget_gpu_hours": RT.BUDGET_GPU_HOURS, "stop": bool(proj_h > RT.BUDGET_GPU_HOURS),
                 "adapter_state_discarded": True}
        cb = ART / "cache_build.json"
        if cb.exists():
            prov["oracle_cache_build"] = json.loads(cb.read_text())
        (ART / "runtime_preflight.json").write_text(json.dumps(prov, indent=2, default=float))
        print(f"[P] {s_per_step*1000:.0f} ms/step (step 1 {step_times[0]*1000:.0f} ms) peak {peak_alloc:.0f} MiB -> "
              f"3 x {RT.STEPS} steps = {proj_h:.2f} GPU-h (budget {RT.BUDGET_GPU_HOURS}) | probe {probe_hash[:16]}", flush=True)
        if proj_h > RT.BUDGET_GPU_HOURS:
            print("[P] STOP: projected cost exceeds the budget", flush=True)
            return 2
        return 0
    if opt_steps != RT.STEPS:
        raise RuntimeError(f"realised steps {opt_steps} != {RT.STEPS}")
    for row in log_rows:
        row["probe_hash"] = probe_hash
    with open(ART / f"training_log_{a.arm}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS); w.writeheader(); w.writerows(log_rows)
    (ART / f"train_provenance_{a.arm}.json").write_text(json.dumps(prov, indent=2, default=float))
    (out / "training_summary.json").write_text(json.dumps({k: prov[k] for k in ("arm", "opt_steps", "wall_s", "s_per_step_mean_2_to_end",
                                                                                 "peak_mem_alloc_MiB", "final_adapter_l2", "probe_hash", "git", "prereg")}, indent=2))
    print(f"[done] {a.arm}: {opt_steps} steps, {wall/60:.1f} min, |W| {prov['final_adapter_l2']:.4f}, probe {probe_hash[:16]} (matches preflight)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
