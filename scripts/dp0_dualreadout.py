"""DP0 — DualReadout-ECG (docs/DP0_DUALREADOUT_PREREGISTRATION.md).

Split of AF-TRAIN (2,400 patients, seed 20261002): DP-TRAIN 2,100 / DP-DEV 300. AF-DEV is never used; AF-LOCK (337,
AF0's split) opens once, after a committed freeze of one development winner; the old V1 TEST stays closed for all of DP0.
Timing detector, SPECIALIST P (WW-L1), SPECIALIST G (SCALEFLOW-COUPLED) and the three DualReadout sharing patterns
(S0 FULL, S1 MIDDLE, S2 STEM-ONLY) are trained once on DP-TRAIN with seed 42.

Stages: split, audit, manifest, train_detector, train_point, train_gen, train_dual <S0|S1|S2>, eval_dev, grad_conflict,
        compute, select, summarize, freeze <S>, eval_lock, lock_summarize
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/dp0_dualreadout.py <stage> [arg]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bf0_run as B  # noqa: E402
import c0_coherentbeat as C0  # noqa: E402
from ppg2ecg.anchorflow import fastfd as FF  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.coherentbeat import ablation as AB  # noqa: E402
from ppg2ecg.coherentbeat import split as SP  # noqa: E402
from ppg2ecg.dualreadout import model as D  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events  # noqa: E402
from ppg2ecg.rhythmfield import model as RM  # noqa: E402
from ppg2ecg.scaleflow import model as SM  # noqa: E402

PREREG = "docs/DP0_DUALREADOUT_PREREGISTRATION.md"
CODE_FILES = ("scripts/dp0_dualreadout.py", "src/ppg2ecg/dualreadout/__init__.py", "src/ppg2ecg/dualreadout/model.py",
              "tests/test_dp0_dualreadout.py")
ART = ROOT / "artifacts/dp0_dualreadout"
OUT = ROOT / "outputs/dp0_dualreadout"
LOCK_FREEZE = ART / "lock_freeze_manifest.json"
AF0_SPLIT = ROOT / "artifacts/af0_anchorflow/split_manifest.json"
FS, T = 128, 512
SEED, SPLIT_SEED, BOOT_SEED, BOOT_N = 42, 20261002, 20261002, 2000
COUNTS = {"dp_train": 2100, "dp_dev": 300}
N_AF_LOCK = 337
PROTO = {"steps": 20000, "batch": 64, "lr": 1e-3, "wd": 0.01, "clip": 1.0}
DET = dict(C0.DET)
SHARES = ("S0", "S1", "S2")
SHARE_LABEL = {"S0": "FULL", "S1": "MIDDLE", "S2": "STEM-ONLY"}
NFE = 8
N_K16, K = 2000, 16
K16_SALT = "dp0-k16-v1"
N_GRAD, GRAD_SEED = 512, 20261002
GRAD_POINTS = {"init": 0, "25pct": 5000, "final": 20000}


def write_json(name, obj):
    p = ART / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(B.clean(obj), indent=1))


def read_json(name):
    return json.loads((ART / name).read_text())


def ci(v, pid):
    return C0.cluster_ci(v, pid, BOOT_N, BOOT_SEED)


# ----------------------------------------------------------------------------------------------- split / seals / data
def dp_split(af_train_patients) -> dict:
    """default_rng(20261002).permutation of the sorted AF-TRAIN patients: first 300 -> DP-DEV, the other 2,100 -> DP-TRAIN."""
    p = np.array(sorted(int(x) for x in af_train_patients))
    perm = np.random.default_rng(SPLIT_SEED).permutation(p.size)
    return {"dp_train": sorted(int(x) for x in p[perm[300:]]), "dp_dev": sorted(int(x) for x in p[perm[:300]])}


def af0_roles() -> dict:
    return json.loads(AF0_SPLIT.read_text())["roles"]


def check_lock_freeze():
    """AF-LOCK stays sealed until a committed, unchanged DP0 lock-freeze manifest names exactly one development winner."""
    if not LOCK_FREEZE.exists():
        raise PermissionError("DP0: AF-LOCK sealed (no lock-freeze manifest)")
    fm = json.loads(LOCK_FREEZE.read_text())
    if fm.get("winner") not in SHARES:
        raise PermissionError("DP0: AF-LOCK sealed (no development winner)")
    for f, h in fm["sha256"].items():
        if B.sha256_file(ROOT / f) != h:
            raise PermissionError(f"DP0: frozen file changed: {f}")
    rel = str(LOCK_FREEZE.relative_to(ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("DP0: lock-freeze manifest not committed")


def check_prereg_committed():
    """DP-DEV outcomes only after the preregistration manifest is committed and unchanged. Returns the code files changed
    since the preregistration (reported in every outcome file; any change needs a dated amendment)."""
    rel = "artifacts/dp0_dualreadout/prereg_manifest.json"
    if not (ROOT / rel).exists():
        raise PermissionError("DP0: no preregistration manifest")
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode == 0
    if not (tracked and clean):
        raise PermissionError("DP0: preregistration manifest not committed")
    pm = json.loads((ROOT / rel).read_text())
    if B.sha256_file(ROOT / PREREG) != pm["prereg"][PREREG]:
        raise PermissionError("DP0: preregistration document changed after the manifest")
    return sorted(f for f, h in pm["code"].items() if B.sha256_file(ROOT / f) != h)      # post-prereg code changes (reported)


def load_test(*_a, **_k):
    """The old V1 TEST is closed for all of DP0 (a final test needs a separately preregistered DP1)."""
    raise PermissionError("DP0: old V1 TEST is closed")


def load_role(role, ex):
    """(X, Y, Pid, wid, ref) for dp_train / dp_dev / af_lock (af_lock only after the committed freeze). wid = window index
    in the frozen ARCH-TRAIN concatenation (also the noise key)."""
    if role not in ("dp_train", "dp_dev", "af_lock"):
        raise ValueError(role)
    if role == "af_lock":
        check_lock_freeze()
        pats = af0_roles()["af_lock"]["patients"]
    else:
        pats = read_json("split_manifest.json")["roles"][role]["patients"]
    X, Y, Pid = C0.load_arch("train")
    ref = C0.reference_peaks("train", Y, ex)
    wid = np.flatnonzero(np.isin(Pid, pats))
    return X[wid], Y[wid], Pid[wid], wid, [ref[i] for i in wid]


def stage_split(ex, dev):
    af = af0_roles()
    sp = dp_split(af["af_train"]["patients"])
    sf = json.loads((ROOT / "artifacts/sf0_scaleflow/split_manifest.json").read_text())["roles"]
    c0r = C0.split_info()["roles"]
    old = SP.old_heldout_patients(C0.manifest())
    allp = set(sp["dp_train"]) | set(sp["dp_dev"])
    checks = {"counts": {k: len(v) for k, v in sp.items()} == COUNTS, "disjoint": not set(sp["dp_train"]) & set(sp["dp_dev"]),
              "union_is_af_train": allp == set(af["af_train"]["patients"]), "no_af_dev": not allp & set(af["af_dev"]["patients"]),
              "no_af_lock": not allp & set(af["af_lock"]["patients"]), "af_lock_count": len(af["af_lock"]["patients"]) == N_AF_LOCK,
              "no_arch_val_holdout": not allp & (set(c0r["val"]["patients"]) | set(c0r["holdout"]["patients"])),
              "no_sf_val": not allp & set(sf["sf_val"]["patients"]), "no_old_val_test": not allp & (set(old["val"]) | set(old["test"]))}
    if not all(checks.values()):
        raise SystemExit(f"STOP: split checks {checks}")
    _, _, Pid = C0.load_arch("train")
    roles = {r: {"patients": sp[r], "n_patients": len(sp[r]), "n_windows": int(np.isin(Pid, sp[r]).sum())} for r in sp}
    roles["af_lock"] = {"source": "artifacts/af0_anchorflow/split_manifest.json (AF0, unopened)", "n_patients": len(af["af_lock"]["patients"]),
                        "n_windows": af["af_lock"]["n_windows"], "sealed": "until the committed DP0 lock freeze"}
    write_json("split_manifest.json", {"source": "AF-TRAIN (artifacts/af0_anchorflow/split_manifest.json)", "seed": SPLIT_SEED,
                                       "rule": "default_rng(20261002).permutation of sorted AF-TRAIN patients: [0:300] DP-DEV, [300:] DP-TRAIN",
                                       "roles": roles, "checks": checks,
                                       "evidence_status": "DP-DEV is a new development evaluation population for DP0; its patients were training "
                                                          "patients of earlier project models (not project-naive, not external)"})
    write_json("split_hashes.json", {r: {"patients_sha256": hashlib.sha256(json.dumps(sp[r]).encode()).hexdigest(),
                                         "window_index_sha256": hashlib.sha256(np.flatnonzero(np.isin(Pid, sp[r])).astype(np.int64).tobytes()).hexdigest()}
                                     for r in sp})
    print(json.dumps({r: (roles[r]["n_patients"], roles[r]["n_windows"]) for r in roles}), flush=True)


# ----------------------------------------------------------------------------------------------- audit / manifest
def widths() -> dict:
    return D.choose_widths(D.n_params(D.specialist_point()), D.n_params(D.specialist_gen()))


def build(name):
    """detector / point / gen / S0 / S1 / S2 (checkpoint suffixes such as S0_init, S0_c5000 share the architecture)."""
    base = name.split("_")[0]
    if base == "detector":
        return RhythmTCN()
    if base == "point":
        return D.specialist_point()
    if base == "gen":
        return D.specialist_gen()
    if base in SHARES:
        w = widths()
        return D.DualReadout(base, w["w_point"], w["w_flow"])
    raise ValueError(name)


def accounting() -> dict:
    pp, pg = D.n_params(D.specialist_point()), D.n_params(D.specialist_gen())
    sep = pp + pg
    out = {"specialist_point": pp, "specialist_gen": pg, "separate_waveform_params": sep, "e1_threshold": D.E1_RATIO * sep,
           "timing_detector_params_not_counted": D.n_params(RhythmTCN()), "widths": widths()}
    for s in SHARES:
        a = build(s).accounting()
        out[s] = a | {"saving": D.saving(a["total"], sep), "E1": D.e1_pass(a["total"], sep)}
    return out


AUDIT_MD = """# DP0 conditioning-path audit (before any DP0 training or DP-DEV outcome)

## WW-L1 (SPECIALIST P; C0-A WWDet 72 x 5, 593,577 parameters)
- PPG [B, 512] -> C0 encoder: stem 1x1 (1 -> 64) and eight residual blocks (64 channels, kernel 5, dilations 1, 2, 4, 8, 16,
  32, 1, 2; GELU) at the full 512-sample resolution -> h [B, 64, 512] (328,832 parameters).
- The event raster [B, 512] is concatenated AFTER the encoder: 1x1 on [h, raster] (65 -> 72) -> five residual blocks
  (72 channels, dilations 1, 2, 4, 8, 16) -> 1x1 -> ECG (264,745 parameters).
- PPG conditioning is a deep learned single-resolution encoder; event conditioning is a raw raster channel at the decoder input.

## SCALEFLOW-COUPLED (SPECIALIST G; ScaleFM(50, coupled), 598,333 parameters)
- No learned condition encoder. Haar(x_t), Haar(PPG), Haar(raster) are computed by the fixed orthonormal two-level Haar
  transform (coarse 128 / mid 128 / fine 256) and stacked as three raw channels at each branch input.
- Each branch (width 50, six time-conditioned residual blocks, dilations 1..32) processes the noisy ECG and the condition
  jointly from its first 1x1 stem; coarse -> mid and coarse / mid -> fine 1x1 projections (16 channels each).
- PPG / event conditioning is therefore entangled with x_t processing at every scale; there is no separable conditioning
  sub-network to share.

## Consequence for DP0
- The only learned PPG conditioning path in either specialist is the WW-L1 encoder. The DP0 shared encoder is that family,
  with the event raster moved into its input (stem 2 -> 64) so that H = E(PPG, raster) carries all condition information.
- Multiscale features: the closest exact resolutions naturally supported by the repository are those of the fixed Haar
  transform: H_512 (encoder output), H_256 = Haar low-pass of H_512, H_128 = Haar low-pass of H_256 (parameter-free).
  Coarse / mid flow branches read H_128, the fine branch reads H_256, the point decoder reads H_512.
- Natural encoder stages: E0 stem (1x1), E1 blocks 1-6 (first dilation cycle 1..32; receptive field grows from 1 to 505
  samples, i.e. the whole 512-sample window), E2 blocks 7-8 (second cycle 1, 2; refinement at full context). There are no
  further natural boundaries (a half-depth split after block 4 falls inside the first dilation cycle).
- x_t never enters the shared encoder; it is private to the flow decoder.
"""


def stage_audit(ex, dev):
    (ART / "audit.md").write_text(AUDIT_MD)
    acc = accounting()
    write_json("parameter_accounting.json", acc)
    write_json("detector_config.json", {"family": "RD1 RhythmTCN", "protocol": DET, "training": "DP-TRAIN only, seed 42",
                                        "events": "threshold 0.35, refractory 32; one event sequence per window shared by every DP0 model",
                                        "params": D.n_params(RhythmTCN()), "frozen": "after training; never changed during DP0"})
    write_json("specialist_point_config.json", {"name": "SPECIALIST P (WW-L1)", "class": "coherentbeat.ablation.WWDet(72, 5) = scaleflow.ww_l1()",
                                                "params": acc["specialist_point"], "loss": "L1", "protocol": PROTO, "seed": SEED,
                                                "training_raster": "reference-R raster (SF0 / AF0 WW-L1 protocol)",
                                                "inference_raster": "frozen DP detector events", "checkpoint": "last step"})
    write_json("specialist_gen_config.json", {"name": "SPECIALIST G (SCALEFLOW-COUPLED)", "class": "scaleflow.ScaleFM(50, coupled=True)",
                                              "params": acc["specialist_gen"], "objective": "linear-path flow matching, x_t = (1 - t) x0 + t x1, u = x1 - x0, MSE",
                                              "protocol": PROTO, "seed": SEED, "training_raster": "reference-R raster (SF0 protocol)",
                                              "inference": {"solver": "Euler", "nfe": NFE, "noise": "scaleflow.window_noise (sha256 'patient:window:20261002')"}})
    write_json("sharing_graph.json", {
        "encoder": {"input": "[PPG, event raster] (2 channels); never x_t", "stem": "Conv1d(2, 64, 1)", "blocks": list(D.ENC_DIL),
                    "stages": D.STAGES, "multiscale": "H_256 = Haar low1(H_512), H_128 = Haar low1(H_256) (fixed, parameter-free)"},
        "patterns": {s: {"label": SHARE_LABEL[s], "shared_stages": list(D.SHARED_STAGES[s]), "shared_blocks": D.N_SHARED_BLOCKS[s],
                         "private_tower_blocks_per_task": 8 - D.N_SHARED_BLOCKS[s],
                         "ownership_prefixes": {"shared": ["stem", "shared_blocks"], "point": ["point_adapter", "point_tower", "point_dec"],
                                                "flow": ["flow_adapter", "flow_tower", "flow_dec"]}} for s in SHARES},
        "readouts": {"point": "point adapter -> point tower -> PointDecoder(w_p) on H_512 -> mu",
                     "flow": "flow adapter -> flow tower -> (H_256, H_128) -> FlowDecoder(w_f)(x_t, t, H) -> velocity; Euler NFE 8 from noise",
                     "never_added": True, "no_anchor_residual_path": True},
        "path_rule": "every readout path has the same architecture in S0 / S1 / S2; private towers are fresh copies of the blocks they replace"})
    write_json("adapter_config.json", {"design": "Conv1d(64, 64, 1) + GELU (same input / output channels), no attention",
                                       "placement": "one point adapter and one flow adapter at the output of the last shared stage",
                                       "params_each": D.n_params(D.Adapter()),
                                       **{s: {"adapters": acc[s]["adapters"], "fraction_of_total": acc[s]["adapter_fraction"]} for s in SHARES},
                                       "limit": "< 5 % of total waveform-model parameters"})
    write_json("training_manifest.json", {
        "specialists": {"protocol": PROTO, "seed": SEED, "batches": "C0 _epoch_batches(n, 64, Generator(42))", "fm_noise": "Generator(cuda).manual_seed(42)"},
        "dual": {"cycles": PROTO["steps"], "point_updates": PROTO["steps"], "flow_updates": PROTO["steps"],
                 "order": "cycle c = 1..20000: odd POINT -> FLOW, even FLOW -> POINT",
                 "optimizer": "ONE AdamW(all parameters, lr 1e-3, weight decay 0.01), constant LR; per substep zero_grad(set_to_none) -> "
                              "task loss backward -> the other task's private parameters have grad None (checked) and are skipped by AdamW -> "
                              "clip_grad_norm 1.0 over shared + this task's private parameters -> step",
                 "point_stream": "batches identical to SPECIALIST P's (Generator(42))", "flow_stream": "batches and FM noise / t identical to SPECIALIST G's",
                 "point_loss": "L1(mu, ECG)", "flow_loss": "MSE(v, x1 - x0) (scaleflow draw order: x0 then t)", "seed": SEED,
                 "training_raster": "reference-R raster", "checkpoints": GRAD_POINTS, "early_stopping": None,
                 "excluded": ["loss weighting", "GradNorm", "PCGrad", "gradient surgery", "adaptive task weighting"]}})


def stage_manifest(ex, dev):
    design = ("split_manifest.json", "split_hashes.json", "detector_config.json", "specialist_point_config.json", "specialist_gen_config.json",
              "sharing_graph.json", "adapter_config.json", "training_manifest.json", "parameter_accounting.json", "audit.md")
    write_json("prereg_manifest.json", {"prereg": {PREREG: B.sha256_file(ROOT / PREREG)}, "code": {f: B.sha256_file(ROOT / f) for f in CODE_FILES},
                                        "design": {f: B.sha256_file(ART / f) for f in design},
                                        "written_before_any_dp_dev_outcome": True, "software": B.software()})


# ----------------------------------------------------------------------------------------------- training
def _save(name, net, secs, nan_steps, extra=None):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.pt"
    meta = {"seed": SEED, "train_seconds": secs, "n_params": D.n_params(net), "nan_steps": nan_steps,
            "peak_gpu_mem_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "optimizer": "AdamW"} | (extra or {})
    torch.save({"state_dict": net.state_dict()} | meta, f)
    meta["sha256"] = B.sha256_file(f)
    write_json(f"checkpoints/{name}.json", meta)


def _train_data(ex, dev):
    X, Y, _, _, ref = load_role("dp_train", ex)
    return (torch.from_numpy(X).to(dev), torch.from_numpy(Y.astype(np.float32)).to(dev),
            torch.from_numpy(AB.event_raster(ref)).to(dev), ref)


def stage_train_detector(ex, dev):
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, _, _, ref = load_role("dp_train", ex)
    Fd = torch.from_numpy(np.stack(list(ex.map(C0._field, ref, chunksize=512))).astype(np.float16)).to(dev)
    Xt = torch.from_numpy(X).to(dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = RhythmTCN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=DET["lr"], weight_decay=DET["wd"])
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    batches, t0, nan_steps = C0._epoch_batches(len(Xt), DET["batch"], g), time.time(), 0
    net.train()
    for _ in range(DET["steps"]):
        b = next(batches).to(dev)
        loss = lossf(net(Xt[b][:, None])[:, 0], Fd[b].float())
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step()
    _save("detector", net, time.time() - t0, nan_steps, {"protocol": DET, "training_role": "dp_train"})


def stage_train_point(ex, dev):
    """SPECIALIST P: WW-L1, point-only L1."""
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    Xt, Yt, Rt, _ = _train_data(ex, dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build("point").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    batches, t0, nan_steps, acc = C0._epoch_batches(len(Xt), PROTO["batch"], torch.Generator().manual_seed(SEED)), time.time(), 0, []
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        b = next(batches).to(dev)
        loss = (net(Xt[b], Rt[b]) - Yt[b]).abs().mean()
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step(); acc.append(loss.item())
        if step % 2000 == 0:
            print(f"[dp0] point step {step} L1 {np.mean(acc):.5f}", flush=True); acc = []
    _save("point", net, time.time() - t0, nan_steps, PROTO | {"loss": "L1", "training_role": "dp_train"})


def stage_train_gen(ex, dev):
    """SPECIALIST G: SCALEFLOW-COUPLED, SF0 flow matching."""
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    Xt, Yt, Rt, _ = _train_data(ex, dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build("gen").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    gd = torch.Generator(device=dev).manual_seed(SEED)
    batches, t0, nan_steps, acc = C0._epoch_batches(len(Xt), PROTO["batch"], torch.Generator().manual_seed(SEED)), time.time(), 0, []
    net.train()
    for step in range(1, PROTO["steps"] + 1):
        b = next(batches).to(dev)
        loss = SM.fm_loss(net, Yt[b], Xt[b], Rt[b], gd)
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), PROTO["clip"])
        opt.step(); acc.append(loss.item())
        if step % 2000 == 0:
            print(f"[dp0] gen step {step} FM {np.mean(acc):.5f}", flush=True); acc = []
    _save("gen", net, time.time() - t0, nan_steps, PROTO | {"loss": "flow-matching MSE", "training_role": "dp_train"})


def stage_train_dual(ex, dev, share):
    """Round-robin multitask training: 20,000 cycles = 20,000 POINT + 20,000 FLOW masked updates of ONE AdamW."""
    if share not in SHARES:
        raise SystemExit(f"STOP: unknown sharing pattern {share}")
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    Xt, Yt, Rt, _ = _train_data(ex, dev)
    torch.manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    net = build(share).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=PROTO["lr"], weight_decay=PROTO["wd"])
    pb = C0._epoch_batches(len(Xt), PROTO["batch"], torch.Generator().manual_seed(SEED))       # = SPECIALIST P's batch stream
    fb = C0._epoch_batches(len(Xt), PROTO["batch"], torch.Generator().manual_seed(SEED))       # = SPECIALIST G's batch stream
    gd = torch.Generator(device=dev).manual_seed(SEED)                                          # = SPECIALIST G's FM noise / t stream

    def point_loss():
        b = next(pb).to(dev)
        return (net.point(Xt[b], Rt[b]) - Yt[b]).abs().mean()

    def flow_loss():
        b = next(fb).to(dev)
        return D.fm_loss(net, Yt[b], Xt[b], Rt[b], gd)

    log = {"point": [], "flow": [], "nan": 0}
    _save(f"{share}_init", net, 0.0, 0, {"cycle": 0, "share": share})
    t0 = time.time()

    def on_cycle(c, task, loss):
        log[task].append(loss.item())
        log["nan"] += int(not torch.isfinite(loss))
        if task == D.cycle_order(c)[1]:
            if c == GRAD_POINTS["25pct"]:
                _save(f"{share}_c{c}", net, time.time() - t0, log["nan"], {"cycle": c, "share": share})
            if c % 2000 == 0:
                print(f"[dp0] {share} cycle {c} L1 {np.mean(log['point']):.5f} FM {np.mean(log['flow']):.5f}", flush=True)
                log["point"], log["flow"] = [], []

    net.train()
    res = D.train_round_robin(net, opt, PROTO["steps"], point_loss, flow_loss, PROTO["clip"], on_cycle)
    secs = time.time() - t0
    if res["counts"] != {"point": PROTO["steps"], "flow": PROTO["steps"]}:
        raise SystemExit(f"STOP: update counts {res['counts']}")
    _save(share, net, secs, log["nan"], PROTO | {"cycle": PROTO["steps"], "share": share, "updates": res["counts"],
                                                  "first_task_odd_even": [res["first_task_by_cycle"][0], res["first_task_by_cycle"][1]],
                                                  "loss": "round-robin L1 / flow-matching MSE", "training_role": "dp_train"})


# ----------------------------------------------------------------------------------------------- inference
def load_net(name, dev):
    ck = torch.load(OUT / f"{name}.pt", map_location="cpu")
    net = build(name)
    net.load_state_dict(ck["state_dict"])
    return net.to(dev).eval()


def _extract(p):
    return extract_events(p, DET["threshold"], DET["refractory"])


@torch.no_grad()
def detector_events(X, dev, ex):
    det = load_net("detector", dev)
    out = [torch.sigmoid(det(torch.from_numpy(X[i:i + 2048]).to(dev)[:, None])[:, 0]).float().cpu().numpy() for i in range(0, len(X), 2048)]
    return [np.asarray(e, int) for e in ex.map(_extract, list(np.concatenate(out)), chunksize=256)]


@torch.no_grad()
def point_out(name, X, R, dev, bs=1024):
    net = load_net(name, dev)
    f = net if name == "point" else net.point
    return np.concatenate([f(torch.from_numpy(X[i:i + bs]).to(dev), torch.from_numpy(R[i:i + bs]).to(dev)).cpu().numpy()
                           for i in range(0, len(X), bs)]).astype(np.float64)


@torch.no_grad()
def gen_out(name, X, R, noise, dev, nfe=NFE, bs=512):
    net = load_net(name, dev)
    out = []
    for i in range(0, len(X), bs):
        x0, x, r = (torch.from_numpy(a[i:i + bs]).to(dev) for a in (noise, X, R))
        out.append((SM.euler(net, x0, x, r, nfe) if name == "gen" else D.euler(net, x0, x, r, nfe)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def k16_subset(n):
    return np.sort(B.salted_rank(K16_SALT, range(n))[:min(N_K16, n)])


def shuffle_perm(n):
    return np.random.default_rng(SPLIT_SEED).permutation(n)


def context(role, ex, dev):
    X, Y, Pid, wid, ref = load_role(role, ex)
    ev = detector_events(X, dev, ex)
    return {"role": role, "X": X, "Y": Y, "Pid": Pid, "wid": wid, "ref": ref, "ev": ev, "R": AB.event_raster(ev),
            "pairs": [BR.matched_pairs(ref[i], ev[i], T) for i in range(len(Y))], "noise": SM.window_noise(Pid, wid),
            "S": k16_subset(len(Y)), "perm": shuffle_perm(len(Y))}


# ----------------------------------------------------------------------------------------------- metrics
def point_metrics(w, Y, ref, Pid, pairs, ex):
    """Per-window beat-aligned corr on the (reference R, detector event) pairs, MAE, patient-macro FP / recall / precision /
    F1 (neurokit on the waveform, +-50 ms), RR / HR errors."""
    evaluable = np.array([len(r) > 0 for r in ref])
    det = list(ex.map(B._peaks, list(w), chunksize=256))
    prf = PMX.rpeak_prf_at(w, Y, FS, 50.0, peaks=(ref, det))
    bl = PMX.beat_level_metrics(w, Y, FS, 50.0, peaks=(ref, det))
    corr = np.array([np.nanmean(BR.pair_correlations(Y[i], w[i], pairs[i])) if len(pairs[i]) else np.nan for i in range(len(Y))])
    pm = RM.patient_macro_rows(prf["n_tp"], prf["n_fp"], prf["n_fn"], Pid)
    return {"corr": corr, "mae": np.abs(w - Y).mean(axis=1), "pm": pm, "rr": bl["rr_mae_ms"], "hr": bl["hr_abs_err"], "det": det,
            "f1_window": np.where(evaluable, prf["rpeak_f1"], np.nan)}


def summarize_point(m, Pid):
    s = m["pm"]["patients"]
    return {"corr": ci(m["corr"], Pid), "mae": ci(m["mae"], Pid), "rr_mae_ms": ci(m["rr"], Pid), "hr_mae_bpm": ci(m["hr"], Pid),
            **{f"pm_{k}": C0.cluster_ci(m["pm"][k], s, BOOT_N, BOOT_SEED) for k in ("fp_rate", "recall", "precision", "f1")},
            "pooled": RM.pooled_prf(**m["pm"]["pooled"])}


def spectral_discrepancy(a, b):
    """AF0 descriptive spectral metric: mean |log PSD ratio| between two waveform sets (Hann, rFFT, bins 1..256)."""
    pa = (np.abs(np.fft.rfft(a * np.hanning(T), axis=1)) ** 2).mean(axis=0)[1:]
    pb = (np.abs(np.fft.rfft(b * np.hanning(T), axis=1)) ** 2).mean(axis=0)[1:]
    return float(np.mean(np.abs(np.log((pa + 1e-12) / (pb + 1e-12)))))


def k16_block(ctx, Ks, mu_ref, ex):
    """K16 characterization on the fixed subset S (descriptive). Ks: [K, |S|, T]; mu_ref: SPECIALIST P on S."""
    S = ctx["S"]
    Y, Pid = ctx["Y"][S], ctx["Pid"][S]
    ev, ref = [ctx["ev"][i] for i in S], [ctx["ref"][i] for i in S]
    ww = np.array([BR.mean_pairwise_rms(Ks[:, i, :]) for i in range(len(S))])
    real_dev = float(np.sqrt(((Y - mu_ref) ** 2).mean()))
    beat_div, n_anchor = B.seed_diversity(Ks, ev)
    real_bd = np.array([BR.within_window_diversity(Y[i], ev[i]) for i in range(len(S))])
    gen_bd = np.array([np.nanmean([BR.within_window_diversity(Ks[k, i], ev[i]) for k in range(Ks.shape[0])]) for i in range(len(S))])
    peaks = [list(ex.map(B._peaks, list(Ks[k]), chunksize=128)) for k in range(Ks.shape[0])]
    rsd, n_sd = B.timing_sd(ref, [[peaks[k][i] for k in range(Ks.shape[0])] for i in range(len(S))])
    hr_ref = np.array([RP.hr_bpm(np.asarray(r, int), FS) for r in ref])
    hr_k = np.array([[RP.hr_bpm(peaks[k][i], FS) for k in range(Ks.shape[0])] for i in range(len(S))])
    with np.errstate(all="ignore"):
        hr_med = np.nanmedian(hr_k, axis=1)
    xbar = Ks.mean(axis=0)
    mb = point_metrics(xbar, Y, ref, Pid, [ctx["pairs"][i] for i in S], ex)
    return {"within_condition_waveform_diversity": ci(ww, Pid), "within_over_sqrt2_real_deviation_from_P": float(np.nanmean(ww) / (np.sqrt(2) * real_dev)),
            "real_rms_deviation_from_P": real_dev, "beat_aligned_diversity": beat_div, "beat_anchors": n_anchor,
            "generated_over_real_beat_diversity_ratio": float(np.nanmean(gen_bd) / np.nanmean(real_bd)),
            "r_time_seed_sd_ms_median": rsd, "r_time_sd_beats": n_sd,
            "k16_consensus_hr_mae": ci(np.abs(hr_med - hr_ref), Pid), "single_sample_hr_mae": ci(np.abs(hr_k[:, 0] - hr_ref), Pid),
            "mean16_waveform": summarize_point(mb, Pid) | {"mean16_minus_P_mae": float(np.abs(xbar - mu_ref).mean())},
            "feature_spread": {"per_window_sd_mean": float(Ks.std(axis=0).mean()), "per_window_peak_to_peak_sd": float(np.ptp(Ks, axis=2).std(axis=0).mean())}}


@torch.no_grad()
def feature_diagnostics(ctx, dev):
    """Descriptive feature statistics on the K16 subset: shared output RMS / channel variance, adapter output norms."""
    S = ctx["S"]
    x, r = torch.from_numpy(ctx["X"][S]).to(dev), torch.from_numpy(ctx["R"][S]).to(dev)
    out = {}
    for s in SHARES:
        net = load_net(s, dev)
        h = torch.cat([net.trunk(x[i:i + 500], r[i:i + 500]) for i in range(0, len(S), 500)])
        pa = torch.cat([net.point_adapter(h[i:i + 500]) for i in range(0, len(S), 500)])
        fa = torch.cat([net.flow_adapter(h[i:i + 500]) for i in range(0, len(S), 500)])
        pf = torch.cat([net.point_features(h[i:i + 500]) for i in range(0, len(S), 500)])
        ff = torch.cat([net.flow_features(h[i:i + 500])[0] for i in range(0, len(S), 500)])
        st = lambda z: {"rms": float(z.pow(2).mean().sqrt()), "channel_variance_mean": float(z.var(dim=(0, 2)).mean()),  # noqa: E731
                        "channel_variance_min": float(z.var(dim=(0, 2)).min()), "dead_channels": int((z.var(dim=(0, 2)) < 1e-6).sum())}
        out[s] = {"shared_output": st(h), "point_adapter_output": st(pa), "flow_adapter_output": st(fa),
                  "point_tower_output": st(pf), "flow_tower_output_H256": st(ff),
                  "point_adapter_norm": float(pa.pow(2).mean().sqrt()), "flow_adapter_norm": float(fa.pow(2).mean().sqrt())}
    return out


def evaluate(ctx, arms_point, arms_gen, ex, dev):
    """Point heads, one-sample generation, PPG shuffle and every bootstrap comparison on one role."""
    X, Y, Pid, R, pairs, ref, perm = ctx["X"], ctx["Y"], ctx["Pid"], ctx["R"], ctx["pairs"], ctx["ref"], ctx["perm"]
    duals = [a for a in arms_gen if a != "gen"]
    pts = {a: point_out(a, X, R, dev) for a in arms_point}
    gens = {a: gen_out(a, X, R, ctx["noise"], dev) for a in arms_gen}
    shuf = {a: gen_out(a, X[perm], R, ctx["noise"], dev) for a in arms_gen}                  # PPG shuffled; raster / noise / model kept
    pm_p = {a: point_metrics(w, Y, ref, Pid, pairs, ex) for a, w in pts.items()}
    pm_g = {a: point_metrics(w, Y, ref, Pid, pairs, ex) for a, w in gens.items()}
    pm_s = {a: point_metrics(w, Y, ref, Pid, pairs, ex) for a, w in shuf.items()}
    res = B.patient_resamples(np.unique(Pid).size, BOOT_N, BOOT_SEED)
    fdb = FF.fd_bootstrap({**{a: gens[a] for a in arms_gen}, **{f"{a}_shuf": shuf[a] for a in arms_gen}}, Y, Pid,
                          [(a, "gen") for a in duals] + [(f"{a}_shuf", a) for a in arms_gen], res)
    mu_P = pts["point"]
    real_dev = float(np.sqrt(((Y - mu_P) ** 2).mean()))
    point = {a: summarize_point(pm_p[a], Pid) | {"fd_descriptive": float(PMX.kanflow_fd(pts[a], Y))} for a in arms_point}
    gen = {a: {"fd": fdb["fd"][a], "nfe": NFE, "single_sample": summarize_point(pm_g[a], Pid),
               "diversity_ratio_rms_dev_from_P": float(np.sqrt(((gens[a] - mu_P) ** 2).mean()) / real_dev),
               "population_sd_ratio": float(gens[a].std(axis=0).mean() / Y.std(axis=0).mean()),
               "spectral_discrepancy": spectral_discrepancy(gens[a], Y)} for a in arms_gen}
    boot, gts = {}, {}
    pats = pm_p["point"]["pm"]["patients"]
    for a in duals:
        assert np.array_equal(pm_p[a]["pm"]["patients"], pats) and np.array_equal(pm_g[a]["pm"]["patients"], pats)
        d = {"corr": ci(pm_p[a]["corr"] - pm_p["point"]["corr"], Pid),
             "fp": C0.cluster_ci(pm_p[a]["pm"]["fp_rate"] - pm_p["point"]["pm"]["fp_rate"], pats, BOOT_N, BOOT_SEED),
             "recall": C0.cluster_ci(pm_p[a]["pm"]["recall"] - pm_p["point"]["pm"]["recall"], pats, BOOT_N, BOOT_SEED),
             "fd": fdb[f"{a}-gen"], "fd_shuf": fdb[f"{a}_shuf-{a}"], "corr_shuf": ci(pm_s[a]["corr"] - pm_g[a]["corr"], Pid),
             "f1_desc": C0.cluster_ci(pm_p[a]["pm"]["f1"] - pm_p["point"]["pm"]["f1"], pats, BOOT_N, BOOT_SEED),
             "mae_desc": ci(pm_p[a]["mae"] - pm_p["point"]["mae"], Pid)}
        boot[a] = d
        acc = read_json("parameter_accounting.json")
        gts[a] = D.gates(d, acc[a]["total"], acc["separate_waveform_params"])
    cond = {a: {"fd_conditioned": fdb["fd"][a], "fd_shuffled": fdb["fd"][f"{a}_shuf"], "fd_shuffled_minus_conditioned": fdb[f"{a}_shuf-{a}"],
                "corr_conditioned": ci(pm_g[a]["corr"], Pid), "corr_shuffled": ci(pm_s[a]["corr"], Pid),
                "corr_shuffled_minus_conditioned": ci(pm_s[a]["corr"] - pm_g[a]["corr"], Pid),
                "pass": bool(fdb[f"{a}_shuf-{a}"][1] > 0 and ci(pm_s[a]["corr"] - pm_g[a]["corr"], Pid)[2] < 0)} for a in arms_gen}
    sanity = {"P_corr_minus_G_single_corr": ci(pm_p["point"]["corr"] - pm_g["gen"]["corr"], Pid),
              "P_fp_minus_G_single_fp": C0.cluster_ci(pm_p["point"]["pm"]["fp_rate"] - pm_g["gen"]["pm"]["fp_rate"], pats, BOOT_N, BOOT_SEED),
              "fd_P": point["point"]["fd_descriptive"], "fd_G": gen["gen"]["fd"]}
    sanity["expected_pattern"] = bool(sanity["P_corr_minus_G_single_corr"][0] > 0 and sanity["P_fp_minus_G_single_fp"][0] < 0 and sanity["fd_G"] < sanity["fd_P"])
    return {"point": point, "gen": gen, "bootstrap": boot, "gates": gts, "condition": cond, "sanity": sanity,
            "arrays": {"pts": pts, "gens": gens}, "windows": int(len(Y)), "patients": int(np.unique(Pid).size)}


def k16_all(ctx, arms, mu_P, dev, ex):
    S = ctx["S"]
    out = {}
    for a in arms:
        Ks = np.stack([gen_out(a, ctx["X"][S], ctx["R"][S], SM.window_noise(ctx["Pid"][S], ctx["wid"][S], k), dev) for k in range(K)])
        out[a] = k16_block(ctx, Ks, mu_P[S], ex)
        print(f"[dp0] k16 {a}: within {out[a]['within_condition_waveform_diversity'][0]:.4f} R-SD {out[a]['r_time_seed_sd_ms_median']:.2f} ms", flush=True)
    return out


def stage_eval_dev(ex, dev):
    changed = check_prereg_committed()
    for n in ("detector", "point", "gen", *SHARES):
        if not (OUT / f"{n}.pt").exists():
            raise SystemExit(f"STOP: missing checkpoint {n}")
    ctx = context("dp_dev", ex, dev)
    Y, Pid = ctx["Y"], ctx["Pid"]
    placed = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    pmp = RM.patient_macro_rows(placed["n_tp"], placed["n_fp"], placed["n_fn"], Pid)
    det = {f"pm_{k}": C0.cluster_ci(pmp[k], pmp["patients"], BOOT_N, BOOT_SEED) for k in ("fp_rate", "recall", "precision", "f1")}
    det["pooled"] = RM.pooled_prf(**pmp["pooled"])
    ev = evaluate(ctx, ("point", *SHARES), ("gen", *SHARES), ex, dev)
    arr = ev.pop("arrays")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "dev_outputs.npz", **{f"point_{k}": v.astype(np.float32) for k, v in arr["pts"].items()},
             **{f"gen_{k}": v.astype(np.float32) for k, v in arr["gens"].items()})
    k16 = k16_all(ctx, ("gen", *SHARES), arr["pts"]["point"], dev, ex)
    meta = {"windows": ev["windows"], "patients": ev["patients"], "k16_windows": int(len(ctx["S"])), "k16_rule": f"salted rank '{K16_SALT}'",
            "shuffle": "default_rng(20261002).permutation over DP-DEV windows; PPG only", "post_prereg_code_changes": changed}
    write_json("dev_point_metrics.json", meta | {"detector_events": det, "point": ev["point"], "sanity": ev["sanity"]})
    write_json("dev_gen_metrics.json", meta | {"gen": ev["gen"], "k16": k16})
    write_json("dev_bootstrap.json", meta | {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": ev["bootstrap"]})
    write_json("dev_gates.json", ev["gates"])
    write_json("condition_shuffle_dev.json", ev["condition"])
    write_json("feature_diagnostics.json", feature_diagnostics(ctx, dev))
    for a in SHARES:
        print(f"[dp0] DP-DEV {a}: gates {json.dumps(ev['gates'][a])}", flush=True)


# ----------------------------------------------------------------------------------------------- gradient interaction
def grad_batches(n_train):
    rng = np.random.default_rng(GRAD_SEED)
    return [np.sort(rng.choice(n_train, PROTO["batch"], replace=False)) for _ in range(N_GRAD)]


def shared_grad(net, loss):
    g = net.groups()["shared"]
    gr = torch.autograd.grad(loss, g, allow_unused=True)
    return torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(gr, g)])


def stage_grad_conflict(ex, dev):
    Xt, Yt, Rt, _ = _train_data(ex, dev)
    bl = grad_batches(len(Xt))
    out = {"minibatches": N_GRAD, "batch": PROTO["batch"], "seed": GRAD_SEED, "population": "DP-TRAIN (reference-R raster, training convention)",
           "fm_draw": "per minibatch i: CPU Generator(20261002 + i) -> x0 then t"}
    for s in SHARES:
        out[s] = {}
        for label, c in GRAD_POINTS.items():
            net = load_net(s if label == "final" else f"{s}_init" if c == 0 else f"{s}_c{c}", dev)
            cos, n_p, n_f = [], [], []
            for i, b in enumerate(bl):
                bt = torch.from_numpy(b).to(dev)
                gp = shared_grad(net, (net.point(Xt[bt], Rt[bt]) - Yt[bt]).abs().mean())
                gen = torch.Generator().manual_seed(GRAD_SEED + i)
                x0 = torch.randn((len(b), T), generator=gen).to(dev)
                t = torch.rand(len(b), generator=gen).to(dev)
                xt, u = SM.fm_path(Yt[bt], x0, t)
                gf = shared_grad(net, ((net.flow(xt, t, Xt[bt], Rt[bt]) - u) ** 2).mean())
                cos.append(float(torch.nn.functional.cosine_similarity(gp, gf, dim=0)))
                n_p.append(float(gp.norm())); n_f.append(float(gf.norm()))
            cos = np.array(cos)
            out[s][label] = {"cycle": c, "median": float(np.median(cos)), "q25": float(np.percentile(cos, 25)), "q75": float(np.percentile(cos, 75)),
                             "iqr": float(np.percentile(cos, 75) - np.percentile(cos, 25)), "fraction_negative": float((cos < 0).mean()),
                             "mean": float(cos.mean()), "point_grad_norm_median": float(np.median(n_p)), "flow_grad_norm_median": float(np.median(n_f)),
                             "shared_params": int(sum(p.numel() for p in net.groups()["shared"]))}
            print(f"[dp0] grad {s} {label}: median {out[s][label]['median']:+.4f} neg {out[s][label]['fraction_negative']:.3f}", flush=True)
    write_json("gradient_conflict.json", out)


# ----------------------------------------------------------------------------------------------- compute
def stage_compute(ex, dev):
    from torch.utils.flop_counter import FlopCounterMode
    assert not B.other_gpu_procs(), B.other_gpu_procs()
    X, _, Pid, wid, _ = load_role("dp_dev", ex)
    pick = B.salted_rank("dp0-latency-v1", range(len(X)))[:50]
    torch.set_num_threads(4)
    lat, mem, flops = {}, {}, {}
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        det = load_net("detector", d)
        nets = {n: load_net(n, d) for n in ("point", "gen", *SHARES)}

        def raster(xi):
            p = torch.sigmoid(det(xi[:, None])[:, 0]).float().cpu().numpy()[0]
            return torch.from_numpy(AB.event_raster([_extract(p)])).to(d)

        def run(n, m, xi, ri, x0, cond):
            """One request; n = 'separate' (SPECIALIST P + SPECIALIST G) or a dual pattern."""
            z = torch.zeros(1, device=d)
            if n == "separate":
                return {"point_only": lambda: nets["point"](xi, ri), "gen_only": lambda: SM.euler(nets["gen"], x0, xi, ri, NFE),
                        "both": lambda: (nets["point"](xi, ri), SM.euler(nets["gen"], x0, xi, ri, NFE)),
                        "one_nfe": lambda: nets["gen"](x0, z, xi, ri)}[m]()
            net = nets[n]
            if m == "both":
                mu, c = net.both(xi, ri)                                           # one shared trunk pass, cached condition
                return mu, D.euler(net, x0, xi, ri, NFE, cond=c)
            return {"point_only": lambda: net.point(xi, ri), "gen_only": lambda: D.euler(net, x0, xi, ri, NFE),
                    "both_uncached": lambda: (net.point(xi, ri), D.euler(net, x0, xi, ri, NFE)),
                    "one_nfe": lambda: net.velocity(x0, z, cond), "encode_flow": lambda: net.encode_flow(xi, ri)}[m]()

        for n in ("separate", *SHARES):
            ms = ("point_only", "gen_only", "both", "one_nfe") + (() if n == "separate" else ("both_uncached", "encode_flow"))
            for m in ms:
                for with_det in ((False, True) if m in ("point_only", "gen_only", "both", "both_uncached") else (False,)):
                    ts = []
                    for rep, i in enumerate(list(pick[:3]) + list(pick)):
                        xi = torch.from_numpy(X[i:i + 1]).to(d)
                        x0 = torch.from_numpy(SM.window_noise(Pid[i:i + 1], wid[i:i + 1])).to(d)
                        with torch.no_grad():
                            ri0 = raster(xi)
                            cond = None if n == "separate" else nets[n].encode_flow(xi, ri0)
                            if dname == "cuda":
                                torch.cuda.synchronize()
                            t0 = time.perf_counter()
                            run(n, m, xi, raster(xi) if with_det else ri0, x0, cond)
                            if dname == "cuda":
                                torch.cuda.synchronize()
                        if rep >= 3:
                            ts.append((time.perf_counter() - t0) * 1000)
                    lat[f"{dname}_{n}_{m}" + ("_with_detector" if with_det else "")] = float(np.median(ts))
                if dname == "cuda":
                    xi = torch.from_numpy(X[:1]).to(d)
                    x0 = torch.from_numpy(SM.window_noise(Pid[:1], wid[:1])).to(d)
                    with torch.no_grad():
                        ri = raster(xi)
                        cond = None if n == "separate" else nets[n].encode_flow(xi, ri)
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                        run(n, m, xi, ri, x0, cond); torch.cuda.synchronize()
                        mem[f"{n}_{m}"] = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
        if dname == "cpu":
            xi = torch.from_numpy(X[:1])
            with torch.no_grad():
                ri = raster(xi)
            z, t0_ = torch.zeros(1, T), torch.zeros(1)

            def fl(fn):
                fc = FlopCounterMode(display=False)
                with fc, torch.no_grad():
                    fn()
                return int(fc.get_total_flops())
            flops["detector"] = fl(lambda: det(xi[:, None]))
            flops["point_specialist"] = fl(lambda: nets["point"](xi, ri))
            flops["gen_specialist_per_nfe"] = fl(lambda: nets["gen"](z, t0_, xi, ri))
            flops["separate_both_nfe8"] = flops["point_specialist"] + NFE * flops["gen_specialist_per_nfe"]
            for s in SHARES:
                net = nets[s]
                with torch.no_grad():
                    h = net.trunk(xi, ri)
                    cond = net.flow_features(h)
                f = {"shared_trunk": fl(lambda: net.trunk(xi, ri)), "point_branch": fl(lambda: net.point_dec(net.point_features(h))),
                     "flow_features": fl(lambda: net.flow_features(h)), "flow_decoder_per_nfe": fl(lambda: net.velocity(z, t0_, cond))}
                f["point_only"] = f["shared_trunk"] + f["point_branch"]
                f["gen_only_nfe8"] = f["shared_trunk"] + f["flow_features"] + NFE * f["flow_decoder_per_nfe"]
                f["both_cached_nfe8"] = f["shared_trunk"] + f["point_branch"] + f["flow_features"] + NFE * f["flow_decoder_per_nfe"]
                f["both_uncached_nfe8"] = f["point_only"] + f["gen_only_nfe8"]
                flops[s] = f
    cks = {p.stem: json.loads(p.read_text()) for p in sorted((ART / "checkpoints").glob("*.json"))}
    write_json("compute_accounting.json", {"latency_ms_batch1": lat, "gpu_peak_inference_mib_above_weights": mem, "flops_one_window": flops,
                                           "training": {k: {"params": v["n_params"], "train_seconds": v["train_seconds"], "peak_gpu_mem_mib": v["peak_gpu_mem_mib"]}
                                                        for k, v in cks.items() if "_" not in k},
                                           "notes": "batch-1 median over 50 salted DP-DEV windows after 3 warm-ups; CPU 4 threads; '_with_detector' includes "
                                                    "detector + event extraction + raster inside the timed request; dual 'both' reuses one shared trunk pass "
                                                    "(cached condition), 'both_uncached' recomputes it per readout; 'one_nfe' = one vector-field evaluation "
                                                    "with the condition already encoded (dual) or the full ScaleFlow forward (G); FLOPs from "
                                                    "torch.utils.flop_counter on one window",
                                           "software": B.software()})
    write_json("checkpoint_hashes.json", {k: v["sha256"] for k, v in cks.items()})


# ----------------------------------------------------------------------------------------------- selection / summaries
def stage_select(ex, dev):
    gts = read_json("dev_gates.json")
    boot = read_json("dev_bootstrap.json")["comparisons"]
    gm = read_json("dev_gen_metrics.json")["gen"]
    pmx = read_json("dev_point_metrics.json")["point"]
    acc = read_json("parameter_accounting.json")
    lat = read_json("compute_accounting.json")["latency_ms_batch1"]
    cands = {s: {"qualified": gts[s]["QUALIFIED"], "gates": gts[s], "saving": acc[s]["saving"], "fd": gm[s]["fd"], "corr": pmx[s]["corr"][0],
                 "latency_both_ms": lat[f"cuda_{s}_both"], "comparisons": boot[s]} for s in SHARES}
    win = D.select_winner(cands)
    rule = "qualify on P1, P2, G1, CONDITION, E1; then largest parameter saving, lower generative FD, higher point corr, lower combined latency"
    write_json("candidate_selection.json", {"candidates": cands, "winner": win, "rule": rule,
                                            "verdict": None if win else "DP0 DEVELOPMENT FAILED"})
    print(f"[dp0] development winner: {win or 'NONE (DP0 DEVELOPMENT FAILED)'}", flush=True)


def fmt(v, nd):
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def stage_summarize(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pmx, gm = read_json("dev_point_metrics.json"), read_json("dev_gen_metrics.json")
    boot, gts = read_json("dev_bootstrap.json")["comparisons"], read_json("dev_gates.json")
    acc, gc = read_json("parameter_accounting.json"), read_json("gradient_conflict.json")
    comp, sel = read_json("compute_accounting.json"), read_json("candidate_selection.json")
    lat = comp["latency_ms_batch1"]
    with open(ART / "table_point_dev.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "morph_corr", "fp_per_window", "recall", "f1_patient_macro", "f1_pooled", "rr_mae_ms", "hr_mae_bpm", "mae",
                    "fd_descriptive", "gpu_latency_ms"])
        for a, lab in (("point", "SPECIALIST P"), *((s, f"{s} point") for s in SHARES)):
            m = pmx["point"][a]
            par = acc["specialist_point"] if a == "point" else acc[a]["point_path"]
            w.writerow([lab, par, fmt(m['corr'][0], 4), fmt(m['pm_fp_rate'][0], 4), fmt(m['pm_recall'][0], 4), fmt(m['pm_f1'][0], 4),
                        fmt(m['pooled']['f1'], 4), fmt(m['rr_mae_ms'][0], 2), fmt(m['hr_mae_bpm'][0], 2), fmt(m['mae'][0], 4),
                        fmt(m['fd_descriptive'], 3), fmt(lat['cuda_separate_point_only' if a == 'point' else f'cuda_{a}_point_only'], 2)])
    with open(ART / "table_gen_dev.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "total_params", "fd", "morph_corr_desc", "fp_desc", "diversity_ratio_rms_dev_from_P", "nfe", "gpu_latency_ms"])
        for a, lab in (("gen", "SPECIALIST G"), *((s, f"{s} gen") for s in SHARES)):
            m = gm["gen"][a]
            par = acc["specialist_gen"] if a == "gen" else acc[a]["total"]
            w.writerow([lab, par, fmt(m['fd'], 3), fmt(m['single_sample']['corr'][0], 4), fmt(m['single_sample']['pm_fp_rate'][0], 4),
                        fmt(m['diversity_ratio_rms_dev_from_P'], 4), NFE, fmt(lat['cuda_separate_gen_only' if a == 'gen' else f'cuda_{a}_gen_only'], 2)])
    fig, ax = plt.subplots(2, 4, figsize=(25, 10.5))
    a = ax[0, 0]
    a.axis("off")
    a.set_title("A  DualReadout-ECG", loc="left")
    a.text(0.0, 0.97, "            PPG + event raster\n                    |\n     shared conditioning encoder\n"
           "   (stem | blocks 1-6 | blocks 7-8)\n          /                \\\n   point adapter       flow adapter\n"
           "   (+ tower S1/S2)     (+ tower S1/S2)\n         |                   |  H_256, H_128 (Haar low-pass)\n"
           "   point decoder       ScaleFlow decoder (x_t, t)\n         |                   |\n        mu              ECG samples (Euler 8)\n\n"
           "S0 FULL: share all   S1 MIDDLE: stem + blocks 1-6\nS2 STEM-ONLY: stem    outputs never added",
           va="top", family="monospace", fontsize=9)
    a = ax[0, 1]
    names = ["separate", *SHARES]
    tot = [acc["separate_waveform_params"]] + [acc[s]["total"] for s in SHARES]
    a.bar(range(4), tot, color=["0.45", "#4a3aa7", "#7a6fd0", "#b3abe6"])
    a.axhline(acc["e1_threshold"], color="k", ls="--", lw=0.8)
    a.set_xticks(range(4), ["P + G", "S0", "S1", "S2"])
    a.set_title("B  waveform parameters (dashed: E1 0.85 x separate)", loc="left")

    def ciplot(a, key, title, margin, lab):
        for i, s in enumerate(SHARES):
            v = boot[s][key]
            a.plot([i, i], [v[1], v[2]], color="k", lw=3)
            a.plot([i], [v[0]], "o", color="#4a3aa7")
        a.axhline(margin, color="k", ls="--", lw=0.8)
        a.axhline(0, color="0.7", lw=0.6)
        a.set_xticks(range(3), SHARES)
        a.set_title(title, loc="left")
        a.set_ylabel(lab)
    ciplot(ax[0, 2], "corr", "C  P1: point corr dual - P (margin -0.02)", -D.M_CORR, "corr")
    ciplot(ax[0, 3], "fp", "C'  P2: FP/window dual - P (margin +0.05)", D.M_FP, "FP / window")
    ciplot(ax[1, 0], "fd", "D  G1: FD dual - G (margin +1.0)", D.M_FD, "FD")
    a = ax[1, 1]
    for i, s in enumerate(SHARES):
        v = boot[s]["fd_shuf"]
        a.plot([i, i], [v[1], v[2]], color="#2a7f3f", lw=3)
        a.plot([i], [v[0]], "o", color="#2a7f3f")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(3), SHARES)
    a.set_title("E  PPG shuffle: FD shuffled - conditioned (> 0)", loc="left")
    a = ax[1, 2]
    for j, (lab, c) in enumerate((("init", "0.6"), ("25pct", "#d59a54"), ("final", "#4a3aa7"))):
        for i, s in enumerate(SHARES):
            g = gc[s][lab]
            a.plot([i + (j - 1) * 0.2] * 2, [g["q25"], g["q75"]], color=c, lw=3, label=lab if i == 0 else None)
            a.plot([i + (j - 1) * 0.2], [g["median"]], "o", color=c)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(3), SHARES)
    a.legend(fontsize=8)
    a.set_title("F  shared-gradient cosine (median, IQR)", loc="left")
    a = ax[1, 3]
    modes = ("point_only", "gen_only", "both")
    for j, m in enumerate(modes):
        a.bar(np.arange(4) + (j - 1) * 0.27, [lat[f"cuda_{n}_{m}"] for n in names], width=0.27, label=m)
    a.set_xticks(range(4), ["P + G", "S0", "S1", "S2"])
    a.legend(fontsize=8)
    a.set_title("G  GPU batch-1 latency (ms, waveform models)", loc="left")
    fig.suptitle(f"DP0 DualReadout — DP-DEV development (AF-LOCK {'pending' if sel['winner'] else 'not opened'}). "
                 f"Winner: {sel['winner'] or 'NONE (DP0 DEVELOPMENT FAILED)'}  |  gates: " +
                 "  ".join(f"{s} " + "".join(k[0] if gts[s][k] else "x" for k in ("P1", "P2", "G1", "CONDITION", "E1")) for s in SHARES), fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure_dev.png", dpi=105)


# ----------------------------------------------------------------------------------------------- lock (sealed)
def stage_freeze(ex, dev, share):
    sel = read_json("candidate_selection.json")
    if sel["winner"] != share or share not in SHARES:
        raise SystemExit("STOP: only the selected development winner can be frozen")
    files = [PREREG, *CODE_FILES, "outputs/dp0_dualreadout/detector.pt", "outputs/dp0_dualreadout/point.pt", "outputs/dp0_dualreadout/gen.pt",
             f"outputs/dp0_dualreadout/{share}.pt", "artifacts/dp0_dualreadout/sharing_graph.json", "artifacts/dp0_dualreadout/training_manifest.json",
             "artifacts/dp0_dualreadout/parameter_accounting.json", "artifacts/dp0_dualreadout/split_manifest.json",
             "artifacts/dp0_dualreadout/candidate_selection.json", "artifacts/af0_anchorflow/split_manifest.json", "src/ppg2ecg/scaleflow/model.py",
             "src/ppg2ecg/anchorflow/fastfd.py", "src/ppg2ecg/evaluation/paper_metrics.py", "src/ppg2ecg/evaluation/rpeaks.py",
             "src/ppg2ecg/probes/rhythm_tcn.py", "scripts/c0_coherentbeat.py", "scripts/bf0_run.py"]
    write_json("lock_freeze_manifest.json", {"winner": share, "nfe": NFE, "noise": "scaleflow.window_noise", "shuffle_seed": SPLIT_SEED,
                                             "bootstrap": {"replicates": BOOT_N, "seed": BOOT_SEED}, "sha256": {f: B.sha256_file(ROOT / f) for f in files}})


def stage_eval_lock(ex, dev):
    check_lock_freeze()
    fm = json.loads(LOCK_FREEZE.read_text())
    win = fm["winner"]
    ctx = context("af_lock", ex, dev)
    Y, Pid = ctx["Y"], ctx["Pid"]
    ev = evaluate(ctx, ("point", win), ("gen", win), ex, dev)
    arr = ev.pop("arrays")
    k16 = k16_all(ctx, ("gen", win), arr["pts"]["point"], dev, ex)
    placed = PMX.rpeak_prf_at(Y, Y, FS, 50.0, peaks=(ctx["ref"], ctx["ev"]))
    pmp = RM.patient_macro_rows(placed["n_tp"], placed["n_fp"], placed["n_fn"], Pid)
    meta = {"role": "AF-LOCK", "winner": win, "windows": ev["windows"], "patients": ev["patients"], "k16_windows": int(len(ctx["S"]))}
    write_json("lock_metrics.json", meta | {"detector_events": {f"pm_{k}": C0.cluster_ci(pmp[k], pmp["patients"], BOOT_N, BOOT_SEED)
                                                                for k in ("fp_rate", "recall", "precision", "f1")},
                                            "point": ev["point"], "gen": ev["gen"], "sanity": ev["sanity"]})
    write_json("lock_bootstrap.json", meta | {"unit": "patient", "replicates": BOOT_N, "seed": BOOT_SEED, "comparisons": ev["bootstrap"]})
    g = ev["gates"][win]
    write_json("lock_gates.json", {"winner": win, **g, "verdict": "CONFIRMED" if g["QUALIFIED"] else "FAILED"})
    write_json("condition_shuffle_lock.json", ev["condition"])
    write_json("k16_lock.json", meta | {"k16": k16})
    print(f"[dp0] AF-LOCK {win}: {json.dumps(g)}", flush=True)


def stage_lock_summarize(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lm, lb, lg = read_json("lock_metrics.json"), read_json("lock_bootstrap.json"), read_json("lock_gates.json")
    win, acc = lg["winner"], read_json("parameter_accounting.json")
    with open(ART / "table_lock.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "params", "morph_corr", "fp_per_window", "recall", "f1", "rr_mae_ms", "hr_mae_bpm", "mae", "fd"])
        for a, lab, par in (("point", "SPECIALIST P", acc["specialist_point"]), (win, f"{win} point", acc[win]["point_path"])):
            m = lm["point"][a]
            w.writerow([lab, par, fmt(m['corr'][0], 4), fmt(m['pm_fp_rate'][0], 4), fmt(m['pm_recall'][0], 4), fmt(m['pm_f1'][0], 4),
                        fmt(m['rr_mae_ms'][0], 2), fmt(m['hr_mae_bpm'][0], 2), fmt(m['mae'][0], 4), fmt(m['fd_descriptive'], 3)])
        for a, lab, par in (("gen", "SPECIALIST G", acc["specialist_gen"]), (win, f"{win} gen", acc[win]["total"])):
            m = lm["gen"][a]
            s = m["single_sample"]
            w.writerow([lab, par, fmt(s['corr'][0], 4), fmt(s['pm_fp_rate'][0], 4), fmt(s['pm_recall'][0], 4), fmt(s['pm_f1'][0], 4),
                        fmt(s['rr_mae_ms'][0], 2), fmt(s['hr_mae_bpm'][0], 2), fmt(s['mae'][0], 4), fmt(m['fd'], 3)])
    c = lb["comparisons"][win]
    fig, ax = plt.subplots(1, 5, figsize=(24, 4.8))
    for a, (key, title, margin) in zip(ax, (("corr", "P1 corr dual - P", -D.M_CORR), ("fp", "P2 FP dual - P", D.M_FP),
                                            ("recall", "P2 recall dual - P", -D.M_RECALL), ("fd", "G1 FD dual - G", D.M_FD),
                                            ("fd_shuf", "PPG shuffle FD shuf - cond", 0.0))):
        v = c[key]
        a.plot([0, 0], [v[1], v[2]], color="k", lw=4)
        a.plot([0], [v[0]], "o", color="#4a3aa7")
        a.axhline(margin, color="k", ls="--", lw=0.8)
        a.set_xticks([])
        a.set_title(title, loc="left")
    fig.suptitle(f"DP0 AF-LOCK confirmation — {win}: {lg['verdict']}  (E1 saving {acc[win]['saving']:.3f})")
    fig.tight_layout()
    fig.savefig(ART / "figure_lock.png", dpi=105)


STAGES = {"split": stage_split, "audit": stage_audit, "manifest": stage_manifest, "train_detector": stage_train_detector,
          "train_point": stage_train_point, "train_gen": stage_train_gen, "eval_dev": stage_eval_dev, "grad_conflict": stage_grad_conflict,
          "compute": stage_compute, "select": stage_select, "summarize": stage_summarize, "eval_lock": stage_eval_lock,
          "lock_summarize": stage_lock_summarize}
ARG_STAGES = {"train_dual": stage_train_dual, "freeze": stage_freeze}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES) + list(ARG_STAGES))
    ap.add_argument("arg", nargs="?")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        if args.stage in ARG_STAGES:
            ARG_STAGES[args.stage](ex, dev, args.arg)
        else:
            STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
