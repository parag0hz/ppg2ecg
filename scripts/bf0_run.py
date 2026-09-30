"""BF0 — Beat-First generator feasibility on VitalDB validation
(docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md, revision 2).

Stages, in order:
  manifest      sha256 of the preregistration and the BF0 code (no data access; written before the prereg commit)
  audit         window / beat counts and input hashes
  train_timing  N5-style timing head on RD1 events of the train split
  train_beat    BeatFlowNet twice on R-aligned train beats: A3 stochastic (OT-CFM) and A2 deterministic (L1);
                train-median template (A1) and RR
  render        validation: events -> mu, sigma -> positions -> arms A1/A2/A3 (16 realizations), oracle arms O1-O3,
                condition shuffles S-PPG / S-RR
  evaluate      metrics, bootstrap CIs, gates G1 G3 G4b G4a G2 (frozen order), diagnostics, verdict
  atlas         qualitative figure of fixed salted validation windows
  compute       parameters, training time, rendering time, batch-1 latency
  figure        the compact BF0 result figure

The test split is never loaded: `load_role` refuses it.
Run: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/bf0_run.py <stage>
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import vm1_evaluate as VM  # noqa: E402
from ppg2ecg.beatfirst import beats as BB  # noqa: E402
from ppg2ecg.beatfirst import model as BM  # noqa: E402
from ppg2ecg.beatfirst import render as BR  # noqa: E402
from ppg2ecg.beatfirst import timing as BT  # noqa: E402
from ppg2ecg.evaluation import metrics as MET  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402
from ppg2ecg.evaluation import rpeaks as RP  # noqa: E402
from ppg2ecg.evaluation.m1_structural import qrs_core_morphology  # noqa: E402
from ppg2ecg.probes.rhythm_tcn import RhythmTCN, extract_events  # noqa: E402

PREREG = "docs/BF0_BEAT_FIRST_FEASIBILITY_PREREGISTRATION.md"
CODE_FILES = ("scripts/bf0_run.py", "src/ppg2ecg/beatfirst/__init__.py", "src/ppg2ecg/beatfirst/beats.py",
              "src/ppg2ecg/beatfirst/model.py", "src/ppg2ecg/beatfirst/render.py", "src/ppg2ecg/beatfirst/timing.py",
              "tests/test_bf0_beatfirst.py")
ART = ROOT / "artifacts/bf0_beat_first"
RUN = ROOT / "outputs/bf0_beat_first"
RD1_CKPT = ROOT / "outputs/rd1_detector/checkpoint_last.pt"
RD1_TRAIN_PEAKS = ROOT / "outputs/rd1_detector/train_rpeaks.npz"
ED1_PARAMS = ROOT / "artifacts/ed1_consensus_decoding/params.json"
ED1_VAL_CACHE = ROOT / "outputs/ed1_cache/samples_I_val.npy"
SPLIT_MANIFEST = ROOT / "data/manifests/split_v1_vitaldb_seed42.json"
FS, T = 128, 512
THRESHOLD, REFRACTORY = 0.35, 32
SEED = 42
TIMING_STEPS, TIMING_BATCH = 6000, 256
BEAT_STEPS, BEAT_BATCH = 20000, 256
LR, WD, CLIP = 1e-3, 0.01, 1.0
K_REAL = 16
BOOT_N, BOOT_N_FD, BOOT_SEED = 2000, 1000, 20260930
SALT_CONFORMAL, SALT_CHECK, SALT_ATLAS, SALT_SHUFFLE = "bf0-conformal-v1", "bf0-peakcheck-v1", "bf0-atlas-v1", "bf0-shuffle-v1"
SALT_LATENCY = "bf0-latency-v1"
N_PEAK_CHECK, N_ATLAS, MIN_MATCHED_REAL, N_LATENCY = 200, 6, 8, 50
ALLOWED_ROLES = ("train", "val")
MARGIN = {"g1_f1": 0.02, "g1_rr_ms": 2.0, "g2_corr": 0.02, "g4a_sd_ms": 1000.0 / FS, "g4b_practical_ratio": 0.25}
PRIMARY_ARMS = ("A1", "A2", "A3", "A3_mean")        # predicted positions; G2 and the correlation diagnostics
SHUFFLE_GROUP = ("A3", "S_PPG", "S_RR")             # predicted positions; shuffle diagnostics
ORACLE_ARMS = ("O1", "O2", "O3")                    # reference R positions
GATE_ORDER = ("G1_rhythm_preservation", "G3_distributional_gain", "G4b_non_collapse", "G4a_timing_invariance",
              "G2_centre_consistency")
VERDICTS = {
    "A": "Case A — G1 FAIL: architecture premise failure; beat morphology rendering destroys rhythm fidelity. NO BF1.",
    "B": "Case B — G1 PASS, G3 FAIL: no demonstrated distributional advantage of stochastic morphology; "
         "generative morphology branch unsupported. NO automatic BF1.",
    "C": "Case C — G1/G3 PASS, G4b FAIL: the stochastic branch fails the meaningful-variation criterion; "
         "no meaningful stochastic-morphology claim.",
    "D": "Case D — G1/G3/G4b PASS, G4a FAIL: stochastic generation has value, but the rhythm-morphology "
         "disentanglement claim fails.",
    "E": "Case E — G1/G3/G4b/G4a PASS, G2 FAIL: distributionally useful and timing-stable, but the distribution "
         "centre is biased (qualified).",
    "F": "Case F — G1/G3/G4b/G4a/G2 PASS: BF0 supports the factorization premise.",
}


# ----------------------------------------------------------------------------------------------- pure helpers
def load_role(role: str):
    if role not in ALLOWED_ROLES:
        raise ValueError(f"BF0 never loads the {role!r} split")
    return VM.load(role)


def compute_gates(d: dict) -> dict:
    """Prereg §4 decision rules applied to the computed statistics ([point, CI low, CI high] triples)."""
    g1, g3, g4b, g4a, g2 = d["G1"], d["G3"], d["G4b"], d["G4a"], d["G2"]
    sd = g4a["A3_median_sd_ms"]
    return {
        "G1_rhythm_preservation": bool(g1["f1_diff"][1] > -MARGIN["g1_f1"] and g1["rr_mae_diff_ms"][2] < MARGIN["g1_rr_ms"]),
        "G3_distributional_gain": bool(all(g3[k][0] < 0 and g3[k][2] < 0 for k in ("A3-A1", "A3-A2"))),
        "G4b_non_collapse": bool(g4b["D(A3)-D(A2)"][1] > 0),
        "G4a_timing_invariance": bool(sd is not None and math.isfinite(sd) and sd <= MARGIN["g4a_sd_ms"]),
        "G2_centre_consistency": bool(g2["A3_mean-A1"][1] > -MARGIN["g2_corr"]),
    }


def verdict_case(g: dict) -> str:
    """Prereg §4: the first failing gate in the frozen order G1 -> G3 -> G4b -> G4a -> G2 decides the case."""
    for gate, case in zip(GATE_ORDER, "ABCDE"):
        if not g[gate]:
            return case
    return "F"


def primary_waveforms(R: dict) -> dict:
    """The one-waveform-per-window arrays used for FD (prereg §3): A3 contributes realization 0 only."""
    out = {"A1": np.asarray(R["A1"]), "A2": np.asarray(R["A2"]), "A3": np.asarray(R["A3"][0])}
    assert np.array_equal(out["A3"], np.asarray(R["A3_primary"])), "A3 primary must be realization 0"
    n = {k: v.shape for k, v in out.items()}
    assert len(set(n.values())) == 1, f"FD sample counts differ across arms: {n}"
    return out


def patient_resamples(n_subjects: int, n_rep: int, seed: int = BOOT_SEED) -> list[np.ndarray]:
    """Replicate r: indices of the patients drawn with replacement (shared by every statistic of the run)."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_subjects, n_subjects) for _ in range(n_rep)]


def patient_bootstrap_indices(pid, n_rep: int, seed: int = BOOT_SEED) -> list[np.ndarray]:
    """Window indices of each replicate: every window of every drawn patient (whole patients, with repeats)."""
    pid = np.asarray(pid)
    subs = np.unique(pid)
    rows = [np.flatnonzero(pid == s) for s in subs]
    return [np.concatenate([rows[k] for k in r]) for r in patient_resamples(len(subs), n_rep, seed)]


def cluster_ci(d, pid, n_rep: int = BOOT_N):
    """Equal-patient-weight mean of per-window values (nan windows skipped) and its patient-clustered 95% CI."""
    d = np.asarray(d, dtype=np.float64)
    pid = np.asarray(pid)
    subs = np.unique(pid)
    per = np.array([np.nanmean(d[pid == s]) if np.isfinite(d[pid == s]).any() else np.nan for s in subs])
    if not np.isfinite(per).any():
        return [float("nan")] * 3
    draws = np.array([np.nanmean(per[r]) for r in patient_resamples(len(subs), n_rep)])
    return [float(np.nanmean(per)), float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def beat_seed(n: int, s: int, j: int) -> int:
    """Noise seed of realization s of beat j in window n (prereg §2.4); s = 0 is the primary render."""
    return 1_000_003 * int(n) + 10_007 * int(s) + int(j)


def salted_rank(salt: str, keys) -> np.ndarray:
    return np.argsort([hashlib.sha256(f"{salt}:{k}".encode()).hexdigest() for k in keys], kind="stable")


def shuffle_donors(owner_pid) -> np.ndarray:
    """Prereg §2.6: order beats by sha256('bf0-shuffle-v1:' + index); the donor of beat k is the beat M/2 places later
    (cyclically), stepping forward one place at a time until the donor belongs to another patient."""
    pid = np.asarray(owner_pid)
    M = pid.size
    order = salted_rank(SALT_SHUFFLE, range(M))
    rank = np.empty(M, int)
    rank[order] = np.arange(M)
    donor = np.empty(M, int)
    for b in range(M):
        k = (rank[b] + M // 2) % M
        for _ in range(M):
            d = order[k]
            if pid[d] != pid[b]:
                break
            k = (k + 1) % M
        else:
            raise ValueError("every beat belongs to one patient; no donor exists")
        donor[b] = d
    return donor


def shuffled_conditions(ppg, rr, donor, which: str):
    """S-PPG: donor PPG segments with the beat's own RR; S-RR: own PPG with the donor's (RR_prev, RR_next)."""
    if which == "S_PPG":
        return ppg[donor], rr
    if which == "S_RR":
        return ppg, rr[donor]
    raise ValueError(which)


def conformal_half(pid: np.ndarray) -> np.ndarray:
    """True = half C (calibration), False = half E (evaluation); patient-level by sha256 parity."""
    subs = np.unique(pid)
    is_c = {int(s): int(hashlib.sha256(f"{SALT_CONFORMAL}{int(s)}".encode()).hexdigest(), 16) % 2 == 0 for s in subs}
    return np.array([is_c[int(s)] for s in pid])


def timing_sd(ref_list, peaks_by_window, min_matched=MIN_MATCHED_REAL):
    """Median over reference beats of the sample SD (ms) of matched detection times across realizations."""
    sds = []
    for ref, per_real in zip(ref_list, peaks_by_window):
        ref = np.asarray(ref, int)
        if ref.size == 0:
            continue
        times = [[] for _ in range(ref.size)]
        for pk in per_real:
            pk = np.asarray(pk, int)
            for i, j in RP.match_rpeaks(ref, pk, FS, 50.0)[0]:
                times[i].append(pk[j])
        sds += [float(np.std(np.asarray(tl, float), ddof=1)) / FS * 1000.0 for tl in times if len(tl) >= min_matched]
    return (float(np.median(sds)) if sds else float("nan")), len(sds)


def window_pair_means(corr_by_arm: dict, arms) -> dict:
    """Per-window mean pair correlation over the pairs that are finite in EVERY listed arm (matched population)."""
    n = len(corr_by_arm[arms[0]])
    out = {a: np.full(n, np.nan) for a in arms}
    n_pairs = 0
    for i in range(n):
        if len(corr_by_arm[arms[0]][i]) == 0:
            continue
        C = np.stack([np.asarray(corr_by_arm[a][i], float) for a in arms])
        ok = np.isfinite(C).all(axis=0)
        n_pairs += int(ok.sum())
        if ok.any():
            for k, a in enumerate(arms):
                out[a][i] = float(C[k, ok].mean())
    return out | {"_n_pairs": n_pairs}


def render_drift(pos_list, peaks_list):
    """Detected R (neurokit) vs placed position, matched within ±50 ms: offsets and detection fraction."""
    offs, n_placed = [], 0
    for pos, pk in zip(pos_list, peaks_list):
        pos, pk = np.asarray(pos, int), np.asarray(pk, int)
        n_placed += pos.size
        offs += [(int(pk[j]) - int(pos[i])) / FS * 1000.0 for i, j in RP.match_rpeaks(pos, pk, FS, 50.0)[0]]
    o = np.asarray(offs, float)
    return {"n_placed": int(n_placed), "n_detected_within_50ms": int(o.size),
            "fraction_detected": float(o.size / max(n_placed, 1)),
            "median_abs_offset_ms": float(np.median(np.abs(o))) if o.size else float("nan"),
            "mean_offset_ms": float(np.mean(o)) if o.size else float("nan"),
            "fraction_abs_offset_le_1_sample": float(np.mean(np.abs(o) <= 1000.0 / FS)) if o.size else float("nan")}


def seed_diversity(renders, anchors):
    """Mean over anchors of the pairwise RMS across realizations of the same 83-sample window
    (A3: placed positions; iMF: reference R). Returns (value, number of anchors)."""
    vals = []
    for n, pos in enumerate(anchors):
        for p in np.asarray(pos, int):
            if p - BR.BW_BEFORE < 0 or p + BR.BW_AFTER > renders.shape[-1]:
                continue
            vals.append(BR.mean_pairwise_rms(np.asarray(renders[:, n, p - BR.BW_BEFORE:p + BR.BW_AFTER], np.float64)))
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def clean(o):
    """JSON-safe copy: numpy scalars to Python, nan / inf to None."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(float(o)) else None
    return o


# ----------------------------------------------------------------------------------------------- io helpers
def other_gpu_procs():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [ln.strip() for ln in q.stdout.splitlines() if ln.strip() and int(ln.split(",")[0]) != os.getpid()]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def write_json(name: str, obj) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(clean(obj), indent=1))


def software() -> dict:
    import neurokit2
    import scipy
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.processor())
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "neurokit2": neurokit2.__version__, "platform": platform.platform(),
            "cpu": cpu, "n_cpu": os.cpu_count(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda,
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()}


def split_list(idx: np.ndarray, off: np.ndarray):
    return [idx[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def pack_list(lst):
    off = np.cumsum([0] + [len(a) for a in lst])
    return (np.concatenate([np.asarray(a) for a in lst]) if off[-1] else np.zeros(0)), off


def _peaks(sig):
    return np.asarray(RP.detect_rpeaks(np.asarray(sig, np.float64), FS), int)


def _events(p):
    return extract_events(p, THRESHOLD, REFRACTORY)


def train_reference_peaks(Ytr, ex):
    d = np.load(RD1_TRAIN_PEAKS)
    peaks = split_list(d["idx"], d["off"])
    assert len(peaks) == len(Ytr), (len(peaks), len(Ytr))
    chk = salted_rank(SALT_CHECK, range(len(Ytr)))[:N_PEAK_CHECK]
    fresh = list(ex.map(_peaks, [Ytr[i] for i in chk]))
    for i, f in zip(chk, fresh):
        assert np.array_equal(np.asarray(peaks[i], int), f), f"RD1 train peak cache differs at window {i}"
    return peaks


def val_reference_peaks(Yv, ex):
    f = RUN / "val_rpeaks.npz"
    if f.exists():
        d = np.load(f)
        return [np.asarray(p, int) for p in split_list(d["idx"], d["off"])]
    P = list(ex.map(_peaks, list(Yv), chunksize=256))
    idx, off = pack_list(P)
    np.savez(f, idx=idx.astype(int), off=off)
    return P


def load_detector(dev):
    net = RhythmTCN().to(dev)
    net.load_state_dict(torch.load(RD1_CKPT, map_location="cpu")["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def detector_fields(net, X, dev, bs=2048):
    out = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(dev)[:, None]
        out.append(torch.sigmoid(net(xb)[:, 0]).float().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------- manifest / audit
def stage_manifest(ex, dev):
    write_json("prereg_manifest.json", {
        "prereg": PREREG, "prereg_sha256": sha256_file(ROOT / PREREG),
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES},
        "parent_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "written_before_any_bf0_number": True})


def stage_audit(ex, dev):
    Xtr, Ytr, Ptr = load_role("train")
    Xv, Yv, Pv = load_role("val")
    ref_tr = train_reference_peaks(Ytr, ex)
    ref_v = val_reference_peaks(Yv, ex)
    E, _, _, W = BB.extract_training_beats(Xtr, Ytr, ref_tr)
    half = conformal_half(Pv)
    n_win = np.array([np.sum(Pv == s) for s in np.unique(Pv)])
    res = {"prereg": PREREG,
           "train": {"windows": int(len(Xtr)), "patients": int(len(np.unique(Ptr))),
                     "reference_peaks": int(sum(len(p) for p in ref_tr)), "training_beats": int(len(E)),
                     "windows_with_training_beats": int(len(np.unique(W)))},
           "val": {"windows": int(len(Xv)), "patients": int(len(np.unique(Pv))),
                   "windows_per_patient": {"min": int(n_win.min()), "median": float(np.median(n_win)), "max": int(n_win.max())},
                   "reference_peaks": int(sum(len(p) for p in ref_v)),
                   "windows_without_reference_peaks": int(sum(len(p) == 0 for p in ref_v)),
                   "half_C_patients": int(len(np.unique(Pv[half]))), "half_E_patients": int(len(np.unique(Pv[~half])))},
           "patient_overlap_train_val": int(len(np.intersect1d(np.unique(Ptr), np.unique(Pv)))),
           "test_loaded": False}
    assert res["patient_overlap_train_val"] == 0
    write_json("audit.json", res)
    write_json("input_hashes.json", {
        "rd1_checkpoint": sha256_file(RD1_CKPT), "rd1_train_peaks_cache": sha256_file(RD1_TRAIN_PEAKS),
        "split_manifest": sha256_file(SPLIT_MANIFEST), "ed1_val_sample_cache": sha256_file(ED1_VAL_CACHE),
        "ed1_params": sha256_file(ED1_PARAMS), "prereg": sha256_file(ROOT / PREREG),
        "code": {f: sha256_file(ROOT / f) for f in CODE_FILES}, "software": software()})
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- training
def _timing_chunk(args):
    field, x, events, ref = args
    F, R = [], []
    for f, xx, ev, rp in zip(field, x, events, ref):
        r = BT.matched_residuals(ev, rp)
        m = np.isfinite(r)
        if m.any():
            F.append(BT.event_features(f, xx, np.asarray(ev)[m]).astype(np.float16))
            R.append(r[m].astype(np.float32))
    return (np.concatenate(F) if F else None), (np.concatenate(R) if R else None)


def stage_train_timing(ex, dev):
    assert not other_gpu_procs(), f"another GPU job is running: {other_gpu_procs()}"
    torch.manual_seed(SEED)
    Xtr, Ytr, _ = load_role("train")
    ref = train_reference_peaks(Ytr, ex)
    det = load_detector(dev)
    field = detector_fields(det, Xtr, dev)
    events = list(ex.map(_events, list(field), chunksize=512))
    step = 4000
    jobs = [(field[i:i + step], Xtr[i:i + step], events[i:i + step], ref[i:i + step]) for i in range(0, len(Xtr), step)]
    parts = [p for p in ex.map(_timing_chunk, jobs) if p[0] is not None]
    F = torch.from_numpy(np.concatenate([p[0] for p in parts]).astype(np.float32))
    R = torch.from_numpy(np.concatenate([p[1] for p in parts]))
    n_events = int(sum(len(e) for e in events))
    print(f"[bf0] timing: {len(R):,} matched of {n_events:,} train events; residual sd {float(R.std()):.1f} ms", flush=True)
    head = BT.TimingHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(SEED)
    Fd, Rd = F.to(dev), R.to(dev)
    t0, acc, nan_steps = time.time(), [], 0
    head.train()
    for s in range(1, TIMING_STEPS + 1):
        idx = torch.randint(0, len(Rd), (TIMING_BATCH,), generator=g).to(dev)
        mu, ls = head(Fd[idx])
        loss = BT.gaussian_nll(Rd[idx], mu, ls).mean()
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward(); opt.step(); acc.append(loss.item())
        if s % 1000 == 0:
            print(f"[bf0] timing step {s} NLL {np.mean(acc):.4f}", flush=True); acc = []
    secs = time.time() - t0
    RUN.mkdir(parents=True, exist_ok=True)
    meta = {"steps": TIMING_STEPS, "seed": SEED, "n_train_events": int(len(R)), "n_train_detections": n_events,
            "train_seconds": secs, "n_params": sum(p.numel() for p in head.parameters()), "nan_steps": nan_steps,
            "checkpoint": "last step (no selection)"}
    torch.save({"state_dict": head.state_dict()} | meta, RUN / "timing_head.pt")
    write_json("checkpoint_timing_head.json", meta | {"sha256": sha256_file(RUN / "timing_head.pt")})


def _train_beat_model(kind: str, E, P, RR, dev):
    """kind 'stochastic' (A3, OT-CFM) or 'deterministic' (A2, L1); same data, architecture, optimiser, steps and seed."""
    torch.manual_seed(SEED)
    net = BM.BeatFlowNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    g = torch.Generator().manual_seed(SEED)
    Ed, Pd, RRd = (torch.from_numpy(a).to(dev) for a in (E, P, RR))
    t0, acc, nan_steps = time.time(), [], 0
    net.train()
    for s in range(1, BEAT_STEPS + 1):
        idx = torch.randint(0, len(Ed), (BEAT_BATCH,), generator=g)
        if kind == "stochastic":
            x0 = torch.randn((BEAT_BATCH, BB.SEG_LEN), generator=g).to(dev)
            t = torch.rand((BEAT_BATCH,), generator=g).to(dev)
            idx = idx.to(dev)
            loss = BM.cfm_loss(net, Ed[idx], Pd[idx], RRd[idx], x0, t)
        else:
            idx = idx.to(dev)
            loss = BM.l1_loss(net, Ed[idx], Pd[idx], RRd[idx])
        nan_steps += int(not torch.isfinite(loss))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step(); acc.append(loss.item())
        if s % 1000 == 0:
            print(f"[bf0] {kind} step {s} loss {np.mean(acc):.5f}", flush=True); acc = []
    return net, time.time() - t0, nan_steps


def stage_train_beat(ex, dev):
    assert not other_gpu_procs(), f"another GPU job is running: {other_gpu_procs()}"
    Xtr, Ytr, _ = load_role("train")
    ref = train_reference_peaks(Ytr, ex)
    E, P, RR, _ = BB.extract_training_beats(Xtr, Ytr, ref)
    template = np.median(E, axis=0)
    rr_all = np.concatenate([np.diff(np.asarray(p, float)) / FS for p in ref if len(p) >= 2])
    rr_median = float(np.median(rr_all))
    print(f"[bf0] beat: {len(E):,} training beats; train median RR {rr_median:.3f} s", flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez(RUN / "template.npz", template=template, rr_median=rr_median)
    cfg = {"segment": {"n_before": BB.N_BEFORE, "n_after": BB.N_AFTER, "length": BB.SEG_LEN},
           "network": {"channels": BM.CH, "kernel": BM.KERNEL, "dilations": list(BM.DILATIONS), "t_dim": BM.T_DIM,
                       "cond_hidden": BM.COND_HIDDEN},
           "optimiser": {"name": "AdamW", "lr": LR, "weight_decay": WD, "grad_clip": CLIP, "steps": BEAT_STEPS,
                         "batch": BEAT_BATCH, "seed": SEED},
           "sampler": {"name": "Euler", "steps": BM.EULER_STEPS}, "n_train_beats": int(len(E)),
           "train_median_rr_s": rr_median, "template": "sample-wise median of the training segments",
           "timing_head": {"steps": TIMING_STEPS, "batch": TIMING_BATCH, "lr": LR, "weight_decay": WD, "seed": SEED},
           "data": {"train_role": "train", "reference_peaks": rel(RD1_TRAIN_PEAKS)}}
    write_json("train_config.json", cfg)
    for kind, fname in (("stochastic", "beat_stochastic.pt"), ("deterministic", "beat_deterministic.pt")):
        net, secs, nan_steps = _train_beat_model(kind, E, P, RR, dev)
        meta = {"kind": kind, "steps": BEAT_STEPS, "seed": SEED, "n_params": BM.n_params(net),
                "n_train_beats": int(len(E)), "train_seconds": secs, "nan_steps": nan_steps,
                "checkpoint": "last step (no selection)"}
        torch.save({"state_dict": net.state_dict()} | meta, RUN / fname)
        write_json(f"checkpoint_{kind}.json", meta | {"sha256": sha256_file(RUN / fname)})


# ----------------------------------------------------------------------------------------------- render
def _positions(events, mu):
    pos = np.clip(np.asarray(events, int) + np.round(np.asarray(mu, float) * FS / 1000.0).astype(int), 0, T - 1)
    order = np.argsort(pos, kind="stable")
    return pos[order], order


def conditions(X, positions, rr_median):
    """Per placed beat: PPG segment, (RR_prev, RR_next), owner (window, beat index)."""
    PPG, RR, OWN = [], [], []
    for n, pos in enumerate(positions):
        prev, nxt = BB.neighbour_rr(pos, rr_median)
        for j, p in enumerate(pos):
            PPG.append(BB.padded_segment(X[n], int(p)))
            RR.append((prev[j], nxt[j]))
            OWN.append((n, j))
    return (np.asarray(PPG, np.float32).reshape(-1, BB.SEG_LEN), np.asarray(RR, np.float32).reshape(-1, 2),
            np.asarray(OWN, int).reshape(-1, 2))


@torch.no_grad()
def sample_stochastic(net, ppg, rr, own, s, dev, bs=4096):
    x0 = torch.stack([torch.randn((BB.SEG_LEN,), generator=torch.Generator().manual_seed(beat_seed(n, s, j)))
                      for n, j in own]) if len(own) else torch.zeros((0, BB.SEG_LEN))
    out = [BM.euler_sample(net, torch.from_numpy(ppg[i:i + bs]).to(dev), torch.from_numpy(rr[i:i + bs]).to(dev),
                           x0[i:i + bs].to(dev)).cpu().numpy() for i in range(0, len(ppg), bs)]
    return np.concatenate(out) if out else np.zeros((0, BB.SEG_LEN))


@torch.no_grad()
def predict_deterministic(net, ppg, rr, dev, bs=4096):
    out = [BM.deterministic_predict(net, torch.from_numpy(ppg[i:i + bs]).to(dev), torch.from_numpy(rr[i:i + bs]).to(dev)).cpu().numpy()
           for i in range(0, len(ppg), bs)]
    return np.concatenate(out) if out else np.zeros((0, BB.SEG_LEN))


def assemble_all(beats, positions, empty_fill):
    out, c = [], 0
    for pos in positions:
        k = len(pos)
        out.append(BR.assemble(beats[c:c + k], pos, T, empty_fill)[0])
        c += k
    assert c == len(beats)
    return np.asarray(out, np.float32)


def stage_render(ex, dev):
    assert not other_gpu_procs(), f"another GPU job is running: {other_gpu_procs()}"
    Xv, Yv, Pv = load_role("val")
    ref = val_reference_peaks(Yv, ex)
    det = load_detector(dev)
    field = detector_fields(det, Xv, dev)
    events = list(ex.map(_events, list(field), chunksize=256))
    head = BT.TimingHead().to(dev).eval()
    head.load_state_dict(torch.load(RUN / "timing_head.pt", map_location="cpu")["state_dict"])
    nets = {}
    for kind, fname in (("stochastic", "beat_stochastic.pt"), ("deterministic", "beat_deterministic.pt")):
        nets[kind] = BM.BeatFlowNet().to(dev).eval()
        nets[kind].load_state_dict(torch.load(RUN / fname, map_location="cpu")["state_dict"])
    tz = np.load(RUN / "template.npz")
    template, rr_median = tz["template"], float(tz["rr_median"])
    empty_fill = float(np.median(np.concatenate([template[:8], template[-8:]])))

    MU, SG, POS, EVS, RES, CONF = [], [], [], [], [], []
    for n in range(len(Xv)):
        ev = np.asarray(events[n], int)
        if ev.size:
            with torch.no_grad():
                mu, ls = head(torch.from_numpy(BT.event_features(field[n], Xv[n], ev)).to(dev))
            mu = mu.float().cpu().numpy()
            sg = torch.exp(BT.clamp_log_sigma(ls)).float().cpu().numpy()
        else:
            mu, sg = np.zeros(0), np.zeros(0)
        pos, order = _positions(ev, mu)
        ev, mu, sg = ev[order], mu[order], sg[order]
        MU.append(mu); SG.append(sg); POS.append(pos); EVS.append(ev)
        RES.append(BT.matched_residuals(ev, ref[n])); CONF.append(field[n][ev] if ev.size else np.zeros(0))
    ORC = [np.sort(np.asarray(r, int)) for r in ref]      # oracle positions: reference R, used only here and in evaluate

    ppg, rr, own = conditions(Xv, POS, rr_median)
    oppg, orr, oown = conditions(Xv, ORC, rr_median)
    donor = shuffle_donors(Pv[own[:, 0]])
    assert np.all(Pv[own[donor, 0]] != Pv[own[:, 0]])
    save = {"pid": Pv, "half_c": conformal_half(Pv), "donor": donor}
    timing = {}
    t0 = time.time()
    A3 = np.stack([assemble_all(sample_stochastic(nets["stochastic"], ppg, rr, own, s, dev), POS, empty_fill)
                   for s in range(K_REAL)])
    timing["A3_16_realizations_s"] = time.time() - t0
    save["A3"] = A3                                      # float32, like every other arm
    save["A3_primary"] = A3[0]
    save["A3_mean"] = A3.astype(np.float64).mean(axis=0).astype(np.float32)
    t0 = time.time()
    save["A2"] = assemble_all(predict_deterministic(nets["deterministic"], ppg, rr, dev), POS, empty_fill)
    timing["A2_s"] = time.time() - t0
    t0 = time.time()
    save["A1"] = np.asarray([BR.render_template(template, p, T, empty_fill) for p in POS], np.float32)
    timing["A1_s"] = time.time() - t0
    save["O3"] = assemble_all(sample_stochastic(nets["stochastic"], oppg, orr, oown, 0, dev), ORC, empty_fill)
    save["O2"] = assemble_all(predict_deterministic(nets["deterministic"], oppg, orr, dev), ORC, empty_fill)
    save["O1"] = np.asarray([BR.render_template(template, p, T, empty_fill) for p in ORC], np.float32)
    for which in ("S_PPG", "S_RR"):
        sp, sr = shuffled_conditions(ppg, rr, donor, which)
        save[which] = assemble_all(sample_stochastic(nets["stochastic"], sp, sr, own, 0, dev), POS, empty_fill)
    for name, L in (("pos", POS), ("events", EVS), ("mu", MU), ("sigma", SG), ("resid", RES), ("conf", CONF)):
        save[f"{name}_idx"], save[f"{name}_off"] = pack_list(L)
    np.savez(RUN / "render_val.npz", **save)
    n_empty = int(sum(len(p) == 0 for p in POS))
    nonfinite = {k: int(np.sum(~np.isfinite(np.asarray(save[k], np.float64))))
                 for k in ("A1", "A2", "A3", "A3_mean", "O1", "O2", "O3", "S_PPG", "S_RR")}
    write_json("render_manifest.json", {
        "windows": int(len(Xv)), "predicted_position_beats": int(len(ppg)), "oracle_beats": int(len(oppg)),
        "rd1_events": int(sum(len(e) for e in EVS)), "windows_without_positions": n_empty,
        "empty_fill_value": empty_fill, "train_median_rr_s": rr_median, "nonfinite_samples": nonfinite,
        "arms": {"A1": "template at predicted positions", "A2": "deterministic learned beat at predicted positions",
                 "A3": f"stochastic beat, realizations 0..{K_REAL - 1} at predicted positions (0 = primary)",
                 "A3_mean": "pointwise mean of the 16 A3 renders",
                 "O1/O2/O3": "template / deterministic / stochastic (s = 0) at reference R",
                 "S_PPG": "A3 s = 0 with donor PPG segments", "S_RR": "A3 s = 0 with donor RR pairs"},
        "wall_clock_s": timing, "outputs": rel(RUN / "render_val.npz")})
    write_json("seed_manifest.json", {
        "training_seed": SEED, "beat_noise_seed": "1_000_003*n + 10_007*s + j (window n, realization s, beat j)",
        "realizations": K_REAL, "primary_realization": 0,
        "roles": {"0": "primary render: FD (G3), single-waveform metrics, G1, G4b, S-PPG / S-RR / O3 noise",
                  "0-15": "G2 16-sample mean, G4a timing SD, seed-to-seed diversity"},
        "never": ["choosing a best-looking realization", "averaging the 16 realizations for G3 FD"]})
    write_json("shuffle_manifest.json", {
        "rule": "order predicted-position beats by sha256('bf0-shuffle-v1:' + global beat index); donor = the beat "
                "M/2 places later (cyclic), stepping forward one place until the donor belongs to another patient",
        "salt": SALT_SHUFFLE, "n_beats": int(len(donor)), "applies_to": {"S_PPG": "PPG segment", "S_RR": "(RR_prev, RR_next)"},
        "donor_sha256": hashlib.sha256(np.asarray(donor, np.int64).tobytes()).hexdigest(),
        "same_patient_pairs": int(np.sum(Pv[own[donor, 0]] == Pv[own[:, 0]])), "donor": [int(d) for d in donor]})
    print(f"[bf0] rendered {len(Xv):,} windows, {len(ppg):,} predicted-position beats, {n_empty} windows without positions", flush=True)


# ----------------------------------------------------------------------------------------------- evaluate
_FD: dict = {}


def _fd_init(a, b, Y):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    _FD["a"], _FD["b"], _FD["Y"] = a, b, Y


def _fd_draw(idx):
    a, b, Y = _FD["a"], _FD["b"], _FD["Y"]
    return PMX.kanflow_fd(a[idx], Y[idx]) - PMX.kanflow_fd(b[idx], Y[idx])


def fd_diff_ci(a, b, Y, pid):
    """FD(a) - FD(b); each replicate resamples whole patients and uses the same windows for both arms. Every replicate
    must keep >= FD_SMALL_SET windows so `kanflow_fd` stays in its raw-waveform regime."""
    jobs = patient_bootstrap_indices(pid, BOOT_N_FD, BOOT_SEED)
    n_min = min(len(j) for j in jobs)
    assert n_min >= PMX.FD_SMALL_SET, f"a bootstrap replicate has {n_min} windows: kanflow_fd would switch regime"
    with ProcessPoolExecutor(16, initializer=_fd_init, initargs=(a, b, Y)) as pool:
        draws = np.array(list(pool.map(_fd_draw, jobs, chunksize=8)))
    point = PMX.kanflow_fd(a, Y) - PMX.kanflow_fd(b, Y)
    return [float(point), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))], n_min


def _structure(args):
    pred, gt, pk = args
    m = qrs_core_morphology(pred, gt, pk)
    return m["qrs_deriv_rmse"], m["qrs_curvature_err"]


def arm_metrics(w, Yv, ref, pk, evaluable):
    prf = PMX.rpeak_prf_at(w, Yv, FS, 50.0, peaks=(ref, pk))
    beat = PMX.beat_level_metrics(w, Yv, FS, 50.0, peaks=(ref, pk))
    return {"f1": np.where(evaluable, prf["rpeak_f1"], np.nan), "rr_mae_ms": beat["rr_mae_ms"],
            "hr_abs_err": beat["hr_abs_err"], "morph_corr_detected": beat["morph_corr"],
            "pcc": MET.signal_metrics(w, Yv)["pcc"]}


def stage_evaluate(ex, dev):
    Xv, Yv, Pv = load_role("val")
    N = len(Yv)
    ref = val_reference_peaks(Yv, ex)
    evaluable = np.array([len(r) > 0 for r in ref])       # F1 only where the reference has >= 1 beat
    R = dict(np.load(RUN / "render_val.npz"))
    L = {k: split_list(R[f"{k}_idx"], R[f"{k}_off"]) for k in ("pos", "events", "mu", "sigma", "resid", "conf")}
    pos = [np.asarray(p, int) for p in L["pos"]]
    assert np.array_equal(R["pid"], Pv)
    import ed1_consensus_decode as ED
    S, P_imf = ED.samples_and_peaks("I", "val", Xv, ex, dev)
    ch = json.loads(ED1_PARAMS.read_text())["I"]["chosen"]
    decoded = ED.decode_all(S, P_imf, int(round(ch["w_ms"] / 1000 * FS)), ch["theta"], ch["b"], ex)[0]
    A3 = R["A3"].astype(np.float64)
    fdw = primary_waveforms(R)
    waves = {"A1": fdw["A1"], "A2": fdw["A2"], "A3": fdw["A3"], "A3_mean": R["A3_mean"],
             "O1": R["O1"], "O2": R["O2"], "O3": R["O3"], "S_PPG": R["S_PPG"], "S_RR": R["S_RR"],
             "imf_single": S[0], "imf_decoded": decoded}
    waves = {k: np.asarray(v, np.float64) for k, v in waves.items()}
    anchor = {k: (ref if (k in ORACLE_ARMS or k.startswith("imf")) else pos) for k in waves}

    # ---- detections (one neurokit pass per waveform)
    a3_peaks = [list(ex.map(_peaks, list(A3[s]), chunksize=256)) for s in range(K_REAL)]
    peaks = {"A3": a3_peaks[0], "imf_single": [p[0] for p in P_imf]}
    for k in waves:
        if k not in peaks:
            peaks[k] = list(ex.map(_peaks, list(waves[k]), chunksize=256))

    # ---- per-window metrics
    per, pooled = {}, {}
    real_beats = np.concatenate([BR.beat_windows(Yv[i], ref[i]) for i in range(N)])
    for name, w in waves.items():
        per[name] = arm_metrics(w, Yv, ref, peaks[name], evaluable)
        st = np.array(list(ex.map(_structure, [(w[i], Yv[i], ref[i]) for i in range(N)], chunksize=128)))
        per[name]["s4_qrs_deriv_rmse"], per[name]["s5_qrs_curvature_err"] = st[:, 0], st[:, 1]
        per[name]["diversity"] = np.array([BR.within_window_diversity(w[i], anchor[name][i]) for i in range(N)])
        arm_beats = np.concatenate([BR.beat_windows(w[i], anchor[name][i]) for i in range(N)])
        pooled[name] = {"fd": float(PMX.kanflow_fd(w, Yv)), "beat_fd": float(PMX.fid_frechet(arm_beats, real_beats)),
                        "n_beat_windows": int(len(arm_beats))}
    d_real = np.array([BR.within_window_diversity(Yv[i], ref[i]) for i in range(N)])

    # ---- matched-pair beat-aligned correlation (prereg §3): one pair set per position family
    pairs_pred = [BR.matched_pairs(ref[i], pos[i], T) for i in range(N)]
    pairs_orc = [BR.matched_pairs(ref[i], ref[i], T) for i in range(N)]
    corr = {a: [BR.pair_correlations(Yv[i], waves[a][i], pairs_pred[i]) for i in range(N)]
            for a in ("A1", "A2", "A3", "A3_mean", "S_PPG", "S_RR")}
    corr |= {a: [BR.pair_correlations(Yv[i], waves[a][i], pairs_orc[i]) for i in range(N)] for a in ORACLE_ARMS}
    wm_primary, wm_shuffle, wm_orc = (window_pair_means(corr, g) for g in (PRIMARY_ARMS, SHUFFLE_GROUP, ORACLE_ARMS))
    for a in PRIMARY_ARMS:
        per[a]["beat_corr_matched"] = wm_primary[a]
    for a in ("S_PPG", "S_RR"):
        per[a]["beat_corr_matched"] = wm_shuffle[a]
    per["A3"]["beat_corr_matched_shuffle_population"] = wm_shuffle["A3"]
    for a in ORACLE_ARMS:
        per[a]["beat_corr_matched"] = wm_orc[a]

    pos_bl = PMX.beat_level_metrics(Yv, Yv, FS, 50.0, peaks=(ref, pos))
    pos_seq = {"f1": np.where(evaluable, PMX.rpeak_prf_at(Yv, Yv, FS, 50.0, peaks=(ref, pos))["rpeak_f1"], np.nan),
               "rr_mae_ms": pos_bl["rr_mae_ms"], "hr_abs_err": pos_bl["hr_abs_err"]}
    ev_seq = [np.asarray(e, int) for e in L["events"]]
    ev_bl = PMX.beat_level_metrics(Yv, Yv, FS, 50.0, peaks=(ref, ev_seq))
    evs = {"f1": np.where(evaluable, PMX.rpeak_prf_at(Yv, Yv, FS, 50.0, peaks=(ref, ev_seq))["rpeak_f1"], np.nan),
           "rr_mae_ms": ev_bl["rr_mae_ms"], "hr_abs_err": ev_bl["hr_abs_err"]}
    summary = {name: {m: cluster_ci(v, Pv) for m, v in d.items()} | pooled[name] for name, d in per.items()}
    summary["positions_as_peaks"] = {m: cluster_ci(v, Pv) for m, v in pos_seq.items()}
    summary["rd1_events_as_peaks"] = {m: cluster_ci(v, Pv) for m, v in evs.items()}
    summary["real"] = {"diversity": cluster_ci(d_real, Pv)}

    # ---- gate statistics (frozen order G1 -> G3 -> G4b -> G4a -> G2)
    g1 = {"f1_diff": cluster_ci(per["A3"]["f1"] - pos_seq["f1"], Pv),
          "rr_mae_diff_ms": cluster_ci(per["A3"]["rr_mae_ms"] - pos_seq["rr_mae_ms"], Pv),
          "A3_render": {"f1": summary["A3"]["f1"], "rr_mae_ms": summary["A3"]["rr_mae_ms"]},
          "placed_positions": summary["positions_as_peaks"]}
    ci_a1, n_min = fd_diff_ci(waves["A3"], waves["A1"], Yv, Pv)
    ci_a2, _ = fd_diff_ci(waves["A3"], waves["A2"], Yv, Pv)
    g3 = {"FD": {k: pooled[k]["fd"] for k in ("A1", "A2", "A3")}, "A3-A1": ci_a1, "A3-A2": ci_a2,
          "min_replicate_windows": n_min}
    common = np.isfinite(per["A3"]["diversity"]) & np.isfinite(d_real)
    ratio = cluster_ci(np.where(common, per["A3"]["diversity"], np.nan), Pv)[0] / cluster_ci(np.where(common, d_real, np.nan), Pv)[0]
    sd_a3_seed, n_a3_seed = seed_diversity(A3, pos)
    sd_imf_seed, n_imf_seed = seed_diversity(S, ref)
    g4b = {"D(A3)-D(A2)": cluster_ci(per["A3"]["diversity"] - per["A2"]["diversity"], Pv),
           "D": {k: summary[k]["diversity"] for k in ("A1", "A2", "A3")}, "D_real": summary["real"]["diversity"],
           "D(A3)/D(real)": ratio, "practical_target_ratio": MARGIN["g4b_practical_ratio"],
           "practical_target_met": bool(ratio >= MARGIN["g4b_practical_ratio"]),
           "seed_to_seed_diversity": {"A3_at_placed_positions": sd_a3_seed, "A3_n_beats": n_a3_seed,
                                      "iMF_at_reference_R": sd_imf_seed, "iMF_n_beats": n_imf_seed}}
    per_window_a3 = [[a3_peaks[s][n] for s in range(K_REAL)] for n in range(N)]
    sd_a3, n_a3 = timing_sd(ref, per_window_a3)
    sd_imf, n_imf = timing_sd(ref, P_imf)
    g4a = {"A3_median_sd_ms": sd_a3, "A3_n_beats": n_a3, "iMF_median_sd_ms": sd_imf, "iMF_n_beats": n_imf,
           "threshold_ms": MARGIN["g4a_sd_ms"]}
    g2 = {"A3_mean-A1": cluster_ci(per["A3_mean"]["beat_corr_matched"] - per["A1"]["beat_corr_matched"], Pv),
          "A3_mean-A2 (secondary comparator)": cluster_ci(per["A3_mean"]["beat_corr_matched"] - per["A2"]["beat_corr_matched"], Pv),
          "levels": {k: summary[k]["beat_corr_matched"] for k in PRIMARY_ARMS},
          "n_matched_pairs": wm_primary["_n_pairs"]}
    detail = {"G1": g1, "G3": g3, "G4b": g4b, "G4a": g4a, "G2": g2}
    gates = compute_gates(detail)
    case = verdict_case(gates)

    # ---- diagnostics
    shuffles = {}
    for k in ("S_PPG", "S_RR"):
        ci, _ = fd_diff_ci(waves[k], waves["A3"], Yv, Pv)
        shuffles[k] = {"fd": pooled[k]["fd"], "fd_minus_A3": ci, "beat_fd": pooled[k]["beat_fd"],
                       "diversity": summary[k]["diversity"],
                       "diversity_minus_A3": cluster_ci(per[k]["diversity"] - per["A3"]["diversity"], Pv),
                       "beat_corr_matched": summary[k]["beat_corr_matched"],
                       "beat_corr_minus_A3": cluster_ci(per[k]["beat_corr_matched"] - per["A3"]["beat_corr_matched_shuffle_population"], Pv),
                       "f1": summary[k]["f1"]}
    shuffles["A3_conditioned"] = {"fd": pooled["A3"]["fd"], "beat_fd": pooled["A3"]["beat_fd"],
                                  "diversity": summary["A3"]["diversity"],
                                  "beat_corr_matched": summary["A3"]["beat_corr_matched_shuffle_population"],
                                  "f1": summary["A3"]["f1"]}
    oracle = {k: {m: summary[k][m] for m in ("fd", "beat_fd", "f1", "rr_mae_ms", "hr_abs_err", "beat_corr_matched",
                                              "morph_corr_detected", "diversity", "s4_qrs_deriv_rmse", "s5_qrs_curvature_err")}
              for k in ORACLE_ARMS}
    oracle["O3-O1 FD"] = fd_diff_ci(waves["O3"], waves["O1"], Yv, Pv)[0]
    oracle["O3-O2 FD"] = fd_diff_ci(waves["O3"], waves["O2"], Yv, Pv)[0]
    drift = {k: render_drift(anchor[k], peaks[k]) for k in ("A1", "A2", "A3", "O1", "O2", "O3")}
    drift["A3_all_16_realizations"] = render_drift([pos[n] for s in range(K_REAL) for n in range(N)],
                                                   [a3_peaks[s][n] for s in range(K_REAL) for n in range(N)])
    half = R["half_c"]
    cat = lambda key, sel: np.concatenate([L[key][n] for n in range(N) if sel[n]] or [np.zeros(0)])  # noqa: E731
    rc, mc, sc = cat("resid", half), cat("mu", half), cat("sigma", half)
    re, me, se, ce = cat("resid", ~half), cat("mu", ~half), cat("sigma", ~half), cat("conf", ~half)
    ok_c, ok_e = np.isfinite(rc), np.isfinite(re)
    scores = np.abs(rc[ok_c] - mc[ok_c]) / sc[ok_c]
    tert = np.quantile(ce[ok_e], [1 / 3, 2 / 3]) if ok_e.any() else [np.nan, np.nan]
    cov = {"n_matched_C": int(ok_c.sum()), "n_matched_E": int(ok_e.sum()), "confidence_tertile_edges": tert}
    for a in BT.ALPHAS:
        q = BT.conformal_quantile(scores, a)
        entry = {"q": q, "coverage": BT.coverage(re, me, se, q),
                 "coverage_uncalibrated_gaussian": BT.coverage(re, me, se, BT.gaussian_z(a)),
                 "mean_width_ms": float(np.mean(2 * q * se[ok_e])) if ok_e.any() else float("nan")}
        for t_name, m in (("low", ce < tert[0]), ("mid", (ce >= tert[0]) & (ce < tert[1])), ("high", ce >= tert[1])):
            entry[f"coverage_conf_{t_name}"] = BT.coverage(re[m], me[m], se[m], q)
            entry[f"mean_width_ms_conf_{t_name}"] = float(np.mean(2 * q * se[m & ok_e])) if (m & ok_e).any() else float("nan")
        cov[str(a)] = entry
    gt_stat = np.array([BR.boundary_statistic(Yv[i], ref[i]) for i in range(N)])
    thr = float(np.nanpercentile(gt_stat, 99.9))
    art = {"threshold": thr}
    for name in ("A1", "A2", "A3", "O1", "O2", "O3"):
        stat = np.array([BR.boundary_statistic(waves[name][i], anchor[name][i]) for i in range(N)])
        art[name] = float(np.mean(stat > thr))
    missing = {"windows": N, "windows_without_positions": int(sum(len(p) == 0 for p in pos)),
               "windows_without_reference_peaks": int((~evaluable).sum()),
               "per_arm_nan_windows": {k: {m: int(np.sum(~np.isfinite(v))) for m, v in d.items()} for k, d in per.items()},
               "G1_paired_windows": {"f1": int(np.sum(np.isfinite(per["A3"]["f1"] - pos_seq["f1"]))),
                                     "rr_mae": int(np.sum(np.isfinite(per["A3"]["rr_mae_ms"] - pos_seq["rr_mae_ms"])))},
               "G2_matched_pairs": wm_primary["_n_pairs"], "shuffle_matched_pairs": wm_shuffle["_n_pairs"],
               "oracle_matched_pairs": wm_orc["_n_pairs"],
               "reference_beats_with_window_inside": int(len(real_beats)),
               "G2_windows": int(np.sum(np.isfinite(per["A3_mean"]["beat_corr_matched"] - per["A1"]["beat_corr_matched"]))),
               "G4b_windows": int(np.sum(np.isfinite(per["A3"]["diversity"] - per["A2"]["diversity"]))),
               "G4b_ratio_windows": int(common.sum()),
               "G4a_beats": {"A3": n_a3, "iMF": n_imf}, "G3_windows_per_arm": {k: int(len(waves[k])) for k in ("A1", "A2", "A3")}}

    head = {"prereg": PREREG, "verdict_case": case, "verdict": VERDICTS[case]}
    write_json("gate_metrics.json", head | {"gates": gates, "gate_order": list(GATE_ORDER), "margins": MARGIN, "detail": detail})
    write_json("bootstrap.json", {"unit": "patient", "per_window_replicates": BOOT_N, "fd_replicates": BOOT_N_FD,
                                  "seed": BOOT_SEED, "weighting": "equal patient weight for per-window metrics",
                                  "fd_min_replicate_windows": n_min, "ci": "percentile 2.5 / 97.5", "summary_ci": summary})
    write_json("waveform_metrics.json", {k: summary[k] for k in waves} | {
        "positions_as_peaks": summary["positions_as_peaks"], "rd1_events_as_peaks": summary["rd1_events_as_peaks"],
        "artifact_rate_J": art,
        "notes": {"imf_decoded": "ED1 parameters were chosen on this validation split (in-sample comparator)",
                  "beat_corr_matched": "reference window at R vs arm window at its matched anchor, pairs shared by the group",
                  "morph_corr_detected": "rpeaks.morphology_corr on each arm's own neurokit detections (populations differ)"}})
    write_json("diversity_metrics.json", g4b | {"per_arm": {k: summary[k]["diversity"] for k in waves}})
    write_json("timing_metrics.json", {"G4a": g4a, "render_drift": drift, "positions_as_peaks": summary["positions_as_peaks"],
                                       "rd1_events_as_peaks": summary["rd1_events_as_peaks"], "coverage": cov})
    write_json("oracle_metrics.json", oracle)
    write_json("shuffle_metrics.json", shuffles)
    write_json("missingness.json", missing)
    write_json("result.json", head | {"gates": gates, "detail": detail, "oracle": oracle, "shuffles": shuffles,
                                      "render_drift": drift, "coverage": cov, "artifact_rate_J": art, "missingness": missing})
    print(json.dumps(clean({"gates": gates, "verdict": VERDICTS[case]}), indent=1))


# ----------------------------------------------------------------------------------------------- atlas / compute / figure
def stage_atlas(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Xv, Yv, Pv = load_role("val")
    R = dict(np.load(RUN / "render_val.npz"))
    pos = split_list(R["pos_idx"], R["pos_off"])
    sig = split_list(R["sigma_idx"], R["sigma_off"])
    S0 = np.load(ED1_VAL_CACHE, mmap_mode="r")[0]
    pick = salted_rank(SALT_ATLAS, range(len(Yv)))[:N_ATLAS]
    fig, axes = plt.subplots(N_ATLAS, 1, figsize=(12, 2.3 * N_ATLAS), sharex=True)
    t = np.arange(T) / FS
    for ax, i in zip(axes, pick):
        ax.plot(t, Yv[i], color="0.6", lw=1.5, label="reference ECG")
        ax.plot(t, R["A3_primary"][i], color="#0E7A86", lw=1.0, label="A3 stochastic (s = 0)")
        ax.plot(t, R["A2"][i], color="#4A3AA7", lw=0.8, ls=":", label="A2 deterministic")
        ax.plot(t, R["A1"][i], color="#B0621B", lw=0.8, ls="--", label="A1 template")
        ax.plot(t, np.asarray(S0[i], float), color="#A8434F", lw=0.6, alpha=0.6, label="iMF single sample")
        for p, s in zip(pos[i], sig[i]):
            ax.axvspan((p - s * FS / 1000) / FS, (p + s * FS / 1000) / FS, color="#0E7A86", alpha=0.12, lw=0)
        ax.set_ylabel(f"win {i}")
    axes[0].legend(ncol=5, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    ART.mkdir(parents=True, exist_ok=True)
    fig.savefig(ART / "atlas.png", dpi=150)


@torch.no_grad()
def _pipeline_once(x, det, head, net, kind, template, rr_median, dev):
    field = torch.sigmoid(det(torch.from_numpy(x[None, None].astype(np.float32)).to(dev))[0, 0]).float().cpu().numpy()
    ev = np.asarray(_events(field), int)
    if ev.size == 0:
        return 0
    mu, _ = head(torch.from_numpy(BT.event_features(field, x, ev)).to(dev))
    pos, _ = _positions(ev, mu.float().cpu().numpy())
    if kind == "template":
        BR.render_template(template, pos, T, 0.0)
        return len(pos)
    ppg, rr, own = conditions(x[None], [pos], rr_median)
    beats = sample_stochastic(net, ppg, rr, own, 0, dev) if kind == "stochastic" else predict_deterministic(net, ppg, rr, dev)
    BR.assemble(beats, pos, T, 0.0)
    return len(pos)


def stage_compute(ex, dev):
    assert not other_gpu_procs(), f"another GPU job is running: {other_gpu_procs()}"
    Xv, _, _ = load_role("val")
    tz = np.load(RUN / "template.npz")
    template, rr_median = tz["template"], float(tz["rr_median"])
    pick = salted_rank(SALT_LATENCY, range(len(Xv)))[:N_LATENCY]
    lat = {}
    torch.set_num_threads(4)
    for dname in ("cpu", "cuda"):
        d = torch.device(dname)
        det = load_detector(d)
        head = BT.TimingHead().to(d).eval()
        head.load_state_dict(torch.load(RUN / "timing_head.pt", map_location="cpu")["state_dict"])
        for kind, fname in (("template", None), ("deterministic", "beat_deterministic.pt"), ("stochastic", "beat_stochastic.pt")):
            net = None
            if fname:
                net = BM.BeatFlowNet().to(d).eval()
                net.load_state_dict(torch.load(RUN / fname, map_location="cpu")["state_dict"])
            for i in pick[:3]:
                _pipeline_once(Xv[i], det, head, net, kind, template, rr_median, d)
            ts, nb = [], []
            for i in pick:
                if dname == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                nb.append(_pipeline_once(Xv[i], det, head, net, kind, template, rr_median, d))
                if dname == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
            lat[f"{dname}_{kind}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
                                      "mean_beats_per_window": float(np.mean(nb))}
    ck = {k: json.loads((ART / f"checkpoint_{k}.json").read_text()) for k in ("stochastic", "deterministic", "timing_head")}
    rm = json.loads((ART / "render_manifest.json").read_text())
    write_json("compute_accounting.json", {
        "params": {"rd1_detector": sum(p.numel() for p in load_detector(torch.device("cpu")).parameters()),
                   "timing_head": ck["timing_head"]["n_params"], "beat_stochastic": ck["stochastic"]["n_params"],
                   "beat_deterministic": ck["deterministic"]["n_params"]},
        "train_seconds": {k: ck[k]["train_seconds"] for k in ck},
        "nan_steps": {k: ck[k]["nan_steps"] for k in ck},
        "render_wall_clock_s_all_validation": rm["wall_clock_s"],
        "network_forward_passes_per_window": {"rd1_detector": 1, "timing_head": 1, "beat_template": 0,
                                              "beat_deterministic": 1, "beat_stochastic": BM.EULER_STEPS,
                                              "note": "beat passes are batched over the window's beats"},
        "batch1_latency_ms": lat,
        "latency_protocol": f"{N_LATENCY} salted validation windows ('{SALT_LATENCY}'), 3 warm-up, CPU 4 threads; "
                            "detector + timing head + beat model (s = 0) + overlap-add",
        "other_gpu_processes": other_gpu_procs(), "software": software()})


def stage_figure(ex, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = json.loads((ART / "gate_metrics.json").read_text())
    orc = json.loads((ART / "oracle_metrics.json").read_text())
    shf = json.loads((ART / "shuffle_metrics.json").read_text())
    det = g["detail"]
    ok = lambda k: "PASS" if g["gates"][k] else "FAIL"  # noqa: E731
    C = {"A1": "#B0621B", "A2": "#4A3AA7", "A3": "#0E7A86", "iMF": "#A8434F", "ref": "#8A94A3"}

    def bars(a, labels, trip, colors, ylabel):
        vals = [t[0] if isinstance(t, list) else t for t in trip]
        a.bar(range(len(vals)), vals, color=colors, width=0.6)
        for k, t in enumerate(trip):                          # 95% CI only where one exists
            if isinstance(t, list):
                a.errorbar(k, t[0], yerr=[[t[0] - t[1]], [t[2] - t[0]]], color="k", capsize=4, lw=1)
        a.set_xticks(range(len(vals)), labels, fontsize=8)
        a.set_ylabel(ylabel, fontsize=9)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    a = ax[0, 0]
    a.axis("off")
    steps = ("PPG", "fixed rhythm events\n(RD1 detector + timing head)",
             "beat morphology model\n(template / deterministic / stochastic)", "fixed-time placement\n(overlap-add, no warp)", "ECG")
    for k, txt in enumerate(steps):
        y = 0.92 - k * 0.21
        a.text(0.5, y, txt, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#E3F1F3", ec="#0E7A86"))
        if k < len(steps) - 1:
            a.annotate("", xy=(0.5, y - 0.13), xytext=(0.5, y - 0.07), arrowprops=dict(arrowstyle="->"))
    a.set_title("A  BF0 architecture", loc="left", fontsize=10)
    pp, rr = det["G1"]["placed_positions"], det["G1"]["A3_render"]
    bars(ax[0, 1], ["placed positions", "stochastic render (s=0)"], [pp["f1"], rr["f1"]], [C["ref"], C["A3"]], "R-peak F1 (±50 ms)")
    ax[0, 1].set_title(f"B1  G1 {ok('G1_rhythm_preservation')}: ΔF1 {det['G1']['f1_diff'][0]:+.3f} "
                       f"[{det['G1']['f1_diff'][1]:+.3f}, {det['G1']['f1_diff'][2]:+.3f}]", loc="left", fontsize=10)
    bars(ax[0, 2], ["placed positions", "stochastic render (s=0)"], [pp["rr_mae_ms"], rr["rr_mae_ms"]], [C["ref"], C["A3"]], "RR-MAE (ms)")
    ax[0, 2].set_title(f"B2  ΔRR-MAE {det['G1']['rr_mae_diff_ms'][0]:+.2f} ms "
                       f"[{det['G1']['rr_mae_diff_ms'][1]:+.2f}, {det['G1']['rr_mae_diff_ms'][2]:+.2f}]", loc="left", fontsize=10)
    a = ax[0, 3]
    vals = [det["G3"]["FD"]["A1"], det["G3"]["FD"]["A2"], det["G3"]["FD"]["A3"], orc["O1"]["fd"], orc["O2"]["fd"], orc["O3"]["fd"]]
    bars(a, ["template", "determ.", "stoch. s=0", "GT-R\ntemplate", "GT-R\ndeterm.", "GT-R\nstoch."], vals,
         [C["A1"], C["A2"], C["A3"]] * 2, "FD (lower = more realistic)")
    for k in range(3, 6):
        a.patches[k].set_alpha(0.45)
    a.set_title(f"C  G3 {ok('G3_distributional_gain')}: stoch−templ {det['G3']['A3-A1'][0]:+.1f} "
                f"[{det['G3']['A3-A1'][1]:+.1f}, {det['G3']['A3-A1'][2]:+.1f}]\n"
                f"stoch−determ {det['G3']['A3-A2'][0]:+.1f} [{det['G3']['A3-A2'][1]:+.1f}, {det['G3']['A3-A2'][2]:+.1f}]",
                loc="left", fontsize=10)
    bars(ax[1, 0], ["iMF 16 samples", "BF0 16 realizations"], [det["G4a"]["iMF_median_sd_ms"], det["G4a"]["A3_median_sd_ms"]],
         [C["iMF"], C["A3"]], "R-time SD across draws (ms)")
    ax[1, 0].axhline(det["G4a"]["threshold_ms"], color="k", ls="--", lw=0.8)
    ax[1, 0].set_title(f"D1  G4a {ok('G4a_timing_invariance')} (threshold 1 sample = 7.8 ms)", loc="left", fontsize=10)
    sdv = det["G4b"]["seed_to_seed_diversity"]
    bars(ax[1, 1], ["iMF 16 samples\n(at reference R)", "BF0 16 realizations\n(at placed R)"],
         [sdv["iMF_at_reference_R"], sdv["A3_at_placed_positions"]], [C["iMF"], C["A3"]], "seed-to-seed beat RMS")
    ax[1, 1].set_title("D2  morphology variation across draws (secondary)", loc="left", fontsize=10)
    D = det["G4b"]["D"]
    bars(ax[1, 2], ["template", "determ.", "stoch. s=0", "real ECG"], [D["A1"], D["A2"], D["A3"], det["G4b"]["D_real"]],
         [C["A1"], C["A2"], C["A3"], C["ref"]], "within-window beat diversity D")
    ax[1, 2].set_title(f"D3  G4b {ok('G4b_non_collapse')}: D(stoch)/D(real) {det['G4b']['D(A3)/D(real)']:.2f} "
                       f"(practical target 0.25)", loc="left", fontsize=10)
    bars(ax[1, 3], ["conditioned", "S-PPG", "S-RR"], [shf["A3_conditioned"]["fd"], shf["S_PPG"]["fd"], shf["S_RR"]["fd"]],
         [C["A3"], C["ref"], C["ref"]], "FD")
    ax[1, 3].set_title("E  condition shuffles (diagnostic, not a gate)", loc="left", fontsize=10)
    fig.suptitle(f"BF0 — {g['verdict']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(ART / "figure.png", dpi=150)


STAGES = {"manifest": stage_manifest, "audit": stage_audit, "train_timing": stage_train_timing,
          "train_beat": stage_train_beat, "render": stage_render, "evaluate": stage_evaluate, "atlas": stage_atlas,
          "compute": stage_compute, "figure": stage_figure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    args = ap.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with ProcessPoolExecutor(10) as ex:
        STAGES[args.stage](ex, dev)


if __name__ == "__main__":
    main()
