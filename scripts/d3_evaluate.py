"""D3 evaluation (docs/D3_PENGUIN_SIX_DATASET_PREREGISTRATION.md §5-§8).

Frozen-checkpoint forward inference only. Computes PENGUIN's own task metric per dataset:
  ABP  (uci_bp, mimicbp)          SBP error / DBP error, mmHg, 8 s windows
  Resp (bidmc_resp, wesad_resp)   RR error, breaths/min, 60 s = 15 consecutive 4 s windows

iMF is reported at NFE 1, 2 and 4 -- ALL of them; no NFE may be selected after seeing results (prereg §6).
Uncertainty: subject-clustered bootstrap, 2000 replicates, seed 20260904.

Run: .venv/bin/python scripts/d3_evaluate.py --corpus uci_bp
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import penguin_metrics as PM
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.flow.samplers import heun_sample, nfe_of
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/d3_penguin_six_dataset"
BOOT_N, BOOT_SEED, BATCH = 2000, 20260904, 64

CORPORA = {
    # key: (processed dir, split manifest, task, segment_len, {arm: checkpoint})
    "uci_bp": ("uci_bp_8s", "split_d3_uci_bp_seed42.json", "ABP", 8,
               {"iMF": "outputs/d3_uci_bp_seed42/checkpoint_best.pt"}),
    "mimicbp": ("mimicbp_8s", "split_a7_mimicbp_official.json", "ABP", 8,
                {"iMF": "outputs/a7_imeanflow_mimicbp_seed42/checkpoint_best.pt",
                 "OT-CFM": "outputs/a7_otcfm_mimicbp_seed42/checkpoint_best.pt"}),
    "bidmc_resp": ("bidmc_resp_4s", "split_d3_bidmc_resp_seed42.json", "Resp", 4,
                   {"iMF": "outputs/d3_bidmc_resp_seed42/checkpoint_best.pt"}),
    "wesad_resp": ("wesad_resp_4s", "split_d3_wesad_resp_seed42.json", "Resp", 4,
                   {"iMF": "outputs/d3_wesad_resp_seed42/checkpoint_best.pt"}),
}
NFES = {"iMF": (1, 2, 4), "OT-CFM": (50,)}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_test(proc: str, manifest: str):
    split = json.loads((ROOT / "data/manifests" / manifest).read_text())["splits"][0]
    assert not ({"kjd", "ssx"} & set(split["test"])), "forbidden subject in the test split"
    X, Y, S, W = [], [], [], []
    for s in split["test"]:
        d = np.load(ROOT / "data/processed" / proc / f"{s}.npz")
        X.append(d["x"].astype(np.float32))
        Y.append(d["y"].astype(np.float64))
        S.append(np.full(len(d["x"]), s))
        # mimicbp_8s predates the D1 corpus contract and stores `segment_idx` instead of `window_index`
        idx = d["window_index"] if "window_index" in d.files else d["segment_idx"]
        W.append(np.asarray(idx).astype(np.int32))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S), np.concatenate(W), split


def build(ckpt: Path, dev):
    """iMF checkpoints store the MeanFlowS5 wrapper (keys prefixed `backbone.`); OT-CFM arms store the BARE
    backbone and are integrated with the upstream Heun sampler instead of the MeanFlow schedule."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    bare = not any(k.startswith("backbone.") for k in ck["state_dict"])
    backbone = build_penguin_backbone(**ck["model_cfg"]).to(dev).eval()
    if bare:
        backbone.load_state_dict(ck["state_dict"])
        backbone.requires_grad_(False)
        return backbone, ck, "otcfm"
    cfg = ck.get("imf_cfg", {}) or {}
    net = MeanFlowS5(backbone, cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    return net, ck, "imf"


@torch.no_grad()
def generate(net, X, e0, nfe, dev, kind="imf"):
    """iMF: the uniform MeanFlow schedule. OT-CFM: upstream Heun, where NFE = 2*steps (samplers.nfe_of)."""
    outs, got = [], set()
    t0 = time.perf_counter()
    steps = nfe // 2 if kind == "otcfm" else None
    if kind == "otcfm":
        assert nfe_of("heun", steps) == nfe, f"Heun cannot realise NFE {nfe}"
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        if kind == "otcfm":
            v = lambda x, t, _p=pp: net.forward_step(x, _p, t)  # noqa: E731
            z, k = heun_sample(v, e0[i:i + BATCH].to(dev), steps)
        else:
            z, k = ER.sample_meanflow_schedule(net, pp, e0[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        got.add(int(k))
        outs.append(z.squeeze(1).float().cpu().numpy())
    assert got == {nfe}, (nfe, got)
    return np.concatenate(outs).astype(np.float64), time.perf_counter() - t0


def cluster_bootstrap(vals, subj, n=BOOT_N, seed=BOOT_SEED):
    """Subject-macro mean with a subject-clustered bootstrap; subjects carry equal weight."""
    subs = np.unique(subj)
    per = np.array([np.nanmean(vals[subj == s]) for s in subs])
    if not np.isfinite(per).any():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = np.array([np.nanmean(per[rng.integers(0, len(subs), len(subs))]) for _ in range(n)])
    return float(np.nanmean(per)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--source-seed", type=int, default=0)
    ap.add_argument("--nfes", default=None,
                    help="override the frozen per-arm NFE list, e.g. 50. DEVIATION D3-2: the preregistration froze "
                         "iMF at 1/2/4 and a 50-NFE column supplied by OT-CFM, but no OT-CFM arm exists for the "
                         "three new corpora, so the 50-NFE budget match is completed with iMF itself. ALL of "
                         "1/2/4/50 are reported; nothing is selected.")
    args = ap.parse_args()
    proc, manifest, task, seg, arms = CORPORA[args.corpus]
    nfe_override = tuple(int(x) for x in args.nfes.split(",")) if args.nfes else None
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, Y, SUB, WIDX, split = load_test(proc, manifest)
    T = X.shape[1]
    print(f"[d3] {args.corpus} ({task}, {seg}s): {len(X)} test windows, {len(split['test'])} subjects, T={T}", flush=True)

    rows, manifest_rows = [], []
    e0 = torch.randn(len(X), 1, T, generator=torch.Generator().manual_seed(args.source_seed))
    for arm, ck_rel in arms.items():
        p = ROOT / ck_rel
        if not p.exists():
            print(f"[d3] {arm}: checkpoint absent, skipped")
            continue
        net, ck, kind = build(p, dev)
        manifest_rows.append({"arm": arm, "path": ck_rel, "sha256": sha256(p), "epoch": int(ck.get("epoch", -1))})
        for nfe in (nfe_override if nfe_override else NFES[arm]):
            pred, dt = generate(net, X, e0, nfe, dev, kind)
            if task == "ABP":
                # SBP/DBP are per 8 s window, in the label's own mmHg (prereg §4: ABP is never normalised)
                per = {"SBP": PM.sbp_error(pred, Y), "DBP": PM.dbp_error(pred, Y)}
                subj_of = SUB
            else:
                # RR needs 60 s = 15 consecutive 4 s windows, exactly train.py:41-47
                k = PM.segments_per_metric_window(PM.RESP_WINDOW_S, seg)
                per, sub60 = {}, []
                p60, y60 = [], []
                for s in np.unique(SUB):
                    m = SUB == s
                    pp, yy = PM.concat_windows(pred[m], k), PM.concat_windows(Y[m], k)
                    if len(pp):
                        p60.append(pp)
                        y60.append(yy)
                        sub60 += [s] * len(pp)
                p60, y60 = np.concatenate(p60), np.concatenate(y60)
                per = {"RR": PM.resp_rate_error(p60, y60)}
                subj_of = np.asarray(sub60)
            for name, v in per.items():
                mean, lo, hi = cluster_bootstrap(v, subj_of)
                rows.append({"corpus": args.corpus, "task": task, "arm": arm, "nfe": nfe, "metric": name,
                             "subject_macro_mean": mean, "ci_lo": lo, "ci_hi": hi,
                             "pooled_mean": float(np.nanmean(v)), "n_units": int(np.isfinite(v).sum()),
                             "n_subjects": int(len(np.unique(subj_of))), "gen_seconds": dt})
                print(f"[d3]   {arm} NFE{nfe} {name}: {mean:.4f} [{lo:.4f}, {hi:.4f}]  n={int(np.isfinite(v).sum())}", flush=True)
        del net
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    # merge with any earlier rows for this corpus, keyed by (arm, nfe, metric); a rerun replaces its own rows
    f = OUT / f"metrics_{args.corpus}.csv"
    prev = {(r["arm"], int(r["nfe"]), r["metric"]): r for r in csv.DictReader(open(f))} if f.exists() else {}
    prev.update({(r["arm"], int(r["nfe"]), r["metric"]): r for r in rows})
    merged = sorted(prev.values(), key=lambda r: (r["arm"], int(r["nfe"]), r["metric"]))
    with open(f, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(merged)
    (OUT / f"meta_{args.corpus}.json").write_text(json.dumps(
        {"corpus": args.corpus, "task": task, "segment_len_s": seg, "processed": proc, "manifest": manifest,
         "test_subjects": split["test"], "n_test_windows": int(len(X)), "source_seed": args.source_seed,
         "bootstrap": {"n": BOOT_N, "seed": BOOT_SEED, "rule": "subject-clustered, equal subject weight"},
         "resp_window_s": PM.RESP_WINDOW_S if task == "Resp" else None,
         "checkpoints": manifest_rows}, indent=1))
    print(f"[d3] wrote {f.name} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
