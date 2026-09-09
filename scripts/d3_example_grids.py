"""Per-dataset n x n example grids for D3 (one PNG per dataset).

Uses the SAME checkpoint that produced each D3 table row, and inference only — no metric is computed here.
Window selection is DETERMINISTIC and stated in every caption: the first n^2 windows of the first evaluated
subject in natural order, taken in stored order. Nothing is inspected before selection and nothing is replaced.

WildPPG uses the A4 arm on its VALIDATION cohort (an0, k2s): the A4 split's test subjects are kjd/ssx, which
the standing rule forbids loading, so the A4 report evaluated on val and this figure follows it.

Run: .venv/bin/python scripts/d3_example_grids.py [--n 4]
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.flow.imeanflow import MeanFlowS5
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/d3_bench/figures"
FS, BATCH, SRC_SEED = 128, 64, 0

# key: (title, processed dir, split manifest, which split, checkpoint, task, seg_len, NFE, target unit)
SPEC = {
    "dalia":      ("PPG-DaLiA", "v0_8s", "split_a3_testS1_valS11.json", "test",
                   "outputs/a3_imeanflow_ppgdalia_testS1_seed42/checkpoint_best.pt", "ECG", 8, 4, "norm. amplitude"),
    "wildppg":    ("WildPPG", "wildppg_8s", "split_a4_wildppg_seed42.json", "val",
                   "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt", "ECG", 8, 2, "norm. amplitude"),
    "bidmc_resp": ("BIDMC (respiration)", "bidmc_resp_4s", "split_d3_bidmc_resp_seed42.json", "test",
                   "outputs/d3_bidmc_resp_seed42/checkpoint_best.pt", "Resp", 4, 2, "norm. amplitude"),
    "wesad_resp": ("WESAD (respiration)", "wesad_resp_4s", "split_d3_wesad_resp_seed42.json", "test",
                   "outputs/d3_wesad_resp_seed42/checkpoint_best.pt", "Resp", 4, 50, "norm. amplitude"),
    "uci_bp":     ("UCI-BP (arterial pressure)", "uci_bp_8s", "split_d3_uci_bp_seed42.json", "test",
                   "outputs/d3_uci_bp_seed42/checkpoint_best.pt", "ABP", 8, 50, "mmHg"),
    "mimicbp":    ("MIMIC-BP (arterial pressure)", "mimicbp_8s", "split_a7_mimicbp_official.json", "test",
                   "outputs/a7_imeanflow_mimicbp_seed42/checkpoint_best.pt", "ABP", 8, 4, "mmHg"),
}
FORBIDDEN = ("kjd", "ssx")


def load_subject(proc: str, manifest: str, which: str, n_take: int):
    split = json.loads((ROOT / "data/manifests" / manifest).read_text())["splits"][0]
    subs = [s for s in split[which] if s not in FORBIDDEN]
    assert subs, f"{manifest}:{which} has no loadable subject"
    s = sorted(subs)[0]
    d = np.load(ROOT / "data/processed" / proc / f"{s}.npz")
    k = min(n_take, len(d["x"]))
    # three corpus generations ship three different index keys; v0_8s ships none, so fall back to position
    for key in ("window_index", "segment_idx"):
        if key in d.files:
            idx = np.asarray(d[key])
            break
    else:
        idx = np.arange(len(d["x"]))
    return s, d["x"][:k].astype(np.float32), d["y"][:k].astype(np.float64), idx[:k], subs


@torch.no_grad()
def predict(ckpt: Path, X: np.ndarray, nfe: int, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("imf_cfg", {}) or {}
    net = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg.get("cond_mode", "h_only"),
                     h_scale=cfg.get("h_scale", 1.0)).to(dev).eval()
    net.load_state_dict(ck["state_dict"])
    net.requires_grad_(False)
    e = torch.randn(len(X), 1, X.shape[1], generator=torch.Generator().manual_seed(SRC_SEED))
    out = []
    for i in range(0, len(X), BATCH):
        pp = torch.from_numpy(X[i:i + BATCH]).to(dev).unsqueeze(1)
        z, k = ER.sample_meanflow_schedule(net, pp, e[i:i + BATCH].to(dev), ER.UNIFORM[nfe])
        assert k == nfe
        out.append(z.squeeze(1).float().cpu().numpy())
    del net
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float64)


def grid(key: str, n: int, dev) -> Path:
    title, proc, manifest, which, ckpt, task, seg, nfe, unit = SPEC[key]
    s, X, Y, widx, subs = load_subject(proc, manifest, which, n * n)
    P = predict(ROOT / ckpt, X, nfe, dev)
    t = np.arange(X.shape[1]) / FS
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 2.3 * n), squeeze=False)
    for i in range(n * n):
        a = axes[i // n][i % n]
        if i >= len(X):
            a.axis("off")
            continue
        a.plot(t, Y[i], color="black", lw=0.9, label="target" if i == 0 else None)
        a.plot(t, P[i], color="#B03A2E", lw=0.8, alpha=0.9, label="generated" if i == 0 else None)
        a2 = a.twinx()
        a2.plot(t, X[i], color="#1F618D", lw=0.5, alpha=0.45)
        a2.set_yticks([])
        a.set_title(f"{s} · w{int(widx[i])}", fontsize=8)
        a.tick_params(labelsize=7)
        if i % n == 0:
            a.set_ylabel(unit, fontsize=8)
        if i // n == n - 1:
            a.set_xlabel("time (s)", fontsize=8)
        if i == 0:
            a.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"{title} — target vs generated, iMeanFlow NFE {nfe}", fontsize=13)
    fig.text(0.5, 0.008,
             f"Black = ground-truth {task}; red = generated; thin blue (right axis, unlabelled) = the conditioning PPG. "
             f"{seg} s windows at {FS} Hz. WINDOW SELECTION IS DETERMINISTIC AND WAS NOT INSPECTED BEFORE SELECTION: "
             f"the first {n*n} stored windows of subject {s}, the first of the {which} split "
             f"({len(subs)} subject{'s' if len(subs) != 1 else ''}) in natural order. "
             + ("ABP is left in RAW mmHg by PENGUIN's config, so the vertical scale is absolute pressure. "
                if task == "ABP" else "")
             + f"Checkpoint: {ckpt}.",
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    p = OUT / f"d3_examples_{key}.png"
    fig.savefig(p, dpi=200)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    made = []
    for k in ([args.only] if args.only else list(SPEC)):
        try:
            p = grid(k, args.n, dev)
            made.append(str(p.relative_to(ROOT)))
            print(f"[grid] {p.relative_to(ROOT)}", flush=True)
        except Exception as e:                                    # a missing corpus must not kill the rest
            print(f"[grid] {k}: SKIPPED — {type(e).__name__}: {e}", flush=True)
    (OUT / "example_grids_manifest.json").write_text(json.dumps(
        {"n": args.n, "source_seed": SRC_SEED, "files": made,
         "selection_rule": "first n^2 stored windows of the first subject of the named split, natural order; "
                           "not inspected before selection",
         "spec": {k: dict(zip(("title", "processed", "manifest", "split", "checkpoint", "task", "seg_s", "nfe", "unit"), v))
                  for k, v in SPEC.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
