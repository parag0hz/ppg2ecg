"""LAT1 — single-window inference latency on CPU and GPU (no training, no metric)."""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cd1_evaluate as CDE  # noqa: E402
import u2_evaluate as U2  # noqa: E402
from ppg2ecg.models import build_penguin_backbone  # noqa: E402
from ppg2ecg.models.imf_dit import ImfDiT1d  # noqa: E402

N_REP, THREADS = 50, 4          # 4 CPU threads ~ a wearable-class core budget
RUNS = [("PENGUIN-50", "C", 50), ("PENGUIN-1", "C", 1), ("iMF-1", "I", 1), ("CD-1", "D", 1), ("smallDiT-1", "S", 1)]
CK = {"C": "outputs/v1_vitaldb_armC_seed42", "I": "outputs/v1_vitaldb_armI_seed42",
      "D": "outputs/cd1_vitaldb_armD_seed42", "S": "outputs/vm1_S_seed42"}
OUT = ROOT / "artifacts/lat1_latency"


def build(arm, dev):
    p = ROOT / CK[arm] / "checkpoint_last.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if arm == "S":
        net = ImfDiT1d(**ck["model_cfg"]); net.load_state_dict(ck["state_dict"]); return net.to(dev).eval(), "S"
    if arm == "D":
        net = build_penguin_backbone(**ck["model_cfg"]); net.load_state_dict(ck["state_dict"]); return net.to(dev).eval(), "D"
    return U2.build(p, dev)[0], arm


@torch.no_grad()
def once(net, kind, x, dev, nfe):
    if kind == "S":
        import vm1_evaluate as VM
        cfg = tuple(float(v) for v in np.load(ROOT / "outputs/vm1_eval/arm_S.npz")["cfg"])
        return VM.generate(net, x, 0, nfe, cfg, dev)[0]
    if kind == "D":
        return CDE.sample(net, x, 0, nfe, dev)[0]
    e0 = torch.randn(len(x), 1, x.shape[1], generator=torch.Generator().manual_seed(0))
    return U2.generate(net, x, e0, nfe, dev, kind)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    x = np.load(ROOT / "data/processed/v1_vitaldb" / (json.loads((ROOT / "data/manifests/split_v1_vitaldb_seed42.json").read_text())["splits"][0]["test"][0] + ".npz"))["x"][:1].astype(np.float32)
    res = {}
    for dev_name in ("cpu", "cuda"):
        dev = torch.device(dev_name)
        for label, arm, nfe in RUNS:
            net, kind = build(arm, dev)
            for _ in range(3):
                once(net, kind, x, dev, nfe)
            ts = []
            for _ in range(N_REP):
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter(); once(net, kind, x, dev, nfe)
                if dev_name == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            res[f"{dev_name}|{label}"] = {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)),
                                          "nfe": nfe, "threads": THREADS if dev_name == "cpu" else None}
            print(f"[lat1] {dev_name:4s} {label:11s} NFE {nfe:2d}  median {np.median(ts):8.2f} ms", flush=True)
            del net
            if dev_name == "cuda":
                torch.cuda.empty_cache()
    for k in list(res):
        if k.endswith("iMF-1") or k.endswith("CD-1"):
            res[k + " x16 (consensus)"] = {"median_ms": res[k]["median_ms"] * 16, "nfe": res[k]["nfe"] * 16}
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
