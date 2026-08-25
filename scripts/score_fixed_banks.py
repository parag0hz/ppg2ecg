"""Score an existing checkpoint on the fixed validation (t, z) banks (post-hoc val_cfm_fixed), e.g. the A0 checkpoint,
so that A0 and A0-b are compared on the same deterministic criterion (A0-b prereg §5 rule 1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import ppg2ecg.utils.mkl_warmup  # noqa: F401  (MKL warm-up must precede `import torch`, docs/ENVIRONMENT.md)
import torch

from ppg2ecg.data.splits import read_manifest
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.training.valbank import bank_hash, fixed_cfm_loss, make_banks

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--manifest", default="data/manifests/split_p0_holdout_seed42.json")
    ap.add_argument("--n-val-banks", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=1000)
    args = ap.parse_args()
    split = read_manifest(ROOT / args.manifest)[0]
    d = np.load(ROOT / args.processed / f"{split['val'][0]}.npz")
    x, y = torch.from_numpy(d["x"]).cuda(), torch.from_numpy(d["y"]).cuda()
    banks = make_banks(len(x), x.shape[1], args.n_val_banks, args.bank_seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_penguin_backbone(**ck["model_cfg"]).cuda().eval()
    model.load_state_dict(ck["state_dict"])
    mean, per = fixed_cfm_loss(model, x, y, banks)
    res = {"checkpoint": args.checkpoint, "epoch": ck.get("epoch"), "val_subject": split["val"], "n_val": int(len(x)), "bank_hash": bank_hash(banks), "val_cfm_fixed": mean, "per_bank": per}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
