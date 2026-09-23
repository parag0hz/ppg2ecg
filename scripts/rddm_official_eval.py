"""R0 — run RDDM's OFFICIAL evaluator (`external/RDDM/std_eval.py::eval_diffusion`, commit 7d53488, unmodified) on the
released checkpoint and on our reconstruction of its inputs (`scripts/rddm_build_inputs.py`).

`std_eval.get_datasets` is called by the official code without a data path, so it reads the hard-coded relative path
"../../ingenuity_NAS/21ds94_nas/21ds94_mount/AAAI24/datasets/". This runner creates that relative layout as a symlink to
`outputs/rddm_run/datasets` and changes the working directory so the path resolves — no upstream line is edited.
The checkpoint directory is passed through `eval_diffusion`'s own `PATH` argument.

Order of calls (the official module seeds once at import with seed 31, so the RNG stream depends on this order):
window 4 s for WESAD, CAPNO, DALIA, BIDMC (the paper's Table 2: RMSE, FD), then window 8 s for WESAD, DALIA (Table 3: HR),
then window 8 s for CAPNO, BIDMC (not in the paper; reported as extra).

Run: PYTHONPATH=external/RDDM:outputs/rddm_env/pydeps .venv/bin/python scripts/rddm_official_eval.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/rddm_run"
CKPT = ROOT / "data/pretrained/rddm"
OUT = ROOT / "artifacts/rddm_r0"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nas = RUN / "cwd/ingenuity_NAS/21ds94_nas/21ds94_mount/AAAI24"
    nas.mkdir(parents=True, exist_ok=True)
    link = nas / "datasets"
    if not link.exists():
        link.symlink_to(RUN / "datasets", target_is_directory=True)
    work = RUN / "cwd/a/b"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    assert Path("../../ingenuity_NAS/21ds94_nas/21ds94_mount/AAAI24/datasets/WESAD/ppg_test_4sec.npy").exists()
    sys.path.insert(0, str(ROOT / "external/RDDM"))
    import std_eval  # noqa: E402  (official; seeds 31 at import)

    plan = [(4, d) for d in ("WESAD", "CAPNO", "DALIA", "BIDMC")] + [(8, "WESAD"), (8, "DALIA"), (8, "CAPNO"), (8, "BIDMC")]
    res = {"evaluator": "external/RDDM/std_eval.py::eval_diffusion @ 7d5348843c3985c211a23ae5105a2d9497d5156a (unmodified)",
           "checkpoint_dir": str(CKPT), "nT": 10, "batch_size": 512, "order": [f"{w}s:{d}" for w, d in plan], "results": {}}
    for w, d in plan:
        t0 = time.time()
        m = std_eval.eval_diffusion(window_size=w, EVAL_DATASETS=[d], nT=10, batch_size=512, PATH=str(CKPT) + "/", device="cuda")
        m = {k: float(v) for k, v in m.items()}
        m["seconds"] = round(time.time() - t0, 1)
        res["results"][f"{d}|{w}s"] = m
        print(f"[rddm-official] {d} {w}s  RMSE {m['RMSE_score']:.4f}  FD {m['FD']:.3f}  MAE_HR {m['MAE_HR_ECG']:.3f}  ({m['seconds']} s)", flush=True)
        (OUT / "official_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
