"""RDDM driver: official loader + official data pipeline + official sampler, with fixed iteration order and seeded
noise, saving per-window samples so every metric can be recomputed later. No upstream line is edited.

- Model: `diffusion.load_pretrained_DPM(PATH, nT, type="RDDM")` (official). The sampler is the official
  `RDDM.forward(mode="sample")`; the two condition encoders run once per window and their output is reused by every
  sample of that window (their input is the same PPG; this is how the official code calls them too, once per batch).
- Data: `data.get_datasets(DATA_PATH, [corpus], window_size)` (official, including its per-window min-max, `ppg_clean` and
  `ecg_clean(pantompkins1985)`), iterated with shuffle=False.
- Noise: the official sampler draws from torch's global CPU RNG; before every batch the RNG is seeded with
  `seed = 1_000_003 * draw + batch_index` (batch size fixed at 512), so draw k of window i is reproducible and draws
  are independent across k.
- NFE: counted with forward hooks on eps_model / region_model (no code change) and asserted = 2 * nT per sample.

Run (R0, K = 1):  PYTHONPATH=external/RDDM:outputs/rddm_env/pydeps .venv/bin/python scripts/rddm_driver.py r0
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/RDDM"))
sys.path.insert(0, str(ROOT / "scripts"))
import data as RD  # noqa: E402  (official)
import metrics as RM  # noqa: E402  (official)
from diffusion import load_pretrained_DPM  # noqa: E402  (official)
import v1_evaluate as V  # noqa: E402
from ppg2ecg.evaluation import paper_metrics as PMX  # noqa: E402

RUN, CKPT = ROOT / "outputs/rddm_run", ROOT / "data/pretrained/rddm"
RAW, OUT = ROOT / "outputs/rddm_raw", ROOT / "artifacts/rddm_r0"
FS, BS = 128, 512
CORPORA = ("WESAD", "CAPNO", "DALIA", "BIDMC")


class Model:
    def __init__(self, nT=10, dev="cuda"):
        self.nT, self.dev = nT, dev
        self.dpm, self.c1, self.c2 = load_pretrained_DPM(PATH=str(CKPT) + "/", nT=nT, type="RDDM", device=dev)
        self.calls = {"eps": 0, "region": 0}
        self.dpm.eps_model.register_forward_hook(lambda *a: self.calls.__setitem__("eps", self.calls["eps"] + 1))
        self.dpm.region_model.register_forward_hook(lambda *a: self.calls.__setitem__("region", self.calls["region"] + 1))

    @torch.no_grad()
    def sample(self, ppg_windows, draw):
        """ppg_windows: float tensor [n, 1, 512] (already cleaned by the official Dataset). Returns [n, 512] numpy."""
        out = []
        for b, i in enumerate(range(0, len(ppg_windows), BS)):
            x = ppg_windows[i:i + BS].float().to(self.dev)
            co1, co2 = self.c1(x), self.c2(x)
            before = dict(self.calls)
            torch.manual_seed(1_000_003 * draw + b)
            y = self.dpm(cond1=co1, cond2=co2, mode="sample", window_size=512)
            assert self.calls["eps"] - before["eps"] == self.nT and self.calls["region"] - before["region"] == self.nT
            out.append(y[:, 0].float().cpu().numpy())
        return np.concatenate(out)


def load_corpus(name, window):
    _, ds = RD.get_datasets(DATA_PATH=str(RUN / "datasets") + "/", datasets=[name], window_size=window)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, num_workers=16)
    E, P = [], []
    for ecg, ppg, _roi in dl:
        E.append(ecg.float()); P.append(ppg.float())
    meta = np.load(RUN / "datasets" / name / f"meta_{window}sec.npz")
    return torch.cat(E)[:, 0].numpy(), torch.cat(P), meta["subject"], meta["window_start_s"]


def gen_8s(model, P8, draw):
    """Official 8 s protocol (std_eval): split the cleaned 8 s PPG into two 4 s halves, generate each, concatenate."""
    halves = [model.sample(P8[:, :, j * 512:(j + 1) * 512], draw * 2 + j) for j in range(2)]
    return np.concatenate(halves, axis=1)


def their_metrics(real, fake, window):
    rmse = float(np.sqrt(np.mean((fake - real) ** 2)))
    fds = [RM.calculate_FD(torch.from_numpy(real[i:i + BS].reshape(-1, 512)), torch.from_numpy(fake[i:i + BS].reshape(-1, 512)))
           for i in range(0, len(real), BS) if len(real[i:i + BS]) > 1]
    return {"RMSE": rmse, "FD_batch512_fixed_order": float(np.mean(fds)), "MAE_HR_hamilton": float(RM.MAE_hr(real, fake, window_size=window))}


def our_metrics(real, fake, subj):
    tab = PMX.paper_metric_table(fake.astype(np.float64), real.astype(np.float64), fs=FS, with_quadratic=False)
    ci = lambda v: V.cluster_ci(v, subj)  # noqa: E731
    return {"HR_neurokit": ci(tab["hr_abs_err"]), "Rpeak_F1_50ms": ci(tab["rpeak_f1_50ms"]), "RR_MAE_ms": ci(tab["rr_mae_ms"]),
            "MAE": ci(np.abs(fake - real).mean(1)), "RMSE_window": ci(np.sqrt(((fake - real) ** 2).mean(1))),
            "n_windows": int(len(real)), "n_subjects": int(len(np.unique(subj)))}


def latency(model, x1):
    res = {}
    for dev in ("cuda", "cpu"):
        m = model if dev == "cuda" else None
        if dev == "cpu":
            torch.set_num_threads(4)
            m = Model(nT=model.nT, dev="cpu")
        xx = x1.to(dev)
        with torch.no_grad():
            for _ in range(2):
                m.dpm(cond1=m.c1(xx), cond2=m.c2(xx), mode="sample", window_size=512)
            ts = []
            for _ in range(20 if dev == "cuda" else 10):
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                m.dpm(cond1=m.c1(xx), cond2=m.c2(xx), mode="sample", window_size=512)
                if dev == "cuda": torch.cuda.synchronize()
                ts.append(1000 * (time.perf_counter() - t0))
        res[dev] = {"median_ms_per_window_batch1": float(np.median(ts)), "threads": 4 if dev == "cpu" else None}
    return res


def r0():
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True)
    model = Model(nT=10)
    res = {"sampler": "official RDDM ancestral sampler, nT = 10", "nfe_per_sample": 20, "condition_encoder_passes_per_window": 2,
           "params": {"eps_model": 45828129, "region_model": 45828129, "cond1": 26926016, "cond2": 26926016, "total": 145508290},
           "draw": 0, "corpora": {}}
    for name in CORPORA:
        real4, P4, subj4, t4 = load_corpus(name, 4)
        t0 = time.time(); fake4 = model.sample(P4, draw=0); gen_s = time.time() - t0
        np.savez_compressed(RAW / f"r0_{name}_4s.npz", fake=fake4.astype(np.float16), real=real4.astype(np.float16), subject=subj4, t0=t4)
        real8, P8, subj8, t8 = load_corpus(name, 8)
        fake8 = gen_8s(model, P8, draw=0)
        np.savez_compressed(RAW / f"r0_{name}_8s.npz", fake=fake8.astype(np.float16), real=real8.astype(np.float16), subject=subj8, t0=t8)
        r = {"4s": {**their_metrics(real4, fake4, 4), **our_metrics(real4, fake4, subj4)},
             "8s": {**their_metrics(real8, fake8, 8), **our_metrics(real8, fake8, subj8)},
             "gpu_ms_per_window_batch512": 1000 * gen_s / len(P4)}
        res["corpora"][name] = r
        print(f"[rddm-r0] {name}: 4s RMSE {r['4s']['RMSE']:.4f} FD {r['4s']['FD_batch512_fixed_order']:.3f} | 8s HR(hamilton) {r['8s']['MAE_HR_hamilton']:.3f} "
              f"| ours 4s HR {r['4s']['HR_neurokit'][0]:.3f} F1 {r['4s']['Rpeak_F1_50ms'][0]:.3f} RR {r['4s']['RR_MAE_ms'][0]:.1f}", flush=True)
        (OUT / "driver_r0.json").write_text(json.dumps(res, indent=1, default=float))
    res["latency"] = latency(model, P4[:1])
    (OUT / "driver_r0.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["latency"], indent=1))


def invalid_schedule_diagnostic():
    """Boundary diagnostic, NOT a candidate allocation: the official loader's own nT argument with the released weights
    (trained on the 10-step schedule). Shows what reduced-step sampling does; reported as such."""
    real4, P4, subj4, _ = load_corpus("CAPNO", 4)
    res = {}
    for nT in (10, 5, 2):
        m = Model(nT=nT)
        fake = m.sample(P4, draw=0)
        res[f"nT={nT}"] = {**their_metrics(real4, fake, 4), **our_metrics(real4, fake, subj4), "nfe_per_sample": 2 * nT}
        print(f"[rddm-diag] CAPNO nT={nT}: RMSE {res[f'nT={nT}']['RMSE']:.4f} FD {res[f'nT={nT}']['FD_batch512_fixed_order']:.2f} HR {res[f'nT={nT}']['HR_neurokit'][0]:.2f} F1 {res[f'nT={nT}']['Rpeak_F1_50ms'][0]:.3f}", flush=True)
        del m; torch.cuda.empty_cache()
    (OUT / "invalid_schedule_diagnostic.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["r0", "diag"])
    a = ap.parse_args()
    {"r0": r0, "diag": invalid_schedule_diagnostic}[a.mode]()
