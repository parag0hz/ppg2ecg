"""Official PPGFlowECG inference path, driven from our side (external/PPGFlowECG @ 56b2cd2 is never edited).

Loading: the official `engine.solver.Trainer` is constructed with the official config
(`config/latent_rectified_flow.yaml`), with only two paths overridden through the config dict: `solver.results_folder`
(a directory whose `checkpoints/checkpoint-10.pt` is a symlink to the released file) and `solver.vae.checkpoint` (the
released `VAE-iter-40000.pth`). `Trainer.load(10)` then loads model / EMA / optimiser with strict `load_state_dict`, and
the VAE encoders / decoder are loaded by `Trainer.load_vae_ecg` / `load_vae_ppg` exactly as released.

Sampling: identical to the body of the official `Trainer.sample_shift` —
  latent_data = vae_encoder_ecg(ecg)[0]          # official: only its shape is used
  latent_cond = vae_encoder_ppg(ppg)[0]          # official: stochastic VAE posterior sample of the PPG latent
  sample      = ema.ema_model.sample_shift(shape=latent_data.shape, cond=latent_cond, ...)
  sample      = vae_decoder_ecg(sample)
with ONE difference that must be stated: the released `Trainer.sample_shift` does not forward its `sampling_steps`
argument to the model, so the model's own default (`num_steps=10`) is always used. Here the step count S is passed to the
model's own official method `RectifiedFlow.sample_shift(shape, num_steps=S, cond=...)` — the same explicit-Euler loop,
the argument the paper's §4.5 varies. At S = 10 the two paths are the same computation.
Randomness: torch's global generators, seeded before every call; nothing is added or removed.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
UP = ROOT / "external/PPGFlowECG"
CKPT_FLOW = ROOT / "ppgflowecg/checkpoint-10.pt"
CKPT_VAE = ROOT / "ppgflowecg/VAE-iter-40000.pth"
RUN = ROOT / "outputs/ppgflowecg_external/run"
os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(UP))
from engine.solver import Trainer  # noqa: E402  (official)
from utils.io_utils import instantiate_from_config, load_yaml_config  # noqa: E402  (official)


class PFE:
    def __init__(self):
        cfg = load_yaml_config(str(UP / "config/latent_rectified_flow.yaml"))
        (RUN / "checkpoints").mkdir(parents=True, exist_ok=True)
        link = RUN / "checkpoints/checkpoint-10.pt"
        if not link.exists():
            link.symlink_to(CKPT_FLOW)
        cfg["solver"]["results_folder"] = str(RUN)
        cfg["solver"]["vae"]["checkpoint"] = str(CKPT_VAE)
        self.cfg = cfg
        model = instantiate_from_config(cfg["model"]).cuda()
        self.tr = Trainer(config=cfg, args=SimpleNamespace(), model=model, dataloader=[None], logger=None)
        self.tr.load(10)
        self.flow = self.tr.ema.ema_model
        self.calls = {"flow_net": 0, "enc_ecg": 0, "enc_ppg": 0, "dec_ecg": 0}
        for key, mod in (("flow_net", self.flow.model), ("enc_ecg", self.tr.vae_encoder_ecg),
                         ("enc_ppg", self.tr.vae_encoder_ppg), ("dec_ecg", self.tr.vae_decoder_ecg)):
            mod.register_forward_hook(lambda *a, _k=key: self.calls.__setitem__(_k, self.calls[_k] + 1))
        self.device = self.tr.device

    @torch.no_grad()
    def sample(self, ppg, ecg, S, seed, device=None):
        """ppg, ecg: float tensors [B, 1280] (official preprocessing). Returns generated ECG [B, 1280] (numpy float32).
        `device` is only for the CPU latency measurement (modules moved there by the caller)."""
        torch.manual_seed(seed)
        dev = self.device if device is None else device
        ppg = ppg.unsqueeze(-1).to(dev).float()
        ecg = ecg.unsqueeze(-1).to(dev).float()
        latent_data = self.tr.vae_encoder_ecg(ecg)[0]
        latent_cond = self.tr.vae_encoder_ppg(ppg)[0]
        x = self.flow.sample_shift(shape=latent_data.shape, num_steps=int(S), cond=latent_cond, report=None)
        x = self.tr.vae_decoder_ecg(x)
        return x[..., 0].float().cpu().numpy()

    def timed(self, ppg, ecg, S, seed, reps=5, device=None):
        """Median wall-clock (ms) of one full sampling call (encoders + S Euler steps + decoder + host copy); 1 warm-up."""
        ts = []
        for r in range(reps + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            self.sample(ppg, ecg, S, seed, device=device)
            torch.cuda.synchronize()
            if r:
                ts.append(1000 * (time.perf_counter() - t0))
        return float(sorted(ts)[len(ts) // 2])
