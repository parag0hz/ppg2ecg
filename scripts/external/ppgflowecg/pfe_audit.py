"""PPGFlowECG external validation — steps 1–2: audit, strict checkpoint load, NFE accounting, latency.
No VitalDB data is read here: NFE / latency use synthetic N(0, 1) inputs of the model's input shape.

Writes artifacts/ppgflowecg_external/{audit.json, checkpoint_hashes.txt, nfe_counts.json, latency.json}.
Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=outputs/pfe_env/core:scripts/external/ppgflowecg .venv/bin/python \
     scripts/external/ppgflowecg/pfe_audit.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

import official as O
from pfe_model import CKPT_FLOW, CKPT_VAE, PFE, ROOT, UP

OUT = ROOT / "artifacts/ppgflowecg_external"
USER_DIR = ROOT / "ppgflowecg"
SS = (5, 10, 15, 20, 25)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(UP), *a], capture_output=True, text=True).stdout.strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = False
    # ---------------------------------------------------------------- hashes / repository
    files = sorted(p.name for p in USER_DIR.iterdir())
    hashes = {p.name: sha256(p) for p in sorted(USER_DIR.iterdir()) if p.is_file()}
    src = O.source_hashes()
    lines = [f"{v}  ppgflowecg/{k}" for k, v in hashes.items()]
    lines += [f"{v}  external/PPGFlowECG@{O.UP_COMMIT[:7]}:{k}  (extracted official function source)" for k, v in src.items()]
    (OUT / "checkpoint_hashes.txt").write_text("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- official load (strict)
    pfe = PFE()
    ck = torch.load(str(CKPT_FLOW), map_location="cpu", weights_only=False)
    strict = {}
    for name, mod, sd in (("model", pfe.tr.model, ck["model"]), ("ema", pfe.tr.ema, ck["ema"])):
        r = mod.load_state_dict(sd, strict=True)            # re-applies the same tensors; returns incompatible keys
        strict[name] = {"missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)}
    vae = torch.load(str(CKPT_VAE), map_location="cpu", weights_only=False)
    for name, mod in (("encoder_ecg", pfe.tr.vae_encoder_ecg), ("decoder_ecg", pfe.tr.vae_decoder_ecg),
                      ("encoder_ppg<-encoder_ecg", pfe.tr.vae_encoder_ppg)):
        key = name.split("<-")[-1]
        r = mod.load_state_dict(vae[key], strict=True)
        strict[f"vae.{name}"] = {"missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)}
    enc_same = all(torch.equal(vae["encoder_ecg"][k], vae["encoder_ppg"][k]) for k in vae["encoder_ecg"])
    ema_eq_model = all(torch.equal(a, b) for a, b in zip(pfe.flow.state_dict().values(), pfe.tr.model.state_dict().values()))
    npar = lambda m: int(sum(p.numel() for p in m.parameters()))  # noqa: E731
    drop = sorted({float(m.p) for m in pfe.flow.modules() if isinstance(m, torch.nn.Dropout)})

    # ---------------------------------------------------------------- NFE accounting (hooks)
    g = torch.Generator().manual_seed(0)
    ppg = torch.randn(4, 1280, generator=g); ecg = torch.randn(4, 1280, generator=g)
    nfe = {}
    for S in SS:
        for k in pfe.calls:
            pfe.calls[k] = 0
        y = pfe.sample(ppg, ecg, S, seed=1)
        nfe[str(S)] = dict(pfe.calls, output_shape=list(y.shape), finite=bool(np.isfinite(y).all()))
    # the released Trainer.sample_shift (official batch loop) with sampling_steps=5: does the step count reach the model?
    for k in pfe.calls:
        pfe.calls[k] = 0
    torch.manual_seed(7)
    off = pfe.tr.sample_shift([(ppg, ecg, torch.zeros(4))], shape=[1280, 1], sampling_steps=5,
                              save_dir=str(ROOT / "outputs/ppgflowecg_external/run/official_sample_shift"))
    official_calls = dict(pfe.calls)
    ours10 = pfe.sample(ppg, ecg, 10, seed=7)
    off_arr = np.asarray(off[0] if isinstance(off, tuple) else off)[..., 0]
    same_as_ours_S10 = bool(np.array_equal(off_arr.astype(np.float32), ours10))
    max_abs_diff = float(np.abs(off_arr - ours10).max())
    nfe_doc = {
        "definition": "calls per sampling call; one call processes the whole batch, so per-sample counts are identical",
        "per_S": nfe,
        "vector_field_nfe_equals_S": all(nfe[str(S)]["flow_net"] == S for S in SS),
        "fixed_overhead_per_sample": {"vae_encoder_ecg (official: only its output shape is used)": 1,
                                      "vae_encoder_ppg (stochastic posterior sample)": 1, "vae_decoder_ecg": 1},
        "euler_update": "x <- x + (1/S) * v(x, t), t on linspace(0, 1, S+1)[:-1]; one vector-field call per step",
        "released_Trainer.sample_shift": {
            "called_with_sampling_steps": 5, "flow_net_calls": official_calls["flow_net"],
            "finding": "sampling_steps is not forwarded to the model (engine/solver.py:276-280); the model default num_steps=10 runs",
            "identical_to_our_driver_at_S10_same_seed": same_as_ours_S10, "max_abs_diff_vs_our_S10": max_abs_diff},
    }
    (OUT / "nfe_counts.json").write_text(json.dumps(nfe_doc, indent=1))

    # ---------------------------------------------------------------- latency
    lat = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "unit": "ms, median of reps after 1 warm-up",
           "includes": "encoder_ecg + encoder_ppg + S Euler steps + decoder + device->host copy", "gpu_batch1": {},
           "gpu_batch256_per_window": {}, "cpu_batch1": {}, "cpu_threads": torch.get_num_threads()}
    p1, e1 = ppg[:1], ecg[:1]
    pb = torch.randn(256, 1280, generator=g); eb = torch.randn(256, 1280, generator=g)
    for S in SS:
        lat["gpu_batch1"][str(S)] = pfe.timed(p1, e1, S, seed=3, reps=20)
        lat["gpu_batch256_per_window"][str(S)] = pfe.timed(pb, eb, S, seed=3, reps=5) / 256
    mods = (pfe.flow, pfe.tr.vae_encoder_ecg, pfe.tr.vae_encoder_ppg, pfe.tr.vae_decoder_ecg)
    for m in mods:
        m.cpu()
    for S in SS:
        lat["cpu_batch1"][str(S)] = pfe.timed(p1, e1, S, seed=3, reps=3, device=torch.device("cpu"))
    for m in mods:
        m.cuda()
    b1 = lat["gpu_batch1"]
    lat["gpu_batch1_marginal_ms_per_euler_step"] = float(np.polyfit(SS, [b1[str(S)] for S in SS], 1)[0])
    lat["gpu_batch1_fixed_overhead_ms"] = float(np.polyfit(SS, [b1[str(S)] for S in SS], 1)[1])
    (OUT / "latency.json").write_text(json.dumps(lat, indent=1))

    # ---------------------------------------------------------------- audit
    audit = {
        "1_is_git_repository": {"ppgflowecg/ (user-provided)": False,
                                "note": "contains only the paper PDF and the two released checkpoints; no source code",
                                "source_used": "external/PPGFlowECG (git submodule of https://github.com/PKUDigitalHealth/PPGFlowECG)"},
        "2_exact_git_commit": git("rev-parse", "HEAD"),
        "3_git_status": {"porcelain": git("status", "--porcelain"), "clean": git("status", "--porcelain") == ""},
        "4_checkpoint_filenames": [f for f in files if f.endswith((".pt", ".pth"))],
        "5_checkpoint_sha256": {"checkpoint-10.pt": hashes.get("checkpoint-10.pt")},
        "6_vae_cardioalign_sha256": {"VAE-iter-40000.pth": hashes.get("VAE-iter-40000.pth")},
        "7_paper_pdf": [f for f in files if f.endswith(".pdf")],
        "8_config_files": sorted(str(p.relative_to(UP)) for p in (UP / "config").glob("*")),
        "9_official_sampling": "engine/solver.py Trainer.sample_shift: vae_encoder_ecg(ecg)[0] (shape only) -> vae_encoder_ppg(ppg)[0] "
                               "(cond) -> ema.ema_model.sample_shift(shape, cond) -> vae_decoder_ecg",
        "10_model_load_path": "main.py: instantiate_from_config(config['model']) -> Trainer(config, ...) (loads VAE via "
                              "load_vae_ecg / load_vae_ppg) -> Trainer.load(milestone=10): results_folder/checkpoints/checkpoint-10.pt",
        "11_sample_and_sample_shift": "model/latent_rectified_flow/rectified_flow.py: sample(shape, num_steps) = explicit Euler from "
                                      "x ~ N(0, I) at t = 0 to t = 1 (data), timesteps linspace(0,1,S+1)[:-1]; sample_shift(shape, "
                                      "num_steps=10, cond) = sample(...)",
        "12_preprocessing_assumptions": "data_process_to_npz/step1.py: per 10-s window, PPG Butterworth band-pass 0.5-8 Hz order 3 (mne, "
                                        "IIR), ECG high-pass 0.5 Hz order 5 + 50 Hz notch, resample_poly to 128 Hz, per-window "
                                        "z-score; step2.py: Savitzky-Golay (PPG 7/2, ECG 11/2) + quality selection. Native rates "
                                        "hard-coded in step1 for the released pipeline: ECG 125 Hz, PPG 25 Hz.",
        "13_expected_ppg_input_shape": "[B, 1280, 1] (10 s x 128 Hz; the Trainer unsqueezes [B, 1280])",
        "14_expected_ecg_output_shape": "[B, 1280, 1]",
        "15_latent_dimensions": "[B, 4, 40] (VAE_Encoder: 8 channels -> mean / log-variance 4 each, length 1280/32)",
        "16_cardioalign_load_sequence": "Trainer.__init__: load_vae_ecg -> VAE_Encoder <- encoder_ecg, VAE_Decoder <- decoder_ecg; "
                                        "load_vae_ppg -> VAE_Encoder <- encoder_ecg (sic), VAE_Decoder <- decoder_ppg; all eval(), "
                                        "requires_grad False",
        "strict_load": strict,
        "strict_load_ok": all(not v["missing"] and not v["unexpected"] for v in strict.values()),
        "checkpoint_step": int(ck["step"]),
        "checkpoint_keys": sorted(ck.keys()),
        "vae_checkpoint_keys": sorted(vae.keys()),
        "vae_encoder_ppg_equals_encoder_ecg_in_checkpoint": enc_same,
        "ema_equals_online_model": ema_eq_model,
        "sampler_network": "EMA model (ema.ema_model), as in the official Trainer.sample_shift",
        "parameters": {"flow_net": npar(pfe.flow), "vae_encoder": npar(pfe.tr.vae_encoder_ppg), "vae_decoder_ecg": npar(pfe.tr.vae_decoder_ecg)},
        "flow_dropout_p_values": drop,
        "flow_training_flag": bool(pfe.flow.training),
        "stochasticity_sources": ["vae_encoder_ecg posterior noise (consumed, output unused)", "vae_encoder_ppg posterior noise (cond)",
                                  "Euler initial noise x ~ N(0, I)"],
        "sampling_steps_forwarding": nfe_doc["released_Trainer.sample_shift"]["finding"],
        "provenance_wording": "authors' officially released checkpoint; training provenance is strongly implied to be MCMED by the "
                              "released training configuration but is not explicitly documented for the checkpoint itself",
        "upstream_modified": False,
        "official_function_source_sha256": src,
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=1))
    print(json.dumps({k: audit[k] for k in ("2_exact_git_commit", "3_git_status", "strict_load_ok", "checkpoint_step",
                                           "vae_encoder_ppg_equals_encoder_ecg_in_checkpoint", "ema_equals_online_model",
                                           "parameters", "flow_dropout_p_values", "flow_training_flag")}, indent=1))
    print(json.dumps(nfe_doc, indent=1)); print(json.dumps(lat, indent=1))


if __name__ == "__main__":
    main()
