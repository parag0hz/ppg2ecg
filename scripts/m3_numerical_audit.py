"""M3 §10-§12 — TRAIN12-only numerical audit, t-bin diagnostic and gradient diagnostic.

NO an0, NO k2s, NO kjd, NO ssx. No training, no checkpoint, no validation metric.
Hard gates A0-1..A0-5 (prereg §10 / spec §10). The t-bin and gradient reports are DESCRIPTIVE ONLY and may never
become tuning: no lambda change, no t reweighting, no sampler change, no low-t exclusion follows from them.

Run: .venv/bin/python scripts/m3_numerical_audit.py
"""
from __future__ import annotations

import ppg2ecg.utils.mkl_warmup  # noqa: F401

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ppg2ecg.flow.endpoint_structure import LAMBDA_SEC, clean_endpoint, sec_loss
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss
from ppg2ecg.flow.interval_exposure import sample_tr_c1
from ppg2ecg.models import build_penguin_backbone
from ppg2ecg.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/m3_sec_imeanflow"
MANIFEST = ROOT / "data/manifests/split_a4_wildppg_seed42.json"
PROCESSED = ROOT / "data/processed/wildppg_8s"
FORBIDDEN = ("kjd", "ssx")
VAL_FORBIDDEN_HERE = ("an0", "k2s")          # §10: the audit is TRAIN12-only
T_BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001))
N_GRAD_MICRO = 16


def load_train(n: int):
    split = json.loads(MANIFEST.read_text())["splits"][0]
    subs = sorted(split["train"])
    assert not (set(subs) & set(FORBIDDEN + VAL_FORBIDDEN_HERE)), "audit must be TRAIN12-only"
    per = -(-n // len(subs))
    xs, ys = [], []
    for s in subs:
        d = np.load(PROCESSED / f"{s}.npz")
        k = min(per, len(d["x"]), n - sum(len(a) for a in xs))
        if k <= 0:
            break
        xs.append(d["x"][:k].astype(np.float32))
        ys.append(d["y"][:k].astype(np.float32))
    return np.concatenate(xs), np.concatenate(ys), subs


def q(a, name):
    a = np.asarray(a, dtype=np.float64)
    return {f"{name}_min": float(a.min()), f"{name}_median": float(np.median(a)),
            f"{name}_p95": float(np.percentile(a, 95)), f"{name}_p99": float(np.percentile(a, 99)),
            f"{name}_max": float(a.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_ppg, y_ecg, subs = load_train(args.n)
    seed_everything(42, deterministic=True)
    net = MeanFlowS5(build_penguin_backbone(h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0, sample_rate=128),
                     cond_mode="h_only", h_scale=1.0).to(dev)
    net.train()
    tr_gen = torch.Generator()
    tr_gen.manual_seed(43)                    # seed + 1, as train_a2 does for the (t, r) stream
    torch.manual_seed(42)

    rows, base_losses, aux_losses = [], [], []
    for i in range(0, len(x_ppg), args.batch):
        ppg = torch.from_numpy(x_ppg[i:i + args.batch]).to(dev).unsqueeze(1)
        ecg = torch.from_numpy(y_ecg[i:i + args.batch]).to(dev).unsqueeze(1)
        B = len(ppg)
        t, r, _ = sample_tr_c1(B, tr_gen, arm="B", p_mean=-0.4, p_std=1.0, data_proportion=0.5)
        t, r = t.to(dev), r.to(dev)
        e = torch.randn(B, 1, ecg.shape[-1], device=dev)
        base, _info = imeanflow_loss(net, ecg, ppg, e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward")
        tt = t.reshape(-1, 1, 1)
        z_t = (1 - tt) * ecg + tt * e
        x0 = clean_endpoint(net, z_t, ppg, t)
        aux, d = sec_loss(x0, ecg)
        base_losses.append(float(base.detach()))
        aux_losses.append(float(aux.detach()))
        finite = torch.isfinite(x0).flatten(1).all(1)
        for b in range(B):
            rows.append({"t": float(t[b, 0]), "L1": float(d["L1"][b]), "L2": float(d["L2"][b]),
                         "s1": float(d["s1"][b]), "s2": float(d["s2"][b]),
                         "L_SEC": float(0.5 * d["L1"][b] + 0.5 * d["L2"][b]),
                         "x0_finite": bool(finite[b])})
        # forward-mode JVP retains a large graph (~0.5 GiB/sample at T=1024); nothing here is backpropagated, so
        # release both graphs before the next batch allocates. Without this the audit OOMs at batch 32.
        del base, _info, x0, aux, d, z_t, tt, e, ppg, ecg, finite
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    t_ = np.array([r_["t"] for r_ in rows])
    L1 = np.array([r_["L1"] for r_ in rows])
    L2 = np.array([r_["L2"] for r_ in rows])
    LS = np.array([r_["L_SEC"] for r_ in rows])
    s1 = np.array([r_["s1"] for r_ in rows])
    s2 = np.array([r_["s2"] for r_ in rows])
    finite_all = bool(np.isfinite(L1).all() and np.isfinite(L2).all() and np.isfinite(LS).all()
                      and all(r_["x0_finite"] for r_ in rows))

    # ---- §12 gradient diagnostic on exactly 16 deterministic microbatches
    seed_everything(42, deterministic=True)
    net2 = MeanFlowS5(build_penguin_backbone(h_dim=128, ssm_block_num=4, ssm_ratio=2.0, mlp_ratio=2.0, sample_rate=128),
                      cond_mode="h_only", h_scale=1.0).to(dev)
    net2.train()
    g2 = torch.Generator()
    g2.manual_seed(43)
    torch.manual_seed(42)
    nb, ns, cos, tot = [], [], [], []
    for i in range(N_GRAD_MICRO):
        ppg = torch.from_numpy(x_ppg[i * 16:(i + 1) * 16]).to(dev).unsqueeze(1)
        ecg = torch.from_numpy(y_ecg[i * 16:(i + 1) * 16]).to(dev).unsqueeze(1)
        B = len(ppg)
        t, r, _ = sample_tr_c1(B, g2, arm="B", p_mean=-0.4, p_std=1.0, data_proportion=0.5)
        t, r = t.to(dev), r.to(dev)
        e = torch.randn(B, 1, ecg.shape[-1], device=dev)
        flat = lambda gs: torch.cat([g.flatten() for g in gs])  # noqa: E731
        base, _ = imeanflow_loss(net2, ecg, ppg, e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward")
        gb = flat(torch.autograd.grad(base, list(net2.parameters()), retain_graph=False, allow_unused=True,
                                      materialize_grads=True))
        tt = t.reshape(-1, 1, 1)
        aux, _ = sec_loss(clean_endpoint(net2, (1 - tt) * ecg + tt * e, ppg, t), ecg)
        gsx = flat(torch.autograd.grad(aux, list(net2.parameters()), allow_unused=True, materialize_grads=True))
        nb.append(float(gb.norm()))
        ns.append(float((LAMBDA_SEC * gsx).norm()))
        # The S5 backbone carries COMPLEX parameters (the SSM Lambda), so the flattened gradient is ComplexFloat
        # and torch.nn.functional.cosine_similarity rejects it. The real part of the Hermitian inner product is the
        # ordinary Euclidean cosine of the equivalent real vectors, which is what is wanted here.
        cos.append(float(torch.real(torch.sum(gb.conj() * gsx)) / (gb.norm() * gsx.norm() + 1e-30)))
        tot.append(float((gb + LAMBDA_SEC * gsx).norm()))
        assert torch.isfinite(gb).all() and torch.isfinite(gsx).all(), "non-finite gradient"
        del gb, gsx, base, aux, e, ppg, ecg
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    gates = {
        "A0-1_all_finite": finite_all and bool(np.isfinite(nb).all() and np.isfinite(ns).all()),
        "A0-2_frac_s1_below_1e-6_lt_0.1pct": bool(float((s1 < 1e-6).mean()) < 0.001),
        "A0-3_frac_s2_below_1e-6_lt_0.1pct": bool(float((s2 < 1e-6).mean()) < 0.001),
        "A0-4_no_nan_inf_gradient": bool(np.isfinite(nb).all() and np.isfinite(ns).all() and np.isfinite(tot).all()),
        "A0-5_sec_gradient_norm_positive": bool(min(ns) > 0),
    }
    tbins = []
    for lo, hi in T_BINS:
        m = (t_ >= lo) & (t_ < hi)
        tbins.append({"bin": f"[{lo:.1f},{min(hi,1.0):.1f})", "n": int(m.sum()),
                      "mean_L1": float(L1[m].mean()) if m.any() else None,
                      "mean_L2": float(L2[m].mean()) if m.any() else None,
                      "mean_L_SEC": float(LS[m].mean()) if m.any() else None})

    audit = {
        "scope": "TRAIN12 only", "subjects": subs, "n_windows": int(len(x_ppg)),
        "forbidden_absent": {"an0": True, "k2s": True, "kjd": True, "ssx": True},
        "x0_hat_finite": finite_all,
        **q(s1, "s1"), **q(s2, "s2"), **q(L1, "L1"), **q(L2, "L2"), **q(LS, "L_SEC"),
        "frac_s1_lt_1e-6": float((s1 < 1e-6).mean()), "frac_s2_lt_1e-6": float((s2 < 1e-6).mean()),
        "L_iMF_mean": float(np.mean(base_losses)), "lambda_L_SEC_mean": float(LAMBDA_SEC * np.mean(aux_losses)),
        "aux_over_base_ratio": float(LAMBDA_SEC * np.mean(aux_losses) / np.mean(base_losses)),
        "lambda_SEC": LAMBDA_SEC, "gates": gates, "all_gates_pass": bool(all(gates.values())),
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }
    (OUT / "train_numerical_audit.json").write_text(json.dumps(audit, indent=1))
    (OUT / "gradient_diagnostic.json").write_text(json.dumps({
        "n_microbatches": N_GRAD_MICRO, "descriptive_only": True,
        "grad_norm_base_mean": float(np.mean(nb)), "grad_norm_lambda_sec_mean": float(np.mean(ns)),
        "grad_norm_total_mean": float(np.mean(tot)),
        "cosine_mean": float(np.mean(cos)), "cosine_min": float(np.min(cos)), "cosine_max": float(np.max(cos)),
        "ratio_lambda_sec_over_base": float(np.mean(ns) / np.mean(nb)),
        "policy": "no gate; lambda, sampler, layer choice and weighting may not change because of these numbers"},
        indent=1))
    (OUT / "t_bin_diagnostic.json").write_text(json.dumps({
        "bins": tbins, "descriptive_only": True,
        "policy": "no gate depends on monotonicity; no t reweighting, no low-t exclusion, no lambda change"}, indent=1))

    print(json.dumps({"gates": gates, "all_pass": audit["all_gates_pass"],
                      "frac_s1_lt_1e-6": audit["frac_s1_lt_1e-6"], "frac_s2_lt_1e-6": audit["frac_s2_lt_1e-6"],
                      "L1_median": audit["L1_median"], "L2_median": audit["L2_median"],
                      "L_SEC_median": audit["L_SEC_median"],
                      "L_iMF_mean": audit["L_iMF_mean"], "lambda_L_SEC_mean": audit["lambda_L_SEC_mean"],
                      "aux_over_base_ratio": audit["aux_over_base_ratio"]}, indent=1))
    print("\nt-bins:", json.dumps(tbins, indent=1))
    print("\ngradient:", json.dumps({"||g_base||": float(np.mean(nb)), "||0.1*g_sec||": float(np.mean(ns)),
                                     "cosine": float(np.mean(cos))}, indent=1))
    print("\nA0 VERDICT:", "PASS — ready to train arm E" if audit["all_gates_pass"] else "FAIL — STOP")
    return 0 if audit["all_gates_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
