"""A6 gradient-flow hard test (prereg §6): module-level gradient norms of the full-backbone deterministic MSE control at
optimizer steps 0, 1 and 5 on real PPG/ECG windows; also checks final-layer input non-zero and absence of target leakage
(the prediction must not change when the target changes). Writes artifacts/a6_capacity_control/gradient_flow.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import ppg2ecg.utils.mkl_warmup  # noqa: F401
import numpy as np
import torch

from ppg2ecg.models.regressor import REGRESSOR_MODELS

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"pre_conv_target (state stem)": "backbone.pre_conv_target", "timestep_embedder": "backbone.timestep_embedder", "pre_conv_ppg (PPG stem)": "backbone.pre_conv_ppg", "ssm_ppg (PPG-stream S5)": ".ssm_ppg", "ssm_target (target-stream S5)": ".ssm_target", "adaLN_modulation (blocks)": "flow_ssm_list.*.adaLN_modulation", "pre/post_attn (cross-stream MLPs)": "_attn_", "mlp_ppg/mlp_target": ".mlp_", "final_layer": "backbone.final_layer", "cross_attn (never called)": "cross_attn", "revin (never called)": "revin"}


def group_of(name):
    for g, pat in GROUPS.items():
        if pat.startswith("flow_ssm_list.*."):
            if "flow_ssm_list." in name and name.endswith(pat.split("*.")[1] + ".1.weight") or ("flow_ssm_list." in name and pat.split("*.")[1] in name):
                return g
        elif pat in name:
            return g
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="full_backbone", choices=list(REGRESSOR_MODELS))
    ap.add_argument("--processed", default="data/processed/v0_8s")
    ap.add_argument("--subject", default="S3")
    ap.add_argument("--out", default="artifacts/a6_capacity_control/gradient_flow.json")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 5, 20, 50, 100, 200, 300])
    ap.add_argument("--x-const", default=None, help="float or fixed_normal:<seed> (default: class default)")
    ap.add_argument("--t-const", type=float, default=None)
    ap.add_argument("--cond-scale", type=float, default=1.0)
    ap.add_argument("--reference-otcfm", action="store_true", help="log the same cascade for the generative OT-CFM objective on the unmodified backbone (reference)")
    ap.add_argument("--h-dim", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.reference_otcfm:
        from ppg2ecg.models import build_penguin_backbone, count_params

        bb = build_penguin_backbone(n_step=1, sample_rate=128, h_dim=args.h_dim, ssm_block_num=args.blocks)

        class _Ref(torch.nn.Module):  # OT-CFM training objective through upstream train_flow (loss on the velocity)
            def __init__(self):
                super().__init__()
                self.backbone = bb

            def forward(self, ppg):
                return self.backbone.forward_step(torch.randn_like(ppg), ppg, torch.rand(ppg.shape[0], device=ppg.device))

            def loss(self, ppg, y):
                self.backbone.train_flow(ppg.squeeze(1), y.squeeze(1))
                return ((self.backbone.pred_dx_t - self.backbone.dx_t) ** 2).mean()

        cls, count = _Ref, lambda m: count_params(m.backbone, exclude_prefixes=("cross_attn", "revin"))
        model = _Ref().to(device)
        args.out = args.out.replace(".json", "_reference_otcfm.json")
    else:
        cls, count = REGRESSOR_MODELS[args.model]
        kw = {}
        if args.model == "full_backbone":
            kw = {"x_const": (float(args.x_const) if args.x_const is not None and ":" not in args.x_const else args.x_const), "t_const": args.t_const, "cond_scale": args.cond_scale}
        model = cls(h_dim=args.h_dim, ssm_block_num=args.blocks, **kw).to(device)
    d = np.load(ROOT / args.processed / f"{args.subject}.npz")
    x_all, y_all = torch.from_numpy(d["x"][:64]).unsqueeze(1).to(device), torch.from_numpy(d["y"][:64]).unsqueeze(1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rec = {"model": cls.__name__, "objective": "otcfm (reference)" if args.reference_otcfm else "mse", "x_const": getattr(model, "x_const", None), "t_const": getattr(model, "t_const", None), "cond_scale": getattr(model, "cond_scale", None), "params": count(model), "steps": {}, "checks": {}}
    torch.manual_seed(42)
    # final-layer input non-zero at init & no target leakage
    with torch.no_grad():
        feats = {}
        h = model.backbone.final_layer.register_forward_hook(lambda m, i, o: feats.__setitem__("final_in", i[0].detach()))
        out_a = model(x_all[:8])
        h.remove()
        out_b = model(x_all[:8])  # same PPG -> identical prediction regardless of any target
        rec["checks"]["final_layer_input_abs_mean_init"] = float(feats["final_in"].abs().mean())
        rec["checks"]["final_layer_input_nonzero"] = bool(feats["final_in"].abs().max() > 0)
        rec["checks"]["deterministic"] = bool(torch.equal(out_a, out_b))
        rec["checks"]["no_target_in_forward_signature"] = "target" not in cls.forward.__code__.co_varnames
        rec["checks"]["output_identically_zero_at_init (upstream zero-init of final linear)"] = bool(out_a.abs().max() == 0)
    max_step = max(args.steps)
    for step in range(max_step + 1):
        idx = slice((step * 8) % 64, (step * 8) % 64 + 8)
        opt.zero_grad()
        loss = model.loss(x_all[idx], y_all[idx]) if args.reference_otcfm else ((model(x_all[idx]) - y_all[idx]) ** 2).mean()
        loss.backward()
        if step in args.steps:
            norms = {}
            for n, p in model.named_parameters():
                g = group_of(n)
                norms.setdefault(g, {"grad_norm_sq": 0.0, "n_params": 0, "n_tensors": 0, "n_zero_grad_tensors": 0})
                gn = float(p.grad.norm()) if p.grad is not None else 0.0
                norms[g]["grad_norm_sq"] += gn**2
                norms[g]["n_params"] += p.numel()
                norms[g]["n_tensors"] += 1
                norms[g]["n_zero_grad_tensors"] += int(gn == 0.0)
            rec["steps"][str(step)] = {"loss": float(loss), "groups": {g: {"grad_norm": v["grad_norm_sq"] ** 0.5, "n_params": v["n_params"], "n_tensors": v["n_tensors"], "n_zero_grad_tensors": v["n_zero_grad_tensors"]} for g, v in norms.items()}}
        opt.step()
    with torch.no_grad():
        model.eval()
        torch.manual_seed(0)
        pa, pb = model(x_all[:8]), model(x_all[8:16])
        rec["checks"]["output_depends_on_ppg_after_training_steps"] = bool(not torch.allclose(pa, pb))
        rec["checks"]["output_not_constant_over_time_after_training_steps"] = bool(pa.std(dim=-1).min() > 0)
        if not args.reference_otcfm:  # direct PPG-dependence measure: MSE with the correct vs a deranged PPG on held-out windows
            xs, ys = x_all[32:64], y_all[32:64]
            perm = torch.roll(torch.arange(len(xs)), 1)
            mse_ok, mse_sh = float(((model(xs) - ys) ** 2).mean()), float(((model(xs[perm]) - ys) ** 2).mean())
            rec["checks"]["mse_correct_ppg"], rec["checks"]["mse_shuffled_ppg"], rec["checks"]["shuffle_penalty_positive"] = mse_ok, mse_sh, bool(mse_sh > mse_ok)
    active = [g for g in GROUPS if "never called" not in g]
    late = [s for s in args.steps if s >= 5]
    rec["checks"]["only_final_layer_at_step0 (adaLN-Zero design, same as generative model)"] = all(rec["steps"]["0"]["groups"][g]["grad_norm"] == 0 for g in active if g != "final_layer") if "0" in rec["steps"] else None
    rec["checks"]["all_active_groups_nonzero_at_steps_ge5"] = all(rec["steps"][str(s)]["groups"][g]["grad_norm"] > 0 for s in late for g in active if g in rec["steps"][str(s)]["groups"])
    first, last = rec["steps"][str(late[0])]["groups"], rec["steps"][str(late[-1])]["groups"]
    rec["checks"]["pathway_gradients_grow_from_step5_to_last"] = {g: bool(last[g]["grad_norm"] > first[g]["grad_norm"]) for g in active if g in first}
    rec["checks"]["never_called_groups_zero"] = all(rec["steps"][str(s)]["groups"][g]["grad_norm"] == 0 for s in args.steps for g in GROUPS if "never called" in g and g in rec["steps"][str(s)]["groups"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    for s in args.steps:
        print(f"step {s}: loss {rec['steps'][str(s)]['loss']:.4f} | " + " | ".join(f"{g.split(' ')[0]} {v['grad_norm']:.2e}" + (f" ({v['n_zero_grad_tensors']}/{v['n_tensors']} zero)" if v['n_zero_grad_tensors'] else "") for g, v in rec["steps"][str(s)]["groups"].items()))
    # PASS criterion (prereg §3): final layer only at step 0 (adaLN-Zero design), all active groups non-zero at steps >= 5, never-called
    # groups zero, deterministic, no target in the forward pass. PPG dependence after these few small-batch steps is informational only —
    # the decisive PPG-dependence test is the 12-epoch training screening (prereg §2b, state_constant_screening.json).
    pass_keys = ["final_layer_input_nonzero", "deterministic", "no_target_in_forward_signature", "only_final_layer_at_step0 (adaLN-Zero design, same as generative model)", "all_active_groups_nonzero_at_steps_ge5", "never_called_groups_zero"]
    rec["checks"]["PASS"] = all(bool(rec["checks"].get(k)) for k in pass_keys)
    out.write_text(json.dumps(rec, indent=1))
    print("checks:", rec["checks"])
    print("PASS" if rec["checks"]["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
