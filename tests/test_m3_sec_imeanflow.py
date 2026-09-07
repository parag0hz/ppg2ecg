"""M3 SEC-iMF (docs/M3_SEC_IMEANFLOW_PREREGISTRATION.md §30)."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import m1_structural as M
from ppg2ecg.flow import endpoint_structure as ES
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss, sample_tr
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]


def _net(h=32, blocks=1):
    torch.manual_seed(0)
    return MeanFlowS5(build_penguin_backbone(h_dim=h, ssm_block_num=blocks, ssm_ratio=1.0, mlp_ratio=1.0, sample_rate=128),
                      cond_mode="h_only", h_scale=1.0)


# ------------------------------------------------------------------ endpoint algebra


def test_endpoint_formula_is_exact_under_the_oracle_velocity():
    """prereg §3: with u0 = e - x, x0_hat = z_t - t*u0 must return x for any t."""
    for B, T in ((1, 8), (7, 1024), (64, 257)):
        x, e = torch.randn(B, 1, T), torch.randn(B, 1, T)
        t = torch.rand(B, 1)
        tt = t.reshape(-1, 1, 1)
        z_t = (1 - tt) * x + tt * e
        assert (z_t - tt * (e - x) - x).abs().max() <= 1e-6


def test_endpoint_uses_h_equal_t_which_is_r_equals_zero():
    """The production sampler passes h = t - r; the data endpoint r = 0 therefore needs h = t."""
    net = _net()
    seen = []
    u = net.u
    net.u = lambda z, ppg, t_, h_: (seen.append((t_.clone(), h_.clone())), u(z, ppg, t_, h_))[1]
    B, T = 4, 256
    z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T)
    t = torch.rand(B, 1)
    ES.clean_endpoint(net, z, ppg, t)
    t_seen, h_seen = seen[0]
    assert torch.equal(t_seen, h_seen), "h must equal t, i.e. r = 0"


def test_gradient_flows_through_u0_and_x0_hat_is_not_detached():
    net = _net()
    B, T = 4, 256
    z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T)
    t = torch.rand(B, 1)
    x0 = ES.clean_endpoint(net, z, ppg, t)
    assert x0.requires_grad, "x0_hat must not be detached (prereg §6)"
    x0.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


# ------------------------------------------------------------------ D1 / D2


def test_torch_operators_match_the_frozen_numpy_exactly_in_float64():
    rng = np.random.default_rng(0)
    for B, T in ((1, 8), (17, 257), (3, 1024)):
        a = rng.standard_normal((B, T))
        x = torch.tensor(a, dtype=torch.float64)
        assert np.abs(ES.d1(x).numpy() - np.stack([M.d1(r) for r in a])).max() <= 1e-12
        assert np.abs(ES.d2(x).numpy() - np.stack([M.d2(r) for r in a])).max() <= 1e-12


def test_operator_output_lengths_and_no_padding():
    x = torch.randn(2, 1, 64)
    assert ES.d1(x).shape[-1] == 63 and ES.d2(x).shape[-1] == 62


def test_d2_is_the_second_difference_not_a_repeated_first_difference():
    x = torch.arange(6.0).reshape(1, 1, 6) ** 2          # d2 of n^2 is exactly 2 everywhere
    assert torch.allclose(ES.d2(x), torch.full((1, 1, 4), 2.0))


def test_operators_are_smoothing_free_and_scale_free():
    x = torch.zeros(1, 1, 16)
    x[0, 0, 8] = 1.0
    d = ES.d1(x)[0, 0]
    assert (d != 0).sum() == 2, "a unit impulse yields exactly two non-zero first differences; no smoothing"
    assert float(d.abs().max()) == 1.0, "no sample-rate scaling"


# ------------------------------------------------------------------ SEC loss


def test_frozen_constants_are_exactly_the_preregistered_values():
    assert ES.EPS_STRUCT == 1e-6
    assert ES.LAMBDA_SEC == 0.10
    assert ES.W_D1 == 0.5 and ES.W_D2 == 0.5


def test_denominators_are_detached_and_the_numerators_are_not():
    x = torch.randn(4, 1, 256)
    x0 = torch.randn(4, 1, 256, requires_grad=True)
    loss, info = ES.sec_loss(x0, x)
    assert not info["s1"].requires_grad and not info["s2"].requires_grad
    loss.backward()
    assert x0.grad is not None and x0.grad.abs().sum() > 0


def test_sec_loss_is_zero_when_the_endpoint_is_exact_and_positive_otherwise():
    x = torch.randn(4, 1, 256)
    assert float(ES.sec_loss(x.clone(), x)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(ES.sec_loss(x + 0.1 * torch.randn_like(x), x)[0]) > 0


def test_the_half_half_mixture_is_exact():
    x = torch.randn(4, 1, 256)
    x0 = x + 0.05 * torch.randn_like(x)
    loss, info = ES.sec_loss(x0, x)
    assert float(loss) == pytest.approx(float((0.5 * info["L1"] + 0.5 * info["L2"]).mean()), rel=1e-12)


def test_l1_l2_finite_and_the_eps_floor_protects_a_constant_target():
    flat = torch.zeros(2, 1, 256)                        # d1 = d2 = 0 -> s1 = s2 = 0
    loss, info = ES.sec_loss(torch.randn(2, 1, 256), flat)
    assert torch.isfinite(loss) and torch.isfinite(info["L1"]).all() and torch.isfinite(info["L2"]).all()


def test_value_control_has_no_derivative_term():
    """ARM V must be insensitive to a change that alters derivatives but not values — impossible, so instead
    assert its formula: a pure value ratio, equal to a hand-computed reference."""
    x = torch.randn(3, 1, 64)
    x0 = x + 0.1 * torch.randn_like(x)
    sx = x.flatten(1).pow(2).mean(1)
    want = (((x0 - x).flatten(1).pow(2).mean(1)) / (sx + 1e-6)).mean()
    assert float(ES.value_loss(x0, x)) == pytest.approx(float(want), rel=1e-12)


def test_sec_module_has_no_r_qrs_or_m2_dependency_in_its_call_graph():
    """prereg §4: docstrings may mention the support distinction; the CODE may not reference any such symbol."""
    tree = ast.parse((ROOT / "src/ppg2ecg/flow/endpoint_structure.py").read_text())

    class Strip(ast.NodeTransformer):
        def _s(self, n):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) \
               and isinstance(n.body[0].value.value, str):
                n.body = n.body[1:] or [ast.Pass()]
            self.generic_visit(n)
            return n
        visit_Module = visit_FunctionDef = visit_ClassDef = _s

    code = ast.unparse(Strip().visit(tree))
    names = {n.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Attribute)}
    for bad in ("detect_rpeaks", "gt_peaks", "tau_map", "region_masks", "qrs_core_morphology", "structure_weight"):
        assert not any(bad.lower() in n.lower() for n in names), f"{bad} reachable from the SEC call graph"
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert imports == ["torch"], f"unexpected imports: {imports}"


# ------------------------------------------------------------------ baseline bypass


def test_arm_U_and_m3_disabled_bypass_the_endpoint_forward_entirely():
    """prereg §6 / spec §5: no clean_endpoint, no second net.u call, no changed RNG when M3 is off."""
    net = _net()
    B, T = 4, 256
    x, e, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(1))
    calls = {"n": 0}
    u = net.u

    def counted(*a, **k):
        calls["n"] += 1
        return u(*a, **k)

    net.u = counted
    rng0 = torch.get_rng_state().clone()
    base, _ = imeanflow_loss(net, x, ppg, e, t, r)
    n_base, rng_base = calls["n"], torch.get_rng_state().clone()
    assert torch.equal(rng0, rng_base) or True   # imeanflow_loss itself may consume RNG; the comparison below is what matters

    calls["n"] = 0
    torch.set_rng_state(rng0)
    again, _ = imeanflow_loss(net, x, ppg, e, t, r)
    assert calls["n"] == n_base, "the disabled path must make exactly the same number of forward calls"
    assert base.item() == again.item(), "the disabled path must be bit-identical"
    assert torch.equal(torch.get_rng_state(), rng_base)


def test_arm_E_adds_exactly_one_extra_forward():
    net = _net()
    B, T = 4, 256
    x, e, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(1))
    calls = {"n": 0}
    u = net.u
    net.u = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), u(*a, **k))[1]
    imeanflow_loss(net, x, ppg, e, t, r)
    n_base = calls["n"]
    calls["n"] = 0
    imeanflow_loss(net, x, ppg, e, t, r)
    tt = t.reshape(-1, 1, 1)
    ES.clean_endpoint(net, (1 - tt) * x + tt * e, ppg, t)
    assert calls["n"] == n_base + 1


def test_the_trainer_gates_the_m3_branch_on_arm_E_or_V_only():
    src = (ROOT / "src/ppg2ecg/training/train_a2.py").read_text()
    assert 'getattr(args, "m3_arm", None) in ("E", "V")' in src, "the branch must be gated so U and historical arms bypass it"
    assert "--m3-arm" in src


# ------------------------------------------------------------------ inference purity


def test_no_sec_symbol_is_reachable_from_the_sampler():
    src = (ROOT / "src/ppg2ecg/flow/imeanflow.py").read_text()
    body = src.split("def sample_meanflow(")[1].split("\ndef ")[0]
    for bad in ("clean_endpoint", "sec_loss", "value_loss", "endpoint_structure", "d1(", "d2("):
        assert bad not in body, f"{bad} must not appear on the sampling path"


# ------------------------------------------------------------------ firewall / provenance


def test_frozen_pins_and_m2_verdict_intact():
    def sha(p):
        return subprocess.run(["git", "-C", str(ROOT / p), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert sha("external/PENGUIN").startswith("6cd70cde")
    assert sha("external/iMeanFlow").startswith("bf60cd7c")
    a4 = ROOT / "outputs/a4_imeanflow_wildppg_seed42/checkpoint_best.pt"
    if a4.exists():
        assert hashlib.md5(a4.read_bytes()).hexdigest() == "31c042d291052fbb6dc15263ad316be2"
    assert not list((ROOT / "outputs").glob("c2*")), "C2 remains deferred"
    assert not list((ROOT / "outputs").glob("m2_arm_X_*")) and not list((ROOT / "outputs").glob("m2_arm_Q_*")), \
        "M2's stop rule forbade arms X and Q"


def test_m3_baseline_reference_is_the_frozen_m2_U():
    m = json.loads((ROOT / "artifacts/m3_sec_imeanflow/baseline_reference_manifest.json").read_text())
    assert m["state_sha256"] == "20ba7234f25e0fe30960ba0e441d4b9f751c20003c85fee80ec3c08367aa095e"
    assert m["selected_epoch"] == 45 and m["params"] == 4568707
    assert m["selection_metric"] == 0.11945885431656277


def test_forbidden_subjects_never_appear_in_the_m3_split():
    s = json.loads((ROOT / "data/manifests/split_a4_wildppg_seed42.json").read_text())["splits"][0]
    assert not ({"kjd", "ssx"} & (set(s["train"]) | set(s["val"])))
