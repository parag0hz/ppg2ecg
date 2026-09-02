"""R3 tests — docs/R3_DISENTANGLED_RHYTHM_FUSION_PREREGISTRATION.md (3d779fc) section 23.

Unconditional tests use a tiny random backbone; real-checkpoint / real-data tests skip when the files are
absent. No optimizer step on the real generator; no training.
"""
import ast
import copy
import csv
import hashlib
import inspect
import itertools
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.flow import rhythm_fusion as RF
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss, sample_meanflow, sample_tr
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
GEN, TCN, AN0 = ROOT / RT.GENERATOR_CKPT, ROOT / RT.RHYTHM_CKPT, ROOT / "data/processed/wildppg_8s/an0.npz"
TINY = dict(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)


def _code(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _tiny(family="gtf", gate_mode="adaptive"):
    torch.manual_seed(0)
    bb = build_penguin_backbone(**TINY)
    for m in bb.modules():
        if isinstance(m, torch.nn.Linear) and float(m.weight.abs().sum()) == 0.0:
            torch.nn.init.normal_(m.weight, std=0.05); torch.nn.init.normal_(m.bias, std=0.05)
    base = MeanFlowS5(copy.deepcopy(bb)).eval(); base.requires_grad_(False)
    mod = RF.build_r3_module(family, gate_mode, c_hidden=TINY["h_dim"])
    net = RF.FusionMeanFlowS5(copy.deepcopy(bb), mod).eval(); net.backbone.requires_grad_(False)
    return base, net


# ---------------------------------------------------------------- parameters / init
def test_parameter_counts_and_names_match_the_prereg():
    tf = RF.build_r3_module("tf", None); gtf = RF.build_r3_module("gtf", "adaptive"); cst = RF.build_r3_module("gtf", "const")
    assert RF.n_params(tf) == RF.EXPECTED_TF_PARAMS == 12_768 <= RF.PARAM_BUDGET
    assert RF.EXPECTED_TF_PARAMS / RF.GENERATOR_PARAMS < RF.PARAM_BUDGET_FRAC
    assert RF.n_params(gtf) == RF.n_params(cst) == RF.EXPECTED_GTF_PARAMS == 12_849
    assert RF.n_params(gtf.gate) == RF.EXPECTED_GATE_PARAMS == 81 < RF.GATE_PARAM_BUDGET
    assert RF.param_names(tf) == RF.TF_PARAM_NAMES and RF.param_names(gtf) == RF.GTF_PARAM_NAMES
    for arm in ("gtf_true", "gtf_shuffle", "gtf_const", "gtf_oracle"):
        assert RF.n_params(RF.build_r3_module(RF.ARM_FAMILY[arm], RF.ARM_GATE_MODE[arm])) == RF.EXPECTED_GTF_PARAMS


def test_initialisation_is_identical_within_families_and_on_the_shared_fusion_subset():
    hs = {arm: RF.params_sha256(RF.build_r3_module(RF.ARM_FAMILY[arm], RF.ARM_GATE_MODE[arm])) for arm in RF.TRAINED_ARMS}
    assert hs["tf_true"] == hs["tf_shuffle"]
    assert hs["gtf_true"] == hs["gtf_shuffle"] == hs["gtf_const"] == hs["gtf_oracle"]
    fs = {arm: RF.params_sha256(RF.build_r3_module(RF.ARM_FAMILY[arm], RF.ARM_GATE_MODE[arm]), only_prefix="fusion.") for arm in RF.TRAINED_ARMS}
    assert len(set(fs.values())) == 1                                   # gate constructed after the out-projection


def test_output_projection_and_gate_final_layer_init():
    m = RF.build_r3_module("gtf", "adaptive")
    assert torch.all(m.fusion.out.weight == 0) and torch.all(m.fusion.out.bias == 0)
    assert torch.all(m.gate[2].weight == 0) and float(m.gate[2].bias) == pytest.approx(math.log(9.0))
    g = m.gate_values(torch.rand(2, 1, 1024))
    assert torch.allclose(g, torch.full_like(g, 0.9), atol=1e-6)
    assert not any(k.startswith("fusion.pe") for k in m.state_dict())    # PE buffers are non-persistent


# ---------------------------------------------------------------- hook / parity / JVP
def test_hook_is_target_side_and_pre_conv_ppg_receives_ppg_only():
    base, net = _tiny()
    seen = {}
    def _cap(key):
        def hook(m, i, o):
            seen[key] = i[0].clone()                                      # return None: never replace the output
        return hook
    h1 = net.backbone.pre_conv_ppg.register_forward_hook(_cap("ppg_in"))
    h2 = net.backbone.pre_conv_target.register_forward_hook(_cap("z_in"))
    B, T = 2, 256
    z, ppg, s = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.rand(B, 1, T)
    net.u(z, RT.make_ppg2(ppg, s), torch.full((B, 1), 0.6), torch.full((B, 1), 0.2))
    h1.remove(); h2.remove()
    assert torch.equal(seen["ppg_in"], ppg) and torch.equal(seen["z_in"], z)
    src = _code(ROOT / "src/ppg2ecg/flow/rhythm_fusion.py")
    body = src[src.index("class FusionMeanFlowS5"):src.index("def trainable_names")]
    assert "z_e = z_e + delta" in body and "pre_conv_ppg(ppg)" in body and "pre_conv_ppg(ppg) +" not in body
    assert "pre_conv_ppg" not in src[src.index("class RhythmCrossFusionAdapter"):src.index("class FusionMeanFlowS5")]


def test_step0_parity_zero_real_shuffled_scaffold_and_nonzero_changes_output():
    for family, gm in (("tf", None), ("gtf", "adaptive"), ("gtf", "const")):
        base, net = _tiny(family, gm)
        B, T = 3, 256
        z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T); t, h = torch.full((B, 1), 0.7), torch.full((B, 1), 0.3)
        ref = base.u(z, ppg, t, h)
        s = torch.rand(B, 1, T)
        for sc in (torch.zeros(B, 1, T), s, s.flip(0)):
            assert torch.equal(net.u(z, RT.make_ppg2(ppg, sc), t, h), ref)
            assert torch.isfinite(net.r3.fusion.attn(torch.randn(B, 1024, 32), torch.randn(B, 256, 32))).all()
        with torch.no_grad():
            net.r3.fusion.out.weight.normal_(0, 0.05)
        assert not torch.equal(net.u(z, RT.make_ppg2(ppg, s), t, h), ref)
        net.cancel_direct_route = True                                  # identity at zero output (section 18.2)
        with torch.no_grad():
            net.r3.fusion.out.weight.zero_()
        assert torch.equal(net.u(z, RT.make_ppg2(ppg, s), t, h), ref)


def test_frozen_loss_and_sampler_pass_through_with_finite_r3_gradients_only():
    base, net = _tiny("gtf", "adaptive")
    B, T = 2, 256
    ppg, e, x, s = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T), torch.rand(B, 1, T)
    zb, _ = sample_meanflow(base, ppg, e, n_steps=4); zn, _ = sample_meanflow(net, RT.make_ppg2(ppg, s), e, n_steps=4)
    assert torch.equal(zb, zn)
    with torch.no_grad():
        net.r3.fusion.out.weight.normal_(0, 0.05)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(1))
    loss, _ = imeanflow_loss(net, x, RT.make_ppg2(ppg, s), e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward")
    loss.backward()
    grads = {n: p.grad for n, p in net.r3.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert float(sum(g.abs().sum() for g in grads.values())) > 0
    RF.assert_frozen_have_no_grad(net)
    RF.assert_only_r3_trainable(net)


def test_global_tcn_receives_no_gradient_through_the_scaffold():
    from ppg2ecg.probes.rhythm_tcn import RhythmTCN
    base, net = _tiny("gtf", "adaptive")
    torch.manual_seed(3); tcn = RhythmTCN((1, 2)).eval(); tcn.requires_grad_(False)
    B, T = 2, 256
    ppg, e, x = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    s = RT.scaffold_from_ppg(tcn, ppg)
    with torch.no_grad():
        net.r3.fusion.out.weight.normal_(0, 0.05)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(2))
    loss, _ = imeanflow_loss(net, x, RT.make_ppg2(ppg, s), e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward"); loss.backward()
    RF.assert_frozen_have_no_grad(net, tcn)


def test_module_does_not_use_fused_attention():
    src = _code(ROOT / "src/ppg2ecg/flow/rhythm_fusion.py")
    assert "scaled_dot_product_attention" not in src and "MultiheadAttention" not in src


# ---------------------------------------------------------------- positional encoding / tokens
def test_positional_encoding_deterministic_and_on_the_same_sample_axis():
    pe1, pe2 = RF.sinusoidal_pe(RF.QUERY_POSITIONS), RF.sinusoidal_pe(RF.QUERY_POSITIONS)
    assert torch.equal(pe1, pe2) and pe1.shape == (1024, 32)
    tok = RF.sinusoidal_pe(RF.TOKEN_POSITIONS)
    assert tok.shape == (256, 32) and torch.equal(tok, pe1[::4])          # token j sits at sample 4j
    m = RF.build_r3_module("tf", None)
    assert torch.equal(m.fusion.pe_q, pe1) and torch.equal(m.fusion.pe_k, tok)


def test_rhythm_tokens_preserve_temporal_order():
    m = RF.build_r3_module("tf", None)
    with torch.no_grad():
        m.fusion.tok.weight.zero_(); m.fusion.tok.weight[:, 0, RF.TOK_P] = 1.0   # centre tap
    for t in (0, 4, 400, 1020):
        s = torch.zeros(1, 1, 1024); s[0, 0, t] = 1.0
        out = m.fusion.tok(s)[0, 0]
        assert out.shape == (256,) and int(out.argmax()) == t // 4 and float(out.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------- gate features
def test_gate_features_no_gt_edges_and_const_mode():
    assert list(inspect.signature(RF.gate_features).parameters) == ["s", "const"]
    assert list(inspect.signature(RF.R3Fusion.gate_values).parameters) == ["self", "s"]
    s = torch.rand(2, 1, 1024)
    f = RF.gate_features(s)
    assert f.shape == (2, 3, 1024) and torch.equal(f[:, 0], s[:, 0]) and float(f[0, 2, 0]) == 0.0
    assert float(f[0, 1, 0]) == pytest.approx(float(s[0, 0, :17].mean()), abs=1e-6)              # truncated edge mean
    assert float(f[0, 1, 500]) == pytest.approx(float(s[0, 0, 484:517].mean()), abs=1e-6)
    assert float(f[0, 2, 7]) == pytest.approx(abs(float(s[0, 0, 7] - s[0, 0, 6])), abs=1e-6)
    fc = RF.gate_features(s, const=True)
    assert torch.allclose(fc, fc[..., :1].expand_as(fc)) and torch.allclose(fc[:, :, 0], f.mean(-1), atol=1e-6)
    src = _code(ROOT / "src/ppg2ecg/flow/rhythm_fusion.py")
    gf = src[src.index("def gate_features"):src.index("def make_gate")]
    assert "ecg" not in gf.lower() and "detect_rpeaks" not in gf and "site" not in gf


def test_roll_is_256_samples():
    s = torch.arange(1024.0).reshape(1, 1, 1024)
    assert RF.PHASE_SHIFT_SAMPLES == 256 and float(RF.roll_scaffold(s)[0, 0, 256]) == 0.0


# ---------------------------------------------------------------- streams / driver statics
def test_three_tensor_loader_order_and_probe_hash_determinism():
    from torch.utils.data import DataLoader, TensorDataset
    N = 3000; x = torch.arange(N, dtype=torch.float32).reshape(N, 1)
    l2 = DataLoader(TensorDataset(x, x.clone()), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    l3 = DataLoader(TensorDataset(x, x.clone(), torch.arange(N)), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    for (a, _), (b, _, i), _ in zip(l2, l3, range(10)):
        assert torch.equal(a, b) and torch.equal(a.reshape(-1).long(), i)
    def run(seed):
        h = hashlib.sha256(); tr = torch.Generator().manual_seed(seed + 1); torch.manual_seed(seed)
        for k in range(4):
            t, r, _ = sample_tr(32, tr); RT.probe_update(h, torch.arange(k * 32, (k + 1) * 32), t, r, torch.randn(32, 1, 1024))
        return h.hexdigest()
    assert run(42) == run(42) != run(43)


def test_driver_statics_firewall_no_validation_oracle_only_in_its_branch():
    src = _code(ROOT / "src/ppg2ecg/training/train_r3_fusion.py")
    for bad in ("an0", "k2s", "fixed_imf_mse", "make_imf_banks", "gen_diag", "split['val']", "split['test']", "kjd", "ssx"):
        assert bad not in src, bad
    assert "assert_no_test_subjects(split['train'])" in src and "sample_tr_c1(Bc, tr_gen, arm='B', **TR_KW)" in src
    assert "torch.randn(Bc, 1, T, device=dev)" in src and "AdamW(net.r3.parameters()" in src
    body = src[src.index("def scaffold("):src.index("s_max.append")]
    assert "ecg" not in body and "y_t" not in body and body.count("oracle_t[idx_c]") == 1
    assert "scaffold_from_ppg(tcn, ppg_c.unsqueeze(1))" in body and "scaffold_from_ppg(tcn, x_t[partner_t[idx_c]].unsqueeze(1))" in body
    assert "if scaffold_kind == 'oracle':" in src and src.count("oracle_t = ") == 2
    assert "steps != RF.STEPS" in src and "R2_PROBE_HASH" in src
    for path, form in (("scripts/r3_prepare.py", "assert_no_test_subjects(VAL)"), ("scripts/r3_evaluate.py", "assert_no_test_subjects(VAL)"), ("scripts/r3_visual_atlas.py", "assert_no_test_subjects(V.VAL)")):
        s = _code(ROOT / path); assert form in s and "kjd" not in s and "ssx" not in s and "split['test']" not in s, path


def test_evaluator_imports_r2_scoring_verbatim_and_workers_can_pickle_it():
    import importlib.util, pickle, sys
    src = _code(ROOT / "scripts/r3_evaluate.py")
    assert "spec_from_file_location('r2_evaluate'" in src and "sys.modules[_spec.name] = R2E" in src
    assert "R2E.score(" in src and "R2E.paired(" in src and "R2E.subset_paired(" in src and "R2E.macro_rows(" in src
    for fn in ("def _score_chunk", "def _chance_chunk", "def macro_rows", "def score(", "def paired("):
        assert fn not in src
    assert "paired_subject_bootstrap(" not in src and RT.BOOT_N == RF.BOOT_N and RT.BOOT_SEED == RF.BOOT_SEED
    assert "n_boot=RF.BOOT_N, seed=RF.BOOT_SEED" in src and "default_rng(seed)" in src
    spec = importlib.util.spec_from_file_location("r2_evaluate", ROOT / "scripts/r2_evaluate.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    assert pickle.loads(pickle.dumps(mod._score_chunk)) is mod._score_chunk and pickle.loads(pickle.dumps(mod._peaks)) is mod._peaks

def test_frozen_constants():
    assert RF.STEPS == 2200 and RF.CKPT_STEPS == (0, 550, 1100, 2200) and RF.PREFLIGHT_STEPS == 100 and RF.BUDGET_GPU_HOURS == 6.0
    assert RF.LR == 1e-3 and RF.WEIGHT_DECAY == 0.01 and RF.BATCH == 64 and RF.MICRO_BATCH == 32 and RF.SEED == 42
    assert RF.BOOT_N == 2000 and RF.BOOT_SEED == 20260902 and RF.GATE_MIN_EFFECT == 0.02 and RF.GATE_BEATS_DEV_MAX == 0.20
    assert RF.NONINFERIORITY_MARGIN == -0.005 and RF.ORACLE_LIFT_MARGIN == 0.010
    assert RF.D_MODEL == 32 and RF.N_HEADS == 4 and (RF.TOK_K, RF.TOK_S, RF.TOK_P) == (7, 4, 3) and RF.GATE_POOL == 33
    assert set(RF.ORACLE_ARMS) == {"GTF-ORACLE", "ADD-ORACLE"}


# ---------------------------------------------------------------- shuffle manifest / decision
def test_r3_shuffle_manifest_matches_rule_when_present():
    p = ROOT / "artifacts/r3_rhythm_fusion/shuffle_manifest.csv"
    if not p.exists():
        pytest.skip("manifest absent")
    rows = list(csv.DictReader(open(p)))
    for pop, key, n in (("train", "partner_train_row", 293271), ("eval", "partner_pop_row", 2048), ("viz", "partner_pop_row", 64)):
        rs = [r for r in rows if r["population"] == pop]; assert len(rs) == n
        sub = np.array([r["subject"] for r in rs]); site = np.array([r["site"] for r in rs]); wi = np.array([int(r["window_index"]) for r in rs])
        partner = np.array([int(r[key]) for r in rs]); RT.assert_derangement(partner); assert np.array_equal(partner, RT.shuffle_partner(sub, site, wi))


def test_decision_is_total_and_narratives_true():
    keys = ("U1", "U2", "U3", "U4", "U5", "U6", "G1", "G2", "G3", "G4", "G6")
    n = 0
    for bits in itertools.product([True, False], repeat=11):
        rec = dict(zip(keys, bits))
        for ev_ci_tf, ev_ci_gtf, g5n, g5s, struct, prot_extra in itertools.product([True, False], repeat=6):
            rec["G5"] = g5n and g5s                                          # consistent record
            prot = struct and prot_extra
            rec2 = {**rec, "ev_tf": rec["U1"] and rec["U2"], "ev_gtf": rec["G1"] and rec["G2"], "ev_ci_tf": ev_ci_tf or (rec["U1"] and rec["U2"]),
                    "ev_ci_gtf": ev_ci_gtf or (rec["G1"] and rec["G2"]), "deg_tf": not rec["U4"] or not rec["U5"], "deg_gtf": not rec["G3"] or not rec["G4"],
                    "g5_noninferior": g5n, "g5_structure": g5s, "gtf_vs_tf_structure": struct, "gtf_vs_tf_protects": prot}
            out = RF.decide_verdict_r3(rec2); n += 1
            assert out["verdict"] in RF.VERDICTS
            if all(rec[f"U{i}"] for i in range(1, 7)):
                assert out["verdict"] == RF.VERDICTS[0] and out["prefers_tf_over_gtf"] == (not prot)
            elif all(rec[f"G{i}"] for i in range(1, 7)):
                assert out["verdict"] == RF.VERDICTS[1]
                assert out["necessity"] == ("SEPARATED" if (not rec["U4"] or struct) else "NOT_SEPARATED")
            elif out["verdict"] == RF.VERDICTS[2]:
                assert (rec2["ev_tf"] or rec2["ev_gtf"]) and all(rec2[f"deg_{x}"] for x in ("tf", "gtf") if rec2[f"ev_{x}"])
            else:
                assert out["verdict"] == RF.VERDICTS[3] and out["codes"] and "D_RESIDUAL_UNCLASSIFIED" not in out["codes"]
                assert ("D_NO_EVENT_GAIN" in out["codes"]) == (not rec2["ev_tf"] and not rec2["ev_gtf"])
                assert ("D_SUBTHRESHOLD" in out["codes"]) == ((rec2["ev_ci_tf"] and not rec2["ev_tf"]) or (rec2["ev_ci_gtf"] and not rec2["ev_gtf"]))
                for c, pred in (("D_TF_U3_FAIL", not rec["U3"]), ("D_TF_U6_CATASTROPHE", not rec["U6"]), ("D_GTF_G6_CATASTROPHE", not rec["G6"]),
                                ("D_GTF_G5_NONINFERIORITY_FAIL", not g5n), ("D_GTF_G5_STRUCTURE_VS_CONST_FAIL", not g5s)):
                    if c in out["codes"]:
                        assert pred, c
                for c in out["codes"]:
                    assert c in RF.RESIDUAL_CODES
    assert n == 2048 * 64
    for v_add in ("improves", "unresolved", "worsens"):
        for pt in (-0.01, 0.005, 0.02):
            for s4, s5 in itertools.product(("improves", "unresolved", "worsens"), repeat=2):
                assert RF.oracle_reading(v_add, pt, s4, s5) in RF.ORACLE_READINGS

def test_firewall_and_source_bank():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "ssx"))
    e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))
    assert hashlib.sha256(e0.numpy().tobytes()).hexdigest() == "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"


@pytest.mark.skipif(not (GEN.exists() and TCN.exists() and AN0.exists()), reason="frozen checkpoints / data absent")
def test_real_checkpoint_step0_parity_cpu():
    ck = torch.load(GEN, map_location="cpu", weights_only=False); cfg = ck["imf_cfg"]
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg["cond_mode"], h_scale=cfg["h_scale"]).eval(); base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    mod = RF.build_r3_module("gtf", "adaptive", c_hidden=int(ck["model_cfg"]["h_dim"]))
    net = RF.FusionMeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), mod, cond_mode=cfg["cond_mode"], h_scale=cfg["h_scale"]).eval()
    missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
    assert unexpected == [] and set(missing) == {"r3." + n for n in RF.GTF_PARAM_NAMES}
    net.requires_grad_(False)
    tcn, _ = RT.load_rhythm_tcn(TCN, torch.device("cpu"))
    X = np.load(AN0)["x"]; idx = ER.select_subset("x4-event-nfe-v2", "an0", len(X), 1024)[:2]
    pp = torch.from_numpy(X[idx].astype(np.float32)).unsqueeze(1); e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))[:2]
    with torch.no_grad():
        zb, _ = ER.sample_meanflow_schedule(base, pp, e0, ER.UNIFORM[2]); s = RT.scaffold_from_ppg(tcn, pp)
        for sc in (torch.zeros_like(s), s, s.flip(0)):
            z, _ = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, sc), e0, ER.UNIFORM[2]); assert torch.equal(zb, z)


def test_evaluator_variant_arms_map_to_their_own_source_modules():
    src = _code(ROOT / "scripts/r3_evaluate.py")
    assert "VARIANT_SRC = {'PHASE-TF': 'TF-TRUE', 'PHASE-GTF': 'GTF-TRUE', 'NODIRECT-TF': 'TF-TRUE', 'NODIRECT-GTF': 'GTF-TRUE'}" in src
    assert src.count("VARIANT_SRC[arm]") == 2 and "endswith('TF')" not in src


def test_evaluator_site_gate_bootstrap_indexes_the_site_masked_gate_array():
    src = _code(ROOT / "scripts/r3_evaluate.py")
    assert "G8m = G8[m]" in src and "float(G8m[idx].mean())" in src and "float(G8[idx].mean())" not in src
