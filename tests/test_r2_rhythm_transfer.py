"""R2 tests — docs/R2_RHYTHM_SCAFFOLD_TRANSFER_PREREGISTRATION.md (f954e07) section 23.

Unconditional tests use a tiny random backbone; real-checkpoint / real-data tests skip when the files are
absent. No optimizer step on the real generator; no training.
"""
import ast
import copy
import hashlib
import inspect
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from ppg2ecg.evaluation import event_reliability as ER
from ppg2ecg.evaluation import rpeaks as R
from ppg2ecg.evaluation.paired_stats import paired_subject_bootstrap
from ppg2ecg.flow import rhythm_transfer as RT
from ppg2ecg.flow.imeanflow import MeanFlowS5, imeanflow_loss, sample_meanflow, sample_tr
from ppg2ecg.models import build_penguin_backbone

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / RT.GENERATOR_CKPT
TCN = ROOT / RT.RHYTHM_CKPT
AN0 = ROOT / "data/processed/wildppg_8s/an0.npz"
TINY = dict(n_step=1, sample_rate=128, h_dim=16, ssm_block_num=2, ssm_ratio=2.0, mlp_ratio=2.0)


def _tiny_pair():
    torch.manual_seed(0)
    bb = build_penguin_backbone(**TINY)
    for m in bb.modules():                       # adaLN-Zero init gates the PPG path to exactly zero; open it as a
        if isinstance(m, torch.nn.Linear) and float(m.weight.abs().sum()) == 0.0:   # trained checkpoint would
            torch.nn.init.normal_(m.weight, std=0.05); torch.nn.init.normal_(m.bias, std=0.05)
    base = MeanFlowS5(copy.deepcopy(bb)).eval()
    base.requires_grad_(False)                   # the frozen reference, as every evaluation script loads it
    net = RT.RhythmMeanFlowS5(copy.deepcopy(bb), TINY["h_dim"]).eval()
    net.backbone.requires_grad_(False)           # CPU conv kernels differ by 3e-8 when weights require grad;
    return base, net                             # the protocol compares frozen vs frozen (bit-exact)


def _code(path):
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


# ---------------------------------------------------------------- frozen sets / trainable set
def test_only_adapter_trainable_and_count_is_h_dim():
    _, net = _tiny_pair()
    assert RT.trainable_names(net) == ["rhythm_adapter.proj.weight"]
    assert RT.n_trainable(net) == TINY["h_dim"]
    RT.assert_only_adapter_trainable(net)
    net.backbone.final_layer.linear.weight.requires_grad_(True)
    with pytest.raises(RuntimeError):
        RT.assert_only_adapter_trainable(net)


def test_adapter_is_zero_init_1x1_conv_without_bias():
    a = RT.RhythmAdapter(128)
    assert isinstance(a.proj, torch.nn.Conv1d) and a.proj.kernel_size == (1,) and a.proj.bias is None
    assert a.proj.in_channels == 1 and a.proj.out_channels == 128 and torch.all(a.proj.weight == 0)
    assert sum(p.numel() for p in a.parameters()) == 128
    a2 = RT.RhythmAdapter(128)
    assert torch.equal(a.proj.weight, a2.proj.weight)                 # identical init across arms


def test_step0_equals_baseline_with_zero_and_real_scaffold_and_differs_when_nonzero():
    base, net = _tiny_pair()
    B, T = 3, 256
    z, ppg = torch.randn(B, 1, T), torch.randn(B, 1, T)
    t, h = torch.full((B, 1), 0.7), torch.full((B, 1), 0.3)
    ref = base.u(z, ppg, t, h)
    for s in (torch.zeros(B, 1, T), torch.rand(B, 1, T), torch.sigmoid(torch.randn(B, 1, T) * 3)):
        assert torch.equal(net.u(z, RT.make_ppg2(ppg, s), t, h), ref)
    with torch.no_grad():
        net.rhythm_adapter.proj.weight.copy_(torch.randn_like(net.rhythm_adapter.proj.weight) * 0.1)
    assert not torch.equal(net.u(z, RT.make_ppg2(ppg, torch.rand(B, 1, T)), t, h), ref)
    assert torch.equal(net.u(z, RT.make_ppg2(ppg, torch.zeros(B, 1, T)), t, h), ref)   # zero scaffold -> exact baseline


def test_sampler_and_loss_accept_ppg2_unchanged_with_finite_adapter_grads():
    base, net = _tiny_pair()
    B, T = 2, 256
    ppg, e, x = torch.randn(B, 1, T), torch.randn(B, 1, T), torch.randn(B, 1, T)
    s = torch.rand(B, 1, T)
    zb, kb = sample_meanflow(base, ppg, e, n_steps=4)
    zn, kn = sample_meanflow(net, RT.make_ppg2(ppg, s), e, n_steps=4)
    assert torch.equal(zb, zn) and kb == kn == 4
    with torch.no_grad():
        net.rhythm_adapter.proj.weight.fill_(0.05)
    t, r, _ = sample_tr(B, torch.Generator().manual_seed(1))
    loss, info = imeanflow_loss(net, x, RT.make_ppg2(ppg, s), e, t, r, norm_p=1.0, norm_eps=0.01, jvp_mode="forward")
    loss.backward()
    g = net.rhythm_adapter.proj.weight.grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    RT.assert_frozen_have_no_grad(net)


def test_load_state_dict_of_plain_checkpoint_misses_only_adapter():
    base, net = _tiny_pair()
    missing, unexpected = net.load_state_dict(base.state_dict(), strict=False)
    assert missing == ["rhythm_adapter.proj.weight"] and unexpected == []


# ---------------------------------------------------------------- streams
def test_three_tensor_loader_has_the_same_order_as_the_two_tensor_loader():
    N = 5000
    x = torch.arange(N, dtype=torch.float32).reshape(N, 1); y = x.clone(); idx = torch.arange(N)
    g1, g2 = torch.Generator().manual_seed(RT.SEED), torch.Generator().manual_seed(RT.SEED)
    l2 = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True, generator=g1)
    l3 = DataLoader(TensorDataset(x, y, idx), batch_size=64, shuffle=True, generator=g2)
    for (a, _), (b, _, i), _ in zip(l2, l3, range(20)):
        assert torch.equal(a, b) and torch.equal(a.reshape(-1).long(), i)


def test_probe_hash_is_deterministic_and_seed_sensitive():
    def run(seed):
        h = hashlib.sha256(); tr = torch.Generator().manual_seed(seed + 1); torch.manual_seed(seed)
        for k in range(4):
            t, r, _ = sample_tr(32, tr); e = torch.randn(32, 1, 1024)
            RT.probe_update(h, torch.arange(k * 32, (k + 1) * 32), t, r, e)
        return h.hexdigest()
    assert run(42) == run(42) and run(42) != run(43)


def test_scaffold_function_takes_ppg_only_and_no_ecg():
    assert list(inspect.signature(RT.scaffold_from_ppg).parameters) == ["tcn", "ppg"]
    src = _code(ROOT / "src/ppg2ecg/flow/rhythm_transfer.py")
    fn = src[src.index("def scaffold_from_ppg"):src.index("def _oracle_one")]
    assert "ecg" not in fn.lower() and "detect_rpeaks" not in fn
    assert "detect_rpeaks" in src[src.index("def _oracle_one"):src.index("def shuffle_key")]   # only the ORACLE path


def test_driver_never_reads_validation_or_selection_and_has_the_firewall():
    src = _code(ROOT / "src/ppg2ecg/training/train_r2_adapter.py")
    for bad in ("an0", "k2s", "fixed_imf_mse", "make_imf_banks", "gen_diag", "split['val']", 'split["val"]', "kjd", "ssx"):
        assert bad not in src, bad
    assert "assert_no_test_subjects(split['train'])" in src and "sample_tr_c1(Bc, tr_gen, arm='B', **TR_KW)" in src
    assert "torch.randn(Bc, 1, T, device=dev)" in src and "AdamW(net.rhythm_adapter.parameters()" in src
    for path in ("scripts/r2_build_caches.py", "scripts/r2_evaluate.py", "scripts/r2_visual_atlas.py"):
        s = _code(ROOT / path)
        assert "assert_no_test_subjects" in s and "kjd" not in s and "ssx" not in s, path


def test_frozen_constants():
    assert RT.STEPS == 2200 and RT.CKPT_STEPS == (0, 550, 1100, 2200) and RT.PREFLIGHT_STEPS == 100
    assert RT.LR == 1e-3 and RT.WEIGHT_DECAY == 0.01 and RT.BATCH == 64 and RT.MICRO_BATCH == 32 and RT.SEED == 42
    assert RT.BOOT_N == 2000 and RT.BOOT_SEED == 20260902 and RT.GATE_MIN_EFFECT == 0.02 and RT.GATE_BEATS_DEV_MAX == 0.20
    assert RT.PHASE_SHIFT_SAMPLES == 256 and RT.PERSIST_TOL_MS == 250.0 and RT.SHUFFLE_SALT == "r2-rhythm-shuffle-v1"


# ---------------------------------------------------------------- shuffle
def test_shuffle_is_bijective_fixed_point_free_on_all_stratum_sizes():
    subs, sites, wi = [], [], []
    for k, (sub, site, n) in enumerate((("a", "sternum", 2), ("a", "head", 3), ("b", "sternum", 8), ("b", "wrist", 251), ("c", "ankle", 5546))):
        subs += [sub] * n; sites += [site] * n; wi += list(range(n))
    p = RT.shuffle_partner(subs, sites, wi)
    RT.assert_derangement(p)
    subs, sites = np.array(subs), np.array(sites)
    assert np.all(subs[p] == subs) and np.all(sites[p] == sites)          # partner stays in the stratum


def test_shuffle_singleton_stratum_stops_and_duplicates_raise():
    with pytest.raises(RuntimeError):
        RT.shuffle_partner(["a", "a", "b"], ["s", "s", "s"], [1, 2, 3])
    with pytest.raises(RuntimeError):
        RT.shuffle_partner(["a", "a"], ["s", "s"], [1, 1])


def test_shuffle_is_salt_dependent_and_order_invariant():
    subs, sites, wi = ["a"] * 40, ["s"] * 40, list(range(100, 140))
    p1 = RT.shuffle_partner(subs, sites, wi); p2 = RT.shuffle_partner(subs, sites, wi, salt="other")
    assert not np.array_equal(p1, p2)
    perm = np.random.default_rng(0).permutation(40)
    p3 = RT.shuffle_partner(subs, sites, np.asarray(wi)[perm])
    for i in range(40):                                                   # same window -> same partner window
        assert wi[p1[i]] == np.asarray(wi)[perm][p3[np.flatnonzero(perm == i)[0]]]


# ---------------------------------------------------------------- ablation primitives
def test_roll_scaffold_is_a_plus256_circular_shift():
    s = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 1024)
    r = RT.roll_scaffold(s)
    assert float(r[0, 0, 256]) == 0.0 and float(r[0, 0, 0]) == 768.0
    assert np.array_equal(r.numpy(), np.roll(s.numpy(), 256, axis=-1))


def test_phase_phi_and_strata():
    assert RT.phase_phi(np.arange(0, 1024, 128)) == pytest.approx(0.0) and RT.phi_stratum(0.0) == "in_phase"
    assert RT.phase_phi(np.arange(0, 1024, 100)) == pytest.approx(0.56) and RT.phi_stratum(0.56) == "anti_phase"
    assert RT.phi_stratum(0.25) == "rest" and RT.phi_stratum(0.95) == "in_phase" and RT.phi_stratum(float("nan")) == "undefined"
    assert np.isnan(RT.phase_phi([100]))


def test_persistence_matcher_is_one_to_one_at_250ms():
    gt = np.array([100, 300])
    d = RT.persistence_deltas(gt, {1: np.array([110, 700]), 2: np.array([]), 4: np.array([101, 305, 306])})
    assert d.shape == (2, 3)
    assert d[0, 0] == pytest.approx(10 / 128 * 1000) and np.isnan(d[1, 0])
    assert np.isnan(d[:, 1]).all()
    assert d[0, 2] == pytest.approx(1 / 128 * 1000) and d[1, 2] == pytest.approx(5 / 128 * 1000)
    d2 = RT.persistence_deltas(np.array([100, 120]), {4: np.array([110])})            # one pred cannot serve two GT beats
    assert np.isnan(d2).sum() == 1


# ---------------------------------------------------------------- verdict
def test_verdict_is_total_and_exactly_one():
    V = ("improves", "unresolved", "worsens")
    for items in itertools.product([True, False], repeat=5):
        for v_OB, v_TB, v_OT in itertools.product(V, V, V):
            rec = {f"item{i+1}": items[i] for i in range(5)}
            rec |= {"v_OB": v_OB, "v_TB": v_TB, "v_OT": v_OT, "v_SB": "unresolved", "item1_point": 0.01, "item1_ci_positive": True}
            verdict, reason = RT.decide_verdict(rec)
            assert verdict in RT.VERDICTS
            i1, i2, i3, i4, i5 = items
            if all(items):
                assert verdict == RT.VERDICTS[0]
            elif i1 and i2 and not i4:
                assert verdict == RT.VERDICTS[1]
            elif v_OB == "improves" and v_OT == "improves":
                assert verdict == RT.VERDICTS[2]
            else:
                assert verdict == RT.VERDICTS[3] and reason
            assert RT.oracle_case(v_OB, v_TB, v_OT) in ("case1", "case2", "case3", "case4", "other")


def test_residual_reason_never_claims_true_lost_when_it_won():
    rec = dict(item1=True, item2=True, item3=False, item4=True, item5=True, v_OB="improves", v_TB="improves", v_OT="unresolved",
               v_SB="unresolved", item1_point=0.05, item1_ci_positive=True)
    verdict, reason = RT.decide_verdict(rec)
    assert verdict == RT.VERDICTS[3] and "does not beat" not in reason and "item 3" in reason


def test_paired_bootstrap_orientation_seed_and_n():
    subj = np.array(["an0"] * 50 + ["k2s"] * 50)
    a, b = np.zeros(100), np.ones(100)
    r = paired_subject_bootstrap(a, b, subj, "higher_better", n_boot=RT.BOOT_N, seed=RT.BOOT_SEED)
    assert r["verdict"] == "improves" and r["seed"] == 20260902 and r["n_boot"] == 2000
    r2 = paired_subject_bootstrap(b, a, subj, "lower_better", n_boot=RT.BOOT_N, seed=RT.BOOT_SEED)
    assert r2["verdict"] == "improves" and r2["point"] == pytest.approx(1.0)


# ---------------------------------------------------------------- frozen population / bank
def test_source_bank_seed0_sha256():
    e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))
    assert hashlib.sha256(e0.numpy().tobytes()).hexdigest() == "868085798050102eb815e1700c8e9edb4cb9e740a314407ee1e471a99419160f"


def test_frozen_subset_matches_nfe_subset_json():
    p = ROOT / "artifacts/x4_0_event_reliability/nfe_subset.json"
    if not p.exists():
        pytest.skip("frozen subset artifact absent")
    frozen = json.loads(p.read_text())
    for s, n in (("an0", 22183), ("k2s", 27017)):
        assert ER.select_subset("x4-event-nfe-v2", s, n, 1024).tolist() == list(frozen[s])


def test_firewall():
    with pytest.raises(ER.WildPPGTestFirewallError):
        ER.assert_no_test_subjects(("an0", "kjd"))


# ---------------------------------------------------------------- real checkpoint (skipped without files)
@pytest.mark.skipif(not (GEN.exists() and TCN.exists() and AN0.exists()), reason="frozen checkpoints / data absent")
def test_real_checkpoint_parity_and_hashes_cpu():
    dev = torch.device("cpu")
    net, ck, gmeta = RT.load_generator(GEN, dev)
    tcn, tmeta = RT.load_rhythm_tcn(TCN, dev)
    assert gmeta["state_dict_sha256"] == RT.EXPECTED_GENERATOR_STATE_SHA and gmeta["round"] == 45 and gmeta["h_dim"] == 128
    assert tmeta["state_dict_sha256"] == RT.EXPECTED_RHYTHM_STATE_SHA and tmeta["params"] == 328897
    assert RT.n_trainable(net) == 128
    cfg = ck["imf_cfg"]
    base = MeanFlowS5(build_penguin_backbone(**ck["model_cfg"]), cond_mode=cfg["cond_mode"], h_scale=cfg["h_scale"]).eval()
    base.load_state_dict(ck["state_dict"]); base.requires_grad_(False)
    d = np.load(AN0); X = d["x"]
    idx = ER.select_subset("x4-event-nfe-v2", "an0", len(X), 1024)[:2]
    pp = torch.from_numpy(X[idx].astype(np.float32)).unsqueeze(1)
    e0 = torch.randn(2048, 1, 1024, generator=torch.Generator().manual_seed(0))[:2]
    with torch.no_grad():
        zb, _ = ER.sample_meanflow_schedule(base, pp, e0, ER.UNIFORM[2])
        s = RT.scaffold_from_ppg(tcn, pp)
        z0, _ = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, torch.zeros_like(pp)), e0, ER.UNIFORM[2])
        z1, _ = ER.sample_meanflow_schedule(net, RT.make_ppg2(pp, s), e0, ER.UNIFORM[2])
    assert torch.equal(zb, z0) and torch.equal(zb, z1)
    assert s.shape == pp.shape and float(s.min()) >= 0 and float(s.max()) <= 1


# ---------------------------------------------------------------- review additions
def test_evaluator_passes_bootstrap_seed_and_n_explicitly():
    src = _code(ROOT / "scripts/r2_evaluate.py")
    n_calls = src.count("paired_subject_bootstrap(")
    n_pinned = src.count("n_boot=RT.BOOT_N, seed=RT.BOOT_SEED")
    assert n_calls >= 2 and n_pinned == n_calls, (n_calls, n_pinned)
    assert "default_rng(RT.BOOT_SEED)" in src


def test_firewall_calls_are_exact_and_no_test_split_indexing():
    forms = {"src/ppg2ecg/training/train_r2_adapter.py": "assert_no_test_subjects(split['train'])",
             "scripts/r2_build_caches.py": "assert_no_test_subjects(list(split['train']) + list(VAL))",
             "scripts/r2_evaluate.py": "assert_no_test_subjects(VAL)", "scripts/r2_visual_atlas.py": "assert_no_test_subjects(V.VAL)"}
    for path, form in forms.items():
        src = _code(ROOT / path)
        assert form in src, path
        assert "split['test']" not in src and 'split["test"]' not in src, path


def test_driver_oracle_field_only_in_oracle_branch_and_scaffold_never_sees_ecg():
    src = _code(ROOT / "src/ppg2ecg/training/train_r2_adapter.py")
    body = src[src.index("def scaffold("):src.index("s_max.append")]
    assert "ecg" not in body and "y_t" not in body
    assert "scaffold_from_ppg(tcn, ppg_c.unsqueeze(1))" in body and "scaffold_from_ppg(tcn, x_t[partner_t[idx_c]].unsqueeze(1))" in body
    assert body.count("oracle_t[idx_c]") == 1
    assert src.count("oracle_t = ") == 2                                     # None init + the oracle branch load only
    assert "if a.arm == 'oracle':" in src


def test_score_and_chance_chunks_match_c0_verbatim():
    import importlib.util
    def load(path):
        spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    c0, r2 = load("scripts/analyze_c0_compression_target.py"), load("scripts/r2_evaluate.py")
    def fn_dump(mod, name):
        tree = ast.parse(Path(mod.__file__).read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        return ast.dump(node)
    assert fn_dump(c0, "_chance_chunk") == fn_dump(r2, "_chance_chunk")
    assert fn_dump(c0, "_peaks") == fn_dump(r2, "_peaks")
    assert c0.PRIMARY == r2.PRIMARY and c0.RATIO_RAW == r2.RATIO_RAW
    # synthetic ECG-like windows: sharp pulses every 96 samples plus noise; shared keys must agree exactly
    rng = np.random.default_rng(3)
    tt = np.arange(1024)
    def ecg_like(phase, amp=1.0):
        x = np.zeros(1024)
        for r0 in range(phase, 1024, 96):
            x += amp * np.exp(-((tt - r0) ** 2) / (2 * 2.0 ** 2)) - 0.3 * amp * np.exp(-((tt - r0 - 6) ** 2) / (2 * 3.0 ** 2))
        return x + 0.02 * rng.standard_normal(1024)
    gt = np.stack([ecg_like(40), ecg_like(60), ecg_like(20)])
    pred = np.stack([ecg_like(43, 0.9), ecg_like(60, 0.5) + 0.1 * rng.standard_normal(1024), np.zeros(1024)])
    gt_pk = [R.detect_rpeaks(g, 128) for g in gt]
    rows_c0 = c0._score_chunk((pred, gt, gt_pk)); rows_r2, _ = r2._score_chunk((pred, gt, gt_pk))
    for a, b in zip(rows_c0, rows_r2):
        for k in a:
            va, vb = a[k], b[k]
            assert (np.isnan(va) and np.isnan(vb)) or va == vb, k
    pk = [R.detect_rpeaks(p_, 128) for p_ in pred]
    assert c0._chance_chunk((gt_pk, pk)) == r2._chance_chunk((gt_pk, pk))


def test_real_shuffle_manifest_matches_rule_on_all_populations():
    p = ROOT / "artifacts/r2_rhythm_transfer/shuffle_manifest.csv"
    if not p.exists():
        pytest.skip("shuffle manifest absent")
    import csv as _csv
    rows = list(_csv.DictReader(open(p)))
    expect = {"train": (293271, 48), "eval": (2048, 8), "viz": (64, 8)}
    for pop, (n, n_strata) in expect.items():
        rs = [r for r in rows if r["population"] == pop]
        assert len(rs) == n, (pop, len(rs))
        sub = np.array([r["subject"] for r in rs]); site = np.array([r["site"] for r in rs]); wi = np.array([int(r["window_index"]) for r in rs])
        assert len(set(zip(sub, site))) == n_strata
        key = "partner_train_row" if pop == "train" else "partner_pop_row"
        partner = np.array([int(r[key]) for r in rs])
        RT.assert_derangement(partner)
        assert np.all(sub[partner] == sub) and np.all(site[partner] == site)
        assert np.array_equal(partner, RT.shuffle_partner(sub, site, wi))
        assert all(int(rs[int(partner[i])]["array_pos"]) == int(r["partner_array_pos"]) for i, r in enumerate(rs))
