"""U2 §6 gate: `--max-steps` must give an exact, corpus-independent optimizer-step budget,
and must be a no-op at its default so every frozen run is unaffected.

Run BEFORE any U2 training. Protocol: docs/U2_PAIRED_IMEANFLOW_VS_PENGUIN_PREREGISTRATION.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ppg2ecg.training import train_a0, train_a2

ROOT = Path(__file__).resolve().parents[1]
SRC = {
    "train_a0": (ROOT / "src/ppg2ecg/training/train_a0.py").read_text(),
    "train_a2": (ROOT / "src/ppg2ecg/training/train_a2.py").read_text(),
}
BREAK_BLOCK = (
    '                if args.max_steps and state["opt_steps"] >= args.max_steps:\n'
    '                    break  # U2 §6: exact, corpus-independent step budget\n'
)
STOP_TERM = ' or bool(args.max_steps and state["opt_steps"] >= args.max_steps)'


# ------------------------------------------------------------------ the flag is a no-op by default
@pytest.mark.parametrize("mod", [train_a0, train_a2])
def test_flag_defaults_to_none(mod):
    assert mod.parse_args(["--out-dir", "/tmp/u2"]).max_steps is None
    assert mod.parse_args(["--out-dir", "/tmp/u2", "--max-steps", "14000"]).max_steps == 14000


@pytest.mark.parametrize("name", sorted(SRC))
def test_both_guards_short_circuit_on_the_default(name):
    """`args.max_steps and ...` means None never reaches the comparison, so the default path
    executes exactly the pre-U2 sequence of optimizer steps."""
    src = SRC[name]
    assert src.count(BREAK_BLOCK) == 1, "the break block must appear exactly once"
    assert src.count(STOP_TERM) == 1, "the stop term must appear exactly once"
    for frag in ('if args.max_steps and state["opt_steps"]', 'bool(args.max_steps and state["opt_steps"]'):
        assert frag in src


def test_the_new_lines_are_textually_identical_in_both_trainers():
    """Compute matching is only meaningful if the two arms stop by the same rule."""
    assert BREAK_BLOCK in SRC["train_a0"] and BREAK_BLOCK in SRC["train_a2"]
    assert STOP_TERM in SRC["train_a0"] and STOP_TERM in SRC["train_a2"]
    arg_a0 = [l for l in SRC["train_a0"].splitlines() if '"--max-steps"' in l]
    arg_a2 = [l for l in SRC["train_a2"].splitlines() if '"--max-steps"' in l]
    assert len(arg_a0) == len(arg_a2) == 1 and arg_a0[0] == arg_a2[0]


@pytest.mark.parametrize("name", sorted(SRC))
def test_counter_increments_once_per_optimizer_step(name):
    src = SRC[name]
    assert src.count('state["opt_steps"] += 1') == 1
    assert src.count("                opt.step()\n") + src.count("model.optimize(") >= 1
    assert '"opt_steps": state["opt_steps"]' in src
    assert '"max_steps": args.max_steps' in src


@pytest.mark.parametrize("name", sorted(SRC))
def test_stop_reason_is_recorded(name):
    assert 'max_steps({args.max_steps})' in SRC[name]


# ------------------------------------------------------------------ the budget is exact
def realised_steps(n_train, batch=64, spr=220, rounds=100000, max_steps=None):
    """Faithful simulation of `batch_rounds` (round ends at min(steps left in epoch, spr))
    plus the U2 break."""
    per_epoch = -(-n_train // batch)
    it, total = 0, 0
    for _ in range(rounds):
        take = min(spr, per_epoch - it)
        if max_steps is not None:
            take = min(take, max_steps - total)
        if take <= 0:
            break
        total += take
        it = 0 if it + take >= per_epoch else it + take
        if max_steps is not None and total >= max_steps:
            break
    return total


def test_without_the_flag_the_budget_is_corpus_dependent():
    """The failure mode U2 exists to avoid — C2's measured case, reproduced."""
    assert realised_steps(293271, rounds=66) == 14409 != 66 * 220
    # a corpus whose epoch is shorter than a round loses steps every round
    assert realised_steps(4_400, rounds=66) < 66 * 220


@pytest.mark.parametrize("n_train", [4_400, 18_700, 27_400, 240_000, 527_000, 592_000])
def test_with_the_flag_the_budget_is_exactly_n_on_every_corpus(n_train):
    assert realised_steps(n_train, max_steps=14_000) == 14_000


def test_budget_is_independent_of_round_length():
    for spr in (55, 110, 220, 440):
        assert realised_steps(592_000, spr=spr, max_steps=14_000) == 14_000
