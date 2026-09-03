# O1 — Preregistration Amendment 1 (verdict evaluation order)

**Frozen and pushed before any probe was trained and before any validation window was scored.**

| | |
|---|---|
| Amends | `docs/O1_ECG_COMPONENT_EXTRACTABILITY_PREREGISTRATION.md` §16, committed as `b972dea` |
| Discovered | while writing `tests/test_o1_component_extractability.py`, i.e. at commit-order step 7 — **before** step 9 (runtime preflight) and step 11 (probe training). No probe existed, no validation row had been scored, no metric had been computed |
| Changes | the *order* in which two verdicts are checked. **No threshold, no classification rule, no metric, no cohort and no target is changed.** |

## The defect

Frozen §16 states verdict **A** ("COMPONENT-WISE EXTRACTABILITY HETEROGENEITY SUPPORTED") requires

> ≥ 2 primary components classified STRONG/PARTIAL **and** ≥ 2 others classified RHYTHM/STATIC or NO CLEAR,

and that the verdicts are evaluated in the order A → B → C → D. Verdict **B** ("EXTRACTABILITY DOMINATED BY
RHYTHM / STATIC INFORMATION") is defined for the case where

> most morphology targets fail to beat B2 or SS-SHUFFLE while rhythm targets succeed.

Those two are not disjoint. With three rhythm targets (T1–T3) and six morphology targets (T4–T9), a result in
which **only the rhythm targets are extractable** satisfies A's literal criteria (high = 3 ≥ 2, low = 6 ≥ 2)
while being exactly the pattern B was written to name. Because A is tested first, B would be effectively
unreachable: it could only fire when fewer than two components are high or fewer than two are low.

## The amendment

Verdict **B is evaluated before verdict A** in, and only in, the case where the STRONG/PARTIAL set contains
**no morphology target** (none of T4–T9) and at least one rhythm target (T1–T3) is STRONG/PARTIAL, and more
morphology targets are low than high. In every other case the frozen order A → B → C → D is unchanged, as are
verdict C (checked first, unchanged) and verdict D (the residual bucket).

Implemented in `o1_targets.decide_o1`; the returned dict records `rhythm_only_high` and
`amendment_1_applied` so that any run states plainly whether the amended branch was taken. The report will
disclose the amendment and, if the amended branch is used, will also state what the unamended rule would have
returned.

## Why this is not result-dependent

No probe existed when this was written: `outputs/o1_*` was empty, `probe_metrics.csv` did not exist, and the
validation subjects had never been loaded by any O1 script. The change was forced by the structure of the rule
(a reachability defect), not by any observed number. The alternative — leaving the defect in place and
reporting an A verdict for a rhythm-only result — would have attached the wrong label to the most likely
outcome, which is the failure mode a preregistration exists to prevent.
