# DP1 — DualReadout-ECG final test — PREREGISTRATION DRAFT

**Status: DRAFT. Not frozen, not executed.**
- Written after DP0 CONFIRMED (`docs/DP0_DUALREADOUT_REPORT.md`).
- The old V1 TEST stays closed until a separate, committed and frozen DP1 preregistration exists and its freshness
  audit is clean.
- Nothing in this draft authorizes opening TEST.

## 1. Frozen from DP0 (to be hashed in the DP1 preregistration)

- **Architecture:** DualReadout S1 MIDDLE:
  - stem + blocks 1–6 shared;
  - per task: adapter + blocks 7–8;
  - point decoder width 71; flow decoder width 30;
  - Haar low-pass condition taps;
  - 943,372 parameters.
- **Training:** round-robin, one AdamW, 20,000 + 20,000 updates. Specialists P (WW-L1) and G (SCALEFLOW-COUPLED) with the
  DP0 protocol.
- **Detector:** RD1 / C0 protocol; events at threshold 0.35, refractory 32.
- **Inference and statistics:**
  - Euler NFE 8;
  - noise `scaleflow.window_noise`;
  - PPG-shuffle seed 20261002;
  - metrics and bootstrap (2,000 patient replicates, seed 20261002).
- **Gates and margins (unchanged):**

  | gate | rule |
  |---|---|
  | P1 | corr CI lower > −0.02 |
  | P2 | FP CI upper < +0.05 and recall CI lower > −0.01 |
  | G1 | FD CI upper < +1.0 |
  | CONDITION | FD CI > 0 and corr CI < 0 |
  | E1 | ≤ 0.85 · P_sep |

## 2. Open decisions for the frozen DP1 version (to be fixed before any TEST access)

- **Training population:**
  - (a) DP-TRAIN only, i.e. the DP0 models as frozen; or
  - (b) retrain on DP-TRAIN ∪ DP-DEV ∪ AF-LOCK with the frozen protocol. Option (b) uses more data but is no longer
    the exact confirmed artifact.
- **Seeds:**
  - several training seeds per model (for example 42 plus four predeclared seeds), the same for P, G and S1;
  - primary analysis either seed 42 (the confirmed models) or the seed-averaged effect, chosen before TEST;
  - seed variability is reported, especially for P2, whose DP-DEV and AF-LOCK upper bounds (0.0498 and 0.0484) were
    near the +0.05 margin.
- **Multiplicity:** whether all five gates must pass on TEST, as in DP0 (proposed: yes, no PARTIAL).

## 3. TEST freshness audit (metadata only, before any TEST array is read)

- Confirm from the repository history and every committed manifest that no model selection, threshold or preprocessing
  choice in BF0 … DP0 used old V1 TEST outcomes.
- Record any earlier TEST access with its commit; if one exists, document its scope.
- Verify that the TEST patient list is disjoint from every DP0 training, development and lock population.
- The audit result (CLEAN / NOT CLEAN) is committed before the final freeze.

## 4. Final freeze and single evaluation

- **Freeze:** a committed manifest hashes the preregistration, code, checkpoints (every seed), detector, splits and the
  audit. The TEST loader refuses without it.
- **Evaluation:** TEST (1,156 patients) is opened exactly once.
  - P, G and S1 are evaluated with the same gates.
  - There is no retraining, no candidate change and no return to earlier populations.
- **Report:** point and generative readouts separately (never a combined score), condition use, parameter efficiency,
  compute and K16 characterization.

## 5. Claim boundary (unchanged from DP0)

The final test can support at most this claim: one shared architecture preserves both specialist capabilities with
fewer parameters on the held-out TEST population.

It does not support:
- clinical validity or external generalization beyond VitalDB;
- calibrated uncertainty;
- state of the art or a first-of-its-kind claim.
