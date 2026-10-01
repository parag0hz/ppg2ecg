# R1 old V1 TEST freshness audit

**NOT PERFORMED — no R1 candidate qualified on ARCH-VAL.**

- **Why it was skipped:** preregistration §9 runs this metadata-only audit only after a candidate qualifies.
  `candidate_selection.json` records `selected: NONE`, so the audit, the final-test freeze and the TEST evaluation were
  all skipped.
- **TEST was never touched:** no TEST waveform, outcome or metric was loaded by R1. `load_test()` was never called, and
  `final_test_freeze_manifest.json` was never created.
- **Disclosure from the preregistration:** earlier program phases (8/25–9/25) report V1 TEST metrics for earlier models.
  Whether those records meet the contamination rule was not evaluated, because the audit did not run. Any future use of
  the V1 TEST must run this audit first.

VERDICT: NOT PERFORMED
