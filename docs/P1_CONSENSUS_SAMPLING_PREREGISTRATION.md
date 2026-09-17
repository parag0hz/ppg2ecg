# P1 — Consensus sampling on validation subjects (preregistration)

Frozen before any P1 number exists. Evaluation only: no weight update, no test subject.

| | |
|---|---|
| Question | Does pooling K cheap iMF samples at the event level reduce HR error vs one iMF sample, and where does it land against one PENGUIN (OT-CFM) 50-NFE sample? |
| Data | U2 **validation** subjects only: DaLiA `subject8`,`subject10`; WildPPG `e61`,`qm9`. Test subjects never loaded; WildPPG `kjd`/`ssx` asserted absent. 2,048 windows per subject by exact `linspace` (4 s, 128 Hz). |
| Checkpoints | U2 `checkpoint_last.pt` (both arms at exactly 14,000 steps), unchanged. |
| Samples | Noise seeds 0..15, shared across arms. PENGUIN NFE 50 (Heun 25) seeds 0..3; PENGUIN NFE 1 and iMF NFE 1, 2 seeds 0..15. |
| Single-sample score | Per-window metric averaged over seeds 0..3 (= the U2 definition). |
| Consensus HR (primary) | Median over the K samples of each sample's detected HR (NaN skipped) → abs error vs reference HR. |
| Consensus peaks (secondary) | Every sample's detected R peaks → vote density (Gaussian σ = 20 ms, one vote per peak) → local maxima ≥ K/2 votes, ≥ 250 ms apart → F1 @ 50 ms vs reference peaks. |
| K | 4 and 16 (first K seeds). PENGUIN-50 consensus only at K = 4 (200 NFE, cost reference). |
| Reported | Subject-macro HR error and R-peak F1 per (arm, NFE, K), and NFE cost. Two subjects per corpus: descriptive, no CI, no superiority claim. |
| GO rule | On **both** corpora, iMF (NFE 1 or 2) at K = 16 lowers subject-macro HR error by ≥ 10 % relative to the same NFE single-sample score, and consensus F1 is not lower than single-sample F1 by more than 0.01. Otherwise NO-GO for consensus sampling. |
| Also stated, not a gate | Whether iMF K = 16 (16 or 32 NFE) is below PENGUIN-50 single-sample HR error. |
| Fixed | Detector `neurokit`, tolerance 50 ms, all rules above. Nothing is tuned after results. |
| Outputs | `artifacts/p1_consensus/` (JSON/CSV); samples as float16 npz in `outputs/p1_consensus/` (not in git) for P2. |

## Amendment 1 (before any P1 number; frozen document above left unchanged)

The consensus-peak rule above uses a Gaussian of σ = 20 ms scaled so one vote peaks at 1.0. Sample-to-sample
R-peak spread is ≈ 50 ms (X4-0), so 16 agreeing samples spread over ±50 ms reach only ≈ 6 < K/2 and the rule
would reject beats every sample produced. Found by reading the rule against X4-0, not from any P1 output.
**Replacement:** count the samples' peaks inside ±50 ms (the F1 tolerance) of each position; keep local maxima
with ≥ K/2 votes, ≥ 250 ms apart; position = median of the votes inside that ±50 ms box. Everything else,
including the GO rule (which is on HR, not on this rule), is unchanged.
