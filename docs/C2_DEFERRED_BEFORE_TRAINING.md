# C2 — Deferred Before Any Weight Update

**Status: C2 training was intentionally deferred. No C2 weight update ever occurred.**

The C2 preregistration `docs/C2_COMPUTE_MATCHED_MULTISEED_INTERVAL_PREREGISTRATION.md`
(commit `f5120f9`) and its implementation (commit `877841d`) **remain frozen and unmodified**. This note
records the deferral; it does not amend the preregistration.

## What was completed

Everything up to but excluding training:

- the repository audit and the compute-budget arithmetic correction (66 rounds = **14,409** realised
  optimiser steps, not 14,520, because `batch_rounds` truncates three rounds at epoch boundaries);
- the per-seed RNG-control preflight, which **passed for all five seeds** — within each seed the three
  arms share model-init, window-order, Gaussian-noise and validation-bank hashes while their (t, r)
  hashes all differ, and across seeds the shared streams all differ;
- the sampler-exposure Monte Carlo, reproducing P(h ≥ 0.5) = 0.0422 / 0.0210 / 0.2710 for B / H25 / H50
  with P(h = 0) = 0.5000 preserved in every arm;
- the metadata-only visual-atlas cohort, 8 windows in each of 8 subject × site strata.

These are all **read-only or forward-only**: not one optimiser step was taken.

## Verified absence of any C2 weight update

At commit `877841d`, checked before writing this note:

| check | result |
|---|---|
| C2 output directory (`outputs/*c2*`) | **none** |
| any checkpoint file matching `*c2*` | **none** |
| any run directory for seeds 40 / 41 / 43 / 44 | **none** |
| `opt_steps` present in any `training_summary.json` | **none** (the counter was added at `877841d`, so any post-`877841d` training run would record it) |
| live `train_a2` process | **none** |
| `artifacts/c2_compute_matched_multiseed/` contents | `rng_control.json`, `sampler_exposure.csv` only — both produced by the read-only preflight |
| C1 output directories | unmodified since 2026-09-01/02 |
| frozen A4 checkpoint md5 | `31c042d291052fbb6dc15263ad316be2`, unchanged |

**No C2 hypothesis was evaluated. Neither H1 nor H2 was tested. No C2 gate fired.**

## Reason for the deferral

C2 is a 15-run replication costing roughly **54 GPU-hours** on the single RTX 5090. It is postponed until
the C1 structural phenomenon is better characterised, because C2 would spend that budget replicating an
effect whose *nature* is not yet established: C1 showed H50 improving aggregate QRS-energy and
peak-to-peak calibration, but did not distinguish a genuine local QRS-structure improvement from a broad
fitting gain or a purely aggregate calibration effect.

`docs/M1_C1_STRUCTURAL_MECHANISM_AUDIT_PREREGISTRATION.md` performs that triage on the **existing** C1
checkpoints with no new training. Its outcome determines whether a compute-matched replication is worth
running at all, and in what form.

## Resumption

C2 may be resumed later against its frozen preregistration `f5120f9` without amendment. If it is resumed,
the deferral and its reason must be disclosed in the C2 report, since the decision to run C2 will by then
have been informed by M1.
