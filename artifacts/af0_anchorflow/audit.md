# AF0 audit (2026-10-02) — before any AF0 training or AF-DEV metric

- **Closed lines untouched:** BF0, D0, C0, C0-A, E0, R1, SF0. AF0 imports the C0 / C0-A / SF0 / R1 modules unchanged.
- **Split (`split_manifest.json`):**

  | role | patients | windows |
  |---|---|---|
  | AF-TRAIN | 2,400 | 159,545 |
  | AF-DEV | 300 | 20,319 |
  | AF-LOCK | 337 | 22,133 |

  - All checks pass: disjoint; the union is SF-TRAIN; no SF-VAL, ARCH-VAL / HOLDOUT, or old V1 validation / test
    patient.
  - Not project-naive (earlier models trained on these patients).
- **Reused conventions:**
  - RD1 detector protocol;
  - SF0 WW-L1 (anchor) and SCALEFLOW-COUPLED (B1);
  - SF0 Haar;
  - SF0 noise map;
  - C0 / C0-A metric code and the R1 patient-macro rows.
- **Parameter matching (`baseline_configs.json`, count only):**

  | model | params |
  |---|---|
  | B2 | 597,727 (decoder width 63) |
  | B3 | 593,635 (width 50) |
  | B4 base | 598,483 (width 50) |
  | anchor | 593,577 |
  | B1 | 598,333 |

- **Decisions left open by the specification:**
  - the anchor is trained on the reference-R raster (SF0 protocol); μ for residual training uses the frozen detector's
    in-sample AF-TRAIN events, the same raster type as inference;
  - B2 uses the global waveform normalization (no bands exist);
  - base coupling = C1, the concatenative form validated in SF0;
  - every structural variant is re-matched to about 600k;
  - D7 diversity ratio = std(r_gen) / std(r_real) over AF-DEV windows;
  - the anchor shuffle adds the true μ to the residual generated under the wrong μ condition;
  - NFE variants do not count as new configurations (no training);
  - FD bootstrap via exact sufficient statistics, verified against `kanflow_fd`.
- **Synthetic dry run:** all stages from detector training to candidate evaluation ran in a scratch directory (30
  steps). The numbers mean nothing.
