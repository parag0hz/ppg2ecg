# DP1 — old V1 TEST freshness audit

> **VERDICT: NOT FRESH — DP1 TEST NOT FRESH.**
> - Committed records show that old V1 TEST outcomes were used for **performance reporting** (at least 17 result
>   reports) and for **hypothesis generation** that DP0 / DP1 inherit.
> - The TEST outcome of the exact timing-detector family and convention used by DP0 / DP1 was computed and reported.
> - Per DP1 spec §5 the final TEST is **not run**, and DP1 stops here. No DP1 preregistration, seed 43 / 44 training,
>   DP-DEV multi-seed evaluation, compute baseline or final freeze was performed.
>
> Machine-readable record: `artifacts/dp1_dualreadout/test_freshness_audit.json`.

## 1. Scope and method

- **Audited population:** the old V1 TEST.
  - Source: `data/manifests/split_v1_vitaldb_seed42.json`, `splits[0]["test"]`.
  - Size: 1,224 cases, 1,156 patients (counts from the manifest only). The records report 19,543 windows.
- **Metadata only.** No TEST ECG / PPG waveform, TEST array or cached TEST output was loaded or opened. The audit read:
  - the git history (268 commits);
  - committed reports, preregistrations and design documents;
  - artifact and output file names;
  - the split manifest (patient / case counts).
- **Pre-audit check.** The repository was verified at `724b883` (= origin/main, clean tracked tree).
  - The DP0 prereg (`b4247f7`) and lock freeze (`fc39649`) are ancestors of HEAD.
  - The DP0 lock-freeze hashes still match.
  - The frozen S1 reproduces exactly: 943,372 total; 246,720 shared; 8,320 adapter parameters; saving 20.85 %;
    widths 71 / 30.
  - The seed-42 detector / P / G / S1 checkpoints match their DP0 hashes.
- **Rules applied:**
  - **DP1 spec §5:** prior use of TEST outcomes for architecture selection, loss design, checkpoint selection,
    threshold tuning, metric selection, hypothesis generation or performance reporting means NOT FRESH.
  - **The rule frozen in SF0 §11** (`dbf60d7`) states that TEST is NOT CLEAN if committed records show V1 TEST
    outcomes that were:
    - (a) computed for any model, detector or convention the line reuses, explicitly including "the RD1 / C0 detector
      family with its threshold / refractory / target convention"; or
    - (b) cited as motivation for a design choice or hypothesis of the line or of the closed line it follows.

## 2. Findings by category

| category | status | evidence |
|---|---|---|
| performance reporting | **FOUND** | at least 17 committed result reports on the V1 TEST (§3) |
| hypothesis generation | **FOUND** | the RD1 TEST result and the BF0 design document (§4) |
| architecture selection | **FOUND (indirect)** | the discriminative event stage inherited by DP0 was motivated with TEST numbers (§4). No DP0 sharing pattern, width or adapter was chosen on TEST |
| threshold tuning | NOT FOUND | threshold 0.35 / refractory 32 come from the R1 rhythm probe (`71aefb4`, 2026-09-02, before VitalDB entered the project on 2026-09-17). The detector with this convention was later evaluated on TEST (rule (a)) |
| checkpoint selection | NOT FOUND | BF0 … DP0 use last-step checkpoints |
| loss design | NOT FOUND for DP0 | DP0 uses L1 and the SF0 FM MSE. LW1 evaluated auxiliary losses on TEST; none is used by DP0 |
| metric selection | NOT ESTABLISHED | the metric suite predates the VitalDB split. FBC1 selected against legacy-test-derived cells; this is not used by DP0 |

**SF0 rule (a): triggered.** RD1 (`02628f3`) trained the RhythmTCN detector and reported it on the V1 TEST:
- detector: 328,897 parameters, Gaussian σ = 20 ms targets, threshold 0.35, refractory 32;
- reported result: F1 0.7725, RR-MAE 7.67 ms on 1,156 patients / 19,543 windows.

DP0 and DP1 reuse exactly this detector family and convention (`artifacts/dp0_dualreadout/detector_config.json`).

**SF0 rule (b): triggered.** The BF0 design document (`504395d`, `docs/BEAT_FIRST_ARCHITECTURE_DESIGN_KO.md`) bases
the discriminative event stage on V1 TEST numbers. DP0 descends from that closed chain:
BF0 → C0 → C0-A → R1 → SF0 → AF0 → DP0.

## 3. Performance reporting on the V1 TEST (committed records; verbatim excerpts)

| commit | record | excerpt |
|---|---|---|
| `5d3d274` | `docs/V1_VITALDB_PAIRED_REPORT.md` | "Test = 1,156 patients / 1,224 cases / 19,543 windows" |
| `4507165` | `docs/SR1_SEED_REPLICATION_REPORT.md` | "Nine evaluations on the V1 test set" |
| `4507165` | `docs/PZ3_PER_PATIENT_COMPARISON_REPORT.md` | "V1 test: 1,156 patients" |
| `1d53ccc` | `docs/BB1_BACKBONE_SENSITIVITY_REPORT.md` | "Test = 1,156 VitalDB patients / 19,543 windows" |
| `547f062` | `docs/RF1_REFLOW_REPORT.md` | "Evaluated with the SR1 protocol on the V1 test set" |
| `70206eb` | `docs/AB1_CONSENSUS_ABLATION_REPORT.md` | "V1 VitalDB test (1,156 patients, 19,543 windows)" |
| `f596b79` | `docs/ED1_EVENT_CONSENSUS_DECODING_REPORT.md` | "VitalDB V1 test: 1,156 patients, 19,543 windows" |
| `f777a16` | `docs/LW1_WEIGHTED_AUX_LOSS_REPORT.md` | "Test = 1,156 patients / 19,543 windows" |
| `a926bc8` | `docs/KN1_KAN_FFN_REPORT.md` | "test = 1,156" |
| `102520f` | `docs/DW1_DEPTH_VS_WIDTH_REPORT.md` | "V1 VitalDB test (1,156 patients, 19,543 windows)" |
| `102520f` | `docs/DB1_DISCRIMINATIVE_HR_BASELINE_REPORT.md` | "Test = 1,156 patients" |
| `02628f3` | `docs/RD1_DIRECT_RPEAK_DETECTOR_REPORT.md` | "V1 test: 1,156 patients, 19,543 windows" |
| `489885c` | `docs/RDDM_EXTERNAL_CONSENSUS_REPORT.md` | "19,543 V1 test windows / 1,156 patients" |
| `eb3f2d7` | `docs/PPGFLOWECG_EXTERNAL_FIXED_BUDGET_REPORT.md` | "V1 test split only" |
| `2d86f9e` | `docs/M1_PILOT_GAIN_PREDICTION_REPORT.md` | "5,000-replicate patient-clustered bootstrap (1,156 test patients" |
| `77764da` | `docs/M2_ACTION_VALUE_REPORT.md` | "(1,156 patients, seed 20260925)" |
| `c9c2004` | `docs/FBC1_CALIBRATION_ALLOCATION_REPORT.md` | "These cells were derived from 1,156 legacy-test patients" |

Cached TEST outputs also exist, listed by name only and not opened:
- `outputs/rd1_detector/{test_metrics, test_rpeaks, decoded_I_test, decoded_D_test}.npz`
- `outputs/ed1_cache/samples_{I,D}_test.npy`
- `outputs/m1_pilot_gain_prediction/{test_predictions_full, table_test}.npz`
- `outputs/m2_action_value/{rows_test, test_policy_rows}.npz`
- `outputs/db1_hr_regressor/test_errors.npz`

## 4. Hypothesis generation that DP0 / DP1 inherit (verbatim excerpts)

- **`02628f3` RD1 report:** "# RD1 — A direct PPG → R-peak detector matches consensus decoding on F1 and beats it on
  RR-MAE; generation is not necessary for event estimation". The results section is headed "## Results (V1 VitalDB
  test)".
- **`504395d`** (BF0 preregistration commit), `docs/BEAT_FIRST_ARCHITECTURE_DESIGN_KO.md`:
  - "VitalDB 수치는 test 1,156명 기준 (`artifacts/paper_tables/numbers.json`)이다."
  - "타이밍은 PPG에서 추정되고, 직접 검출기가 생성기보다 정확하다. RD1 F1 0.772 vs 샘플 하나 0.645–0.674 (VitalDB). … |
    타이밍은 판별 이벤트 단이 정한다"
- **Inheritance:** that event-stage decision (an RD1-family detector producing an event raster that conditions the
  waveform model) is carried unchanged by C0, C0-A, E0, R1, SF0, AF0 and DP0. In DP0 every model, both readouts and
  every control receives this raster.

## 5. Mitigating facts (recorded, not decisive under the rules)

- BF0, D0, C0, C0-A, E0, R1, SF0, AF0 and DP0 never loaded V1 TEST waveforms: each sealed TEST, and each report states
  that it was never opened.
- DP0 selected its candidate on DP-DEV and confirmed it on AF-LOCK only. No DP0 sharing pattern, width, adapter, loss,
  threshold or checkpoint was chosen from TEST outcomes.
- **Known earlier, but never audited:**
  - SF0 §11 disclosed: "Earlier-phase reporting of V1 TEST metrics is disclosed here as known."
  - The R1 freshness audit was NOT PERFORMED.
  - This is the first time the audit has been run. Under both rule sets it ends NOT FRESH.

## 6. Consequence

- **DP1 TEST NOT FRESH.** The old V1 TEST cannot serve as an untouched final test for DualReadout-ECG.
- **Stopped at §5, as the DP1 spec requires.** None of the following was performed:
  - TEST loading, the DP1 preregistration and the final freeze;
  - seed 43 / 44 training and DP-DEV multi-seed characterization;
  - the cached compute baseline and the TEST evaluation.
- **DP0's status is unchanged:** CONFIRMED on AF-LOCK, which is an internal locked set and not project-naive.
