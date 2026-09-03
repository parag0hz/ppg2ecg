# O2 — Canonical event warp: frozen specification and mechanical audit

Written **before** the preregistration is committed and **before** any round-trip or generator result exists
(commit-order steps 3–4). It fixes the warp operator exactly; the Stage-0 round-trip numbers that decide whether
the operator is accepted are produced afterwards and reported in `docs/O2_ORACLE_EVENT_CANONICALIZATION_REPORT.md`.

Base commit `8cb8c76` (clean tree). Implementation: `src/ppg2ecg/evaluation/o2_warp.py`.

**Oracle disclosure.** The GT ECG R schedule is used to build this coordinate at training *and* at inference.
O2 is a target-leakage diagnostic and is **not deployable**.

---

## 1. What the operator is

For one 8 s window (`FS = 128`, `L = 1024`) with GT R peaks `r = [r_1 … r_K]` obtained from the frozen project
detector (`rpeaks.detect_rpeaks`, the same call used by R1/Q1/M1/O1 — no detector switching):

**Canonical beat schedule** (`canonical_positions`), for `K ≥ 3`:

```
q_1 = r_1 ,   q_K = r_K ,   q_k = r_1 + (k-1)/(K-1) · (r_K - r_1)
```

so beat count, first beat phase, last beat phase and the total cardiac span are preserved while the interior
inter-beat intervals are equalised. For `K < 3` the window uses the **identity warp** and is *not* dropped; its
fraction is recorded.

**QRS-preserving anchors** (`build_anchors`), with `W = 10` samples (78.125 ms — exactly the M1/O1 QRS-core
half-width `CORE`):

```
boundary      0      -> 0
per beat k:   r_k - W -> q_k - W ,   r_k -> q_k ,   r_k + W -> q_k + W      (only if strictly inside the window)
boundary      1023   -> 1023
```

The three anchors per beat pin the local slope to **exactly 1** across the QRS core, so the schedule change is
absorbed by the inter-beat regions rather than by stretching the QRS complex itself. `tau = f_r(t)` is the
monotone piecewise-linear interpolation of `(src → dst)`; `t = f_r^{-1}(tau)` is the same interpolation with the
coordinates exchanged.

**Anchor-validity rule (frozen).** Candidate anchors are sorted by source position and kept in order only while
**both** coordinates strictly increase (by more than `EPS = 1e-3` samples). If any beat's **centre** anchor
(`r_k → q_k`) would be dropped, or a boundary anchor is lost, the window is **invalid** and falls back to the
identity warp; the count is reported. Preregistration §7 STOP rule: if **more than 0.5 %** of windows need a
fallback for any reason other than `K < 3`, O2 stops.

## 2. Resampling

One fixed deterministic linear resampler: `torch.nn.functional.grid_sample(mode="bilinear",
align_corners=True, padding_mode="border")` applied as 1-D sampling through `[B, C, 1, L]`. No cubic
interpolation, no learned resampling, no anti-alias search.

- `x_can(tau) = x(f^{-1}(tau))` — sample the raw signal at `f^{-1}(tau)`.
- `x(t) = x_can(f(t))` — sample the canonical signal at `f(t)`.
- **Identity rows are returned bit-exactly**, without touching the interpolator (`apply_warp` selects the
  original tensor for those rows).
- **No amplitude Jacobian scaling and no renormalisation after warping**: the signal is treated as a value, not
  as a density under a coordinate change.

## 3. What is warped, and what is not

| tensor | treatment |
|---|---|
| PPG (model input) | warped into the canonical coordinate with the **same** `f_r` as the ECG |
| GT ECG (training target) | warped with the same `f_r` |
| Gaussian source `e ~ N(0, I)` | drawn **directly in canonical coordinates**; never warped |
| generated ECG | inverse-warped back to raw time with the same `f_r`; no shift, no peak snapping, no event editing |

**The model receives no event feature of any kind** — no GT R binary map, no Gaussian R field, no phase
channel, no beat-count or RR scalar, no event embedding, no R token. The only oracle intervention is the
temporal coordinate transformation. This is what separates O2 from R2/R3.

## 4. Mechanical audit (verified before any data-dependent result)

Checked in `tests/test_o2_oracle_canonicalization.py` and reproduced here on a worked example
(`r = [120, 340, 551, 790, 1000]`, `K = 5`):

| property | result |
|---|---|
| canonical schedule | `q = [120, 340, 560, 780, 1000]` (equal 220-sample spacing, ends pinned) |
| `f(r_k) = q_k` | exact to 1e-9 |
| `f^{-1}(q_k) = r_k` | exact to 1e-9 |
| slope inside every QRS core | exactly `1.0` |
| global slope range | 0.913 … 1.053 (inter-beat segments only) |
| monotone `f`, monotone `f^{-1}`, all finite | yes |
| boundary anchors | `0 → 0`, `1023 → 1023` exact |
| identity rows | returned bit-exactly (`torch.equal`) |
| PPG and ECG | warped by the same object in the same call |

## 5. Diagnostic-only variant (never trained)

`CenterOnlyWarp` uses only the centre anchors `r_k → q_k` (no ±W). It exists solely to test whether the local
slope-1 construction actually protects QRS shape, by comparing round-trip T6/T7/T8 against the primary
operator. It does **not** enter any O2 verdict.

## 6. Stage-0 acceptance thresholds (frozen here, evaluated after this document is pushed)

On the frozen 2,048-window development cohort, round-trip `x_rt = W^{-1}(W(x))` of the **GT ECG** must satisfy
**all** of:

| id | quantity | threshold |
|---|---|---|
| R0-1 | median raw waveform RMSE | ≤ 0.020 |
| R0-2 | median normalised absolute error of T6 (max abs derivative) | ≤ 0.020 |
| R0-3 | median normalised absolute error of T7 (curvature energy) | ≤ 0.020 |
| R0-4 | median normalised absolute error of T4 (QRS p2p) and T8 (QRS width) | ≤ 0.020 each |
| R0-5 | detector F1@50 between original and round-trip ECG | ≥ 0.98 |
| R0-6 | median beat-count difference | 0 |

Normalisation uses the **exact O1 train IQRs** (`artifacts/o1_component_extractability/target_scaling.json`).
If any check fails the verdict is **CANONICALIZATION OPERATOR REJECTED**, no generator is trained, and O2 stops.
These thresholds are not loosened.
