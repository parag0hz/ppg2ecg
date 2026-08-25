# WildPPG Audit (for A4 replication)

Audited 2026-08-26 by a multi-agent workflow (12 agents: locate / acquire+inspect / PENGUIN usage / adversarial verification)
plus direct inspection; every item below was verified on the downloaded files unless marked INFERRED.

## 1. Source, paper, licence
| Item | Value |
|---|---|
| Dataset | **WildPPG: A Real-World PPG Dataset of Long Continuous Recordings** — Meier, Demirel, Holz (ETH Zürich), NeurIPS 2024 Datasets & Benchmarks (DOI 10.52202/079017-0073; arXiv:2412.17540). Code: `github.com/eth-siplab/WildPPG` |
| Official data source | ETH polybox public share `https://polybox.ethz.ch/index.php/s/NWTuyNojU7aya1y` (the link given in PENGUIN's README) — anonymous, read-only, **no login / consent form / password**; per-file WebDAV URLs `…/public.php/dav/files/NWTuyNojU7aya1y/data/WildPPG_Part_<id>.mat`. (Hugging Face `eth-siplab/WildPPG` hosts a processed file **without raw ECG** — unusable here.) |
| Licence | **Data: CC BY-NC-SA 4.0**; code: GPL-3.0 (paper §4.1; `LICENSE.md` in `github_image.zip`). Caveats recorded verbatim: `LICENSE.md` still carries a "released for review purposes only" clause that was never removed after publication (INFERRED stale); the review-time datasheet says CC BY-SA (no NC); the HF card says "mit". We treat the data as CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike) — compatible with this non-commercial research use; the datasheet also forbids diagnostic use and de-anonymisation. Ethics: ETH EK 2022-N-44, participant consent to publication. |
| Download | 2026-08-26 01:38–01:50 KST, 16 `.mat` + `github_image.zip`, **19,608,554,456 B**, every size equal to the server's PROPFIND size; `data/raw/WildPPG/CHECKSUMS.sha256` (17 sha256), `INVENTORY.json` (per-file structure). |

## 2. Content (verified on all 16 files)
- 16 participants (ids `an0 e61 fex k2s kjd l38 n31 ngh p5d p9p qm9 ssx trh tz8 u7y w4p`), one MATLAB v5 file each (scipy `loadmat`, ~2–4 s/file).
- Four time-synchronised devices per participant: **sternum, head, wrist, ankle**. Each has `acc_{x,y,z}` (128 Hz), `altitude`, `temperature` (0.5 Hz), `ppg_g` (530 nm), `ppg_ir` (950 nm), `ppg_r` (660 nm) at **128 Hz** (values in [0,1], ADC fraction); **ECG (lead I, 128 Hz, int32 raw ADC ±131071) exists only at the sternum**.
- Every 128-Hz channel of a participant has exactly the same length; N is a multiple of 1024 (= 8 s), ECG and every PPG site have identical duration (alignment check passes for all 16). Total ECG **216.8 h** (12.3–15.2 h per participant, mean 13.55 h).
- Missing data: **no NaN anywhere**; recording gaps are constant-filled (PPG = 1.0, acc = 0) — documented in `notes` for an0 (ankle, 15 s), k2s (wrist, 53 min), l38 (wrist, 27 min); an undocumented flat run exists in qm9 sternum (526 s). Three participants (fex, kjd, p5d) carry the note "ECG recordings of this participant are noisy".
- Activities: outdoor hiking day (walking, resting, cycling segments per the paper); no per-sample activity labels in the release.

## 3. How PENGUIN uses WildPPG (code `6cd70cd`, verified line by line)
- Loader `src/utils/load_data.py:27-77`: green PPG (`colors: [g]`) at **all four sites** (`locations: [sternum, head, wrist, ankle]`); windows per site are concatenated **site-major as separate samples** and the sternum ECG windows are **tiled ×4** (each ECG window is the target of 4 PPG windows). Red/IR PPG, accelerometer, altitude, temperature and `notes` are never used. Site identity is not stored.
- Windowing `sliding_window_view[::fs·segment_len]` (non-overlapping), fs read from the file (128). Preprocessing identical to DaLiA (FFT resample to 128·segment_len — a no-op length for WildPPG, still applied; PPG Butterworth 4th-order 0.5–4 Hz band-pass, ECG 0.5 Hz high-pass, both `filtfilt`; per-window z-score and min-max to [−1, 1]).
- Config bookkeeping `sample_num 390216 = 97554 ECG windows × 4 sites` and `duration 867.15 h = 4 × 216.8 h` → **8-second windows**, again contradicting the shipped `segment_len: 4` (same pattern as DaLiA); with 4 s the HR metric compresses time 2× (see PENGUIN_AUDIT §20).
- Split: `16 // 8 = 2 val, 2 test, 12 train` subjects, `random.sample(glob(...))` → **machine-dependent** (glob order); single run. Paper: "6:1:1 cross-subject, no subject overlap"; Table 1 WildPPG HR error **PENGUIN 12.97 bpm** (RDDM 16.02, RespDiff 20.57, CycleGAN 23.21, PaPaGei-S 55.42; ablation w/o PPG cond. 21.75).
- No NaN/constant/quality handling: the 861 constant-gap PPG windows (8 s) become deterministic ripple artefacts after z-score; noisy-ECG subjects are kept.
- Paper vs code: paper is silent on the PPG site (code: 4 sites pooled), the window length (code bookkeeping: 8 s, shipped 4 s), and the HR-metric pathology; the "6:1:1" claim corresponds to the 12/2/2 subject split.

## 4. Decisions frozen for A4 (docs/A3_A4_REPLICATION_PREREGISTRATION.md, Part II amendment)
1. **PPG channel/site (rule 1 — PENGUIN's explicit code choice):** green PPG at all four sites, each site's windows as separate samples paired with the sternum ECG (tiled). Per-site metrics are also reported (site identity is stored in our npz).
2. **Window / rate:** 8 s @ 128 Hz (1024 samples), PENGUIN filters and per-window normalisation — `scripts/build_processed_wildppg.py` → `data/processed/wildppg_8s/` (**389,355 windows**; MANIFEST with per-file sha256).
3. **Gap handling (documented deviation from PENGUIN):** windows with non-finite or zero-variance PPG/ECG are dropped **(861 windows, 0.22 %)** instead of being kept as filter-ripple artefacts; applied identically to both objectives. Noisy-ECG participants are kept (as PENGUIN).
4. **Split (deterministic, subject-level):** sorted ids → `random.Random(42).sample` → val **an0, k2s**; test **kjd, ssx**; train the other 12 — `data/manifests/split_a4_wildppg_seed42.json` (sha256 `bc168144…`). Note: test participant kjd is one of the "noisy ECG" participants — kept as drawn.
5. **Leakage gate:** subject-disjoint, window-hash-disjoint, window-local normalisation, PPG/ECG alignment (equal window counts per site), no NaN — see `docs/EXPERIMENT_LOG.md` for the run.
6. **Validation / test subsets (compute):** deterministic uniform stride subsample to ≤ 4,096 windows of the concatenated val (an0+k2s ≈ 49 k) and test (kjd+ssx ≈ 47 k) arrays; the same subsets for every arm (paired noise); fixed banks built on the val subset.
7. **Schedule unit:** validation round = min(epoch, 220 optimizer steps) (`--val-every-steps 220`), patience 20 rounds, min_delta 1e-4, max 300 rounds; effective batch 64 (iMF micro-batch 32).
