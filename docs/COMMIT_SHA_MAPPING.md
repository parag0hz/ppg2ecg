# Commit SHA mapping (history rewrite 2026-08-27)

The repository's history was rewritten once more on 2026-08-27 to redact a third-party e-mail/account name that an earlier
log entry had quoted, and to re-assert the single author identity `parag0hz <131474134+parag0hz@users.noreply.github.com>` on every
commit (author and committer). File contents are otherwise byte-identical: `git diff` between the pre-rewrite tip and the new tip is
empty. Every SHA cited in the reports and pre-registrations was updated in place to the new value; this table records the mapping so
that older provenance files (`outputs/*/provenance.json`, which store the SHA that was current when a run started) remain traceable.

| old | new | commit |
|---|---|---|
| `9fd4ce5` | `d535ae4` | Initial commit |
| `a15b354` | `a15b354` | session 0: audit PENGUIN and prepare reproducible baseline |
| `f2d814b` | `13566ca` | merge: incorporate GitHub initial commit (local README kept) |
| `20cc6cd` | `5e0aa35` | experiment A0: reproduce PENGUIN on PPG-DaLiA |
| `1ae155c` | `f5dc344` | A0-b: pre-registration + deterministic fixed-bank checkpoint selection (pre-launch freeze) |
| `1ba600c` | `03e4db2` | experiment A0-b: stabilize PENGUIN checkpoint selection |
| `b5e8cc9` | `84804c9` | chore: register external/iMeanFlow as a submodule pinned at bf60cd7 |
| `5276bb9` | `bf2a024` | A2: Improved MeanFlow implementation, tests, audit and pre-registration (pre-launch freeze) |
| `1a2f9da` | `7112476` | A2 amendment (pre-result): h_scale=1000 for the shared-embedder interval conditioning; review fixes |
| `62c2b15` | `78e5518` | A2 amendment 2 (pre-result): h-only conditioning (official iMF design); h_scale=1000 diverged |
| `219a0b2` | `80e2229` | experiment A2: test iMeanFlow one-step PPG-to-ECG |
| `41f565e` | `aaca4be` | A3/A4 replication pre-registration freeze (Part I: DaLiA test S1); generic launchers |
| `73a3dfd` | `172fee1` | trainers: optional step-based validation rounds (A4 schedule rule, prereg Part II §7); default per-epoch behaviour unchanged |
| `2457fd5` | `0cdc91b` | A4 Part II freeze: WildPPG audit, deterministic split, subset/round rules, dataset-agnostic gates (before any A4 training) |
| `633c9f8` | `b5848c1` | experiment A3: test one-step recovery on new DaLiA subject |
| `5dd2b2d` | `ce13126` | A3 results: provenance, metrics, NFE curve, recovery, figures (small files only) |
| `0c7ddb0` | `bae0142` | experiment A4: replicate one-step recovery on WildPPG |
| `cc28ad9` | `7860940` | preregister A5 conditional-mean control |
| `d961000` | `8ca11ad` | amend A5 preregistration: learned constant state token (zero-state dead-start) |
| `b295a63` | `e0a63ae` | experiment A5: test conditional-mean hypothesis |
| `2fc7841` | `50a77a8` | preregister A6 capacity-matched conditional-mean control |
| `f4a115f` | `84223f0` | A7 dataset audit + preregister A7 PPG->ABP generalisation (MIMIC-BP) |
| `ac0016b` | `76fb54a` | experiment A6: capacity-match conditional-mean control |
| `9b65d1b` | `fed5b8c` | experiment A7: test one-step structural attenuation on ABP |
| `27f1424` | `d6ca9dd` | preregister A8 ABP target-scale sensitivity control |
| `0a58449` | `5e78f3d` | docs: remove the third-party account name/e-mail from the authorship-fix log entry |
