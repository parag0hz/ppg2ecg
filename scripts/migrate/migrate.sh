#!/usr/bin/env bash
# Move the project to another machine. Code, docs, artifacts (results) and data/manifests travel through git
# (27 MB); this script copies only what git deliberately does not hold: processed windows, checkpoints and caches.
#
#   core   6.88 GiB  processed windows + current-generation checkpoints + paired eval arrays  -> required
#   cache  1.30 GiB  HR / sample caches                                                       -> strongly recommended
#   expd   2.53 GiB  upstream PENGUIN checkpoints + BIDMC / MIMIC-BP windows                  -> only if EXP-D runs
#
# NOT copied, on purpose: data/raw (102 GB, only needed to rebuild processed windows),
# the 8-second-window corpora and the historical 8 s checkpoints (those stages are finished and a 16 GB card
# cannot retrain them anyway), data/pretrained (VM1's ImageNet weights, that stage is done).
#
# Usage, from THIS machine (push):
#     bash scripts/migrate/migrate.sh user@newhost:/path/to/ppg2ecg-one-step [core|cache|expd|all]
# or from the NEW machine (pull), after `git clone`:
#     bash scripts/migrate/migrate.sh --pull user@thishost:/home/kwy00/ppg2ecg-one-step [core|cache|expd|all]
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST_DIR="$SRC_ROOT/scripts/migrate"
RSYNC_OPTS=(-a -r -h --partial --info=progress2 --no-implied-dirs)

PULL=0
if [[ "${1:-}" == "--pull" ]]; then PULL=1; shift; fi
REMOTE="${1:?usage: migrate.sh [--pull] user@host:/path/to/repo [core|cache|expd|all]}"
WHICH="${2:-core}"
case "$WHICH" in
  core|cache|expd) SETS=("$WHICH") ;;
  all)             SETS=(core cache expd) ;;
  *) echo "unknown set '$WHICH' (core|cache|expd|all)" >&2; exit 2 ;;
esac

for s in "${SETS[@]}"; do
  list="$LIST_DIR/files_$s.txt"
  echo "== $s  ($(wc -l < "$list") paths)"
  if [[ $PULL -eq 1 ]]; then
    rsync "${RSYNC_OPTS[@]}" --files-from="$list" "$REMOTE/" "$SRC_ROOT/"
  else
    rsync "${RSYNC_OPTS[@]}" --files-from="$list" "$SRC_ROOT/" "$REMOTE/"
  fi
done

cat <<'EOF'

Done. On the new machine:
  1. git clone --recurse-submodules https://github.com/parag0hz/ppg2ecg.git ppg2ecg-one-step
     (code, docs, artifacts and data/manifests are all in git — 27 MB)
  2. verify the submodule pins:  git -C external/PENGUIN rev-parse --short HEAD  -> 6cd70cd
                                 git -C external/iMeanFlow rev-parse --short HEAD -> bf60cd7
     Neither may be modified.
  3. python -m venv .venv && .venv/bin/pip install -e .      (pyproject.toml; torch >= 2.11 with cu128/cu130
     for Blackwell — the 5080 is sm_120, same as the 5090, so the wheel that works here works there)
  4. .venv/bin/python -m pytest -x -o addopts="" -q          # 793 tests must pass
  5. Verification of the GPU change (do this once, and record it):
     regenerate ONE cached cell and compare with the value produced on the 5090, e.g. iMF (K 32, S 1) on the
     VitalDB test set should give HR 6.454 (seed 42, artifacts/dw2_multiseed/grid.csv).  Do NOT recompute cells
     that already have published numbers — copy their caches instead, and never mix numbers from two machines
     inside one table.
  6. On a 16 GB card: train one run at a time (current-generation training peaks at 11.3 GiB) and keep the
     inference batch at 256 or below (dw1_depth_width.BS / tt_expa_solver.BS = 256 -> 6.0 GiB).
EOF
