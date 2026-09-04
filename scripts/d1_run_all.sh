#!/usr/bin/env bash
# D1 multi-dataset benchmark orchestrator (docs/D1_MULTI_DATASET_BENCHMARK_PREREGISTRATION.md §9).
#
# Runs, inside a detached tmux session so it survives an SSH disconnect and STRICTLY IN SEQUENCE (one 32 GB GPU,
# never two trainings at once):
#   1. scripts/d1_train.py    --corpus {dalia,bidmc,capnobase,wildppg,vitaldb}   (prereg D1 §4: wildppg is a NEW run
#      over the 14 subjects left after the never-loaded kjd/ssx, so it does NOT reuse the frozen C1 arm-B run)
#   2. scripts/d1_evaluate.py --corpus {wildppg,dalia,bidmc,capnobase,vitaldb}
#   3. scripts/d1_figures.py --all and scripts/d1_report.py --all, if those scripts exist
# Idempotent: a corpus whose TRAINING_DONE / EVAL_DONE marker exists is skipped. A failed stage is logged, marked
# failed in STATUS.json, and the run CONTINUES with the next corpus.
#
# Progress:  outputs/d1_bench/PROGRESS.log  (append-only, ISO timestamps)
# Machine:   outputs/d1_bench/STATUS.json   ({corpus: {train, eval, started, finished, seconds, stages}}; the
#            pseudo-corpus "_final" carries the figures/report stages)
set -uo pipefail

ROOT=/home/kwy00/ppg2ecg-one-step
PY="$ROOT/.venv/bin/python"
SESSION=d1bench
BENCH="$ROOT/outputs/d1_bench"
PROGRESS="$BENCH/PROGRESS.log"
STATUS="$BENCH/STATUS.json"
# Small corpora first so a configuration fault surfaces in minutes rather than hours; the two large ones last.
TRAIN_CORPORA=(dalia bidmc capnobase wildppg vitaldb)
EVAL_CORPORA=(dalia bidmc capnobase wildppg vitaldb)
# Per-subject test-window cap. 0 = no cap. Only the two large corpora are capped; the small ones evaluate every
# window. The realised counts are written into eval_meta.json and every CSV, and disclosed in RESULTS.md.
declare -A EVAL_CAP=([dalia]=0 [bidmc]=0 [capnobase]=0 [wildppg]=1024 [vitaldb]=1024)

# ---------------------------------------------------------------------------- launcher (default invocation)
if [ "${1:-}" != "--inner" ]; then
  echo "D1 benchmark launcher"
  echo "  attach:   tmux attach -t $SESSION"
  echo "  progress: tail -f $PROGRESS"
  echo "  status:   cat $STATUS"
  if ! command -v tmux >/dev/null 2>&1; then
    echo "ERROR: tmux is not installed" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session '$SESSION' already exists -- not starting a second one"
    exit 0
  fi
  mkdir -p "$BENCH"
  tmux new-session -d -s "$SESSION" -c "$ROOT" "bash $ROOT/scripts/d1_run_all.sh --inner"
  tmux set-option -t "$SESSION" remain-on-exit on >/dev/null 2>&1
  echo "started detached tmux session '$SESSION'"
  exit 0
fi

# ---------------------------------------------------------------------------- worker (inside tmux)
mkdir -p "$BENCH"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$PROGRESS"; }

# status_update <corpus> <stage> <state> [started] [finished] [seconds]
status_update() {
  "$PY" - "$STATUS" "$@" <<'PYEOF'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
corpus, stage, state = sys.argv[2], sys.argv[3], sys.argv[4]
started, finished, seconds = (list(sys.argv[5:8]) + ["", "", ""])[:3]
d = json.loads(p.read_text()) if p.exists() else {}
skeleton = {"started": None, "finished": None, "seconds": 0, "stages": {}}
if stage in ("train", "eval"):  # the pseudo-corpus "_final" has neither, so it must not sprout phantom stages
    skeleton = {"train": "pending", "eval": "pending"} | skeleton
c = d.setdefault(corpus, skeleton)
c[stage] = state
st = c["stages"].setdefault(stage, {})
st["state"] = state
for key, val in (("started", started), ("finished", finished)):
    if val:
        st[key] = val
if seconds:
    st["seconds"] = float(seconds)
starts = [v["started"] for v in c["stages"].values() if v.get("started")]
fins = [v["finished"] for v in c["stages"].values() if v.get("finished")]
c["started"] = min(starts) if starts else None
c["finished"] = max(fins) if fins else None
c["seconds"] = sum(float(v.get("seconds", 0)) for v in c["stages"].values())
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(d, indent=1, sort_keys=True))
PYEOF
}

# run_stage <corpus> <stage> <marker-file-or-empty> <cmd...>
run_stage() {
  local corpus="$1" stage="$2" marker="$3"
  shift 3
  local stage_log="$BENCH/${corpus}_${stage}.log"
  if [ -n "$marker" ] && [ -f "$marker" ]; then
    log "SKIP  $corpus/$stage (marker exists: ${marker#"$ROOT/"})"
    status_update "$corpus" "$stage" done
    return 0
  fi
  local t0 s0 t1 rc
  s0="$(date -Is)"
  t0="$(date +%s)"
  status_update "$corpus" "$stage" running "$s0"
  log "START $corpus/$stage : $*"
  "$@" >>"$stage_log" 2>&1
  rc=$?
  t1="$(date +%s)"
  if [ "$rc" -eq 0 ]; then
    status_update "$corpus" "$stage" done "" "$(date -Is)" "$((t1 - t0))"
    log "DONE  $corpus/$stage in $((t1 - t0))s"
  else
    status_update "$corpus" "$stage" failed "" "$(date -Is)" "$((t1 - t0))"
    log "FAIL  $corpus/$stage exit $rc after $((t1 - t0))s -- continuing; see ${stage_log#"$ROOT/"}"
    tail -n 5 "$stage_log" | while IFS= read -r line; do log "      | $line"; done
  fi
  return 0
}

cd "$ROOT" || exit 1
log "==== D1 benchmark run start (host $(hostname), session $SESSION, pid $$) ===="
log "attach: tmux attach -t $SESSION | progress: tail -f $PROGRESS"

for c in "${TRAIN_CORPORA[@]}"; do
  run_stage "$c" train "$ROOT/outputs/d1_${c}_seed42/TRAINING_DONE" "$PY" "$ROOT/scripts/d1_train.py" --corpus "$c"
done

for c in "${EVAL_CORPORA[@]}"; do
  run_stage "$c" eval "$ROOT/outputs/d1_${c}_seed42/EVAL_DONE" "$PY" "$ROOT/scripts/d1_evaluate.py" \
    --corpus "$c" --max-test-windows-per-subject "${EVAL_CAP[$c]}"
done

if [ -f "$ROOT/scripts/d1_figures.py" ]; then
  run_stage _final figures "" "$PY" "$ROOT/scripts/d1_figures.py" --all
else
  log "WARN  scripts/d1_figures.py not present -- figures skipped"
  status_update _final figures skipped
fi

if [ -f "$ROOT/scripts/d1_report.py" ]; then
  run_stage _final report "" "$PY" "$ROOT/scripts/d1_report.py" --all
else
  log "WARN  scripts/d1_report.py not present -- report skipped"
  status_update _final report skipped
fi

log "==== D1 benchmark run finished ===="
"$PY" - "$STATUS" <<'PYEOF' | tee -a "$PROGRESS"
import json, sys

for k, v in sorted(json.load(open(sys.argv[1])).items()):
    stages = " ".join(f"{s}={v[s]}" for s in ("train", "eval", "figures", "report") if s in v)
    print(f"  {k:12s} {stages} {v.get('seconds', 0):.0f}s")
PYEOF
