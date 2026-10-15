#!/usr/bin/env bash
# Finish every kev_r4 result after training (steps 5 and 6), resumably:
#   1. dev eval (gates G1-G3) + confidence trace caches (train/dev/calibration/risk_validation)
#   2. select-heads, finalize (frozen before the audit is decoded)
#   3. audit traces + audit (gate G4); test eval only if G1-G3 pass on dev
# Every step is skipped when its output exists, so the script can simply be rerun.
#
# Caches are filled by a worker pool: every minute, while a cache split has a part nobody
# holds, a worker is started on a GPU that runs fewer than TARGET<gpu> cache workers
# (workers started by earlier runs count too). A worker that dies (out of memory on a
# shared GPU) is simply replaced; within a part, decoded documents are checkpointed, so
# it loses at most one document. Run inside tmux: a session end must not kill it.
#
#   tmux new -s kev_r4 scripts/run_kev_r4.sh
#   TARGET0=3 TARGET1=1 scripts/run_kev_r4.sh        # cache workers per GPU
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R4=/home/jovyan/bachdx2/l2r4kie/artifacts/training-optimization
R4_AUDIT=/home/jovyan/bachdx2/l2r4kie/artifacts/token-review-v2/selected/attention/audit_predictions.jsonl
C=configs/confidence/kev.yaml
CACHE=artifacts/kev-r4/confidence/cache
OUT=artifacts/kev-r4/confidence/selected
LOG=artifacts/kev-r4/logs
TARGET0=${TARGET0:-3}
TARGET1=${TARGET1:-1}
mkdir -p $LOG

say() { echo "[$(date +%F' '%T)] $*" | tee -a $LOG/run.log; }

retry() {  # retry <log> <command...>
  local log=$1; shift
  for attempt in $(seq 1 30); do
    echo "attempt $attempt $(date +%T): $*" >> "$log"
    "$@" >> "$log" 2>&1 && return 0
    sleep 60
  done
  return 1
}

running() {  # running <gpu>: live cache workers on cuda:<gpu>
  pgrep -fc "[.]venv/bin/l2r4kie cache-traces .*device=cuda:$1\$"
}

pending() {  # pending "<splits>": print the splits with a part nobody holds, or parts left unmerged
  uv run --quiet python - $CACHE $C $1 <<'EOF'
import json, math, os, socket, sys
from pathlib import Path

cache, config, splits = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]

def held(lock):
    try:
        host, pid = lock.read_text().split()
        os.kill(int(pid), 0)
        return host == socket.gethostname()
    except (OSError, ValueError):
        return False

for split in splits:
    if (cache / f'{split}.pt').exists():
        continue
    provenance = cache / f'{split}.provenance.json'
    if not provenance.exists():  # not started
        print(split)
        continue
    record = json.loads(provenance.read_text())
    parts = math.ceil(len(record['documents']) / record['part_size'])
    free = [n for n in range(parts) if not (cache / f'{split}.part{n:04d}.pt').exists()
            and not held(cache / f'{split}.part{n:04d}.lock')]
    unmerged = all((cache / f'{split}.part{n:04d}.pt').exists() for n in range(parts)) \
        and not held(cache / f'{split}.merge.lock')
    if free or unmerged:
        print(split)
EOF
}

pool() {  # pool "<splits>": keep the cache workers going until every split is merged
  local splits=$1 n=0 todo gpu target
  while true; do
    todo=$(pending "$splits" | tr '\n' ' ')
    local merged=1
    for split in $splits; do [ -f $CACHE/$split.pt ] || merged=0; done
    [ $merged = 1 ] && return 0
    if [ -n "$todo" ]; then
      for gpu in 0 1; do
        target=$((gpu == 0 ? TARGET0 : TARGET1))
        if [ "$(running $gpu)" -lt $target ]; then
          n=$((n + 1))
          echo "start $(date +%T) cuda:$gpu splits: $todo" >> $LOG/pool_${splits// /_}.log
          uv run l2r4kie cache-traces --config $C --split $todo --set device=cuda:$gpu \
            >> $LOG/pool_${splits// /_}_$n.log 2>&1 &
          sleep 30  # one model load at a time
        fi
      done
    fi
    sleep 60
  done
}

evaluate() {  # evaluate <split> <args...>
  local split=$1; shift
  [ -f artifacts/kev-r4/eval-$split/metrics.json ] && return 0
  retry $LOG/eval_$split.log uv run l2r4kie evaluate --config configs/extractor/kev_r4.yaml \
    --adapter artifacts/kev-r4 --split $split --fields 24 --output artifacts/kev-r4/eval-$split "$@" \
    --set device=cuda:0
}

gates_pass() {  # pre-registered dev gates (docs/results/kev_extractor.md)
  uv run python -c "
import json, sys
m = json.load(open('artifacts/kev-r4/eval-dev/metrics.json'))
ok = m['scalar_exact_match'] >= .89334 and m['coordinate_rate'] < .005 and m['truncated_rate'] <= .02292
print(json.dumps({k: m[k] for k in ('scalar_exact_match', 'coordinate_rate', 'truncated_rate')}), 'pass' if ok else 'fail')
sys.exit(0 if ok else 1)"
}

say "start (cache workers: $TARGET0 on cuda:0, $TARGET1 on cuda:1)"
evaluate dev --documents-file $R4/cohort_plan.json &
EVAL=$!
pool "dev train calibration risk_validation"
wait $EVAL
say "dev eval and caches done"

if [ ! -f $OUT/selection.json ]; then
  uv run l2r4kie select-heads --config $C > $LOG/select_heads.log 2>&1 || { say "ERROR: select-heads"; exit 1; }
fi
say "heads selected: $(grep -o '"primary": "[^"]*"' $OUT/selection.json | head -1)"
if [ ! -f $OUT/frozen.json ]; then
  uv run l2r4kie finalize --config $C --device cuda:0 > $LOG/finalize.log 2>&1 || { say "ERROR: finalize"; exit 1; }
fi
say "frozen"

TEST=
if gates_pass >> $LOG/run.log 2>&1; then
  say "dev gates G1-G3 pass: test eval"
  evaluate test --limit 116 &
  TEST=$!
  TARGET0=$((TARGET0 > 1 ? TARGET0 - 1 : 1))  # the test eval decodes on cuda:0 too
else
  say "dev gates G1-G3 fail: no test eval (pre-registered)"
fi
pool "audit"
if [ ! -f $OUT/audit_report.json ]; then
  uv run l2r4kie audit --config $C --baseline-predictions $R4_AUDIT --device cuda:0 > $LOG/audit.log 2>&1 \
    || { say "ERROR: audit"; exit 1; }
fi
say "audit done"
[ -n "$TEST" ] && wait $TEST
say "finished"
