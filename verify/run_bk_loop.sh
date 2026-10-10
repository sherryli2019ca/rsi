#!/bin/bash
# Experiment BK (verify/PREREGISTRATION_BK.md): one loop (lineage) of a block,
# rounds 0..9 from the block's seeded calibration or from the round it was
# forked in, then the held-out deployment of the final incumbent and the
# shadow evaluation or replay (verify/bk.py post). The selection mode is read
# before every round (runs/bk_members: a lineage's arms shrink when it forks).
# Started by verify/run_bk.sh, or by verify/bk.py fork_lineage for a fork.
# Usage: bash verify/run_bk_loop.sh   (from the loop's checkout)
# Resume-safe; a second copy started while one runs exits at once.
set -u
D=tau2_airline
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
PY=/home/user/venv-tau2/bin/python
NS=$(basename "$HERE")
LIMIT=${LIMIT:-85}
mkdir -p runs/rrsi/$D/logs runs/verify/$D
exec 9> runs/bk.lock
flock -n 9 || { echo "[bk:$NS] already running"; exit 0; }
export PYTHONPATH="$HERE" RRSI_USAGE_LOG=runs/rrsi/$D.usage.jsonl
[ -f runs/rrsi/$D/frontier.json ] || { echo "[bk:$NS] not seeded" >> runs/rrsi/$D/logs/run.log; exit 1; }
settled() { $PY -c "import json;print(len(json.load(open('runs/rrsi/$D/frontier.json'))['trajectory'])-1)"; }
for t in $(seq 0 9); do
  for attempt in 1 2 3; do
    [ "$(settled)" -ge $((t + 1)) ] && break
    $PY -m verify.bk spend --limit "$LIMIT" >> runs/rrsi/$D/logs/run.log 2>&1 || {
      echo "[bk:$NS] spend limit \$$LIMIT reached before round $t" >> runs/rrsi/$D/logs/run.log; exit 2; }
    SEL=$($PY -m verify.bk selection --run "$HERE") || exit 1
    echo "[bk:$NS] round $t attempt $attempt ($SEL) $(date -u +%H:%M:%S)" >> runs/rrsi/$D/logs/run.log
    RRSI_ROUND=$t $PY rrsi.py --domain $D --runs runs/rrsi --branch-ns $NS --selection $SEL \
      round --t $t >> runs/rrsi/$D/logs/r$t.log 2>&1
  done
  [ "$(settled)" -ge $((t + 1)) ] || { echo "[bk:$NS] round $t did not settle" >> runs/rrsi/$D/logs/run.log; exit 1; }
done
touch runs/bk_loop.done
$PY -m verify.bk deploy --run "$HERE" > runs/verify/$D/deploy.log 2>&1 || exit 1
$PY -m verify.bk post --run "$HERE" > runs/verify/$D/post.log 2>&1 || exit 1
touch runs/bk.done
