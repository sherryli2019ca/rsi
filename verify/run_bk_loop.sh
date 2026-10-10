#!/bin/bash
# Experiment BK (verify/PREREGISTRATION_BK.md): one loop of a block, rounds 0..9
# from the block's seeded calibration, then the held-out deployment of the
# final incumbent and, by arm, the shadow evaluation (seqfull, seqcost) or the
# replay of the sequential rules (full). Started by verify/run_bk.sh.
# Usage: NS=bk_b1_s SEL=seqfull verify/run_bk_loop.sh   (from the loop's checkout)
# Resume-safe; a second copy started while one runs exits at once.
set -u
D=tau2_airline
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
PY=/home/user/venv-tau2/bin/python
NS=${NS:?set NS, e.g. bk_b1_s}
SEL=${SEL:?set SEL=full, seqfull, seqcost, seqhist or seqadm}
LIMIT=${LIMIT:-120}
mkdir -p runs/rrsi/$D/logs runs/verify/$D
exec 9> runs/bk.lock
flock -n 9 || { echo "[bk:$NS] already running"; exit 0; }
export PYTHONPATH="$HERE" RRSI_USAGE_LOG=runs/rrsi/$D.usage.jsonl
[ -f runs/rrsi/$D/frontier.json ] || { echo "[bk:$NS] not seeded" >> runs/rrsi/$D/logs/run.log; exit 1; }
A="--domain $D --runs runs/rrsi --branch-ns $NS --selection $SEL"
settled() { $PY -c "import json;print(len(json.load(open('runs/rrsi/$D/frontier.json'))['trajectory'])-1)"; }
for t in $(seq 0 9); do
  for attempt in 1 2 3; do
    [ "$(settled)" -ge $((t + 1)) ] && break
    $PY -m verify.bk spend --limit "$LIMIT" >> runs/rrsi/$D/logs/run.log 2>&1 || {
      echo "[bk:$NS] spend limit \$$LIMIT reached before round $t" >> runs/rrsi/$D/logs/run.log; exit 2; }
    echo "[bk:$NS] round $t attempt $attempt $(date -u +%H:%M:%S)" >> runs/rrsi/$D/logs/run.log
    $PY rrsi.py $A round --t $t >> runs/rrsi/$D/logs/r$t.log 2>&1
  done
  [ "$(settled)" -ge $((t + 1)) ] || { echo "[bk:$NS] round $t did not settle" >> runs/rrsi/$D/logs/run.log; exit 1; }
done
touch runs/bk_loop.done
$PY -m verify.bk deploy --run "$HERE" > runs/verify/$D/deploy.log 2>&1 || exit 1
case "$SEL" in
  seqfull|seqcost) $PY -m verify.cl shadow --run "$HERE" > runs/verify/$D/shadow.log 2>&1 || exit 1 ;;
  full)    $PY -m verify.bk replay --run "$HERE" > runs/verify/$D/replay.log 2>&1 || exit 1 ;;
esac
touch runs/bk.done
