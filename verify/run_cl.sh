#!/bin/bash
# Experiment CL (verify/PREREGISTRATION_CL.md, approved by Dr Cao 2026-10-09):
# RRSI on tau2 airline with sequential full evaluation as its selection step,
# rounds 0..9, then held-out deployments and the shadow evaluation.
# Usage: NS=cl1 verify/run_cl.sh    (from a worktree pinned to the registered commit)
# Resume-safe: rerun the same command after a container restart.
set -u
D=tau2_airline
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
PY=/home/user/venv-tau2/bin/python
NS=${NS:?set NS=cl1 or cl2}
export PYTHONPATH="$HERE" RRSI_USAGE_LOG=runs/rrsi/$D.usage.jsonl
mkdir -p runs/rrsi/$D/logs runs/verify/$D
A="--domain $D --runs runs/rrsi --branch-ns $NS --selection seqfull"
EV=evolve/$NS/$D
if [ ! -f runs/rrsi/$D/frontier.json ]; then
  $PY rrsi.py $A baseline --job base > runs/rrsi/$D/logs/baseline.log 2>&1 || exit 1
fi
for L in base2 base3; do
  [ -f runs/rrsi/$D/jobs/heldout_$L/eval.json ] || \
    $PY rrsi.py $A heldout --label $L --set evolve --ref $EV > runs/rrsi/$D/logs/$L.log 2>&1
done
[ -f runs/rrsi/$D/calibration.json ] && grep -q heldout_base3 runs/rrsi/$D/calibration.json || \
  $PY rrsi.py $A calibrate --jobs base,heldout_base2,heldout_base3 > runs/rrsi/$D/logs/calibrate.log 2>&1
settled() { $PY -c "import json;print(len(json.load(open('runs/rrsi/$D/frontier.json'))['trajectory'])-1)"; }
for t in $(seq 0 9); do
  for attempt in 1 2 3; do
    [ "$(settled)" -ge $((t + 1)) ] && break
    echo "[cl:$NS] round $t attempt $attempt $(date -u +%H:%M:%S)" >> runs/rrsi/$D/logs/run.log
    $PY rrsi.py $A round --t $t >> runs/rrsi/$D/logs/r$t.log 2>&1
  done
  [ "$(settled)" -ge $((t + 1)) ] || { echo "[cl:$NS] round $t did not settle" >> runs/rrsi/$D/logs/run.log; exit 1; }
done
touch runs/cl_loop.done
$PY -m verify.cl deploy --run "$HERE" > runs/verify/$D/deploy.log 2>&1 || exit 1
$PY -m verify.cl shadow --run "$HERE" > runs/verify/$D/shadow.log 2>&1 || exit 1
touch runs/cl.done
