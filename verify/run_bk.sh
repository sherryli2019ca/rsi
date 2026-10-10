#!/bin/bash
# Experiment BK (verify/PREREGISTRATION_BK.md): one block. Calibrates the noise
# band once (baseline and two repeated base evaluations, as in run_il.sh),
# seeds the block's loops with it (one per arm of verify/bk.py's DESIGN),
# starts them, and deploys the block's base on the held-out tasks.
# Usage: BLOCK=b1 bash verify/run_bk.sh   (from /home/user/bk_<block>_cal, a
# detached checkout of the registered commit). Resume-safe: rerun the same command.
set -u
D=tau2_airline
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
B=${BLOCK:?set BLOCK, e.g. b1}
[ "$(basename "$HERE")" = "bk_${B}_cal" ] || { echo "run from /home/user/bk_${B}_cal"; exit 1; }
PY=/home/user/venv-tau2/bin/python
COMMIT=$(git rev-parse HEAD)
export PYTHONPATH="$HERE" RRSI_USAGE_LOG=runs/rrsi/$D.usage.jsonl
mkdir -p runs/rrsi/$D/logs runs/verify/$D
NS=bk_${B}_cal
A="--domain $D --runs runs/rrsi --branch-ns $NS --selection full"
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
grep -q heldout_base3 runs/rrsi/$D/calibration.json || exit 1
touch runs/cal.done
ARMS=$($PY -m verify.bk arms) || exit 1          # e.g. "f=full s=seqfull c=seqcost ..."
for x in $ARMS; do
  a=${x%%=*}; W=/home/user/bk_${B}_$a
  [ -d "$W" ] || git worktree add --detach "$W" "$COMMIT" > /dev/null 2>&1 || exit 1
  [ "$(git -C "$W" rev-parse HEAD)" = "$COMMIT" ] || { echo "$W is not at $COMMIT"; exit 1; }
  $PY -m verify.bk seed --cal "$HERE" --run "$W" --ns bk_${B}_$a --selection ${x#*=} || exit 1
done
for x in $ARMS; do
  a=${x%%=*}; W=/home/user/bk_${B}_$a
  [ -f "$W/runs/bk.done" ] && continue
  (cd "$W" && NS=bk_${B}_$a SEL=${x#*=} setsid nohup bash verify/run_bk_loop.sh \
     >> runs/bk.out 2>&1 < /dev/null &)
done
[ -f runs/verify/$D/deploy_base.done ] || \
  $PY -m verify.bk deploy-base --run "$HERE" > runs/verify/$D/deploy_base.log 2>&1
