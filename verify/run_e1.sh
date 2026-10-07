#!/bin/bash
# Experiment E1 for one tau2 domain: RRSI reference trajectory + verification evidence follower.
# Usage: verify/run_e1.sh tau2_retail|tau2_airline   (needs Dr Cao's approval of the E1 quote)
set -u
D=$1
cd /home/user/rsi
PY=/home/user/venv-tau2/bin/python
export PYTHONPATH=/home/user/rsi RRSI_USAGE_LOG=runs/rrsi/$D.usage.jsonl
mkdir -p runs/rrsi/$D/logs runs/verify/$D
A="--domain $D --runs runs/rrsi"
if [ ! -f runs/rrsi/$D/frontier.json ]; then
  $PY rrsi.py $A baseline --job base > runs/rrsi/$D/logs/baseline.log 2>&1 || exit 1
fi
for L in base2 base3; do
  [ -f runs/rrsi/$D/jobs/heldout_$L/eval.json ] || \
    $PY rrsi.py $A heldout --label $L --set evolve --ref evolve/$D > runs/rrsi/$D/logs/$L.log 2>&1
done
$PY rrsi.py $A calibrate --jobs base,heldout_base2,heldout_base3 > runs/rrsi/$D/logs/calibrate.log 2>&1
$PY -m verify.follow --domain $D > runs/verify/$D/follow.log 2>&1 &
$PY rrsi.py $A run > runs/rrsi/$D/logs/run.log 2>&1
wait
touch runs/e1_$D.done
