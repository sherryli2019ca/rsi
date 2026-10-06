#!/usr/bin/env bash
# Budget sweep on top of run_all.sh's shared stages. Usage: bash agent_exp/run_budgets.sh SEED
set -uo pipefail
SEED=$1; OUT=runs/real_s$SEED
P="python -m agent_exp.pipeline --out $OUT --seed $SEED --n_train 240 --n_test 120 --workers 8"
for B in 10 20 30; do
  for m in Replay-each Uncertainty Uncertainty-MF CARVE-full-only CARVE CARVE-calibrated CARVE-robust-check; do
    [ -f $OUT/verify_${m}_$B.json ] && grep -q patched_success $OUT/verify_${m}_$B.json && continue
    $P --stage all --method $m --budget $B > $OUT/log_${m}_$B.txt 2>&1 &
  done
  wait
done
