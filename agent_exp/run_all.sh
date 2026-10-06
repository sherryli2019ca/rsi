#!/usr/bin/env bash
# Full real-agent study for one seed: shared stages, then every method.
# Usage: bash agent_exp/run_all.sh SEED [BUDGET]
set -euo pipefail
SEED=$1; BUDGET=${2:-60}; OUT=runs/real_s$SEED
P="python -m agent_exp.pipeline --out $OUT --seed $SEED --n_train 240 --n_test 120 --workers 16"
$P --stage taxonomy
$P --stage patches
$P --stage calibrate
$P --stage all --method LLM-only --budget $BUDGET     # also caches the unpatched baseline
for m in Replay-each Uncertainty Uncertainty-MF CARVE-full-only CARVE CARVE-calibrated CARVE-robust-check; do
  $P --stage all --method $m --budget $BUDGET > $OUT/log_$m.txt 2>&1 &
done
wait
