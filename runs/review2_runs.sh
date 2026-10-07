#!/bin/bash
# Second-review experiments A, B, C, D, F (approved 2026-10-07 04:16Z)
cd /home/user/rsi
PY="env PYTHONPATH=. /home/user/venv-tau2/bin/python"
F='WARN|INFO|DEBUG|^ |^\{|^\}|^\]|^\['
$PY -m agent_exp.review2_runs --stage baseline --out runs/tau2_retail --domain tau2_retail --workers 24 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_A_retail.log &
$PY -m agent_exp.review2_runs --stage baseline --out runs/tau2_airline --domain tau2_airline --workers 24 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_A_airline.log &
$PY -m agent_exp.bank_heldout --out runs/tau2_airline --domain tau2_airline --budgets 10 --n_seeds 5 --trials 3 --workers 24 --methods LLM-only HarnessFix HarnessFix-gate CARVE Net 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_B_airline.log &
$PY -m agent_exp.review2_runs --stage clean --out runs/tau2_retail --workers 24 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_F_retail.log &
( $PY -m agent_exp.review2_runs --stage nostep --out runs/tau2_retail --workers 24 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_C_retail.log
  $PY -m agent_exp.review2_runs --stage loo --out runs/tau2_retail --workers 24 2>&1 | grep --line-buffered -v -E "$F" > runs/r2_D_retail.log ) &
wait
touch runs/review2_runs.done
