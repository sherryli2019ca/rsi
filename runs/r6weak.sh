#!/bin/bash
cd /home/user/rsi
for d in tau2_retail tau2_airline; do
  PYTHONPATH=. /home/user/venv-tau2/bin/python -m agent_exp.reanalysis --out runs/$d --domain $d --stage decide --settings weak --workers 2 > runs/$d/r6weak.log 2>&1 &
done
wait
touch runs/r6weak.done
