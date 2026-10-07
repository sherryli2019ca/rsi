#!/bin/bash
cd /home/user/rsi
PY="env PYTHONPATH=. /home/user/venv-tau2/bin/python"
for d in tau2_retail tau2_airline; do
  ( $PY -m agent_exp.reanalysis --out runs/$d --domain $d --stage decide --settings netabl --workers 2 > runs/r3_netabl_$d.log 2>&1
    $PY -m agent_exp.reanalysis3 --out runs/$d --domain $d > runs/r3_$d.log 2>&1 ) &
done
wait
touch runs/r3.done
