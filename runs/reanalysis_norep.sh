#!/bin/bash
cd /home/user/rsi
for d in tau2_retail tau2_airline; do
  BANK_NOREPLACE=1 PYTHONPATH=. /home/user/venv-tau2/bin/python -m agent_exp.reanalysis --out runs/$d --domain $d --stage decide --settings norep > runs/reanalysis_norep_$d.log 2>&1
done
touch runs/reanalysis_norep.done
