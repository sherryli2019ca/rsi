#!/bin/bash
cd /home/user/rsi
for d in tau2_retail tau2_airline; do
  PYTHONPATH=. /home/user/venv-tau2/bin/python -m agent_exp.reanalysis --out runs/$d --domain $d --stage decide 2>&1 | grep -v -E "WARN|INFO|DEBUG|^ |^\{|^\}|^\]|^\[" > runs/reanalysis_$d.log
done
