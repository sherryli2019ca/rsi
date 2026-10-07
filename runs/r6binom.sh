#!/bin/bash
cd /home/user/rsi
for d in tau2_retail tau2_airline; do
  PYTHONPATH=. /home/user/venv-tau2/bin/python -m agent_exp.reanalysis5 --out runs/$d --domain $d --only binom audit split clean payback judges > runs/$d/r6binom.log 2>&1
done
touch runs/r6binom.done
