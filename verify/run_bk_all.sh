#!/bin/bash
# Experiment BK: every block of verify/bk.py's DESIGN, two blocks at a time.
# Usage: COMMIT=<registered> setsid nohup bash verify/run_bk_all.sh &
# (from /home/user/rsi). Resume-safe: rerun after a container restart.
set -u
COMMIT=${COMMIT:?set COMMIT to the registered commit}
cd /home/user/rsi
PY=/home/user/venv-tau2/bin/python
read -r -a BLOCKS <<< "$(git show "$COMMIT:verify/bk.py" | PYTHONPATH= $PY -c "
import ast, sys
src = sys.stdin.read(); m = ast.parse(src)
v = {t.id: n.value for n in m.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
d = ast.literal_eval(v['DESIGNS']); print(' '.join(d[ast.literal_eval(v['DESIGN'])]['blocks']))")"
for ((i = 0; i < ${#BLOCKS[@]}; i += 2)); do
  wave="${BLOCKS[*]:i:2}"
  for B in $wave; do
    C=/home/user/bk_${B}_cal
    [ -d "$C" ] || git worktree add --detach "$C" "$COMMIT" > /dev/null 2>&1 || exit 1
    (cd "$C" && mkdir -p runs && BLOCK=$B setsid nohup bash verify/run_bk.sh >> runs/bk_block.out 2>&1 < /dev/null &)
  done
  sleep 10
  # the next wave starts when every loop (lineage) of this wave has finished
  # its rounds; a lineage is forked only while its parent runs a round, so
  # when all lineages that exist are finished, no new one can appear
  while :; do
    tot=0; n=0
    for B in $wave; do
      for L in /home/user/bk_${B}_*/runs/bk_lineage.json; do
        [ -f "$L" ] || continue
        tot=$((tot + 1))
        [ -f "$(dirname "$L")/bk_loop.done" ] && n=$((n + 1))
      done
    done
    [ $tot -ge $(wc -w <<< "$wave") ] && [ $n -eq $tot ] && break
    if ! pgrep -f "bash verify/run_bk(_loop)?\\.sh" > /dev/null; then
      echo "wave $wave stopped before its loops finished ($n of $tot); see runs/*.out"; exit 1
    fi
    sleep 120
  done
done
