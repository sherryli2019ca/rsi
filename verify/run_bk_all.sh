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
NARMS=$(git show "$COMMIT:verify/bk.py" | PYTHONPATH= $PY -c "
import ast, sys
m = ast.parse(sys.stdin.read())
v = {t.id: n.value for n in m.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
d = ast.literal_eval(v['DESIGNS']); print(len(d[ast.literal_eval(v['DESIGN'])]['arms']))")
for ((i = 0; i < ${#BLOCKS[@]}; i += 2)); do
  wave="${BLOCKS[*]:i:2}"
  for B in $wave; do
    C=/home/user/bk_${B}_cal
    [ -d "$C" ] || git worktree add --detach "$C" "$COMMIT" > /dev/null 2>&1 || exit 1
    (cd "$C" && mkdir -p runs && BLOCK=$B setsid nohup bash verify/run_bk.sh >> runs/bk_block.out 2>&1 < /dev/null &)
  done
  sleep 10
  # the next wave starts when this wave's loops have finished their rounds
  need=$(( $(wc -w <<< "$wave") * NARMS ))
  while :; do
    n=0
    for B in $wave; do
      n=$(( n + $(ls /home/user/bk_${B}_*/runs/bk_loop.done 2>/dev/null | wc -l) ))
    done
    [ $n -ge $need ] && break
    if ! pgrep -f "bash verify/run_bk(_loop)?\\.sh" > /dev/null; then
      echo "wave $wave stopped before its loops finished ($n of $need); see runs/*.out"; exit 1
    fi
    sleep 120
  done
done
