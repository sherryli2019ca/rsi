#!/bin/bash
# Experiment BK: all four blocks in two waves of two (eight loops at a time, as
# in IL). Usage: COMMIT=<registered> setsid nohup bash verify/run_bk_all.sh &
# (from /home/user/rsi). Resume-safe: rerun after a container restart.
set -u
COMMIT=${COMMIT:?set COMMIT to the registered commit}
cd /home/user/rsi
for wave in "b1 b2" "b3 b4"; do
  for B in $wave; do
    C=/home/user/bk_${B}_cal
    [ -d "$C" ] || git worktree add --detach "$C" "$COMMIT" > /dev/null 2>&1 || exit 1
    (cd "$C" && mkdir -p runs && BLOCK=$B setsid nohup bash verify/run_bk.sh >> runs/bk_block.out 2>&1 < /dev/null &)
  done
  sleep 10
  # the next wave starts when this wave's loops have finished their rounds
  while :; do
    n=0
    for B in $wave; do for a in f s h a; do
      [ -f /home/user/bk_${B}_$a/runs/bk_loop.done ] && n=$((n + 1))
    done; done
    [ $n -ge 8 ] && break
    if ! pgrep -f "bash verify/run_bk(_loop)?\.sh" > /dev/null; then
      echo "wave $wave stopped before its loops finished ($n of 8); see runs/*.out"; exit 1
    fi
    sleep 120
  done
done
