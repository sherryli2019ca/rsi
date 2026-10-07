#!/bin/bash
# Live comparison of RRSI's own selection with a verification-evidence rule.
# Every arm starts from ONE shared evaluation of the base harness (same evolve
# trials, same noise band) and then evolves on its own branch namespace, with
# the same domain, models, T, m, k and cost accounting (r<t>/selection.json,
# <arm>/<domain>.usage.jsonl). Large runs need approval of their quote first.
#
#   verify/run_compare.sh <domain> <selection> [rrsi.py options, e.g. --T 5]
#     selection: full     native RRSI: every candidate evaluated on the whole
#                         evolve set, k trials per task (the control)
#                net@40 | replaynull@40 | replay@40 | sample@40 | judge | none
#
# Output: runs/compare/sel_<selection>/<domain>/ (e.g. sel_full, sel_net40),
# shared base in runs/compare/base/<domain>/. Held-out deployment afterwards:
#   python -m verify.deploy --domain <domain> --ref evolve/sel_net40/<domain> --k 10
# Env: COMPARE_ROOT (default runs/compare), CALIB_JOBS (default three base
# evaluations: base,heldout_base2,heldout_base3; "base" alone bootstraps),
# BASE_FROM (runs root of an existing native run, e.g. /home/user/e1/runs/rrsi:
# its base evaluation and noise band are reused, and that run is the native
# control, so only the evidence arm has to be run).
set -eu
D=$1; SEL=$2; shift 2
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
PY=${RRSI_PY:-/home/user/venv-tau2/bin/python}
ROOT=${COMPARE_ROOT:-runs/compare}
ARM=sel_$(echo "$SEL" | tr -d '@')
CAL=${CALIB_JOBS:-base,heldout_base2,heldout_base3}
export PYTHONPATH="$HERE"
B=${BASE_FROM:-$ROOT/base}
A=$ROOT/$ARM
mkdir -p "$A/$D/logs"

# 1) shared base evaluation and noise band (once per domain)
if [ -z "${BASE_FROM:-}" ] && { [ ! -f "$B/$D/calibration.json" ] || [ ! -f "$B/$D/calibrated_$CAL" ]; }; then
  mkdir -p "$B/$D/logs"
  export RRSI_USAGE_LOG=$B/$D.usage.jsonl
  [ -f "$B/$D/frontier.json" ] || \
    $PY rrsi.py --domain "$D" --runs "$B" --branch-ns cmp_base "$@" baseline --job base
  for J in ${CAL//,/ }; do
    [ "$J" = base ] && continue
    [ -f "$B/$D/jobs/$J/eval.json" ] || \
      $PY rrsi.py --domain "$D" --runs "$B" --branch-ns cmp_base "$@" heldout \
        --label "${J#heldout_}" --set evolve --ref "evolve/cmp_base/$D"
  done
  $PY rrsi.py --domain "$D" --runs "$B" --branch-ns cmp_base "$@" calibrate --jobs "$CAL"
  touch "$B/$D/calibrated_$CAL"
fi

# 2) seed the arm with the base at t = 0 (same commit, trials and noise band)
if [ ! -f "$A/$D/frontier.json" ]; then
  mkdir -p "$A/$D/jobs"
  cp -r "$B/$D/jobs/base" "$A/$D/jobs/"
  cp "$B/$D/calibration.json" "$A/$D/"
  head -1 "$B/$D/history.jsonl" > "$A/$D/history.jsonl"
  C0=$($PY - "$B/$D" "$A/$D" "$SEL" "$ARM" <<'PYEOF'
import json, sys
src, dst, sel, ns = sys.argv[1:]
fr = json.load(open(f"{src}/frontier.json"))
t0 = fr["trajectory"][0]
ev = json.load(open(f"{src}/jobs/{t0['job']}/eval.json"))
inc = {"t": 0, "commit": t0["commit"], "job": t0["job"], "S": ev["S"], "C": ev["C"],
       "extra": ev.get("extra") or {}}
if fr["incumbent"]["t"] == 0:
    inc["harness_tree"] = fr["incumbent"]["harness_tree"]
cfg = dict(fr["config"], selection=sel, branch_ns=ns)
json.dump({"domain": fr["domain"], "incumbent": inc, "S_star": ev["S"], "trajectory": [t0],
           "config": cfg}, open(f"{dst}/frontier.json", "w"), indent=1)
print(t0["commit"])
PYEOF
)
  git rev-parse -q --verify "refs/heads/evolve/$ARM/$D" >/dev/null || git branch "evolve/$ARM/$D" "$C0"
  $PY - "$A/$D/frontier.json" "$D" <<'PYEOF'
import json, subprocess, sys
p, d = sys.argv[1:]
fr = json.load(open(p))
if "harness_tree" not in fr["incumbent"]:
    out = subprocess.run(["git", "rev-parse", f"{fr['incumbent']['commit']}:domains/{d}/harness"],
                         capture_output=True, text=True, check=True).stdout.strip()
    fr["incumbent"]["harness_tree"] = out[:12]
    json.dump(fr, open(p, "w"), indent=1)
PYEOF
fi

# 3) evolve with the arm's selection rule
RRSI_USAGE_LOG=$A/$D.usage.jsonl $PY rrsi.py --domain "$D" --runs "$A" --branch-ns "$ARM" \
  --selection "$SEL" "$@" run
