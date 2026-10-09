"""How the Phase B proposers read the traces (post hoc; not registered).

  python -m attrib.phaseb_reading [--out results/attrib/phaseB_reading.json]

From each draft's action log (r<t>/<A|B>/proposal.json under every Phase B
run copy): read_trace calls per draft, the share of failing traces read, the
share of reads of a failing trace that ask for a window of at most 5 steps (a
read without a window returns the whole trace),
and how often a read's window covers the step each attribution method named
for that trace (one step of slack, since read_trace counts steps from 1). The
no-attribution and RRSI groups give the coverage a proposer reaches without
being shown the step. The post hoc oracle group (largest rescue gain of each
trace's own profile) is included once its rounds exist.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from attrib.phaseb import GROUPS, NOREAD_GROUPS, POSTHOC_GROUPS, STATES, STEP_GROUPS, parse_state, run_dir

OUT = Path(__file__).resolve().parents[1] / "results" / "attrib" / "phaseB_reading.json"


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _posthoc() -> tuple:
    return tuple(g for g in POSTHOC_GROUPS
                 if g not in NOREAD_GROUPS and all((run_dir(g, s) / f"r{parse_state(s)[2]}" / "A").exists() for s in STATES))


def named_steps() -> dict:
    out = {}
    for g in STEP_GROUPS + _posthoc():
        for s in STATES:
            t = parse_state(s)[2]
            for p in (run_dir(g, s) / f"r{t}" / "analysis" / "digests").glob("*_failure.json"):
                d = json.loads(p.read_text())
                out[(g, s, str(d["task_id"]))] = d["decisive_step"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    named = named_steps()
    named_by = STEP_GROUPS + _posthoc()
    res = {}
    for g in GROUPS + _posthoc():
        st = {"drafts": 0, "reads": [], "failing_read": [], "narrow": [],
              **{f"covers_{m}": [] for m in named_by}}
        for s in STATES:
            t = parse_state(s)[2]
            fails = {k[2] for k in named if k[1] == s and k[0] in STEP_GROUPS}
            for v in "AB":
                p = run_dir(g, s) / f"r{t}" / v / "proposal.json"
                if not p.exists():
                    continue
                log = json.loads(p.read_text())["log"]
                log = ast.literal_eval(log) if isinstance(log, str) else log
                reads = [a for a in log if a.get("action") == "read_trace"]
                fr = [a for a in reads if str(a.get("task_id")) in fails]
                st["drafts"] += 1
                st["reads"].append(len(reads))
                st["failing_read"].append(len({str(a["task_id"]) for a in fr}) / len(fails))
                for a in fr:
                    lo, hi = a.get("from_step"), a.get("to_step")
                    lo, hi = (lo if lo is not None else 0), (hi if hi is not None else 10 ** 6)  # whole trace
                    st["narrow"].append(hi - lo <= 4)
                    for m in named_by:
                        k = named.get((m, s, str(a["task_id"])))
                        if k is not None:
                            st[f"covers_{m}"].append(lo <= k + 1 and k <= hi)
        res[g] = {"drafts": st["drafts"], "reads_per_draft": _mean(st["reads"]),
                  "share_failing_traces_read": _mean(st["failing_read"]),
                  "failing_reads": len(st["narrow"]), "share_narrow": _mean(st["narrow"]),
                  **{f"share_covering_{m}": _mean(st[f"covers_{m}"]) for m in named_by}}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
