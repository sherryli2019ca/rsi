"""New runs requested in the second review round (approved experiments A, C, D, F;
B is bank_heldout.py on airline).

  A  baseline   unpatched agent on every regression task, `--n` clean runs each,
                run alongside the other new runs (contemporaneous, task-matched r0)
  C  nostep     retail held-out sets with the step-limit patch removed, same
                episodes as heldout.json
  D  loo        the Apply-attributed retail set with one patch left out at a time,
                same episodes as heldout.json
  F  clean      extra clean (no injected fault) retail held-out episodes for the
                unpatched agent and four policies; episode e uses the patch set of
                the policy's seed e mod (number of seeds)

  python -m agent_exp.review2_runs --stage baseline --out runs/tau2_airline --domain tau2_airline
  python -m agent_exp.review2_runs --stage nostep|loo|clean --out runs/tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading

import numpy as np

from agent_exp.bank import Domain, _load, _pmap, _save

STEP = "CFG.max_steps"
C_METHODS = ["LLM-only", "HarnessFix-gate", "Uncertainty", "CARVE", "Net"]
F_METHODS = ["LLM-only", "Net", "HarnessFix-gate", "HarnessFix"]
_lock = threading.Lock()


def heldout_episodes(D, trials):
    """The episode list of bank_heldout.py."""
    rng = random.Random(7)
    mix = [None, None] + list(D.faults)
    return [(tid, rng.choice(mix), t) for tid in D.split["test"] for t in range(trials)]


def stage_baseline(llm, D, args):
    path = os.path.join(args.out, "baseline_A.json")
    res = _load(path) or {}
    bank = _load(os.path.join(args.out, "bank.json"))
    tasks = sorted({r["task"] for v in bank["reg"].values() for r in v})
    jobs = [(t, r) for t in tasks for r in range(len(res.get(t, [])), args.n)]
    print("baseline jobs", len(jobs), "tasks", len(tasks), flush=True)

    def one(job):
        tid, _ = job
        tr = D.m.run_agent(llm, D.tasks[tid], D.m.components_with([], {}))
        with _lock:
            res.setdefault(tid, []).append(tr.reward)
        return tr.reward

    for s in range(0, len(jobs), 4 * args.workers):
        _pmap(one, jobs[s:s + 4 * args.workers], args.workers)
        with _lock:
            _save(path, res)
    ys = [y for v in res.values() for y in v]
    print("baseline fail rate", round(1 - float(np.mean(ys)), 3), len(ys), flush=True)


def run_sets(llm, D, args, named):
    """Run each named patch set on the held-out episodes, caching runs by set
    in heldout.json (as bank_heldout.py does)."""
    path = os.path.join(args.out, "heldout.json")
    res = _load(path)
    trials = len(res["runs"]["none"]) // len(D.split["test"])
    test = heldout_episodes(D, trials)
    tag = args.stage
    res.setdefault(tag, {})
    for name, ps in named.items():
        key = json.dumps(ps, sort_keys=True) if ps else "none"
        res[tag][name] = key
        if key not in res["runs"]:
            def one(job):
                tid, v, _ = job
                fl = [v] if v else []
                return D.m.run_agent(llm, D.tasks[tid], D.m.components_with(fl, ps), fl).reward
            res["runs"][key] = _pmap(one, test, args.workers)
            _save(path, res)
        print(tag, name, round(float(np.mean(res["runs"][key])), 3), flush=True)
    _save(path, res)


def stage_nostep(llm, D, args):
    res = _load(os.path.join(args.out, "heldout.json"))
    named = {}
    for m in C_METHODS:
        ps = json.loads(res["sets"][f"{m}|10|0"])
        named[m] = {k: v for k, v in ps.items() if k != STEP}
    run_sets(llm, D, args, named)


def stage_loo(llm, D, args):
    res = _load(os.path.join(args.out, "heldout.json"))
    ps = json.loads(res["sets"]["LLM-only|10|0"])
    run_sets(llm, D, args, {f"-{c}": {k: v for k, v in ps.items() if k != c} for c in ps})


def stage_clean(llm, D, args):
    held = _load(os.path.join(args.out, "heldout.json"))
    path = os.path.join(args.out, "heldout_clean.json")
    res = _load(path) or {"episodes": [], "runs": {}}
    eps = [(tid, t) for tid in D.split["test"] for t in range(args.n)]
    res["episodes"] = eps
    groups = {"unpatched": [{}]}
    for m in F_METHODS:
        keys = sorted(k for k in held["sets"] if k.startswith(f"{m}|10|"))
        groups[m] = [json.loads(held["sets"][k]) for k in keys]
    for m, sets in groups.items():
        if len(res["runs"].get(m, [])) == len(eps):
            continue

        def one(e):
            tid, _ = eps[e]
            ps = sets[e % len(sets)]
            return D.m.run_agent(llm, D.tasks[tid], D.m.components_with([], ps)).reward
        res["runs"][m] = _pmap(one, range(len(eps)), args.workers)
        _save(path, res)
        print("clean", m, round(float(np.mean(res["runs"][m])), 3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    ap.add_argument("--stage", required=True, choices=["baseline", "nostep", "loo", "clean"])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    from agent_exp.llm import LLM

    if args.n is None:
        args.n = {"baseline": 10, "clean": 3}.get(args.stage)
    llm, D = LLM(), Domain(args.domain)
    {"baseline": stage_baseline, "nostep": stage_nostep, "loo": stage_loo,
     "clean": stage_clean}[args.stage](llm, D, args)
    print(json.dumps({k: dict(llm.io)[k] for k in llm.io}))


if __name__ == "__main__":
    main()
