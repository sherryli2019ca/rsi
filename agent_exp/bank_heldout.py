"""Held-out success on tau2 after applying the patches each method accepts.

For each method and budget, run the bank-resampled verification for a few
seeds, collect the distinct accepted patch sets, apply each to the agent and
run it on the held-out (test-split) tasks, under the same mix of clean and
fault-injected variants as training.

  python -m agent_exp.bank_heldout --out runs/tau2_retail --budgets 10 --n_seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from agent_exp.bank import (METHODS, Domain, _load, _pmap, _save, ground_truth, problem,
                            run_method)


def patch_set(D, acc, patch, patches, post_val):
    """Component -> patch text; when several accepted cells touch one component,
    keep the cell with the larger value."""
    out, best = {}, {}
    for i, j in zip(*np.nonzero(acc)):
        cid = D.comp_ids[j]
        if cid not in out or post_val[i, j] > best[cid]:
            out[cid] = patches[f"{i},{j}"][int(patch[i, j])]
            best[cid] = post_val[i, j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--budgets", type=float, nargs="+", default=[10])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--methods", nargs="+", default=None)
    args = ap.parse_args()
    from agent_exp.llm import LLM

    llm, D = LLM(), Domain(args.domain)
    attrs = _load(os.path.join(args.out, "attributions.json"))
    tax = _load(os.path.join(args.out, "taxonomy.json"))
    bank = _load(os.path.join(args.out, "bank.json"))
    patches = _load(os.path.join(args.out, "patches.json"))
    ev = _load(os.path.join(args.out, "evaluation_rep20.json")) or \
        _load(os.path.join(args.out, "evaluation.json"))
    cats, counts, members, f = problem(D, attrs, tax)
    inb, E, Q, b = ground_truth(bank, *counts.shape)
    costs, cal, b_hat = ev["costs"], ev["calibration"], ev["b_hat"]

    # held-out episodes: each test task under `trials` variants drawn from the
    # training mix (clean twice as often as each fault)
    rng = random.Random(7)
    mix = [None, None] + list(D.faults)
    test = [(tid, rng.choice(mix), t) for tid in D.split[args.test_split]
            for t in range(args.trials)]

    path = os.path.join(args.out, "heldout.json")
    res = _load(path) or {"sets": {}, "runs": {}}

    def run_set(key, comps_patch):
        def one(job):
            tid, v, _ = job
            fl = [v] if v else []
            tr = D.m.run_agent(llm, D.tasks[tid], D.m.components_with(fl, comps_patch), fl)
            return tr.reward
        return _pmap(one, test, args.workers)

    if "none" not in res["runs"]:
        res["runs"]["none"] = run_set("none", {})
        _save(path, res)
    for m in args.methods or METHODS:
        for B in args.budgets:
            for s in range(args.n_seeds):
                acc, patch, spent = run_method(m, bank, counts, f, inb, costs, b_hat, s, B,
                                               cal=cal)
                ps = patch_set(D, acc, patch, patches, f[:, None] * np.ones_like(acc, float))
                key = json.dumps(ps, sort_keys=True)
                res["sets"][f"{m}|{int(B)}|{s}"] = key
                if key not in res["runs"]:
                    res["runs"][key] = run_set(key, ps)
                    _save(path, res)
            ys = [np.mean(res["runs"][res["sets"][f"{m}|{int(B)}|{s}"]])
                  for s in range(args.n_seeds)]
            print(m, int(B), round(float(np.mean(ys)), 3), [round(float(y), 3) for y in ys],
                  flush=True)
    print("unpatched", round(float(np.mean(res["runs"]["none"])), 3))
    _save(path, res)


if __name__ == "__main__":
    main()
