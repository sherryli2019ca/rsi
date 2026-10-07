"""Analysis of the second-review held-out runs (experiments C, D, F on retail; B
is summarised by heldout_breakdown.py on airline). No new episodes.

  C  each policy's set with vs without the step-limit patch, same episodes
  D  the Apply-attributed set vs the same set with one patch left out
  F  clean (no injected fault) episodes, old and new pooled: each policy vs
     Apply attributed and vs the unpatched agent, with noninferiority at 0.05

Differences are paired on the same episodes (C, D, old clean episodes) or the
same (task, trial) slots (new clean episodes), with standard errors clustered
by task.

  python -m agent_exp.review2_report --out runs/tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from agent_exp.bank import Domain, _load, _save
from agent_exp.review2_runs import F_METHODS, STEP, heldout_episodes

MARGIN = 0.05


def clustered(d, tasks):
    d, tasks = np.asarray(d, float), np.asarray(tasks)
    m = d.mean()
    sums = np.array([(d[tasks == u] - m).sum() for u in np.unique(tasks)])
    se = float(np.sqrt((sums ** 2).sum()) / len(d))
    return [float(m), se, float(m - 1.645 * se), float(m + 1.645 * se)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    args = ap.parse_args()
    D = Domain(args.domain)
    H = _load(os.path.join(args.out, "heldout.json"))
    run = {k: np.array(v, float) for k, v in H["runs"].items()}
    trials = len(run["none"]) // len(D.split["test"])
    test = heldout_episodes(D, trials)
    tasks = np.array([t for t, _, _ in test])
    var = np.array([str(v) for _, v, _ in test])
    groups = {"all": var == var, "clean": var == "None", "injected": var != "None",
              "no_step_fault": var != "F_steps", "step_fault": var == "F_steps"}
    out = {"n": {g: int(s.sum()) for g, s in groups.items()}}

    def diff(a, b):
        return {g: clustered(a[s] - b[s], tasks[s]) for g, s in groups.items()}

    # C: step-limit patch removed
    C = {}
    for m, key in H.get("nostep", {}).items():
        full = run[H["sets"][f"{m}|10|0"]]
        ns = run[key]
        C[m] = {"rate_full": {g: float(full[s].mean()) for g, s in groups.items()},
                "rate_nostep": {g: float(ns[s].mean()) for g, s in groups.items()},
                "full_minus_nostep": diff(full, ns),
                "nostep_minus_unpatched": diff(ns, run["none"]),
                "n_patches": len(json.loads(key)) if key != "none" else 0}
    out["C"] = C

    # D: leave one patch out of the Apply set
    full_key = H["sets"]["LLM-only|10|0"]
    full = run[full_key]
    Dd = {"full_minus_unpatched": diff(full, run["none"])}
    contrib = {}
    for name, key in H.get("loo", {}).items():
        contrib[name[1:]] = diff(full, run[key])
    Dd["contribution"] = contrib
    # sum of single-patch contributions vs the whole set's effect (interaction)
    for g, s in groups.items():
        tot = sum(full[s] - run[H["loo"][f"-{c}"]][s] for c in contrib)
        inter = (full[s] - run["none"][s]) - tot
        Dd.setdefault("interaction", {})[g] = clustered(inter, tasks[s])
    out["D"] = Dd

    # F: clean episodes, old (heldout.json, seed-averaged) and new (heldout_clean.json)
    Fc = _load(os.path.join(args.out, "heldout_clean.json"))
    if Fc:
        cl = groups["clean"]
        new_tasks = np.array([t for t, _ in Fc["episodes"]])
        old = {"unpatched": run["none"][cl]}
        for m in F_METHODS:
            keys = [v for k, v in H["sets"].items() if k.startswith(f"{m}|10|")]
            old[m] = np.mean([run[k] for k in keys], 0)[cl]
        new = {m: np.array(v, float) for m, v in Fc["runs"].items()}
        allt = np.concatenate([tasks[cl], new_tasks])
        F = {"n_old": int(cl.sum()), "n_new": len(new_tasks),
             "drift_unpatched_new_minus_old": [float(new["unpatched"].mean() - old["unpatched"].mean())]}
        for m in old:
            r = {"old": float(old[m].mean()), "new": float(new[m].mean()),
                 "pooled": float(np.concatenate([old[m], new[m]]).mean())}
            for ref in ("unpatched", "LLM-only"):
                if m == ref:
                    continue
                d = np.concatenate([old[m] - old[ref], new[m] - new[ref]])
                c = clustered(d, allt)
                r[f"minus_{ref}"] = c + [bool(c[3] < MARGIN)] if ref == "LLM-only" else c
                r[f"minus_{ref}_new_only"] = clustered(new[m] - new[ref], new_tasks)
            F[m] = r
        out["F"] = F
    _save(os.path.join(args.out, "review2_heldout.json"), out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
