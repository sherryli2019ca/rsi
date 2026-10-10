"""Zero-cost analyses for the eleventh review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r11 joint --traj r1=DIR r2=DIR r3=DIR [--json results/r11/report.json]

joint: the live-minus-offline contrast of verify/posthoc_r10.py (live: the IL
comparison, seqfull minus full transfer at round 10, 6 v 6 loops; offline: the
matched airline rounds 0-9 of r1 to r3, seqfull minus full summed over ten
rounds) with one joint bootstrap instead of two independent ones. Each draw
resamples the 20 held-out tasks once and uses them everywhere, and resamples
the units within strata that keep shared units shared: the six sequential
loops; the four full loops that appear only live (f1..f4); r2 and r3, which
appear in the live full arm and in the offline analysis (a drawn unit enters
both); and r1 (offline only). Within each drawn offline trajectory the rounds
are resampled in incumbent blocks, as in the offline analysis; a variant keeps
the rounds fixed. For r2 and r3, full evaluation's ten round values sum, task
by task, to the trajectory's transfer at round 10, so with the rounds fixed
the shared part cancels exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import posthoc_r10 as R

A = R.A


def _round_diffs(T, ts):
    """Per round: seqfull minus full round value on the task multiset ts."""
    return [A.round_value(r, "seqfull", ts) - A.round_value(r, "full", ts) for r in T["rows"]]


def _offline_sum(T, block_of, ts, rng, fixed_rounds):
    rows = T["rows"]
    if fixed_rounds:
        idx = range(len(rows))
    else:
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(block_of[T["name"]][row["t"]], []).append(i)
        keys = list(groups)
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
    d = [A.round_value(rows[i], "seqfull", ts) - A.round_value(rows[i], "full", ts) for i in idx]
    return 10 * float(np.mean(d))


def joint(trajs: dict, js: Path | None, B: int = 4000, seed: int = 21) -> dict:
    tables, block_of = R._merged(trajs, ["r1", "r2", "r3"], R.DOM, R.T_LAST)
    off_T = {T["name"].split("/")[0]: T for T in tables}
    arms, allr, ho, tasks, delta, new = R._il()
    S = list(arms["seq"])
    F_only = [n for n in arms["full"] if n not in ("r2", "r3")]
    shared = ["r2", "r3"]
    tf = lambda n, ts: float(np.mean([ho[n][0][t] - ho[n][1][t] for t in ts]))

    # point estimates and the exact identity for the shared units
    ident = {n: (tf(n, tasks), sum(A.round_value(r, "full") for r in off_T[n]["rows"])) for n in shared}
    live_pt = np.mean([tf(n, tasks) for n in S]) - np.mean([tf(n, tasks) for n in F_only + shared])
    off_pt = float(np.mean([10 * np.mean(_round_diffs(off_T[n], None)) for n in ("r1", "r2", "r3")]))
    res = {"live": 100 * float(live_pt), "offline_sum10": 100 * off_pt,
           "difference": 100 * float(live_pt - off_pt),
           "identity_full_sum_equals_transfer": {n: [100 * a, 100 * b] for n, (a, b) in ident.items()}}
    rng = np.random.default_rng(seed)
    for label, fixed in (("joint", False), ("joint_fixed_rounds", True)):
        live, off, gap = [], [], []
        for _ in range(B):
            ts = list(rng.choice(tasks, len(tasks)))
            s = list(rng.choice(S, len(S)))
            f = list(rng.choice(F_only, len(F_only)))
            sh = list(rng.choice(shared, len(shared)))
            lv = np.mean([tf(n, ts) for n in s]) - np.mean([tf(n, ts) for n in f + sh])
            of = np.mean([_offline_sum(off_T[n], block_of, ts, rng, fixed) for n in ["r1"] + sh])
            live.append(lv)
            off.append(of)
            gap.append(lv - of)
        pc = lambda x: [100 * float(np.percentile(x, 5)), 100 * float(np.percentile(x, 95))]
        res[label] = {"live_ci90": pc(live), "offline_ci90": pc(off), "difference_ci90": pc(gap),
                      "p_live_below_offline": float(np.mean(np.array(gap) < 0)),
                      "corr_live_offline": float(np.corrcoef(live, off)[0, 1])}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("joint")
    j.add_argument("--traj", nargs="+", required=True)
    j.add_argument("--json", default="results/r11/report.json")
    a = ap.parse_args()
    if a.cmd == "joint":
        joint(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
