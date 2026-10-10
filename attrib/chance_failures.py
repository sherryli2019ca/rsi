"""Rescuable failures that often recover without correction (post hoc).

A failure counts as a chance failure here when uncorrected replays from some step
succeeded at least half the time (null success >= 0.5 at a step with null
replays; steps without an applicable correction have none). Reports how many of
the registered rescuable failures are chance failures, by domain, and the
registered ranking on the rest.

  python -m attrib.chance_failures /home/user/attrib_runs/main --out results/attrib/chance_failures.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.admissibility import DOMAINS
from attrib.analyze import _boot, _mean, load
from attrib.robustness import LLM, METHODS, kendall

CUT = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = {d: load(Path(a.main) / d) for d in DOMAINS}
    rows, counts = [], {}
    for d, v in data.items():
        n = chance = 0
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            nulls = [s["null"] for s in g["steps"] if s.get("null") is not None]
            top = max(nulls, default=0.0)
            n += 1
            chance += top >= CUT
            r = {"cluster": f"{d}/{v['fails'][f]['task_id']}", "chance": top >= CUT}
            for m in METHODS:
                k = v["picks"][m].get(f)
                r[m] = g["R"][k] if isinstance(k, int) and 0 <= k < len(g["R"]) else 0.0
            rows.append(r)
        counts[d] = {"rescuable": n, "chance": chance}
    reg = {m: _mean([r[m] for r in rows]) for m in METHODS}
    rest = [r for r in rows if not r["chance"]]
    mean = {m: round(_mean([r[m] for r in rest]), 4) for m in METHODS}
    best = max(LLM, key=lambda m: mean[m])
    rng = random.Random(0)
    res = {"cut": CUT, "counts": counts, "chance": sum(r["chance"] for r in rows), "n": len(rows),
           "reliable": {"n": len(rest), "mean": mean, "best_llm": best,
                        "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
                        "first_write_minus_best_llm": _boot(
                            [{"d": r["first_write"] - r[best], "cluster": r["cluster"]} for r in rest],
                            lambda xs: _mean([x["d"] for x in xs]), rng),
                        "kendall_tau_vs_registered": round(kendall(mean, reg), 3)}}
    print(json.dumps({k: v for k, v in res.items() if k != "reliable"}),
          json.dumps({k: v for k, v in res["reliable"].items() if k != "mean"}))
    print(sorted(mean.items(), key=lambda x: -x[1]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
