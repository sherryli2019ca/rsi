"""Zero-cost analyses for the sixteenth review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r16 report --traj r1=DIR/runs r2=... r3=... e3=... [--json results/r16/report.json]

resolution (question 2): every measured candidate of r1-r3 and E3 (the 183 of
the paper), its deployment gain over its incumbent (mean over held-out tasks of
the paired per-task difference in success rate, as in the decision values) and
a 95% bootstrap interval over held-out tasks (B = 4000). A candidate is clearly
positive if the interval lies above 0, clearly negative if below, and
unresolved otherwise; the intervals are per candidate and not adjusted, so
about 2.5% of candidates with no effect would land on each side by chance.

shared (question 1): the registered primary analysis of r2+r3 (verify/e1r.py:
blocks of rounds sharing an incumbent and held-out tasks resampled within each
trajectory and domain, one task draw per trajectory and domain, shared by every
round, candidate, incumbent and rule of it) repeated with one task draw per
domain shared by both trajectories, which also carries the correlation of
task-level gains across trajectories that evaluate on the same tasks. Decision
values, differences from full evaluation with 90% intervals and the one-sided
non-inferiority p at the registered margin of 0.75 points, Holm-adjusted
within the registered families, for both versions, at the registered
B = 2000 and at B = 10000 to show the Monte Carlo error of p values near 0.05.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analyze as A
from .e1r import NI_RULES, SEQ_NI, _tables
from .posthoc_e1 import _dv, _per_round

B_RES = 4000


def resolution(trajs: dict, seed: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    out = {"rows": [], "by_set": {}}
    for name, P in trajs.items():
        d = json.loads((P / "verify" / "e1_analysis_guard_posthoc.json").read_text())
        for T in d["tables"]:
            for row in T["rows"]:
                for v, dep in row["dep"].items():
                    x = np.array([float(g) for g in dep.values()])
                    bs = x[rng.integers(0, len(x), (B_RES, len(x)))].mean(axis=1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    cls = "positive" if lo > 0 else "negative" if hi < 0 else "unresolved"
                    out["rows"].append({"traj": name, "domain": T["name"], "t": row["t"], "variant": v,
                                        "gain": 100 * float(x.mean()), "ci95": [100 * float(lo), 100 * float(hi)],
                                        "n_tasks": len(x), "class": cls})
    groups = {}
    for r in out["rows"]:
        keys = ("e3", "all") if r["traj"] == "e3" else (f"tau2/{r['domain']}", "tau2", "all")
        for k in keys:
            groups.setdefault(k, []).append(r)
    for k, rs in groups.items():
        c = {cl: sum(1 for r in rs if r["class"] == cl) for cl in ("positive", "negative", "unresolved")}
        out["by_set"][k] = {"n": len(rs), **c,
                            "point_positive": sum(1 for r in rs if r["gain"] > 0),
                            "point_nonpositive": sum(1 for r in rs if r["gain"] <= 0),
                            "median_ci_width": float(np.median([r["ci95"][1] - r["ci95"][0] for r in rs]))}
    return out


def _per_round_shared(tables, rng, block_of):
    """As posthoc_e1._per_round with blocks, but one task draw per domain,
    shared by every trajectory of that domain."""
    draws = {}
    for T in tables:
        dom = T["name"].split("/", 1)[1]
        draws.setdefault(dom, set()).update(t for row in T["rows"] for d in row["dep"].values() for t in d)
    draws = {dom: list(rng.choice(sorted(ts), len(ts))) for dom, ts in sorted(draws.items())}
    out = {}
    for T in tables:
        rows = T["rows"]
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(block_of[T["name"]][row["t"]], []).append(i)
        keys = list(groups)
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
        tasks = draws[T["name"].split("/", 1)[1]]
        for i in idx:
            row = rows[i]
            for rule, probs in row["probs"].items():
                drun = sum(p * (row["dcr"].get(c) or 0.0) for c, p in probs.items() if c not in (None, "null"))
                out.setdefault(rule, []).append((A.round_value(row, rule, tasks), drun,
                                                 row["cost"][rule]["episode_equivalents"]))
    return out


def shared(trajs: dict, B: int = A.B) -> dict:
    tables, block_of = _tables(trajs, ["r2", "r3"])
    point = _dv(_per_round(tables))
    pc = lambda xs: [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    res = {}
    for label, seed, fn in (("registered", 2, lambda rng: _per_round(tables, rng, block_of)),
                            ("shared_tasks", 2, lambda rng: _per_round_shared(tables, rng, block_of))):
        rng = np.random.default_rng(seed)
        boots = [_dv(fn(rng)) for _ in range(B)]
        r = {"full": {"est": 100 * point["full"], "ci90": [100 * x for x in pc([b["full"] for b in boots])]}}
        for fam_name, fam in (("primary", NI_RULES), ("sequential", SEQ_NI)):
            ps, rows = {}, {}
            for rule in fam:
                if rule not in point:
                    continue
                xs = np.array([b[rule] - b["full"] for b in boots])
                ps[rule] = float(np.mean(xs <= -A.MARGIN))
                rows[rule] = {"diff": 100 * (point[rule] - point["full"]),
                              "ci90": [100 * x for x in pc(xs)], "p_ni": ps[rule]}
            for rule, p in A.holm(ps).items():
                rows[rule]["p_holm"] = p
            r[fam_name] = rows
        xs = np.array([-b["full"] for b in boots])
        r["keep"] = {"diff": -100 * point["full"], "ci90": [100 * x for x in pc(xs)],
                     "p_ni": float(np.mean(xs <= -A.MARGIN))}
        res[label] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report"])
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--json", default="results/r16/report.json")
    ap.add_argument("--B", type=int, nargs="+", default=[A.B, 10000],
                    help="bootstrap sizes for the shared-task comparison (the registered analysis used 2000)")
    a = ap.parse_args()
    trajs = {k: Path(v) for k, v in (s.split("=", 1) for s in a.traj)}
    res = {"resolution": resolution({k: trajs[k] for k in ("r1", "r2", "r3", "e3") if k in trajs}),
           "shared": {f"B={b}": shared(trajs, b) for b in a.B}}
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=1))
    print(json.dumps({"resolution": res["resolution"]["by_set"], "shared": res["shared"]}, indent=1))


if __name__ == "__main__":
    main()
