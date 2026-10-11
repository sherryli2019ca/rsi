"""Zero-cost analyses for the self-review of paper 1 v23 (post hoc, not registered).

  python -m verify.posthoc_simple sim --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_simple report --traj r1=DIR/runs r2=... r3=... r4=... e3=... [--json results/r13/simple.json]

sim: two simpler ways to halve the cost of RRSI's full evaluation, decided by
RRSI's own rule with its calibrated noise band, as a practitioner who simply ran
fewer episodes would: `single`, one of the two trials of every evolve task
(chosen at random per task), and `half`, both trials of a random half of the
evolve tasks. The candidate's score is its mean reward on those episodes, its
cost change its mean tokens there relative to the incumbent's full evaluation,
and it is rejected if its harness errors exceed 2% of the episodes it ran.
M = 400 draws per round. Writes <out>/e1_simple.json.

report: (a) for r2+r3, airline r1-r4 and AppWorld, each simple rule's episodes,
recall of full evaluation's acceptances, acceptances where full evaluation keeps
the incumbent and decision value minus full evaluation's, beside the registered
sequential rule; (b) the noise-band replays of verify/posthoc_r13.py restricted
to airline rounds 0-9 of r1-r3, the rounds whose summed values equal a loop's
transfer at round 10, as ten-round differences from full evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from . import analyze as A
from . import posthoc_e1 as P
from .posthoc_r7 import _load_tables, summarise
from .posthoc_r8 import _stats
from .posthoc_r13 import MULT, ABS

RULES = ("single", "half")
SETS = (("primary", ["r2", "r3"], None), ("airline", ["r1", "r2", "r3", "r4"], "tau2_airline"),
        ("e3", ["e3"], None))


def _decide(R: dict, cfg: dict, sub: dict, inc_tok: float):
    """RRSI's rule at the calibrated band on subsampled evaluations {v: pairs}."""
    cap = cfg.get("max_harness_error_rate", 0.02)
    best, bestS = None, None
    for v, pk in sub.items():
        if not pk or sum(isinstance(p[2], P._Err) for p in pk) > cap * len(pk):
            continue
        S = float(np.mean([p[2] for p in pk]))
        tc = [p[5] for p in pk if p[5]]
        dC = (float(np.mean(tc)) - inc_tok) / inc_tok if tc and inc_tok else 0.0
        dS = S - R["S_inc"]
        delta = R["delta"]
        if S < R["S_star"] - delta:
            continue
        nu = R["cands"][v]["decision"].get("novelty") or 0
        if dS > delta:
            ok = dC <= cfg["beta0"] + cfg["beta1"] * dS
        else:
            ok = cfg["w_s"] * dS - cfg["w_c"] * dC + cfg["w_n"] * nu > 0
        if ok and (best is None or S > bestS):
            best, bestS = v, S
    return best


def _sim_round(R: dict, cfg: dict, M: int, seed: int):
    vs = list(R["cands"])
    if not vs:
        return {k: {None: 1.0} for k in RULES}, {k: 0.0 for k in RULES}
    pairs = {v: A._pairs(R["inc"], R["cands"][v]["full"]) for v in vs}
    inc_tok = [p[6] for v in vs for p in pairs[v] if p[6]]
    inc_tok = float(np.mean(inc_tok)) if inc_tok else 0.0
    rng = random.Random(seed)
    probs = {k: {} for k in RULES}
    eps = {k: 0.0 for k in RULES}
    for _ in range(M):
        for k in RULES:
            sub = {}
            for v in vs:
                by = {}
                for p in pairs[v]:
                    by.setdefault(p[0], []).append(p)
                tasks = sorted(by)
                if k == "single":
                    sub[v] = [rng.choice(by[t]) for t in tasks]
                else:
                    keep = rng.sample(tasks, (len(tasks) + 1) // 2)
                    sub[v] = [p for t in keep for p in by[t]]
                eps[k] += len(sub[v])
            w = _decide(R, cfg, sub, inc_tok)
            probs[k][w] = probs[k].get(w, 0) + 1
    return ({k: {c: x / M for c, x in d.items()} for k, d in probs.items()},
            {k: e / M for k, e in eps.items()})


def _sim_main(M: int = 400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(sys.argv[1:])
    tables = []
    for name in a.domains.split(","):
        D = A.load(name, Path(a.runs), Path(a.out))
        rows = []
        for R in D["rounds"]:
            probs, eps = _sim_round(R, D["cfg"], M, 7000 + 1000 * R["t"])
            rows.append({"t": R["t"], "probs": probs,
                         "cost": {k: {"episodes": e, "episode_equivalents": e} for k, e in eps.items()}})
        tables.append({"name": name, "rows": rows})
        print(f"{name}: {len(rows)} rounds")
    (Path(a.out) / "e1_analysis.json").write_text(json.dumps({"tables": tables, "M": M}, indent=1))


def sim(runs: Path, out: Path, domains: str):
    orig = A.main

    def hook(mod):
        mod.main = _sim_main
    try:
        P._guard(runs, out, domains=domains, dest="e1_simple.json", hook=hook)
    finally:
        A.main = orig


def _merged(trajs: dict, names, dom=None, src="e1_simple.json", keys=None, tmax=None):
    tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
    sims = {}
    for n in names:
        for T in json.loads((Path(trajs[n]) / "verify" / src).read_text())["tables"]:
            sims[f"{n}/{T['name']}"] = {r["t"]: r for r in T["rows"]}
    out = []
    for T in tables:
        if dom and not T["name"].endswith("/" + dom):
            continue
        rows = []
        for r in T["rows"]:
            if tmax is not None and r["t"] > tmax:
                continue
            s = sims[T["name"]][r["t"]]
            pr = {k: v for k, v in s["probs"].items() if keys is None or k in keys}
            co = {k: v for k, v in s["cost"].items() if keys is None or k in keys}
            base = {k: r["probs"][k] for k in ("full", "seqfull") if k in r["probs"]}
            bco = {k: r["cost"][k] for k in base}
            rows.append({**r, "probs": {**base, **pr}, "cost": {**bco, **co}})
        out.append({**T, "rows": rows})
    return out, block_of


def report(trajs: dict, js: Path | None):
    res = {"simple": {}, "ten_rounds": {}}
    for name, names, dom in SETS:
        if not all(n in trajs for n in names):
            continue
        tables, block_of = _merged(trajs, names, dom)
        s = summarise(tables, block_of, "full")
        full_ep = s["full"]["episodes"]
        res["simple"][name] = {}
        for k in ("seqfull",) + RULES:
            st = _stats(tables, k)
            res["simple"][name][k] = {"episode_share": s[k]["episodes"] / full_ep, "recall": st["recall"],
                                      "accept_when_full_keeps": st["accept_when_full_keeps"],
                                      "agree": s[k]["agree_with_full"], "dv_diff": s[k]["diff_vs_full"]}
    # ten-round sums of the noise-band replays on airline rounds 0-9 of r1-r3
    names = ["r1", "r2", "r3"]
    if all(n in trajs for n in names):
        bands = [f"m{k:g}" for k in MULT] + [f"a{100 * x:g}" for x in ABS]
        for b in bands:
            keys = [f"{r}|{b}" for r in ("full", "seqfull", "seqcost")]
            tables, block_of = _merged(trajs, names, "tau2_airline", "e1_r13_seq.json", keys, tmax=9)
            for T in tables:
                for r in T["rows"]:
                    r["probs"] = {k: r["probs"][k] for k in keys}
                    r["cost"] = {k: r["cost"][k] for k in keys}
            s = summarise(tables, block_of, keys[0])
            res["ten_rounds"][b] = {
                "rounds": sum(len(T["rows"]) for T in tables),
                "full_10": 10 * s[keys[0]]["dv"][0],
                **{k.split("|")[0] + "_minus_full_10": [10 * x for x in s[k]["diff_vs_" + keys[0]]]
                   for k in keys[1:]}}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    pp = lambda x: f"{100 * x:+.2f}"
    for name, x in res["simple"].items():
        print(f"== {name}")
        for k, v in x.items():
            d = v["dv_diff"]
            print(f"  {k:8s} ep {v['episode_share']:.2f} recall {v['recall']:.2f} agree {v['agree']:.2f} "
                  f"acc-when-full-keeps {v['accept_when_full_keeps']:.2f} dD {pp(d[0])} [{pp(d[1])},{pp(d[2])}]")
    print("== ten rounds, airline r1-r3 rounds 0-9 (points over ten rounds)")
    for b, v in res["ten_rounds"].items():
        a, c = v["seqfull_minus_full_10"], v["seqcost_minus_full_10"]
        print(f"  {b:6s} full {100 * v['full_10']:+.1f}  seq-full {100 * a[0]:+.2f} [{100 * a[1]:+.2f},{100 * a[2]:+.2f}]"
              f"  cost-full {100 * c[0]:+.2f} [{100 * c[1]:+.2f},{100 * c[2]:+.2f}]")
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--runs", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--domains", required=True)
        a = ap.parse_args()
        sim(Path(a.runs), Path(a.out), a.domains)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report"])
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--json", default="results/r13/simple.json")
    a = ap.parse_args()
    report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
