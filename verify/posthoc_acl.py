"""Zero-cost analysis for the ACL-form review of paper 1 v25 (post hoc, not registered).

  python -m verify.posthoc_acl seqsim --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_acl report --traj r1=DIR/runs r2=... r3=... r4=... e3=... [--json results/acl/report.json]

Question 3 of the review: is the sequential rule's miscalibration specific to
RRSI's in-band cost/novelty branch? seqsim re-decides the recorded rounds under
a success-only acceptance rule, RRSI's Algorithm 2 without that branch
(w_s = w_c = w_n = 0): a candidate is admitted only if S >= S* - delta,
dS > delta and the cost budget holds, at the calibrated band. Full evaluation
admits by that rule, and the registered sequential rule (seqfull) and the
cost-aware rule (seqcost) predict it, with posthoc_r13's orders, seeds, gamma
and batches (at the calibrated band and RRSI's own weights, posthoc_r13
reproduces the recorded decisions). Writes <out>/e1_acl_succ.json.

report: per set (r2+r3; airline r1-r4; AppWorld), for RRSI's rule (from
posthoc_r13's e1_r13_seq.json at the calibrated band) and the success-only
rule, full evaluation's acceptances, and for each sequential rule its recall
of them, the share of full evaluation's admitted candidates it stops, its
candidate episodes as a share of full evaluation's, and its decision value
minus full evaluation's (block bootstrap, 90%).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import analyze as A
from . import posthoc_e1 as P
from . import posthoc_r13 as R13
from .posthoc_r7 import _load_tables

SUCC = {"w_s": 0.0, "w_c": 0.0, "w_n": 0.0}
DEST = "e1_acl_succ.json"
RULESETS = (("rrsi", "e1_r13_seq.json"), ("success_only", DEST))


def _seqsim_main(M: int = 400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(sys.argv[1:])
    bands = R13._bands
    R13._bands = lambda d: {"m1": d}
    tables, crow = [], []
    try:
        for name in a.domains.split(","):
            D = A.load(name, Path(a.runs), Path(a.out))
            cfg = {**D["cfg"], **SUCC}
            rows = []
            for R in D["rounds"]:
                probs, eps, cand, adm, _ = R13._sim_round(R, cfg, M, 1000 * R["t"])
                rows.append({"t": R["t"], "probs": probs, "delta": R["delta"],
                             "cost": {k: {"episodes": e, "episode_equivalents": e} for k, e in eps.items()}})
                for v, C in R["cands"].items():
                    crow.append({"domain": name, "t": R["t"], "cand": v, "N": len(A._pairs(R["inc"], C["full"])),
                                 "dS": C["decision"].get("delta_S"),
                                 "admissible": {b: v in x for b, x in adm.items()},
                                 **{k: {"p_drop": x[0] / M, "episodes": x[1] / M} for k, x in cand[v].items()}})
            tables.append({"name": name, "rows": rows})
            print(f"{name}: {len(rows)} rounds")
    finally:
        R13._bands = bands
    (Path(a.out) / "e1_analysis.json").write_text(json.dumps({"tables": tables, "cands": crow, "M": M}, indent=1))


def seqsim(runs: Path, out: Path, domains: str):
    orig = A.main

    def hook(mod):
        mod.main = _seqsim_main
    try:
        P._guard(runs, out, domains=domains, dest=DEST, hook=hook)
    finally:
        A.main = orig


def _merged(trajs: dict, names, dom, fname: str):
    tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
    sims = {}
    for n in names:
        for T in json.loads((Path(trajs[n]) / "verify" / fname).read_text())["tables"]:
            sims[f"{n}/{T['name']}"] = {r["t"]: r for r in T["rows"]}
    out = []
    for T in tables:
        if dom and not T["name"].endswith("/" + dom):
            continue
        rows = []
        for r in T["rows"]:
            s = sims[T["name"]][r["t"]]
            rows.append({**r, "probs": s["probs"], "cost": s["cost"], "delta": s["delta"]})
        out.append({**T, "rows": rows})
    return out, block_of


def report(trajs: dict, js: Path | None):
    res = {}
    for name, names, dom, _ in R13.SETS:
        if name == "airline_mult" or not all(n in trajs for n in names):
            continue
        res[name] = {}
        for rs, fname in RULESETS:
            crow = []
            for n in names:
                crow += [{**r, "key": f"{n}/{r['domain']}"}
                         for r in json.loads((Path(trajs[n]) / "verify" / fname).read_text())["cands"]
                         if dom is None or r["domain"] == dom]
            tables, block_of = _merged(trajs, names, dom, fname)
            x = R13._band_stats(tables, block_of, crow, "m1")
            x["admitted_candidates"] = sum(c["admissible"]["m1"] for c in crow)
            x["candidates"] = len(crow)
            res[name][rs] = x
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    pp = lambda x: f"{100 * x:+.2f}"
    for name, x in res.items():
        print(f"== {name}")
        for rs, y in x.items():
            line = (f"{rs:13s} full acc {y['full_accepts']:3d}/{y['rounds']} (in band {y['full_accepts_in_band']:2d}),"
                    f" admitted {y['admitted_candidates']}/{y['candidates']}  D(full) {pp(y['full_dv'][0])}")
            for r, z in y["rules"].items():
                d = z["dv_diff"]
                line += (f" | {r} recall {z['recall'] if z['recall'] is None else round(z['recall'], 2)}"
                         f" stop-adm {z['stops_admitted'] or 0:.2f} ep {z['candidate_episode_share']:.2f}"
                         f" dD {pp(d[0])} [{pp(d[1])},{pp(d[2])}]")
            print(line)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["seqsim", "report"])
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--domains")
    ap.add_argument("--traj", nargs="+")
    ap.add_argument("--json", default="results/acl/report.json")
    a = ap.parse_args()
    if a.cmd == "seqsim":
        seqsim(Path(a.runs), Path(a.out), a.domains)
    else:
        report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
