"""Zero-cost analyses for the twelfth review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r12 seqsim --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r12 report --traj r1=DIR/runs r2=... r3=... r4=... e3=... [--json results/r12/report.json]

seqsim: the cost-aware stopping rule of verify/posthoc_r10.py (seqcost, gamma
0.05) with the final changes in success and cost jointly normal instead of
independent (question 4 of the review). Given the final dS = x, the final dC is
normal with mean dC + rho (sd_c / sd) (x - m) and variance sd_c^2 (1 - rho^2),
where m and dC are the running estimates and sd and sd_c their predictive
standard deviations, as in posthoc_r10.p_cost. rho is fixed (0, +-0.3, +-0.6),
estimated from the candidate's pairs so far (`emp`: the correlation of the
paired success difference and the paired token difference relative to the
incumbent's mean, clipped to [-0.9, 0.9], 0 before 10 pairs), or taken from
all of the candidate's pairs (`final`, a diagnostic that cannot be run).
rho = 0 is posthoc_r10's seqcost; the M = 400 orders and seeds are
posthoc_r10's. Writes <out>/e1_r12_seq.json, with each candidate's correlation
over all its pairs.

report: per set (r2+r3; airline r1-r4; AppWorld) the distribution of the
candidates' correlations, and for every variant the share of full evaluation's
admitted candidates it keeps, the share of its stops that full evaluation
admits, its episodes as a share of full evaluation's, and per round its recall
of full evaluation's acceptances and its decision value minus full evaluation's.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from . import analyze as A
from . import posthoc_e1 as P
from .posthoc_r7 import _load_tables, summarise
from .posthoc_r8 import _stats
from .posthoc_r10 import _ncdf

GAMMA = 0.05
VARIANTS = {"seqcost": 0.0, "seqcost[rho-0.6]": -0.6, "seqcost[rho-0.3]": -0.3, "seqcost[rho+0.3]": 0.3,
            "seqcost[rho+0.6]": 0.6, "seqcost[rho=emp]": "emp", "seqcost[rho=final]": "final"}
SETS = (("primary", ["r2", "r3"], None), ("airline", ["r1", "r2", "r3", "r4"], "tau2_airline"),
        ("e3", ["e3"], None))


def p_cost_rho(m: float, s2: float, n: int, N: int, delta: float, L: float, dC: float, sc2: float,
               rho: float, nu: float, cfg: dict, K: int = 400) -> float:
    """posthoc_r10.p_cost with the final dC correlated with the final dS (rho)."""
    from scipy.special import ndtri
    b0, b1 = cfg["beta0"], cfg["beta1"]
    ws, wc, wn = cfg["w_s"], cfg["w_c"], cfg["w_n"]
    r = (N - n) / N
    sd = r * math.sqrt(s2 * (1.0 / n + 1.0 / (N - n))) if n < N else 0.0
    sdc = r * math.sqrt(sc2 * (1.0 / n + 1.0 / (N - n))) if n < N else 0.0
    if sd == 0.0:
        rho = 0.0

    def c_le(y, x):                  # P(final dC <= y | final dS = x)
        y, x = np.asarray(y, float), np.asarray(x, float)
        if sdc == 0.0:
            return (dC <= y).astype(float)
        mu = dC + (rho * sdc / sd) * (x - m) if rho else dC
        return _ncdf((y - mu) / (sdc * math.sqrt(1.0 - rho * rho)))

    def mass(lo, hi, fn, open_lo):
        if sd == 0.0:
            inside = (m > lo if open_lo else m >= lo) and m <= hi
            return float(fn(np.array([m]))[0]) if inside else 0.0
        ua = float(_ncdf((lo - m) / sd)) if lo > -math.inf else 0.0
        ub = float(_ncdf((hi - m) / sd)) if hi < math.inf else 1.0
        if ub - ua < 1e-12:
            return 0.0
        u = np.clip(ua + (np.arange(K) + 0.5) / K * (ub - ua), 1e-300, 1.0 - 1e-16)
        return (ub - ua) * float(np.mean(fn(m + sd * ndtri(u))))

    pa = mass(max(delta, L), math.inf, (lambda x: c_le(b0 + b1 * x, x)) if b1 > 0
              else (lambda x: c_le(np.full_like(x, b0), x)), True)
    if L < delta:
        pb = mass(L, delta, (lambda x: c_le((ws * x + wn * nu) / wc - 1e-12, x)) if wc > 0
                  else (lambda x: ((ws * x + wn * nu) > 0).astype(float)), False)
    else:
        pb = 0.0
    return pa + pb


def _cost_terms(pk):
    """Per-pair success differences and token differences relative to the
    incumbent's mean, over pairs with both token counts (posthoc_r10._sc2)."""
    ok = [(p[2] - p[3], p[5], p[6]) for p in pk if p[5] and p[6]]
    if len(ok) < 2:
        return np.zeros(0), np.zeros(0)
    h = float(np.mean([b for _, _, b in ok]))
    return np.array([d for d, _, _ in ok], float), np.array([(a - b) / h for _, a, b in ok])


def _corr(d, e) -> float:
    if len(d) < 10 or d.std() == 0 or e.std() == 0:
        return 0.0
    return float(np.clip(np.corrcoef(d, e)[0, 1], -0.9, 0.9))


def _sim_round(R: dict, cfg: dict, M: int, seed: int):
    vs = list(R["cands"])
    probs = {k: {} for k in VARIANTS}
    eps = {k: 0.0 for k in VARIANTS}
    cand = {v: {k: [0, 0] for k in VARIANTS} for v in vs}       # drops, episodes
    if not vs:
        return {k: {None: 1.0} for k in VARIANTS}, {k: 0.0 for k in VARIANTS}, cand, {}
    cap = cfg.get("max_harness_error_rate", 0.02)
    s_inc = R["S_inc"]
    floor_var = s_inc * (1 - s_inc)
    delta, L = R["delta"], R["S_star"] - R["delta"] - s_inc
    pairs = {v: A._pairs(R["inc"], R["cands"][v]["full"]) for v in vs}
    final = {v: _corr(*_cost_terms(pairs[v])) for v in vs}
    rng = random.Random(seed)
    for _ in range(M):
        orders = {}
        for v in vs:                # the same draws as posthoc_e1's seq_probs and posthoc_r10
            o = list(pairs[v])
            rng.shuffle(o)
            orders[v] = o
        for k, rho_k in VARIANTS.items():
            alive = []
            for v in vs:
                order = orders[v]
                N = len(order)
                nu = R["cands"][v]["decision"].get("novelty") or 0
                n, dropped = 0, False
                while n < N:
                    n = min(N, n + P.SEQ_BATCH)
                    pk = order[:n]
                    if sum(isinstance(p[2], P._Err) for p in pk) > cap * N:
                        dropped = True
                        break
                    if n == N:
                        break
                    diff = [p[2] - p[3] for p in pk]
                    m = float(np.mean(diff))
                    sv = float(np.var(diff, ddof=1)) if n > 1 else 0.0
                    s2 = max(sv, floor_var)
                    dC = A._rel_cost([p[5] for p in pk], [p[6] for p in pk])
                    d, e = _cost_terms(pk)
                    sc2 = float(np.var(e, ddof=1)) if len(e) > 1 else 0.0
                    rho = (_corr(d, e) if rho_k == "emp" else final[v] if rho_k == "final" else rho_k)
                    p = p_cost_rho(m, s2, n, N, delta, L, dC, sc2, rho, nu, cfg)
                    if p < GAMMA:
                        dropped = True
                        break
                eps[k] += n
                cand[v][k][1] += n
                if dropped:
                    cand[v][k][0] += 1
                else:
                    alive.append(v)
            adm = [v for v in alive if R["cands"][v]["decision"].get("admissible")]
            w = max(adm, key=lambda v: R["cands"][v]["decision"].get("S") or 0.0) if adm else None
            probs[k][w] = probs[k].get(w, 0) + 1
    return ({k: {c: x / M for c, x in d.items()} for k, d in probs.items()},
            {k: e / M for k, e in eps.items()}, cand, final)


def _seqsim_main(M: int = 400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(sys.argv[1:])
    tables, crow = [], []
    for name in a.domains.split(","):
        D = A.load(name, Path(a.runs), Path(a.out))
        cfg = D["cfg"]
        rows = []
        for R in D["rounds"]:
            probs, eps, cand, final = _sim_round(R, cfg, M, 1000 * R["t"])
            rows.append({"t": R["t"], "probs": probs,
                         "cost": {k: {"episodes": e, "episode_equivalents": e} for k, e in eps.items()}})
            for v, C in R["cands"].items():
                dec = C["decision"]
                prs = A._pairs(R["inc"], C["full"])
                d, e = _cost_terms(prs)
                crow.append({"domain": name, "t": R["t"], "cand": v, "N": len(prs),
                             "admissible": bool(dec.get("admissible")), "selected": R["accepted"] == v,
                             "rho": final[v], "rho_raw": float(np.corrcoef(d, e)[0, 1])
                             if len(d) > 2 and d.std() > 0 and e.std() > 0 else None,
                             **{k: {"p_drop": x[0] / M, "episodes": x[1] / M} for k, x in cand[v].items()}})
        tables.append({"name": name, "rows": rows})
        print(f"{name}: {len(rows)} rounds")
    (Path(a.out) / "e1_analysis.json").write_text(json.dumps({"tables": tables, "cands": crow, "M": M},
                                                             indent=1))


def seqsim(runs: Path, out: Path, domains: str):
    orig = A.main

    def hook(mod):
        mod.main = _seqsim_main
    try:
        P._guard(runs, out, domains=domains, dest="e1_r12_seq.json", hook=hook)
    finally:
        A.main = orig


def _merged(trajs: dict, names, dom=None):
    tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
    sims = {}
    for n in names:
        for T in json.loads((Path(trajs[n]) / "verify" / "e1_r12_seq.json").read_text())["tables"]:
            sims[f"{n}/{T['name']}"] = {r["t"]: r for r in T["rows"]}
    out = []
    for T in tables:
        if dom and not T["name"].endswith("/" + dom):
            continue
        rows = []
        for r in T["rows"]:
            s = sims[T["name"]][r["t"]]
            rows.append({**r, "probs": {**r["probs"], **s["probs"]}, "cost": {**r["cost"], **s["cost"]}})
        out.append({**T, "rows": rows})
    return out, block_of


def report(trajs: dict, js: Path | None):
    res = {}
    for name, names, dom in SETS:
        crow = []
        for n in names:
            crow += [r for r in json.loads((Path(trajs[n]) / "verify" / "e1_r12_seq.json").read_text())["cands"]
                     if dom is None or r["domain"] == dom]
        raw = np.array([r["rho_raw"] for r in crow if r["rho_raw"] is not None])
        x = {"candidates": len(crow),
             "rho": {"median": float(np.median(raw)), "q10": float(np.percentile(raw, 10)),
                     "q90": float(np.percentile(raw, 90)), "share_abs_gt_0.3": float(np.mean(np.abs(raw) > 0.3))},
             "variants": {}}
        tables, block_of = _merged(trajs, names, dom)
        s = summarise(tables, block_of, "full")
        full_ep = s["full"]["episodes"]
        adm = np.array([r["admissible"] for r in crow], float)
        N = np.array([r["N"] for r in crow], float)
        for k in VARIANTS:
            pd = np.array([r[k]["p_drop"] for r in crow])
            ep = np.array([r[k]["episodes"] for r in crow])
            st = _stats(tables, k)
            x["variants"][k] = {
                "kept_admitted": float(1 - (pd * adm).sum() / adm.sum()),
                "stopped_admitted": float((pd * adm).sum() / pd.sum()) if pd.sum() else None,
                "candidate_episode_share": float(ep.sum() / N.sum()),
                "episode_share": s[k]["episodes"] / full_ep, "recall": st["recall"],
                "dv_diff": s[k]["diff_vs_full"]}
        res[name] = x
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    for name, x in res.items():
        r = x["rho"]
        print(f"{name}: {x['candidates']} candidates, correlation of paired success and cost differences "
              f"median {r['median']:+.2f} [10-90%: {r['q10']:+.2f}, {r['q90']:+.2f}], |rho|>0.3 in {r['share_abs_gt_0.3']:.0%}")
        for k, v in x["variants"].items():
            d = v["dv_diff"]
            print(f"  {k:20s} kept {v['kept_admitted']:.2f}  stop-admitted {v['stopped_admitted'] or 0:.2f}  "
                  f"episodes {v['episode_share']:.2f}  recall {v['recall']:.2f}  "
                  f"D-full {100 * d[0]:+.2f} [{100 * d[1]:+.2f}, {100 * d[2]:+.2f}]")
    return res


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "seqsim":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--runs", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--domains", required=True)
        a = ap.parse_args()
        seqsim(Path(a.runs), Path(a.out), a.domains)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report"])
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--json", default="results/r12/report.json")
    a = ap.parse_args()
    report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
