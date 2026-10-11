"""Zero-cost analysis for the thirteenth review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r13 seqsim --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r13 report --traj r1=DIR/runs r2=... r3=... r4=... e3=... [--json results/r13/report.json]
  python -m verify.posthoc_r13 tables [--json results/r13/report.json] [--tex paper/tables/bands.tex]

seqsim: the fidelity of the registered sequential rule (seqfull) and of the
cost-aware rule (seqcost, posthoc_r10) to full evaluation when RRSI's noise band
is set to other values than the calibrated one (question 3 of the review). For
each band delta' the recorded rounds are re-decided: full evaluation admits a
candidate if S' >= S* - delta' and RRSI's cost rule holds at delta' (and no
domain guard was violated), which at the calibrated band reproduces the
recorded decisions; the sequential rules stop with delta' and the floor
S* - delta' in their predictive probability and admit what full evaluation at
delta' admits among the candidates they do not stop. Bands are multiples of
each trajectory's calibrated band (`m0.5` ... `m3`, every domain) and fixed
values in points (`a2.5` ... `a17.5`, applied to every domain but read for
airline only). The M = 400 orders and seeds are posthoc_r10's. Writes
<out>/e1_r13_seq.json.

report: per set (r2+r3; airline r1-r4; AppWorld) and band, full evaluation's
acceptances (and the share inside the band), and for each sequential rule its
recall of full evaluation's acceptances, the share of full evaluation's
admitted candidates it stops, its candidate episodes as a share of full
evaluation's, and its decision value minus full evaluation's at that band
(block bootstrap, 90%).
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
from .posthoc_r10 import p_cost, _sc2

GAMMA = 0.05
MULT = (0.5, 1.0, 1.5, 2.0, 3.0)
ABS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175)
SETS = (("primary", ["r2", "r3"], None, "m"), ("airline", ["r1", "r2", "r3", "r4"], "tau2_airline", "a"),
        ("airline_mult", ["r1", "r2", "r3", "r4"], "tau2_airline", "m"), ("e3", ["e3"], None, "m"))


def _bands(delta: float) -> dict:
    out = {f"m{k:g}": k * delta for k in MULT}
    out.update({f"a{100 * x:g}": x for x in ABS})
    return out


def _admit(dec: dict, S_star: float, delta: float, cfg: dict) -> bool:
    """RRSI's Algorithm 2 (rrsi/selection.py judge) at noise band delta."""
    if dec.get("delta_S") is None or dec.get("S") is None or dec.get("guards"):
        return False
    if dec["S"] < S_star - delta:
        return False
    dS, dC, nu = dec["delta_S"], dec["delta_C"] or 0.0, dec.get("novelty") or 0
    if dS > delta:
        return dC <= cfg["beta0"] + cfg["beta1"] * dS
    return cfg["w_s"] * dS - cfg["w_c"] * dC + cfg["w_n"] * nu > 0


def _sim_round(R: dict, cfg: dict, M: int, seed: int):
    vs = list(R["cands"])
    bands = _bands(R["delta"])
    keys = [f"{r}|{b}" for b in bands for r in ("full", "seqfull", "seqcost")]
    probs = {k: {} for k in keys}
    eps = {k: 0.0 for k in keys}
    cand = {v: {k: [0, 0] for k in keys} for v in vs}           # drops, episodes
    adm = {b: [v for v in vs if _admit(R["cands"][v]["decision"], R["S_star"], d, cfg)] for b, d in bands.items()}
    check = sorted(adm["m1"]) == sorted(v for v in vs if R["cands"][v]["decision"].get("admissible"))
    if not vs:
        return {k: {None: 1.0} for k in keys}, {k: 0.0 for k in keys}, cand, adm, check
    cap = cfg.get("max_harness_error_rate", 0.02)
    s_inc = R["S_inc"]
    floor_var = s_inc * (1 - s_inc)
    pairs = {v: A._pairs(R["inc"], R["cands"][v]["full"]) for v in vs}
    S = lambda v: R["cands"][v]["decision"].get("S") or 0.0
    for b in bands:
        w = max(adm[b], key=S) if adm[b] else None
        probs[f"full|{b}"] = {w: 1.0}
        eps[f"full|{b}"] = float(sum(len(pairs[v]) for v in vs))
        for v in vs:
            cand[v][f"full|{b}"][1] = len(pairs[v]) * M
    rng = random.Random(seed)
    for _ in range(M):
        orders = {}
        for v in vs:                # the same draws as posthoc_e1's seq_probs and posthoc_r10
            o = list(pairs[v])
            rng.shuffle(o)
            orders[v] = o
        for b, delta in bands.items():
            L = R["S_star"] - delta - s_inc
            for rule in ("seqfull", "seqcost"):
                k = f"{rule}|{b}"
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
                        if rule == "seqfull":
                            p = P.p_admissible(m, s2, n, N, delta, L, dC, nu, cfg)
                        else:
                            p = p_cost(m, s2, n, N, delta, L, dC, _sc2(pk), nu, cfg)
                        if p < GAMMA:
                            dropped = True
                            break
                    eps[k] += n
                    cand[v][k][1] += n
                    if dropped:
                        cand[v][k][0] += 1
                    else:
                        alive.append(v)
                ok = [v for v in alive if v in adm[b]]
                w = max(ok, key=S) if ok else None
                probs[k][w] = probs[k].get(w, 0) + 1
    for b in bands:
        for rule in ("seqfull", "seqcost"):
            k = f"{rule}|{b}"
            probs[k] = {c: x / M for c, x in probs[k].items()}
            eps[k] /= M
    return probs, eps, cand, adm, check


def _seqsim_main(M: int = 400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(sys.argv[1:])
    tables, crow, mism = [], [], 0
    for name in a.domains.split(","):
        D = A.load(name, Path(a.runs), Path(a.out))
        cfg = D["cfg"]
        rows = []
        for R in D["rounds"]:
            probs, eps, cand, adm, check = _sim_round(R, cfg, M, 1000 * R["t"])
            mism += not check
            rows.append({"t": R["t"], "probs": probs, "delta": R["delta"],
                         "cost": {k: {"episodes": e, "episode_equivalents": e} for k, e in eps.items()}})
            for v, C in R["cands"].items():
                dec = C["decision"]
                crow.append({"domain": name, "t": R["t"], "cand": v, "N": len(A._pairs(R["inc"], C["full"])),
                             "dS": dec.get("delta_S"),
                             "admissible": {b: v in x for b, x in adm.items()},
                             **{k: {"p_drop": x[0] / M, "episodes": x[1] / M} for k, x in cand[v].items()}})
        tables.append({"name": name, "rows": rows})
        print(f"{name}: {len(rows)} rounds")
    print(f"rounds whose re-decided admissions at the calibrated band differ from the record: {mism}")
    (Path(a.out) / "e1_analysis.json").write_text(json.dumps({"tables": tables, "cands": crow, "M": M,
                                                             "mismatch_at_calibrated_band": mism}, indent=1))


def seqsim(runs: Path, out: Path, domains: str):
    orig = A.main

    def hook(mod):
        mod.main = _seqsim_main
    try:
        P._guard(runs, out, domains=domains, dest="e1_r13_seq.json", hook=hook)
    finally:
        A.main = orig


def _merged(trajs: dict, names, dom=None):
    tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
    sims = {}
    for n in names:
        for T in json.loads((Path(trajs[n]) / "verify" / "e1_r13_seq.json").read_text())["tables"]:
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


def _keep(c) -> bool:
    return c in (None, "null")


def _band_stats(tables, block_of, crow, b: str) -> dict:
    keys = [f"{r}|{b}" for r in ("full", "seqfull", "seqcost")]
    sub = [{**T, "rows": [{**r, "probs": {k: r["probs"][k] for k in keys},
                           "cost": {k: r["cost"][k] for k in keys}} for r in T["rows"]]} for T in tables]
    s = summarise(sub, block_of, keys[0])
    full_acc, inband = [], 0
    rec = {k: [] for k in keys[1:]}
    for T in sub:
        for r in T["rows"]:
            f = r["probs"][keys[0]]
            fc = max(f, key=f.get)
            if _keep(fc):
                continue
            full_acc.append((T["name"], r["t"], fc))
            for k in rec:
                rec[k].append(r["probs"][k].get(fc, 0.0))
    deltas = {(T["name"], r["t"]): r.get("delta") for T in sub for r in T["rows"]}
    adm = np.array([c["admissible"][b] for c in crow], float)
    N = np.array([c["N"] for c in crow], float)
    out = {"rounds": sum(len(T["rows"]) for T in sub), "full_accepts": len(full_acc),
           "full_dv": s[keys[0]]["dv"], "rules": {}}
    dS = {(c["key"], c["t"], c["cand"]): c["dS"] for c in crow}
    for name, t, fc in full_acc:
        bv = _bands(deltas[(name, t)] or 0.0)[b]
        x = dS.get((name, t, fc))
        inband += x is not None and x <= bv
    out["full_accepts_in_band"] = inband
    for k in keys[1:]:
        pd = np.array([c[k]["p_drop"] for c in crow])
        ep = np.array([c[k]["episodes"] for c in crow])
        out["rules"][k.split("|")[0]] = {
            "recall": float(np.mean(rec[k])) if rec[k] else None,
            "agree": s[k]["agree_with_" + keys[0]],
            "stops_admitted": float((pd * adm).sum() / adm.sum()) if adm.sum() else None,
            "stopped_admitted": float((pd * adm).sum() / pd.sum()) if pd.sum() else None,
            "candidate_episode_share": float(ep.sum() / N.sum()),
            "dv": s[k]["dv"], "dv_diff": s[k]["diff_vs_" + keys[0]]}
    return out


def report(trajs: dict, js: Path | None):
    res = {}
    for name, names, dom, fam in SETS:
        if not all(n in trajs for n in names):
            continue
        crow = []
        for n in names:
            crow += [{**r, "key": f"{n}/{r['domain']}"}
                     for r in json.loads((Path(trajs[n]) / "verify" / "e1_r13_seq.json").read_text())["cands"]
                     if dom is None or r["domain"] == dom]
        tables, block_of = _merged(trajs, names, dom)
        bands = [f"m{k:g}" for k in MULT] if fam == "m" else [f"a{100 * x:g}" for x in ABS]
        res[name] = {"bands": {b: _band_stats(tables, block_of, crow, b) for b in bands},
                     "calibrated": {T["name"]: T["rows"][0].get("delta") for T in tables if T["rows"]}}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    pp = lambda x: f"{100 * x:+.2f}"
    for name, x in res.items():
        print(f"== {name}  calibrated bands {({k: round(100 * v, 1) for k, v in x['calibrated'].items()})}")
        for b, y in x["bands"].items():
            line = (f"{b:6s} full acc {y['full_accepts']:3d}/{y['rounds']} (in band {y['full_accepts_in_band']:2d})"
                    f"  D(full) {pp(y['full_dv'][0])}")
            for r, z in y["rules"].items():
                d = z["dv_diff"]
                line += (f" | {r} recall {z['recall'] if z['recall'] is None else round(z['recall'], 2)}"
                         f" stop-adm {z['stops_admitted'] or 0:.2f} ep {z['candidate_episode_share']:.2f}"
                         f" dD {pp(d[0])} [{pp(d[1])},{pp(d[2])}]")
            print(line)
    return res


def tables(js: Path, tex: Path):
    res = json.loads(js.read_text())
    lab = {"primary": "r2+r3", "airline": "Airline r1--r4", "e3": "AppWorld"}
    pc = lambda x: "--" if x is None else f"{100 * x:.0f}"
    dd = lambda x: f"${100 * x:+.2f}$".replace("+0.00", "0.00").replace("-0.00", "0.00")
    rows = []
    for name in ("primary", "airline", "e3"):
        if name not in res:
            continue
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{lab[name]}}}}} \\\\")
        for b, y in res[name]["bands"].items():
            band = (f"${b[1:]}\\times$" if b.startswith("m") else f"{float(b[1:]):g}")
            r, c = y["rules"]["seqfull"], y["rules"]["seqcost"]
            rows.append(f"{band} & {y['full_accepts']} ({y['full_accepts_in_band']}) & {pc(r['recall'])} & "
                        f"{pc(r['candidate_episode_share'])} & {dd(r['dv_diff'][0])} & {pc(c['recall'])} & "
                        f"{pc(c['candidate_episode_share'])} & {dd(c['dv_diff'][0])} \\\\")
    out = ("\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
           " & & \\multicolumn{3}{c}{Registered rule} & \\multicolumn{3}{c}{Cost-aware rule} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n"
           "Band & Full acc. & Recall & Ep. & $\\Delta D$ & Recall & Ep. & $\\Delta D$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tex.write_text(out)
    print(out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "tables":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--json", default="results/r13/report.json")
        ap.add_argument("--tex", default="paper/tables/bands.tex")
        a = ap.parse_args()
        tables(Path(a.json), Path(a.tex))
        return
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
    ap.add_argument("--json", default="results/r13/report.json")
    a = ap.parse_args()
    report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
