"""Post-hoc analyses for the tenth review of paper 1 (not registered).

  python -m verify.posthoc_r10 seqsim --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r10 report --traj r1=DIR r2=DIR ... [--json FILE]
  python -m verify.posthoc_r10 tables --json FILE [--tex paper/tables]

seqsim: sequential full evaluation as registered (gamma 0.05, batches of 10,
  the running cost change held fixed; `seqfull`) and two variants that can be
  run without knowing the final cost change. `seqcost` integrates the
  predictive probability over the final cost change, taken as normal around
  its running estimate with variance ((N-n)/N)^2 s_c^2 (1/n + 1/(N-n)), where
  s_c^2 is the per-pair variance of the token difference relative to the
  incumbent's mean, independent of the final dS; at gamma 0.05, 0.10 and 0.20.
  `seqscore` never stops for cost: it stops only when the predicted probability
  that the final dS reaches the floor falls below 0.05. For every round, M
  random orders of each candidate's paired episodes, common to all variants
  and the same as the guarded analysis's seqfull, give each rule's choice
  probabilities and episodes; for every candidate, the share of orders in which
  it is stopped. Writes <out>/e1_r10_seq.json.

report: (a) the matched comparison: full and sequential full evaluation on
  airline rounds 0-9 of the trajectories that full evaluation drove (r1-r3),
  where the sum of full evaluation's round values equals the trajectory's
  transfer at round 10, against the independent loops of experiment IL;
  (b) the calibrated noise band of every loop and the transfer difference
  adjusted for it; (c) accounting in the live loops: accepted changes, rounds
  in which full evaluation would have accepted (shadow), the candidates'
  complete evaluations and the early-stopped estimates of dropped candidates;
  (d) the cost-aware variants on the recorded trajectories (calibration of
  the stops, recall of full evaluation's admissions, episodes, decision value).
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
from .posthoc_r9 import _branch

GAMMA = 0.05
VARIANTS = {"seqfull": ("hold", 0.05), "seqcost": ("cost", 0.05), "seqcost[g10]": ("cost", 0.10),
            "seqcost[g20]": ("cost", 0.20), "seqscore": ("score", 0.05)}
DOM = "tau2_airline"
T_LAST = 9


def _ncdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def p_cost(m: float, s2: float, n: int, N: int, delta: float, L: float, dC: float, sc2: float,
           nu: float, cfg: dict, K: int = 400) -> float:
    """posthoc_e1.p_admissible with the final cost change normal around dC
    (variance as for dS, from the per-pair variance sc2) and independent of the
    final dS. Each region's mass in dS is exact; the cost condition is averaged
    over K quantiles of dS within it. With sc2 = 0 it equals p_admissible up to
    the quadrature."""
    from scipy.special import ndtri
    b0, b1 = cfg["beta0"], cfg["beta1"]
    ws, wc, wn = cfg["w_s"], cfg["w_c"], cfg["w_n"]
    r = (N - n) / N
    sd = r * math.sqrt(s2 * (1.0 / n + 1.0 / (N - n))) if n < N else 0.0
    sdc = r * math.sqrt(sc2 * (1.0 / n + 1.0 / (N - n))) if n < N else 0.0

    def c_le(y):                     # P(final dC <= y)
        y = np.asarray(y, float)
        return (dC <= y).astype(float) if sdc == 0.0 else _ncdf((y - dC) / sdc)

    def mass(lo, hi, fn, open_lo):
        """P(lo < dS <= hi) (or lo <= dS) times the mean of fn over it."""
        if sd == 0.0:
            inside = (m > lo if open_lo else m >= lo) and m <= hi
            return float(fn(np.array([m]))[0]) if inside else 0.0
        ua = float(_ncdf((lo - m) / sd)) if lo > -math.inf else 0.0
        ub = float(_ncdf((hi - m) / sd)) if hi < math.inf else 1.0
        if ub - ua < 1e-12:
            return 0.0
        u = np.clip(ua + (np.arange(K) + 0.5) / K * (ub - ua), 1e-300, 1.0 - 1e-16)
        return (ub - ua) * float(np.mean(fn(m + sd * ndtri(u))))

    pa = mass(max(delta, L), math.inf, (lambda x: c_le(b0 + b1 * x)) if b1 > 0
              else (lambda x: np.full_like(x, float(c_le(b0)))), True)
    if L < delta:
        pb = mass(L, delta, (lambda x: c_le((ws * x + wn * nu) / wc - 1e-12)) if wc > 0
                  else (lambda x: ((ws * x + wn * nu) > 0).astype(float)), False)
    else:
        pb = 0.0
    return pa + pb


def p_score(m: float, s2: float, n: int, N: int, L: float) -> float:
    """Predictive probability that the final dS reaches the floor L."""
    if n >= N:
        return 1.0 if m >= L else 0.0
    sd = (N - n) / N * math.sqrt(s2 * (1.0 / n + 1.0 / (N - n)))
    return float(1.0 - _ncdf((L - m) / sd)) if sd > 0 else (1.0 if m >= L else 0.0)


def _sc2(pk) -> float:
    ok = [(p[5], p[6]) for p in pk if p[5] and p[6]]
    if len(ok) < 2:
        return 0.0
    h = float(np.mean([b for _, b in ok]))
    e = [(a - b) / h for a, b in ok]
    return float(np.var(e, ddof=1))


def _sim_round(R: dict, cfg: dict, M: int, seed: int):
    vs = list(R["cands"])
    probs = {k: {} for k in VARIANTS}
    eps = {k: 0.0 for k in VARIANTS}
    cand = {v: {k: [0, 0] for k in VARIANTS} for v in vs}       # drops, episodes
    if not vs:
        return {k: {None: 1.0} for k in VARIANTS}, {k: 0.0 for k in VARIANTS}, cand
    cap = cfg.get("max_harness_error_rate", 0.02)
    s_inc = R["S_inc"]
    floor_var = s_inc * (1 - s_inc)
    delta, L = R["delta"], R["S_star"] - R["delta"] - s_inc
    pairs = {v: A._pairs(R["inc"], R["cands"][v]["full"]) for v in vs}
    rng = random.Random(seed)
    for _ in range(M):
        orders = {}
        for v in vs:                # the same draws as posthoc_e1's seq_probs
            o = list(pairs[v])
            rng.shuffle(o)
            orders[v] = o
        for k, (mode, gamma) in VARIANTS.items():
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
                    if mode == "hold":
                        p = P.p_admissible(m, s2, n, N, delta, L, dC, nu, cfg)
                    elif mode == "cost":
                        p = p_cost(m, s2, n, N, delta, L, dC, _sc2(pk), nu, cfg)
                    else:
                        p = p_score(m, s2, n, N, L)
                    if p < gamma:
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
            {k: e / M for k, e in eps.items()}, cand)


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
            probs, eps, cand = _sim_round(R, cfg, M, 1000 * R["t"])
            rows.append({"t": R["t"], "probs": probs,
                         "cost": {k: {"episodes": e, "episode_equivalents": e} for k, e in eps.items()}})
            for v, C in R["cands"].items():
                dec = C["decision"]
                N = len(A._pairs(R["inc"], C["full"]))
                adm = bool(dec.get("admissible"))
                crow.append({"domain": name, "t": R["t"], "cand": v, "N": N, "admissible": adm,
                             "selected": R["accepted"] == v, "nu": dec.get("novelty") or 0,
                             "branch": _branch(dec.get("reason", ""), adm, dec.get("novelty") or 0),
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
        P._guard(runs, out, domains=domains, dest="e1_r10_seq.json", hook=hook)
    finally:
        A.main = orig


# ----------------------------------------------------------------- report --
def _merged(trajs: dict, names, dom=None, tmax=None):
    """The e1_r9_curves.json tables of the named trajectories with the seqsim
    rules added, restricted to a domain and to rounds t <= tmax."""
    tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
    if tables is None:
        return None, None
    sims = {}
    for n in names:
        f = Path(trajs[n]) / "verify" / "e1_r10_seq.json"
        if f.exists():
            for T in json.loads(f.read_text())["tables"]:
                sims[f"{n}/{T['name']}"] = {r["t"]: r for r in T["rows"]}
    out = []
    for T in tables:
        if dom and not T["name"].endswith("/" + dom):
            continue
        rows = []
        for r in T["rows"]:
            if tmax is not None and r["t"] > tmax:
                continue
            s = sims.get(T["name"], {}).get(r["t"])
            probs, cost = dict(r["probs"]), dict(r["cost"])
            if s:
                for k in VARIANTS:
                    lab = k if k != "seqfull" else "seqfull_sim"
                    probs[lab], cost[lab] = s["probs"][k], s["cost"][k]
            rows.append({**r, "probs": probs, "cost": cost})
        out.append({**T, "rows": rows})
    return out, block_of


RULES_D = ("full", "seqfull", "seqfull_sim", "seqcost", "seqcost[g10]", "seqcost[g20]", "seqscore",
           "sample@60", "none")


def _rule_summary(tables, block_of):
    s = summarise(tables, block_of, "full")
    res = {}
    for r in ("keep",) + RULES_D:
        if r not in s:
            continue
        x = s[r]
        row = {"dv": x["dv"], "diff_vs_full": x.get("diff_vs_full"), "p_ni": x.get("p_ni"),
               "episodes": x.get("episodes")}
        if r != "keep":
            st = _stats(tables, r)
            row.update(recall=st["recall"], accept_when_full_keeps=st["accept_when_full_keeps"],
                       full_accepts=st["full_accepts"])
        res[r] = row
    full_ep = res["full"]["episodes"]
    for r in res:
        if res[r].get("episodes") is not None and full_ep:
            res[r]["episode_share"] = res[r]["episodes"] / full_ep
    res["rounds"] = sum(len(T["rows"]) for T in tables)
    res["sums"] = {T["name"]: {r: float(sum(A.round_value(row, r) for row in T["rows"]))
                               for r in ("full", "seqfull") if r in T["rows"][0]["probs"]}
                   for T in tables if T["rows"]}
    return res


def _offline_draws(tables, block_of, B: int, seed: int = 11):
    """Bootstrap draws (incumbent blocks and held-out tasks, as summarise) of
    D(seqfull) - D(full) per round."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pr = P._per_round(tables, rng, block_of)
        out.append(np.mean([x[0] for x in pr["seqfull"]]) - np.mean([x[0] for x in pr["full"]]))
    return np.array(out)


def _il():
    from . import il
    arms = {"seq": dict(il.SEQ), "full": dict(il.FULL)}
    allr = {**arms["seq"], **arms["full"], **il.R1}
    ho = {n: il._transfer(Path(r)) for n, r in allr.items()}
    tasks = sorted(ho["r2"][1])
    delta = {n: json.loads((Path(r) / "runs" / "rrsi" / DOM / "calibration.json").read_text())["delta"]
             for n, r in allr.items()}
    return arms, allr, ho, tasks, delta, il.NEW


def _ols(y, X):
    from scipy import stats
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(r @ r / dof * np.linalg.inv(X.T @ X)))
    t = stats.t.ppf(0.95, dof)
    return [{"b": float(bb), "ci90": [float(bb - t * s), float(bb + t * s)],
             "p": float(2 * stats.t.sf(abs(bb / s), dof))} for bb, s in zip(b, se)], dof


def _delta_analysis(B: int = 2000, seed: int = 12) -> dict:
    arms, allr, ho, tasks, delta, new = _il()
    tf = lambda n, ts: float(np.mean([ho[n][0][t] - ho[n][1][t] for t in ts]))
    S, F = list(arms["seq"]), list(arms["full"])
    sets = {"primary": (S, F), "new_only": ([n for n in S if n in new], [n for n in F if n in new]),
            "with_r1": (S, F + ["r1"])}
    rng = np.random.default_rng(seed)
    res = {"delta": delta, "transfer": {n: tf(n, tasks) for n in allr}, "sets": {}}
    for name, (s, f) in sets.items():
        loops = s + f
        d = np.array([delta[n] for n in loops])
        a = np.array([n in s for n in loops], float)

        def fit(ts, idx=None):
            ii = range(len(loops)) if idx is None else idx
            y = np.array([100 * tf(loops[i], ts) for i in ii])
            dd = d[list(ii)]
            X = np.column_stack([np.ones(len(y)), a[list(ii)], dd - dd.mean()])
            return y, X
        y, X = fit(tasks)
        coef, dof = _ols(y, X)
        bs = []
        si, fi = [i for i, n in enumerate(loops) if n in s], [i for i, n in enumerate(loops) if n in f]
        for _ in range(B):
            ts = list(rng.choice(tasks, len(tasks)))
            idx = list(rng.choice(si, len(si))) + list(rng.choice(fi, len(fi)))
            yy, XX = fit(ts, idx)
            if np.linalg.matrix_rank(XX) < 3:
                continue
            bs.append(np.linalg.lstsq(XX, yy, rcond=None)[0][1])
        res["sets"][name] = {"seq": s, "full": f, "unadjusted": float(np.mean(y[a == 1]) - np.mean(y[a == 0])),
                             "adjusted": coef[1], "slope_per_0.01": {k: (v / 100 if k != "ci90" else [z / 100 for z in v])
                                                                    for k, v in coef[2].items() if k in ("b", "ci90")},
                             "dof": dof, "adjusted_boot_ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                             "mean_delta": {"seq": float(np.mean([delta[n] for n in s])),
                                            "full": float(np.mean([delta[n] for n in f]))},
                             "corr": {k: float(np.corrcoef([delta[n] for n in g], [tf(n, tasks) for n in g])[0, 1])
                                      for k, g in (("seq", s), ("full", f))}}
    return res


def _accounting() -> dict:
    arms, allr, *_ = _il()
    res = {"loops": {}}

    def full_rows(root):
        rows = []
        for t in range(T_LAST + 1):
            for x in json.loads((Path(root) / "runs/rrsi" / DOM / f"r{t}" / "decisions.json").read_text()):
                if x.get("delta_S") is None:          # stopped by the critic or a gate: not evaluated
                    continue
                rows.append({"t": t, "adm": bool(x["admissible"]), "dS": x["delta_S"]})
        return rows

    def seq_rows(root):
        sh = {r["t"]: r for r in json.loads((Path(root) / "runs/verify" / DOM / "shadow.json").read_text())}
        rows, drops = [], []
        for t in range(T_LAST + 1):
            dec = {x["variant"]: x for x in json.loads(
                (Path(root) / "runs/rrsi" / DOM / f"r{t}" / "decisions.json").read_text())}
            sel = json.loads((Path(root) / "runs/rrsi" / DOM / f"r{t}" / "selection.json").read_text())
            for v, s in sh[t]["candidates"].items():
                if s.get("completed"):
                    rows.append({"t": t, "adm": bool(s["admissible"]), "dS": s["dS"]})
                    rec = sel["candidates"][v]
                    drops.append({"early": rec["m"], "full": s["dS"], "n": rec["n"],
                                  "early_dC": rec["dC"], "full_dC": s.get("dC"),
                                  "agree_round": sh[t]["seq_choice"] == sh[t]["full_choice"]})
                else:
                    rows.append({"t": t, "adm": bool(dec[v]["admissible"]), "dS": dec[v]["delta_S"]})
        return rows, drops, sh

    allrows = {"seq": [], "full": []}
    alldrops = []
    for arm, M in arms.items():
        for n, root in M.items():
            if arm == "seq":
                rows, drops, sh = seq_rows(root)
                alldrops += drops
                fa = sum(1 for r in sh.values() if r["full_choice"] is not None)
                made = sum(1 for r in sh.values() if r["full_choice"] is not None
                           and r["seq_choice"] == r["full_choice"])
            else:
                rows = full_rows(root)
                fa = made = None
            allrows[arm] += rows
            res["loops"][n] = {"arm": arm, "evaluated": len(rows), "admissible": sum(r["adm"] for r in rows),
                               "rounds_full_accepts": fa, "seq_made": made}
    for arm, rows in allrows.items():
        dS = np.array([r["dS"] for r in rows])
        res[arm] = {"evaluated": len(rows), "admissible_share": float(np.mean([r["adm"] for r in rows])),
                    "mean_dS": float(dS.mean()), "share_dS_pos": float(np.mean(dS > 0))}
    e = np.array([d["early"] for d in alldrops])
    f = np.array([d["full"] for d in alldrops])
    nz = [(a, b) for a, b in zip(e, f) if a != 0 and b != 0]
    res["dropped"] = {"n": len(alldrops), "early_mean": float(e.mean()), "full_mean": float(f.mean()),
                      "corr": float(np.corrcoef(e, f)[0, 1]),
                      "sign_flip": float(np.mean([np.sign(a) != np.sign(b) for a, b in nz])),
                      "mean_episodes": float(np.mean([d["n"] for d in alldrops])),
                      "in_agreeing_rounds": sum(d["agree_round"] for d in alldrops)}
    return res


def _cand_summary(rows: list[dict]) -> dict:
    out = {}
    for k in VARIANTS:
        pd = np.array([r[k]["p_drop"] for r in rows])
        adm = np.array([r["admissible"] for r in rows], float)
        sel = np.array([r["selected"] for r in rows], float)
        N = np.array([r["N"] for r in rows], float)
        ep = np.array([r[k]["episodes"] for r in rows])
        x = {"stopped": float(pd.sum()), "stopped_admitted": float((pd * adm).sum() / pd.sum()) if pd.sum() else None,
             "admitted_stopped": float((pd * adm).sum() / adm.sum()) if adm.sum() else None,
             "selected_stopped": float((pd * sel).sum() / sel.sum()) if sel.sum() else None,
             "episode_share": float(ep.sum() / N.sum()), "by_branch": {}}
        for b in sorted({r["branch"] for r in rows}):
            ii = [i for i, r in enumerate(rows) if r["branch"] == b]
            x["by_branch"][b] = {"n": len(ii), "stopped_share": float(pd[ii].mean())}
        out[k] = x
    return out


SIM_SETS = (("primary", ["r2", "r3"], None), ("airline", ["r1", "r2", "r3", "r4"], "tau2_airline"),
            ("e3", ["e3"], None))


def report(trajs: dict, js: Path | None, B: int = 2000):
    res = {"matched": {}, "noise_band": _delta_analysis(B), "accounting": _accounting(), "costaware": {}}
    # (a) matched: airline rounds 0-9 of the trajectories driven by full evaluation
    for name, names in (("air_r1r3_t9", ["r1", "r2", "r3"]), ("air_r2r3_t9", ["r2", "r3"]),
                        ("air_r1r4_t9", ["r1", "r2", "r3", "r4"]), ("air_r1r3_all", ["r1", "r2", "r3"]),
                        ("primary_all", ["r2", "r3"])):
        dom = None if name.startswith("primary") else DOM
        tmax = None if name.endswith("_all") else T_LAST
        tables, block_of = _merged(trajs, names, dom, tmax)
        if tables is None:
            continue
        res["matched"][name] = _rule_summary(tables, block_of)
        if name == "air_r1r3_t9":
            off = 10 * _offline_draws(tables, block_of, B)
            res["matched"][name]["sum10_diff_ci90"] = [float(np.percentile(off, 5)), float(np.percentile(off, 95))]
    # live minus offline difference: loops resampled within arm with held-out tasks (as verify/il.py)
    arms, allr, ho, tasks, delta, new = _il()
    tf = lambda n, ts: float(np.mean([ho[n][0][t] - ho[n][1][t] for t in ts]))
    rng = np.random.default_rng(13)
    S, F = list(arms["seq"]), list(arms["full"])
    live = []
    for _ in range(B):
        ts = list(rng.choice(tasks, len(tasks)))
        live.append(100 * (np.mean([tf(n, ts) for n in rng.choice(S, len(S))]) -
                           np.mean([tf(n, ts) for n in rng.choice(F, len(F))])))
    live = np.array(live)
    tables, block_of = _merged(trajs, ["r1", "r2", "r3"], DOM, T_LAST)
    off = 100 * 10 * _offline_draws(tables, block_of, B, seed=14)
    gap = live - off
    pt_live = 100 * (np.mean([tf(n, tasks) for n in S]) - np.mean([tf(n, tasks) for n in F]))
    pt_off = 100 * 10 * (res["matched"]["air_r1r3_t9"]["seqfull"]["dv"][0] - res["matched"]["air_r1r3_t9"]["full"]["dv"][0])
    res["live_vs_offline"] = {"live": float(pt_live), "offline_sum10": float(pt_off),
                              "difference": float(pt_live - pt_off),
                              "ci90": [float(np.percentile(gap, 5)), float(np.percentile(gap, 95))],
                              "p_live_below_offline": float(np.mean(gap < 0))}
    # (d) cost-aware stopping on the recorded trajectories
    for name, names, dom in SIM_SETS:
        crow = []
        for n in names:
            f = Path(trajs[n]) / "verify" / "e1_r10_seq.json"
            if f.exists():
                crow += [r for r in json.loads(f.read_text())["cands"] if dom is None or r["domain"] == dom]
        if not crow:
            continue
        x = {"candidates": _cand_summary(crow)}
        tables, block_of = _merged(trajs, names, dom)
        if tables is not None:
            x["rounds"] = _rule_summary(tables, block_of)
        res["costaware"][name] = x
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    _print(res)
    return res


def _pp(x, d=2):
    return "  -  " if x is None else f"{100 * x:+.{d}f}"


def _print(res):
    for name, s in res["matched"].items():
        print(f"\n[matched {name}] rounds {s['rounds']}")
        for r in ("keep",) + RULES_D:
            if r not in s:
                continue
            x = s[r]
            d = x.get("diff_vs_full")
            print(f"  {r:14s} D {_pp(x['dv'][0])} [{_pp(x['dv'][1])},{_pp(x['dv'][2])}]"
                  + (f" diff {_pp(d[0])} [{_pp(d[1])},{_pp(d[2])}]" if d else "")
                  + (f" recall {x['recall']:.2f} ep {x['episode_share']:.2f}" if x.get("recall") is not None else ""))
        print("  sums", json.dumps(s["sums"]))
        if "sum10_diff_ci90" in s:
            print("  10-round diff ci90", [round(100 * v, 2) for v in s["sum10_diff_ci90"]])
    print("\nlive vs offline", json.dumps(res["live_vs_offline"]))
    nb = res["noise_band"]
    for k, s in nb["sets"].items():
        print(f"noise band {k}: unadjusted {s['unadjusted']:+.2f}, adjusted {s['adjusted']['b']:+.2f} "
              f"{[round(v, 2) for v in s['adjusted']['ci90']]} p {s['adjusted']['p']:.3f} boot "
              f"{[round(v, 2) for v in s['adjusted_boot_ci90']]}; mean delta {s['mean_delta']}; corr {s['corr']}")
    print("\naccounting", json.dumps({k: v for k, v in res["accounting"].items() if k != "loops"}))
    for name, x in res["costaware"].items():
        print(f"\n[cost-aware {name}]")
        for k, c in x["candidates"].items():
            rr = x.get("rounds", {}).get(k if k != "seqfull" else "seqfull_sim", {})
            d = rr.get("diff_vs_full")
            print(f"  {k:13s} stopped {c['stopped']:.1f} stopped-admitted {_pp(c['stopped_admitted'], 0)}% "
                  f"admitted-stopped {_pp(c['admitted_stopped'], 0)}% episodes {c['episode_share']:.2f}"
                  + (f" | D diff {_pp(d[0])} [{_pp(d[1])},{_pp(d[2])}] recall {rr['recall']:.2f}" if d else "")
                  + f" | branches {json.dumps({b: round(v['stopped_share'], 2) for b, v in c['by_branch'].items()})}")


# ----------------------------------------------------------------- tables --
def _minus(row: str) -> str:
    """Typeset the minus signs of a table row's numbers."""
    import re
    return re.sub(r"(?<![-\w])-(?=\d)", "$-$", row)


def tables(js: Path, out: Path):
    res = json.loads(js.read_text())
    out.mkdir(parents=True, exist_ok=True)
    m = res["matched"]["air_r1r3_t9"]
    lo = res["live_vs_offline"]
    # the loops of the registered comparison, historical and concurrent, with the recorded rounds
    nb = res["noise_band"]
    tr, dl = nb["transfer"], nb["delta"]
    acc = json.loads(Path("results/il/analysis.json").read_text())["accepted_changes"]
    groups = [("Live, all (registered)", nb["sets"]["primary"]["seq"], nb["sets"]["primary"]["full"], "primary"),
              (r"\quad historical", ["cl1", "cl2"], ["r2", "r3"], None),
              (r"\quad concurrent", nb["sets"]["new_only"]["seq"], nb["sets"]["new_only"]["full"], "new_only")]
    pt = lambda ns: 100 * float(np.mean([tr[n] for n in ns]))
    md = lambda ns: 100 * float(np.mean([dl[n] for n in ns]))
    rows = [f"Recorded rounds 0--9, offline & 3 & {10 * 100 * m['seqfull']['dv'][0]:+.1f} & "
            f"{10 * 100 * m['full']['dv'][0]:+.1f} & {lo['offline_sum10']:+.1f} [{100 * m['sum10_diff_ci90'][0]:+.1f}, "
            f"{100 * m['sum10_diff_ci90'][1]:+.1f}] & -- & -- & {md(['r1', 'r2', 'r3']):.1f} \\\\"]
    for lab, s_, f_, key in groups:
        diff = pt(s_) - pt(f_)
        if key:
            x = nb["sets"][key]
            ci_u = json.loads(Path("results/il/analysis.json").read_text())[key]["transfer"]["ci90"]
            d_cell = f"{diff:+.1f} [{100 * ci_u[0]:+.1f}, {100 * ci_u[1]:+.1f}]"
            a = x["adjusted"]
            a_cell = f"{a['b']:+.1f} [{a['ci90'][0]:+.1f}, {a['ci90'][1]:+.1f}]"
        else:
            d_cell, a_cell = f"{diff:+.1f}", "--"
        rows.append(f"{lab} & {len(s_)}+{len(f_)} & {pt(s_):+.1f} & {pt(f_):+.1f} & {d_cell} & {a_cell} & "
                    f"{sum(acc[n] for n in s_)}/{sum(acc[n] for n in f_)} & {md(s_):.1f}/{md(f_):.1f} \\\\")
    rows = [_minus(r) for r in rows]
    lines = [r"\begin{tabular}{@{}lcrrrrcc@{}}", r"\toprule",
             r" & Loops & Sequential & Full & Difference [90\%] & $\delta$-adjusted [90\%] & Accepted & Mean $\delta$ \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (out / "r10_loops.tex").write_text("\n".join(lines) + "\n")
    print((out / "r10_loops.tex").read_text())
    rows = []
    names = {"seqfull": "Registered (cost held)", "seqcost": r"Cost-aware, $\gamma=0.05$",
             "seqcost[g10]": r"Cost-aware, $\gamma=0.10$", "seqcost[g20]": r"Cost-aware, $\gamma=0.20$",
             "seqscore": "Never stop for cost"}
    for k, lab in names.items():
        cells = []
        for s in ("primary", "airline", "e3"):
            x = res["costaware"].get(s)
            if not x:
                cells += ["--"] * 3
                continue
            c = x["candidates"][k]
            cells += [f"{100 * c['stopped_admitted']:.0f}\\%" if c["stopped_admitted"] is not None else "--",
                      f"{100 * (1 - c['admitted_stopped']):.0f}\\%", f"{100 * c['episode_share']:.0f}\\%"]
        rows.append(lab + " & " + " & ".join(cells) + r" \\")
    lines = [r"\begin{tabular}{@{}l" + "rrr" * 3 + "@{}}", r"\toprule",
             r" & \multicolumn{3}{c}{$\tau^2$ r2+r3} & \multicolumn{3}{c}{$\tau^2$ airline r1--r4} & \multicolumn{3}{c}{AppWorld} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
             r"Stopping rule & Stop adm. & Kept & Ep. & Stop adm. & Kept & Ep. & Stop adm. & Kept & Ep. \\",
             r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (out / "r10_costaware.tex").write_text("\n".join(lines) + "\n")
    print((out / "r10_costaware.tex").read_text())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seqsim")
    s.add_argument("--runs", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--domains", default="tau2_retail,tau2_airline")
    r = sub.add_parser("report")
    r.add_argument("--traj", nargs="+", required=True)
    r.add_argument("--json", type=Path)
    t = sub.add_parser("tables")
    t.add_argument("--json", type=Path, required=True)
    t.add_argument("--tex", type=Path, default=Path("paper/tables"))
    a = ap.parse_args()
    if a.cmd == "seqsim":
        seqsim(a.runs, a.out, a.domains)
    elif a.cmd == "report":
        report(dict(x.split("=", 1) for x in a.traj), a.json)
    else:
        tables(a.json, a.tex)


if __name__ == "__main__":
    main()
