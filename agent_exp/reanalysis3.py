"""Third-review reanalysis of the tau2 banks (no new agent runs).

Changes to the primary scoring, all requested in review:
  deployable  every accepted set is reduced to one patch per component before it
              is scored (the cell of the most frequent category is kept, as when
              patch sets are deployed on held-out tasks)
  support     an applied set cannot lower the failure rate on previously solved
              tasks below zero: each patch's rho_k is clipped at -r0_k and the
              set's summed rho at minus the mean baseline failure rate
  defaults    value of information is reported against two defaults: the
              Apply-attributed heuristic and the Bayes default of Proposition 1
              (no evidence, the priors of the net-effect method), with the gap
              between them

Harm models (sensitivity): hier (normal, empirical Bayes), pool (tau = 0),
tau2x (tau doubled), nopool (each patch its own Beta posterior), task
(tasks resampled before fitting the hierarchical model). A posterior predictive
check compares the spread of patch failure rates with draws from the fitted model.

  python -m agent_exp.reanalysis3 --out runs/tau2_retail --domain tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from agent_exp import reanalysis as RA
from agent_exp.reanalysis import (CTX, base_draws, base_r0, boot_bank, repair, truth,
                                  yardstick)

MODELS = ("hier", "pool", "tau2x", "nopool", "task")
SHOW = ["LLM-only", "Replay-each", "HarnessFix", "HarnessFix-gate", "Uncertainty",
        "Uncertainty-MF", "CARVE-full-only", "CARVE", "CARVE-calibrated",
        "CARVE-robust-check", "Net"]


# ---- deployable sets and capped harm -----------------------------------------------
def deploy(cells, f):
    """One patch per component: keep the cell of the most frequent category."""
    best = {}
    for i, j, k in cells:
        if j not in best or f[i] > f[best[j][0]]:
            best[j] = (i, j, k)
    return sorted(best.values())


def value(cells, T, f, R, c, ctype, cap):
    rep = repair(cells, T, f)
    reg = sum(R[i, j, k] for i, j, k in cells)
    if cap is not None:
        reg = max(reg, -cap)
    nw = sum(not T["E"][i, j] for i, j, k in cells)
    return rep - reg - c * (len(cells) if ctype == "apply" else nw)


def bayes_default(c, ctype):
    """Proposition 1's default without evidence: in every bank cell, the patch with
    the largest prior expected net effect, applied when positive. Priors are those
    of the net-effect method (edge prior from attribution counts, q ~ Beta(2,2),
    rho with prior mean 0); all patches of a cell are exchangeable, so patch 0."""
    from sim.testbed import edge_prior
    f, inb, counts = CTX["f"], CTX["inb"], CTX["counts"]
    pi = edge_prior(counts, 0.7)
    cells = []
    for i, j in zip(*np.nonzero(inb)):
        pen = c if ctype == "apply" else c * (1 - pi[i, j])
        if pi[i, j] * f[i] * 0.5 - pen > 0:
            cells.append((int(i), int(j), 0))
    return cells


def perfect(T, f, R, c, ctype, inb, cap, within=None):
    """Best deployable set with every cell's effect known: per component the cell
    and patch with the largest positive net effect (a greedy solution, so a lower
    bound on the constrained optimum). `within` restricts to given components."""
    best = {}
    I, J = inb.shape
    for i in range(I):
        for j in range(J):
            if not inb[i, j] or (within is not None and j not in within):
                continue
            pen = c if (ctype == "apply" or not T["E"][i, j]) else 0.0
            d = f[i] * T["Q"][i, j] * T["E"][i, j] - R[i, j] - pen
            k = int(np.argmax(d))
            if d[k] > 0 and (j not in best or d[k] > best[j][0]):
                best[j] = (d[k], (i, j, k))
    return sorted(v[1] for v in best.values())


def vpi3(default, T, f, R, c, ctype, inb, cap):
    """Value of perfect information over a deployable default, split sequentially:
    harm (drop default patches with negative net effect), choice (best cell and
    patch on the default's components), discovery (the rest)."""
    v0 = value(default, T, f, R, c, ctype, cap)
    keep = []
    for i, j, k in default:
        pen = c if (ctype == "apply" or not T["E"][i, j]) else 0.0
        if f[i] * T["Q"][i, j, k] * T["E"][i, j] - R[i, j, k] - pen >= 0:
            keep.append((i, j, k))
    v1 = value(keep, T, f, R, c, ctype, cap)
    comps = {j for _, j, _ in default}
    ch = perfect(T, f, R, c, ctype, inb, cap, within=comps)
    v2 = max(v1, value(ch, T, f, R, c, ctype, cap))
    pi = perfect(T, f, R, c, ctype, inb, cap)
    v3 = max(v2, value(pi, T, f, R, c, ctype, cap))
    return {"default": v0, "harm": v1 - v0, "select": v2 - v1, "discover": v3 - v2,
            "total": v3 - v0}


# ---- harm models ---------------------------------------------------------------
def fit_rho(nf, n, r0, rng, model, scale_tau=1.0):
    """One posterior draw of rho (absolute excess failure over the task-matched
    baseline), clipped to the support rho >= -r0."""
    if model == "nopool":
        p = rng.beta(nf + 0.5, n - nf + 0.5)
        rho = np.where(n > 0, p - r0, 0.0)
        return np.maximum(rho, -r0), float(np.mean(rho[n > 0])), np.nan
    has = n > 0
    y = np.where(has, nf / np.maximum(n, 1), 0.0) - r0
    pbar = nf[has].sum() / n[has].sum()
    v = np.where(has, pbar * (1 - pbar) / np.maximum(n, 1), np.inf)
    yy, vv = y[has], v[has]
    tau2 = 0.0 if model == "pool" else max(0.0, np.var(yy) - vv.mean()) * scale_tau ** 2
    wts = 1 / (vv + tau2)
    mu = (wts * yy).sum() / wts.sum() + rng.standard_normal() * np.sqrt(1 / wts.sum())
    if tau2 == 0:
        rho = np.full(n.shape, mu)
    else:
        prec = 1 / v + 1 / tau2
        m = (np.where(has, y / v, 0.0) + mu / tau2) / prec
        rho = m + rng.standard_normal(n.shape) / np.sqrt(prec)
    return np.maximum(rho, -r0), float(mu), float(np.sqrt(tau2))


def task_tables():
    """Per patch and regression task: runs and failures; per task: baseline runs."""
    bank, tasks = CTX["bank"], CTX["base_tasks"]
    col = {t: n for n, t in enumerate(tasks)}
    I, J = CTX["I"], CTX["J"]
    N = np.zeros((I, J, 3, len(tasks)))
    NF = np.zeros_like(N)
    for key, runs in bank["reg"].items():
        i, j, k = map(int, key.split(","))
        for r in runs:
            N[i, j, k, col[r["task"]]] += 1
            NF[i, j, k, col[r["task"]]] += 1 - r["y"]
    return N, NF


def draw_harm(model, rng, T, N, NF):
    """(rho, set-level cap) for one posterior draw under a harm model."""
    nf0, n0 = CTX["base_nf"], CTX["base_n"]
    if model == "task":
        wt = np.bincount(rng.integers(len(n0), size=len(n0)), minlength=len(n0)).astype(float)
        n = (N * wt).sum(-1)
        nf = (NF * wt).sum(-1)
        p0 = rng.binomial(n0.astype(int), nf0 / n0) / n0
        r0 = np.where(n > 0, ((N * wt) @ p0) / np.maximum(n, 1), (wt * p0).sum() / wt.sum())
        cap = float((wt * p0).sum() / wt.sum())
        rho, mu, tau = fit_rho(nf, n, r0, rng, "hier")
        return rho, cap, mu, tau
    p0 = base_draws(1, rng)[0]
    r0 = base_r0(p0)
    rho, mu, tau = fit_rho(T["nf"], T["n"], r0, rng, "hier" if model == "tau2x" else model,
                           scale_tau=2.0 if model == "tau2x" else 1.0)
    return rho, float(p0.mean()), mu, tau


def ppc(T, rng, n_sim=2000):
    """Posterior predictive check of the hierarchical model on the patches with
    20 regression runs: spread of failure rates, number with no failure, maximum."""
    sel = T["n"] >= 20
    obs = T["nf"][sel] / T["n"][sel]
    stats_obs = np.array([obs.std(), (T["nf"][sel] == 0).sum(), obs.max()])
    sims = []
    for _ in range(n_sim):
        p0 = base_draws(1, rng)[0]
        r0 = base_r0(p0)
        rho, mu, tau = fit_rho(T["nf"], T["n"], r0, rng, "hier")
        row = []
        # posterior predictive (patch effects as fitted) and mixed predictive
        # (new patch effects drawn from the fitted population N(mu, tau^2))
        for r in (rho, mu + tau * rng.standard_normal(rho.shape)):
            p = np.clip(r0 + r, 0, 1)[sel]
            y = rng.binomial(T["n"][sel].astype(int), p) / T["n"][sel]
            row += [y.std(), (y == 0).sum(), y.max()]
        sims.append(row)
    sims = np.array(sims)
    return {"obs": stats_obs.tolist(), "sim_mean": sims.mean(0).tolist(),
            "p_upper": (sims >= np.tile(stats_obs, 2)).mean(0).tolist(),
            "stats": ["sd", "n_zero", "max"] * 2, "kinds": ["posterior"] * 3 + ["mixed"] * 3}


# ---- evaluation ------------------------------------------------------------------
def groups_of(dec, name, budgets, n_seed, f, dep):
    g = {}
    for key, v in dec.items():
        nm, m, B, s = key.split("|")
        if nm == name and int(B) in budgets and int(s) < n_seed:
            g.setdefault(f"{m}|{B}", []).append(deploy(v[0], f) if dep else
                                                 [tuple(x) for x in v[0]])
    return g


def evaluate(dec, T, w, model="hier", dep=True, cap=True, n_draw=300, n_seed=40,
             budgets=(10, 40, 80), with_vpi=True, seed=1):
    rng = np.random.default_rng(seed)
    f, inb, bank = CTX["f"], CTX["inb"], CTX["bank"]
    I, J = inb.shape
    o = yardstick(T, f)
    N, NF = task_tables()
    groups = groups_of(dec, "main", budgets, n_seed, f, dep)
    defaults = {}
    for c in (0.0, 0.02):
        apply_ = groups_of(dec, "main", (10,), 1, f, dep)["LLM-only|10"][0]
        bd = bayes_default(c, "wrong")
        defaults[c] = {"apply": apply_, "bayes": deploy(bd, f) if dep else bd}
        groups[f"Bayes-default-{c}|0"] = [defaults[c]["bayes"]]
    vals = {c: {g: [] for g in groups} for c in (0.0, 0.02)}
    mus, taus, vp = [], [], {}
    for d in range(n_draw):
        rho, base, mu, tau = draw_harm(model, rng, T, N, NF)
        mus.append(mu)
        taus.append(tau)
        Tb = truth(boot_bank(bank, rng), I, J, CTX["ev"]["r0"], w)
        R = w * rho
        capv = w * base if cap else None
        for c in vals:
            for g, sets in groups.items():
                vals[c][g].append(np.mean([value(cs, Tb, f, R, c, "wrong", capv)
                                           for cs in sets]) / o)
            if with_vpi:
                for dn, dc in defaults[c].items():
                    for k, v in vpi3(dc, Tb, f, R, c, "wrong", inb, capv).items():
                        vp.setdefault(f"{c}|{dn}", {}).setdefault(k, []).append(v / o)
    q = lambda v: [float(np.mean(v)), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
    out = {"mu": q(mus), "tau": float(np.nanmean(taus)) if not np.all(np.isnan(taus)) else None,
           "vpi": {k: {kk: q(vv) for kk, vv in v.items()} for k, v in vp.items()}}
    for c, vs in vals.items():
        res = {}
        for g, v in vs.items():
            v = np.array(v)
            B = g.split("|")[1]
            ref = np.array(vs["LLM-only|10"])
            dlt = v - ref
            res[g] = {"mean": float(v.mean()), "lo": float(np.quantile(v, 0.05)),
                      "hi": float(np.quantile(v, 0.95)), "p_better": float((dlt > 0).mean())}
        for B in budgets:
            gs = [g for g in vs if g.endswith(f"|{B}")]
            M = np.array([vs[g] for g in gs])
            best = np.bincount(M.argmax(0), minlength=len(gs)) / M.shape[1]
            res[f"p_best|{B}"] = {g.split("|")[0]: float(b) for g, b in zip(gs, best)}
        out[f"{c}"] = res
    out["defaults"] = {f"{c}": {k: [list(map(int, x)) for x in v] for k, v in d.items()}
                       for c, d in defaults.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--draws", type=int, default=300)
    args = ap.parse_args()
    CTX.update(RA.load(args.out, args.domain))
    dec = RA._load(os.path.join(args.out, "reanalysis_decisions.json"))
    ev = CTX["ev"]
    w = ev["w"]
    T = truth(CTX["bank"], CTX["I"], CTX["J"], ev["r0"], w)
    CTX["T"] = T
    F = 1 / (1 + w)
    rep = {"F": F, "yardstick": yardstick(T, CTX["f"]),
           "yardstick_abs": yardstick(T, CTX["f"]) * F}
    rep["primary"] = evaluate(dec, T, w, n_draw=args.draws)
    rep["undeployed"] = evaluate(dec, T, w, dep=False, n_draw=args.draws, with_vpi=True)
    rep["uncapped"] = evaluate(dec, T, w, cap=False, n_draw=args.draws)
    for m in MODELS[1:]:
        rep[f"model|{m}"] = evaluate(dec, T, w, model=m, n_draw=args.draws)
    rep["ppc"] = ppc(T, np.random.default_rng(5))
    netabl = {k: v for k, v in dec.items() if k.startswith("netabl|")}
    if netabl:
        rep["netabl"] = netabl_eval(dec, netabl, T, w, args.draws)
    RA._save(os.path.join(args.out, "reanalysis3.json"), rep)
    p = rep["primary"]
    print(json.dumps({"mu": p["mu"], "tau": p["tau"], "ppc": rep["ppc"],
                      "yard_abs": rep["yardstick_abs"]}, indent=1))


def netabl_eval(dec, netabl, T, w, n_draw, budgets=(10, 40, 80)):
    """Net-effect method with its regression term clipped / rescaled or not,
    scored like the primary table."""
    merged = {k: v for k, v in dec.items() if k.startswith("main|Net|") or
              k.startswith("main|LLM-only|")}
    merged.update({k.replace("netabl|", "main|", 1): v for k, v in netabl.items()})
    return evaluate(merged, T, w, n_draw=n_draw, budgets=budgets, with_vpi=False, n_seed=40)


if __name__ == "__main__":
    main()
