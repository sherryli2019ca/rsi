"""Fourth-review reanalysis of the tau2 banks (no new agent runs).

  split     policy evidence separated from outcome assessment: on each of five
            random halvings of the bank, policies ran on half A
            (reanalysis_decisions.json, split{t}); their deployable sets are scored
            here on half B only (edges and repair rates from a bootstrap of B's
            replays, harm from B's regression runs against the task-matched
            unpatched baseline, hierarchical model as in reanalysis3)
  prior     Proposition 1's Bayes default with the edge prior at the audited
            analyst accuracy (0.41 retail, 0.29 airline) instead of the 0.7 that
            every policy used
  regev     how many of each policy's deployed patches have 20 regression runs
            (the bank was extended only for patches the original policies accepted)
  payback   held-out gain of each verified set over Apply attributed per future
            episode, drawn jointly with a per-change cost, against the episodes
            the verification spent: probability of paying back within N episodes

  python -m agent_exp.reanalysis4 --out runs/tau2_retail --domain tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from agent_exp import reanalysis as RA
from agent_exp.reanalysis import CTX, base_draws, boot_bank, split_bank, truth, yardstick
from agent_exp.reanalysis3 import SHOW, deploy, fit_rho, groups_of, value

BUDGETS = (10, 40, 80)
HORIZONS = (100, 300, 1000, 3000, 10000)


def matched_r0(bank, p0):
    """Task-matched baseline failure rate of every patch for the regression runs
    of `bank` (a bank half)."""
    tasks = CTX["base_tasks"]
    col = {t: n for n, t in enumerate(tasks)}
    M = np.zeros((CTX["I"], CTX["J"], 3, len(tasks)))
    for key, runs in bank["reg"].items():
        i, j, k = map(int, key.split(","))
        for r in runs:
            M[i, j, k, col[r["task"]]] += 1
    n = M.sum(-1)
    return np.where(n > 0, (M @ p0) / np.maximum(n, 1), p0.mean())


def split_eval(dec, w, o, n_draw=100, n_seed=40):
    f, I, J = CTX["f"], CTX["I"], CTX["J"]
    rng = np.random.default_rng(3)
    vals = {c: {} for c in (0.0, 0.02)}
    n_edges = []
    for t in range(RA.N_SPLITS):
        _, Bh = split_bank(CTX["bank"], t)
        TB = truth(Bh, I, J, CTX["ev"]["r0"], w)
        n_edges.append(int(TB["E"].sum()))
        groups = groups_of(dec, f"split{t}", BUDGETS, n_seed, f, True)
        for _ in range(n_draw):
            p0 = base_draws(1, rng)[0]
            rho, _, _ = fit_rho(TB["nf"], TB["n"], matched_r0(Bh, p0), rng, "hier")
            Tb = truth(boot_bank(Bh, rng), I, J, CTX["ev"]["r0"], w)
            R, cap = w * rho, w * float(p0.mean())
            for c in vals:
                for g, sets in groups.items():
                    vals[c].setdefault(g, []).append(
                        np.mean([value(cs, Tb, f, R, c, "wrong", cap) for cs in sets]) / o)
    out = {"n_edges_B": n_edges}
    for c, vs in vals.items():
        ref = np.array(vs["LLM-only|10"])
        res = {}
        for g, v in vs.items():
            v = np.array(v)
            res[g] = {"mean": float(v.mean()), "lo": float(np.quantile(v, 0.05)),
                      "hi": float(np.quantile(v, 0.95)), "p_better": float((v - ref > 0).mean())}
        for B in BUDGETS:
            gs = [g for g in vs if g.endswith(f"|{B}")]
            M = np.array([vs[g] for g in gs])
            best = np.bincount(M.argmax(0), minlength=len(gs)) / M.shape[1]
            res[f"p_best|{B}"] = {g.split("|")[0]: float(b) for g, b in zip(gs, best)}
        out[f"{c}"] = res
    return out


def prior_eval(T, w, o, r_hats, n_draw=300):
    """Value of the Bayes default under edge priors with different analyst
    accuracies, primary harm model."""
    from sim.testbed import edge_prior
    f, inb, counts = CTX["f"], CTX["inb"], CTX["counts"]
    rng = np.random.default_rng(1)
    sets = {}
    for r in r_hats:
        pi = edge_prior(counts, r)
        sets[f"{r:.2f}"] = deploy([(int(i), int(j), 0) for i, j in zip(*np.nonzero(inb))
                                   if pi[i, j] * f[i] * 0.5 > 0], f)
        sets[f"{r:.2f}|cw"] = deploy([(int(i), int(j), 0) for i, j in zip(*np.nonzero(inb))
                                      if pi[i, j] * f[i] * 0.5 - 0.02 * (1 - pi[i, j]) > 0], f)
    vals = {k: [] for k in sets}
    for _ in range(n_draw):
        p0 = base_draws(1, rng)[0]
        rho, _, _ = fit_rho(T["nf"], T["n"], RA.base_r0(p0), rng, "hier")
        Tb = truth(boot_bank(CTX["bank"], rng), CTX["I"], CTX["J"], CTX["ev"]["r0"], w)
        for k, s in sets.items():
            c = 0.02 if k.endswith("cw") else 0.0
            vals[k].append(value(s, Tb, f, w * rho, c, "wrong", w * float(p0.mean())) / o)
    return {k: {"n": len(sets[k]), "mean": float(np.mean(v)), "lo": float(np.quantile(v, .05)),
                "hi": float(np.quantile(v, .95))} for k, v in vals.items()}


def regev(dec, f, B=40):
    n = {key: len(v) for key, v in CTX["bank"]["reg"].items()}
    out = {}
    for m in SHOW:
        sets = groups_of(dec, "main", (B,), 40, f, True).get(f"{m}|{B}", [])
        cells = [c for s in sets for c in s]
        if cells:
            out[m] = float(np.mean([n.get(f"{i},{j},{k}", 0) >= 20 for i, j, k in cells]))
    return out


def payback(domain, dec, F_abs, rng, n_draw=20000):
    """Held-out gain per future episode of each verified set over Apply attributed,
    g ~ N(paired mean, task-clustered se^2), plus the change cost saved by
    applying fewer patches, c_a ~ U(0, 0.02) of the failure mass per change per
    episode; verification spends its measured budget once. Returns P(pays back
    within N) for each horizon N and P(g + saving > 0)."""
    H = json.load(open(f"runs/{domain}/heldout_breakdown.json"))
    nA = H["LLM-only|10"]["n_patches"]
    out = {}
    for key, r in H.items():
        if not isinstance(r, dict) or "vs_apply" not in r or key.startswith("LLM-only"):
            continue
        m = key.split("|")[0]
        spent = [v[1] for k, v in dec.items() if k.startswith(f"main|{m}|10|")]
        C = float(np.mean(spent)) if spent else 10.0
        mean, se = r["vs_apply"]["all"][:2]
        g = mean + se * rng.standard_normal(n_draw)
        ca = rng.uniform(0, 0.02, n_draw)
        g_c = g + ca * F_abs * (nA - r["n_patches"])
        res = {"C": C, "mean": mean, "se": se, "dn": nA - r["n_patches"]}
        for lab, gg in (("c0", g), ("ca", g_c)):
            res[lab] = {"p_pos": float((gg > 0).mean()),
                        **{f"N{N}": float((N * gg > C).mean()) for N in HORIZONS}}
        out[m] = res
    return out


def judge_needed(b, q, cost):
    """Sensitivity a judge with false-positive rate eps needs for single-step
    checks to beat replays at a given cost ratio (Prop. cfid)."""
    from agent_exp.judge_configs import break_even
    out = {}
    for eps in (0.05, 0.1, 0.2):
        ss = np.linspace(eps, 1, 400)
        ok = [s for s in ss if break_even(s, eps, b, q) > cost]
        out[f"{eps}"] = float(ok[0]) if ok else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--draws", type=int, default=100)
    args = ap.parse_args()
    CTX.update(RA.load(args.out, args.domain))
    dec = RA._load(os.path.join(args.out, "reanalysis_decisions.json"))
    ev = CTX["ev"]
    w = ev["w"]
    T = truth(CTX["bank"], CTX["I"], CTX["J"], ev["r0"], w)
    CTX["T"] = T
    o = yardstick(T, CTX["f"])
    rep = {"split": split_eval(dec, w, o, n_draw=args.draws)}
    rep["prior"] = prior_eval(T, w, o, (0.7, ev["attr_acc_injected"]))
    rep["regev"] = regev(dec, CTX["f"])
    H = json.load(open(os.path.join(args.out, "heldout_breakdown.json")))
    F_abs = 1 - H["unpatched"]["all"][0]
    rep["payback"] = payback(os.path.basename(args.out.rstrip("/")), dec, F_abs,
                             np.random.default_rng(9))
    J = RA._load(os.path.join(args.out, "judge_configs_report.json"))
    if J:
        rep["judge_needed"] = {"regen_cost": J["regen_cost"],
                               "at_regen": judge_needed(J["b"], J["q"], J["regen_cost"]),
                               "at_short": judge_needed(J["b"], J["q"],
                                                        J["configs"]["short_pro"]["cost"])}
    RA._save(os.path.join(args.out, "reanalysis4.json"), rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "split"}, indent=1))
    print(json.dumps({g: round(v["mean"], 2) for g, v in rep["split"]["0.0"].items()
                      if not g.startswith("p_best")}))


if __name__ == "__main__":
    main()
