"""All synthetic experiments reported in the paper.

Usage:  python -m sim.experiments [main|prior|fidelity|misspec|scale|rsi|all]
Results are written to results/<name>.json.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from sim.rsi_loop import rsi_run
from sim.run import episode
from sim.testbed import WorldConfig

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
CPS = [0, 10, 20, 30, 50, 75, 100, 150, 200]
METHODS = ["LLM-only", "Replay-each", "Replay-each-3patch", "Sequential-each", "Uncertainty",
           "Thompson", "Uncertainty-MF", "CARVE-full-only",
           "CARVE-myopic", "CARVE-uniform-prior", "CARVE-one-patch",
           "CARVE-no-label-noise", "CARVE"]


def _ep(args):
    cfg, method, seed, cps, costs, overrides = args
    overrides = dict(overrides or {})
    c_fp = overrides.pop("c_fp", 0.02)
    res = episode(cfg, method, seed, cps, costs=costs, c_fp=c_fp,
                  model_overrides=overrides or None)
    return {str(k): v for k, v in res.items()}


def run_grid(jobs, procs=4):
    with Pool(procs) as p:
        return p.map(_ep, jobs, chunksize=1)


def summarise(results, cps):
    g = np.array([[r[str(c)][0] for c in cps] for r in results])
    prec = np.array([[r[str(c)][1] for c in cps] for r in results])
    rec = np.array([[r[str(c)][2] for c in cps] for r in results])
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    n = len(results)
    return {
        "gain_mean": g.mean(0).tolist(), "gain_se": (g.std(0, ddof=1) / np.sqrt(n)).tolist(),
        "f1_mean": f1.mean(0).tolist(), "f1_se": (f1.std(0, ddof=1) / np.sqrt(n)).tolist(),
        "prec_mean": prec.mean(0).tolist(), "rec_mean": rec.mean(0).tolist(),
        "spent_mean": float(np.mean([r[str(cps[-1])][3] for r in results])),
        "n": n,
    }


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as fh:
        json.dump(obj, fh, indent=1)


def exp_main(seeds=50):
    cfg = WorldConfig()
    out = {"checkpoints": CPS, "methods": {}}
    for m in METHODS:
        res = run_grid([(cfg, m, s, CPS, (1.0, 0.1), None) for s in range(seeds)])
        out["methods"][m] = summarise(res, CPS)
        print("main", m, np.round(out["methods"][m]["gain_mean"], 3), flush=True)
    save("main", out)


def exp_prior(seeds=30, budget=50):
    """Vary LLM attribution accuracy r."""
    out = {"budget": budget, "r": [], "methods": {}}
    rs = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
    ms = ["LLM-only", "Replay-each", "CARVE-uniform-prior", "CARVE"]
    out["r"] = rs
    for m in ms:
        out["methods"][m] = []
        for r in rs:
            cfg = replace(WorldConfig(), r=r)
            res = run_grid([(cfg, m, s, [0, budget], (1.0, 0.1), None) for s in range(seeds)])
            out["methods"][m].append(summarise(res, [0, budget]))
        print("prior", m, [round(x["gain_mean"][1], 3) for x in out["methods"][m]], flush=True)
    save("prior", out)


def exp_fidelity(seeds=30, budget=50):
    """Vary single-step cost and quality."""
    out = {"budget": budget, "cost": {}, "quality": {}}
    costs = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
    for m in ["CARVE-full-only", "CARVE"]:
        out["cost"][m] = []
        for c in costs:
            res = run_grid([(WorldConfig(), m, s, [budget], (1.0, c), None) for s in range(seeds)])
            out["cost"][m].append(summarise(res, [budget]))
        print("fid-cost", m, [round(x["gain_mean"][0], 3) for x in out["cost"][m]], flush=True)
    out["cost_values"] = costs
    quals = [(0.95, 0.05), (0.85, 0.15), (0.75, 0.25), (0.65, 0.35), (0.55, 0.45)]
    for m in ["CARVE-full-only", "CARVE"]:
        out["quality"][m] = []
        for sens, fpr in quals:
            cfg = replace(WorldConfig(), sens=sens, fpr=fpr)
            res = run_grid([(cfg, m, s, [budget], (1.0, 0.1), None) for s in range(seeds)])
            out["quality"][m].append(summarise(res, [budget]))
        print("fid-qual", m, [round(x["gain_mean"][0], 3) for x in out["quality"][m]], flush=True)
    out["quality_values"] = quals
    save("fidelity", out)


def exp_misspec(seeds=30, budget=50):
    """CARVE with wrong nuisance parameters (true world uses defaults)."""
    cases = {
        "correct": None,
        "b x0.2": {"b": 0.01}, "b x3": {"b": 0.15},
        "lam 1.0": {"lam": 1.0}, "lam 0.6": {"lam": 0.6},
        "sens/fpr .95/.05": {"sens": 0.95, "fpr": 0.05},
        "sens/fpr .7/.3": {"sens": 0.7, "fpr": 0.3},
        "q prior B(1,1)": {"q_a": 1.0, "q_b": 1.0},
        "q prior B(5,2)": {"q_a": 5.0, "q_b": 2.0},
    }
    out = {"budget": budget, "cases": {}}
    for name, ov in cases.items():
        res = run_grid([(WorldConfig(), "CARVE", s, [budget], (1.0, 0.1), ov) for s in range(seeds)])
        out["cases"][name] = summarise(res, [budget])
        print("misspec", name, round(out["cases"][name]["gain_mean"][0], 3), flush=True)
    save("misspec", out)


def exp_scale(seeds=20):
    """Larger graph: 20 categories x 50 components."""
    cfg = replace(WorldConfig(), I=20, J=50)
    cps = [0, 25, 50, 100, 200, 300]
    out = {"checkpoints": cps, "methods": {}}
    for m in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF", "CARVE-full-only",
              "CARVE"]:
        res = run_grid([(cfg, m, s, cps, (1.0, 0.1), None) for s in range(seeds)])
        out["methods"][m] = summarise(res, cps)
        print("scale", m, np.round(out["methods"][m]["gain_mean"], 3), flush=True)
    save("scale", out)


def exp_joint(seeds=30):
    """Stress test: 30% of categories need two components changed together,
    which violates CARVE's cell-independence assumption."""
    cfg = replace(WorldConfig(), joint_rate=0.3)
    cps = [0, 20, 50, 100, 200]
    out = {"checkpoints": cps, "methods": {}}
    for m in ["LLM-only", "Replay-each", "Uncertainty", "Thompson", "Uncertainty-MF", "CARVE"]:
        res = run_grid([(cfg, m, s, cps, (1.0, 0.1), None) for s in range(seeds)])
        out["methods"][m] = summarise(res, cps)
        print("joint", m, np.round(out["methods"][m]["gain_mean"], 3), flush=True)
    save("joint", out)


def exp_stress(seeds=30, budget=50):
    """Violations of the model's assumptions, CARVE vs the strongest baseline."""
    base = WorldConfig()
    cases = {
        "default": base,
        "labels lam=0.5": replace(base, lam=0.5),
        "judge fooled by bad patches": replace(base, fooled_fpr=0.6),
        "analyst overconfident (r_hat+0.3)": replace(base, extra={"r_bias": 0.3}),
        "analyst underconfident (r_hat-0.3)": replace(base, extra={"r_bias": -0.3}),
    }
    out = {"budget": budget, "cases": {}}
    for name, cfg in cases.items():
        out["cases"][name] = {}
        ms = ["Uncertainty-MF", "CARVE"] + (["CARVE-no-label-noise"] if "lam" in name else [])
        for m in ms:
            res = run_grid([(cfg, m, s, [budget], (1.0, 0.1), None) for s in range(seeds)])
            out["cases"][name][m] = summarise(res, [budget])
        print("stress", name, {m: round(v["gain_mean"][0], 3) for m, v in out["cases"][name].items()},
              flush=True)
    # regression cost sweep (both decision rule and evaluation use c_fp)
    out["c_fp"] = {}
    for c in [0.005, 0.02, 0.05, 0.1]:
        out["c_fp"][str(c)] = {}
        for m in ["LLM-only", "Uncertainty-MF", "CARVE"]:
            res = run_grid([(base, m, s, [budget], (1.0, 0.1), {"c_fp": c}) for s in range(seeds)])
            out["c_fp"][str(c)][m] = summarise(res, [budget])
        print("c_fp", c, {m: round(v["gain_mean"][0], 3) for m, v in out["c_fp"][str(c)].items()},
              flush=True)
    save("stress", out)


def exp_fooled(seeds=30, budget=50):
    """Correlated check errors: the judge is fooled by bad patches (FPR 0.6)."""
    out = {"budget": budget, "cases": {}}
    for name, cfg in {"default": WorldConfig(),
                      "judge fooled": replace(WorldConfig(), fooled_fpr=0.6)}.items():
        out["cases"][name] = {}
        for m in ["Uncertainty", "Uncertainty-MF", "CARVE-full-only", "CARVE-robust-check",
                  "CARVE-robust-est", "CARVE"]:
            res = run_grid([(cfg, m, s, [budget], (1.0, 0.1), None) for s in range(seeds)])
            out["cases"][name][m] = summarise(res, [budget])
        print("fooled", name, {m: round(v["gain_mean"][0], 3)
                               for m, v in out["cases"][name].items()}, flush=True)
    save("fooled", out)


def exp_calibrated(seeds=30, budgets=(20, 50, 100)):
    """Simulation with nuisance parameters measured on the real agents
    (results/calibrated_params.json, written by agent_exp.calib_params):
    check sensitivity / false-positive rate, spurious recovery, analyst accuracy
    and the measured single-step / full-replay cost ratio."""
    P = json.load(open(os.path.join(OUT, "calibrated_params.json")))
    out = {"params": P, "budgets": list(budgets), "envs": {}}
    cps = [0] + list(budgets)
    for env, p in P.items():
        cfg = replace(WorldConfig(), sens=p["sens"], fpr=p["fpr"], b=p["b"], r=p["r"])
        costs = (1.0, p["cost_ratio"])
        out["envs"][env] = {}
        for m in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF",
                  "CARVE-full-only", "CARVE"]:
            res = run_grid([(cfg, m, s, cps, costs, None) for s in range(seeds)])
            out["envs"][env][m] = summarise(res, cps)
            print("calibrated", env, m, np.round(out["envs"][env][m]["gain_mean"], 3), flush=True)
    save("calibrated", out)


def exp_prop1(reps=4000, delta=0.02):
    """Single-cell check of Proposition 1: predicted vs simulated observations to a
    decision under a sequential test with known p1, p0."""
    from scipy.stats import bernoulli  # noqa: F401  (scipy is a dependency anyway)
    rng = np.random.default_rng(0)
    p1, p0 = 0.8 * (6 / 9) + 0.2 * 0.05, 0.05
    kl = lambda a, b: a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    D1, D0 = kl(p1, p0), kl(p0, p1)
    A = np.log((1 - delta) / delta)
    llr1, llr0 = np.log(p1 / p0), np.log((1 - p1) / (1 - p0))
    rows = []
    for ell in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]:
        for e, p, D in [(1, p1, D1), (0, p0, D0)]:
            ns, wrong = [], 0
            for _ in range(reps):
                L, n = ell, 0
                while -A < L < A:
                    L += llr1 if rng.random() < p else llr0
                    n += 1
                ns.append(n)
                wrong += (L >= A) != (e == 1)
            pred = max(0.0, (A - ell) / D) if e == 1 else max(0.0, (A + ell) / D)
            rows.append({"ell": ell, "e": e, "pred": pred, "sim": float(np.mean(ns)),
                         "wrong": wrong / reps})
            print("prop1", rows[-1], flush=True)
    save("prop1", {"p1": p1, "p0": p0, "delta": delta, "rows": rows})


def _rsi(args):
    cfg, m, s = args
    return rsi_run(cfg, m, s)


def exp_rsi(seeds=30):
    cfg = WorldConfig()
    out = {"methods": {}}
    from sim.testbed import World
    # oracle ceiling: all true edges fixed with their best patch
    ceil = []
    for s in range(seeds):
        w = World(cfg, np.random.default_rng(s))
        keep = np.prod(np.where(w.E, 1 - w.Q.max(-1), 1.0), axis=1)
        ceil.append(1 - 0.6 * (w.f * keep).sum())
    out["oracle_success"] = float(np.mean(ceil))
    for m in ["LLM-only", "Replay-each", "Uncertainty", "Uncertainty-MF", "CARVE-full-only",
              "CARVE"]:
        with Pool(4) as p:
            H = np.array(p.map(_rsi, [(cfg, m, s) for s in range(seeds)]))
        out["methods"][m] = {
            "cost": H[:, :, 0].mean(0).tolist(),
            "success_mean": H[:, :, 1].mean(0).tolist(),
            "success_se": (H[:, :, 1].std(0, ddof=1) / np.sqrt(seeds)).tolist(),
        }
        print("rsi", m, np.round(H[:, :, 1].mean(0), 3), flush=True)
    save("rsi", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    exps = {"main": exp_main, "prior": exp_prior, "fidelity": exp_fidelity,
            "misspec": exp_misspec, "scale": exp_scale, "rsi": exp_rsi,
            "joint": exp_joint, "stress": exp_stress, "prop1": exp_prop1,
            "fooled": exp_fooled, "calibrated": exp_calibrated}
    for k, fn in exps.items():
        if which in (k, "all"):
            fn()
