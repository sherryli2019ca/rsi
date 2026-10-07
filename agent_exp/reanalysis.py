"""Reanalysis of the tau2 intervention banks requested in review (no new agent runs).

Stage `decide` runs every policy offline under several settings and caches the
patches it accepts; stage `report` scores the cached decisions under several
definitions of ground truth and cost.

Settings (policy side):
  main       full bank, internal change cost c = 0.02, budgets 10/20/40/80
  c0, c01, c05   same with internal c = 0, 0.01, 0.05 (only c-dependent policies)
  nostep     the step-limit component is removed from the bank
  split{t}   the bank is split in half at random (replays, null replays and
             regression runs of every patch / category); policies see half A,
             edges, repair rates and regressions are adjudicated on half B

Scoring (report side):
  sig        regressions charged only when significant (one-sided binomial vs r0)
  cont       every accepted patch charged its unthresholded regression estimate
             w (fail/n - r0), with a posterior bootstrap over the
             regression runs and r0
  per-wrong vs per-applied change cost c, c in {0, 0.01, 0.02, 0.05}
  natural / injected: failure shares f recomputed from natural or injected
             failures only

  python -m agent_exp.reanalysis --out runs/tau2_retail --domain tau2_retail --stage decide
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from agent_exp.bank import (METHODS, Domain, _load, _save, ground_truth, problem, rerun_stats,
                            run_method)

C_DEP = ["Uncertainty", "Uncertainty-MF", "CARVE-full-only", "CARVE", "CARVE-calibrated",
         "CARVE-robust-check", "Net"]
ALL = METHODS + ["HarnessFix-gate", "Net"]
N_SPLITS = 5
CTX = {}


def load(out, domain):
    D = Domain(domain)
    attrs = _load(os.path.join(out, "attributions.json"))
    tax = _load(os.path.join(out, "taxonomy.json"))
    bank = _load(os.path.join(out, "bank.json"))
    data = _load(os.path.join(out, "traces.json"))
    ev = _load(os.path.join(out, "evaluation_v4.json"))
    cats, counts, members, f = problem(D, attrs, tax)
    I, J = counts.shape
    inb = ground_truth(bank, I, J)[0]
    nat = np.array([sum(attrs[u]["true_fault"] is None for u in m) for m in members], float)
    inj = np.array([len(m) for m in members], float) - nat
    step = D.comp_ids.index("CFG.max_steps")
    ctx = dict(D=D, attrs=attrs, tax=tax, bank=bank, data=data, ev=ev, cats=cats,
               counts=counts, f=f, inb=inb, I=I, J=J, f_nat=nat / nat.sum(),
               f_inj=inj / inj.sum(), step=step, comp_ids=D.comp_ids)
    # unpatched runs on every regression task (second-review experiment A): each
    # patch's baseline is the failure rate of the tasks its regression runs used
    base = _load(os.path.join(out, "baseline_A.json"))
    if base:
        tasks = sorted(base)
        col = {t: n for n, t in enumerate(tasks)}
        M = np.zeros((I, J, 3, len(tasks)))
        for key, runs in bank["reg"].items():
            i, j, k = map(int, key.split(","))
            for r in runs:
                M[i, j, k, col[r["task"]]] += 1
        nfail = np.array([len(base[t]) - sum(base[t]) for t in tasks], float)
        nrun = np.array([len(base[t]) for t in tasks], float)
        ctx.update(base=base, base_tasks=tasks, base_M=M, base_nf=nfail, base_n=nrun)
    return ctx


def base_r0(p0):
    """Task-matched baseline failure rate of every patch, given per-task unpatched
    failure rates p0 (patches without regression runs get the mean)."""
    M = CTX["base_M"]
    n = M.sum(-1)
    return np.where(n > 0, (M @ p0) / np.maximum(n, 1), p0.mean())


def base_draws(n, rng):
    """Draws of each regression task's unpatched failure rate, resampling its runs
    (a per-task Jeffreys posterior would add about 0.5/11 to every task with no
    failures in 10 runs, biasing the matched baseline upwards)."""
    nf, nr = CTX["base_nf"], CTX["base_n"]
    return rng.binomial(nr.astype(int), nf / nr, (n, len(nf))) / nr


def split_bank(bank, t):
    """Two disjoint halves of every list of outcomes in the bank."""
    rng = np.random.default_rng(1000 + t)
    A = {"cells": {}, "null": {}, "reg": {}}
    B = {"cells": {}, "null": {}, "reg": {}}
    for key, c in bank["cells"].items():
        n = len(c["y"])
        p = rng.permutation(n)
        for H, idx in ((A, p[:n // 2]), (B, p[n // 2:])):
            H["cells"][key] = {k: [v[x] for x in idx] for k, v in c.items()}
    bycat = {}
    for key, v in bank["null"].items():
        bycat.setdefault(v["cat"], []).append(key)
    for cat, keys in bycat.items():
        p = rng.permutation(len(keys))
        for H, idx in ((A, p[:len(keys) // 2]), (B, p[len(keys) // 2:])):
            for x in idx:
                H["null"][keys[x]] = bank["null"][keys[x]]
    for key, runs in bank["reg"].items():
        p = rng.permutation(len(runs))
        A["reg"][key] = [runs[x] for x in p[:len(runs) // 2]]
        B["reg"][key] = [runs[x] for x in p[len(runs) // 2:]]
    return A, B


def split_bank_grouped(bank, t):
    """Two halves that share no source trajectory and no regression task: within
    each category the failed trajectories that replays and null replays start
    from are split in half, and so are the regression tasks; every recorded
    outcome goes to the half of its source. Categories with a single source
    trajectory fall back to splitting outcomes (counted in `fallback`)."""
    rng = np.random.default_rng(2000 + t)
    A = {"cells": {}, "null": {}, "reg": {}}
    B = {"cells": {}, "null": {}, "reg": {}}
    src = {}
    for key, c in bank["cells"].items():
        src.setdefault(int(key.split(",")[0]), set()).update(c["trace"])
    for v in bank["null"].values():
        src.setdefault(v["cat"], set()).add(v["trace"])
    inA, fallback = {}, []
    for cat, us in src.items():
        us = sorted(us)
        if len(us) < 2:
            fallback.append(cat)
            continue
        p = rng.permutation(len(us))
        half = len(us) // 2 + (len(us) % 2) * int(rng.integers(2))
        for n, x in enumerate(p):
            inA[us[x]] = n < half
    for key, c in bank["cells"].items():
        cat = int(key.split(",")[0])
        n = len(c["y"])
        if cat in fallback:
            p = rng.permutation(n)
            sides = (p[:n // 2], p[n // 2:])
        else:
            sides = ([x for x in range(n) if inA[c["trace"][x]]],
                     [x for x in range(n) if not inA[c["trace"][x]]])
        for H, idx in zip((A, B), sides):
            H["cells"][key] = {k: [v[x] for x in idx] for k, v in c.items()}
    for key, v in bank["null"].items():
        if v["cat"] in fallback:
            (A if rng.random() < 0.5 else B)["null"][key] = v
        else:
            (A if inA[v["trace"]] else B)["null"][key] = v
    tasks = sorted({r["task"] for runs in bank["reg"].values() for r in runs})
    p = rng.permutation(len(tasks))
    tA = {tasks[x] for x in p[:len(tasks) // 2]}
    for key, runs in bank["reg"].items():
        A["reg"][key] = [r for r in runs if r["task"] in tA]
        B["reg"][key] = [r for r in runs if r["task"] not in tA]
    A["fallback"] = B["fallback"] = fallback
    return A, B


def inputs_from(bank):
    """Experiment costs and judge calibration estimated from one bank half (as
    stage_evaluate does on the full bank)."""
    cells = [c for c in bank["cells"].values() if c["y"]]
    tf = np.mean([t for c in cells for t in c["tok_full"]])
    ts = np.mean([t for c in cells for t in c["tok_single"]])
    regs = [r["tok"] for v in bank["reg"].values() for r in v]
    costs = (1.0, ts / tf, (np.mean(regs) if regs else tf) / tf)
    ys = np.array([y for c in cells for y in c["y"]])
    zs = np.array([z for c in cells for z in c["z"]])
    bad = np.array([np.mean(c["y"]) < 0.3 for c in cells for _ in c["y"]])
    cal = {"sens": float(zs[ys == 1].mean()), "fpr": float(zs[ys == 0].mean())}
    cal["fpr_bad"] = float(zs[(ys == 0) & bad].mean()) if ((ys == 0) & bad).any() else cal["fpr"]
    cal["fpr_good"] = float(zs[(ys == 0) & ~bad].mean()) if ((ys == 0) & ~bad).any() else cal["fpr"]
    return costs, cal


OVR = {}


def setting(name):
    """(bank seen by policies, in-bank mask, internal c, methods, budgets, seeds)."""
    bank, inb = CTX["bank"], CTX["inb"].copy()
    if name == "main":
        return bank, inb, 0.02, ALL, (10, 20, 40, 80), 200
    if name in ("c0", "c01", "c05"):
        return bank, inb, {"c0": 0.0, "c01": 0.01, "c05": 0.05}[name], C_DEP, (10, 40, 80), 100
    if name == "nostep":
        inb[:, CTX["step"]] = False
        return bank, inb, 0.02, ALL, (10, 40, 80), 100
    if name == "norep":
        # run with BANK_NOREPLACE=1: outcomes drawn without replacement
        assert os.environ.get("BANK_NOREPLACE") == "1"
        return bank, inb, 0.02, ALL, (10, 40, 80), 100
    if name == "netabl":
        # net-effect regression term without clipping / rescaling (third review)
        return bank, inb, 0.02, ["Net-noclip", "Net-norescale", "Net-raw"], (10, 40, 80), 40
    if name.startswith("split"):
        A, _ = split_bank(bank, int(name[5:]))
        return A, inb, 0.02, ALL, (10, 40, 80), 40
    r_aud = float(CTX["ev"]["attr_acc_injected"])
    if name == "audit":
        # fifth review: edge priors at the audited analyst accuracy
        OVR[name] = {"r_hat": r_aud}
        return bank, inb, 0.02, ALL, (10, 40, 80), 40
    if name.startswith("clean"):
        # fifth review: halves share no source trajectory or regression task, and
        # the policy's prior, costs and judge calibration come from its own half
        A, _ = split_bank_grouped(bank, int(name[5:]))
        costs, cal = inputs_from(A)
        OVR[name] = {"r_hat": r_aud, "costs": costs, "cal": cal}
        return A, inb, 0.02, ALL, (10, 40, 80), 20
    raise ValueError(name)


def _job(args):
    name, m, B, s = args
    bank, inb, c, _, _, _ = SET[name]
    ev = CTX["ev"]
    o = OVR.get(name, {})
    b_hat = float(np.mean([v["y"] for v in bank["null"].values()]))
    acc, patch, spent = run_method(m, bank, CTX["counts"], CTX["f"], inb, o.get("costs", ev["costs"]),
                                   b_hat, s, B, c_fp=c, cal=o.get("cal", ev["calibration"]),
                                   r0=ev["r0"], w=ev["w"], r_hat=o.get("r_hat", 0.7))
    cells = [[int(i), int(j), int(patch[i, j])] for i, j in zip(*np.nonzero(acc))]
    return f"{name}|{m}|{int(B)}|{s}", [cells, float(spent)]


SET = {}


def stage_decide(args):
    path = os.path.join(args.out, "reanalysis_decisions.json")
    dec = _load(path) or {}
    names = args.settings or (["main", "c0", "c01", "c05", "nostep"] +
                              [f"split{t}" for t in range(N_SPLITS)])
    for name in names:
        SET[name] = setting(name)
    jobs = []
    for name in names:
        _, _, _, methods, budgets, seeds = SET[name]
        for m in methods:
            for B in budgets:
                for s in range(seeds):
                    if f"{name}|{m}|{int(B)}|{s}" not in dec:
                        jobs.append((name, m, B, s))
    # slow (Net) jobs first so the pool stays busy
    jobs.sort(key=lambda j: (j[1] != "Net", -j[2]))
    print(len(jobs), "jobs", flush=True)
    with Pool(args.workers) as pool:
        for n, (key, val) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4)):
            dec[key] = val
            if n % 500 == 499:
                _save(path, dec)
                print(n + 1, flush=True)
    _save(path, dec)


# ---- scoring -------------------------------------------------------------------------
def truth(bank, I, J, r0, w, alpha=0.05):
    """Edges, repair rates, regression estimates (significant and continuous) and
    regression run counts from one bank (or bank half)."""
    from scipy.stats import binom
    inb, E, Q, b = ground_truth(bank, I, J)
    K = 3
    R_sig, R_cont = np.zeros((I, J, K)), np.zeros((I, J, K))
    nf, n = np.zeros((I, J, K)), np.zeros((I, J, K))
    for key, runs in bank["reg"].items():
        i, j, k = map(int, key.split(","))
        n[i, j, k] = len(runs)
        nf[i, j, k] = len(runs) - sum(r["y"] for r in runs)
        if not runs:
            continue
        r0k = r0[i, j, k] if np.ndim(r0) else r0
        ex = w * (nf[i, j, k] / n[i, j, k] - r0k)
        R_cont[i, j, k] = ex
        if binom.sf(nf[i, j, k] - 1, n[i, j, k], r0k) < alpha:
            R_sig[i, j, k] = max(0.0, ex)
    # edge-free repair estimate: unclipped (y - b) / (1 - b) of every cell
    Qraw = np.zeros((I, J, K))
    for key, c in bank["cells"].items():
        i, j, k = map(int, key.split(","))
        Qraw[i, j, k] = (np.mean(c["y"]) - b[i]) / max(1e-9, 1 - b[i])
    return dict(E=E, Q=Q, Qraw=Qraw, b=b, R_sig=R_sig, R_cont=R_cont, nf=nf, n=n)


def repair(cells, T, f, edge_free=False):
    """Repaired failure mass of a set of accepted (i, j, k)."""
    by = {}
    for i, j, k in cells:
        q = T["Qraw"][i, j, k] if edge_free else (T["Q"][i, j, k] if T["E"][i, j] else 0.0)
        by.setdefault(i, []).append(q)
    g = 0.0
    for i, qs in by.items():
        pos = [q for q in qs if q > 0]
        g += f[i] * ((1 - np.prod([1 - q for q in pos])) + sum(q for q in qs if q < 0))
    return g


def score(cells, T, f, R, c, ctype, edge_free=False):
    rep = repair(cells, T, f, edge_free)
    reg = sum(R[i, j, k] for i, j, k in cells)
    nw = sum(not T["E"][i, j] for i, j, k in cells)
    return rep - reg - c * (len(cells) if ctype == "apply" else nw)


def oracle(T, f, R, c, ctype, inb, edge_free=False):
    """Best value from knowing every cell's measurements: per cell the patch with the
    largest net value, kept when positive (cells are scored independently)."""
    cells = []
    I, J = inb.shape
    for i in range(I):
        for j in range(J):
            if not inb[i, j] or (not edge_free and not T["E"][i, j]):
                continue
            q = T["Qraw"][i, j] if edge_free else T["Q"][i, j]
            pen = c if (ctype == "apply" or not T["E"][i, j]) else 0.0
            v = f[i] * q - R[i, j] - pen
            k = int(np.argmax(v))
            if v[k] > 0:
                cells.append((i, j, k))
    return score(cells, T, f, R, c, ctype, edge_free), cells


def yardstick(T, f):
    """Repair-only oracle: failure mass repaired by the true edges' best patches,
    with no regression or change cost. Every value is reported as a fraction of
    this fixed quantity, so that noisy regression estimates cannot inflate the
    denominator."""
    cells = [(i, j, int(np.argmax(T["Q"][i, j]))) for i, j in zip(*np.nonzero(T["E"]))]
    return repair(cells, T, f)


def summarise(dec, name, methods, budgets, T, f, R, c, ctype, inb, edge_free=False, o=None):
    o = yardstick(CTX["T"], f) if o is None else o
    out = {}
    for m in methods:
        for B in budgets:
            vals = [score(v[0], T, f, R, c, ctype, edge_free) for k, v in dec.items()
                    if k.startswith(f"{name}|{m}|{B}|")]
            if vals:
                v = np.array(vals) / o
                out[f"{m}|{B}"] = [float(v.mean()), float(v.std() / np.sqrt(len(v)))]
    return o, out


def stage_report(args):
    dec = _load(os.path.join(args.out, "reanalysis_decisions.json"))
    ev, I, J, f, inb = CTX["ev"], CTX["I"], CTX["J"], CTX["f"], CTX["inb"]
    r0, w = ev["r0"], ev["w"]
    T = truth(CTX["bank"], I, J, r0, w)
    CTX["T"] = T
    rm = matched_r0(CTX["bank"], CTX["data"])
    Tm = truth(CTX["bank"], I, J, rm, w)
    T["R_contm"] = Tm["R_cont"]
    regs = ("sig", "cont", "contm")
    if "base" in CTX:
        # same scorings against the task-matched unpatched baseline (experiment
        # A; a binomial test that ignores the baseline's own noise)
        TA = truth(CTX["bank"], I, J, base_r0(CTX["base_nf"] / CTX["base_n"]), w)
        T["R_sigA"], T["R_contA"] = TA["R_sig"], TA["R_cont"]
        regs += ("sigA", "contA")
    rep = {"r0": r0, "r0_matched": rm, "w": w, "n_edges": int(T["E"].sum()),
           "yardstick": yardstick(T, f)}
    for c in (0.0, 0.02):
        for ctype in ("wrong", "apply"):
            rep[f"oracle|sig|{ctype}|{c}"] = oracle(T, f, T["R_sig"], c, ctype, inb)[0] / rep["yardstick"]
    B4, B3 = (10, 20, 40, 80), (10, 40, 80)

    # 1. main table under significant vs continuous regression estimates
    for reg in regs:
        for c in (0.0, 0.02):
            o, s = summarise(dec, "main", ALL, B4, T, f, T[f"R_{reg}"], c, "wrong", inb)
            rep[f"main|{reg}|wrong|{c}"] = {"oracle": o, "res": s}

    # 2. change-cost sweep: per wrong vs per applied change, policies told the same c
    for c, nm in ((0.0, "c0"), (0.01, "c01"), (0.02, "main"), (0.05, "c05")):
        for ctype in ("wrong", "apply"):
            merged = {k.replace("main|", f"{nm}|", 1): v for k, v in dec.items()
                      if k.startswith("main|") and k.split("|")[1] not in C_DEP}
            merged.update({k: v for k, v in dec.items() if k.startswith(f"{nm}|")})
            o, s = summarise(merged, nm, ALL, B3, T, f, T["R_sig"], c, ctype, inb)
            rep[f"sweep|{ctype}|{c}"] = {"oracle": o, "res": s}

    # 3. without the step-limit component
    inb_ns = inb.copy()
    inb_ns[:, CTX["step"]] = False
    Tns = dict(T, E=T["E"] & inb_ns)
    for c in (0.0, 0.02):
        o, s = summarise(dec, "nostep", ALL, B3, Tns, f, T["R_sig"], c, "wrong", inb_ns)
        rep[f"nostep|{c}"] = {"oracle": o, "res": s, "n_edges": int(Tns["E"].sum()),
                              "oracle_nostep": oracle(Tns, f, T["R_sig"], c, "wrong", inb_ns)[0] / o}

    # 4. natural vs injected failure shares (same decisions, reweighted value)
    for nm, fw in (("natural", CTX["f_nat"]), ("injected", CTX["f_inj"])):
        for c in (0.0, 0.02):
            o, s = summarise(dec, "main", ALL, B3, T, fw, T["R_sig"], c, "wrong", inb)
            rep[f"{nm}|{c}"] = {"oracle": o, "res": s}

    # 5. split bank: adjudicate and evaluate on the half the policy never saw
    split = {}
    for t in range(N_SPLITS):
        _, Bh = split_bank(CTX["bank"], t)
        Tb = truth(Bh, I, J, r0, w)
        split.setdefault("n_edges", []).append(int(Tb["E"].sum()))
        split.setdefault("agree_edges", []).append(int((Tb["E"] & T["E"]).sum()))
        split.setdefault("yard_B", []).append(yardstick(Tb, f) / yardstick(T, f))
        for nm, kw in (("edge", {}), ("edgefree", {"edge_free": True})):
            for c in (0.0, 0.02):
                o, s = summarise(dec, f"split{t}", ALL, B3, Tb, f, Tb["R_cont" if kw else "R_sig"],
                                 c, "wrong", inb, **kw)
                for k, v in s.items():
                    split.setdefault(f"{nm}|{c}|{k}", []).append(v[0])
                split.setdefault(f"{nm}|{c}|oracle", []).append(o)
    rep["split"] = {k: [float(np.mean(v)), float(np.std(v) / np.sqrt(len(v)))]
                    if not k.endswith("edges") else v for k, v in split.items()}

    # 6. what perfect verification could add to the no-evidence default, by source
    for reg in regs:
        for c, ctype in ((0.0, "wrong"), (0.02, "wrong"), (0.02, "apply")):
            rep[f"vpi|{reg}|{ctype}|{c}"] = vpi_decomp(dec, T, f, T[f"R_{reg}"], c, ctype, inb)

    # 7. posterior expected utility: hierarchical regression posterior, r0
    # posterior and a bootstrap of the bank's adjudication, decisions fixed
    # (with experiment A the primary posterior uses the task-matched unpatched
    # baseline; "post_paired" keeps the paired-trial r0 of the training runs)
    for name in ("main", "norep"):
        if any(k.startswith(f"{name}|") for k in dec):
            rep[f"post|{name}"] = posterior_eval(dec, name, T, r0, w, use_base="base" in CTX)
            if "base" in CTX and name == "main":
                rep[f"post_paired|{name}"] = posterior_eval(dec, name, T, r0, w)
    rep["post_sig|norep"] = {f"{c}": summarise(dec, "norep", ALL, B3, T, f, T["R_sig"], c, "wrong", inb)[1]
                             for c in (0.0, 0.02)} if any(k.startswith("norep|") for k in dec) else None

    # 8. one patch per component, as deployed: keep the cell of the most frequent category
    comp = {}
    for k, v in dec.items():
        if k.startswith("main|"):
            best = {}
            for i, j, kk in v[0]:
                if j not in best or f[i] > f[best[j][0]]:
                    best[j] = (i, j, kk)
            comp[k] = [[list(x) for x in best.values()], v[1]]
    for c in (0.0, 0.02):
        rep[f"component|{c}"] = summarise(comp, "main", ALL, B3, T, f, T["R_sig"], c, "wrong", inb)[1]

    # 9. regression evidence: per-patch CIs, equivalence, baselines
    rep["regression"] = regression_report(dec, T, r0, w)
    _save(os.path.join(args.out, "reanalysis.json"), rep)
    print(json.dumps({k: v for k, v in rep.items() if k in ("n_edges", "regression")}, indent=1))


def vpi_decomp(dec, T, f, R, c, ctype, inb, o=None):
    """Realised value of perfect information over the apply-attributed default, per
    cell and additive (Prop. 1): avoided harm of default patches with negative net
    effect, better patch choice in default cells, and discovery of cells the
    default leaves out. Units: fraction of the oracle's value."""
    default = {(i, j): k for i, j, k in dec["main|LLM-only|10|0"][0]}
    I, J = inb.shape
    harm = select = disc = 0.0
    for i in range(I):
        for j in range(J):
            if not inb[i, j]:
                continue
            pen = c if (ctype == "apply" or not T["E"][i, j]) else 0.0
            d = f[i] * T["Q"][i, j] * T["E"][i, j] - R[i, j] - pen
            best = max(0.0, d.max())
            if (i, j) in default:
                dk = d[default[i, j]]
                harm += max(0.0, -dk)
                select += best - max(0.0, dk)
            else:
                disc += best
    o = yardstick(T, f) if o is None else o
    return {"harm": harm / o, "select": select / o, "discover": disc / o,
            "default": score(dec["main|LLM-only|10|0"][0], T, f, R, c, ctype) / o}


def hier_rho(nf, n, r0, rng):
    """One posterior draw of every patch's absolute excess failure rate rho_k, with
    rho_k ~ N(mu, tau^2) across patches (empirical Bayes, method of moments) given
    a draw of r0; patches without runs get mu."""
    has = n > 0
    y = np.where(has, nf / np.maximum(n, 1), 0.0) - r0
    pbar = nf[has].sum() / n[has].sum()
    v = np.where(has, pbar * (1 - pbar) / np.maximum(n, 1), np.inf)
    yy, vv = y[has], v[has]
    tau2 = max(0.0, np.var(yy) - vv.mean())
    wts = 1 / (vv + tau2)
    mu_hat = (wts * yy).sum() / wts.sum()
    mu = mu_hat + rng.standard_normal() * np.sqrt(1 / wts.sum())
    if tau2 == 0:
        return np.full(n.shape, mu), mu, tau2
    prec = 1 / v + 1 / tau2
    m = (np.where(has, y / v, 0.0) + mu / tau2) / prec
    return m + rng.standard_normal(n.shape) / np.sqrt(prec), mu, tau2


def boot_bank(bank, rng):
    """Nonparametric bootstrap of the replay outcomes behind the edge labels."""
    B = {"cells": {}, "null": {}, "reg": bank["reg"]}
    for key, c in bank["cells"].items():
        idx = rng.integers(len(c["y"]), size=len(c["y"]))
        B["cells"][key] = {"y": [c["y"][x] for x in idx]}
    bycat = {}
    for key, v in bank["null"].items():
        bycat.setdefault(v["cat"], []).append(key)
    for cat, keys in bycat.items():
        for n, x in enumerate(rng.integers(len(keys), size=len(keys))):
            B["null"][f"{cat}|b{n}"] = bank["null"][keys[x]]
    return B


def posterior_eval(dec, name, T, r0, w, n_draw=300, n_seed=40, budgets=(10, 40, 80),
                   use_base=False):
    """Value of each policy's decisions under uncertainty about harm and about the
    bank's adjudication: each draw takes r0 from its posterior, every patch's
    excess failure rate from a hierarchical posterior, and edges and repair rates
    from a bootstrap of the replays. Values are fractions of the fixed yardstick."""
    rng = np.random.default_rng(1)
    f, inb, bank = CTX["f"], CTX["inb"], CTX["bank"]
    I, J = inb.shape
    r0s, _ = r0_draws(CTX["data"], n_draw, rng)
    if use_base:
        p0s = base_draws(n_draw, rng)
        r0s = [base_r0(p) for p in p0s]
    o = yardstick(T, f)
    groups = {}
    for key, v in dec.items():
        nm, m, B, s = key.split("|")
        if nm == name and int(B) in budgets and int(s) < n_seed:
            groups.setdefault(f"{m}|{B}", []).append(v[0])
    vals = {c: {g: [] for g in groups} for c in (0.0, 0.02)}
    mus, taus, edges, vpis = [], [], [], {}
    for d in range(n_draw):
        rho, mu, tau2 = hier_rho(T["nf"], T["n"], r0s[d], rng)
        mus.append(mu)
        taus.append(np.sqrt(tau2))
        Tb = truth(boot_bank(bank, rng), I, J, r0, w)
        edges.append(int(Tb["E"].sum()))
        R = w * rho
        for c in (0.0, 0.02):
            vd = vpi_decomp(dec, Tb, f, R, c, "wrong", inb, o=o)
            for k, v in vd.items():
                vpis.setdefault(f"{c}", {}).setdefault(k, []).append(v)
        for g, sets in groups.items():
            rep_ = np.mean([repair(cs, Tb, f) for cs in sets])
            reg_ = np.mean([sum(R[i, j, k] for i, j, k in cs) for cs in sets])
            nw = np.mean([sum(not Tb["E"][i, j] for i, j, k in cs) for cs in sets])
            for c in vals:
                vals[c][g].append((rep_ - reg_ - c * nw) / o)
    out = {"mu": [float(np.mean(mus)), float(np.quantile(mus, 0.05)), float(np.quantile(mus, 0.95))],
           "tau": float(np.mean(taus)),
           "edges": [float(np.mean(edges)), int(np.min(edges)), int(np.max(edges))],
           "vpi": {c: {k: [float(np.mean(v)), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
                       for k, v in d.items()} for c, d in vpis.items()}}
    for c, vs in vals.items():
        res = {}
        for g, v in vs.items():
            v = np.array(v)
            B = g.split("|")[1]
            ref = np.array(vs[f"LLM-only|{B}"])
            dlt = v - ref
            res[g] = {"mean": float(v.mean()), "lo": float(np.quantile(v, 0.05)),
                      "hi": float(np.quantile(v, 0.95)),
                      "d_lo": float(np.quantile(dlt, 0.05)), "d_hi": float(np.quantile(dlt, 0.95)),
                      "p_better": float((dlt > 0).mean())}
        for B in budgets:
            gs = [g for g in vs if g.endswith(f"|{B}")]
            M = np.array([vs[g] for g in gs])
            best = np.bincount(M.argmax(0), minlength=len(gs)) / M.shape[1]
            res[f"p_best|{B}"] = {g.split("|")[0]: float(b) for g, b in zip(gs, best)}
        out[f"{c}"] = res
    return out


def matched_r0(bank, data):
    """Paired-trial r0 restricted to the tasks used in regression runs, weighted by
    how often each was used."""
    by = {}
    for d in data:
        if d["variant"] is None:
            by.setdefault(d["task_id"], {})[d["trial"]] = d["reward"]
    use = {}
    for v in bank["reg"].values():
        for r in v:
            use[r["task"]] = use.get(r["task"], 0) + 1
    num = den = 0.0
    for t, c in use.items():
        v = by.get(t, {})
        if 0 in v and 1 in v and (v[0] or v[1]):
            after = ([v[1]] if v[0] else []) + ([v[0]] if v[1] else [])
            num += c * (1 - np.mean(after))
            den += c
    return num / den


def r0_draws(data, n, rng):
    """Posterior draws of the natural re-run failure rate (Jeffreys prior)."""
    by = {}
    for d in data:
        if d["variant"] is None:
            by.setdefault(d["task_id"], {})[d["trial"]] = d["reward"]
    pairs = [(v[0], v[1]) for v in by.values() if 0 in v and 1 in v]
    after = [b for a, b in pairs if a] + [a for a, b in pairs if b]
    fails = len(after) - sum(after)
    return rng.beta(fails + 0.5, len(after) - fails + 0.5, n), len(after)


def regression_report(dec, T, r0, w, n_draw=4000, margins=(0.05, 0.10, 0.15)):
    from scipy.stats import norm
    rng = np.random.default_rng(0)
    bank, data = CTX["bank"], CTX["data"]
    r0s, n_r0 = r0_draws(data, n_draw, rng)
    acc_keys = sorted({tuple(c) for k, v in dec.items() if k.startswith("main|") for c in v[0]})
    keys20 = [k for k in acc_keys if T["n"][k] >= 20]
    out = {"n_r0_obs": n_r0, "accepted_patches": len(acc_keys),
           "accepted_with_20_runs": len(keys20)}
    # per-patch excess failure and equivalence (posterior over patch rate and r0)
    rows = []
    for k in keys20:
        nf, n = T["nf"][k], T["n"][k]
        p = rng.beta(nf + 0.5, n - nf + 0.5, n_draw)
        rho = p - r0s
        lo, hi = np.quantile(rho, [0.05, 0.95])
        rows.append(dict(key=list(map(int, k)), edge=bool(T["E"][k[:2]]), fail=nf / n,
                         rho=float(np.mean(rho)), lo=float(lo), hi=float(hi)))
    out["patches"] = rows
    for m in margins:
        # TOST at level 0.05 <=> the 90% interval lies inside (-m, m)
        out[f"equivalent_{m}"] = sum(r["hi"] < m and r["lo"] > -m for r in rows)
        out[f"harm_excluded_{m}"] = sum(r["hi"] < m for r in rows)
    # pooled excess of wrong-edge and true-edge patches (task-clustered s.e.)
    for grp in ("wrong", "true"):
        sel = [k for k in keys20 if bool(T["E"][k[:2]]) == (grp == "true")]
        runs = [(r["task"], 1 - r["y"]) for k in sel for r in bank["reg"][f"{k[0]},{k[1]},{k[2]}"]]
        y = np.array([x for _, x in runs], float)
        tasks = sorted({t for t, _ in runs})
        idx = {t: n for n, t in enumerate(tasks)}
        tot = np.zeros(len(tasks))
        cnt = np.zeros(len(tasks))
        for t, x in runs:
            tot[idx[t]] += x
            cnt[idx[t]] += 1
        mean = y.mean()
        se_task = np.sqrt(((tot - mean * cnt) ** 2).sum()) / cnt.sum()
        draws = mean + se_task * rng.standard_normal(n_draw) - r0s
        lo, hi = np.quantile(draws, [0.05, 0.95])
        out[f"pooled_{grp}"] = dict(n_patches=len(sel), n_runs=len(y), fail=float(mean),
                                    se_task=float(se_task), rho=float(draws.mean()),
                                    lo=float(lo), hi=float(hi))
    # baselines for r0: paired-trial estimate, the same estimate restricted to and
    # weighted by the tasks used in regression runs, and the mean over all patches
    by = {}
    for d in data:
        if d["variant"] is None:
            by.setdefault(d["task_id"], {})[d["trial"]] = d["reward"]
    use = {}
    for v in bank["reg"].values():
        for r in v:
            use[r["task"]] = use.get(r["task"], 0) + 1
    num = den = 0.0
    for t, c in use.items():
        v = by.get(t, {})
        if 0 in v and 1 in v and (v[0] or v[1]):
            after = ([v[1]] if v[0] else []) + ([v[0]] if v[1] else [])
            num += c * (1 - np.mean(after))
            den += c
    allruns = [1 - r["y"] for v in bank["reg"].values() for r in v]
    out["r0_paired"] = r0
    out["r0_matched"] = num / den
    out["fail_all_patches"] = float(np.mean(allruns))
    # task-adjusted comparison between patches: is any patch worse than the other
    # patches on the same tasks? (leave-one-out task rates, normal approximation)
    task_f, task_n = {}, {}
    for v in bank["reg"].values():
        for r in v:
            task_f[r["task"]] = task_f.get(r["task"], 0) + 1 - r["y"]
            task_n[r["task"]] = task_n.get(r["task"], 0) + 1
    worse = []
    for k in keys20:
        runs = bank["reg"][f"{k[0]},{k[1]},{k[2]}"]
        ex = var = obs = 0.0
        for r in runs:
            t = r["task"]
            fo, no = task_f[t] - (1 - r["y"]), task_n[t] - 1
            p = (fo + 0.5) / (no + 1)
            ex += p
            var += p * (1 - p)
            obs += 1 - r["y"]
        worse.append(float(norm.sf((obs - ex - 0.5) / np.sqrt(max(var, 1e-9)))))
    out["task_adjusted_worse_p05"] = int(sum(p < 0.05 for p in worse))
    out["task_adjusted_tested"] = len(worse)
    # continuous-regression policy values with posterior uncertainty
    out["cont_bootstrap"] = cont_bootstrap(dec, T, r0s, w, rng)
    if "base" in CTX:
        out["A"] = regression_vs_base(T, keys20, margins, rng, n_draw)
    return out


def regression_vs_base(T, keys20, margins, rng, n_draw):
    """Excess failure of patches over the unpatched agent on the same tasks
    (experiment A). Per patch: posterior of its failure rate minus the posterior
    of its task-matched baseline. Pooled: run-weighted mean of (patched failure -
    unpatched failure rate of that task), with a bootstrap over tasks that also
    resamples the unpatched runs within each task."""
    bank, base = CTX["bank"], CTX["base"]
    tasks = CTX["base_tasks"]
    nf0, n0 = CTX["base_nf"], CTX["base_n"]
    out = {"n_tasks": len(tasks), "n_runs": int(n0.sum()),
           "fail_unpatched": float(nf0.sum() / n0.sum()),
           "fail_unpatched_matched": float(np.mean([base_r0(nf0 / n0)[k] for k in keys20]))}
    p0s = base_draws(n_draw, rng)
    M = CTX["base_M"]
    rows = []
    for k in keys20:
        nf, n = T["nf"][k], T["n"][k]
        p = rng.beta(nf + 0.5, n - nf + 0.5, n_draw)
        r0k = (p0s @ M[k]) / M[k].sum()
        rho = p - r0k
        lo, hi = np.quantile(rho, [0.05, 0.95])
        rows.append(dict(key=list(map(int, k)), edge=bool(T["E"][k[:2]]), fail=nf / n,
                         base=float(r0k.mean()), rho=float(rho.mean()), lo=float(lo),
                         hi=float(hi)))
    out["patches"] = rows
    for m in margins:
        out[f"equivalent_{m}"] = sum(r["hi"] < m and r["lo"] > -m for r in rows)
        out[f"harm_excluded_{m}"] = sum(r["hi"] < m for r in rows)
    out["harm_significant"] = sum(r["lo"] > 0 for r in rows)
    out["benefit_significant"] = sum(r["hi"] < 0 for r in rows)
    col = {t: n for n, t in enumerate(tasks)}
    for grp in ("all", "wrong", "true"):
        sel = [k for k in keys20 if grp == "all" or bool(T["E"][k[:2]]) == (grp == "true")]
        tf, tn = np.zeros(len(tasks)), np.zeros(len(tasks))
        for k in sel:
            for r in bank["reg"][f"{k[0]},{k[1]},{k[2]}"]:
                tf[col[r["task"]]] += 1 - r["y"]
                tn[col[r["task"]]] += 1
        use = np.nonzero(tn)[0]

        def est(idx, p0):
            return float((tf[idx] - tn[idx] * p0[idx]).sum() / tn[idx].sum())
        point = est(use, nf0 / n0)
        bs = []
        for _ in range(2000):
            idx = rng.choice(use, len(use))
            pf = rng.binomial(tn[idx].astype(int), np.clip(tf[idx] / tn[idx], 0, 1))
            p0 = rng.binomial(n0[idx].astype(int), nf0[idx] / n0[idx]) / n0[idx]
            bs.append(float((pf - tn[idx] * p0).sum() / tn[idx].sum()))
        lo, hi = np.quantile(bs, [0.05, 0.95])
        out[f"pooled_{grp}"] = dict(n_patches=len(sel), n_runs=int(tn.sum()),
                                    fail=float(tf.sum() / tn.sum()), rho=point,
                                    lo=float(lo), hi=float(hi), p_harm=float(np.mean(np.array(bs) > 0)))
    return out


def cont_bootstrap(dec, T, r0s, w, rng, n=1000):
    """Policy value under continuous regression estimates, drawing each patch's
    failure rate and r0 from their posteriors; decisions are fixed."""
    f, inb = CTX["f"], CTX["inb"]
    I, J, K = T["n"].shape
    res = {}
    groups = {}
    for key, v in dec.items():
        name, m, B, s = key.split("|")
        if name == "main" and B in ("10", "40", "80"):
            groups.setdefault(f"{m}|{B}", []).append(v[0])
    rep = {g: np.mean([repair(cs, T, f) for cs in v]) for g, v in groups.items()}
    freq = {}
    for g, v in groups.items():
        fr = np.zeros((I, J, K))
        for cs in v:
            for i, j, k in cs:
                fr[i, j, k] += 1
        freq[g] = fr / len(v)
    vals = {g: [] for g in groups}
    for d in range(n):
        r0 = r0s[d]
        p = rng.beta(T["nf"] + 0.5, T["n"] - T["nf"] + 0.5)
        R = np.where(T["n"] > 0, w * (p - r0), 0.0)
        o = yardstick(T, f)
        for g in groups:
            vals[g].append((rep[g] - (freq[g] * R).sum()) / o)
    for g, v in vals.items():
        v = np.array(v)
        res[g] = [float(v.mean()), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
    # how often each policy is best at its budget
    for B in ("10", "40", "80"):
        gs = [g for g in groups if g.endswith(f"|{B}")]
        M = np.array([vals[g] for g in gs])
        best = np.bincount(M.argmax(0), minlength=len(gs)) / M.shape[1]
        res[f"p_best|{B}"] = {g.split("|")[0]: float(b) for g, b in zip(gs, best)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--stage", default="decide")
    ap.add_argument("--settings", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CTX.update(load(args.out, args.domain))
    if args.stage == "decide":
        stage_decide(args)
    else:
        stage_report(args)


if __name__ == "__main__":
    main()
