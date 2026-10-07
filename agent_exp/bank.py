"""Offline intervention bank on a real benchmark (tau2-bench retail) and
evaluation of verification policies by resampling from it.

Stages (cached under --out):
  collect    run the agent on training tasks: clean runs (natural failures) and
             runs with one injected component fault
  attribute  analyst root cause, step and component for every failure
  taxonomy   induce failure categories and label failures
  patches    3 candidate patches for every cell in the bank
  bank       for each bank cell: n paired (full replay, single-step check) outcomes
             per patch on failures of that category, null replays (no patch) from
             the same steps to estimate spurious recovery b, and regression runs
             (whole episodes on previously successful tasks with the patch applied)
  evaluate   resample the bank to run every policy at several budgets and seeds

Ground truth: an edge (category i, component j) is true when its best patch's
full-replay success exceeds the category's null-replay success by at least
`delta` (one-sided Fisher exact test p < 0.05). This needs no injected faults,
so natural failures count; for injected failures we also report the injected
component.

Run with an interpreter where tau2-bench is installed:
  python -m agent_exp.bank --out runs/tau2_retail --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from agent_exp.analyst import attribute, induce_taxonomy, judge_step, label, make_patches
from carve.model import FULL, SINGLE, CategoryBPosterior, GraphPosterior, ObsParams
from carve.policies import (CARVE, LLMOnly, ReplayEach, UncertaintyMF, UncertaintySampling,
                            bayes_decide)
from sim.testbed import edge_prior


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.__dict__)
    os.replace(tmp, path)


def _pmap(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


class Domain:
    def __init__(self, name, split="train"):
        if name not in ("tau2_retail", "tau2_airline"):
            raise ValueError(name)
        os.environ["TAU2_DOMAIN"] = name.split("_", 1)[1]
        from agent_exp import tau2_env as m
        from agent_exp.tau2_env import Step, Trace

        self.m, self.Step, self.Trace = m, Step, Trace
        self.docs = m.COMPONENT_DOCS
        self.comp_ids = list(m.COMPONENT_DOCS)
        self.faults = m.FAULTS
        self.tasks = {t.id: t for t in m.get_tasks("base")}
        self.split = m.get_tasks_split()
        self.split["base"] = list(self.tasks)
        self.train = self.split[split]

    def trace_from(self, d):
        t = self.Trace(d["task_id"], d["question"], d["faults"], stopped=d["stopped"],
                       tokens=d["tokens"], reward=d["reward"])
        t.steps = [self.Step(**s) for s in d["steps"]]
        t.opening = d.get("opening")
        return t

    def gold_text(self, task):
        acts = "; ".join(f"{a.name}({json.dumps(a.arguments)})"
                         for a in task.evaluation_criteria.actions or [])
        info = task.evaluation_criteria.communicate_info or []
        return (f"The expected outcome was reached by these actions: {acts}" +
                (f"; the agent also had to tell the user: {info}" if info else ""))


# ---- shared stages -----------------------------------------------------------
def stage_collect(llm, D, args):
    path = os.path.join(args.out, "traces.json")
    data = _load(path) or []
    done = {(d["task_id"], d["variant"], d["trial"]) for d in data}
    jobs = []
    for tid in D.train:
        for trial in range(args.clean_trials):
            jobs.append((tid, None, trial))
        for f in D.faults:
            jobs.append((tid, f, 0))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]

    def one(j):
        tid, v, trial = j
        f = [v] if v else []
        tr = D.m.run_agent(llm, D.tasks[tid], D.m.components_with(f), f)
        d = json.loads(json.dumps(tr, default=lambda o: o.__dict__))
        d.update(variant=v, trial=trial, uid=f"{tid}|{v}|{trial}")
        return d

    for s in range(0, len(jobs), 4 * args.workers):
        data.extend(_pmap(one, jobs[s:s + 4 * args.workers], args.workers))
        _save(path, data)
    return data


def stage_attribute(llm, D, args, data):
    path = os.path.join(args.out, "attributions.json")
    attrs = _load(path) or {}
    todo = [d for d in data if not d["reward"] and d["uid"] not in attrs]

    def one(d):
        tr = D.trace_from(d)
        a = attribute(llm, tr, D.gold_text(D.tasks[d["task_id"]]),
                      D.m.components_with(tr.faults), docs=D.docs)
        a["true_fault"] = d["variant"]
        return d["uid"], a

    for uid, a in _pmap(one, todo, args.workers):
        attrs[uid] = a
    _save(path, attrs)
    return attrs


def stage_taxonomy(llm, args, attrs):
    path = os.path.join(args.out, "taxonomy.json")
    tax = _load(path)
    if tax is None:
        ids = sorted(attrs)
        cats = induce_taxonomy(llm, [attrs[i]["root_cause"] for i in ids])
        labs = _pmap(lambda i: label(llm, attrs[i]["root_cause"], cats), ids, args.workers)
        tax = {"categories": cats, "labels": dict(zip(ids, labs))}
        _save(path, tax)
    return tax


def problem(D, attrs, tax):
    cats = tax["categories"] + [{"name": "other", "definition": "unclassified"}]
    I, J = len(cats), len(D.comp_ids)
    counts = np.zeros((I, J), dtype=int)
    members = [[] for _ in range(I)]
    for uid, a in attrs.items():
        i = tax["labels"][uid]
        members[i].append(uid)
        counts[i, D.comp_ids.index(a["component"])] += 1
    f = np.array([len(m) for m in members], dtype=float)
    return cats, counts, members, f / f.sum()


def bank_cells(D, args, counts, members):
    """Every attributed cell, plus an equal number of random unattributed cells of
    populated categories (at most --max_cells in total)."""
    rng = random.Random(args.seed + 3)
    att = [tuple(map(int, c)) for c in zip(*np.nonzero(counts))]
    att.sort(key=lambda c: -counts[c])
    att = att[:args.max_cells * 2 // 3]
    pop = [i for i in range(len(members)) if len(members[i]) >= 2]
    rest = [(i, j) for i in pop for j in range(len(D.comp_ids)) if counts[i, j] == 0]
    rng.shuffle(rest)
    return att + rest[:max(0, min(len(att), args.max_cells - len(att)))]


def stage_patches(llm, D, args, attrs, tax):
    path = os.path.join(args.out, "patches.json")
    cats, counts, members, f = problem(D, attrs, tax)
    have = _load(path) or {}
    cells = [c for c in bank_cells(D, args, counts, members) if f"{c[0]},{c[1]}" not in have]

    def one(c):
        i, j = c
        ex = [attrs[u]["root_cause"] for u in members[i]]
        return f"{i},{j}", make_patches(llm, D.comp_ids[j], D.m.components_with(), cats[i],
                                        ex, 3, docs=D.docs)

    for key, v in _pmap(one, cells, args.workers):
        have[key] = v
    _save(path, have)
    return have


def stage_bank(llm, D, args, data, attrs, tax, patches):
    path = os.path.join(args.out, "bank.json")
    bank = _load(path) or {"cells": {}, "null": {}, "reg": {}}
    cats, counts, members, f = problem(D, attrs, tax)
    by_uid = {d["uid"]: d for d in data}
    rng = random.Random(args.seed + 5)
    meter = llm.meter

    def replay(uid, patch_cid=None, patch_text=None, single=False):
        d = by_uid[uid]
        tr = D.trace_from(d)
        a = attrs[uid]
        step = max(0, min(a["step"], len(tr.steps) - 1))
        p = {patch_cid: patch_text} if patch_cid else {}
        comps = D.m.components_with(tr.faults, p)
        task = D.tasks[d["task_id"]]
        meter(reset=True)
        if not single:
            new = D.m.run_agent(llm, task, comps, tr.faults, prefix=tr, start_step=step)
            return new.reward, meter(reset=True)
        one = D.m.run_agent(llm, task, comps, tr.faults, prefix=tr, start_step=step,
                            max_new_steps=1)
        z = 0
        if len(one.steps) > step:
            z = judge_step(llm, tr, step, a["root_cause"], one.steps[step].assistant)
        return int(z), meter(reset=True)

    jobs = []
    cells = bank_cells(D, args, counts, members)
    samples = {}
    for i in sorted({c[0] for c in cells}):
        samples[i] = [rng.choice(members[i]) for _ in range(args.n_rep)]
        for r, uid in enumerate(samples[i]):
            if f"{i}|{r}" not in bank["null"]:
                jobs.append(("null", i, None, None, r, uid))
    ok_runs = [d for d in data if d["reward"] and d["variant"] is None]
    # --reg_keys: top up regression runs only for these patches (no other jobs)
    reg_keys = set(json.load(open(args.reg_keys))) if args.reg_keys else None
    if reg_keys is not None:
        jobs = []
    for i, j in cells:
        for k in range(3):
            key = f"{i},{j},{k}"
            have = bank["cells"].get(key, {"y": [], "z": []})
            if reg_keys is None:
                for r in range(len(have["y"]), args.n_rep):
                    jobs.append(("pair", i, j, k, r, samples[i][r]))
            elif key not in reg_keys:
                continue
            for r in range(len(bank["reg"].get(key, [])), args.n_reg):
                jobs.append(("reg", i, j, k, r, rng.choice(ok_runs)["uid"]))

    def one(job):
        kind, i, j, k, r, uid = job
        if kind == "null":
            y, t = replay(uid)
            return job, {"y": y, "tok": t}
        cid, text = D.comp_ids[j], patches[f"{i},{j}"][k]
        if kind == "pair":
            y, tf = replay(uid, cid, text)
            z, ts = replay(uid, cid, text, single=True)
            return job, {"y": y, "z": z, "tok_full": tf, "tok_single": ts, "trace": uid}
        d = by_uid[uid]
        meter(reset=True)
        tr = D.m.run_agent(llm, D.tasks[d["task_id"]], D.m.components_with([], {cid: text}))
        return job, {"y": tr.reward, "tok": meter(reset=True), "task": d["task_id"]}

    for s in range(0, len(jobs), 8 * args.workers):
        for job, res in _pmap(one, jobs[s:s + 8 * args.workers], args.workers):
            kind, i, j, k, r, uid = job
            if kind == "null":
                bank["null"][f"{i}|{r}"] = dict(res, cat=i, trace=uid)
            elif kind == "pair":
                c = bank["cells"].setdefault(f"{i},{j},{k}", {"y": [], "z": [], "tok_full": [],
                                                               "tok_single": [], "trace": []})
                for key in ("y", "z", "tok_full", "tok_single", "trace"):
                    c[key].append(res[key])
            else:
                bank["reg"].setdefault(f"{i},{j},{k}", []).append(res)
        _save(path, bank)
    return bank


# ---- evaluation by resampling ------------------------------------------------
def fisher_greater(a, n, b, m):
    from scipy.stats import fisher_exact
    return fisher_exact([[a, n - a], [b, m - b]], alternative="greater")[1]


def ground_truth(bank, I, J, K=3, delta=0.2):
    inb = np.zeros((I, J), bool)
    Q = np.zeros((I, J, K))
    E = np.zeros((I, J), bool)
    b = np.zeros(I)
    nulls = {}
    for v in bank["null"].values():
        nulls.setdefault(v["cat"], []).append(v["y"])
    for i, ys in nulls.items():
        b[i] = np.mean(ys)
    for key, c in bank["cells"].items():
        i, j, k = map(int, key.split(","))
        inb[i, j] = True
        ys = c["y"]
        nb = nulls.get(i, [0])
        Q[i, j, k] = max(0.0, (np.mean(ys) - b[i]) / max(1e-9, 1 - b[i]))
        if np.mean(ys) - np.mean(nb) >= delta and \
                fisher_greater(sum(ys), len(ys), sum(nb), len(nb)) < 0.05:
            E[i, j] = True
    return inb, E, Q, b


def net_gain(acc, patch, E, Q, f, c_fp):
    gain = 0.0
    for i in range(len(f)):
        hit = np.flatnonzero(acc[i] & E[i])
        gain += f[i] * (1 - np.prod([1 - Q[i, j, patch[i, j]] for j in hit]))
    return gain - c_fp * (acc & ~E).sum()


class BankWorld:
    """Draws recorded outcomes from the bank. By default with replacement; with
    BANK_NOREPLACE=1 each list is drawn without replacement (a fresh random order
    once it is exhausted), so a policy never sees the same record twice until it
    has seen them all."""

    def __init__(self, bank, rng):
        self.c, self.reg, self.rng = bank["cells"], bank["reg"], rng
        self.null = list(bank["null"].values())
        self.replace = os.environ.get("BANK_NOREPLACE", "0") != "1"
        self._order = {}

    def _draw(self, key, n):
        if self.replace:
            return int(self.rng.integers(n))
        q = self._order.get(key)
        if not q:
            q = self._order[key] = list(self.rng.permutation(n))
        return int(q.pop())

    def intervene(self, i, j, k, fid):
        c = self.c[f"{i},{j},{k}"]
        r = self._draw(("cell", i, j, k), len(c["y"]))
        return int(c["y"][r] if fid == FULL else c["z"][r])

    def control(self, i):
        ys = [v["y"] for v in self.null if v["cat"] == i] or [0]
        return int(ys[self._draw(("null", i), len(ys))])

    def regress(self, i, j, k):
        runs = self.reg.get(f"{i},{j},{k}") or [{"y": 1}]
        return int(runs[self._draw(("reg", i, j, k), len(runs))]["y"])


class HarnessFix:
    """Do-then-verify with a regression constraint (after HarnessFix): hypotheses in
    order of attribution count; for each, try patches in turn; a patch is
    accepted when it repairs at least `need` of `n_val` failure replays and
    breaks at most `max_reg` of `n_reg` previously successful tasks."""

    name = "HarnessFix"

    def __init__(self, counts, n_val=3, need=2, n_reg=2, max_reg=0, patches=3):
        self.counts, self.n_val, self.need = counts, n_val, need
        self.n_reg, self.max_reg, self.patches = n_reg, max_reg, patches

    def run(self, world, inb, budget, cost_full, cost_reg):
        order = np.argsort(-self.counts, axis=None, kind="stable")
        I, J = self.counts.shape
        acc = np.zeros((I, J), bool)
        patch = np.zeros((I, J), int)
        spent = 0.0
        for o in order:
            i, j = np.unravel_index(o, (I, J))
            if self.counts[i, j] == 0 or not inb[i, j]:
                continue
            for k in range(self.patches):
                need_cost = self.n_val * cost_full
                if spent + need_cost > budget:
                    return acc, patch, spent
                s = sum(world.intervene(i, j, k, FULL) for _ in range(self.n_val))
                spent += need_cost
                if s < self.need:
                    continue
                if spent + self.n_reg * cost_reg > budget:
                    return acc, patch, spent
                broken = sum(1 - world.regress(i, j, k) for _ in range(self.n_reg))
                spent += self.n_reg * cost_reg
                if broken <= self.max_reg:
                    acc[i, j], patch[i, j] = True, k
                    break
        return acc, patch, spent


class HarnessFixGate:
    """Closer to HarnessFix's own protocol: apply the whole bundle of attributed
    repairs (LLM-only's set, one patch index per attempt), run it on fresh
    validation episodes and promote it if it solves at least `min_net` more
    tasks than before and breaks at most `max_reg`. A validation episode is a
    failure of category i with probability F f_i (solved if the bundle's patch
    for i repairs a replay, else with the category's null rate) or a previously
    successful task (broken if a regression run of a bundle patch fails).
    Up to three attempts, one per patch index, each with a third of the budget."""

    name = "HarnessFix-gate"

    def __init__(self, counts, min_net=1, max_reg=2, attempts=3):
        from carve.policies import LLMOnly
        self.cells = LLMOnly(counts).decide(None, None)[0]
        self.min_net, self.max_reg, self.attempts = min_net, max_reg, attempts

    def run(self, world, inb, f, w, budget, cost_ep, rng):
        acc0 = self.cells & inb
        I, J = acc0.shape
        cells = list(zip(*np.nonzero(acc0)))
        p_fail = 1.0 / (1.0 + w)
        n_val = int(budget / self.attempts / cost_ep)
        spent = 0.0
        for k in range(self.attempts):
            if n_val < 1 or not cells:
                break
            net = broken = 0
            for _ in range(n_val):
                if rng.random() < p_fail:
                    i = rng.choice(len(f), p=f)
                    js = [j for ii, j in cells if ii == i]
                    y = world.intervene(i, js[0], k, FULL) if js else world.control(i)
                    net += y
                else:
                    ii, jj = cells[rng.integers(len(cells))]
                    b = 1 - world.regress(ii, jj, k)
                    broken += b
                    net -= b
                spent += cost_ep
            if net >= self.min_net and broken <= self.max_reg:
                return acc0, np.full((I, J), k), spent
        return np.zeros((I, J), bool), np.zeros((I, J), int), spent


def regression_value(n_fail, n, r0, w, a0=0.5, b0=10.0, grid=np.linspace(0.0005, 0.9995, 400)):
    """Posterior expected regression cost of a patch, in failure-mass units.

    rho is the patch's excess failure probability on previously successful tasks
    (prior Beta(a0, b0)); a regression run fails with prob r0 + (1 - r0) rho,
    where r0 is the natural re-run failure rate; w = successes / failures
    converts a rate on successful tasks into failure-mass units."""
    pf = r0 + (1 - r0) * grid
    lw = (a0 - 1) * np.log(grid) + (b0 - 1) * np.log1p(-grid) + \
        n_fail * np.log(pf) + (n - n_fail) * np.log1p(-pf)
    pw = np.exp(lw - lw.max())
    return float((pw * grid).sum() / pw.sum()) * w


def run_method(method, bank, counts, f, inb, costs, b_hat, seed, budget, c_fp=0.02,
               r_hat=0.7, cal=None, r0=0.1, w=2.8):
    rng = np.random.default_rng(seed)
    world = BankWorld(bank, rng)
    I, J = counts.shape
    if method == "HarnessFix":
        return HarnessFix(counts).run(world, inb, budget, 1.0, costs[2])
    if method == "HarnessFix-gate":
        return HarnessFixGate(counts).run(world, inb, f, w, budget, costs[2], rng)
    if method.startswith("Net"):
        # episode-priced net-effect selection (agent_exp/netsel.py)
        from agent_exp.netsel import run_netsel
        return run_netsel(bank, counts, f, inb, costs, b_hat, seed, budget, r0, w,
                          use_reg=method != "Net-noreg", use_null=method != "Net-nonull",
                          use_prior=method != "Net-flat",
                          n_ctrl=2 if method == "Net+ctrl" else 0,
                          c_fp=0.0 if method == "Net-c0" else c_fp)
    params = ObsParams(b=b_hat, lam=0.8, sens=0.85, fpr=0.15)
    if method == "CARVE-calibrated":
        params.sens, params.fpr = cal["sens"], cal["fpr"]
    if method == "CARVE-robust-check":
        params.sens, params.fpr, params.fooled_fpr = cal["sens"], cal["fpr_good"], cal["fpr_bad"]
    plus = method.startswith("CARVE+")
    ctrl = method in ("CARVE+", "CARVE+ctrl")
    gate = method in ("CARVE+", "CARVE+gate")
    # CARVE-R: evidence rule + Bayesian regression gate (borrowed from HarnessFix:
    # accept only verified patches, and only if they do not regress)
    evid = False
    pw = {"CARVE-R": 0.5, "CARVE+evidence": 0.5, "CARVE+flat": 0.0}.get(method, 1.0)
    bgate = method in ("CARVE-R", "CARVE+bgate")
    if ctrl:
        post = CategoryBPosterior(edge_prior(counts, r_hat), 3, params)
    else:
        post = GraphPosterior(edge_prior(counts, r_hat), 3, params)
    c = costs[:2]
    rho = 1.5 / J                                   # base edge density of the prior
    post.base_logit = float(np.log(rho / (1 - rho)))
    if plus or method == "CARVE-R":
        method = "CARVE"
    pol = {"CARVE": lambda: CARVE(costs=c, c_fp=c_fp, prior_weight=pw),
           "CARVE-calibrated": lambda: CARVE(costs=c, c_fp=c_fp),
           "CARVE-robust-check": lambda: CARVE(costs=c, c_fp=c_fp),
           "CARVE-full-only": lambda: CARVE(costs=c, c_fp=c_fp, fidelities=(FULL,)),
           "Uncertainty": lambda: UncertaintySampling(c_fp=c_fp),
           "Uncertainty-MF": lambda: UncertaintyMF(c_fp=c_fp),
           "LLM-only": lambda: LLMOnly(counts),
           "Replay-each": lambda: ReplayEach(counts)}[method]()
    pol.blocked = ~inb
    spent = 0.0
    n_ctrl, n_reg = 2, 2
    reserve = min(0.3 * budget, 3 * n_reg * costs[2]) if (gate or bgate) else 0.0
    while True:
        act = pol.select(post, f)
        if act is None:
            break
        i, j, k, fid = act
        if ctrl and post.n_ctrl[i] < n_ctrl:
            # control replays of the category before trusting its outcomes
            if spent + 1.0 > budget - reserve:
                break
            post.control(i, world.control(i))
            pol._cache = None
            spent += 1.0
            continue
        if spent + c[fid] > budget - reserve:
            break
        o = world.intervene(i, j, k, fid)
        post.update(i, j, k, fid, o)
        if hasattr(pol, "record"):
            pol.record(i, j, k, fid, o)
        spent += c[fid]
    acc, patch = pol.decide(post, f)
    acc &= inb
    if evid:
        # the data, not only the prior, must favour the edge: at least one full
        # replay and a positive log-likelihood ratio from the observations
        llr = (post.log_ev1 - post.log_ev0).sum(-1)
        nf = post.n_obs[..., FULL].sum(-1)
        acc &= (nf >= 1) & (llr > 0)
    if bgate:
        # sequential Bayesian regression check: run up to 3 regression episodes
        # per accepted cell and keep it only while its expected repair exceeds
        # its expected regression cost
        pe, q = post.p_edge(), post.q_mean()
        val = pe * f[:, None] * q.max(-1) - (1 - pe) * c_fp
        for i, j in sorted(zip(*np.nonzero(acc)), key=lambda c: -val[c]):
            n = nfail = 0
            while True:
                cost = regression_value(nfail, n, r0, w)
                if val[i, j] - cost <= 0:
                    acc[i, j] = False
                    break
                if n >= 3:
                    break
                if spent + costs[2] > budget:
                    acc[i, j] = n > 0
                    break
                nfail += 1 - world.regress(i, j, patch[i, j])
                n += 1
                spent += costs[2]
    if gate:
        # regression gate: each accepted patch must not break previously
        # successful tasks; cells that cannot be gated within budget are dropped
        val = post.p_edge() * f[:, None]
        for i, j in sorted(zip(*np.nonzero(acc)), key=lambda c: -val[c]):
            if spent + n_reg * costs[2] > budget:
                acc[i, j] = False
                continue
            broken = sum(1 - world.regress(i, j, patch[i, j]) for _ in range(n_reg))
            spent += n_reg * costs[2]
            if broken:
                acc[i, j] = False
    return acc, patch, spent


METHODS = ["LLM-only", "Replay-each", "HarnessFix", "Uncertainty", "Uncertainty-MF",
           "CARVE-full-only", "CARVE", "CARVE-calibrated", "CARVE-robust-check"]
# variants tried on the retail development bank only (none beat plain CARVE there,
# so none is carried to the held-out airline bank)
DEV_METHODS = ["CARVE+ctrl", "CARVE+gate", "CARVE+", "CARVE+evidence", "CARVE+flat",
               "CARVE+bgate", "CARVE-R"]


def rerun_stats(data):
    """Natural re-run failure rate r0 = P(fail | succeeded in the other clean
    trial), and w = successes / failures in the training runs."""
    by = {}
    for d in data:
        if d["variant"] is None:
            by.setdefault(d["task_id"], {})[d["trial"]] = d["reward"]
    pairs = [(v[0], v[1]) for v in by.values() if 0 in v and 1 in v]
    after = [b for a, b in pairs if a] + [a for a, b in pairs if b]
    r0 = 1 - float(np.mean(after)) if after else 0.1
    ok = sum(d["reward"] for d in data)
    return r0, ok / max(len(data) - ok, 1)


def reg_excess(bank, I, J, K, r0, w, alpha=0.05):
    """Measured regression cost of each patch in failure-mass units. As for edges,
    a patch counts as regressing only when its failure rate on previously
    successful tasks exceeds the natural re-run rate r0 significantly (one-sided
    binomial test); clipping noisy estimates at zero would otherwise charge every
    patch for sampling noise."""
    from scipy.stats import binom
    R = np.zeros((I, J, K))
    for key, runs in bank["reg"].items():
        i, j, k = map(int, key.split(","))
        n = len(runs)
        nf = n - sum(r["y"] for r in runs)
        if binom.sf(nf - 1, n, r0) < alpha:
            R[i, j, k] = w * max(0.0, (nf / n - r0) / (1 - r0))
    return R


def net_gain_reg(acc, patch, E, Q, f, R):
    """Repaired failure mass minus measured regressions of every accepted patch
    (true or false edge)."""
    g = net_gain(acc, patch, E, Q, f, 0.0)
    return g - sum(R[i, j, patch[i, j]] for i, j in zip(*np.nonzero(acc)))


def stage_evaluate(D, args, attrs, tax, bank, data=None):
    cats, counts, members, f = problem(D, attrs, tax)
    I, J = counts.shape
    inb, E, Q, b = ground_truth(bank, I, J, delta=args.delta)
    cells = list(bank["cells"].values())
    tf = np.mean([t for c in cells for t in c["tok_full"]])
    ts = np.mean([t for c in cells for t in c["tok_single"]])
    tr = np.mean([r["tok"] for v in bank["reg"].values() for r in v]) if bank["reg"] else tf
    costs = (1.0, ts / tf, tr / tf)
    ys = np.array([y for c in cells for y in c["y"]])
    zs = np.array([z for c in cells for z in c["z"]])
    qbar = {k: np.mean(c["y"]) for k, c in bank["cells"].items()}
    bad = np.array([qbar[k] < 0.3 for k, c in bank["cells"].items() for _ in c["y"]])
    cal = {"sens": float(zs[ys == 1].mean()), "fpr": float(zs[ys == 0].mean()),
           "fpr_bad": float(zs[(ys == 0) & bad].mean()) if ((ys == 0) & bad).any() else None,
           "fpr_good": float(zs[(ys == 0) & ~bad].mean()) if ((ys == 0) & ~bad).any() else None}
    cal["fpr_bad"] = cal["fpr_bad"] if cal["fpr_bad"] is not None else cal["fpr"]
    cal["fpr_good"] = cal["fpr_good"] if cal["fpr_good"] is not None else cal["fpr"]
    b_hat = float(np.mean([v["y"] for v in bank["null"].values()]))
    faulted = [a for a in attrs.values() if a["true_fault"]]
    attr_acc = float(np.mean([a["component"] == D.faults[a["true_fault"]][0] for a in faulted]))
    oracle = net_gain(E, Q.argmax(-1), E, Q, f, args.c_fp)
    r0, w = rerun_stats(data) if data else (0.1, 2.8)
    R = reg_excess(bank, I, J, 3, r0, w)
    # regression-aware oracle: true edges with the patch maximising f*q - R,
    # keeping only cells whose contribution is positive
    pbest = (f[:, None, None] * Q - R).argmax(-1)
    keep = E & (np.take_along_axis(f[:, None, None] * Q - R, pbest[..., None], -1)[..., 0] > 0)
    oracle_r = net_gain_reg(keep, pbest, E, Q, f, R)
    out = {"r0": r0, "w": w, "oracle_gain_reg": oracle_r, "costs": costs, "calibration": cal, "b_hat": b_hat, "n_true_edges": int(E.sum()),
           "n_bank_cells": int(inb.sum()), "attr_acc_injected": attr_acc,
           "n_failures": len(attrs), "n_natural": len(attrs) - len(faulted),
           "oracle_gain": oracle, "true_edges": [[cats[i]["name"], D.comp_ids[j]]
                                                 for i, j in zip(*np.nonzero(E))],
           "results": {}}
    for m in args.methods or METHODS:
        for B in args.budgets:
            rows = []
            for s in range(args.n_seeds):
                acc, patch, spent = run_method(m, bank, counts, f, inb, costs, b_hat, s, B,
                                               c_fp=args.c_fp, cal=cal, r0=r0, w=w)
                g = net_gain(acc, patch, E, Q, f, args.c_fp)
                gr = net_gain_reg(acc, patch, E, Q, f, R)
                tp = int((acc & E).sum())
                rows.append([g / oracle if oracle > 0 else 0.0,
                             tp / max(int(acc.sum()), 1), tp / max(int(E.sum()), 1), spent,
                             gr / oracle_r if oracle_r > 0 else 0.0,
                             (gr - args.c_fp * int((acc & ~E).sum())) / oracle_r
                             if oracle_r > 0 else 0.0])
            r = np.array(rows)
            out["results"][f"{m}|{B}"] = {"gain": r[:, 0].mean(), "gain_se": r[:, 0].std() /
                                          np.sqrt(len(r)), "precision": r[:, 1].mean(),
                                          "recall": r[:, 2].mean(), "spent": r[:, 3].mean(),
                                          "gain_reg": r[:, 4].mean(),
                                          "gain_reg_se": r[:, 4].std() / np.sqrt(len(r)),
                                          # measured regressions plus a fixed cost per wrong change
                                          "gain_both": r[:, 5].mean(),
                                          "gain_both_se": r[:, 5].std() / np.sqrt(len(r))}
            print(m, B, {k: round(v, 3) for k, v in out["results"][f"{m}|{B}"].items()})
    _save(os.path.join(args.out, f"evaluation{args.eval_tag}.json"), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    ap.add_argument("--task_split", default="train")
    ap.add_argument("--stage", default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--clean_trials", type=int, default=2)
    ap.add_argument("--max_cells", type=int, default=36)
    ap.add_argument("--n_rep", type=int, default=10)
    ap.add_argument("--n_reg", type=int, default=4)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--c_fp", type=float, default=0.02)
    ap.add_argument("--n_seeds", type=int, default=200)
    ap.add_argument("--budgets", type=float, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--eval_tag", default="")
    ap.add_argument("--reg_keys", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from agent_exp.llm import LLM

    llm, D = LLM(), Domain(args.domain, args.task_split)
    data = stage_collect(llm, D, args)
    if args.stage == "collect":
        return
    attrs = stage_attribute(llm, D, args, data)
    tax = stage_taxonomy(llm, args, attrs)
    if args.stage == "taxonomy":
        return
    patches = stage_patches(llm, D, args, attrs, tax)
    bank = stage_bank(llm, D, args, data, attrs, tax, patches)
    if args.stage == "bank":
        return
    stage_evaluate(D, args, attrs, tax, bank, data)
    print(json.dumps({k: dict(llm.io)[k] for k in llm.io}))


if __name__ == "__main__":
    main()
