"""End-to-end real-LLM experiment on ShopDesk with injected faults.

Stages (each cached under --out so a run can be resumed):
  collect    run the fault-injected agent population on training tasks
  attribute  LLM root-cause attribution of every failure (step + component)
  taxonomy   induce failure categories from the root causes and label failures
  verify     spend an intervention budget with a selection policy (CARVE or a
             baseline), using real replays from the attributed step
  evaluate   apply the accepted patches and measure held-out success

Ground truth: each task runs under exactly one injected fault (or none), so the
true component of every failure is known; a (category, component) edge is true
when at least `edge_share` of the category's failures came from that
component's fault.

Usage:
  python -m agent_exp.pipeline --out runs/r1 --stage all --method CARVE --budget 60
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from agent_exp.agent import COMPONENT_DOCS, FAULTS, Step, Trace, components_with, run_agent
from agent_exp.analyst import attribute, induce_taxonomy, judge_step, label, make_patches
from agent_exp.env import ShopDB, check, make_tasks
from carve.model import FULL, SINGLE, GraphPosterior, ObsParams
from carve.policies import CARVE, LLMOnly, ReplayEach, UncertaintyMF, UncertaintySampling
from sim.testbed import edge_prior

COMP_IDS = list(COMPONENT_DOCS)
VARIANTS = list(FAULTS) + [None]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _save(path, obj):
    tmp = f"{path}.{os.getpid()}.tmp"           # atomic: methods run concurrently
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.__dict__)
    os.replace(tmp, path)


def _trace_from(d) -> Trace:
    t = Trace(d["task_id"], d["question"], d["faults"], answer=d["answer"],
              stopped=d["stopped"], tokens=d["tokens"])
    t.steps = [Step(**s) for s in d["steps"]]
    return t


def _pmap(fn, items, workers):
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def stage_collect(llm, args, db, tasks):
    path = os.path.join(args.out, "traces.json")
    data = _load(path) or []
    rng = random.Random(args.seed)
    variants = [rng.choice(VARIANTS) for _ in tasks]     # fixed before any model call

    def one(n):
        task, v = tasks[n], variants[n]
        faults = [v] if v else []
        tr = run_agent(llm, db, task, components_with(faults), faults)
        return {"trace": tr, "gold": task.answer, "ok": check(tr.answer, task.answer)}

    w = getattr(args, "workers", 1)
    while len(data) < len(tasks):
        idx = range(len(data), min(len(tasks), len(data) + max(4 * w, 1)))
        data.extend(_pmap(one, list(idx), w))
        _save(path, data)
        data = _load(path)
    return data


def stage_attribute(llm, args, data):
    path = os.path.join(args.out, "attributions.json")
    attrs = _load(path) or {}
    todo = [d for d in data if not d["ok"] and
            (d["trace"]["task_id"] if isinstance(d["trace"], dict)
             else d["trace"].task_id) not in attrs]

    def one(d):
        tr = _trace_from(d["trace"]) if isinstance(d["trace"], dict) else d["trace"]
        a = attribute(llm, tr, f"The correct answer was: {d['gold']}", components_with(tr.faults))
        a["true_fault"] = tr.faults[0] if tr.faults else None
        return tr.task_id, a

    for tid, a in _pmap(one, todo, getattr(args, "workers", 1)):
        attrs[tid] = a
    _save(path, attrs)
    return attrs


def stage_taxonomy(llm, args, attrs):
    path = os.path.join(args.out, "taxonomy.json")
    tax = _load(path)
    if tax is None:
        ids = sorted(attrs)
        cats = induce_taxonomy(llm, [attrs[i]["root_cause"] for i in ids])
        labs = _pmap(lambda i: label(llm, attrs[i]["root_cause"], cats), ids,
                     getattr(args, "workers", 1))
        labels = dict(zip(ids, labs))
        tax = {"categories": cats, "labels": labels}
        _save(path, tax)
    return tax


def build_problem(attrs, tax, edge_share):
    cats = tax["categories"] + [{"name": "other", "definition": "unclassified"}]
    I, J = len(cats), len(COMP_IDS)
    counts = np.zeros((I, J), dtype=int)
    prov = np.zeros((I, J), dtype=int)
    members = [[] for _ in range(I)]
    for tid, a in attrs.items():
        i = tax["labels"][tid]
        members[i].append(tid)
        counts[i, COMP_IDS.index(a["component"])] += 1
        if a["true_fault"]:
            prov[i, COMP_IDS.index(FAULTS[a["true_fault"]][0])] += 1
    size = np.maximum(np.array([len(m) for m in members]), 1)[:, None]
    truth = prov / size >= edge_share
    f = np.array([len(m) for m in members], dtype=float)
    return cats, counts, truth, members, f / f.sum()


class RealWorld:
    """Interventions on recorded failures with real model calls."""

    def __init__(self, llm, db, data, attrs, cats, members, rng, k=3, cache=None):
        self.llm, self.db, self.attrs, self.cats = llm, db, attrs, cats
        # candidate patches are cached on disk so every method and the
        # calibration stage test the same patch texts
        self.cache = cache
        self.shared = (_load(cache) or {}) if cache else {}
        self.members, self.rng, self.k = members, rng, k
        self.by_id = {(d["trace"]["task_id"] if isinstance(d["trace"], dict)
                       else d["trace"].task_id): d for d in data}
        self.patches = {}                       # (i, j) -> list of patch texts
        self.tokens = {"full": [], "single": [], "patch": 0}
        self.log = []

    def get_patches(self, i, j):
        key = f"{i},{j}"
        if (i, j) not in self.patches and key in self.shared:
            self.patches[(i, j)] = self.shared[key]
        if (i, j) not in self.patches:
            before = sum(self.llm.tokens.values())
            ex = [self.attrs[t]["root_cause"] for t in self.members[i]]
            self.patches[(i, j)] = make_patches(self.llm, COMP_IDS[j],
                                                components_with(), self.cats[i], ex, self.k)
            self.tokens["patch"] += sum(self.llm.tokens.values()) - before
            if self.cache:
                self.shared = _load(self.cache) or {}
                self.shared[key] = self.patches[(i, j)]
                _save(self.cache, self.shared)
        return self.patches[(i, j)]

    def intervene(self, i, j, k, fid):
        tid = self.rng.choice(self.members[i])
        d = self.by_id[tid]
        tr = _trace_from(d["trace"]) if isinstance(d["trace"], dict) else d["trace"]
        a = self.attrs[tid]
        step = max(0, min(a["step"], len(tr.steps) - 1))
        comps = components_with(tr.faults, {COMP_IDS[j]: self.get_patches(i, j)[k]})
        task = type("T", (), {"tid": tid, "question": tr.question})()
        before = sum(self.llm.tokens.values())
        if fid == FULL:
            new = run_agent(self.llm, self.db, task, comps, tr.faults, prefix=tr, start_step=step)
            o = int(check(new.answer, d["gold"]))
            self.tokens["full"].append(sum(self.llm.tokens.values()) - before)
        else:
            new = run_agent(self.llm, self.db, task, comps, tr.faults, prefix=tr,
                            start_step=step, max_new_steps=1)
            o = judge_step(self.llm, tr, step, a["root_cause"], new.steps[step].assistant)
            self.tokens["single"].append(sum(self.llm.tokens.values()) - before)
        self.log.append({"cell": [i, j], "patch": k, "fidelity": fid, "trace": tid, "outcome": o})
        return o

    def unit_costs(self, prior=(1.0, 0.1)):
        """(1, c_S/c_F, mean full-replay tokens) from this run's measurements,
        falling back to `prior` for a fidelity not yet observed."""
        full = np.mean(self.tokens["full"]) if self.tokens["full"] else None
        single = np.mean(self.tokens["single"]) if self.tokens["single"] else None
        if full is None or single is None:
            return 1.0, float(prior[1]), float(full or 0.0)
        return 1.0, float(single / full), float(full)


def stage_verify(llm, args, db, data, attrs, tax):
    cats, counts, truth, members, f = build_problem(attrs, tax, args.edge_share)
    rng = random.Random(args.seed + 1)
    world = RealWorld(llm, db, data, attrs, cats, members, rng,
                      cache=os.path.join(args.out, "patches.json"))
    n_fail = len(attrs)
    r_hat = np.mean([a["component"] == FAULTS[a["true_fault"]][0]
                     for a in list(attrs.values())[:args.audit] if a["true_fault"]] or [0.5])
    params = ObsParams(b=0.05, lam=0.8, sens=0.85, fpr=0.15)
    if args.method == "CARVE-calibrated":
        cal = _load(os.path.join(args.out, "calibration.json"))
        params.sens, params.fpr = cal["sens"], cal["p_pass_given_fail"][0]
    if args.method == "CARVE-robust-check":
        # judge parameters measured by the calibration stage on separate draws
        cal = _load(os.path.join(args.out, "calibration.json"))
        params.sens, params.fpr = cal["sens"], cal["fpr_good"]
        params.fooled_fpr = cal["fpr_bad"]
    post = GraphPosterior(edge_prior(counts, r_hat), world.k, params)
    m = args.method
    # cost ratio c_S/c_F measured in the calibration stage (tokens per check vs
    # per full replay); refined online from this run's own measurements
    cal = _load(os.path.join(args.out, "calibration.json"))
    c0 = (1.0, cal["tok_single"] / cal["tok_full"]) if cal and "tok_full" in cal else (1.0, 0.1)
    pol = {"CARVE": lambda: CARVE(costs=c0),
           "CARVE-full-only": lambda: CARVE(costs=c0, fidelities=(FULL,)),
           "CARVE-calibrated": lambda: CARVE(costs=c0),
           "CARVE-robust-check": lambda: CARVE(costs=c0),
           "Uncertainty": UncertaintySampling,
           "Uncertainty-MF": UncertaintyMF,
           "LLM-only": lambda: LLMOnly(counts),
           "Replay-each": lambda: ReplayEach(counts)}[m]()
    spent = 0.0
    while True:
        act = pol.select(post, f)
        if act is None:
            break
        i, j, k, fid = act
        c = c0[fid]
        if spent + c > args.budget:
            break
        o = world.intervene(i, j, k, fid)
        post.update(i, j, k, fid, o)
        if hasattr(pol, "record"):
            pol.record(i, j, k, fid, o)
        spent += c
    acc, patch = pol.decide(post, f)
    tp = int((acc & truth).sum())
    res = {"method": m, "budget": args.budget, "spent_units": spent, "cost_ratio_used": c0[1],
           "unit_costs": world.unit_costs(c0), "tokens": dict(llm.tokens),
           "io": {k: list(v) for k, v in getattr(llm, "io", {}).items()},
           "precision": tp / max(int(acc.sum()), 1), "recall": tp / max(int(truth.sum()), 1),
           "accepted": [[COMP_IDS[j], cats[i]["name"], int(patch[i, j])]
                        for i, j in zip(*np.nonzero(acc))],
           "patch_texts": {f"{i},{j}": v for (i, j), v in world.patches.items()},
           "log": world.log, "n_failures": n_fail}
    # patches the evaluation stage will apply: best patch per accepted component
    apply = {}
    for i, j in zip(*np.nonzero(acc)):
        if (i, j) not in world.patches:          # e.g. LLM-only never replayed it
            world.get_patches(i, j)
        apply.setdefault(COMP_IDS[j], world.patches[(i, j)][patch[i, j]])
    res["apply"] = apply
    _save(os.path.join(args.out, f"verify_{m}_{int(args.budget)}.json"), res)
    return res


def stage_patches(llm, args, attrs, tax):
    """Pre-generate the candidate patches of every populated cell, so methods
    that run concurrently share one patch set without racing on the cache."""
    path = os.path.join(args.out, "patches.json")
    cats, counts, truth, members, f = build_problem(attrs, tax, args.edge_share)
    have = _load(path) or {}
    cells = [(i, j) for i in range(len(cats)) if members[i] for j in range(len(COMP_IDS))
             if f"{i},{j}" not in have]

    def one(c):
        i, j = c
        ex = [attrs[t]["root_cause"] for t in members[i]]
        return f"{i},{j}", make_patches(llm, COMP_IDS[j], components_with(), cats[i], ex, 3)

    for key, v in _pmap(one, cells, getattr(args, "workers", 1)):
        have[key] = v
    _save(path, have)
    return have


def stage_calibrate(llm, args, db, data, attrs, tax, n_cells=16, reps=3):
    """Judge calibration: for (cell, patch) pairs, run the single-step check and a
    full replay on the same failure trace and step, and tabulate the judge's
    pass rate against the replay outcome."""
    path = os.path.join(args.out, "calibration.json")
    cal = _load(path)
    if cal is not None:
        return cal
    cats, counts, truth, members, f = build_problem(attrs, tax, args.edge_share)
    rng = random.Random(args.seed + 7)
    world = RealWorld(llm, db, data, attrs, cats, members, rng,
                      cache=os.path.join(args.out, "patches.json"))
    # half the cells are the most-attributed ones, half are random populated
    # categories paired with a random component, so wrong edges are covered too
    order = [tuple(map(int, np.unravel_index(o, counts.shape)))
             for o in np.argsort(-counts, axis=None, kind="stable") if counts.flat[o] > 0]
    cells = order[:n_cells // 2]
    pop = [i for i in range(len(cats)) if members[i]]
    while len(cells) < n_cells and pop:
        c = (rng.choice(pop), rng.randrange(len(COMP_IDS)))
        if c not in cells:
            cells.append(c)
    jobs = []
    for i, j in cells:
        world.get_patches(i, j)
        for k in range(world.k):
            for _ in range(reps):
                jobs.append((i, j, k, rng.choice(members[i])))

    def one(job):
        i, j, k, tid = job
        d = world.by_id[tid]
        tr = _trace_from(d["trace"]) if isinstance(d["trace"], dict) else d["trace"]
        a = attrs[tid]
        step = max(0, min(a["step"], len(tr.steps) - 1))
        comps = components_with(tr.faults, {COMP_IDS[j]: world.patches[(i, j)][k]})
        task = type("T", (), {"tid": tid, "question": tr.question})()
        meter = getattr(llm, "meter", lambda reset=False: 0)
        meter(reset=True)
        new = run_agent(llm, db, task, comps, tr.faults, prefix=tr, start_step=step)
        y = int(check(new.answer, d["gold"]))
        tok_full = meter(reset=True)
        one_ = run_agent(llm, db, task, comps, tr.faults, prefix=tr, start_step=step,
                         max_new_steps=1)
        if len(one_.steps) <= step:     # episode could not take a step
            z = 0
        else:
            z = judge_step(llm, tr, step, a["root_cause"], one_.steps[step].assistant)
        tok_single = meter(reset=True)
        return {"cell": [i, j], "patch": k, "trace": tid, "y": y, "z": int(z),
                "true_edge": bool(truth[i, j]), "tok_full": tok_full,
                "tok_single": tok_single}

    rows = _pmap(one, jobs, getattr(args, "workers", 1))
    by_patch = {}
    for r in rows:
        by_patch.setdefault((r["cell"][0], r["cell"][1], r["patch"]), []).append(r["y"])
    succ = {key: float(np.mean(v)) for key, v in by_patch.items()}
    for r in rows:
        r["patch_full_success"] = succ[(r["cell"][0], r["cell"][1], r["patch"])]

    def rate(sel):
        sel = list(sel)
        return (float(np.mean([r["z"] for r in sel])) if sel else None, len(sel))

    bad = [r for r in rows if r["patch_full_success"] < 0.3]
    good = [r for r in rows if r["patch_full_success"] >= 0.3]
    cal = {
        "n": len(rows), "n_pairs": len(by_patch),
        "p_pass_given_fail": rate(r for r in rows if not r["y"]),
        "p_pass_given_success": rate(r for r in rows if r["y"]),
        "p_pass_bad_patch": rate(bad),
        "p_pass_given_fail_bad_patch": rate(r for r in bad if not r["y"]),
        "p_pass_given_fail_good_patch": rate(r for r in good if not r["y"]),
        "replay_success": float(np.mean([r["y"] for r in rows])),
        "tok_full": float(np.mean([r["tok_full"] for r in rows])),
        "tok_single": float(np.mean([r["tok_single"] for r in rows])),
        "patches": {f"{a},{b}": v for (a, b), v in world.patches.items()},
        "rows": rows,
    }
    cal["sens"] = cal["p_pass_given_success"][0] or 0.85
    cal["fpr_good"] = cal["p_pass_given_fail_good_patch"][0] or 0.15
    cal["fpr_bad"] = cal["p_pass_given_fail_bad_patch"][0] or cal["fpr_good"]
    _save(path, cal)
    return cal


def stage_evaluate(llm, args, db, res):
    test = make_tasks(db, args.n_test, seed=args.seed + 99)
    rng = random.Random(args.seed + 2)
    variants = [rng.choice(VARIANTS) for _ in test]

    def one(n):
        task, v = test[n], variants[n]
        faults = [v] if v else []
        n_ = run_agent(llm, db, task, components_with(faults, res["apply"]), faults)
        return int(check(n_.answer, task.answer))

    def base(n):
        task, v = test[n], variants[n]
        faults = [v] if v else []
        return int(check(run_agent(llm, db, task, components_with(faults), faults).answer,
                         task.answer))

    # the unpatched baseline is shared by every method: cache it
    bpath = os.path.join(args.out, f"eval_base_{args.n_test}.json")
    w = getattr(args, "workers", 1)
    b_ok = _load(bpath)
    if b_ok is None:
        b_ok = _pmap(base, range(len(test)), w)
        _save(bpath, b_ok)
    n_ok = _pmap(one, range(len(test)), w) if res["apply"] else list(b_ok)
    out = {"base_success": sum(b_ok) / len(test), "patched_success": sum(n_ok) / len(test),
           "base_ok": b_ok, "patched_ok": n_ok,
           "test_faults": variants}
    res.update(out)
    _save(os.path.join(args.out, f"verify_{args.method}_{int(args.budget)}.json"), res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--method", default="CARVE")
    ap.add_argument("--budget", type=float, default=60)
    ap.add_argument("--n_train", type=int, default=240)
    ap.add_argument("--n_test", type=int, default=120)
    ap.add_argument("--edge_share", type=float, default=0.3)
    ap.add_argument("--audit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from agent_exp.llm import LLM

    llm = LLM()
    db = ShopDB(args.seed)
    tasks = make_tasks(db, args.n_train, seed=args.seed + 1)
    data = stage_collect(llm, args, db, tasks)
    if args.stage == "collect":
        return
    attrs = stage_attribute(llm, args, data)
    tax = stage_taxonomy(llm, args, attrs)
    if args.stage in ("attribute", "taxonomy"):
        return
    if args.stage == "patches":
        stage_patches(llm, args, attrs, tax)
        return
    if args.stage == "calibrate":
        cal = stage_calibrate(llm, args, db, data, attrs, tax)
        print(json.dumps({k: v for k, v in cal.items() if k not in ("rows", "patches")}))
        return
    vpath = os.path.join(args.out, f"verify_{args.method}_{int(args.budget)}.json")
    if args.stage == "evaluate":                 # re-evaluate a finished verification
        print(stage_evaluate(llm, args, db, _load(vpath))["patched_success"])
        return
    res = stage_verify(llm, args, db, data, attrs, tax)
    print(json.dumps({k: res[k] for k in ("method", "precision", "recall", "spent_units")}))
    if args.stage == "all":
        print(stage_evaluate(llm, args, db, res))


if __name__ == "__main__":
    main()
