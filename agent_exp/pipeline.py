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

import numpy as np

from agent_exp.agent import COMPONENT_DOCS, FAULTS, Step, Trace, components_with, run_agent
from agent_exp.analyst import attribute, induce_taxonomy, judge_step, label, make_patches
from agent_exp.env import ShopDB, check, make_tasks
from carve.model import FULL, SINGLE, GraphPosterior, ObsParams
from carve.policies import CARVE, LLMOnly, ReplayEach, UncertaintySampling
from sim.testbed import edge_prior

COMP_IDS = list(COMPONENT_DOCS)
VARIANTS = list(FAULTS) + [None]


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _save(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.__dict__)


def _trace_from(d) -> Trace:
    t = Trace(d["task_id"], d["question"], d["faults"], answer=d["answer"],
              stopped=d["stopped"], tokens=d["tokens"])
    t.steps = [Step(**s) for s in d["steps"]]
    return t


def stage_collect(llm, args, db, tasks):
    path = os.path.join(args.out, "traces.json")
    data = _load(path) or []
    rng = random.Random(args.seed)
    for task in tasks[len(data):]:
        v = rng.choice(VARIANTS)
        faults = [v] if v else []
        tr = run_agent(llm, db, task, components_with(faults), faults)
        data.append({"trace": tr, "gold": task.answer, "ok": check(tr.answer, task.answer)})
        _save(path, data)
    return data


def stage_attribute(llm, args, data):
    path = os.path.join(args.out, "attributions.json")
    attrs = _load(path) or {}
    for d in data:
        tid = d["trace"]["task_id"] if isinstance(d["trace"], dict) else d["trace"].task_id
        if d["ok"] or tid in attrs:
            continue
        tr = _trace_from(d["trace"]) if isinstance(d["trace"], dict) else d["trace"]
        a = attribute(llm, tr, d["gold"], components_with(tr.faults))
        a["true_fault"] = tr.faults[0] if tr.faults else None
        attrs[tid] = a
        _save(path, attrs)
    return attrs


def stage_taxonomy(llm, args, attrs):
    path = os.path.join(args.out, "taxonomy.json")
    tax = _load(path)
    if tax is None:
        ids = sorted(attrs)
        cats = induce_taxonomy(llm, [attrs[i]["root_cause"] for i in ids])
        labels = {i: label(llm, attrs[i]["root_cause"], cats) for i in ids}
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

    def __init__(self, llm, db, data, attrs, cats, members, rng, k=3):
        self.llm, self.db, self.attrs, self.cats = llm, db, attrs, cats
        self.members, self.rng, self.k = members, rng, k
        self.by_id = {(d["trace"]["task_id"] if isinstance(d["trace"], dict)
                       else d["trace"].task_id): d for d in data}
        self.patches = {}                       # (i, j) -> list of patch texts
        self.tokens = {"full": [], "single": [], "patch": 0}
        self.log = []

    def get_patches(self, i, j):
        if (i, j) not in self.patches:
            before = sum(self.llm.tokens.values())
            ex = [self.attrs[t]["root_cause"] for t in self.members[i]]
            self.patches[(i, j)] = make_patches(self.llm, COMP_IDS[j],
                                                components_with(), self.cats[i], ex, self.k)
            self.tokens["patch"] += sum(self.llm.tokens.values()) - before
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

    def unit_costs(self):
        full = np.mean(self.tokens["full"]) if self.tokens["full"] else 1.0
        single = np.mean(self.tokens["single"]) if self.tokens["single"] else 0.1 * full
        return 1.0, float(single / full), float(full)


def stage_verify(llm, args, db, data, attrs, tax):
    cats, counts, truth, members, f = build_problem(attrs, tax, args.edge_share)
    rng = random.Random(args.seed + 1)
    world = RealWorld(llm, db, data, attrs, cats, members, rng)
    n_fail = len(attrs)
    r_hat = np.mean([a["component"] == FAULTS[a["true_fault"]][0]
                     for a in list(attrs.values())[:args.audit] if a["true_fault"]] or [0.5])
    post = GraphPosterior(edge_prior(counts, r_hat), world.k,
                          ObsParams(b=0.05, lam=0.8, sens=0.85, fpr=0.15))
    m = args.method
    pol = {"CARVE": lambda: CARVE(costs=(1.0, 0.1)),
           "CARVE-full-only": lambda: CARVE(costs=(1.0, 0.1), fidelities=(FULL,)),
           "Uncertainty": UncertaintySampling,
           "LLM-only": lambda: LLMOnly(counts),
           "Replay-each": lambda: ReplayEach(counts)}[m]()
    spent = 0.0
    while True:
        act = pol.select(post, f)
        if act is None:
            break
        i, j, k, fid = act
        c = pol.costs[fid] if hasattr(pol, "costs") else (1.0, 0.1)[fid]
        if spent + c > args.budget:
            break
        o = world.intervene(i, j, k, fid)
        post.update(i, j, k, fid, o)
        if hasattr(pol, "record"):
            pol.record(i, j, k, fid, o)
        spent += c
        if hasattr(pol, "costs") and world.tokens["full"]:
            # re-estimate the single-step / full cost ratio from measured tokens
            pol.costs = world.unit_costs()[:2]
            pol._cache = None
    acc, patch = pol.decide(post, f)
    tp = int((acc & truth).sum())
    res = {"method": m, "budget": args.budget, "spent_units": spent,
           "unit_costs": world.unit_costs(), "tokens": dict(llm.tokens),
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


def stage_evaluate(llm, args, db, res):
    test = make_tasks(db, args.n_test, seed=args.seed + 99)
    rng = random.Random(args.seed + 2)
    ok_base = ok_new = 0
    for task in test:
        v = rng.choice(VARIANTS)
        faults = [v] if v else []
        b = run_agent(llm, db, task, components_with(faults), faults)
        n = run_agent(llm, db, task, components_with(faults, res["apply"]), faults)
        ok_base += check(b.answer, task.answer)
        ok_new += check(n.answer, task.answer)
    out = {"base_success": ok_base / len(test), "patched_success": ok_new / len(test)}
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
    res = stage_verify(llm, args, db, data, attrs, tax)
    print(json.dumps({k: res[k] for k in ("method", "precision", "recall", "spent_units")}))
    if args.stage == "all":
        print(stage_evaluate(llm, args, db, res))


if __name__ == "__main__":
    main()
