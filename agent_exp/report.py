"""Aggregate the real-agent runs (runs/real_s*) into results/real.json and
paper/tables/real.tex.

Usage: python -m agent_exp.report [runs/real_s0 runs/real_s1 ...]
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

from agent_exp.agent import FAULTS
from agent_exp.pipeline import FULL, SINGLE, build_problem

METHODS = [("LLM-only", "LLM attribution only"),
           ("Replay-each", "Replay-each (do-then-verify)"),
           ("Uncertainty", "Uncertainty sampling"),
           ("Uncertainty-MF", "Uncertainty + cheap checks"),
           ("CARVE-full-only", "CARVE, full replay only"),
           ("CARVE", "CARVE (default check model)"),
           ("CARVE-calibrated", "CARVE, calibrated check"),
           ("CARVE-robust-check", "CARVE, robust check model")]


def load(p):
    with open(p) as fh:
        return json.load(fh)


def seed_summary(d, budgets=(10, 20, 30, 60)):
    traces = load(f"{d}/traces.json")
    attrs = load(f"{d}/attributions.json")
    tax = load(f"{d}/taxonomy.json")
    cal = load(f"{d}/calibration.json")
    cats, counts, truth, members, f = build_problem(attrs, tax, 0.3)
    faulted = [a for a in attrs.values() if a["true_fault"]]
    out = {
        "n_train": len(traces),
        "train_success": float(np.mean([t["ok"] for t in traces])),
        "n_failures": len(attrs),
        "n_failures_no_fault": sum(1 for a in attrs.values() if not a["true_fault"]),
        "attr_acc": float(np.mean([a["component"] == FAULTS[a["true_fault"]][0]
                                   for a in faulted])),
        "n_categories": len(tax["categories"]),
        "n_true_edges": int(truth.sum()),
        "calibration": {k: v for k, v in cal.items() if k not in ("rows", "patches")},
        "methods": {},
    }
    fail_by_fault = {}
    for t in traces:
        f_ = (t["trace"]["faults"] or ["none"])[0]
        fail_by_fault.setdefault(f_, []).append(1 - t["ok"])
    out["failure_rate_by_fault"] = {k: float(np.mean(v)) for k, v in fail_by_fault.items()}
    for m, _ in METHODS:
      for budget in budgets:
        fn = f"{d}/verify_{m}_{budget}.json"
        if m == "LLM-only":
            fn = f"{d}/verify_{m}_60.json"
        try:
            r = load(fn)
        except FileNotFoundError:
            continue
        log = r["log"]
        out["methods"][f"{m}|{budget}"] = {
            "precision": r["precision"], "recall": r["recall"],
            "n_accepted": len(r["accepted"]), "accepted": r["accepted"],
            "n_full": sum(1 for x in log if x["fidelity"] == FULL),
            "n_single": sum(1 for x in log if x["fidelity"] == SINGLE),
            "spent_units": r["spent_units"],
            "base_success": r.get("base_success"),
            "patched_success": r.get("patched_success"),
            "tokens_verify": sum(v for k, v in r["tokens"].items()),
        }
    return out


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    if len(xs) == 1:
        return f"{xs[0]:.2f}"
    return f"{np.mean(xs):.2f}\\,{{\\scriptsize$\\pm${np.std(xs, ddof=1) / np.sqrt(len(xs)):.2f}}}"


def f1(p, r):
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def main():
    dirs = sys.argv[1:] or sorted(glob.glob("runs/real_s*"))
    seeds = {d: seed_summary(d) for d in dirs}
    with open("results/real.json", "w") as fh:
        json.dump(seeds, fh, indent=1)
    S = list(seeds.values())
    rows = []
    for m, name in METHODS:
        cells = []
        for B in (10, 20, 30, 60):
            got = [s["methods"].get(f"{m}|{B}") for s in S]
            got = [g for g in got if g]
            cells.append(ms([f1(g["precision"], g["recall"]) for g in got]) if got else "--")
        got = [s["methods"].get(f"{m}|60") for s in S]
        cells.append(ms([g["patched_success"] for g in got if g]))
        n_s = np.mean([g["n_single"] for g in got if g])
        cells.append(f"{n_s:.0f}")
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if m == "Uncertainty-MF":
            rows.append("\\midrule")
    base = [s["methods"]["LLM-only|60"]["base_success"] for s in S]
    rows.append("\\midrule")
    rows.append(f"Unpatched agent & & & & & {ms(base)} & \\\\")
    with open("paper/tables/real.tex", "w") as fh:
        fh.write("\n".join(rows) + "\n")
    for d, s in seeds.items():
        print(d, {k: v for k, v in s.items() if k not in ("methods", "calibration")})
        print("  cal", s["calibration"])


if __name__ == "__main__":
    main()
