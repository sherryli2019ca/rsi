"""Judges asked for the step with the largest rescue gain instead of the
earliest decisive mistake, or given less information (post hoc; not registered).

The registered judge prompts ask for the earliest step whose correction would
most likely have rescued the task, while the metric scores the rescue gain of
the named step. The rescue variants (attrib.methods all_at_once_rescue,
binary_search_rescue) change only that definition. This script scores them
against their registered versions on the same rescuable failures.

  python -m attrib.rescue_prompt /home/user/attrib_runs/main [--variant rescue|gain|blind] [--out <json>]

Variants: rescue = the step whose correction most likely rescues the task
(first review); gain = the step whose correction most increases success over
letting the agent act again, the quantity R_k measures (second review);
blind = the registered definition, given only what counterfactual search sees
(policy and tools, no hidden customer instructions, no grading section), so
search@40 and search@0 (its first suspect, no replay) are compared with judges
under matched information.

Per pair: mean R(k-hat) by domain and pooled, the paired difference
(rescue minus registered) with a cluster-bootstrap interval, exact agreement
with the decisive step, mean relative position of the named step, the share of
failures on which the two prompts name the same step, and dollars per trace.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.analyze import _boot, _mean, _usage_dollars, load, score

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
VARIANTS = {"rescue": (("all_at_once_flash", "all_at_once_rescue_flash"),
                      ("all_at_once_pro", "all_at_once_rescue_pro"),
                      ("binary_search_pro", "binary_search_rescue_pro")),
            "gain": (("all_at_once_flash", "all_at_once_gain_flash"),
                     ("all_at_once_pro", "all_at_once_gain_pro"),
                     ("binary_search_pro", "binary_search_gain_pro")),
            "blind": (("all_at_once_pro", "all_at_once_blind_pro"),
                      ("binary_search_pro", "binary_search_blind_pro"))}
REF = ("first_write",)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--variant", choices=tuple(VARIANTS), default="rescue")
    ap.add_argument("--out")
    args = ap.parse_args()
    PAIRS = VARIANTS[args.variant]
    rng = random.Random(0)
    data = {d: load(Path(args.main) / d) for d in DOMAINS}
    ref = REF + (("search@40", "search@0") if args.variant == "blind" else ())
    for d, v in data.items():
        if "search@0" in ref:
            sr = json.loads((Path(args.main) / d / "search" / "result.json").read_text())["failures"]
            v["picks"]["search@0"] = {x["fid"]: (x["suspects"][0] if x["suspects"] else None) for x in sr}
            v["cost"]["search@0"] = _usage_dollars(Path(args.main) / d / "search" / "usage.jsonl").get(
                "search", 0.0) / max(len(sr), 1)
    methods = [m for p in PAIRS for m in p] + list(ref)
    rows = []
    for d, v in data.items():
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            r = {"domain": d, "cluster": f"{d}/{v['fails'][fid]['task_id']}", "fid": fid}
            for m in methods:
                k = v["picks"][m].get(fid)
                s = score(g, k)
                r[m], r[m + "_exact"] = s["R"], float(s["exact"])
                r[m + "_pos"] = k / max(g["n_steps"] - 1, 1) if isinstance(k, int) else None
                r[m + "_step"] = k
            rows.append(r)
    res = {"n_rescuable": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS}, "methods": {}, "pairs": {}}
    for m in methods:
        res["methods"][m] = {
            "R": {**{d: round(_mean([r[m] for r in rows if r["domain"] == d]), 4) for d in DOMAINS},
                  "pooled": _boot(rows, lambda rs, m=m: _mean([r[m] for r in rs]), rng)},
            "exact": round(_mean([r[m + "_exact"] for r in rows]), 4),
            "relative_position": round(_mean([r[m + "_pos"] for r in rows if r[m + "_pos"] is not None]), 3),
            "no_step": sum(not isinstance(r[m + "_step"], int) for r in rows),
            "dollars_per_trace": {d: round(data[d]["cost"].get(m, 0.0), 5) for d in DOMAINS}}
    for a, b in PAIRS:
        res["pairs"][f"{b}-{a}"] = {
            **{d: _boot([{"d": r[b] - r[a], "cluster": r["cluster"]} for r in rows if r["domain"] == d],
                        lambda rs: _mean([x["d"] for x in rs]), rng) for d in DOMAINS},
            "pooled": _boot([{"d": r[b] - r[a], "cluster": r["cluster"]} for r in rows],
                            lambda rs: _mean([x["d"] for x in rs]), rng),
            "same_step": round(_mean([float(r[a + "_step"] == r[b + "_step"]) for r in rows]), 3),
            "variant_minus_first_write": _boot([{"d": r[b] - r["first_write"], "cluster": r["cluster"]} for r in rows],
                                              lambda rs: _mean([x["d"] for x in rs]), rng)}
    if args.variant == "blind":
        res["pairs"]["search@40-search@0"] = {
            **{d: _boot([{"d": r["search@40"] - r["search@0"], "cluster": r["cluster"]} for r in rows
                         if r["domain"] == d], lambda rs: _mean([x["d"] for x in rs]), rng) for d in DOMAINS},
            "pooled": _boot([{"d": r["search@40"] - r["search@0"], "cluster": r["cluster"]} for r in rows],
                            lambda rs: _mean([x["d"] for x in rs]), rng)}
        for s_ in ("search@40", "search@0"):
            for _, b in PAIRS:
                res["pairs"][f"{s_}-{b}"] = {
                    **{d: _boot([{"d": r[s_] - r[b], "cluster": r["cluster"]} for r in rows if r["domain"] == d],
                                lambda rs: _mean([x["d"] for x in rs]), rng) for d in DOMAINS},
                    "tau2": _boot([{"d": r[s_] - r[b], "cluster": r["cluster"]} for r in rows
                                   if r["domain"] != "appworld"], lambda rs: _mean([x["d"] for x in rs]), rng),
                    "pooled": _boot([{"d": r[s_] - r[b], "cluster": r["cluster"]} for r in rows],
                                    lambda rs: _mean([x["d"] for x in rs]), rng)}
    # dollars for the whole re-run (all failures, not only rescuable ones)
    spent = 0.0
    for d in DOMAINS:
        n = {m: len(data[d]["picks"][m]) for _, m in PAIRS}
        spent += sum(data[d]["cost"][m] * n[m] for m in n)
    res["rerun_dollars"] = round(spent, 2)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
