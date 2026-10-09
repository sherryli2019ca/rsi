"""A zero-cost attribution rule that uses the grading report (post hoc; not
registered): the first step at which the agent's state-changing calls depart
from tau2's gold actions.

Walk the agent's steps in order. A state-changing call (not a read prefix of
domains.tau2.common, transfers included) that matches an unused gold action
(same name, same arguments, list order ignored) uses it up; the first one that
matches none names its step. If every state-changing call matched, the agent
left a gold write undone or failed on what it said, and the rule names the
last step. AppWorld ships no gold action sequence, so the rule is tau2 only.

  python -m attrib.gold_rule /home/user/attrib_runs/main [--out <json>]

Writes <main>/<domain>/methods/gold_deviation/<fid>.json for the two tau2
domains and prints the mean R(k-hat) on rescuable failures by domain and
pooled, with paired differences against the first write and the judges.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path

from attrib.analyze import _boot, _mean, load, score

DOMAINS = ("tau2_retail", "tau2_airline")
COMPARE = ("first_write", "all_at_once_pro", "binary_search_pro", "search@40")


def _norm(x):
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return sorted((_norm(v) for v in x), key=lambda v: json.dumps(v, sort_keys=True))
    return x


def gold_deviation(rec: dict) -> dict:
    from domains.tau2.common import READ_PREFIXES
    gold = rec.get("gold_actions") or []
    if isinstance(gold, str):
        gold = ast.literal_eval(gold)
    unused = [(g["name"], _norm(g.get("arguments") or {})) for g in gold
              if not g["name"].startswith(READ_PREFIXES)]
    for st in rec["steps"]:
        for b in st["assistant"]:
            if b["type"] != "tool_use" or b["name"].startswith(READ_PREFIXES):
                continue
            key = (b["name"], _norm(b.get("input") or {}))
            if key in unused:
                unused.remove(key)
            else:
                return {"step": st["index"], "top3": [st["index"]], "component": None, "calls": 0}
    n = int(rec["n_steps"])
    return {"step": n - 1, "top3": [n - 1], "component": None, "calls": 0, "fallback": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    args = ap.parse_args()
    main_dir = Path(args.main)
    for d in DOMAINS:
        out = main_dir / d / "methods" / "gold_deviation"
        out.mkdir(parents=True, exist_ok=True)
        for f in json.loads((main_dir / d / "failures.json").read_text()):
            rec = json.loads(Path(f["trace"]).read_text())
            (out / f"{f['fid']}.json").write_text(json.dumps({"fid": f["fid"], "method": "gold_deviation",
                                                               **gold_deviation(rec)}, indent=1))
    rng = random.Random(0)
    data = {d: load(main_dir / d) for d in DOMAINS}
    rows = []
    for d, v in data.items():
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            r = {"domain": d, "cluster": f"{d}/{v['fails'][fid]['task_id']}", "fid": fid}
            for m in ("gold_deviation",) + COMPARE:
                r[m] = score(g, v["picks"][m].get(fid))["R"]
                r[m + "_exact"] = score(g, v["picks"][m].get(fid))["exact"]
            r["fallback"] = json.loads((main_dir / d / "methods" / "gold_deviation" / f"{fid}.json")
                                       .read_text()).get("fallback", False)
            rows.append(r)
    res = {"n_rescuable": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS},
           "fallback_share": _mean([float(r["fallback"]) for r in rows]),
           "R": {m: {**{d: _mean([r[m] for r in rows if r["domain"] == d]) for d in DOMAINS},
                     "tau2": _boot(rows, lambda rs, m=m: _mean([r[m] for r in rs]), rng)}
                 for m in ("gold_deviation",) + COMPARE},
           "exact": {m: _mean([float(r[m + "_exact"]) for r in rows]) for m in ("gold_deviation",) + COMPARE},
           "gold_minus": {m: _boot([{"d": r["gold_deviation"] - r[m], "cluster": r["cluster"]} for r in rows],
                                   lambda rs: _mean([x["d"] for x in rs]), rng) for m in COMPARE}}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
