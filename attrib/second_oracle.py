"""A second oracle for the rescue profiles (post hoc, second review of paper 2).

The registered profiles (gt/a) use one oracle model to propose corrections. Here
the same protocol was re-run with a different oracle model (gt/flash; same K,
same replays per correction, null replays shared with gt/a where both oracles
flagged the step, attrib.groundtruth3 --oracle-model --null-from). This script
compares the two profiles and re-scores every method under the second one and
under their average.

  python -m attrib.second_oracle /home/user/attrib_runs/main [--rep flash] [--out <json>]

  agreement   rescuable under both / either, Cohen's kappa, decisive step equal
              (and within one step) when both call the failure rescuable,
              Pearson r of R_k over all steps, proposal share by oracle
  ranking     mean R(k-hat) per method on failures rescuable under the profile,
              Kendall tau against the registered ranking, first write's rank,
              first write minus binary search and minus the best LLM method
              (cluster bootstrap), for profile = second oracle and = mean of both
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.analyze import _boot, _mean, load
from attrib.robustness import LLM, METHODS, kendall

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
THRESH = 0.5
EXTRA = ("all_at_once_gain_pro", "binary_search_gain_pro")


def _pearson(xs, ys) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** .5
    sy = sum((y - my) ** 2 for y in ys) ** .5
    return sxy / (sx * sy) if sx and sy else float("nan")


def _share(g: dict) -> float:
    st = [sum(s.get("verdict") == "mistake" for s in x["samples"]) / len(x["samples"]) for x in g["steps"]]
    return _mean(st)


def _ranking(data: dict, prof: dict, rng, methods) -> dict:
    rows = []
    for d, v in data.items():
        for f, R in prof[d].items():
            if not R or max(R) < THRESH:
                continue
            r = {"domain": d, "cluster": f"{d}/{v['fails'][f]['task_id']}"}
            for m in methods:
                k = v["picks"][m].get(f)
                r[m] = R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0
            kd = R.index(max(R))
            r["decisive"] = R[kd]
            rows.append(r)
    mean = {m: round(_mean([r[m] for r in rows]), 4) for m in methods}
    best = max(LLM, key=lambda m: mean[m])

    def diff(a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    out = {"n_rescuable": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS},
           "mean": mean, "by_domain": {m: {d: round(_mean([r[m] for r in rows if r["domain"] == d]), 4)
                                           for d in DOMAINS} for m in methods},
           "decisive": round(_mean([r["decisive"] for r in rows]), 4),
           "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
           "best_llm": best,
           "first_write_minus_binary_search": diff("first_write", "binary_search_pro"),
           "first_write_minus_best_llm": diff("first_write", best)}
    for m in EXTRA:
        if m in methods:
            out[f"first_write_minus_{m}"] = diff("first_write", m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--rep", default="flash")
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(0)
    main_dir = Path(args.main)
    data = {d: load(main_dir / d) for d in DOMAINS}
    second = {d: {f["fid"]: f for f in json.loads((main_dir / d / "gt" / args.rep / "result.json").read_text())["failures"]}
              for d in DOMAINS}
    methods = [m for m in METHODS + EXTRA if all(m in v["picks"] for v in data.values())]
    agree = {}
    xs_all, ys_all = [], []
    for d in DOMAINS:
        a, b = data[d]["gt"], second[d]
        both = [f for f in a if f in b]
        ra = [a[f]["decisive"] is not None for f in both]
        rb = [b[f]["decisive"] is not None for f in both]
        n = len(both)
        po = sum(x == y for x, y in zip(ra, rb)) / n
        pa, pb = sum(ra) / n, sum(rb) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        dd = [f for f in both if a[f]["decisive"] is not None and b[f]["decisive"] is not None]
        xs, ys = [], []
        for f in both:
            if len(a[f]["R"]) == len(b[f]["R"]):
                xs += a[f]["R"]
                ys += b[f]["R"]
        xs_all += xs
        ys_all += ys
        agree[d] = {"failures": n, "rescuable_registered": sum(ra), "rescuable_second": sum(rb),
                    "rescuable_both": sum(x and y for x, y in zip(ra, rb)), "agreement": round(po, 3),
                    "kappa": round((po - pe) / (1 - pe), 3) if pe < 1 else None,
                    "decisive_equal": round(_mean([a[f]["decisive"] == b[f]["decisive"] for f in dd]), 3),
                    "decisive_within1": round(_mean([abs(a[f]["decisive"] - b[f]["decisive"]) <= 1 for f in dd]), 3),
                    "step_pearson": round(_pearson(xs, ys), 3), "steps": len(xs),
                    "proposal_share_registered": round(_mean([_share(a[f]) for f in both]), 4),
                    "proposal_share_second": round(_mean([_share(b[f]) for f in both]), 4),
                    "max_R_registered": round(_mean([a[f]["max_R"] for f in both]), 4),
                    "max_R_second": round(_mean([b[f]["max_R"] for f in both]), 4)}
    agree["pooled_step_pearson"] = round(_pearson(xs_all, ys_all), 3)
    reg = {m: _mean([v["gt"][f]["R"][k] if isinstance(k := v["picks"][m].get(f), int) and 0 <= k < len(v["gt"][f]["R"])
                     else 0.0 for v in data.values() for f in v["gt"] if v["gt"][f]["decisive"] is not None])
           for m in METHODS}
    prof2 = {d: {f: g["R"] for f, g in second[d].items()} for d in DOMAINS}
    avg = {d: {f: [(x + y) / 2 for x, y in zip(data[d]["gt"][f]["R"], second[d][f]["R"])]
               for f in second[d] if f in data[d]["gt"] and len(second[d][f]["R"]) == len(data[d]["gt"][f]["R"])}
           for d in DOMAINS}
    res = {"agreement": agree, "registered_mean": {m: round(x, 4) for m, x in reg.items()}}
    for name, prof in (("second", prof2), ("average", avg)):
        r = _ranking(data, prof, rng, methods)
        r["kendall_tau_vs_registered"] = round(kendall(r["mean"], reg), 3)
        res[name] = r
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
