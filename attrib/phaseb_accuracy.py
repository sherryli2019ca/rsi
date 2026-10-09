"""Attribution accuracy on Phase B's own failing traces (post hoc, second review
of paper 2, Q3).

Phase B gave the proposer step pointers from first write, binary search and
counterfactual search on the 81 failing traces of its 8 states; Phase A's
accuracy was measured on other failures. Here those 81 traces got rescue
profiles under the registered protocol (attrib.groundtruth3, $PHASEB_ROOT/gt/
<domain>/a), and the pointers Phase B actually supplied are scored on them.

  python -m attrib.phaseb_accuracy [--out <json>]

RRSI's own analysis gave no pointer, but its trace digests cite evidence steps;
as in Phase A (attrib.methods rrsi_digest) the earliest cited step stands for it.

Per group: mean R(k-hat) on rescuable failures and on all failures (by domain
and pooled, cluster bootstrap over tasks), exact agreement with the decisive
step, failures without a pointer; paired differences between groups; the
oracle pointer (argmax R_k, the post hoc Phase B group) as the ceiling.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from attrib.analyze import _boot, _mean
from attrib.phaseb import DOMAINS, PB, STATES, parse_state, picks, run_dir, state_dir

GROUPS = ("rrsi", "first_write", "binary_search", "cf_search", "oracle")


def rrsi_steps(dom: str) -> dict:
    """{fid: earliest step cited as evidence by the round's own RRSI digest}."""
    out = {}
    for s in STATES:
        if parse_state(s)[1] != dom:
            continue
        t = parse_state(s)[2]
        fails = json.loads((state_dir(s) / "failures.json").read_text())
        for f in fails:
            p = run_dir("rrsi", s) / f"r{t}" / "analysis" / "digests" / f"{f['task_id']}_failure.json"
            if not p.exists():
                continue
            dg = json.loads(p.read_text())
            steps = sorted({int(m.group(1)) for e in dg.get("evidence") or []
                            if (m := re.search(r"step\s+(\d+)", str((e or {}).get("where", ""))))
                            and int(m.group(1)) < f["n_steps"]})
            out[f["fid"]] = steps[0] if steps else f["n_steps"] - 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(0)
    rows, all_rows, n_fail, max_r, pointed = [], [], {}, {}, {}
    for dom in DOMAINS:
        gt = json.loads((PB / "gt" / dom / "a" / "result.json").read_text())["failures"]
        fails = {f["fid"]: f for f in json.loads((PB / "gt" / f"{dom}_failures.json").read_text())}
        pk = picks(dom)
        pk["rrsi"] = rrsi_steps(dom)
        n_fail[dom] = len(gt)
        max_r[dom] = round(_mean([g["max_R"] for g in gt]), 4)
        pointed[dom] = sum(pk["oracle"].get(g["fid"]) is not None for g in gt)
        for g in gt:
            a = {"domain": dom, "cluster": f"{dom}/{fails[g['fid']]['task_id']}"}
            for grp in GROUPS:
                k = pk[grp].get(g["fid"])
                a[grp] = g["R"][k] if isinstance(k, int) and 0 <= k < len(g["R"]) else 0.0
            all_rows.append(a)
            if g["decisive"] is None:
                continue
            r = {"domain": dom, "cluster": f"{dom}/{fails[g['fid']]['task_id']}"}
            for grp in GROUPS:
                k = pk[grp].get(g["fid"])
                ok = isinstance(k, int) and 0 <= k < len(g["R"])
                r[grp] = g["R"][k] if ok else 0.0
                r[grp + "_exact"] = float(ok and k == g["decisive"])
                r[grp + "_none"] = float(not ok)
            rows.append(r)

    def m(key, sub=lambda r: True):
        return _boot([r for r in rows if sub(r)], lambda xs: _mean([x[key] for x in xs]), rng)

    def diff(a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    res = {"failures": n_fail, "mean_max_R": max_r, "oracle_pointers": pointed,
           "R_all_failures": {g: _boot(all_rows, lambda xs, g=g: _mean([x[g] for x in xs]), rng) for g in GROUPS},
           "rescuable": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS},
           "groups": {g: {"R": {**{d: round(_mean([r[g] for r in rows if r["domain"] == d]), 4) for d in DOMAINS},
                                "pooled": m(g)},
                          "exact": round(_mean([r[g + "_exact"] for r in rows]), 4),
                          "no_pointer": int(sum(r[g + "_none"] for r in rows))} for g in GROUPS},
           "rrsi_digests": {d: len(rrsi_steps(d)) for d in DOMAINS},
           "diffs": {f"{a}-{b}": diff(a, b) for a, b in (("first_write", "rrsi"), ("binary_search", "rrsi"),
                                                         ("cf_search", "rrsi"), ("first_write", "binary_search"),
                                                         ("first_write", "cf_search"),
                                                         ("binary_search", "cf_search"),
                                                         ("oracle", "first_write"),
                                                         ("oracle", "binary_search"),
                                                         ("oracle", "cf_search"))}}
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
