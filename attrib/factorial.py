"""Replays x grading information (post hoc, third review of paper 2), and the
post hoc rows of the main table.

Counterfactual search was registered without the grading criteria the judges
read. attrib.search --informed reran its suspect call with them (same replay
test, budget 40) on the 157 rescuable failures, so search gives a 2x2:

                 no replay (first suspect)   replays (budget 40)
  no grading     search@0                    search@40          (registered run)
  grading        search_inf@0                search_inf@40      (informed run)

  python -m attrib.factorial /home/user/attrib_runs/main [--out <json>]

Per cell: mean R(k-hat) on rescuable failures by domain [95%] and pooled,
exact share, dollars per failure. Contrasts (cluster bootstrap over tasks,
paired by failure): the effect of grading information at each replay level,
the effect of replays at each information level, both main effects and the
interaction; and the judges at the two information levels (registered vs
attrib.rescue_prompt --variant blind), for comparison without replays.
Also the same per-method statistics for the post hoc judge variants
(largest gain over null; no grading information), for the main table.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.analyze import _boot, _mean, _usage_dollars, load
from attrib.pilot_report import replay_dollars

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
JUDGES = ("all_at_once_pro", "binary_search_pro", "all_at_once_gain_pro", "binary_search_gain_pro",
          "all_at_once_blind_pro", "binary_search_blind_pro")


def informed(main: Path, d: str, v: dict) -> tuple[dict, dict, dict]:
    """picks for search_inf@40 and search_inf@0, and dollars per failure."""
    sd = main / d / "search_informed"
    res = json.loads((sd / "result.json").read_text())["failures"]
    at40 = {x["fid"]: x["at40"]["step"] for x in res}
    at0 = {x["fid"]: (x["suspects"][0] if x["suspects"] else None) for x in res}
    sus = _usage_dollars(sd / "usage.jsonl").get("search", 0.0) / max(len(res), 1)
    rep = 0.0
    for x in res:
        for e in x["log"]:
            if e["spent"] > 40:
                break
            for kind in ("c", "n"):
                for j in range(4):
                    p = sd / "replays" / f"{x['fid']}_s{e['i']}_{kind}{j}.json"
                    if p.exists():
                        rep += replay_dollars(json.loads(p.read_text()))
    return at40, at0, {"search_inf@40": sus + rep / len(res), "search_inf@0": sus}


def _rows(data: dict, picks: dict) -> list[dict]:
    rows = []
    for d, v in data.items():
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            r = {"domain": d, "cluster": f"{d}/{v['fails'][fid]['task_id']}"}
            for m, pk in picks.items():
                k = pk[d].get(fid)
                ok = isinstance(k, int) and 0 <= k < len(g["R"])
                r[m] = g["R"][k] if ok else 0.0
                r[m + "_exact"] = float(ok and k == g["decisive"])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    args = ap.parse_args()
    main_dir = Path(args.main)
    rng = random.Random(0)
    data = {d: load(main_dir / d) for d in DOMAINS}
    picks, dollars = {}, {}
    for m in ("first_write", "search@40") + JUDGES:
        picks[m] = {d: data[d]["picks"][m] for d in DOMAINS}
        dollars[m] = {d: data[d]["cost"].get(m, 0.0) for d in DOMAINS}
    search_res = {d: {x["fid"]: x for x in json.loads((main_dir / d / "search" / "result.json").read_text())["failures"]}
                  for d in DOMAINS}
    picks["search@0"] = {d: {f: (x["suspects"][0] if x["suspects"] else None) for f, x in search_res[d].items()}
                         for d in DOMAINS}
    dollars["search@0"] = {d: _usage_dollars(main_dir / d / "search" / "usage.jsonl").get("search", 0.0)
                           / max(len(search_res[d]), 1) for d in DOMAINS}
    picks["search_inf@40"], picks["search_inf@0"] = {}, {}
    dollars["search_inf@40"], dollars["search_inf@0"] = {}, {}
    for d in DOMAINS:
        a40, a0, dl = informed(main_dir, d, data[d])
        picks["search_inf@40"][d], picks["search_inf@0"][d] = a40, a0
        for k, x in dl.items():
            dollars[k][d] = x
    rows = _rows(data, picks)

    def mean(m, dom=None):
        rs = [r for r in rows if dom is None or r["domain"] == dom]
        return _boot(rs, lambda xs: _mean([x[m] for x in xs]), rng)

    def diff(f, dom=None):
        rs = [{"d": f(r), "cluster": r["cluster"]} for r in rows if dom is None or r["domain"] == dom]
        return _boot(rs, lambda xs: _mean([x["d"] for x in xs]), rng)

    methods = {}
    for m in picks:
        methods[m] = {"R": {**{d: mean(m, d) for d in DOMAINS}, "pooled": mean(m)},
                      "exact": round(_mean([r[m + "_exact"] for r in rows]), 4),
                      "dollars_per_failure": round(sum(dollars[m].values()) / 3, 5)}
    s0, s40, i0, i40 = "search@0", "search@40", "search_inf@0", "search_inf@40"
    contrasts = {
        "info_at_no_replay": lambda r: r[i0] - r[s0],
        "info_at_replay": lambda r: r[i40] - r[s40],
        "replay_at_no_info": lambda r: r[s40] - r[s0],
        "replay_at_info": lambda r: r[i40] - r[i0],
        "info_main": lambda r: ((r[i0] - r[s0]) + (r[i40] - r[s40])) / 2,
        "replay_main": lambda r: ((r[s40] - r[s0]) + (r[i40] - r[i0])) / 2,
        "interaction": lambda r: (r[i40] - r[i0]) - (r[s40] - r[s0]),
        "info_judge_all_at_once": lambda r: r["all_at_once_pro"] - r["all_at_once_blind_pro"],
        "info_judge_binary_search": lambda r: r["binary_search_pro"] - r["binary_search_blind_pro"],
        "search_inf@40-all_at_once_pro": lambda r: r[i40] - r["all_at_once_pro"],
        "search_inf@40-binary_search_pro": lambda r: r[i40] - r["binary_search_pro"],
        "search_inf@40-binary_search_gain_pro": lambda r: r[i40] - r["binary_search_gain_pro"],
        "first_write-search_inf@40": lambda r: r["first_write"] - r[i40],
        "search_inf@0-all_at_once_pro": lambda r: r[i0] - r["all_at_once_pro"],
    }
    res = {"n_rescuable": {d: sum(r["domain"] == d for r in rows) for d in DOMAINS}, "methods": methods,
           "contrasts": {k: {"pooled": diff(f), "tau2": _boot(
               [{"d": f(r), "cluster": r["cluster"]} for r in rows if r["domain"] != "appworld"],
               lambda xs: _mean([x["d"] for x in xs]), rng),
               "appworld": diff(f, "appworld")} for k, f in contrasts.items()}}
    flips = {}
    for d in DOMAINS:
        res_i = json.loads((main_dir / d / "search_informed" / "result.json").read_text())["failures"]
        flips[d] = {"informed_flip": round(_mean([not x["at40"]["fallback"] for x in res_i]), 3),
                    "registered_flip": round(_mean([not search_res[d][x["fid"]]["at40"]["fallback"] for x in res_i]), 3),
                    "informed_replays": round(_mean([x["at40"]["replays"] for x in res_i]), 1),
                    "registered_replays": round(_mean([search_res[d][x["fid"]]["at40"]["replays"] for x in res_i]), 1)}
    res["search_flips"] = flips
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
