"""Post-hoc analyses for the second external review of paper 2 (not registered).

  python -m attrib.review2 /home/user/attrib_runs/main [--out <json>]

  null_regeneration   how often a null replay from step k (which keeps steps
                      0..k-1 and lets the agent act again) reproduced the
                      observed action at k: same text and calls (tau2) or the
                      same code (AppWorld), and the same calls alone (tau2)
  decomposition       R_k = (m_k / K) * gbar_k: m_k of the K oracle samples
                      gave a correction that was replayed (proposal share),
                      gbar_k is their mean gain over null (conditional
                      effectiveness); at each method's named step on rescuable
                      failures, and over all steps by kind of step
  search_first        counterfactual search's first-ranked suspect without any
                      replay (search@0), against search@40 and the judges
  split_sample        rescuable failures and decisive steps chosen on two of the
                      four oracle samples and two of the four null replays,
                      methods scored on the other two samples and nulls;
                      averaged over all 6 x 2 splits
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

from attrib.analyze import _boot, _mean, load, score
from attrib.robustness import METHODS, LLM, kendall

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
K = 4


def _same_calls(a: list, b: list) -> bool:
    ca = [(x["name"], json.dumps(x.get("input"), sort_keys=True)) for x in a if x["type"] == "tool_use"]
    cb = [(x["name"], json.dumps(x.get("input"), sort_keys=True)) for x in b if x["type"] == "tool_use"]
    return ca == cb


def _same_text(a: list, b: list) -> bool:
    ta = " ".join(x["text"].strip() for x in a if x["type"] == "text")
    tb = " ".join(x["text"].strip() for x in b if x["type"] == "text")
    return ta == tb


def null_regeneration(main: Path, d: str, v: dict) -> dict:
    rd = main / d / "gt" / "a" / "replays"
    traces = {}
    n = moved = exact = calls = 0
    for p in rd.glob("*_n*.json"):
        fid, rest = p.stem.rsplit("_k", 1)
        k = int(rest.split("_")[0])
        if fid not in v["fails"]:
            continue
        x = json.loads(p.read_text())
        rp = x.get("replay") or {}
        n += 1
        if rp.get("start_effective") != k:
            moved += 1
            continue
        if fid not in traces:
            traces[fid] = {s["index"]: s for s in json.loads(Path(v["fails"][fid]["trace"]).read_text())["steps"]}
        orig = traces[fid].get(k)
        new = next((s for s in x["steps"] if s["index"] == k), None)
        if orig is None or new is None:
            continue
        if d == "appworld":
            same = (orig.get("code") or "").strip() == (new.get("code") or "").strip()
            exact += same
            calls += same
        else:
            sc = _same_calls(orig["assistant"], new["assistant"])
            calls += sc
            exact += sc and _same_text(orig["assistant"], new["assistant"])
    started = n - moved
    return {"null_replays": n, "start_moved": moved, "same_action": round(exact / started, 4),
            "same_calls_or_code": round(calls / started, 4)}


def _parts(st: dict) -> tuple[int, list]:
    gains = []
    for s in st["samples"]:
        if s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2 \
                and st["null"] is not None:
            gains.append(s["corrected"] - st["null"])
    return len(gains), gains


def decomposition(data: dict, main: Path) -> dict:
    from attrib.report_paper import aw_kind, tau2_kind
    named = {}
    for m in METHODS + ("decisive",):
        share, cond, R = [], [], []
        for d, v in data.items():
            for f, g in v["gt"].items():
                if g["decisive"] is None:
                    continue
                k = g["decisive"] if m == "decisive" else v["picks"][m].get(f)
                if not (isinstance(k, int) and 0 <= k < len(g["steps"])):
                    share.append(0.0)
                    R.append(0.0)
                    continue
                mk, gains = _parts(g["steps"][k])
                share.append(mk / K)
                R.append(sum(gains) / K)
                if mk:
                    cond.append(sum(gains) / mk)
        named[m] = {"proposal_share": round(_mean(share), 4), "conditional_gain": round(_mean(cond), 4),
                    "n_with_proposal": len(cond), "R": round(_mean(R), 4)}
    by_kind = {}
    for d, v in data.items():
        kind = aw_kind if d == "appworld" else tau2_kind
        acc = defaultdict(lambda: {"steps": 0, "share": [], "cond": [], "null": []})
        for f, g in v["gt"].items():
            steps = {s["index"]: s for s in json.loads(Path(v["fails"][f]["trace"]).read_text())["steps"]}
            for st in g["steps"]:
                a = acc[kind(steps[st["k"]])]
                mk, gains = _parts(st)
                a["steps"] += 1
                a["share"].append(mk / K)
                if mk:
                    a["cond"].append(sum(gains) / mk)
                if st["null"] is not None:
                    a["null"].append(st["null"])
        by_kind[d] = {kk: {"steps": a["steps"], "proposal_share": round(_mean(a["share"]), 4),
                           "conditional_gain": round(_mean(a["cond"]), 4) if a["cond"] else None,
                           "null": round(_mean(a["null"]), 4) if a["null"] else None}
                      for kk, a in sorted(acc.items())}
    return {"named": named, "by_kind": by_kind}


def search_first(main: Path, data: dict, rng) -> dict:
    rows = []
    for d, v in data.items():
        sr = {x["fid"]: x for x in json.loads((main / d / "search" / "result.json").read_text())["failures"]}
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            sus = sr[f]["suspects"]
            r = {"domain": d, "cluster": f"{d}/{v['fails'][f]['task_id']}",
                 "search@0": score(g, sus[0] if sus else None)["R"]}
            for m in ("search@40", "all_at_once_pro", "first_write"):
                r[m] = score(g, v["picks"][m].get(f))["R"]
            rows.append(r)

    def diff(a, b, sub):
        rs = [{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rows if sub(r)]
        return _boot(rs, lambda xs: _mean([x["d"] for x in xs]), rng)
    tau = lambda r: r["domain"] != "appworld"  # noqa: E731
    aw = lambda r: r["domain"] == "appworld"  # noqa: E731
    return {"R": {m: {**{d: round(_mean([r[m] for r in rows if r["domain"] == d]), 4) for d in DOMAINS},
                      "pooled": _boot(rows, lambda xs, m=m: _mean([x[m] for x in xs]), rng)}
                  for m in ("search@0", "search@40", "all_at_once_pro")},
            "search40_minus_search0": {"tau2": diff("search@40", "search@0", tau), "appworld": diff("search@40", "search@0", aw),
                                       "pooled": diff("search@40", "search@0", lambda r: True)},
            "search0_minus_judge_pro": {"tau2": diff("search@0", "all_at_once_pro", tau),
                                        "appworld": diff("search@0", "all_at_once_pro", aw)}}


def _outcomes(main: Path, d: str) -> dict:
    """(fid, k) -> {"null": [rewards by j], "corr": {i: [rewards]}} from the replay files."""
    out = defaultdict(lambda: {"null": {}, "corr": defaultdict(dict)})
    for p in (main / d / "gt" / "a" / "replays").glob("*_k*_*.json"):
        fid, rest = p.stem.rsplit("_k", 1)
        parts = rest.split("_")
        if not parts[0].isdigit() or len(parts) < 2:
            continue
        k = int(parts[0])
        x = json.loads(p.read_text())
        ok = int(x.get("reward", 0) >= 1)
        if parts[1].startswith("n"):
            out[(fid, k)]["null"][int(parts[1][1:])] = ok
        else:
            if not (x.get("replay") or {}).get("forced"):
                continue
            out[(fid, k)]["corr"][int(parts[1][1:])][int(parts[2])] = ok
    return out


def _split_profile(g: dict, oc: dict, fid: str, S: tuple, N: tuple) -> list[float]:
    R = []
    for st in g["steps"]:
        o = oc.get((fid, st["k"]))
        nul = [o["null"][j] for j in N if o and j in o["null"]]
        null = sum(nul) / len(nul) if nul else None
        tot = 0.0
        for i in S:
            s = st["samples"][i]
            if s.get("verdict") != "mistake" or null is None or not o:
                continue
            c = o["corr"].get(i, {})
            if len(c) == 2:
                tot += sum(c.values()) / 2 - null
        R.append(tot / len(S))
    return R


def split_sample(main: Path, data: dict, rng) -> dict:
    oc = {d: _outcomes(main, d) for d in DOMAINS}
    splits = []
    for SA in itertools.combinations(range(K), 2):
        SB = tuple(i for i in range(K) if i not in SA)
        for NA, NB in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            splits.append((SA, NA, SB, NB))
    per = []
    stacked = []
    for SA, NA, SB, NB in splits:
        rows = {m: [] for m in METHODS}
        opt = []
        n = 0
        for d, v in data.items():
            for f, g in v["gt"].items():
                RA = _split_profile(g, oc[d], f, SA, NA)
                if max(RA) < 0.5:
                    continue
                RB = _split_profile(g, oc[d], f, SB, NB)
                n += 1
                kA = RA.index(max(RA))
                opt.append((RA[kA], RB[kA]))
                row = {"cluster": f"{d}/{v['fails'][f]['task_id']}"}
                for m in METHODS:
                    k = v["picks"][m].get(f)
                    row[m] = RB[k] if isinstance(k, int) and 0 <= k < len(RB) else 0.0
                    rows[m].append(row[m])
                stacked.append(row)
        per.append({"n": n, "mean": {m: _mean(rows[m]) for m in METHODS},
                    "decisive_in_sample": _mean([a for a, _ in opt]), "decisive_out_of_sample": _mean([b for _, b in opt])})
    mean = {m: round(_mean([p["mean"][m] for p in per]), 4) for m in METHODS}
    best = max(LLM, key=lambda m: mean[m])
    return {"splits": len(splits), "n_rescuable": _mean([p["n"] for p in per]), "mean": mean,
            "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
            "best_llm": best,
            "first_write_minus_binary_search": _boot(
                [{"d": r["first_write"] - r["binary_search_pro"], "cluster": r["cluster"]} for r in stacked],
                lambda xs: _mean([x["d"] for x in xs]), rng),
            "first_write_minus_best_llm": _boot(
                [{"d": r["first_write"] - r[best], "cluster": r["cluster"]} for r in stacked],
                lambda xs: _mean([x["d"] for x in xs]), rng),
            "decisive_gain_in_sample": round(_mean([p["decisive_in_sample"] for p in per]), 4),
            "decisive_gain_out_of_sample": round(_mean([p["decisive_out_of_sample"] for p in per]), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    args = ap.parse_args()
    main_dir = Path(args.main)
    rng = random.Random(0)
    data = {d: load(main_dir / d) for d in DOMAINS}
    res = {"null_regeneration": {d: null_regeneration(main_dir, d, data[d]) for d in DOMAINS},
           "decomposition": decomposition(data, main_dir),
           "search_first": search_first(main_dir, data, rng)}
    sp = split_sample(main_dir, data, rng)
    reg = {m: _mean([score(g, v["picks"][m].get(f))["R"] for v in data.values() for f, g in v["gt"].items()
                     if g["decisive"] is not None]) for m in METHODS}
    sp["kendall_tau_vs_main"] = round(kendall(sp["mean"], reg), 3)
    res["split_sample"] = sp
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
