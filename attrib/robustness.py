"""Robustness of the Phase A method ranking to choices in the ground truth
(post hoc; not registered). Recomputes every step's rescue gain from the
per-sample replay outcomes stored in gt/a/result.json and re-scores the
methods under:

  main        K = 4 oracle samples, null replays subtracted, threshold 0.5
  K=2, K=1    every subset of 2 (6) or 1 (4) of the 4 samples, averaged
  t=0.25 ...  other rescuability thresholds
  no null     corrected success alone, no null replays subtracted
  unique      only failures with exactly one step at or above 0.5

  python -m attrib.robustness /home/user/attrib_runs/main [--out <json>]

For each variant: rescuable failures, pooled mean R(k-hat) per method,
Kendall's tau of the method ordering against main, the first write's rank,
and first write minus binary search with a cluster-bootstrap interval.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from attrib.analyze import _boot, _mean, load

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
METHODS = ("first_write", "last_step", "all_at_once_flash", "all_at_once_pro", "all_at_once_pro_think",
           "step_by_step_pro", "binary_search_pro", "study1_pro", "rrsi_digest_pro", "search@40")
LLM = METHODS[2:]


def sample_gain(step: dict, s: dict, use_null: bool = True) -> float:
    """groundtruth3.profile's per-sample gain."""
    if s.get("verdict") != "mistake" or s.get("corrected") is None or s.get("n") != 2:
        return 0.0
    if not use_null:
        return s["corrected"]
    return s["corrected"] - step["null"] if step["null"] is not None else 0.0


def profile(g: dict, idx=None, use_null=True) -> list[float]:
    out = []
    for st in g["steps"]:
        ss = st["samples"] if idx is None else [st["samples"][i] for i in idx]
        out.append(sum(sample_gain(st, s, use_null) for s in ss) / len(ss))
    return out


def scores(data: dict, idx=None, use_null=True, thr=0.5, unique=False) -> dict:
    rows = {m: [] for m in METHODS}
    n = 0
    for d, v in data.items():
        for fid, g in v["gt"].items():
            R = profile(g, idx, use_null)
            if max(R) < thr:
                continue
            if unique and sum(r >= 0.5 for r in R) != 1:
                continue
            n += 1
            cl = f"{d}/{v['fails'][fid]['task_id']}"
            for m in METHODS:
                k = v["picks"][m].get(fid)
                r = R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0
                rows[m].append({"R": r, "cluster": cl, "fid": fid})
    return {"n": n, "rows": rows, "mean": {m: _mean([r["R"] for r in rows[m]]) for m in METHODS}}


def kendall(a: dict, b: dict) -> float:
    c = dsc = 0
    for x, y in itertools.combinations(METHODS, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        c += s > 0
        dsc += s < 0
    return (c - dsc) / (c + dsc) if c + dsc else 1.0


def rank(mean: dict, m: str) -> int:
    return 1 + sum(mean[x] > mean[m] for x in METHODS if x != m)


def diff(sc: dict, a: str, b: str, rng) -> tuple:
    rb = {r["fid"]: r["R"] for r in sc["rows"][b]}
    rows = [{"d": r["R"] - rb[r["fid"]], "cluster": r["cluster"]} for r in sc["rows"][a]]
    return _boot(rows, lambda rs: _mean([r["d"] for r in rs]), rng)


def summarize(name: str, scs: list[dict], ref: dict, rng) -> dict:
    mean = {m: _mean([s["mean"][m] for s in scs]) for m in METHODS}
    best = max(LLM, key=lambda m: mean[m])
    out = {"variant": name, "n_rescuable": _mean([s["n"] for s in scs]),
           "mean": {m: round(mean[m], 4) for m in METHODS},
           "kendall_tau_vs_main": round(_mean([kendall(s["mean"], ref) for s in scs]), 3),
           "first_write_rank": _mean([rank(s["mean"], "first_write") for s in scs]),
           "best_llm": best, "best_llm_R": round(mean[best], 4)}
    if len(scs) == 1:
        out["first_write_minus_binary_search"] = diff(scs[0], "first_write", "binary_search_pro", rng)
        out["first_write_minus_best_llm"] = diff(scs[0], "first_write", best, rng)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(0)
    data = {d: load(Path(args.main) / d) for d in DOMAINS}
    base = scores(data)
    ref = base["mean"]
    res = [summarize("main", [base], ref, rng)]
    for k in (2, 1):
        res.append(summarize(f"K={k}", [scores(data, idx) for idx in itertools.combinations(range(4), k)], ref, rng))
    for t in (0.25, 0.375, 0.625, 0.75):
        res.append(summarize(f"threshold {t}", [scores(data, thr=t)], ref, rng))
    res.append(summarize("no null", [scores(data, use_null=False)], ref, rng))
    res.append(summarize("unique rescuing step", [scores(data, unique=True)], ref, rng))
    # how many failures a null-free protocol would call rescuable, per domain
    extra = {d: {"rescuable_main": sum(max(profile(g)) >= .5 for g in v["gt"].values()),
                 "rescuable_no_null": sum(max(profile(g, use_null=False)) >= .5 for g in v["gt"].values())}
             for d, v in data.items()}
    # success of null replays by the step's relative position (fifths of the
    # episode); null replays exist only where an oracle sample proposed a correction
    null_by_pos = {}
    for d, v in data.items():
        bins = [[] for _ in range(5)]
        for g in v["gt"].values():
            for st in g["steps"]:
                if st["null"] is not None:
                    bins[min(int(5 * st["k"] / max(g["n_steps"] - 1, 1)), 4)].append(st["null"])
        null_by_pos[d] = [[round(_mean(b), 4), len(b)] for b in bins]
    # mean relative position of the step each method names, rescuable failures
    pos = {m: round(_mean([v["picks"][m][f] / max(g["n_steps"] - 1, 1) for d, v in data.items()
                           for f, g in v["gt"].items() if g["decisive"] is not None
                           and isinstance(v["picks"][m].get(f), int)]), 3) for m in METHODS}
    pos["decisive"] = round(_mean([g["decisive"] / max(g["n_steps"] - 1, 1) for v in data.values()
                                   for g in v["gt"].values() if g["decisive"] is not None]), 3)
    out = {"variants": res, "rescuable_by_domain": extra, "null_by_position": null_by_pos,
           "named_relative_position": pos}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    for r in res:
        fw = r.get("first_write_minus_binary_search")
        print(f"{r['variant']:22s} n={r['n_rescuable']:6.1f} tau={r['kendall_tau_vs_main']:+.2f} "
              f"fw={r['mean']['first_write']:.3f} rank={r['first_write_rank']:.1f} "
              f"best={r['best_llm']}:{r['best_llm_R']:.3f} " + (f"fw-bs={fw}" if fw else ""))
    print(json.dumps(extra))
    print(json.dumps(null_by_pos))
    print(json.dumps(pos))


if __name__ == "__main__":
    main()
