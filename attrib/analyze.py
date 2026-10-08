"""Phase A analysis (attrib/PREREGISTRATION.md, Addendum 1): every method's
named step scored against the protocol-3 rescue profile.

  python -m attrib.analyze <main dir> [--domains tau2_retail tau2_airline]

<main dir>/<domain>/ holds failures.json, methods/<method>_<model>/<fid>.json,
search/result.json, gt/a/result.json (and gt/b for the retest). Writes
<main dir>/analysis.json and prints a table.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.pilot_report import PRICES, replay_dollars

BOOT = 10_000
SEARCH_BUDGETS = (8, 16, 24, 32, 40)


def _usage_dollars(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        u = json.loads(line)
        pi, pc, po = PRICES.get(u.get("model"), PRICES["deepseek-v4-pro"])
        out[u.get("role")] = out.get(u.get("role"), 0.0) + (
            (u.get("in") or 0) * pi + (u.get("cache_read") or 0) * pc + (u.get("out") or 0) * po) / 1e6
    return out


def load(dom_dir: Path) -> dict:
    fails = {f["fid"]: f for f in json.loads((dom_dir / "failures.json").read_text())}
    gt = {f["fid"]: f for f in json.loads((dom_dir / "gt" / "a" / "result.json").read_text())["failures"]}
    picks, cost = {}, {}
    usage = _usage_dollars(dom_dir / "methods" / "usage.jsonl")
    for mdir in sorted(p for p in (dom_dir / "methods").iterdir() if p.is_dir()):
        rows = {}
        for p in mdir.glob("*.json"):
            r = json.loads(p.read_text())
            rows[r["fid"]] = r.get("step")
        picks[mdir.name] = rows
        role = next((mdir.name[:-len(sf)] + ":" + sf[1:] for sf in ("_pro_think", "_pro", "_flash")
                     if mdir.name.endswith(sf)), None)
        if mdir.name.startswith("rrsi_digest"):
            role = "digester"
        cost[mdir.name] = (usage.get(role, 0.0) / max(len(rows), 1)) if role else 0.0
    sp = dom_dir / "search" / "result.json"
    if sp.exists():
        sr = json.loads(sp.read_text())["failures"]
        sus_cost = _usage_dollars(dom_dir / "search" / "usage.jsonl").get("search", 0.0) / max(len(sr), 1)
        rdir = dom_dir / "search" / "replays"
        for b in SEARCH_BUDGETS:
            key = f"at{b}"
            if not all(key in x for x in sr):
                continue
            picks[f"search@{b}"] = {x["fid"]: x[key]["step"] for x in sr}
            rep_d = 0.0
            for x in sr:
                spent = x[key]["replays"]
                for e in x["log"]:
                    if e["spent"] > spent:
                        break
                    for kind in ("c", "n"):
                        for j in range(4):
                            p = rdir / f"{x['fid']}_s{e['i']}_{kind}{j}.json"
                            if p.exists():
                                rep_d += replay_dollars(json.loads(p.read_text()))
            cost[f"search@{b}"] = sus_cost + rep_d / len(sr)
    return {"fails": fails, "gt": gt, "picks": picks, "cost": cost}


def score(gt: dict, step) -> dict:
    R = gt["R"]
    ok = isinstance(step, int) and 0 <= step < len(R)
    r = R[step] if ok else 0.0
    d, e = gt["decisive"], gt["earliest"]
    return {"R": r, "share": (r / gt["max_R"]) if gt["max_R"] > 0 else None,
            "exact": ok and d is not None and step == d,
            "within1": ok and d is not None and abs(step - d) <= 1,
            "earliest": ok and e is not None and step == e}


def _boot(rows: list[dict], f, rng: random.Random) -> tuple:
    """Point estimate and 95% cluster-bootstrap interval (clusters: tasks)."""
    by = {}
    for r in rows:
        by.setdefault(r["cluster"], []).append(r)
    keys = sorted(by)
    est = f(rows)
    draws = []
    for _ in range(BOOT):
        smp = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        v = f(smp)
        if v is not None:
            draws.append(v)
    draws.sort()
    lo, hi = draws[int(.025 * len(draws))], draws[int(.975 * len(draws)) - 1]
    return round(est, 4), round(lo, 4), round(hi, 4)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def analyze(main: Path, domains: list[str]) -> dict:
    rng = random.Random(0)
    data = {d: load(main / d) for d in domains if (main / d / "gt" / "a" / "result.json").exists()}
    methods = sorted(set.intersection(*(set(v["picks"]) for v in data.values()))) if data else []
    out = {"n": {}, "methods": {}}
    for d, v in data.items():
        g = v["gt"]
        out["n"][d] = {"failures": len(g), "rescuable": sum(x["decisive"] is not None for x in g.values()),
                       "mean_max_R": round(_mean([x["max_R"] for x in g.values()]), 3)}
    for m in methods:
        rows = []
        for d, v in data.items():
            for fid, g in v["gt"].items():
                s = score(g, v["picks"][m].get(fid))
                rows.append({**s, "rescuable": g["decisive"] is not None,
                             "cluster": f"{d}/{v['fails'][fid]['task_id']}", "fid": fid, "domain": d})
        resc = [r for r in rows if r["rescuable"]]
        out["methods"][m] = {
            "R_rescuable": _boot(resc, lambda rs: _mean([r["R"] for r in rs]), rng),
            "R_all": _boot(rows, lambda rs: _mean([r["R"] for r in rs]), rng),
            "share_rescuable": round(_mean([r["share"] for r in resc]) or 0, 4),
            "exact": _boot(resc, lambda rs: _mean([float(r["exact"]) for r in rs]), rng),
            "within1": round(_mean([float(r["within1"]) for r in resc]) or 0, 4),
            "earliest": round(_mean([float(r["earliest"]) for r in resc]) or 0, 4),
            "dollars_per_failure": round(_mean([v["cost"].get(m, 0.0) for v in data.values()]) or 0, 5),
            "by_domain": {d: round(_mean([r["R"] for r in resc if r["domain"] == d]) or 0, 4) for d in data}}
        out["methods"][m]["_rows"] = rows
    # primary comparison: search@40 vs all-at-once (pro), paired R on rescuable failures
    a, b = "search@40", "all_at_once_pro"
    if a in out["methods"] and b in out["methods"]:
        ra = {r["fid"]: r for r in out["methods"][a]["_rows"] if r["rescuable"]}
        rb = {r["fid"]: r for r in out["methods"][b]["_rows"] if r["rescuable"]}
        diffs = [{"d": ra[f]["R"] - rb[f]["R"], "cluster": ra[f]["cluster"]} for f in ra]
        est = _boot(diffs, lambda rs: _mean([r["d"] for r in rs]), rng)
        by = {}
        for r in diffs:
            by.setdefault(r["cluster"], []).append(r["d"])
        sums = [sum(v) for v in by.values()]
        obs = abs(sum(sums))
        hits = sum(abs(sum(s if rng.random() < .5 else -s for s in sums)) >= obs - 1e-12 for _ in range(BOOT))
        b01 = sum(ra[f]["exact"] and not rb[f]["exact"] for f in ra)
        b10 = sum(rb[f]["exact"] and not ra[f]["exact"] for f in ra)
        out["primary"] = {"diff_R": est, "permutation_p": round((hits + 1) / (BOOT + 1), 4),
                          "n": len(diffs), "exact_discordant": [b01, b10]}
    # retest
    ret = {}
    for d in data:
        pb = main / d / "gt" / "b" / "result.json"
        if pb.exists():
            gb = {f["fid"]: f for f in json.loads(pb.read_text())["failures"]}
            ga = data[d]["gt"]
            common = sorted(set(gb) & set(ga))
            ret[d] = {"n": len(common), "same_decisive": sum(ga[x]["decisive"] == gb[x]["decisive"] for x in common),
                      "both_found": sum(ga[x]["decisive"] is not None and gb[x]["decisive"] is not None for x in common)}
    out["retest"] = ret
    for m in out["methods"].values():
        m.pop("_rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--domains", nargs="+", default=["tau2_retail", "tau2_airline"])
    args = ap.parse_args()
    res = analyze(Path(args.main), args.domains)
    (Path(args.main) / "analysis.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["n"]), json.dumps(res.get("primary")), json.dumps(res.get("retest")))
    print(f"{'method':28s} {'R resc [95%]':24s} {'exact':18s} {'share':6s} {'$/fail':8s}")
    for m, v in sorted(res["methods"].items(), key=lambda kv: -kv[1]["R_rescuable"][0]):
        r, e = v["R_rescuable"], v["exact"]
        print(f"{m:28s} {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}]   {e[0]:.2f} [{e[1]:.2f},{e[2]:.2f}]  "
              f"{v['share_rescuable']:.2f}  {v['dollars_per_failure']:.4f}")


if __name__ == "__main__":
    main()
