"""Pilot report for the attribution ground truth: dollars spent, what the
curtailed scan would have spent on the same replays, and agreement between
two independent repetitions (reps) of the ground truth.

  python -m attrib.pilot_report <out dir> [--reps a b]

Curtailed scan (exact: it reaches the same decisive step as the full scan):
steps are scanned in order and the scan stops at the first flip; at a step the
corrected replays run first, one by one, and stop once FLIP successes are out
of reach; the null replays run only if the corrected ones reached FLIP, and
stop once the step's flip is settled either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attrib.groundtruth import FLIP, N_REPLAYS, ORACLE_MODEL

# US dollars per million tokens (uncached input, cached input, output), as in
# verify/posthoc_e1.py; the user simulator's tokens at the uncached price
PRICES = {"deepseek-v4-flash": (0.14, 0.0028, 0.28), "deepseek-v4-pro": (0.435, 0.0036, 0.87)}


def replay_dollars(rec: dict) -> float:
    pi, pc, po = PRICES["deepseek-v4-flash"]
    t = rec.get("tokens") or {}
    return ((t.get("agent_in") or 0) * pi + (t.get("agent_cache_read") or 0) * pc
            + (t.get("agent_out") or 0) * po + (t.get("user") or 0) * pi) / 1e6


def oracle_dollars(usage: Path) -> tuple[float, int]:
    pi, pc, po = PRICES[ORACLE_MODEL]
    d, n = 0.0, 0
    if usage.exists():
        for line in usage.read_text().splitlines():
            u = json.loads(line)
            if u.get("role") != "oracle":
                continue
            n += 1
            d += ((u.get("in") or 0) * pi + (u.get("cache_read") or 0) * pc + (u.get("out") or 0) * po) / 1e6
    return d, n


def _curtailed(c: list[int], nl: list[int]) -> tuple[int, bool]:
    """Replays the curtailed rule runs at one step, and whether the step flips."""
    used, s = 0, 0
    for j, r in enumerate(c):
        used, s = used + 1, s + r
        if s + (N_REPLAYS - j - 1) < FLIP:
            return used, False
    allowed, z = s - FLIP, 0          # flip iff null successes <= allowed
    for j, r in enumerate(nl):
        used, z = used + 1, z + r
        if z > allowed:
            return used, False
        if z + (N_REPLAYS - j - 1) <= allowed:
            return used, True
    return used, z <= allowed


def rep_stats(base: Path, rep: str) -> dict:
    res = json.loads((base / rep / "result.json").read_text())
    rdir = base / rep / "replays"
    full_d = sum(replay_dollars(json.loads(p.read_text())) for p in rdir.glob("*.json")
                 if not p.name.startswith("refs_"))
    n_full = sum(1 for p in rdir.glob("*.json") if not p.name.startswith("refs_"))
    cur_n, cur_d, mism = 0, 0.0, 0
    for f in res["failures"]:
        decisive = None
        for s in f["steps"]:
            if s["verdict"] != "mistake":
                continue
            recs = {kind: [json.loads((rdir / f"{f['fid']}_k{s['k']}_{kind}{j}.json").read_text())
                           for j in range(N_REPLAYS)] for kind in "cn"}
            c = [int(r.get("reward", 0) >= 1 and (r.get("replay") or {}).get("forced", False))
                 for r in recs["c"]]
            nl = [int(r.get("reward", 0) >= 1) for r in recs["n"]]
            used, flip = _curtailed(c, nl)
            order = recs["c"] + recs["n"]
            cur_n += used
            cur_d += sum(replay_dollars(r) for r in order[:used])
            if flip:
                decisive = s["k"]
                break
        mism += decisive != f["decisive"]
    od, on = oracle_dollars(base / rep / "usage.jsonl")
    return {"failures": len(res["failures"]), "oracle_calls": on, "oracle_dollars": round(od, 4),
            "replays_full": n_full, "replay_dollars_full": round(full_d, 4),
            "replays_curtailed": cur_n, "replay_dollars_curtailed": round(cur_d, 4),
            "curtailed_mismatch": mism, "result": res}


def agreement(ra: dict, rb: dict) -> dict:
    a = {f["fid"]: f for f in ra["failures"]}
    b = {f["fid"]: f for f in rb["failures"]}
    common = sorted(set(a) & set(b))
    same = sum(a[x]["decisive"] == b[x]["decisive"] for x in common)
    near = sum(a[x]["decisive"] is not None and b[x]["decisive"] is not None
               and abs(a[x]["decisive"] - b[x]["decisive"]) <= 1 for x in common)
    both = sum(a[x]["decisive"] is not None and b[x]["decisive"] is not None for x in common)
    found = {r: sum(d[x]["decisive"] is not None for x in common) for r, d in (("a", a), ("b", b))}
    return {"n": len(common), "same_decisive": same, "within_1_both_found": near,
            "both_found": both, "found": found,
            "pairs": [(x, a[x]["decisive"], b[x]["decisive"], a[x]["flips"], b[x]["flips"]) for x in common]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reps", nargs="+", default=["a", "b"])
    args = ap.parse_args()
    report = {}
    for dom_dir in sorted(p for p in Path(args.out).iterdir() if (p / "failures.json").exists()):
        st = {r: rep_stats(dom_dir, r) for r in args.reps if (dom_dir / r / "result.json").exists()}
        row = {r: {k: v for k, v in s.items() if k != "result"} for r, s in st.items()}
        if len(st) >= 2:
            r1, r2 = list(st)[:2]
            row["agreement"] = agreement(st[r1]["result"], st[r2]["result"])
        report[dom_dir.name] = row
    print(json.dumps(report, indent=1))
    (Path(args.out) / "pilot_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
