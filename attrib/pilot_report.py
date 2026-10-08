"""Pilot report for the attribution ground truth: dollars spent, what the
curtailed scan would have spent on the same replays, and agreement between
two independent repetitions (reps) of the ground truth.

  python -m attrib.pilot_report <out dir> [--reps a b]

Curtailed scan (exact: it reaches the same decisive step as the full scan):
steps are scanned in order and the scan stops at the first flip; at a step the
corrected replays run first and the null replays only if the corrected ones
reached FLIP successes.
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
    """Replays the curtailed scan (attrib.groundtruth.run_curtailed) runs at one
    step, and whether the step flips."""
    if sum(c) < FLIP:
        return len(c), False
    return len(c) + len(nl), sum(c) - sum(nl) >= FLIP


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


def agree(da: dict, db: dict) -> dict:
    """Agreement of two {fid: decisive step or None} maps."""
    common = sorted(set(da) & set(db))
    either = [x for x in common if da[x] is not None or db[x] is not None]
    return {"n": len(common), "same": sum(da[x] == db[x] for x in common),
            "found_either": len(either), "same_when_found": sum(da[x] == db[x] for x in either),
            "found": [sum(d[x] is not None for x in common) for d in (da, db)]}


def multi_rep(dom_dir: Path, reps: list[str]) -> dict:
    """Protocol 1 with 4+ reps: every pair, and the union rule (earliest flip
    of two reps pooled) on disjoint pairs of reps."""
    res = {r: json.loads((dom_dir / r / "result.json").read_text())["failures"] for r in reps}
    dec = {r: {f["fid"]: f["decisive"] for f in fs} for r, fs in res.items()}
    flips = {r: {f["fid"]: set(f["flips"]) for f in fs} for r, fs in res.items()}
    pairs = {f"{a}-{b}": agree(dec[a], dec[b]) for i, a in enumerate(reps) for b in reps[i + 1:]}
    union = {}
    if len(reps) >= 4:
        a, b, c, d = reps[:4]
        for (p, q), (u, v) in (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))):
            du = {x: min(flips[p][x] | flips[q][x], default=None) for x in dec[p]}
            dv = {x: min(flips[u][x] | flips[v][x], default=None) for x in dec[u]}
            union[f"{p}{q}-{u}{v}"] = agree(du, dv)
    return {"pairs": pairs, "union": union}


def _pearson(x: list, y: list) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** .5
    sy = sum((b - my) ** 2 for b in y) ** .5
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy), 3) if sx and sy else None


def protocol3(p3: Path, p1: Path | None) -> dict:
    """Test-retest of protocol 3 (reps a and b): step-level rescue gains and the
    decisive-step label; with protocol-1 reps, the correlation of protocol 3's
    gains with protocol 1's pooled gains (mean over reps of (corrected - null) / 4
    at oracle-flagged steps, 0 elsewhere)."""
    out = {}
    for dom_dir in sorted(p for p in p3.iterdir() if (p / "a" / "result.json").exists()):
        ra, rb = ({f["fid"]: f for f in json.loads((dom_dir / r / "result.json").read_text())["failures"]}
                  for r in ("a", "b"))
        fids = sorted(set(ra) & set(rb))
        xa = [v for x in fids for v in ra[x]["R"]]
        xb = [v for x in fids for v in rb[x]["R"]]
        row = {"failures": len(fids), "steps": len(xa), "step_R_corr": _pearson(xa, xb),
               "decisive": agree({x: ra[x]["decisive"] for x in fids}, {x: rb[x]["decisive"] for x in fids}),
               "earliest": agree({x: ra[x]["earliest"] for x in fids}, {x: rb[x]["earliest"] for x in fids}),
               "within_1_both_found": sum(ra[x]["decisive"] is not None and rb[x]["decisive"] is not None
                                          and abs(ra[x]["decisive"] - rb[x]["decisive"]) <= 1 for x in fids),
               "pairs": [(x, ra[x]["decisive"], rb[x]["decisive"], round(ra[x]["max_R"], 2),
                          round(rb[x]["max_R"], 2)) for x in fids]}
        if p1 is not None and (p1 / dom_dir.name).exists():
            reps = [r for r in "abcd" if (p1 / dom_dir.name / r / "result.json").exists()]
            pooled = {}
            for r in reps:
                for f in json.loads((p1 / dom_dir.name / r / "result.json").read_text())["failures"]:
                    g = [((s["corrected"] - s["null"]) / 4 if s["verdict"] == "mistake"
                          and s.get("n_corrected") == N_REPLAYS and s.get("n_null") == N_REPLAYS else 0.0)
                         for s in f["steps"]]
                    acc = pooled.setdefault(f["fid"], [0.0] * len(g))
                    pooled[f["fid"]] = [a + b / len(reps) for a, b in zip(acc, g)]
            x3 = [(a + b) / 2 for x in fids for a, b in zip(ra[x]["R"], rb[x]["R"])]
            x1 = [v for x in fids for v in pooled[x]]
            row["corr_with_protocol1_pooled"] = _pearson(x3, x1)
        out[dom_dir.name] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reps", nargs="+", default=["a", "b"])
    ap.add_argument("--multi", action="store_true", help="protocol-1 agreement over every pair of --reps")
    ap.add_argument("--protocol2", default=None, help="a protocol-2 output dir with reps a and b")
    ap.add_argument("--protocol3", default=None, help="a protocol-3 output dir with reps a and b "
                    "(`out` is then the protocol-1 pilot dir)")
    args = ap.parse_args()
    if args.protocol3:
        rep = protocol3(Path(args.protocol3), Path(args.out))
        print(json.dumps(rep, indent=1))
        (Path(args.protocol3) / "pilot_agreement.json").write_text(json.dumps(rep, indent=1))
        return
    if args.multi or args.protocol2:
        rep = {}
        for dom_dir in sorted(p for p in Path(args.out).iterdir() if (p / "failures.json").exists()):
            rep[dom_dir.name] = {"protocol1": multi_rep(dom_dir, args.reps)}
            if args.protocol2:
                p2 = Path(args.protocol2) / dom_dir.name
                da, db = ({f["fid"]: f["decisive"] for f in
                           json.loads((p2 / r / "result.json").read_text())["failures"]} for r in ("a", "b"))
                rep[dom_dir.name]["protocol2"] = agree(da, db)
        print(json.dumps(rep, indent=1))
        (Path(args.out) / "pilot_agreement.json").write_text(json.dumps(rep, indent=1))
        return
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
