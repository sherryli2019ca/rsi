"""Zero-cost analyses for the fourteenth review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r14 report --traj r1=DIR/runs r2=... r3=... [--json results/r14/report.json]

lobo (review question 3): experiment BK's contrasts with each block left out
in turn, from the registered lineage model (verify/bk.py) on the three
remaining blocks, with 90% and 95% t intervals. Uses the transfers recorded in
results/bk/analysis.json.

budget (review question 4): what the registered and cost-aware rules' savings
would buy if spent on more rounds. Each BK arm's mean loop cost (as if run
alone, results/bk/analysis.json) less its calibration gives a cost per round;
at full evaluation's ten-round cost a rule runs R = 10 * (cost_f - cal) /
(cost_rule - cal) rounds. On the recorded airline trajectories r1-r3, which
ran 20 rounds, a rule's value at matched cost is its round values summed over
rounds 0 to floor(R) - 1 plus the fraction R - floor(R) of the next round, in
full evaluation's states (the noise-band replays of verify/posthoc_r13.py at
the calibrated band), against full evaluation's sum over rounds 0-9. 90%
intervals resample the three trajectories and the held-out tasks (B = 4000).
Also: full evaluation's mean round value over rounds 10-19, the value of a
further round in these trajectories.

il (review question 1): experiment IL's three registered primary outcomes
(results/il/analysis.json) with 95% Welch t intervals and two-sided p values,
Holm-adjusted over the three; the registration reported 90% bootstrap
intervals only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

from . import analyze as A
from . import bk
from .posthoc_simple import _merged

BK_JSON = Path("results/bk/analysis.json")
IL_JSON = Path("results/il/analysis.json")
RULES = {"seqfull": ("s", "seqfull|m1"), "seqcost": ("c", "seqcost|m1")}


def lobo(res: dict) -> dict:
    Yall = {tuple(k.split("_")): v for k, v in res["transfer"].items()}
    blocks = list(bk.BLOCKS)
    out = {}
    try:
        for drop in blocks:
            bk.BLOCKS = [b for b in blocks if b != drop]
            Y = {k: v for k, v in Yall.items() if k[0] in bk.BLOCKS}
            means = {a: float(np.mean([Y[(b, a)] for b in bk.BLOCKS])) for a in bk.ARMS}
            row = {}
            for k, c in bk.CONTRASTS.items():
                est = bk._contrast(means, c)
                se, df = bk._lineage_se(Y, c)
                x = {"est": est, "se": se, "df": df}
                if df > 0.5 and math.isfinite(df):
                    for lvl, q in (("ci90", .95), ("ci95", .975)):
                        h = float(tdist.ppf(q, df)) * se
                        x[lvl] = [est - h, est + h]
                row[k] = x
            out[drop] = row
    finally:
        bk.BLOCKS = blocks
    return out


def _rounds(res: dict) -> dict:
    loops = res["loops"]
    mean = lambda a, k: float(np.mean([loops[f"{b}_{a}"]["cost"][k] for b in bk.BLOCKS]))
    cal = mean("f", "calibration")
    per = {a: (mean(a, "total") - cal) / 10 for a in ("f", "s", "c")}
    return {"cost_per_round": per, "calibration": cal,
            **{r: 10 * per["f"] / per[a] for r, (a, _) in RULES.items()}}


def _value(rows: list, key: str, R: float, tasks) -> float:
    n, frac = int(math.floor(R)), R - math.floor(R)
    v = [A.round_value(r, key, tasks) for r in rows]
    return float(sum(v[:n]) + (frac * v[n] if n < len(v) else 0.0))


def budget(trajs: dict, res: dict, B: int = 4000, seed: int = 21) -> dict:
    R = _rounds(res)
    keys = ["full|m1"] + [k for _, k in RULES.values()]
    tables, _ = _merged(trajs, ["r1", "r2", "r3"], "tau2_airline", "e1_r13_seq.json", keys)
    rows = [sorted(T["rows"], key=lambda r: r["t"]) for T in tables]
    tasks = [sorted({t for r in rs for d in r["dep"].values() for t in d}) for rs in rows]

    def diffs(idx, ts):
        out = {}
        for rule, (_, key) in RULES.items():
            f10 = [_value(rows[i], "full|m1", 10, ts[i]) for i in idx]
            out[rule + "_equal_rounds"] = float(np.mean([_value(rows[i], key, 10, ts[i]) for i in idx]) - np.mean(f10))
            out[rule + "_equal_cost"] = float(np.mean([_value(rows[i], key, R[rule], ts[i]) for i in idx]) - np.mean(f10))
        out["full_later_round"] = float(np.mean([A.round_value(r, "full|m1", ts[i]) for i in idx for r in rows[i][10:20]]))
        return out
    point = diffs(range(len(rows)), [None] * len(rows))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = list(rng.integers(0, len(rows), len(rows)))
        ts = [list(rng.choice(t, len(t))) for t in tasks]
        draws.append(diffs(idx, ts))
    ci = {k: [float(np.percentile([d[k] for d in draws], q)) for q in (5, 95)] for k in point}
    return {"rounds_at_full_cost": R, "trajectories": [T["name"] for T in tables],
            "point": point, "ci90": ci}


def il_welch() -> dict:
    prim = json.loads(IL_JSON.read_text())["primary"]
    out = {}
    for k, x in prim.items():
        v1, v2 = x["seq_sd"] ** 2 / x["seq"], x["full_sd"] ** 2 / x["full"]
        se = math.sqrt(v1 + v2)
        df = (v1 + v2) ** 2 / (v1 ** 2 / (x["seq"] - 1) + v2 ** 2 / (x["full"] - 1))
        h = float(tdist.ppf(.975, df)) * se
        out[k] = {"diff": x["diff"], "se": se, "df": df, "ci95_welch": [x["diff"] - h, x["diff"] + h],
                  "p_two_sided": float(2 * tdist.sf(abs(x["diff"]) / se, df)), "ci90_registered": x["ci90"]}
    order = sorted(out, key=lambda k: out[k]["p_two_sided"])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * out[k]["p_two_sided"]))
        out[k]["p_holm"] = run
    return out


def report(trajs: dict, js: Path | None):
    res = json.loads(BK_JSON.read_text())
    out = {"lobo": lobo(res), "budget": budget(trajs, res), "il": il_welch()}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(out, indent=1, default=float))
    pp = lambda x: f"{100 * x:+.2f}"
    for drop, row in out["lobo"].items():
        print(f"== without {drop}")
        for k, x in row.items():
            c90, c95 = x.get("ci90"), x.get("ci95")
            print(f"  {k:16s} {pp(x['est'])}" + (f" 90% [{pp(c90[0])},{pp(c90[1])}] 95% [{pp(c95[0])},{pp(c95[1])}]"
                                              if c90 else "  (no interval)") + f" df {x['df']:.1f}")
    b = out["budget"]
    print("== matched cost; rounds at full evaluation's cost:",
          {k: (round(v, 2) if isinstance(v, float) else v) for k, v in b["rounds_at_full_cost"].items()})
    for k, v in b["point"].items():
        c = b["ci90"][k]
        print(f"  {k:22s} {pp(v)} [{pp(c[0])},{pp(c[1])}]")
    print("== IL primary outcomes, Welch")
    for k, x in out["il"].items():
        print(f"  {k:9s} diff {x['diff']:+.4f} 95% [{x['ci95_welch'][0]:+.4f},{x['ci95_welch'][1]:+.4f}] "
              f"df {x['df']:.1f} p {x['p_two_sided']:.3f} Holm {x['p_holm']:.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report"])
    ap.add_argument("--traj", nargs="+", required=True)
    ap.add_argument("--json", default="results/r14/report.json")
    a = ap.parse_args()
    report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
