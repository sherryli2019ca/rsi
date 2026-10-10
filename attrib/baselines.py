"""Chance and position baselines, and how stable the first place is (post hoc).

Scores step pickers that read nothing but the trace length on the registered
rescuable failures, as Table 1 scores the methods:

  random          expected R(k-hat) of a uniformly random step (mean R over steps;
                  exact = expected share naming the best rescue step)
  position@p      the step at relative position p (round(p * (n_steps - 1)));
                  p = the first write's mean relative position, and the best p on
                  a 0.05 grid (chosen on these failures, so optimistic)
  second_to_last  step n_steps - 2

and draws 10,000 task-clustered bootstrap samples of the failures to count how
often each method has the highest mean, within sets of methods:

  registered9     the nine registered methods other than search@40
  registered10    all ten registered methods
  gain_asked      registered9 with the two Pro judges replaced by their
                  gain-asked versions (all_at_once_gain_pro, binary_search_gain_pro)
  gain_aligned    gain_asked plus the aligned search (needs --runs)

Ties share the draw. Also: each baseline's score by domain and pooled with
cluster-bootstrap intervals and its exact agreement with the best rescue step,
and first write minus each baseline with an interval.

  python -m attrib.baselines /home/user/attrib_runs/main --runs <dir with <domain>/search_maxgain/aligned> \
      --methods-from <release>/data/phaseA --out results/attrib/baselines.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.analyze import BOOT, _boot, _mean, load
from attrib.robustness import DOMAINS, METHODS

GAIN = {"all_at_once_pro": "all_at_once_gain_pro", "binary_search_pro": "binary_search_gain_pro"}


def _at(R: list, k) -> float:
    return R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--runs", help="directory holding <domain>/search_maxgain/aligned")
    ap.add_argument("--methods-from", help="directory holding <domain>/methods/<method> for the gain-asked "
                                           "judges when main lacks them (e.g. the release's data/phaseA)")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = {d: load(Path(a.main) / d) for d in DOMAINS}
    for d, v in data.items():
        for m in GAIN.values():
            if m not in v["picks"] and a.methods_from:
                v["picks"][m] = {r["fid"]: r.get("step") for r in (json.loads(p.read_text()) for p in
                                 (Path(a.methods_from) / d / "methods" / m).glob("*.json"))}
    if a.runs:
        for d, v in data.items():
            res = json.loads((Path(a.runs) / d / "search_maxgain" / "aligned" / "result.json").read_text())
            v["picks"]["aligned_search"] = {x["fid"]: x["maxgain"] for x in res["failures"]}
    methods = list(METHODS) + list(GAIN.values()) + (["aligned_search"] if a.runs else [])
    rows = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            R, n = g["R"], g["n_steps"]
            r = {"domain": d, "cluster": f"{d}/{v['fails'][f]['task_id']}", "R": R, "n": n, "dec": g["decisive"],
                 "random": _mean(R), "second_to_last": _at(R, max(n - 2, 0)),
                 "random_exact": 1 / len(R), "second_to_last_exact": float(max(n - 2, 0) == g["decisive"])}
            for m in methods:
                r[m] = _at(R, v["picks"][m].get(f))
            rows.append(r)
    rng = random.Random(0)
    fw_pos = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            k = v["picks"]["first_write"].get(f)
            if g["decisive"] is not None and isinstance(k, int):
                fw_pos.append(k / max(g["n_steps"] - 1, 1))
    pos_fw = round(_mean(fw_pos), 2)

    def at_pos(p):
        return lambda r: _at(r["R"], round(p * (r["n"] - 1)))

    grid = [round(.05 * i, 2) for i in range(21)]
    by_p = {p: _mean([at_pos(p)(r) for r in rows]) for p in grid}
    best_p = max(grid, key=lambda p: by_p[p])
    for r in rows:
        r["position_fw"] = at_pos(pos_fw)(r)
        r["position_best"] = at_pos(best_p)(r)
        for key, p in (("position_fw", pos_fw), ("position_best", best_p)):
            r[key + "_exact"] = float(round(p * (r["n"] - 1)) == r["dec"])
    base = ("random", "position_fw", "position_best", "second_to_last")
    out = {"n": len(rows), "first_write_mean_position": pos_fw, "best_position": best_p,
           "position_curve": {str(p): round(by_p[p], 4) for p in grid}, "scores": {}, "first_write_minus": {},
           "p_first": {}}
    for m in base + tuple(methods):
        out["scores"][m] = {**{d: _boot([r for r in rows if r["domain"] == d], lambda xs, m=m: _mean([x[m] for x in xs]),
                                        rng) for d in DOMAINS},
                            "pooled": _boot(rows, lambda xs, m=m: _mean([x[m] for x in xs]), rng)}
        if m in base:
            out["scores"][m]["exact"] = round(_mean([r[m + "_exact"] for r in rows]), 4)
    for m in base:
        out["first_write_minus"][m] = _boot([{"d": r["first_write"] - r[m], "cluster": r["cluster"]} for r in rows],
                                            lambda xs: _mean([x["d"] for x in xs]), rng)
    reg9 = [m for m in METHODS if m != "search@40"]
    sets = {"registered9": reg9, "registered10": list(METHODS),
            "gain_asked": [GAIN.get(m, m) for m in reg9]}
    if a.runs:
        sets["gain_aligned"] = sets["gain_asked"] + ["aligned_search"]
    by = {}
    for r in rows:
        by.setdefault(r["cluster"], []).append(r)
    keys = sorted(by)
    wins = {s: dict.fromkeys(ms, 0.0) for s, ms in sets.items()}
    for _ in range(BOOT):
        smp = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        for s, ms in sets.items():
            mean = {m: sum(r[m] for r in smp) / len(smp) for m in ms}
            top = max(mean.values())
            best = [m for m in ms if mean[m] >= top - 1e-12]
            for m in best:
                wins[s][m] += 1 / len(best)
    for s, w in wins.items():
        out["p_first"][s] = {m: round(x / BOOT, 3) for m, x in sorted(w.items(), key=lambda x: -x[1]) if x > 0}
    print(json.dumps({k: out[k] for k in ("n", "first_write_mean_position", "best_position", "p_first")}))
    print({m: out["scores"][m]["pooled"] for m in base}, out["first_write_minus"])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
