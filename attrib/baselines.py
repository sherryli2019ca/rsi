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

Ties share the draw. In the same draws, Kendall's tau between each draw's
ordering of the ten registered methods and the registered ordering: the spread
expected from sampling failures alone (null_tau), against which the tau of
other evaluation choices (attrib.robustness) can be read.

Position-corrected scores of every method (registered rescuable failures):

  normalized      R_k / (1 - null success at k): the gain as a share of the room
                  left by uncorrected replays (R_k where k has no null replays)
  position_adj    R_k minus the mean R over all steps of rescuable failures in the
                  same relative-position decile, so a method that only names late
                  steps scores about 0

each with the first write's rank among the ten registered methods and its
difference from the best registered LLM method [95%].

Also: each baseline's score by domain and pooled with
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
from attrib.robustness import DOMAINS, LLM, METHODS, kendall

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
            nulls = [st.get("null") for st in g["steps"]]
            r["Rn"] = [x / (1 - q) if q is not None and q < 1 else x for x, q in zip(R, nulls)]
            r["k"] = {}
            for m in methods:
                k = v["picks"][m].get(f)
                r["k"][m] = k if isinstance(k, int) and 0 <= k < len(R) else None
                r[m] = _at(R, k)
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
    # null distribution of Kendall's tau among bootstrap draws (registered ten methods)
    reg = {m: _mean([r[m] for r in rows]) for m in METHODS}
    taus = []
    rng2 = random.Random(1)
    for _ in range(BOOT):
        smp = [r for k in (rng2.choice(keys) for _ in keys) for r in by[k]]
        taus.append(kendall({m: sum(r[m] for r in smp) / len(smp) for m in METHODS}, reg))
    taus.sort()
    q = lambda p: round(taus[min(int(p * len(taus)), len(taus) - 1)], 3)
    out["null_tau"] = {"q025": q(.025), "q05": q(.05), "q25": q(.25), "median": q(.5),
                       "share_le": {str(t): round(sum(x <= t + 1e-9 for x in taus) / len(taus), 3)
                                    for t in (0.6, 0.778, 0.822, 0.867, 0.889)}}
    # position-corrected scores
    bins = 10
    pos_bin = lambda k, n: min(int(k / max(n - 1, 1) * bins), bins - 1)
    acc = [[] for _ in range(bins)]
    for r in rows:
        for k, x in enumerate(r["R"]):
            acc[pos_bin(k, r["n"])].append(x)
    bmean = [_mean(a) if a else 0.0 for a in acc]
    out["position_bin_mean_R"] = [round(x, 4) for x in bmean]
    allm = list(METHODS) + [m for m in methods if m not in METHODS]
    for r in rows:
        for m in allm:
            k = r["k"][m]
            r[m + "@norm"] = r["Rn"][k] if k is not None else 0.0
            r[m + "@adj"] = r["R"][k] - bmean[pos_bin(k, r["n"])] if k is not None else 0.0
        k2 = max(r["n"] - 2, 0)
        r["second_to_last@norm"] = r["Rn"][k2]
        r["second_to_last@adj"] = r["R"][k2] - bmean[pos_bin(k2, r["n"])]
    out["corrected"] = {}
    for kind in ("norm", "adj"):
        mean = {m: _mean([r[f"{m}@{kind}"] for r in rows]) for m in allm + ["second_to_last"]}
        best = max(LLM, key=lambda m: mean[m])
        out["corrected"][kind] = {
            "mean": {m: round(x, 4) for m, x in sorted(mean.items(), key=lambda x: -x[1])},
            "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
            "best_llm": best,
            "first_write_minus_best_llm": _boot([{"d": r[f"first_write@{kind}"] - r[f"{best}@{kind}"],
                                                  "cluster": r["cluster"]} for r in rows],
                                                lambda xs: _mean([x["d"] for x in xs]), rng),
            "kendall_tau_vs_registered": round(kendall({m: mean[m] for m in METHODS}, reg), 3)}
        for m in ("binary_search_gain_pro", "aligned_search"):
            if m in mean:
                out["corrected"][kind][f"first_write_minus_{m}"] = _boot(
                    [{"d": r[f"first_write@{kind}"] - r[f"{m}@{kind}"], "cluster": r["cluster"]} for r in rows],
                    lambda xs: _mean([x["d"] for x in xs]), rng)
    print(json.dumps({k: out[k] for k in ("n", "first_write_mean_position", "best_position", "p_first")}))
    print(json.dumps(out["null_tau"]), out["position_bin_mean_R"])
    for kind, c in out["corrected"].items():
        print(kind, {k: v for k, v in c.items() if k != "mean"})
        print(list(c["mean"].items()))
    print({m: out["scores"][m]["pooled"] for m in base}, out["first_write_minus"])
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
