"""Selection and scoring on independent draws, and uncertainty from the oracle and the replays (post hoc).

  python -m attrib.independent /home/user/attrib_runs/main [--methods-from <release data/phaseA>] \
      [--runs <dir with <domain>/search_maxgain/aligned>] [--out <json>]

split   each of the 6 x 2 splits of a step's four oracle samples and four null
        replays into halves A and B: failures counted as rescuable on A, every
        method scored on B (as attrib.review2.split_sample, now with the
        gain-asked and blind judges and search@0, which cover all 300
        failures); also every failure scored on B with no selection. Aligned
        search was run on the 157 registered rescuable failures only, so it is
        scored on the A-rescuable failures among them, beside the others on the
        same failures.
draws   two-level bootstrap: tasks resampled with replacement and, within each
        step, its four oracle samples (each with its two correction replays)
        and four null replays resampled with replacement; profile, rescuable
        set and scores recomputed each time. Its intervals include the oracle's
        and the replays' sampling variability, which the task bootstrap of the
        paper's tables (also given here, as tasks_only) holds fixed. fixed157
        keeps the registered 157 failures and redraws only their profiles, so
        aligned search can be compared.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from attrib.analyze import _boot, _mean, load
from attrib.baselines import BLIND, GAIN, _at
from attrib.review2 import _outcomes, _split_profile
from attrib.robustness import DOMAINS, LLM, METHODS

K = 4
DRAWS = 2000
EXTRA = list(GAIN.values()) + list(BLIND) + ["search@0"]
CONTRASTS = ["binary_search_pro", "all_at_once_pro", "binary_search_gain_pro", "all_at_once_gain_pro",
             "second_to_last"]


def _arrays(g: dict, oc: dict, fid: str):
    """Per step: null outcomes (steps x 4), correction means (steps x 4), and whether sample i counts."""
    nul, cor, ok = [], [], []
    for st in g["steps"]:
        o = oc.get((fid, st["k"]))
        if o is None:   # no sample proposed an applicable correction: no replays, R_k = 0
            nul.append([0.0] * K)
            cor.append([0.0] * K)
            ok.append([0.0] * K)
            continue
        nul.append([o["null"][j] for j in range(K)])
        c, v = [], []
        for i in range(K):
            r = o["corr"].get(i, {})
            good = st["samples"][i].get("verdict") == "mistake" and len(r) == 2
            c.append(sum(r.values()) / 2 if good else 0.0)
            v.append(float(good))
        cor.append(c)
        ok.append(v)
    return np.array(nul, float), np.array(cor, float), np.array(ok, float)


def _profile(arr, S, N):
    nul, cor, ok = arr
    pn = np.take_along_axis(nul, N, 1).mean(1)
    return (np.take_along_axis(ok, S, 1) * (np.take_along_axis(cor, S, 1) - pn[:, None])).sum(1) / K


def _scores(R: np.ndarray, picks: dict, n: int) -> dict:
    s = {m: _at(list(R), k) for m, k in picks.items()}
    s["second_to_last"] = float(R[max(n - 2, 0)])
    s["random"] = float(R.mean())
    return s


def split(data, oc, methods, rng) -> dict:
    splits = []
    for SA in itertools.combinations(range(K), 2):
        SB = tuple(i for i in range(K) if i not in SA)
        for NA, NB in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            splits.append((SA, NA, SB, NB))
    per, stacked, stacked_all, stacked_al = [], [], [], []
    for SA, NA, SB, NB in splits:
        resc, allf, al = [], [], []
        for d, v in data.items():
            for f, g in v["gt"].items():
                RA = _split_profile(g, oc[d], f, SA, NA)
                RB = np.array(_split_profile(g, oc[d], f, SB, NB))
                s = _scores(RB, {m: v["picks"][m].get(f) for m in methods}, g["n_steps"])
                s["cluster"] = f"{d}/{v['fails'][f]['task_id']}"
                allf.append(s)
                if max(RA) >= 0.5:
                    resc.append(s)
                    k = v["picks"].get("aligned_search", {}).get(f)
                    if isinstance(k, int):
                        al.append({**s, "aligned_search": _at(list(RB), k)})
        per.append((resc, allf, al))
        stacked += resc
        stacked_all += allf
        stacked_al += al
    cols = list(methods) + ["second_to_last", "random"]

    def summary(sets, rows):
        mean = {m: round(_mean([_mean([r[m] for r in s]) for s in sets]), 4) for m in cols}
        reg = [m for m in METHODS]
        out = {"n": _mean([len(s) for s in sets]), "mean": mean,
               "first_write_rank_registered": 1 + sum(mean[m] > mean["first_write"] for m in reg if m != "first_write"),
               "first_write_rank_all": 1 + sum(mean[m] > mean["first_write"] for m in cols if m != "first_write"),
               "first_write_minus": {}}
        for m in CONTRASTS:
            out["first_write_minus"][m] = _boot([{"d": r["first_write"] - r[m], "cluster": r["cluster"]} for r in rows],
                                                lambda xs: _mean([x["d"] for x in xs]), rng)
        return out

    res = {"splits": len(splits), "rescuable_on_A": summary([p[0] for p in per], stacked),
           "all_failures": summary([p[1] for p in per], stacked_all)}
    if stacked_al:
        res["aligned_subset"] = {
            "n": _mean([len(p[2]) for p in per]),
            "mean": {m: round(_mean([_mean([r[m] for r in p[2]]) for p in per if p[2]]), 4)
                     for m in ["first_write", "binary_search_pro", "binary_search_gain_pro", "aligned_search"]},
            "first_write_minus_aligned_search": _boot(
                [{"d": r["first_write"] - r["aligned_search"], "cluster": r["cluster"]} for r in stacked_al],
                lambda xs: _mean([x["d"] for x in xs]), rng)}
    return res


def draws(data, oc, methods, seed=0) -> dict:
    gen = np.random.default_rng(seed)
    fails = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            fails.append({"cluster": f"{d}/{v['fails'][f]['task_id']}", "arr": _arrays(g, oc[d], f),
                          "picks": {m: v["picks"][m].get(f) for m in methods + ["aligned_search"]
                                    if f in v["picks"].get(m, {})},
                          "n": g["n_steps"], "reg": g["decisive"] is not None, "R": np.array(g["R"])})
            x = fails[-1]
            full = np.tile(np.arange(K), (len(x["R"]), 1))
            assert np.allclose(_profile(x["arr"], full, full), x["R"]), f"profile mismatch {f}"
    by = {}
    for i, x in enumerate(fails):
        by.setdefault(x["cluster"], []).append(i)
    keys = sorted(by)
    cols = list(methods) + ["second_to_last", "random"]
    reg = list(METHODS)
    allm = reg + list(GAIN.values())
    rec = {kind: {"diff": {m: [] for m in CONTRASTS + ["aligned_search"]}, "first_reg": 0, "first_gain": 0,
                  "n": []} for kind in ("tasks_only", "draws", "fixed157")}
    for b in range(DRAWS):
        idx = [i for k in gen.choice(len(keys), len(keys)) for i in by[keys[k]]]
        prof = {}
        for i in set(idx):
            steps = len(fails[i]["R"])
            S = gen.integers(0, K, (steps, K))
            N = gen.integers(0, K, (steps, K))
            prof[i] = _profile(fails[i]["arr"], S, N)
        for kind in rec:
            rows = []
            for i in idx:
                x = fails[i]
                R = x["R"] if kind == "tasks_only" else prof[i]
                keep = x["reg"] if kind == "fixed157" else R.max() >= 0.5
                if keep:
                    rows.append(_scores(R, x["picks"], x["n"]))
            mean = {m: _mean([r.get(m) for r in rows]) for m in cols + ["aligned_search"]}
            rec[kind]["n"].append(len(rows))
            for m in CONTRASTS:
                rec[kind]["diff"][m].append(mean["first_write"] - mean[m])
            if kind == "fixed157":
                rec[kind]["diff"]["aligned_search"].append(mean["first_write"] - mean["aligned_search"])
            rec[kind]["first_reg"] += all(mean["first_write"] >= mean[m] for m in reg)
            rec[kind]["first_gain"] += all(mean["first_write"] >= mean[m] for m in allm)

    def q(xs):
        xs = np.sort(np.array(xs))
        return [round(float(np.quantile(xs, .025)), 4), round(float(np.quantile(xs, .975)), 4)]

    out = {"draws": DRAWS}
    for kind, r in rec.items():
        out[kind] = {"n_rescuable": q(r["n"]) + [round(float(np.mean(r["n"])), 1)],
                     "first_write_minus": {m: q(x) for m, x in r["diff"].items() if x},
                     "p_first_registered": round(r["first_reg"] / DRAWS, 3),
                     "p_first_with_gain_asked": round(r["first_gain"] / DRAWS, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--methods-from")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args()
    data = {d: load(Path(a.main) / d) for d in DOMAINS}
    for d, v in data.items():
        for m in list(GAIN.values()) + list(BLIND):
            if m not in v["picks"] and a.methods_from:
                v["picks"][m] = {r["fid"]: r.get("step") for r in (json.loads(p.read_text()) for p in
                                 (Path(a.methods_from) / d / "methods" / m).glob("*.json"))}
        sr = json.loads((Path(a.main) / d / "search" / "result.json").read_text())["failures"]
        v["picks"]["search@0"] = {x["fid"]: (x["suspects"][0] if x["suspects"] else None) for x in sr}
        if a.runs:
            res = json.loads((Path(a.runs) / d / "search_maxgain" / "aligned" / "result.json").read_text())
            v["picks"]["aligned_search"] = {x["fid"]: x["maxgain"] for x in res["failures"]}
    methods = [m for m in list(METHODS) + EXTRA
               if all(len(v["picks"].get(m, {})) == len(v["gt"]) for v in data.values())]
    oc = {d: _outcomes(Path(a.main), d) for d in DOMAINS}
    out = {"methods": methods, "split": split(data, oc, methods, random.Random(0)),
           "bootstrap": draws(data, oc, methods)}
    print(json.dumps(out, indent=1, default=lambda o: o.item()))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
