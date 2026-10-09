"""Post-hoc analyses for the eighth review of paper 1 (not registered).

  python -m verify.posthoc_r8 accept --traj r1=DIR r2=DIR r3=DIR r4=DIR e3=DIR [--json FILE]
  python -m verify.posthoc_r8 bias --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r8 tables --traj ... --json accept.json --bias-json bias.json

accept: agreement with full evaluation conditional on full evaluation's
  decision. In rounds where full evaluation accepts a candidate c*, how often a
  rule picks c* (recall of acceptances), what full's accepted candidates were
  worth held out, split by whether the rule also accepts them, and the per-round
  decision-value difference split into gains the rule forgoes (c* helped held
  out), harm it avoids (c* hurt), and what it picks instead. In rounds where full
  evaluation keeps the incumbent, how often the rule accepts something and what
  that is worth. The four parts sum to D(rule) - D(full).
  Intervals: block bootstrap (blocks of rounds sharing an incumbent, and held-out
  tasks, within domain and trajectory), as in verify/posthoc_r7.py.

bias: for the coverage diagnostic of posthoc_r7 (Table 24), separates a
  systematic offset of the evidence from the spread of its draws: per candidate,
  the mean of M evidence draws minus full evaluation's evolve dS (signed error),
  the draws' standard deviation, the mean reported standard error, and the share
  of draws whose +-z*se interval covers the draws' own mean (spread-only
  coverage). Writes <out>/e1_r8_bias.json.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import analyze as A
from .posthoc_r7 import FULL_B, SETS, _load_tables

RULES = ("seqfull", "seqsample@80", "sample@80", "sample@40", "net@40", "replaynull@40", "judge",
         "none", "keep")
KEEP = (None, "null")


def _gain(row, c, tasks=None):
    d = row["dep"].get(c) or {}
    ts = tasks if tasks is not None else list(d)
    vals = [d[t] for t in ts if t in d]
    return float(np.mean(vals)) if vals else 0.0


def _probs(row, rule):
    return {"null": 1.0} if rule == "keep" else row["probs"][rule]


def _stats(tables, rule, sample=None):
    """sample: None for the observed rounds, else [(table, row indices, tasks)]."""
    if sample is None:
        sample = [(T, range(len(T["rows"])), None) for T in tables]
    acc, keep = [], []
    for T, idx, tasks in sample:
        for i in idx:
            row = T["rows"][i]
            f = row["probs"]["full"]
            cs = max(f, key=f.get)
            P = _probs(row, rule)
            if cs in KEEP:
                keep.append((1.0 - sum(P.get(k, 0.0) for k in KEEP),
                             sum(p * _gain(row, c, tasks) for c, p in P.items() if c not in KEEP)))
                continue
            g = _gain(row, cs, tasks)
            other = sum(p * _gain(row, c, tasks) for c, p in P.items() if c not in KEEP and c != cs)
            acc.append((P.get(cs, 0.0), g, other))
    a = np.array(acc) if acc else np.zeros((0, 3))
    n = len(acc) + len(keep)
    miss = 1 - a[:, 0]
    out = {"rounds": n, "full_accepts": len(acc),
           "recall": float(a[:, 0].mean()) if len(a) else None,
           "accept_when_full_keeps": float(np.mean([k for k, _ in keep])) if keep else 0.0,
           "full_acc_gain": float(a[:, 1].mean()) if len(a) else None,
           # held-out gain of full's accepted candidates, weighted by whether the rule keeps or drops them
           "gain_kept": float(np.sum(a[:, 0] * a[:, 1]) / a[:, 0].sum()) if a[:, 0].sum() > 0 else None,
           "gain_dropped": float(np.sum(miss * a[:, 1]) / miss.sum()) if miss.sum() > 0 else None,
           "dropped": float(miss.sum()),
           # per-round contributions to D(rule) - D(full), over all rounds
           "forgone_gain": -float(np.sum(miss * np.clip(a[:, 1], 0, None))) / n,
           "avoided_harm": -float(np.sum(miss * np.clip(a[:, 1], None, 0))) / n,
           "substitute": float(np.sum(a[:, 2])) / n,
           # value of what the rule accepts in rounds where full evaluation keeps the incumbent
           "extra": float(sum(x for _, x in keep)) / n}
    out["diff"] = out["forgone_gain"] + out["avoided_harm"] + out["substitute"] + out["extra"]
    out["dropped_positive"] = float(np.sum(miss * (a[:, 1] > 0))) if len(a) else 0.0
    out["dropped_negative"] = float(np.sum(miss * (a[:, 1] < 0))) if len(a) else 0.0
    return out


def _resample(tables, block_of, rng):
    sample = []
    for T in tables:
        rows = T["rows"]
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(block_of[T["name"]][row["t"]], []).append(i)
        keys = list(groups)
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
        all_tasks = sorted({t for row in rows for d in row["dep"].values() for t in d})
        sample.append((T, idx, list(rng.choice(all_tasks, len(all_tasks)))))
    return sample


def accept(trajs: dict, js: Path | None, seed: int = 3):
    res = {}
    for set_name, names in SETS + (("all_tau2", ["r1", "r2", "r3", "r4"]),):
        if not all(n in trajs for n in names):
            continue
        tables, block_of = _load_tables(trajs, names, "e1_analysis_guard_posthoc.json")
        rng = np.random.default_rng(seed)
        boots = [_resample(tables, block_of, rng) for _ in range(A.B)]
        res[set_name] = {}
        for r in RULES:
            point = _stats(tables, r)
            bs = [_stats(tables, r, s) for s in boots]
            ci = {}
            for k in ("recall", "full_acc_gain", "gain_kept", "gain_dropped", "forgone_gain", "avoided_harm",
                      "substitute", "extra", "diff"):
                xs = [b[k] for b in bs if b[k] is not None]
                if xs:
                    ci[k] = [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
            res[set_name][r] = {**point, "ci90": ci}
            pp = lambda x: "  -  " if x is None else f"{100 * x:+.2f}"
            print(f"{set_name:9s} {r:14s} n={point['rounds']:3d} accF={point['full_accepts']:2d} "
                  f"recall {point['recall'] if point['recall'] is None else round(point['recall'], 2)} "
                  f"accKeep {point['accept_when_full_keeps']:.2f} | gain full-acc {pp(point['full_acc_gain'])} "
                  f"kept {pp(point['gain_kept'])} dropped {pp(point['gain_dropped'])} (n {point['dropped']:.1f}: "
                  f"+{point['dropped_positive']:.1f}/-{point['dropped_negative']:.1f}) | forgone "
                  f"{pp(point['forgone_gain'])} avoided {pp(point['avoided_harm'])} subst {pp(point['substitute'])}"
                  f" extra {pp(point['extra'])}"
                  f" = {pp(point['diff'])}")
    if js:
        js.write_text(json.dumps(res, indent=1))
    return res


def bias(runs: Path, out: Path, domains: str, M: int = 200):
    res = {}
    for name in domains.split(","):
        D = A.load(name, runs, out)
        z = float(D["cfg"].get("delta_z", 2.0))
        rows = {}
        for R in D["rounds"]:
            for v, C in R["cands"].items():
                pairs = A._pairs(R["inc"], C["full"])
                target = float(np.mean([p[2] - p[3] for p in pairs])) if pairs else None
                dep = A.task_diffs(C["heldout"], R["inc_heldout"], D["k_h"])
                gain = float(np.mean(list(dep.values()))) if dep else None
                if target is None or gain is None:
                    continue
                # same draws as posthoc_r7.coverage
                rng = random.Random(1000 * R["t"] + sum(map(ord, v)))
                for rule, b in (("sample", 40), ("net", 40), ("replaynull", 40), ("replay", 40),
                                ("sample", FULL_B)):
                    es = []
                    for _ in range(M if b != FULL_B else 1):
                        nv = None
                        if rule in ("net", "replaynull"):
                            nn = b // 2 if rule == "replaynull" else b // 4
                            nv = A._fix(R["null"], A._spread(R["refs"], nn, rng), rng, {})
                        e = A.estimate(rule, b, R, v, rng, None, nv, D["ep_tokens"])
                        if e is None or e[0] is None:
                            continue
                        es.append((e[0], e[1]))
                    if not es:
                        continue
                    x = np.array(es)
                    m = float(x[:, 0].mean())
                    key = "fullse" if b == FULL_B else f"{rule}@{b}"
                    rows.setdefault(key, []).append({
                        "t": R["t"], "cand": v, "target": target, "gain": gain, "mean": m,
                        "sd": float(x[:, 0].std()), "se": float(x[:, 1].mean()),
                        "cover_self": float(np.mean(np.abs(x[:, 0] - m) <= z * x[:, 1])),
                        "cover_evolve": float(np.mean(np.abs(x[:, 0] - target) <= z * x[:, 1])),
                        "cover_deploy": float(np.mean(np.abs(x[:, 0] - gain) <= z * x[:, 1]))})
        res[name] = rows
    (out / "e1_r8_bias.json").write_text(json.dumps(res, indent=1))
    for name, rows in res.items():
        for k, xs in rows.items():
            err = np.array([r["mean"] - r["target"] for r in xs])
            print(f"{name:13s} {k:15s} n={len(xs):3d} signed err {100 * err.mean():+.2f} "
                  f"|err| {100 * np.abs(err).mean():.2f} sd {100 * np.mean([r['sd'] for r in xs]):.2f} "
                  f"se {100 * np.mean([r['se'] for r in xs]):.2f} cover self/evolve/deploy "
                  f"{np.mean([r['cover_self'] for r in xs]):.2f}/{np.mean([r['cover_evolve'] for r in xs]):.2f}/"
                  f"{np.mean([r['cover_deploy'] for r in xs]):.2f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("accept", "bias", "tables"))
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--domains", default="tau2_retail,tau2_airline")
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--json")
    ap.add_argument("--bias-json", help="tables: where to write the bias summary")
    ap.add_argument("--tex", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    a = ap.parse_args()
    if a.cmd == "tables":
        trajs = {k: Path(v) for k, v in (x.split("=", 1) for x in a.traj)}
        f = lambda n: trajs[n] / "verify" / "e1_r8_bias.json"
        tables(Path(a.json), {"tau2": [f(n) for n in ("r1", "r2", "r3", "r4")], "e3": [f("e3")]},
               Path(a.tex), Path(a.bias_json) if a.bias_json else None)
    elif a.cmd == "accept":
        accept({k: v for k, v in (x.split("=", 1) for x in a.traj)}, Path(a.json) if a.json else None)
    else:
        bias(Path(a.runs), Path(a.out), a.domains)



# ---------------------------------------------------------------- tables --
NAMES = {"keep": "Keep incumbent", "seqfull": "Sequential full", "seqsample@80": "Sequential sample@80",
         "sample@80": "Sample@80", "sample@40": "Sample@40", "net@40": "Net@40",
         "replaynull@40": "Replay$-$null@40", "judge": "LLM judge", "none": "None",
         "replay@40": "Replay@40"}


def _pp(x, d=2):
    return "--" if x is None else f"{100 * x:+.{d}f}".replace("-", "$-$")


def bias_summary(files: dict, seed: int = 4):
    """files: {group: [e1_r8_bias.json, ...]} -> per group and evidence: mean signed error
    (draw mean - full evolve dS) with a 90% bootstrap interval over candidates, share of
    candidates with a positive error, mean draw SD, mean reported SE, and the coverage
    of the draws' own mean, of full's evolve dS and of the deployment gain."""
    rng = np.random.default_rng(seed)
    out = {}
    for g, fs in files.items():
        rows = {}
        for f in fs:
            for dom in json.loads(Path(f).read_text()).values():
                for k, xs in dom.items():
                    rows.setdefault(k, []).extend(xs)
        out[g] = {}
        for k, xs in rows.items():
            err = np.array([r["mean"] - r["target"] for r in xs])
            bs = [float(err[rng.integers(0, len(err), len(err))].mean()) for _ in range(2000)]
            out[g][k] = {"n": len(xs), "signed_err": float(err.mean()),
                         "signed_err_ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                         "share_positive": float(np.mean(err > 0)),
                         "draw_sd": float(np.mean([r["sd"] for r in xs])),
                         "se": float(np.mean([r["se"] for r in xs])),
                         "cover_self": float(np.mean([r["cover_self"] for r in xs])),
                         "cover_evolve": float(np.mean([r["cover_evolve"] for r in xs])),
                         "cover_deploy": float(np.mean([r["cover_deploy"] for r in xs]))}
    return out


def tables(acc: Path, bias_files: dict, out: Path, js: Path | None):
    """paper/tables/r8_accept.tex and r8_bias.tex"""
    R = json.loads(acc.read_text())
    lines = []

    def row(name, x):
        rc = "--" if x["recall"] is None else (f"{x['recall']:.2f} {{\\scriptsize [{x['ci90']['recall'][0]:.2f}, "
                                               f"{x['ci90']['recall'][1]:.2f}]}}")
        other = x["substitute"] + x["extra"]
        return " & ".join([name, rc, f"{x['accept_when_full_keeps']:.2f}", _pp(x["gain_kept"]),
                           _pp(x["gain_dropped"]), f"{x['dropped']:.1f}", _pp(x["forgone_gain"]),
                           _pp(x["avoided_harm"]), _pp(other), _pp(x["diff"])]) + " \\\\"
    for r in ("seqfull", "seqsample@80", "sample@80", "sample@40", "net@40", "replaynull@40", "judge",
              "none", "keep"):
        lines.append(row(NAMES[r], R["primary"][r]))
    lines.append("\\midrule")
    for s, lab in (("r1", "r1"), ("pooled", "r1--r3"), ("r4", "r4"), ("e3", "E3")):
        x = R[s]["seqfull"]
        lines.append(row(f"Sequential full, {lab} ({x['full_accepts']})", x))
    (out / "r8_accept.tex").write_text("\n".join(lines) + "\n")
    S = bias_summary(bias_files)
    if js:
        js.write_text(json.dumps(S, indent=1))
    lines = []
    for g, lab in (("tau2", "$\\tau^2$, r1--r4"), ("e3", "AppWorld")):
        if lines:
            lines.append("\\midrule")
        n = sorted({S[g][k]["n"] for k in ("sample@40", "net@40", "replaynull@40", "replay@40")})
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{lab}}} ({'--'.join(map(str, (n[0], n[-1]) if n[0] != n[-1] else n[:1]))} candidates)}} \\\\")
        for k in ("sample@40", "net@40", "replaynull@40", "replay@40"):
            x = S[g][k]
            lines.append(" & ".join([NAMES[k], f"{_pp(x['signed_err'], 1)} {{\\scriptsize [{_pp(x['signed_err_ci90'][0], 1)}, "
                                     f"{_pp(x['signed_err_ci90'][1], 1)}]}}", f"{x['share_positive']:.2f}",
                                     f"{100 * x['draw_sd']:.1f} / {100 * x['se']:.1f}", f"{x['cover_self']:.2f}"])
                         + " \\\\")
    (out / "r8_bias.tex").write_text("\n".join(lines) + "\n")
    for f in ("r8_accept", "r8_bias"):
        print(f"== {f}\n" + (out / f"{f}.tex").read_text())
    for g, v in S.items():
        for k, x in v.items():
            print(g, k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in x.items()})


if __name__ == "__main__":
    main()
