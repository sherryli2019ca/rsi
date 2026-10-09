"""Post-hoc analyses for the seventh review of paper 1 (not registered).

  python -m verify.posthoc_r7 ablate --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r7 coverage --runs ... --out ... --domains ...
  python -m verify.posthoc_r7 report --traj r1=DIR r2=DIR ... [--json FILE]

ablate: the guarded analysis of verify/posthoc_e1.py (common error check) twice
  more, to separate the evidence from the rule that turns it into a decision.
  Both runs add `fullse`, RRSI's full evolve evaluation used as an estimator
  with its own standard error (sample@b with b = every pair), and `seqfullse`,
  its sequential version (seqsample with the full plan), so that full
  evaluation is decided like every other evidence instead of by RRSI's
  calibrated noise band.
    own:    every estimate through RRSI's Algorithm 2 with the evidence's own
            band z * se (as registered), including the within-band branch that
            admits a candidate on cost and novelty.
    strict: one common criterion for all evidence: admit only if dS > z * se and
            the cost condition holds; no within-band admission (no novelty).
  Writes <out>/e1_r7_own.json and <out>/e1_r7_strict.json.

coverage: for every measured candidate and evidence type at budget 40 (and
  fullse), the share of M evidence draws whose interval dS +- z * se covers
  (a) the candidate's full-evaluation dS on the evolve set (the quantity the
  evidence estimates) and (b) its held-out deployment gain. Writes
  <out>/e1_r7_coverage.json.

report: decision values with the block bootstrap (blocks of rounds sharing an
  incumbent, and held-out tasks), per trajectory set: non-inferiority to full
  evaluation at margins 0.1, 0.25, 0.5 and 0.75 points per round (one-sided
  bootstrap p), keeping the incumbent as a zero-cost rule, the share of rounds
  in which each rule makes full evaluation's choice, and the ablations above.

costshare: the dollar cost of the RRSI loop itself per trajectory (model calls
  by role, candidate evaluations, smoke tests, calibration) and the share of it
  sequential full evaluation would have saved. Writes --json.

tables: the appendix tables of the review-7 analyses from the report and
  costshare outputs and each trajectory's coverage file (paper/tables/r7_*.tex):
  python -m verify.posthoc_r7 tables --traj r1=... r2=... r3=... r4=... e3=...
      --json report.json --cost costshare.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from . import analyze as A
from . import posthoc_e1 as P
from .state import rounds

KEEP_RULES = {"none", "judge", "full", "sample@40", "sample@80", "net@40", "replaynull@40",
              "seqfull", "seqsample@80"}
FULL_B = 10 ** 6
MARGINS = (0.001, 0.0025, 0.005, 0.0075)
SETS = (("r1", ["r1"]), ("primary", ["r2", "r3"]), ("pooled", ["r1", "r2", "r3"]),
        ("r4", ["r4"]), ("e3", ["e3"]))


def decide_strict(ests: dict, cfg: dict, z: float):
    """One criterion for all evidence: dS > z * se and the cost condition."""
    best = None
    for v, (dS, se, dC, _nu) in ests.items():
        if dS is None or dS <= z * (se or 0.0):
            continue
        if (dC or 0.0) <= cfg["beta0"] + cfg["beta1"] * dS and (best is None or dS > ests[best][0]):
            best = v
    return best


def _hook(variant: str):
    def hook(mod):
        rules, label = mod.rules, mod.label

        def rules_():
            out = [r for r in rules() if label(*r) in KEEP_RULES]
            return out + [("sample", FULL_B, None), ("seqsample", FULL_B, None)]

        def label_(rule, b, p):
            if b == FULL_B:
                return {"sample": "fullse", "seqsample": "seqfullse"}[rule]
            return label(rule, b, p)
        mod.rules, mod.label = rules_, label_
        if variant == "strict":
            mod.decide = decide_strict
    return hook


def ablate(runs: Path, out: Path, domains: str):
    orig = A.decide
    for variant in ("own", "strict"):
        try:
            P._guard(runs, out, domains=domains, dest=f"e1_r7_{variant}.json", hook=_hook(variant))
        finally:
            A.decide = orig


def coverage(runs: Path, out: Path, domains: str, M: int = 200):
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
                rng = random.Random(1000 * R["t"] + sum(map(ord, v)))
                for rule, b in (("sample", 40), ("net", 40), ("replaynull", 40), ("replay", 40),
                                ("sample", FULL_B)):
                    hit_e, hit_d, n = 0, 0, 0
                    for _ in range(M if b != FULL_B else 1):
                        nv = None
                        if rule in ("net", "replaynull"):
                            nn = b // 2 if rule == "replaynull" else b // 4
                            nv = A._fix(R["null"], A._spread(R["refs"], nn, rng), rng, {})
                        e = A.estimate(rule, b, R, v, rng, None, nv, D["ep_tokens"])
                        if e is None or e[0] is None:
                            continue
                        lo, hi = e[0] - z * e[1], e[0] + z * e[1]
                        hit_e += lo <= target <= hi
                        hit_d += lo <= gain <= hi
                        n += 1
                    if n:
                        key = "fullse" if b == FULL_B else f"{rule}@{b}"
                        rows.setdefault(key, []).append((hit_e / n, hit_d / n))
                j = C["judge"]
                if j is not None:
                    rows.setdefault("judge_sign", []).append((float(np.sign(j) == np.sign(target)),
                                                              float(np.sign(j) == np.sign(gain))))
        res[name] = {k: {"cover_evolve": float(np.mean([a for a, _ in x])),
                         "cover_deploy": float(np.mean([b for _, b in x])), "n_candidates": len(x)}
                     for k, x in rows.items()}
    (out / "e1_r7_coverage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------- cost share --
def costshare(trajs: dict, js: Path | None):
    """Dollar cost of the RRSI loop itself (not of this study's extra evidence or
    held-out deployments) per trajectory and domain: proposal-side model calls
    by role (the judge is study evidence and is left out), candidate full
    evaluations, smoke tests and the calibration runs of the base harness; and
    what sequential full evaluation would have saved, priced at the mean
    episode price of the candidates' evaluations."""
    pi, pc, po = P.PRICES["deepseek-v4-pro"]
    res = {}
    for n, root in trajs.items():
        rr = Path(root) / "rrsi"
        for usage in sorted(rr.glob("*.usage.jsonl")):
            D = usage.name.split(".")[0]
            llm = {}
            for line in usage.read_text().splitlines():
                if not line.strip():
                    continue
                c = json.loads(line)
                if c.get("role") == "judge":
                    continue
                pr = P.PRICES.get(c.get("model"), (pi, pc, po))
                llm[c.get("role")] = llm.get(c.get("role"), 0.0) + (
                    c.get("in", 0) * pr[0] + c.get("cache_read", 0) * pr[1] + c.get("out", 0) * pr[2]) / 1e6
            ep = {"candidate_eval": [], "smoke": [], "calibration": []}
            for job in (rr / D / "jobs").iterdir():
                kind = ("smoke" if job.name.endswith("_smoke") else
                        "candidate_eval" if job.name.startswith("r") else "calibration")
                for f in job.glob("s[0-9]*/*.json"):
                    ep[kind].append(P._episode_dollars(json.loads(f.read_text()).get("tokens") or {}))
            src = Path(root) / "verify" / "e1_analysis_guard_posthoc.json"
            T = [x for x in json.loads(src.read_text())["tables"] if x["name"] == D]
            saved_eps = sum(r["cost"]["full"]["episodes"] - r["cost"]["seqfull"]["episodes"]
                            for r in T[0]["rows"]) if T else 0.0
            price = float(np.mean(ep["candidate_eval"])) if ep["candidate_eval"] else 0.0
            usd = {"llm_" + k: v for k, v in llm.items()}
            usd.update({k: float(np.sum(v)) for k, v in ep.items()})
            total = sum(usd.values())
            res[f"{n}/{D}"] = {"usd": usd, "total_usd": total,
                               "episodes": {k: len(v) for k, v in ep.items()},
                               "share_candidate_eval": usd["candidate_eval"] / total,
                               "share_llm": sum(v for k, v in usd.items() if k.startswith("llm_")) / total,
                               "seqfull_saved_episodes": saved_eps,
                               "seqfull_saved_usd": saved_eps * price,
                               "seqfull_saved_share_of_total": saved_eps * price / total}
    if js:
        js.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, f"total ${v['total_usd']:.2f}  cand-eval {100 * v['share_candidate_eval']:.0f}%  "
                 f"llm {100 * v['share_llm']:.0f}%  seqfull saves {100 * v['seqfull_saved_share_of_total']:.0f}% "
                 f"({v['seqfull_saved_episodes']:.0f} ep)  ", {a: round(b, 2) for a, b in v["usd"].items()})
    return res


# ---------------------------------------------------------------- report --
def _load_tables(trajs: dict, names, src: str):
    tables, block_of = [], {}
    for n in names:
        P_ = Path(trajs[n])
        f = P_ / "verify" / src
        if not f.exists():
            return None, None
        for T in json.loads(f.read_text())["tables"]:
            key = f"{n}/{T['name']}"
            tables.append({**T, "name": key})
            block_of[key] = {r.t: r.inc_commit for r in rounds(P_ / "rrsi" / T["name"])}
    return tables, block_of


def _agree(tables, rule, ref="full"):
    ag = []
    for T in tables:
        for r in T["rows"]:
            f = r["probs"][ref]
            fc = max(f, key=f.get)
            ag.append(r["probs"][rule].get(fc, 0.0))
    return float(np.mean(ag))


def _accept(tables, rule):
    return float(np.mean([1 - r["probs"][rule].get("null", 0.0) - r["probs"][rule].get(None, 0.0)
                          for T in tables for r in T["rows"]]))


def _episodes(tables, rule):
    return float(np.mean([r["cost"][rule]["episodes"] for T in tables for r in T["rows"]]))


def summarise(tables, block_of, ref="full", seed=2):
    pr0 = P._per_round(tables)
    point = P._dv(pr0)
    rng = np.random.default_rng(seed)
    prs = [P._per_round(tables, rng, block_of) for _ in range(A.B)]
    boots = [P._dv(x) for x in prs]

    def net(x, v=10, N=1e4):
        """Net(N) against keeping the incumbent (Eq. net), no change cost."""
        return N * (v * np.mean([a for a, _, _ in x]) - np.mean([d for _, d, _ in x])) - np.mean([c for _, _, c in x])
    nets = {r: [net(pr0[r])] + [float(np.percentile([net(x[r]) for x in prs], q)) for q in (5, 95)]
            for r in pr0}
    point["keep"] = 0.0
    for b in boots:
        b["keep"] = 0.0
    pc = lambda xs: [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    out = {}
    for r in point:
        row = {"dv": [point[r]] + pc([b[r] for b in boots])}
        if r != "keep":
            row["net_1e4_v10"] = nets[r]
            row["episodes"] = _episodes(tables, r)
            row["accept_rate"] = _accept(tables, r)
            row["agree_with_" + ref] = _agree(tables, r, ref)
        if r != ref:
            xs = np.array([b[r] - b[ref] for b in boots])
            row["diff_vs_" + ref] = [point[r] - point[ref]] + pc(xs)
            row["p_ni"] = {f"{100 * m:g}": float(np.mean(xs <= -m)) for m in MARGINS}
        out[r] = row
    return out


def report(trajs: dict, js: Path | None):
    res = {}
    for set_name, names in SETS:
        if not all(n in trajs for n in names):
            continue
        res[set_name] = {}
        for src, ref in (("e1_analysis_guard_posthoc.json", "full"), ("e1_r7_own.json", "fullse"),
                         ("e1_r7_strict.json", "fullse")):
            tables, block_of = _load_tables(trajs, names, src)
            if tables is None:
                continue
            res[set_name][src.replace(".json", "")] = summarise(tables, block_of, ref)
    text = json.dumps(res, indent=1)
    if js:
        js.write_text(text)
    for s, v in res.items():
        for src, rows in v.items():
            print(f"== {s} {src}")
            for r, x in sorted(rows.items(), key=lambda kv: -kv[1]["dv"][0]):
                d = x.get([k for k in x if k.startswith("diff_vs")][0]) if any(k.startswith("diff_vs") for k in x) else None
                pp = lambda a: f"{100 * a:+.2f}"
                print(f"{r:16s} D {pp(x['dv'][0])} [{pp(x['dv'][1])},{pp(x['dv'][2])}]"
                      + (f"  diff {pp(d[0])} [{pp(d[1])},{pp(d[2])}]" if d else "")
                      + (f"  pNI {x['p_ni']}" if "p_ni" in x else "")
                      + (f"  agree {[v_ for k, v_ in x.items() if k.startswith('agree')][0]:.2f}"
                         f" acc {x['accept_rate']:.2f} ep {x['episodes']:.0f}" if "episodes" in x else ""))
    return res


# ---------------------------------------------------------------- tables --
NAMES = {"keep": "Keep incumbent", "full": "Full (RRSI)", "fullse": "Full, own band",
         "seqfull": "Sequential full", "seqfullse": "Sequential full, own band",
         "seqsample@80": "Sequential sample@80", "sample@80": "Sample@80", "sample@40": "Sample@40",
         "net@40": "Net@40", "replaynull@40": "Replay$-$null@40", "judge": "LLM judge", "none": "None"}


def _pp(x):
    return f"{100 * x:+.2f}".replace("-", "$-$")


def _ci(v):
    return f"{_pp(v[0])} {{\\scriptsize [{_pp(v[1])}, {_pp(v[2])}]}}"


def _p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def tables(report: Path, cover: dict, cost: Path, out: Path):
    """paper/tables/r7_margins.tex, r7_ablation.tex, r7_coverage.tex, r7_cost.tex"""
    R = json.loads(report.read_text())
    out.mkdir(parents=True, exist_ok=True)
    g = {s: R[s]["e1_analysis_guard_posthoc"] for s in ("primary", "r1", "r4", "e3")}
    lines = []
    for r in ("seqfull", "seqsample@80", "sample@80", "sample@40", "net@40", "replaynull@40", "keep"):
        x = g["primary"][r]
        cells = [NAMES[r], _ci(x["diff_vs_full"])] + [_p(x["p_ni"][m]) for m in ("0.1", "0.25", "0.5", "0.75")]
        for s in ("primary", "r1", "r4", "e3"):
            y = g[s][r]
            a = y.get("agree_with_full")
            if r == "keep":
                a = 1 - g[s]["full"]["accept_rate"]
            cells.append(f"{a:.2f}")
        cells += [_p(g[s][r]["p_ni"]["0.25"]) for s in ("r1", "r4", "e3")]
        lines.append(" & ".join(cells) + " \\\\")
    (out / "r7_margins.tex").write_text("\n".join(lines) + "\n")
    P2 = R["primary"]
    lines = []
    for r in ("full", "seqfull", "seqsample@80", "sample@80", "sample@40", "net@40", "replaynull@40",
              "judge", "none"):
        reg = P2["e1_analysis_guard_posthoc"][r]
        r_own = {"full": "fullse", "seqfull": "seqfullse"}.get(r, r)
        own, st = P2["e1_r7_own"][r_own], P2["e1_r7_strict"][r_own]
        name = NAMES[r] + (" / own band" if r in ("full", "seqfull") else "")
        lines.append(" & ".join([name, _ci(reg["dv"]), f"{reg['accept_rate']:.2f}", _ci(own["dv"]),
                                 f"{own['accept_rate']:.2f}", _ci(st["dv"]), f"{st['accept_rate']:.2f}"])
                     + " \\\\")
    (out / "r7_ablation.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for k, name in (("sample@40", "Sample@40"), ("net@40", "Net@40"), ("replaynull@40", "Replay$-$null@40"),
                    ("replay@40", "Replay@40"), ("fullse", "Full evaluation")):
        cells = [name]
        for grp in ("r1", "r2r3", "r4", "e3"):
            rows = cover[grp]
            xs = [(v[k]["cover_evolve"], v[k]["cover_deploy"], v[k]["n_candidates"]) for v in rows if k in v]
            n = sum(c for _, _, c in xs)
            ce = sum(a * c for a, _, c in xs) / n
            cd = sum(b * c for _, b, c in xs) / n
            cells.append(f"{ce:.2f} / {cd:.2f}")
        lines.append(" & ".join(cells) + " \\\\")
    (out / "r7_coverage.tex").write_text("\n".join(lines) + "\n")
    C = json.loads(cost.read_text())
    lines = []
    names = {"tau2_retail": "retail", "tau2_airline": "airline", "appworld": "AppWorld"}
    for key, v in C.items():
        t, D = key.split("/")
        if t == "r4" and D == "tau2_retail":
            continue            # stopped after two rounds
        llm = sum(x for k, x in v["usd"].items() if k.startswith("llm_"))
        lines.append(" & ".join([f"{t.upper() if t == 'e3' else t} {names[D]}", f"{v['total_usd']:.2f}",
                                 f"{100 * llm / v['total_usd']:.0f}",
                                 f"{100 * v['share_candidate_eval']:.0f}",
                                 f"{100 * (v['usd']['smoke'] + v['usd']['calibration']) / v['total_usd']:.0f}",
                                 f"{v['seqfull_saved_episodes']:,.0f}".replace(",", "{,}"),
                                 f"{100 * v['seqfull_saved_share_of_total']:.0f}"]) + " \\\\")
    (out / "r7_cost.tex").write_text("\n".join(lines) + "\n")
    for f in ("r7_margins", "r7_ablation", "r7_coverage", "r7_cost"):
        print(f"== {f}\n" + (out / f"{f}.tex").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("ablate", "coverage", "report", "costshare", "tables"))
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--domains", default="tau2_retail,tau2_airline")
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--json")
    ap.add_argument("--cost", help="tables: costshare output")
    ap.add_argument("--tex", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    a = ap.parse_args()
    if a.cmd == "ablate":
        ablate(Path(a.runs), Path(a.out), a.domains)
    elif a.cmd == "coverage":
        coverage(Path(a.runs), Path(a.out), a.domains)
    elif a.cmd == "tables":
        trajs = {k: Path(v) for k, v in (x.split("=", 1) for x in a.traj)}
        cov = lambda n: list(json.loads((trajs[n] / "verify" / "e1_r7_coverage.json").read_text()).values())
        cover = {"r1": cov("r1"), "r2r3": cov("r2") + cov("r3"), "r4": cov("r4"), "e3": cov("e3")}
        tables(Path(a.json), cover, Path(a.cost), Path(a.tex))
    elif a.cmd == "costshare":
        costshare({k: v for k, v in (x.split("=", 1) for x in a.traj)}, Path(a.json) if a.json else None)
    else:
        report({k: v for k, v in (x.split("=", 1) for x in a.traj)}, Path(a.json) if a.json else None)


if __name__ == "__main__":
    main()
