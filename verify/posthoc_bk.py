"""Experiment BK: sensitivity of the registered contrasts (post hoc, not registered).

  python -m verify.posthoc_bk [--json results/bk/analysis.json] [--out results/bk/posthoc.json]
                              [--tex paper/tables/bk_main.tex]

The registered analysis (`verify/bk.py analyze`) reads each of its five
contrasts by its own 90% interval, with no adjustment for multiplicity. This
script reports, from the same lineage model (estimate, standard error,
Satterthwaite df in analysis.json):
- 95% t intervals;
- one-sided p values in the direction of each registered reading (s - f below
  0, c - f below 0, c - s above 0, admission below 0, history below 0), Holm-
  adjusted over the five, and Bonferroni-adjusted intervals (two-sided
  1 - 0.10/5, so that each one-sided reading holds at family level 0.05);
- the registered bootstrap over blocks and held-out tasks (same seed, B = 4000),
  with 95% percentile intervals beside its 90% ones (checked against the
  registered output).
It also writes the main text's table: transfer by block and arm, each arm's
candidate-evaluation episodes and loop cost (as if run alone) and their change
against full evaluation, and the contrasts with 90% and 95% intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as tdist

from . import bk

READING = {"seq_minus_full": -1, "cost_minus_full": -1, "cost_minus_seq": +1, "admission": -1, "history": -1}


def _boot(B: int = 4000, seed: int = 13) -> dict:
    """bk.cmd_analyze's bootstrap, unchanged, returning the draws."""
    ho = {(b, a): bk._ho_tasks(b, a) for b in bk.BLOCKS for a in bk.ARMS}
    tasks = sorted(next(iter(ho.values()))[1])

    def y(b, a, ts):
        f, base = ho[(b, a)]
        return float(np.mean([f[t] - base[t] for t in ts]))
    rng = np.random.default_rng(seed)
    contrasts = {k: c for k, c in bk.CONTRASTS.items() if all(a in bk.ARMS for a in c)}
    boots = {k: [] for k in contrasts}
    for _ in range(B):
        bs = list(rng.choice(bk.BLOCKS, len(bk.BLOCKS)))
        ts = list(rng.choice(tasks, len(tasks)))
        m = {a: float(np.mean([y(b, a, ts) for b in bs])) for a in bk.ARMS}
        for k, c in contrasts.items():
            boots[k].append(bk._contrast(m, c))
    return boots


def analyse(res: dict) -> dict:
    out = {"contrasts": {}}
    C = res["contrasts"]
    p1 = {}
    for k, x in C.items():
        e, se, df = x["est"], x["se"], x["df"]
        q95 = float(tdist.ppf(0.975, df))
        row = {"est": e, "ci90_t": x["ci90_t"], "ci95_t": [e - q95 * se, e + q95 * se]}
        if k in READING:
            p1[k] = float(tdist.cdf(e / se, df)) if READING[k] < 0 else float(tdist.sf(e / se, df))
            qb = float(tdist.ppf(1 - 0.10 / (2 * len(READING)), df))
            row["p_one_sided_reading"] = p1[k]
            row["ci_bonferroni"] = [e - qb * se, e + qb * se]
        out["contrasts"][k] = row
    order = sorted(p1, key=p1.get)
    m, run = len(order), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p1[k]))
        out["contrasts"][k]["p_holm"] = run
    boots = _boot()
    for k, xs in boots.items():
        r = out["contrasts"][k]
        r["ci90_boot"] = [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
        r["ci95_boot"] = [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
        reg = C[k]["ci90_boot"]
        r["boot_matches_registered"] = bool(np.allclose(r["ci90_boot"], reg, atol=1e-9))
    return out


NAMES = {"f": ("Full", "full", "full"), "h": ("History only", "full", "early"),
         "c": ("Cost-aware seq.", "cost-aware", "early"), "a": ("Admission only", "seq.", "full"),
         "s": ("Sequential", "seq.", "early")}
LAB = {"seq_minus_full": "Sequential $-$ full", "cost_minus_full": "Cost-aware $-$ full",
       "cost_minus_seq": "Cost-aware $-$ sequential", "admission": "Admission (main effect)",
       "history": "History (main effect)", "interaction": "Interaction"}


def table(res: dict, post: dict) -> str:
    pp = lambda x: f"{100 * x:+.1f}"
    nb = len(bk.BLOCKS)
    loops = {a: [res["loops"][f"{b}_{a}"] for b in bk.BLOCKS] for a in NAMES}
    ep = {a: np.mean([x["cost"]["eval_episodes"] for x in L]) for a, L in loops.items()}
    usd = {a: np.mean([x["cost"]["total"] for x in L]) for a, L in loops.items()}
    rows = []
    for a, (n, adm, hist) in NAMES.items():
        rel = lambda v, ref, d: ("" if a == "f" else
                                 " ($" + f"{100 * (v / ref - 1):+.{d}f}" + "\\%$)")
        rows.append(f"{n} & {adm} & {hist} & "
                    + " & ".join(f"${pp(res['transfer'][f'{b}_{a}'])}$" for b in bk.BLOCKS)
                    + f" & ${pp(res['arm_means'][a])}$ & {sum(x['accepted'] for x in loops[a])}"
                    + f" & {ep[a]:.0f}{rel(ep[a], ep['f'], 0)} & {usd[a]:.2f}{rel(usd[a], usd['f'], 1)} \\\\")
    rows.append("\\midrule")
    rows.append(f"Contrast & & & \\multicolumn{{{nb}}}{{l}}{{90\\% / 95\\% interval}} & Est. & "
                "\\multicolumn{3}{l}{Holm $p$ (one-sided)} \\\\")
    for k, x in post["contrasts"].items():
        a90, a95 = x["ci90_t"], x["ci95_t"]
        hp = f"{x['p_holm']:.2f}" if "p_holm" in x else "--"
        rows.append(f"\\multicolumn{{3}}{{l}}{{{LAB[k]}}} & \\multicolumn{{{nb}}}{{l}}"
                    f"{{$[{pp(a90[0])},{pp(a90[1])}]$ / $[{pp(a95[0])},{pp(a95[1])}]$}} & ${pp(x['est'])}$ & "
                    f"\\multicolumn{{3}}{{l}}{{{hp}}} \\\\")
    return ("\\begin{tabular}{lll" + "r" * (nb + 2) + "rr}\n\\toprule\n"
            "Arm & Admission & History & " + " & ".join(b.upper() for b in bk.BLOCKS)
            + " & Mean & Acc. & Cand.\\ ep. & Cost (\\$) \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/bk/analysis.json")
    ap.add_argument("--out", default="results/bk/posthoc.json")
    ap.add_argument("--tex", default="paper/tables/bk_main.tex")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())
    post = analyse(res)
    Path(a.out).write_text(json.dumps(post, indent=1))
    tex = table(res, post)
    Path(a.tex).write_text(tex)
    print(tex)
    pp = lambda x: f"{100 * x:+.2f}"
    for k, x in post["contrasts"].items():
        print(f"{k:16s} {pp(x['est'])}  90% t [{pp(x['ci90_t'][0])},{pp(x['ci90_t'][1])}]"
              f"  95% t [{pp(x['ci95_t'][0])},{pp(x['ci95_t'][1])}]"
              f"  95% boot [{pp(x['ci95_boot'][0])},{pp(x['ci95_boot'][1])}]"
              + (f"  p1 {x['p_one_sided_reading']:.3f} Holm {x['p_holm']:.2f}"
                 f"  Bonf [{pp(x['ci_bonferroni'][0])},{pp(x['ci_bonferroni'][1])}]" if "p_holm" in x else "")
              + f"  boot90 matches registered: {x['boot_matches_registered']}")


if __name__ == "__main__":
    main()
