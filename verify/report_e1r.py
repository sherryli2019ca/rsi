"""Paper table of Study 2 with E1R (verify/PREREGISTRATION_E1R.md).

  python -m verify.report_e1r --traj r1=/home/user/e1/runs r2=... r3=... [--out paper/tables]

Writes e1r_rules.tex: per rule (the rows of verify/report_e1.py), the decision
value per round under the common error check, with a 90% interval from the
block bootstrap of verify/e1r.py (blocks of rounds sharing an incumbent, and
held-out tasks, within domain and trajectory), on the first trajectory (r1),
on the registered primary set (r2 and r3) and on all three pooled; then, on the
primary set, how often the rule accepts a candidate, its evidence cost per
round (episodes and episode equivalents) and Net(N) at N = 10^4, v = 10.

  python -m verify.report_e1r --e3-report /home/user/e3/runs/verify/e1r_report.json

writes e3_rules.tex instead, from the E3 report of verify/e1r.py: per rule, the
decision value with its block-bootstrap interval, episodes per round, and the
non-inferiority contrast with full evaluation (difference, one-sided p, Holm).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import analyze as A
from .e1r import _tables, _trajs
from .posthoc_e1 import _dv, _per_round
from .report_e1 import MIDRULES, ROWS, accept_rate

SETS = (("r1", ["r1"]), ("primary", ["r2", "r3"]), ("pooled", ["r1", "r2", "r3"]))


def block_dv(tables, block_of, seed: int = 2):
    point = _dv(_per_round(tables))
    rng = np.random.default_rng(seed)
    boots = [_dv(_per_round(tables, rng, block_of)) for _ in range(A.B)]
    return {k: (point[k], float(np.percentile([b[k] for b in boots], 5)),
                float(np.percentile([b[k] for b in boots], 95))) for k in point}


E3_ROWS = [("full", "Full (RRSI)"), ("seqfull", "Sequential full"), ("seqsample@80", "Sequential sample@80"),
           ("sample@40", "Sample@40"), ("net@40", "Net@40"), ("replaynull@40", "Replay$-$null@40"),
           ("sample@80", "Sample@80"), ("nonecheck@10", "None + check"), ("judgecheck@10", "Judge + check")]


def e3_table(report: Path, out: Path):
    import json
    R = json.loads(report.read_text())["primary"]
    pp = lambda x: f"{100 * x:+.2f}".replace("-", "$-$")
    ni = {**R["ni_primary"], **R["ni_sequential"]}
    lines = []
    for k, name in E3_ROWS:
        d = R["decision_value"][k]
        cells = [name, f"{pp(d['est'])} {{\\scriptsize [{pp(d['ci90_block'][0])}, {pp(d['ci90_block'][1])}]}}",
                 f"{d['episodes_per_round']:.0f}"]
        if k in ni:
            c = ni[k]
            p = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
            cells += [f"{pp(c['est'])} {{\\scriptsize [{pp(c['ci90_block'][0])}, {pp(c['ci90_block'][1])}]}}",
                      f"{p(c['p_ni'])} ({p(c['p_holm'])})"]
        else:
            cells += ["--", "--"]
        lines.append(" & ".join(cells) + " \\\\")
        if k in ("full", "seqsample@80", "replaynull@40", "sample@80"):
            lines.append("\\midrule")
    out.mkdir(parents=True, exist_ok=True)
    (out / "e3_rules.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", help="name=runs directory")
    ap.add_argument("--e3-report", default=None)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    args = ap.parse_args()
    if args.e3_report:
        e3_table(Path(args.e3_report), Path(args.out))
        return
    trajs = _trajs(args.traj)
    res = {name: block_dv(*_tables(trajs, names)) for name, names in SETS}
    prim, _ = _tables(trajs, ["r2", "r3"])
    rows_all = [row for T in prim for row in T["rows"]]
    pr = _per_round(prim)
    pp = lambda x: f"{100 * x:+.2f}".replace("-", "$-$")

    def cell(name, k):
        if k not in res[name]:
            return "--"
        est, lo, hi = res[name][k]
        best = max(res[name][kk][0] for kk, _, _ in ROWS if kk in res[name])
        e = "\\textbf{" + pp(est) + "}" if est >= best else pp(est)
        return f"{e} {{\\scriptsize [{pp(lo)}, {pp(hi)}]}}"

    lines = []
    for i, (k, name, b) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        ep = np.mean([row["cost"][k]["episodes"] for row in rows_all])
        eq = np.mean([row["cost"][k]["episode_equivalents"] for row in rows_all])
        x = pr[k]
        net = 1e4 * (10 * np.mean([a for a, _, _ in x]) - np.mean([d for _, d, _ in x])) - np.mean([c for _, _, c in x])
        cells = [name, b] + [cell(n, k) for n, _ in SETS] + [
            f"{accept_rate(rows_all, k):.2f}", f"{ep:.0f} / {eq:.0f}",
            f"{net:,.0f}".replace("-", "$-$").replace(",", "{,}")]
        lines.append(" & ".join(cells) + " \\\\")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1r_rules.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
