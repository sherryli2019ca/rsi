"""Paper table of Study 2 with E1R (verify/PREREGISTRATION_E1R.md).

  python -m verify.report_e1r --traj r1=/home/user/e1/runs r2=... r3=... [--out paper/tables]

Writes e1r_rules.tex: per rule (the rows of verify/report_e1.py), the decision
value per round under the common error check, with a 90% interval from the
block bootstrap of verify/e1r.py (blocks of rounds sharing an incumbent, and
held-out tasks, within domain and trajectory), on the first trajectory (r1),
on the registered primary set (r2 and r3) and on all three pooled; then, on the
primary set, how often the rule accepts a candidate, its evidence cost per
round (episodes and episode equivalents) and Net(N) at N = 10^4, v = 10.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", required=True, help="name=runs directory")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    args = ap.parse_args()
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
