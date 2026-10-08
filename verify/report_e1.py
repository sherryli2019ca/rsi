"""Paper table of Study 2 (experiment E1) from verify/analyze.py's output.

  python -m verify.report_e1 --analysis runs/verify/e1_analysis.json \
      [--guard runs/verify/e1_analysis_guard_posthoc.json] [--out paper/tables]

Writes e1_rules.tex: per rule, the decision value per round pooled over both
domains with its 90% bootstrap interval, as registered and (with --guard) with
RRSI's harness-error check applied to every rule's evidence episodes (post hoc,
verify/posthoc_e1.py guard), with the post-hoc sequential rules; then, from the guarded analysis when given, how
often the rule accepts a candidate, its evidence cost per round (episodes and
episode equivalents) and Net(N) at N = 10^4, v = 10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [("none", "None", "0"), ("judge", "LLM judge", "0"),
        ("nonecheck@10", "None + check", "10"), ("judgecheck@10", "Judge + check", "10"),
        ("full", "Full (RRSI)", "all"),
        ("seqfull", "Sequential full", "$\\le$all"), ("seqsample@80", "Sequential sample", "$\\le$80")] + [
    (f"{r}@{b}", name, str(b)) for b in (10, 40, 80)
    for r, name in (("sample", "Sample"), ("replay", "Replay"), ("replaynull", "Replay$-$null"),
                    ("net", "Net"))]
MIDRULES = (2, 4, 5, 7, 11, 15)


def accept_rate(rows, rule):
    return float(np.mean([1 - sum(p for c, p in row["probs"][rule].items() if c in (None, "null"))
                          for row in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--guard")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    args = ap.parse_args()
    reg = json.loads(Path(args.analysis).read_text())
    gua = json.loads(Path(args.guard).read_text()) if args.guard else reg
    rows_all = [row for T in gua["tables"] for row in T["rows"]]
    pp = lambda x: f"{100 * x:+.2f}".replace("-", "$-$")
    bold = lambda x, best: "\\textbf{" + pp(x) + "}" if x >= best else pp(x)

    def dv(r, k):
        if k not in r["decision_value"]:
            return "--", "--"
        lo, hi = r["decision_value_ci90"][k]
        best = max(r["decision_value"][kk] for kk, _, _ in ROWS if kk in r["decision_value"])
        return bold(r["decision_value"][k], best), f"[{pp(lo)}, {pp(hi)}]"

    lines = []
    for i, (k, name, b) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        c = gua["cost_per_round"][k]
        net = f"{gua['net_N'][k]['N=10000,v=10']:,.0f}".replace("-", "$-$").replace(",", "{,}")
        cells = [name, b, *dv(reg, k)]
        if args.guard:
            cells += list(dv(gua, k))
        cells += [f"{accept_rate(rows_all, k):.2f}", f"{c['episodes']:.0f} / {c['episode_equivalents']:.0f}", net]
        lines.append(" & ".join(cells) + " \\\\")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_rules.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
