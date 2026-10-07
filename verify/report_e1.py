"""Paper table of Study 2 (experiment E1) from verify/analyze.py's output.

  python -m verify.report_e1 --analysis runs/verify/e1_analysis.json [--out paper/tables]

Writes e1_rules.tex: per rule, the decision value per round pooled over both
domains with its 90% bootstrap interval, the per-domain values, how often the
rule accepts a candidate, its evidence cost per round (episodes and episode
equivalents) and Net(N) at N = 10^4, v = 10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROWS = [("none", "None", "0"), ("judge", "LLM judge", "0"), ("full", "Full (RRSI)", "all")] + [
    (f"{r}@{b}", name, str(b)) for b in (10, 40, 80)
    for r, name in (("sample", "Sample"), ("replay", "Replay"), ("replaynull", "Replay$-$null"),
                    ("net", "Net"))]


def round_value(row, rule):
    return sum(p * float(np.mean(list(row["dep"][c].values())))
               for c, p in row["probs"][rule].items() if c != "null" and row["dep"].get(c))


def accept_rate(rows, rule):
    return float(np.mean([1 - row["probs"][rule].get("null", 0.0) for row in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    args = ap.parse_args()
    r = json.loads(Path(args.analysis).read_text())
    tabs = {T["name"]: T["rows"] for T in r["tables"]}
    pp = lambda x: f"{100 * x:+.2f}".replace("-", "$-$")
    lines = []
    best = max(r["decision_value"][k] for k, _, _ in ROWS)
    for i, (k, name, b) in enumerate(ROWS):
        if i in (3, 7, 11):
            lines.append("\\midrule")
        dv = r["decision_value"][k]
        lo, hi = r["decision_value_ci90"][k]
        dom = [float(np.mean([round_value(row, k) for row in tabs[d]])) for d in ("tau2_retail", "tau2_airline")]
        acc = float(np.mean([accept_rate(tabs[d], k) for d in tabs]))
        c = r["cost_per_round"][k]
        net = r["net_N"][k]["N=10000,v=10"]
        val = pp(dv) if dv < best else "\\textbf{" + pp(dv) + "}"
        net_s = f"{net:,.0f}".replace("-", "$-$").replace(",", "{,}")
        lines.append(f"{name} & {b} & {val} & [{pp(lo)}, {pp(hi)}] & {pp(dom[0])} & {pp(dom[1])} & "
                     f"{acc:.2f} & {c['episodes']:.0f} / {c['episode_equivalents']:.0f} & {net_s} \\\\")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e1_rules.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
