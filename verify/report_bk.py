"""Experiment BK: the descriptive table of the appendix (paper/tables/bk_loops.tex)
from the registered analysis (results/bk/analysis.json, `verify/bk.py analyze`).

Usage: python -m verify.report_bk [--json results/bk/analysis.json] [--tex paper/tables/bk_loops.tex]
"""
import argparse
import json
from pathlib import Path

NAMES = {"f": "Full", "h": "History only", "c": "Cost-aware seq.", "a": "Admission only", "s": "Sequential"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/bk/analysis.json")
    ap.add_argument("--tex", default="paper/tables/bk_loops.tex")
    a = ap.parse_args()
    res = json.loads(Path(a.json).read_text())
    blocks = sorted({k.split("_")[0] for k in res["loops"]})
    rows = []
    for arm, name in NAMES.items():
        L = [res["loops"][f"{b}_{arm}"] for b in blocks]
        forks = [res["forks"].get(f"{b}_{arm}") for b in blocks]
        fk = ", ".join("--" if x is None else str(x) for x in forks)
        rows.append(f"{name} & {sum(x['accepted'] for x in L)} & {sum(x['rounds_full_would_accept'] for x in L)} & "
                    f"{sum(x['stopped_full_choice'] for x in L)} & {sum(x['stopped'] for x in L)} & "
                    f"{sum(x['cost']['eval_episodes'] for x in L) / len(L):.0f} & "
                    f"{sum(x['cost']['total'] for x in L) / len(L):.2f} & {fk} \\\\")
    tex = ("\\begin{tabular}{lrrrrrrl}\n\\toprule\n"
           "Arm & Acc. & Full acc. & Choice stopped & Stopped & Cand.\\ ep. & Cost (\\$) & Fork round \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.tex).write_text(tex)
    print(tex)
    print("noise bands:", {b: round(100 * res["loops"][f"{b}_f"]["delta"], 1) for b in blocks})


if __name__ == "__main__":
    main()
