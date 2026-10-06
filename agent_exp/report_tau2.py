"""Tables for the tau2 intervention banks: paper/tables/tau2.tex (oracle-normalised
net gain on retail and airline) and paper/tables/heldout.tex (retail held-out
success after applying each method's accepted patches at budget 10).

  python -m agent_exp.report_tau2
"""
from __future__ import annotations

import json

import numpy as np

NAMES = [("LLM-only", "LLM attribution only"), ("Replay-each", "Replay-each"),
         ("HarnessFix", "HarnessFix-style"), ("Uncertainty", "Uncertainty sampling"),
         ("Uncertainty-MF", "Uncertainty + cheap checks"),
         ("CARVE-full-only", "\\carve{}, full replay only"), ("CARVE", "\\carve{}"),
         ("CARVE-calibrated", "\\carve{}, measured check"),
         ("CARVE-robust-check", "\\carve{}, robust check")]
BUDGETS = (10, 20, 40, 80)
FILES = {"retail": "runs/tau2_retail/evaluation_rep20.json",
         "airline": "runs/tau2_airline/evaluation_heldout.json"}


def cell(r, best):
    s = f"{r['gain']:.2f}"
    if best:
        s = f"\\textbf{{{s}}}"
    return s + (f"\\,{{\\scriptsize$\\pm${r['gain_se']:.2f}}}" if r["gain_se"] > 0 else "")


def main():
    ev = {d: json.load(open(f)) for d, f in FILES.items()}
    best = {(d, B): max(ev[d]["results"][f"{m}|{B}"]["gain"] for m, _ in NAMES)
            for d in ev for B in BUDGETS}
    rows = []
    for m, name in NAMES:
        cells = []
        for d in ("retail", "airline"):
            for B in BUDGETS:
                r = ev[d]["results"][f"{m}|{B}"]
                cells.append(cell(r, r["gain"] >= best[(d, B)] - 1e-9))
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if m in ("LLM-only", "Uncertainty-MF"):
            rows.append("\\midrule")
    with open("paper/tables/tau2.tex", "w") as fh:
        fh.write("\n".join(rows) + "\n")

    ho = json.load(open("runs/tau2_retail/heldout.json"))
    none = np.array(ho["runs"]["none"], float)
    out = [f"Unpatched agent & {none.mean():.2f}\\,{{\\scriptsize$\\pm${none.std(ddof=1) / np.sqrt(len(none)):.2f}}} & -- \\\\",
           "\\midrule"]
    for m, name in NAMES:
        keys = [ho["sets"][k] for k in sorted(ho["sets"]) if k.startswith(f"{m}|10|")]
        if not keys:
            continue
        ys = [np.mean(ho["runs"][k]) for k in keys]
        n_patch = np.mean([len(json.loads(k)) for k in keys])
        out.append(f"{name} & {np.mean(ys):.2f}\\,{{\\scriptsize$\\pm${np.std(ys, ddof=1) / np.sqrt(len(ys)):.2f}}}"
                   f" & {n_patch:.1f} \\\\")
    with open("paper/tables/heldout.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(rows + out))


if __name__ == "__main__":
    main()
