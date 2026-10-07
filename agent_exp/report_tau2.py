"""Tables for the tau2 intervention banks, from the offline reanalysis
(runs/tau2_*/reanalysis.json) and the held-out breakdown
(runs/tau2_retail/heldout_breakdown.json):

  paper/tables/tau2.tex     value of each policy's patch set, as a fraction of the
                            repair-only oracle, with and without a per-wrong-change cost
  paper/tables/heldout.tex  retail held-out success by episode type
  paper/tables/vpi.tex      where perfect verification could add value (Prop. 1)
  paper/tables/robust.tex   budget-40 values under alternative scorings
  paper/tables/csweep.tex   change-cost sweep, per wrong vs per applied change

  python -m agent_exp.report_tau2
"""
from __future__ import annotations

import json

NAMES = [("LLM-only", "Apply attributed"), ("Replay-each", "Replay-each"),
         ("HarnessFix", "HarnessFix-style, per patch"),
         ("HarnessFix-gate", "HarnessFix-style, bundle"),
         ("Uncertainty", "Uncertainty sampling"),
         ("Uncertainty-MF", "Uncertainty + checks"),
         ("CARVE-full-only", "\\carve{}, replay only"), ("CARVE", "\\carve{}"),
         ("CARVE-calibrated", "\\carve{}, measured check"),
         ("CARVE-robust-check", "\\carve{}, robust check"),
         ("Net", "Net-effect (ours)")]
SHORT = ["LLM-only", "HarnessFix", "HarnessFix-gate", "Uncertainty", "CARVE", "Net"]
BUDGETS = (10, 40, 80)
DOMS = ("retail", "airline")


def load():
    return {d: json.load(open(f"runs/tau2_{d}/reanalysis.json")) for d in DOMS}


def fmt(v, bold=False):
    s = f"{v:.2f}".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if bold else s


def main_table(R):
    groups = [(d, c) for d in DOMS for c in (0.0, 0.02)]
    res = {g: R[g[0]][f"main|sig|wrong|{g[1]}"]["res"] for g in groups}
    best = {(g, B): max(res[g][f"{m}|{B}"][0] for m, _ in NAMES) for g in groups for B in BUDGETS}
    rows = []
    for m, name in NAMES:
        cells = [fmt(res[g][f"{m}|{B}"][0], res[g][f"{m}|{B}"][0] >= best[g, B] - 1e-9)
                 for g in groups for B in BUDGETS]
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if m == "LLM-only":
            rows.append("\\midrule")
    open("paper/tables/tau2.tex", "w").write("\n".join(rows) + "\n")
    return rows


def heldout_table():
    H = json.load(open("runs/tau2_retail/heldout_breakdown.json"))
    cols = ("all", "clean", "no_step_fault")
    rows = ["Unpatched & " + " & ".join(f"{H['unpatched'][c][0]:.2f}" for c in cols) + " & 0 \\\\",
            "\\midrule"]
    for m, name in NAMES:
        r = H.get(f"{m}|10")
        if r:
            rows.append(f"{name} & " + " & ".join(f"{r[c][0]:.2f}" for c in cols) +
                        f" & {r['n_patches']:.1f} \\\\")
    open("paper/tables/heldout.tex", "w").write("\n".join(rows) + "\n")
    ses = [H[k][c][1] for k in H if k != "n_episodes" for c in cols]
    return rows, (min(ses), max(ses)), H["n_episodes"]


def vpi_table(R):
    rows = []
    for d in DOMS:
        for key, lab in (("vpi|sig|wrong|0.0", "measured harm only"),
                         ("vpi|sig|wrong|0.02", "$+$ $c_{\\mathrm w}=0.02$"),
                         ("vpi|sig|apply|0.02", "$+$ $c_{\\mathrm a}=0.02$")):
            v = R[d][key]
            first = d.capitalize() if lab.startswith("measured") else ""
            rows.append(f"{first} & {lab} & {fmt(v['default'])} & {fmt(v['harm'])} & "
                        f"{fmt(v['select'])} & {fmt(v['discover'])} \\\\")
        if d == "retail":
            rows.append("\\midrule")
    open("paper/tables/vpi.tex", "w").write("\n".join(rows) + "\n")
    return rows


def robust_table(R, B=40):
    """One block per domain; columns are alternative scorings at budget B."""
    rows = []
    for d in DOMS:
        r = R[d]
        sp = r["split"]
        cb = r["regression"]["cont_bootstrap"]
        for m in SHORT:
            name = dict(NAMES)[m]
            bb = 10 if m == "LLM-only" else B
            cols = [r["main|sig|wrong|0.02"]["res"][f"{m}|{bb}"][0],
                    r["sweep|apply|0.02"]["res"][f"{m}|{bb}"][0],
                    sp[f"edge|0.02|{m}|{bb}"][0],
                    r["natural|0.02"]["res"][f"{m}|{bb}"][0],
                    r["nostep|0.02"]["res"][f"{m}|{bb}"][0]]
            c = cb[f"{m}|{bb}"]
            cont = f"{fmt(c[0])} {{\\scriptsize[{fmt(c[1])}, {fmt(c[2])}]}}"
            first = d.capitalize() if m == SHORT[0] else ""
            rows.append(f"{first} & {name} & " + " & ".join(fmt(x) for x in cols) +
                        f" & {cont} \\\\")
        if d == "retail":
            rows.append("\\midrule")
    open("paper/tables/robust.tex", "w").write("\n".join(rows) + "\n")
    return rows


def csweep_table(R, B=40):
    rows = []
    cs = (0.0, 0.01, 0.02, 0.05)
    for d in DOMS:
        for m, name in NAMES:
            bb = 10 if m in ("LLM-only",) else B
            cells = [R[d][f"sweep|{t}|{c}"]["res"][f"{m}|{bb}"][0]
                     for t in ("wrong", "apply") for c in cs]
            first = d.capitalize() if m == NAMES[0][0] else ""
            rows.append(f"{first} & {name} & " + " & ".join(fmt(x) for x in cells) + " \\\\")
        if d == "retail":
            rows.append("\\midrule")
    open("paper/tables/csweep.tex", "w").write("\n".join(rows) + "\n")
    return rows


def main():
    R = load()
    for rows in (main_table(R), vpi_table(R), robust_table(R), csweep_table(R)):
        print("\n".join(rows), "\n")
    rows, se, n = heldout_table()
    print("\n".join(rows), "\nheld-out s.e. range", se, n)


if __name__ == "__main__":
    main()
