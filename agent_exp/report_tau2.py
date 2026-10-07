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
    res = {g: {k: [v["mean"]] for k, v in R[g[0]]["post|main"][f"{g[1]}"].items()
               if not k.startswith("p_best")} for g in groups}
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


def sig_table(R):
    """Appendix: the same table with only significant regressions charged."""
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
    open("paper/tables/tau2_sig.tex", "w").write("\n".join(rows) + "\n")
    return rows


def heldout_table():
    H = json.load(open("runs/tau2_retail/heldout_breakdown.json"))
    cols = ("all", "clean", "no_step_fault")
    rows = ["Unpatched & " + " & ".join(f"{H['unpatched'][c][0]:.2f}" for c in cols) + " & & 0 \\\\",
            "\\midrule"]
    for m, name in NAMES:
        r = H.get(f"{m}|10")
        if r:
            va = r["vs_apply"]["all"]
            dv = "--" if m == "LLM-only" else f"{fmt(va[0])} {{\\scriptsize[{fmt(va[2])}, {fmt(va[3])}]}}"
            rows.append(f"{name} & " + " & ".join(f"{r[c][0]:.2f}" for c in cols) +
                        f" & {dv} & {r['n_patches']:.1f} \\\\")
    open("paper/tables/heldout.tex", "w").write("\n".join(rows) + "\n")
    ses = [H[k][c][1] for k in H if k != "n_episodes" and isinstance(H[k], dict) for c in cols]
    return rows, (min(ses), max(ses)), H["n_episodes"]


def vpi_table(R):
    rows = []
    for d in DOMS:
        pv = R[d]["post|main"]["vpi"]
        for n, (v, lab) in enumerate(((pv["0.0"], "posterior, $c=0$"),
                                      (pv["0.02"], "posterior, $c_{\\mathrm w}=0.02$"))):
            first = d.capitalize() if n == 0 else ""
            rows.append(f"{first} & {lab} & " + " & ".join(
                fmt(v[k][0]) for k in ("default", "harm", "select", "discover")) + " \\\\")
        v = R[d]["vpi|sig|wrong|0.0"]
        rows.append(f" & significant only, $c=0$ & {fmt(v['default'])} & {fmt(v['harm'])} & "
                    f"{fmt(v['select'])} & {fmt(v['discover'])} \\\\")
        if d == "retail":
            rows.append("\\midrule")
    open("paper/tables/vpi.tex", "w").write("\n".join(rows) + "\n")
    return rows


def robust_table(R, B=40):
    """One block per domain: posterior value with interval and probability of
    beating Apply attributed, then sensitivity scorings (significant harm,
    c_w = 0.02 unless stated) at budget B."""
    rows = []
    for d in DOMS:
        r = R[d]
        po = r["post|main"]["0.02"]
        sp = r["split"]
        for m in SHORT:
            name = dict(NAMES)[m]
            p = po[f"{m}|{B}"]
            post = f"{fmt(p['mean'])} {{\\scriptsize[{fmt(p['lo'])}, {fmt(p['hi'])}]}}"
            pb = "--" if m == "LLM-only" else f"{p['p_better']:.2f}"
            cols = [r["main|sig|wrong|0.02"]["res"][f"{m}|{B}"][0],
                    r["sweep|apply|0.02"]["res"][f"{m}|{B}"][0],
                    sp[f"edge|0.02|{m}|{B}"][0],
                    r["natural|0.02"]["res"][f"{m}|{B}"][0],
                    r["nostep|0.02"]["res"][f"{m}|{B}"][0],
                    r["component|0.02"][f"{m}|{B}"][0],
                    r["post_sig|norep"]["0.02"][f"{m}|{B}"][0]]
            first = d.capitalize() if m == SHORT[0] else ""
            rows.append(f"{first} & {name} & {post} & {pb} & " + " & ".join(fmt(x) for x in cols) +
                        " \\\\")
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
    for rows in (main_table(R), sig_table(R), vpi_table(R), robust_table(R), csweep_table(R)):
        print("\n".join(rows), "\n")
    rows, se, n = heldout_table()
    print("\n".join(rows), "\nheld-out s.e. range", se, n)


if __name__ == "__main__":
    main()
