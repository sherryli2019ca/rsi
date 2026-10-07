"""Tables for the tau2 intervention banks, from the offline reanalysis
(runs/tau2_*/reanalysis.json) and the held-out breakdown
(runs/tau2_retail/heldout_breakdown.json):

  paper/tables/tau2.tex     value of each policy's patch set, as a fraction of the
                            repair-only oracle, with and without a per-wrong-change cost
  paper/tables/heldout.tex  retail held-out success by episode type
  paper/tables/vpi.tex      where perfect verification could add value (Prop. 1)
  paper/tables/robust.tex   budget-40 values under alternative scorings
  paper/tables/csweep.tex   change-cost sweep, per wrong vs per applied change
  paper/tables/judges.tex   second judge configurations (judge_configs_report.json)

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


def load3():
    return {d: json.load(open(f"runs/tau2_{d}/reanalysis3.json")) for d in DOMS}


def load4():
    return {d: json.load(open(f"runs/tau2_{d}/reanalysis4.json")) for d in DOMS}


def main_table(R5):
    """Posterior value of deployable sets under the task-level binomial harm model,
    budgets 10/40/80, with the Bayes default: on the full bank, and with policy
    evidence and outcome assessment taken from halves that share no source
    trajectory or regression task (policy inputs from the policy's half,
    audited analyst accuracy). Also writes the absolute-success version and the
    appendix variants (outcome-split halves, audited accuracy on the full bank)."""
    def write(groups, res, c, path, scale=None):
        best = {(g, B): max(res[g][f"{m}|{B}"]["mean"] for m, _ in NAMES) for g in groups
                for B in BUDGETS}
        rows = []
        for m, name in NAMES:
            cells = []
            for g in groups:
                for B in BUDGETS:
                    v = res[g][f"{m}|{B}"]["mean"]
                    if scale:
                        cells.append(f"{100 * v * scale[g[0]]:.1f}".replace("-", "$-$"))
                    else:
                        cells.append(fmt(v, v >= best[g, B] - 1e-9))
            rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
            if m == "LLM-only":
                bd = []
                for g in groups:
                    k = f"Bayes-default-{c}|0"
                    for B in BUDGETS:
                        if k in res[g]:
                            v = res[g][k]["mean"]
                            bd.append(f"{100 * v * scale[g[0]]:.1f}".replace("-", "$-$") if scale else fmt(v))
                        else:
                            bd.append("--")
                rows.append("Bayes default (Prop.~\\ref{prop:vpi}) & " + " & ".join(bd) + " \\\\")
                rows.append("\\midrule")
        open(path, "w").write("\n".join(rows) + "\n")
        return rows

    scale = {d: R5[d]["yardstick_abs"] for d in DOMS}
    out = None
    for c, path in (("0.0", "paper/tables/tau2.tex"), ("0.02", "paper/tables/tau2_cw.tex")):
        groups = [(d, k) for d in DOMS for k in ("primary", "clean")]
        res = {g: R5[g[0]][g[1]][c] for g in groups}
        r = write(groups, res, c, path)
        out = out or r
        if c == "0.0":
            write(groups, res, c, "paper/tables/tau2_abs.tex", scale)
    groups = [(d, k) for d in DOMS for k in ("audit", "split")]
    res = {g: R5[g[0]][g[1]]["0.0"] for g in groups}
    write(groups, res, "0.0", "paper/tables/tau2_more.tex")
    return out


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
    A = json.load(open("runs/tau2_airline/heldout_breakdown.json"))
    cols, acols = ("all", "clean", "no_step_fault"), ("all", "clean")

    def vs(r):
        va = r["vs_apply"]["all"]
        return f"{fmt(va[0])} {{\\footnotesize[{fmt(va[2])}, {fmt(va[3])}]}}"
    rows = ["Unpatched & " + " & ".join(f"{H['unpatched'][c][0]:.2f}" for c in cols) + " & & 0 & " +
            " & ".join(f"{A['unpatched'][c][0]:.2f}" for c in acols) + " & & 0 \\\\", "\\midrule"]
    for m, name in NAMES:
        r, a = H.get(f"{m}|10"), A.get(f"{m}|10")
        if r:
            dv = "--" if m == "LLM-only" else vs(r)
            air = " & & & " if not a else (" & ".join(f"{a[c][0]:.2f}" for c in acols) + " & " +
                                           ("--" if m == "LLM-only" else vs(a)) +
                                           f" & {a['n_patches']:.1f}")
            rows.append(f"{name} & " + " & ".join(f"{r[c][0]:.2f}" for c in cols) +
                        f" & {dv} & {r['n_patches']:.1f} & {air} \\\\")
    open("paper/tables/heldout.tex", "w").write("\n".join(rows) + "\n")
    ses = [H[k][c][1] for k in H if k != "n_episodes" and isinstance(H[k], dict) for c in cols]
    return rows, (min(ses), max(ses)), H["n_episodes"]


def ablate_table():
    """Appendix: second-review held-out runs on retail (C, D, F)."""
    R = json.load(open("runs/tau2_retail/review2_heldout.json"))

    def ci(v):
        return f"{fmt(v[0])} {{\\scriptsize[{fmt(v[2])}, {fmt(v[3])}]}}"
    rows = []
    for m, v in R["C"].items():
        rows.append(f"{dict(NAMES)[m]} & {ci(v['full_minus_nostep']['all'])} & "
                    f"{ci(v['full_minus_nostep']['no_step_fault'])} & "
                    f"{ci(v['full_minus_nostep']['step_fault'])} & "
                    f"{ci(v['nostep_minus_unpatched']['no_step_fault'])} \\\\")
    open("paper/tables/ablate_step.tex", "w").write("\n".join(rows) + "\n")
    rows2 = []
    for c, v in R["D"]["contribution"].items():
        rows2.append(f"\\texttt{{{c.replace('_', '\\_')}}} & {ci(v['all'])} & {ci(v['no_step_fault'])} \\\\")
    rows2.append("\\midrule")
    rows2.append(f"Whole set vs unpatched & {ci(R['D']['full_minus_unpatched']['all'])} & "
                 f"{ci(R['D']['full_minus_unpatched']['no_step_fault'])} \\\\")
    rows2.append(f"Whole set $-$ sum of parts & {ci(R['D']['interaction']['all'])} & "
                 f"{ci(R['D']['interaction']['no_step_fault'])} \\\\")
    open("paper/tables/ablate_loo.tex", "w").write("\n".join(rows2) + "\n")
    F = R["F"]
    rows3 = []
    for m in ("unpatched", "LLM-only", "Net", "HarnessFix-gate", "HarnessFix"):
        v = F[m]
        name = "Unpatched" if m == "unpatched" else dict(NAMES)[m]
        du = "--" if m == "unpatched" else ci(v["minus_unpatched"])
        da = "--" if m == "LLM-only" else ci(v["minus_LLM-only"])
        rows3.append(f"{name} & {v['old']:.2f} & {v['new']:.2f} & {v['pooled']:.2f} & {du} & {da} \\\\")
    open("paper/tables/ablate_clean.tex", "w").write("\n".join(rows3) + "\n")
    return rows + rows2 + rows3


def vpi_table(R3, R5):
    """Value of perfect information over two deployable defaults under the
    binomial harm model (posterior means), with the decision-rule gap; indented
    rows: over Apply at c = 0 under the normal-approximation models of the
    previous analysis, and per cell (additive, not deployable)."""
    rows = []
    for d in DOMS:
        P = R5[d]["primary"]
        for n, c in enumerate(("0.0", "0.02")):
            lab_c = "$c=0$" if c == "0.0" else "$c_{\\mathrm w}=0.02$"
            va, vb = P["vpi"][f"{c}|apply"], P["vpi"][f"{c}|bayes"]
            gap = vb["default"][0] - va["default"][0]
            first = d.capitalize() if n == 0 else ""
            rows.append(f"{first} & Apply, {lab_c} & {fmt(va['default'][0])} & {fmt(gap)} & " +
                        " & ".join(fmt(va[k][0]) for k in ("harm", "select", "discover", "total")) + " \\\\")
            if c == "0.0":
                rows.append(f" & Bayes, {lab_c} & {fmt(vb['default'][0])} & -- & " +
                            " & ".join(fmt(vb[k][0]) for k in ("harm", "select", "discover", "total")) + " \\\\")
        for key, lab in (("primary", "normal, partial pooling"), ("model|pool", "normal, $\\tau{=}0$"),
                         ("model|nopool", "no pooling"), ("model|task", "tasks resampled"),
                         ("undeployed", "per cell")):
            u = R3[d][key]["vpi"]["0.0|apply"]
            rows.append(f" & \\quad {lab} & {fmt(u['default'][0])} & & " +
                        " & ".join(fmt(u[k][0]) for k in ("harm", "select", "discover", "total")) + " \\\\")
        if d == "retail":
            rows.append("\\midrule")
    open("paper/tables/vpi.tex", "w").write("\n".join(rows) + "\n")
    return rows


def harm_models_table(R3, R5=None, B=40):
    """Appendix: sensitivity of the posterior to the harm model (c = 0)."""
    labels = [("binom", "Task-level binomial (primary)"), ("primary", "Hierarchical normal"), ("model|pool", "Complete pooling ($\\tau=0$)"),
              ("model|tau2x", "Spread doubled ($2\\tau$)"), ("model|nopool", "No pooling"),
              ("model|task", "Tasks resampled"), ("uncapped", "No support cap"),
              ("undeployed", "Not deployable")]
    rows = []
    for key, lab in labels:
        cells = []
        for d in DOMS:
            if key == "binom":
                P = dict(R5[d]["primary"], mu=R5[d]["binom"]["pooled_rho"])
            else:
                P = R3[d][key]
            r = P["0.0"]
            ver = [(r[f"{m}|{B}"]["mean"], m) for m in SHORT if m != "LLM-only"]
            bv, bm = max(ver)
            mu = P["mu"][0]
            tot = P["vpi"]["0.0|apply"]["total"][0] if P.get("vpi") else float("nan")
            cells += [fmt(mu), fmt(r[f"LLM-only|{B}"]["mean"]), fmt(bv),
                      f"{r[f'{bm}|{B}']['p_better']:.2f}", fmt(tot)]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open("paper/tables/harm_models.tex", "w").write("\n".join(rows) + "\n")
    return rows


def netabl_table(R3):
    rows = []
    for m, lab in (("Net", "As used (clip, rescale)"), ("Net-noclip", "No clip"),
                   ("Net-norescale", "No rescale"), ("Net-raw", "Neither (Eq.~\\ref{eq:delta})")):
        cells = []
        for d in DOMS:
            r = R3[d].get("netabl", {}).get("0.0", {})
            cells += [fmt(r[f"{m}|{B}"]["mean"]) if f"{m}|{B}" in r else "--" for B in BUDGETS]
        rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    open("paper/tables/netabl.tex", "w").write("\n".join(rows) + "\n")
    return rows


def robust_table(R, R3, B=40):
    """One block per domain: posterior value with interval and probability of
    beating Apply attributed, then sensitivity scorings (significant harm,
    c_w = 0.02 unless stated) at budget B."""
    rows = []
    for d in DOMS:
        r = R[d]
        po = R3[d]["primary"]["0.02"]
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


def load5():
    import os
    return {d: json.load(open(f"runs/tau2_{d}/reanalysis5.json")) for d in DOMS
            if os.path.exists(f"runs/tau2_{d}/reanalysis5.json")}


def payback_table(R5):
    """Probability that verification ever pays back (and after 1,000 episodes)
    under gain models and change-cost distributions."""
    lab = dict(NAMES)
    cols = [("normal|none", "ever"), ("t|none", "ever"), ("bootstrap|none", "ever"),
            ("sceptic03|none", "ever"), ("normal|U02", "N1000"), ("normal|U02", "ever"),
            ("normal|U05", "ever"), ("normal|fixed05", "ever")]
    rows = []
    for d in DOMS:
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{d.capitalize()}}}}} \\\\")
        for m, r in R5[d]["payback"].items():
            if m not in lab:
                continue
            cells = [f"{r[k][v]:.2f}" for k, v in cols]
            rows.append(f"{lab[m]} & {fmt(r['mean'])} [{r['se']:.2f}] & {r['dn']:.1f} & "
                        + " & ".join(cells) + " \\\\")
    open("paper/tables/payback.tex", "w").write("\n".join(rows) + "\n")
    return rows


def judges_table(R5):
    labs = (("bank", "Bank judge (analyst, full trajectory)"),
            ("short_pro", "Analyst, failing step only"),
            ("full_flash", "Agent model, full trajectory$^\\dagger$"),
            ("short_flash", "Agent model, failing step only"))
    rows = []
    for d in DOMS:
        J = R5[d]["judges"]
        rows.append(f"\\multicolumn{{10}}{{l}}{{\\emph{{{d.capitalize()}}}}} \\\\")
        for c, lab in labs:
            r = J[c]
            pl = J["patch_level"] if c == "bank" else r["patch_level"]
            cp = r["cost_by_price"]
            rows.append(f"{lab} & {r['cost']:.2f} & {cp['2']:.2f} / {cp['4']:.2f} & "
                        f"{r['break_even']:.2f} & {r['ratio']:.2f} {{\\footnotesize[{r['ratio_90'][0]:.2f}, {r['ratio_90'][1]:.2f}]}} & "
                        f"{pl['sens']:.2f} & {pl['fpr']:.2f} & {pl['break_even']:.2f} & "
                        f"{pl['ratio']:.2f} {{\\footnotesize[{pl['ratio_90'][0]:.2f}, {pl['ratio_90'][1]:.2f}]}} \\\\")
    open("paper/tables/judges.tex", "w").write("\n".join(rows) + "\n")
    return rows


def deploy_table():
    """Main table: held-out success of every policy's patch set (budget 10) in
    percent, the paired difference from Apply attributed in points with a
    task-clustered 90% interval, patches per set and verification episodes."""
    D = {d: json.load(open(f"runs/tau2_{d}/deployment.json")) for d in DOMS}

    def pct(v):
        return f"{100 * v:.1f}"

    def pts(v):
        return f"{100 * v:+.1f}".replace("+", "").replace("-", "$-$")

    def block(a, m):
        if a is None:
            return " & & & & "
        if m == "unpatched":
            dv, C = "", ""
        elif m == "LLM-only":
            dv, C = "--", "0"
        else:
            v = a["vs_apply|all"]
            dv = f"{pts(v[0])} {{\\footnotesize[{pts(v[2])}, {pts(v[3])}]}}"
            C = f"{a['C_verify']:.0f}"
        return f"{pct(a['all'])} & {pct(a['clean'])} & {dv} & {a['n_patches']:.1f} & {C}"
    rows = ["Unpatched & " + block(D["retail"]["arms"]["unpatched"], "unpatched") + " & " +
            block(D["airline"]["arms"]["unpatched"], "unpatched") + " \\\\", "\\midrule"]
    for m, name in NAMES:
        r, a = D["retail"]["arms"].get(m), D["airline"]["arms"].get(m)
        rows.append(f"{name.replace(' (ours)', '')} & {block(r, m)} & {block(a, m)} \\\\")
        if m == "LLM-only":
            rows.append("\\midrule")
    open("paper/tables/deploy.tex", "w").write("\n".join(rows) + "\n")
    return rows


def transfer_data():
    """Figure data: distinct patch sets, gain over the unpatched agent predicted
    from their bank replays (net of spurious recovery) against the observed
    held-out gain, in points, with 90% half-widths."""
    out = {}
    for d in DOMS:
        T = json.load(open(f"runs/tau2_{d}/deployment.json"))["transfer"]
        seen = {}
        for p in T["points"]:
            seen.setdefault((round(p["pred_replay"], 5), round(p["obs"], 5)), p)
        for step in (True, False):
            lines = ["x y e"]
            for p in seen.values():
                if p["has_step"] == step and p["n_patches"] > 0:
                    lines.append(f"{100 * p['pred_replay']:.2f} {100 * p['obs']:.2f} {164.5 * p['se']:.2f}")
            open(f"paper/tables/transfer_{d}_{'step' if step else 'nostep'}.dat", "w").write("\n".join(lines) + "\n")
        out[d] = {"n": len(seen), "corr": T["corr_replay"], "corr_judge": T["corr_judge"]}
    return out


def main():
    R, R3, R4, R5 = load(), load3(), load4(), load5()
    print("\n".join(deploy_table()), "\n")
    print(transfer_data())
    for rows in (main_table(R5), sig_table(R), vpi_table(R3, R5), robust_table(R, R3), csweep_table(R),
                 harm_models_table(R3, R5), netabl_table(R3), judges_table(R5), payback_table(R5)):
        print("\n".join(rows), "\n")
    print("\n".join(ablate_table()), "\n")
    rows, se, n = heldout_table()
    print("\n".join(rows), "\nheld-out s.e. range", se, n)


if __name__ == "__main__":
    main()
