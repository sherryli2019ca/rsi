"""Tables of the attribution paper (paper2/) from the Phase A analyses.

  python -m attrib.report_paper /home/user/attrib_runs/main [--out paper2/tables]

Reads <main>/analysis_<domain>.json for tau2_retail, tau2_airline and appworld
and <main>/analysis_pooled3.json (attrib/analyze.py with --out), and the ground
truth and traces for the kinds of decisive step. Writes:
  main.tex     per method: R(k-hat) on rescuable failures by domain [95%] and
               pooled, exact accuracy pooled, dollars per failure
  kinds_tau2.tex, kinds_aw.tex  the kind of action at every step and at decisive steps
  search.tex   counterfactual search by budget, by domain
  secondary.tex  pooled secondary metrics per method
  whowhen.tex  agent and step accuracy on Who&When (results/attrib/whowhen_scores.json)
  phaseb_groups.tex, phaseb_contrasts.tex, phaseb_points.tex  the repair
               experiment (Phase B) from results/attrib/phaseB_analysis.json
               (attrib/phaseb_analyze.py); R-hat of each group's pointers on Phase
               B's own failures (phaseB_accuracy.json, attrib/phaseb_accuracy.py)
               and the post hoc oracle-pointer group (phaseB_oracle.json,
               attrib/phaseb_oracle.py) and no-reading groups (phaseB_review3.json,
               attrib/phaseb_review3.py --noread) when present
  robustness.tex, nullpos.tex  the ranking under other ground-truth choices and
               null-replay success by step position, from
               results/attrib/phaseA_robustness.json (attrib/robustness.py)
  prompt_variants.tex  judges asked for the step most likely to rescue the
               task or for the largest gain over null, or given no grading
               information, against their registered prompts, and
               counterfactual search with and without replays
               (results/attrib/phaseA_{rescue,gain,blind}_prompt.json,
               attrib/rescue_prompt.py)
  gold_rule.tex  the first deviation from tau2's gold actions against other
               methods (results/attrib/phaseA_gold_rule.json, attrib/gold_rule.py)
  robustness.tex also gets rows for the observed action as control and the
               second oracle (phaseA_orig_control.json, attrib/orig_control.py;
               phaseA_second_oracle.json, attrib/second_oracle.py)
  robustness.tex also gets rows for the re-scorings without state-changing
               corrections (results/attrib/audit_rescore.json, attrib/audit_rescore.py)
  decomp.tex, decomp_kinds.tex  proposal share and gain of a proposed
               correction at each method's step and by kind of step, and a
               split-sample row of robustness.tex (results/attrib/phaseA_review2.json,
               attrib/review2.py)
and prints the numbers the text quotes. Without <main>, only the Phase B and
robustness tables are written.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
DNAME = {"tau2_retail": "Retail", "tau2_airline": "Airline", "appworld": "AppWorld"}
ROWS = [("first_write", "First write", "rule"), ("last_step", "Last step", "rule"),
        ("all_at_once_flash", "All-at-once", "Flash"), ("all_at_once_pro", "All-at-once", "Pro"),
        ("all_at_once_pro_think", "All-at-once", "Pro, thinking"),
        ("step_by_step_pro", "Step-by-step", "Pro"), ("binary_search_pro", "Binary search", "Pro"),
        ("study1_pro", "Root-cause attributor", "Pro"), ("rrsi_digest_pro", "RRSI trace digester", "Pro"),
        ("search@40", "Counterfactual search@40", "Pro, replays")]
MIDRULES = {2, 5, 7}
PB_GROUPS = [("none", "No attribution"), ("rrsi", "RRSI analysis"), ("first_write", "First write"),
             ("binary_search", "Binary search"), ("cf_search", "Counterfactual search@40")]
PB_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "attrib" / "phaseB_analysis.json"
PB_ACC_DEFAULT = PB_DEFAULT.with_name("phaseB_accuracy.json")
PB_ORACLE_DEFAULT = PB_DEFAULT.with_name("phaseB_oracle.json")
PB_R3_DEFAULT = PB_DEFAULT.with_name("phaseB_review3.json")
ROB_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "attrib" / "phaseA_robustness.json"
ROB_ROWS = [("main", "Registered"), ("K=2", "$K=2$"), ("K=1", "$K=1$"),
            ("threshold 0.25", "$\\theta=0.25$"), ("threshold 0.375", "$\\theta=0.375$"),
            ("threshold 0.625", "$\\theta=0.625$"), ("threshold 0.75", "$\\theta=0.75$"),
            ("unique rescuing step", "One rescuing step"), ("no null", "No null replays")]
RESCUE_DEFAULT = ROB_DEFAULT.with_name("phaseA_rescue_prompt.json")
GOLD_DEFAULT = ROB_DEFAULT.with_name("phaseA_gold_rule.json")
REVIEW2_DEFAULT = ROB_DEFAULT.with_name("phaseA_review2.json")
ORIG_DEFAULT = ROB_DEFAULT.with_name("phaseA_orig_control.json")
SECOND_DEFAULT = ROB_DEFAULT.with_name("phaseA_second_oracle.json")
ADMISS_DEFAULT = ROB_DEFAULT.with_name("phaseA_admissibility.json")
FACTORIAL_DEFAULT = ROB_DEFAULT.with_name("phaseA_factorial.json")
REVIEW4_DEFAULT = ROB_DEFAULT.with_name("phaseAB_review4.json")
REVIEW5_DEFAULT = ROB_DEFAULT.with_name("phaseAB_review5.json")
POSTHOC_ROWS = [("all_at_once_gain_pro", "All-at-once, gain", "Pro"),
                ("binary_search_gain_pro", "Binary search, gain", "Pro"),
                ("all_at_once_blind_pro", "All-at-once, no grading", "Pro"),
                ("binary_search_blind_pro", "Binary search, no grading", "Pro"),
                ("search@0", "CF search@0", "Pro"),
                ("search_inf@0", "CF search@0, grading", "Pro"),
                ("search_inf@40", "CF search@40, grading", "Pro, replays")]
VARIANT_ROWS = [("all_at_once_flash", "All-at-once, Flash", ("rescue", "gain")),
                ("all_at_once_pro", "All-at-once, Pro", ("rescue", "gain", "blind")),
                ("binary_search_pro", "Binary search, Pro", ("rescue", "gain", "blind"))]
GOLD_ROWS = [("gold_deviation", "Gold deviation (rule)"), ("first_write", "First write (rule)"),
             ("all_at_once_pro", "All-at-once, Pro"), ("binary_search_pro", "Binary search, Pro"),
             ("search@40", "Counterfactual search@40")]
PB_X = {g: i + 1 for i, (g, _) in enumerate([("none", 0), ("rrsi", 0), ("first_write", 0),
                                             ("binary_search", 0), ("cf_search", 0), ("oracle", 0),
                                             ("rrsi_noread", 0), ("oracle_fix_noread", 0)])}
PB_NOREAD = [("rrsi_noread", "RRSI analysis, no reading", "rrsi"),
             ("oracle_fix_noread", "Oracle fix, no reading", "oracle")]   # post hoc, like the oracle group
PB_NOREAD_CONTRASTS = [("oracle_fix_noread", "rrsi_noread", "Oracle fix $-$ RRSI, no reading"),
                       ("rrsi_noread", "rrsi", "RRSI: no reading $-$ reading"),
                       ("oracle_fix_noread", "oracle", "Oracle fix, no reading $-$ oracle step")]
PB_CLIP = -0.25
T975_DF7 = 2.364624


def _f(x, d=2):
    return f"{x:.{d}f}"


def _ci(t, d=2):
    return f"{_f(t[0], d)} {{\\scriptsize [{_f(t[1], d)}, {_f(t[2], d)}]}}"


def tau2_kind(st: dict) -> str:
    from domains.tau2.common import READ_PREFIXES
    calls = [b["name"] for b in st["assistant"] if b["type"] == "tool_use"]
    if any(c.startswith("transfer") for c in calls):
        return "transfer"
    if any(not c.startswith(READ_PREFIXES) for c in calls):
        return "write"
    return "read" if calls else "utterance"


def aw_kind(st: dict) -> str:
    from attrib.aw import _CALL, is_write
    code = st.get("code") or ""
    calls = _CALL.findall(code)
    if "complete_task" in code:
        return "submit"
    if any(is_write(a, b) for a, b in calls):
        return "write"
    if any(a != "api_docs" for a, _ in calls):
        return "read"
    return "docs" if calls else "other"


def kinds(main: Path, d: str) -> dict:
    fails = {f["fid"]: f for f in json.loads((main / d / "failures.json").read_text())}
    gt = {f["fid"]: f for f in json.loads((main / d / "gt" / "a" / "result.json").read_text())["failures"]}
    kind = aw_kind if d == "appworld" else tau2_kind
    allk, dec = Counter(), Counter()
    rel = []
    for fid, g in gt.items():
        steps = json.loads(Path(fails[fid]["trace"]).read_text())["steps"]
        for st in steps:
            allk[kind(st)] += 1
        if g["decisive"] is not None:
            st = next(s for s in steps if s["index"] == g["decisive"])
            dec[kind(st)] += 1
            rel.append(g["decisive"] / max(g["n_steps"] - 1, 1))
    return {"all": dict(allk), "decisive": dict(dec), "mean_relative_position": sum(rel) / len(rel),
            "decisive_is_last": sum(g["decisive"] == g["n_steps"] - 1 for g in gt.values()
                                    if g["decisive"] is not None)}


def _pp(x, d=2):
    """Percentage points with a sign, typeset minus."""
    return f"{0:.{d}f}" if abs(x * 100) < 0.5 * 10 ** -d else f"{x * 100:+.{d}f}".replace("-", "$-$")


def _sg(x, d=2):
    """Signed number, typeset minus."""
    return f"{x:+.{d}f}".replace("-", "$-$")


def _state_contrast(rows: list[dict], g: str, h: str) -> dict:
    """Mean over states of the difference in mean slot gain, t interval (df 7);
    the registered estimator of attrib/phaseb_analyze.py without its test."""
    import numpy as np
    states = sorted({r["state"] for r in rows})
    d = np.array([np.mean([r["G"] for r in rows if r["group"] == g and r["state"] == s]) -
                  np.mean([r["G"] for r in rows if r["group"] == h and r["state"] == s]) for s in states])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"estimate": float(d.mean()), "ci95": [float(d.mean() - T975_DF7 * se), float(d.mean() + T975_DF7 * se)]}


def robustness_tables(path: Path, out: Path, review2: Path | None = None, orig: Path | None = None,
                      second: Path | None = None, admiss: Path | None = None, review4: Path | None = None,
                      rescore: Path | None = None) -> dict:
    R = json.loads(path.read_text())
    V = {v["variant"]: v for v in R["variants"]}
    short = {m: (name if model in ("rule", "Pro", "Pro, replays") else f"{name}, {model}")
             for m, name, model in ROWS}
    short["all_at_once_pro_think"] = "Judge, thinking"
    lines = []
    for key, label in ROB_ROWS:
        v = V[key]
        lines.append(" & ".join([label, f"{v['n_rescuable']:.0f}", _f(v["mean"]["first_write"]),
                                 f"{short[v['best_llm']]} {_f(v['best_llm_R'])}",
                                 f"{v['first_write_rank']:.0f}", _f(v["kendall_tau_vs_main"])]) + " \\\\")
    if review2 is not None and review2.exists():
        sp = json.loads(review2.read_text())["split_sample"]
        lines.append(" & ".join(["Split samples", f"{sp['n_rescuable']:.0f}", _f(sp["mean"]["first_write"]),
                                 f"{short[sp['best_llm']]} {_f(sp['mean'][sp['best_llm']])}",
                                 f"{sp['first_write_rank']:.0f}", _f(sp["kendall_tau_vs_main"])]) + " \\\\")
    if orig is not None and orig.exists():
        o = json.loads(orig.read_text())
        lines.append(" & ".join(["Observed action as control", f"{o['n_failures']}$^\\dagger$", _f(o["mean"]["first_write"]),
                                 f"{short[o['best_llm']]} {_f(o['mean'][o['best_llm']])}",
                                 f"{o['first_write_rank']}", _f(o["kendall_tau_vs_main"])]) + " \\\\")
    if second is not None and second.exists():
        so = json.loads(second.read_text())
        for key, label in (("second", "Second oracle (Flash)"), ("average", "Both oracles averaged")):
            v = so[key]
            lines.append(" & ".join([label, f"{sum(v['n_rescuable'].values())}", _f(v["mean"]["first_write"]),
                                     f"{short[v['best_llm']]} {_f(v['mean'][v['best_llm']])}",
                                     f"{v['first_write_rank']}", _f(v["kendall_tau_vs_registered"])]) + " \\\\")
    if admiss is not None and admiss.exists():
        A = json.loads(admiss.read_text())
        v = A["rescored"]["unseen_id_or_expanded"]
        lines.append(" & ".join(["Flagged corrections dropped", f"{sum(v['n_rescuable'].values())}",
                                 _f(v["mean"]["first_write"]), f"{short[v['best_llm']]} {_f(v['mean'][v['best_llm']])}",
                                 f"{v['first_write_rank']}", _f(v["kendall_tau_vs_registered"])]) + " \\\\")
        rl = []
        for d in DOMAINS:
            r = A["rates"][d]
            c = A["calls"][d]
            rl.append(" & ".join([DNAME[d], f"{r['n']:,}".replace(",", "{,}"), f"{100 * r['unseen_id']:.1f}",
                                  f"{100 * r['unseen_any']:.1f}", f"{100 * r['expanded']:.1f}",
                                  f"{100 * r['strict']:.1f}", f"{c['obs_mean']:.2f} / {c['corr_mean']:.2f}"]) + " \\\\")
        r = A["rates_decisive_step"]
        rl.append("\\midrule")
        rl.append(" & ".join(["At best rescue steps", f"{r['n']}", f"{100 * r['unseen_id']:.1f}",
                              f"{100 * r['unseen_any']:.1f}", f"{100 * r['expanded']:.1f}",
                              f"{100 * r['strict']:.1f}", ""]) + " \\\\")
        (out / "admissibility.tex").write_text("\n".join(rl) + "\n")
    if review4 is not None and review4.exists() and "blind" in json.loads(review4.read_text()):
        B = json.loads(review4.read_text())["blind"]
        best = B["best_llm_reg10"]
        lines.append(" & ".join(["Oracle without grading", f"{B['n']}$^\\dagger$", _f(B["mean"]["first_write"]),
                                 f"{short[best]} {_f(B['mean'][best])}", f"{B['rank_first_write_reg10']}",
                                 _f(B["kendall_reg10_vs_registered"])]) + " \\\\")
    if rescore is not None and rescore.exists():
        S = json.loads(rescore.read_text())
        for key, label in (("tau2_conflicts", "Conflicting $\\tau^2$ writes dropped"),
                           ("tau2_writes", "All $\\tau^2$ writes dropped"), ("all_writes", "All writes dropped")):
            if key not in S:
                continue
            v = S[key]["pooled"]
            lines.append(" & ".join([label, f"{sum(v['n_rescuable'].values())}", _f(v["mean"]["first_write"]),
                                     f"{short[v['best_llm']]} {_f(v['mean'][v['best_llm']])}",
                                     f"{v['first_write_rank']}", _f(v["kendall_tau_vs_registered"])]) + " \\\\")
    (out / "robustness.tex").write_text("\n".join(lines) + "\n")
    names = {"tau2_retail": "Retail", "tau2_airline": "Airline", "appworld": "AppWorld"}
    marks = {"tau2_retail": "*", "tau2_airline": "square*", "appworld": "triangle*"}
    lines = []
    for d, bins in R["null_by_position"].items():
        pts = " ".join(f"({(i + 0.5) / 5:.1f},{b[0]:.3f})" for i, b in enumerate(bins))
        lines.append(f"\\addplot[mark={marks[d]}, mark size=1.6pt] coordinates {{{pts}}};")
        lines.append(f"\\addlegendentry{{{names[d]}}}")
    (out / "nullpos.tex").write_text("\n".join(lines) + "\n")
    return {"variants": {k: V[k] for k, _ in ROB_ROWS}, "rescuable_by_domain": R["rescuable_by_domain"],
            "null_by_position": R["null_by_position"], "named_relative_position": R["named_relative_position"]}


R4_GROUPS = [("none", "No attribution"), ("rrsi", "RRSI analysis"), ("first_write", "First write"),
             ("binary_search", "Binary search"), ("cf_search", "Counterfactual search@40"),
             ("oracle", "Oracle step"), ("rrsi_noread", "RRSI analysis, no reading"),
             ("oracle_fix_noread", "Oracle fix, no reading")]
R4_STRATA_M = ["first_write", "all_at_once_pro", "binary_search_pro", "search@40", "all_at_once_gain_pro",
               "binary_search_gain_pro", "search_inf@40"]
R4_KIND = {"write": "write", "utterance": "utterance", "submit": "submission", "read": "read", "other": "other code",
           "docs": "docs"}
R4_BLIND_ROWS = ROWS + POSTHOC_ROWS


def _pm(x, d=2):
    return f"${'+' if x >= 0 else '-'}{abs(x):.{d}f}$"


def review5_tables(path: Path, out: Path) -> dict:
    """Post hoc tables for the fifth review: full profiles without grading on a
    random 100 failures (blindfull.tex) and search with the metric's selection
    rule (maxgain.tex)."""
    R = json.loads(path.read_text())
    res = {}
    B = R.get("blindfull") or {}
    if B.get("n"):
        lines = []
        for i, (m, name, model) in enumerate(R4_BLIND_ROWS):
            if i in MIDRULES or i == len(ROWS):
                lines.append("\\midrule")
            lines.append(" & ".join([name, model, _f(B["scores_reg_rescuable_registered"][m]),
                                     _f(B["scores_all_registered"][m]), _f(B["scores_blind_rescuable"][m]),
                                     _f(B["scores_all_blind"][m])]) + " \\\\")
        (out / "blindfull.tex").write_text("\n".join(lines) + "\n")
        res["blindfull"] = {k: v for k, v in B.items() if k != "_rows"}
    M = R.get("maxgain") or {}
    if M.get("scores"):
        S = M["scores"]
        lines = []
        for v, name in (("search", "Registered call"), ("search_informed", "With grading info."),
                        ("aligned", "With grading info., gain")):
            if f"{v}:maxgain" not in S:
                continue
            lines.append(" & ".join([name, _f(S[f"{v}:first_flip"]["mean"]), _f(S[f"{v}:maxgain"]["mean"]),
                                     f"{S[f'{v}:replays']['mean']:.1f}"]) + " \\\\")
        (out / "maxgain.tex").write_text("\n".join(lines) + "\n")
        res["maxgain"] = {"scores": S, "diffs": M.get("diffs")}
    return res


def review4_tables(path: Path, out: Path) -> dict:
    """Tables for the accepted rounds, the kinds of best rescue step, the
    grading-blind oracle and the fresh replays (attrib/review4.py)."""
    R = json.loads(path.read_text())
    res = {}
    if "accepted" in R:
        A = R["accepted"]
        con = {(c["g"], c["h"]): c for c in A["contrasts"]}
        lines = []
        for g, name in R4_GROUPS:
            v = A["groups"][g]
            gains = ", ".join(_pm(x, 1) for x in v["accepted_gains"]) or "--"
            c = con.get((g, "rrsi"))
            cells = [name, f"{v['accepted']}/{v['rounds']}", gains, _pm(v["round_gain"])]
            if c:
                s, r = c["states"]["ci95"], c["runs"]["ci95"]
                cells += [_pm(c["estimate_pp"]), f"[{_pm(s[0])}, {_pm(s[1])}]", f"[{_pm(r[0])}, {_pm(r[1])}]",
                          f"{c['runs']['p_signflip']:.2f}"]
            else:
                cells += ["", "", "", ""]
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
        for g, h, name in (("first_write+binary_search+cf_search+oracle", "rrsi", "Any step pointer $-$ RRSI"),
                           ("oracle+oracle_fix_noread", "rrsi+rrsi_noread", "Oracle evidence $-$ RRSI, pooled"),
                           ("oracle_fix_noread", "rrsi_noread", "Oracle fix $-$ RRSI, no reading")):
            c = con[(g, h)]
            s, r = c["states"]["ci95"], c["runs"]["ci95"]
            lines.append(" & ".join([name, "", "", "", _pm(c["estimate_pp"]), f"[{_pm(s[0])}, {_pm(s[1])}]",
                                     f"[{_pm(r[0])}, {_pm(r[1])}]", f"{c['runs']['p_signflip']:.2f}"]) + " \\\\")
        (out / "accepted.tex").write_text("\n".join(lines) + "\n")
        res["accepted"] = A
    if "strata" in R:
        S = R["strata"]
        lines = []
        for d in DOMAINS:
            for cell, v in S["by_cell"].items():
                dd, k = cell.split("/")
                if dd != d or v["n"] < 5:
                    continue
                lines.append(" & ".join([f"{DNAME[d]}, {R4_KIND.get(k, k)}", str(v["n"])] +
                                        [_f(v[m]) for m in R4_STRATA_M]) + " \\\\")
        lines.append("\\midrule")
        for key, name in (("state_changing", "Best step changes state"), ("other", "Other best steps")):
            v = S["split"][key]
            lines.append(" & ".join([name, str(v["n"])] + [_f(v["mean"][m]) for m in R4_STRATA_M]) + " \\\\")
        v = S["kind_balanced"]
        lines.append(" & ".join([f"Cells weighted equally", str(v["cells"])] +
                                [_f(v["mean"][m]) for m in R4_STRATA_M]) + " \\\\")
        (out / "strata.tex").write_text("\n".join(lines) + "\n")
        res["strata"] = {k: S[k] for k in ("split", "kind_balanced",
                                           "first_write_score_share_from_state_changing_best_steps")}
    if "blind" in R:
        B = R["blind"]
        lines = []
        for i, (m, name, model) in enumerate(R4_BLIND_ROWS):
            if i in MIDRULES or i == len(ROWS):
                lines.append("\\midrule")
            lines.append(" & ".join([name, model, _f(B["registered_same_failures"][m]), _f(B["mean"][m])] +
                                    [_f(B["by_domain"][d]["mean"][m]) for d in DOMAINS]) + " \\\\")
        (out / "blind.tex").write_text("\n".join(lines) + "\n")
        res["blind"] = {k: v for k, v in B.items() if k not in ("steps_by_domain_kind",)}
    if "fresh" in R:
        F = R["fresh"]
        lines = []
        for g, name in (("oracle", "Oracle step"), ("first_write", "First write"), ("binary_search", "Binary search"),
                        ("cf_search", "CF search@40"), ("rrsi", "RRSI (earliest cited)")):
            a, r = F["all"][g], F["rescuable"][g]
            lines.append(" & ".join([name, _f(a["reg"]), _f(a["fresh"]), _f(r["reg"]), _f(r["fresh"])]) + " \\\\")
        (out / "fresh.tex").write_text("\n".join(lines) + "\n")
        res["fresh"] = F
    return res


def variant_tables(rescue: Path, gold: Path, out: Path) -> dict:
    """prompt_variants.tex from the rescue, gain and blind runs of attrib.rescue_prompt
    (phaseA_{rescue,gain,blind}_prompt.json next to `rescue`)."""
    res = {}
    V = {v: json.loads(p.read_text()) for v in ("rescue", "gain", "blind")
         if (p := rescue.with_name(f"phaseA_{v}_prompt.json") if v != "rescue" else rescue).exists()}
    if V:
        M, P = {}, {}
        for v in V.values():  # a registered method keeps the interval of the first run that scores it
            for m, x in v["methods"].items():
                M.setdefault(m, x)
            P.update(v["pairs"])
        names = {"": "earliest (reg.)", "rescue": "most likely rescue", "gain": "largest gain over null",
                 "blind": "earliest, no grading"}

        def row(m, name, definition, diff=None, dollars=None):
            v = M[m]
            cells = [name, definition] + [_f(v["R"][d]) for d in DOMAINS]
            cells += [_ci(v["R"]["pooled"]), _f(v["exact"]), _f(v["relative_position"]),
                      dollars if dollars is not None else f"{sum(v['dollars_per_trace'].values()) / 3:.4f}"]
            cells.append(f"{_sg(diff[0])} {{\\scriptsize [{_sg(diff[1])}, {_sg(diff[2])}]}}" if diff else "")
            return " & ".join(cells) + " \\\\"
        lines = []
        for i, (reg, name, variants) in enumerate(VARIANT_ROWS):
            if i:
                lines.append("\\midrule")
            lines.append(row(reg, name, names[""]))
            for v in variants:
                m = reg.replace("_flash", f"_{v}_flash").replace("_pro", f"_{v}_pro")
                if m in M:
                    lines.append(row(m, "", names[v], P[f"{m}-{reg}"]["pooled"]))
        lines.append("\\midrule")
        if "search@0" in M:
            lines.append(row("search@0", "Counterfactual search@0", "first suspect, no grading"))
            lines.append(row("search@40", "Counterfactual search@40", "no grading",
                             P["search@40-search@0"]["pooled"] if "search@40-search@0" in P else None))
            lines.append("\\midrule")
        lines.append(row("first_write", "First write", "rule", dollars="0"))
        (out / "prompt_variants.tex").write_text("\n".join(lines) + "\n")
        res["prompt_variants"] = {"pairs": P, "rerun_dollars": {v: x["rerun_dollars"] for v, x in V.items()},
                                  "R": {m: M[m]["R"] for m in M}, "position": {m: M[m]["relative_position"] for m in M}}
    if gold.exists():
        G = json.loads(gold.read_text())
        lines = []
        for m, name in GOLD_ROWS:
            v = G["R"][m]
            dm = G["gold_minus"].get(m)
            cells = [name, _f(v["tau2_retail"]), _f(v["tau2_airline"]), _ci(v["tau2"]), _f(G["exact"][m]),
                     f"{_sg(dm[0])} {{\\scriptsize [{_sg(dm[1])}, {_sg(dm[2])}]}}" if dm else ""]
            lines.append(" & ".join(cells) + " \\\\")
        (out / "gold_rule.tex").write_text("\n".join(lines) + "\n")
        res["gold_rule"] = G
    return res


def decomp_tables(path: Path, out: Path) -> dict:
    if not path.exists():
        return {}
    D = json.loads(path.read_text())["decomposition"]
    lines = []
    for i, (m, name, model) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        v = D["named"][m]
        lines.append(" & ".join([name, model, _f(v["proposal_share"]), _f(v["conditional_gain"]), _f(v["R"])]) + " \\\\")
    v = D["named"]["decisive"]
    lines += ["\\midrule", " & ".join(["Best rescue step", "", _f(v["proposal_share"]), _f(v["conditional_gain"]),
                                        _f(v["R"])]) + " \\\\"]
    (out / "decomp.tex").write_text("\n".join(lines) + "\n")
    order = {"tau2_retail": ["utterance", "read", "write"], "tau2_airline": ["utterance", "read", "write"],
             "appworld": ["docs", "read", "write", "submit"]}
    lines = []
    for d, ks in order.items():
        for j, kk in enumerate(ks):
            v = D["by_kind"][d][kk]
            lines.append(" & ".join([DNAME[d] if j == 0 else "", kk, str(v["steps"]), _f(v["proposal_share"]),
                                     _f(v["conditional_gain"]), _f(v["null"])]) + " \\\\")
    (out / "decomp_kinds.tex").write_text("\n".join(lines) + "\n")
    return D


def phaseb_tables(path: Path, out: Path, accuracy: Path | None = None, oracle: Path | None = None,
                  review3: Path | None = None) -> dict:
    """accuracy: attrib.phaseb_accuracy output (R-hat of each group's pointers on
    Phase B's own rescuable failures); oracle: attrib.phaseb_oracle output (the
    post hoc oracle-pointer group), added below a rule when present; review3:
    attrib.phaseb_review3 --noread output (the post hoc no-reading groups), added
    after the oracle group when it holds them."""
    B = json.loads(path.read_text())
    R3 = json.loads(review3.read_text()) if review3 is not None and review3.exists() else {}
    NR = R3.get("noread")
    S = B["secondary"]
    acc, rl = S["S7_phase_a_accuracy"], S["S4_round_level_means"]
    own = json.loads(accuracy.read_text())["groups"] if accuracy is not None and accuracy.exists() else None
    O = json.loads(oracle.read_text()) if oracle is not None and oracle.exists() else None
    groups = list(PB_GROUPS) + ([("oracle", "Oracle step")] if O else [])
    lines = []
    for g, name in groups:
        v = (O["groups"] if g == "oracle" else B["groups"])[g]
        if g == "none":
            pa = "--"
        elif own is not None:
            pa = " / ".join(_f(own[g]["R"][d]) for d in ("tau2_retail", "tau2_airline"))
        else:
            pa = " / ".join(_f(acc[d][g]) for d in ("tau2_retail", "tau2_airline"))
        dep = v["mean_G_deployable"]
        r_l = O["round_level_means"][g] if g == "oracle" else rl[g]
        cells = [name, pa, f"{v['deployable']}/{v['slots']}", _pp(v["mean_G"]),
                 _pp(v["by_domain"]["tau2_retail"]), _pp(v["by_domain"]["tau2_airline"]),
                 _pp(dep) if dep is not None else "--",
                 f"{v['share_positive_deployable']:.0%}".replace("%", "\\%") if dep is not None else "--",
                 str(v["accepted_by_rrsi"]), _pp(r_l)]
        if g == "oracle":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    for g, name, ptr in (PB_NOREAD if NR else []):
        v = NR["groups"][g]
        pa = " / ".join(_f(own[ptr]["R"][d]) for d in ("tau2_retail", "tau2_airline")) if own is not None else "--"
        dep = v["mean_G_deployable"]
        cells = [name, pa, f"{v['deployable']}/{v['slots']}", _pp(v["mean_G"]),
                 _pp(v["by_domain"]["tau2_retail"]), _pp(v["by_domain"]["tau2_airline"]),
                 _pp(dep) if dep is not None else "--",
                 f"{v['share_positive_deployable']:.0%}".replace("%", "\\%") if dep is not None else "--",
                 str(v["accepted_by_rrsi"]), _pp(NR["round_level_means"][g])]
        lines.append(" & ".join(cells) + " \\\\")
    (out / "phaseb_groups.tex").write_text("\n".join(lines) + "\n")

    name = {**dict(PB_GROUPS), "cf_search": "CF search@40", "none": "none", "rrsi": "RRSI"}
    s4 = {(c["g"], c["h"]): c for c in S["S4_round_level"]}
    lines = []
    for i, c in enumerate(B["primary"]):
        if i == 3:
            lines.append("\\midrule")
        r = s4[(c["g"], c["h"])]
        cells = [f"{name[c['g']]} $-$ {name[c['h']]}",
                 f"{_pp(c['estimate'])} {{\\scriptsize [{_pp(c['ci95'][0])}, {_pp(c['ci95'][1])}]}}",
                 f"{c['p_perm']:.2f}", f"{c['p_holm']:.2f}", "yes" if c["equivalent_within_2.5pp"] else "no",
                 f"{_pp(r['estimate'])} {{\\scriptsize [{_pp(r['ci95'][0])}, {_pp(r['ci95'][1])}]}}"]
        lines.append(" & ".join(cells) + " \\\\")
    if O:
        lines.append("\\midrule")
        for h, label in (("rrsi", "Oracle $-$ RRSI"), ("steps", "Oracle $-$ step groups")) if NR else \
                (("rrsi", "Oracle $-$ RRSI"), ("none", "Oracle $-$ none"), ("steps", "Oracle $-$ step groups")):
            c = O["slot_level_vs_step_groups"] if h == "steps" else O["slot_level"][h]
            r = O["round_level"].get(h)
            cells = [label,
                     f"{_pp(c['estimate'])} {{\\scriptsize [{_pp(c['ci95'][0])}, {_pp(c['ci95'][1])}]}}",
                     f"{c['p_perm']:.2f}", "--", "yes" if c["equivalent_within_2.5pp"] else "no",
                     f"{_pp(r['estimate'])} {{\\scriptsize [{_pp(r['ci95'][0])}, {_pp(r['ci95'][1])}]}}" if r else "--"]
            lines.append(" & ".join(cells) + " \\\\")
    if NR:
        sl = {(c["g"], c["h"]): c for c in NR["slot_level"]}
        rl3 = {(c["g"], c["h"]): c for c in NR["round_level"]}
        for g, h, label in PB_NOREAD_CONTRASTS:
            c, r = sl[(g, h)], rl3[(g, h)]
            cells = [label,
                     f"{_pp(c['estimate'])} {{\\scriptsize [{_pp(c['ci95'][0])}, {_pp(c['ci95'][1])}]}}",
                     f"{c['p_perm']:.2f}", "--", "yes" if c["equivalent_within_2.5pp"] else "no",
                     f"{_pp(r['estimate'])} {{\\scriptsize [{_pp(r['ci95'][0])}, {_pp(r['ci95'][1])}]}}"]
            lines.append(" & ".join(cells) + " \\\\")
    (out / "phaseb_contrasts.tex").write_text("\n".join(lines) + "\n")

    # strip plot of every deployable candidate's gain, clipped at PB_CLIP
    lines = []
    for acc, style in ((False, "mark=o, gray"), (True, "mark=*, black")):
        pts = []
        states = sorted({x["state"] for x in B["slots"]})
        extra = (O["slots"] if O else []) + (NR["slots"] if NR else [])
        for r in (x for x in B["slots"] + extra if x["gate"] is None and x["accepted"] == acc):
            jit = (2 * states.index(r["state"]) + (r["variant"] == "B") - 7.5) * 0.025
            pts.append(f"({PB_X[r['group']] + jit:.2f},{max(r['G'], PB_CLIP) * 100:.2f})")
        lines.append(f"\\addplot[only marks, {style}, mark size=1.5pt] coordinates {{{' '.join(pts)}}};")
    (out / "phaseb_points.tex").write_text("\n".join(lines) + "\n")

    # exploratory, not registered: the no-attribution group without its one
    # catastrophic candidate (dropped, or set to 0 as if a gate had stopped it)
    rows = B["slots"]
    worst = min(rows, key=lambda r: r["G"])
    none_wo = [r["G"] for r in rows if r["group"] == "none" and r is not worst]
    zeroed = [{**r, "G": 0.0} if r is worst else r for r in rows]
    big = sorted([r for r in rows if r["G"] < -0.10], key=lambda r: r["G"])
    return {"n_slots": B["n_slots"], "deployable": sum(B["groups"][g]["deployable"] for g, _ in PB_GROUPS),
            "worst": {k: worst[k] for k in ("group", "state", "variant", "commit", "G", "evolve_S", "accepted")},
            "losses_over_10pp": [{k: r[k] for k in ("group", "state", "variant", "G", "accepted")} for r in big],
            "exploratory_none_without_worst_mean_G": sum(none_wo) / len(none_wo),
            "exploratory_vs_none_worst_set_to_0": {g: _state_contrast(zeroed, g, "none")
                                                   for g in ("rrsi", "first_write", "binary_search", "cf_search")},
            "S5_retail_first_write_vs_rrsi": next(c for c in S["S5_by_domain"]["tau2_retail"]
                                                  if c["g"] == "first_write" and c["h"] == "rrsi"),
            "S8": S["S8_agreement"], "S9": [x["diff"] for x in S["S9_drift"]],
            "S10_mean_G": S["S10_original_round"]["mean_G"],
            "S4_signflip": {f"{c['g']}-{c['h']}": c["p_signflip"] for c in S["S4_round_level"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main", nargs="?")
    ap.add_argument("--phaseb", default=str(PB_DEFAULT))
    ap.add_argument("--phaseb-accuracy", default=str(PB_ACC_DEFAULT))
    ap.add_argument("--phaseb-oracle", default=str(PB_ORACLE_DEFAULT))
    ap.add_argument("--phaseb-review3", default=str(PB_R3_DEFAULT))
    ap.add_argument("--robustness", default=str(ROB_DEFAULT))
    ap.add_argument("--rescue-prompt", default=str(RESCUE_DEFAULT))
    ap.add_argument("--gold-rule", default=str(GOLD_DEFAULT))
    ap.add_argument("--review2", default=str(REVIEW2_DEFAULT))
    ap.add_argument("--orig-control", default=str(ORIG_DEFAULT))
    ap.add_argument("--second-oracle", default=str(SECOND_DEFAULT))
    ap.add_argument("--factorial", default=str(FACTORIAL_DEFAULT))
    ap.add_argument("--admissibility", default=str(ADMISS_DEFAULT))
    ap.add_argument("--review4", default=str(REVIEW4_DEFAULT))
    ap.add_argument("--rescore", default="results/attrib/audit_rescore.json")
    ap.add_argument("--review5", default=str(REVIEW5_DEFAULT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper2" / "tables"))
    ap.add_argument("--whowhen", default="/mnt/project-files/results/attrib/whowhen_scores.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pb = phaseb_tables(Path(args.phaseb), out, Path(args.phaseb_accuracy), Path(args.phaseb_oracle),
                       Path(args.phaseb_review3)) \
        if Path(args.phaseb).exists() else None
    rob = robustness_tables(Path(args.robustness), out, Path(args.review2), Path(args.orig_control),
                            Path(args.second_oracle), Path(args.admissibility), Path(args.review4),
                            Path(args.rescore)) \
        if Path(args.robustness).exists() else None
    if Path(args.review4).exists():
        rob = {**(rob or {}), "review4": review4_tables(Path(args.review4), out)}
    if Path(args.review5).exists():
        rob = {**(rob or {}), "review5": review5_tables(Path(args.review5), out)}
    var = variant_tables(Path(args.rescue_prompt), Path(args.gold_rule), out)
    var["decomposition"] = decomp_tables(Path(args.review2), out)
    if not args.main:
        print(json.dumps({"phaseb": pb, "robustness": rob, "variants": var}, indent=1))
        return
    main_dir = Path(args.main)
    A = {d: json.loads((main_dir / f"analysis_{d}.json").read_text()) for d in DOMAINS}
    P = json.loads((main_dir / "analysis_pooled3.json").read_text())
    summary = {"n": P["n"], "retest": P["retest"], "primary": {d: A[d].get("primary") for d in DOMAINS},
               "primary_pooled3": P.get("primary"), "phaseb": pb, "robustness": rob,
               "variants": var}

    lines = []
    for i, (m, name, model) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        cells = [name, model] + [_ci(A[d]["methods"][m]["R_rescuable"]) for d in DOMAINS]
        cells += [_ci(P["methods"][m]["R_rescuable"]), _f(P["methods"][m]["exact"][0]),
                  f"{sum(A[d]['methods'][m]['dollars_per_failure'] for d in DOMAINS) / 3:.4f}"]
        lines.append(" & ".join(cells) + " \\\\")
    fac = Path(args.factorial)
    if fac.exists():
        F = json.loads(fac.read_text())["methods"]
        lines += ["\\midrule", "\\multicolumn{8}{l}{\\emph{Post hoc variants}} \\\\"]
        for m, name, model in POSTHOC_ROWS:
            cells = [name, model] + [_ci(F[m]["R"][d]) for d in DOMAINS]
            cells += [_ci(F[m]["R"]["pooled"]), _f(F[m]["exact"]), f"{F[m]['dollars_per_failure']:.4f}"]
            lines.append(" & ".join(cells) + " \\\\")
        summary["posthoc_rows"] = {m: F[m] for m, _, _ in POSTHOC_ROWS}
    (out / "main.tex").write_text("\n".join(lines) + "\n")
    summary["methods"] = {m: {"pooled": P["methods"][m]["R_rescuable"], "exact": P["methods"][m]["exact"],
                              **{d: A[d]["methods"][m]["R_rescuable"] for d in DOMAINS},
                              "dollars": {d: A[d]["methods"][m]["dollars_per_failure"] for d in DOMAINS}}
                          for m, _, _ in ROWS}

    K = {d: kinds(main_dir, d) for d in DOMAINS}
    summary["kinds"] = K
    order = {"tau2": ["utterance", "read", "write", "transfer"],
             "appworld": ["docs", "read", "write", "submit", "other"]}
    lines = {"tau2": [], "aw": []}
    for d in DOMAINS:
        g = "aw" if d == "appworld" else "tau2"
        ks = order["appworld" if g == "aw" else "tau2"][:4]
        na, nd = sum(K[d]["all"].values()), sum(K[d]["decisive"].values())
        cells = [DNAME[d], f"{na}/{nd}"] + [
            f"{K[d]['all'].get(k, 0) / na:.0%}/{K[d]['decisive'].get(k, 0)}".replace("%", "\\%") for k in ks]
        lines[g].append(" & ".join(cells) + " \\\\")
    for g, ls in lines.items():
        (out / f"kinds_{g}.tex").write_text("\n".join(ls) + "\n")

    lines = []
    for b in (8, 16, 24, 32, 40):
        m = f"search@{b}"
        cells = [str(b)] + [_f(A[d]["methods"][m]["R_rescuable"][0]) for d in DOMAINS] + [
            _f(P["methods"][m]["R_rescuable"][0]),
            f"{sum(A[d]['methods'][m]['dollars_per_failure'] for d in DOMAINS) / 3:.4f}"]
        lines.append(" & ".join(cells) + " \\\\")
    (out / "search.tex").write_text("\n".join(lines) + "\n")

    lines = []
    for i, (m, name, model) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        v = P["methods"][m]
        cells = [name, model, _ci(v["exact"]), _f(v["within1"]), _f(v["earliest"]), _f(v["share_rescuable"]),
                 _ci(v["R_all"])]
        lines.append(" & ".join(cells) + " \\\\")
    (out / "secondary.tex").write_text("\n".join(lines) + "\n")

    ww = Path(args.whowhen)
    if ww.exists():
        W = json.loads(ww.read_text())
        lines = []
        for m, name, model in ROWS:
            hc, ag = W.get(f"{m}/Hand-Crafted"), W.get(f"{m}/Algorithm-Generated")
            if not hc or not ag:
                continue
            lines.append(" & ".join([name, model, f"{hc['agent_acc']:.2f}", f"{hc['step_acc']:.2f}",
                                     f"{ag['agent_acc']:.2f}", f"{ag['step_acc']:.2f}"]) + " \\\\")
        (out / "whowhen.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
