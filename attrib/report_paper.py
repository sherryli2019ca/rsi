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
               (attrib/phaseb_analyze.py)
  robustness.tex, nullpos.tex  the ranking under other ground-truth choices and
               null-replay success by step position, from
               results/attrib/phaseA_robustness.json (attrib/robustness.py)
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
ROB_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "attrib" / "phaseA_robustness.json"
ROB_ROWS = [("main", "Registered"), ("K=2", "$K=2$"), ("K=1", "$K=1$"),
            ("threshold 0.25", "$\\theta=0.25$"), ("threshold 0.375", "$\\theta=0.375$"),
            ("threshold 0.625", "$\\theta=0.625$"), ("threshold 0.75", "$\\theta=0.75$"),
            ("unique rescuing step", "One rescuing step"), ("no null", "No null replays")]
PB_X = {g: i + 1 for i, (g, _) in enumerate([("none", 0), ("rrsi", 0), ("first_write", 0),
                                             ("binary_search", 0), ("cf_search", 0)])}
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


def _state_contrast(rows: list[dict], g: str, h: str) -> dict:
    """Mean over states of the difference in mean slot gain, t interval (df 7);
    the registered estimator of attrib/phaseb_analyze.py without its test."""
    import numpy as np
    states = sorted({r["state"] for r in rows})
    d = np.array([np.mean([r["G"] for r in rows if r["group"] == g and r["state"] == s]) -
                  np.mean([r["G"] for r in rows if r["group"] == h and r["state"] == s]) for s in states])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"estimate": float(d.mean()), "ci95": [float(d.mean() - T975_DF7 * se), float(d.mean() + T975_DF7 * se)]}


def robustness_tables(path: Path, out: Path) -> dict:
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


def phaseb_tables(path: Path, out: Path) -> dict:
    B = json.loads(path.read_text())
    S = B["secondary"]
    acc, rl = S["S7_phase_a_accuracy"], S["S4_round_level_means"]
    lines = []
    for g, name in PB_GROUPS:
        v = B["groups"][g]
        pa = "--" if g == "none" else " / ".join(_f(acc[d][g]) for d in ("tau2_retail", "tau2_airline"))
        dep = v["mean_G_deployable"]
        cells = [name, pa, f"{v['deployable']}/{v['slots']}", _pp(v["mean_G"]),
                 _pp(v["by_domain"]["tau2_retail"]), _pp(v["by_domain"]["tau2_airline"]),
                 _pp(dep) if dep is not None else "--",
                 f"{v['share_positive_deployable']:.0%}".replace("%", "\\%") if dep is not None else "--",
                 str(v["accepted_by_rrsi"]), _pp(rl[g])]
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
    (out / "phaseb_contrasts.tex").write_text("\n".join(lines) + "\n")

    # strip plot of every deployable candidate's gain, clipped at PB_CLIP
    lines = []
    for acc, style in ((False, "mark=o, gray"), (True, "mark=*, black")):
        pts = []
        states = sorted({x["state"] for x in B["slots"]})
        for r in (x for x in B["slots"] if x["gate"] is None and x["accepted"] == acc):
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
    ap.add_argument("--robustness", default=str(ROB_DEFAULT))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper2" / "tables"))
    ap.add_argument("--whowhen", default="/mnt/project-files/results/attrib/whowhen_scores.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pb = phaseb_tables(Path(args.phaseb), out) if Path(args.phaseb).exists() else None
    rob = robustness_tables(Path(args.robustness), out) if Path(args.robustness).exists() else None
    if not args.main:
        print(json.dumps({"phaseb": pb, "robustness": rob}, indent=1))
        return
    main_dir = Path(args.main)
    A = {d: json.loads((main_dir / f"analysis_{d}.json").read_text()) for d in DOMAINS}
    P = json.loads((main_dir / "analysis_pooled3.json").read_text())
    summary = {"n": P["n"], "retest": P["retest"], "primary": {d: A[d].get("primary") for d in DOMAINS},
               "primary_pooled3": P.get("primary"), "phaseb": pb, "robustness": rob}

    lines = []
    for i, (m, name, model) in enumerate(ROWS):
        if i in MIDRULES:
            lines.append("\\midrule")
        cells = [name, model] + [_ci(A[d]["methods"][m]["R_rescuable"]) for d in DOMAINS]
        cells += [_ci(P["methods"][m]["R_rescuable"]), _f(P["methods"][m]["exact"][0]),
                  f"{sum(A[d]['methods'][m]['dollars_per_failure'] for d in DOMAINS) / 3:.4f}"]
        lines.append(" & ".join(cells) + " \\\\")
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
