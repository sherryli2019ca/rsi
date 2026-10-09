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
and prints the numbers the text quotes.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper2" / "tables"))
    ap.add_argument("--whowhen", default="/mnt/project-files/results/attrib/whowhen_scores.json")
    args = ap.parse_args()
    main_dir, out = Path(args.main), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A = {d: json.loads((main_dir / f"analysis_{d}.json").read_text()) for d in DOMAINS}
    P = json.loads((main_dir / "analysis_pooled3.json").read_text())
    summary = {"n": P["n"], "retest": P["retest"], "primary": {d: A[d].get("primary") for d in DOMAINS},
               "primary_pooled3": P.get("primary")}

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
