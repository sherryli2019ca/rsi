"""Zero-cost analyses for the eleventh review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r11 joint --traj r1=DIR r2=DIR r3=DIR [--json results/r11/report.json]
  python -m verify.posthoc_r11 table [--json results/r11/report.json] [--out paper/tables]

table: paper/tables/r10_loops.tex (verify/posthoc_r10.py) with a row for the
joint live-minus-offline contrast, written to r11_loops.tex.

  python -m verify.posthoc_r11 primary --report /home/user/e1r2/runs/verify/e1r_report.json [--out paper/tables]

primary: the main-text table of the primary analysis (r2+r3), rows selected
from the appendix tables e1r_rules.tex (decision value, episodes, Agree),
r8_accept.tex (recall of full evaluation's acceptances) and r7_margins.tex
(keeping the incumbent at the registered margin), with the registered Holm p
values of the two non-inferiority families from the registered report.
Written to r11_primary.tex.

  python -m verify.posthoc_r11 replay --run /home/user/il_f1 ... [--json results/r11/replay.json]

replay: summary of `python -m verify.bk replay` on the four concurrent
full-evaluation loops: rounds in which seqfull or the cost-aware seqcost would
choose differently from full evaluation, and their episodes as a share of full
evaluation's (60 per measured candidate).

joint: the live-minus-offline contrast of verify/posthoc_r10.py (live: the IL
comparison, seqfull minus full transfer at round 10, 6 v 6 loops; offline: the
matched airline rounds 0-9 of r1 to r3, seqfull minus full summed over ten
rounds) with one joint bootstrap instead of two independent ones. Each draw
resamples the 20 held-out tasks once and uses them everywhere, and resamples
the units within strata that keep shared units shared: the six sequential
loops; the four full loops that appear only live (f1..f4); r2 and r3, which
appear in the live full arm and in the offline analysis (a drawn unit enters
both); and r1 (offline only). Within each drawn offline trajectory the rounds
are resampled in incumbent blocks, as in the offline analysis; a variant keeps
the rounds fixed. For r2 and r3, full evaluation's ten round values sum, task
by task, to the trajectory's transfer at round 10, so with the rounds fixed
the shared part cancels exactly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import posthoc_r10 as R

A = R.A


def _round_diffs(T, ts):
    """Per round: seqfull minus full round value on the task multiset ts."""
    return [A.round_value(r, "seqfull", ts) - A.round_value(r, "full", ts) for r in T["rows"]]


def _offline_sum(T, block_of, ts, rng, fixed_rounds):
    rows = T["rows"]
    if fixed_rounds:
        idx = range(len(rows))
    else:
        groups = {}
        for i, row in enumerate(rows):
            groups.setdefault(block_of[T["name"]][row["t"]], []).append(i)
        keys = list(groups)
        idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
    d = [A.round_value(rows[i], "seqfull", ts) - A.round_value(rows[i], "full", ts) for i in idx]
    return 10 * float(np.mean(d))


def joint(trajs: dict, js: Path | None, B: int = 4000, seed: int = 21) -> dict:
    tables, block_of = R._merged(trajs, ["r1", "r2", "r3"], R.DOM, R.T_LAST)
    off_T = {T["name"].split("/")[0]: T for T in tables}
    arms, allr, ho, tasks, delta, new = R._il()
    S = list(arms["seq"])
    F_only = [n for n in arms["full"] if n not in ("r2", "r3")]
    shared = ["r2", "r3"]
    tf = lambda n, ts: float(np.mean([ho[n][0][t] - ho[n][1][t] for t in ts]))

    # point estimates and the exact identity for the shared units
    ident = {n: (tf(n, tasks), sum(A.round_value(r, "full") for r in off_T[n]["rows"])) for n in shared}
    live_pt = np.mean([tf(n, tasks) for n in S]) - np.mean([tf(n, tasks) for n in F_only + shared])
    off_pt = float(np.mean([10 * np.mean(_round_diffs(off_T[n], None)) for n in ("r1", "r2", "r3")]))
    res = {"live": 100 * float(live_pt), "offline_sum10": 100 * off_pt,
           "difference": 100 * float(live_pt - off_pt),
           "identity_full_sum_equals_transfer": {n: [100 * a, 100 * b] for n, (a, b) in ident.items()}}
    rng = np.random.default_rng(seed)
    for label, fixed in (("joint", False), ("joint_fixed_rounds", True)):
        live, off, gap = [], [], []
        for _ in range(B):
            ts = list(rng.choice(tasks, len(tasks)))
            s = list(rng.choice(S, len(S)))
            f = list(rng.choice(F_only, len(F_only)))
            sh = list(rng.choice(shared, len(shared)))
            lv = np.mean([tf(n, ts) for n in s]) - np.mean([tf(n, ts) for n in f + sh])
            of = np.mean([_offline_sum(off_T[n], block_of, ts, rng, fixed) for n in ["r1"] + sh])
            live.append(lv)
            off.append(of)
            gap.append(lv - of)
        pc = lambda x: [100 * float(np.percentile(x, 5)), 100 * float(np.percentile(x, 95))]
        res[label] = {"live_ci90": pc(live), "offline_ci90": pc(off), "difference_ci90": pc(gap),
                      "p_live_below_offline": float(np.mean(np.array(gap) < 0)),
                      "corr_live_offline": float(np.corrcoef(live, off)[0, 1])}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def table(js: Path, out: Path) -> None:
    res = json.loads(js.read_text())
    lo, hi = res["joint"]["difference_ci90"]
    row = (rf"Live minus recorded & & & & {res['difference']:+.1f} [{lo:+.1f}, {hi:+.1f}] "
           r"& -- & -- & -- \\").replace("-", "$-$").replace("$-$$-$", "--")
    lines = (out / "r10_loops.tex").read_text().splitlines()
    i = lines.index(r"\bottomrule")
    (out / "r11_loops.tex").write_text("\n".join(lines[:i] + [r"\midrule", row] + lines[i:]) + "\n")
    print((out / "r11_loops.tex").read_text())


def _rows(path: Path) -> list[list[str]]:
    return [[c.strip() for c in l.rstrip().removesuffix("\\\\").split("&")]
            for l in path.read_text().splitlines() if "&" in l]


def primary(report: Path, out: Path) -> None:
    ni = json.loads(report.read_text())["primary"]
    pv = {**{k: v["p_holm"] for k, v in ni["ni_primary"].items()},
          **{k: v["p_holm"] for k, v in ni["ni_sequential"].items()}}
    e1r = {(r[0], r[1]): r for r in _rows(out / "e1r_rules.tex")}
    acc = {r[0]: r[1].split("{")[0].strip() for r in _rows(out / "r8_accept.tex")}
    keep_p = next(r for r in _rows(out / "r7_margins.tex") if r[0] == "Keep incumbent")[5]
    sel = [("Full (RRSI)", ("Full (RRSI)", "all"), None, None),
           ("Keep incumbent", ("Keep incumbent", "0"), "Keep incumbent", "keep"),
           ("None", ("None", "0"), "None", None),
           ("LLM judge", ("LLM judge", "0"), "LLM judge", None),
           ("Sample@40", ("Sample", "40"), "Sample@40", "sample@40"),
           ("Net@40", ("Net", "40"), "Net@40", "net@40"),
           (r"Replay$-$null@40", (r"Replay$-$null", "40"), r"Replay$-$null@40", "replaynull@40"),
           ("Seq.\\ full", ("Sequential full", r"$\le$all"), "Sequential full", "seqfull"),
           ("Seq.\\ sample", ("Sequential sample", r"$\le$80"), "Sequential sample@80", "seqsample@80")]
    fmt = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}".rstrip("0") if x < 0.1 else f"{x:.2f}"
    rows = []
    for lab, ek, ak, pk in sel:
        r = e1r[ek]
        p = "--" if pk is None else rf"{keep_p}$^\dagger$" if pk == "keep" else fmt(pv[pk])
        rec = "1.00" if ak is None else acc[ak]
        d = "0" if pk == "keep" else re.sub(r"\\textbf\{([^}]*)\}", r"\1", r[3])
        rows.append(f"{lab} & {r[7]} & {d} & {p} & {r[6]} & {rec} \\\\")
        if lab in ("LLM judge", r"Replay$-$null@40"):
            rows.append(r"\midrule")
    (out / "r11_primary.tex").write_text("\n".join(rows) + "\n")
    print((out / "r11_primary.tex").read_text())


def replay(runs: list[Path], js: Path) -> dict:
    res = {}
    for key, ch in (("seqfull", "seq"), ("seqcost", "seqcost")):
        diff = n = full = 0
        for run in runs:
            for r in json.loads((run / "runs/verify/tau2_airline/replay.json").read_text()):
                diff += r["full_choice"] != r[f"{ch}_choice"]
                for c in r["candidates"].values():
                    n += c[f"n_{ch}"]
                    full += 60
        res[key] = {"rounds_different": diff, "episode_share": n / full, "full_episodes": full}
    res["rounds"] = 10 * len(runs)
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("joint")
    j.add_argument("--traj", nargs="+", required=True)
    j.add_argument("--json", default="results/r11/report.json")
    t = sub.add_parser("table")
    t.add_argument("--json", default="results/r11/report.json")
    t.add_argument("--out", default="paper/tables")
    q = sub.add_parser("primary")
    q.add_argument("--report", required=True)
    q.add_argument("--out", default="paper/tables")
    r = sub.add_parser("replay")
    r.add_argument("--run", nargs="+", required=True)
    r.add_argument("--json", default="results/r11/replay.json")
    a = ap.parse_args()
    if a.cmd == "joint":
        joint(dict(x.split("=", 1) for x in a.traj), Path(a.json))
    elif a.cmd == "table":
        table(Path(a.json), Path(a.out))
    elif a.cmd == "primary":
        primary(Path(a.report), Path(a.out))
    else:
        replay([Path(x) for x in a.run], Path(a.json))


if __name__ == "__main__":
    main()
