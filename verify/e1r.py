"""Registered analyses of experiment E1R (verify/PREREGISTRATION_E1R.md).

  python -m verify.e1r prepare --traj r2=/home/user/e1r2/runs [...]
  python -m verify.e1r report  --traj r1=/home/user/e1/runs r2=... r3=... r4=... \
      [--primary r2,r3] [--pooled r1,r2,r3] [--out FILE]

prepare: for each trajectory (a runs directory with rrsi/ and verify/), the
  registered E1 analysis (verify/analyze.py, unchanged), the analysis under the
  common error check with the sequential rules (verify/posthoc_e1.py guard) and
  its robustness summary (posthoc_e1.py robust), each written into that
  trajectory's verify/ directory; outputs that exist are kept.

report: from those outputs,
  - primary (the trajectories in --primary, rounds of both domains): decision
    values under the common error check; non-inferiority of sample@40, net@40
    and replaynull@40 to full evaluation (margin 0.75 points per round,
    one-sided, Holm over the three), and of seqfull and seqsample@80
    (addendum 2, Holm over the two); bootstrap over blocks of rounds that share
    an incumbent and over held-out tasks, within domain and trajectory
    (B = 2000); episodes per round, contrasts with sample@40/@80, Net(10^4);
  - pooled (--pooled) with the same bootstrap, and each trajectory's own
    decision values, so the spread between trajectories sits next to the
    pooled estimate;
  - descriptive, every trajectory (r4 against the others): candidates' mean
    deployment gain and share with a positive gain, the critic's rejection
    rate, the share of measured candidates failing the error check, transfer
    of the final harness, and the mix of measured candidates by corrected
    component label and by files touched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import analyze as A
from .posthoc_e1 import CODE_FILES, _dv, _per_round
from .state import ROOT, rounds

GUARD = "e1_analysis_guard_posthoc.json"
NI_RULES = ("sample@40", "net@40", "replaynull@40")
SEQ_NI = ("seqfull", "seqsample@80")
REPORT_RULES = ("full", "sample@40", "sample@80", "net@40", "replaynull@40", "seqfull", "seqsample@80",
                "nonecheck@10", "judgecheck@10", "judge", "none")
DOMAINS = ("tau2_retail", "tau2_airline")


def _trajs(pairs: list[str]) -> dict:
    return {p.split("=", 1)[0]: Path(p.split("=", 1)[1]) for p in pairs}


def prepare(trajs: dict) -> None:
    py = sys.executable
    for name, P in trajs.items():
        rr, vv = P / "rrsi", P / "verify"
        steps = [("e1_analysis.json", ["-m", "verify.analyze", "--domains", ",".join(DOMAINS)]),
                 (GUARD, ["-m", "verify.posthoc_e1", "guard"]),
                 ("e1_robust_posthoc.json", ["-m", "verify.posthoc_e1", "robust"])]
        for out, cmd in steps:
            if (vv / out).exists():
                print(f"{name}: {out} exists, kept")
                continue
            print(f"{name}: {' '.join(cmd[1:])}", flush=True)
            subprocess.run([py, *cmd, "--runs", str(rr), "--out", str(vv)], cwd=ROOT, check=True)


def _tables(trajs: dict, names: list[str]):
    tables, block_of = [], {}
    for n in names:
        P = trajs[n]
        for T in json.loads((P / "verify" / GUARD).read_text())["tables"]:
            key = f"{n}/{T['name']}"
            tables.append({**T, "name": key})
            block_of[key] = {r.t: r.inc_commit for r in rounds(P / "rrsi" / T["name"])}
    return tables, block_of


def _episodes(tables, rule):
    return float(np.mean([row["cost"][rule]["episodes"] for T in tables for row in T["rows"]]))


def holm(ps: dict) -> dict:
    return A.holm(ps)


def analyse(tables, block_of, seed: int = 2) -> dict:
    point = _dv(_per_round(tables))
    rng = np.random.default_rng(seed)
    boots = [_dv(_per_round(tables, rng, block_of)) for _ in range(A.B)]
    pc = lambda xs: [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    rules = [r for r in REPORT_RULES if r in point]
    res = {"n_rounds": {T["name"]: len(T["rows"]) for T in tables},
           "n_blocks": {T["name"]: len(set(block_of[T["name"]][row["t"]] for row in T["rows"])) for T in tables},
           "decision_value": {r: {"est": point[r], "ci90_block": pc([b[r] for b in boots]),
                                  "episodes_per_round": _episodes(tables, r)} for r in rules}}

    def family(names):
        out, ps = {}, {}
        for r in names:
            if r not in point:
                continue
            xs = np.array([b[r] - b["full"] for b in boots])
            ps[r] = float(np.mean(xs <= -A.MARGIN))
            out[r] = {"est": point[r] - point["full"], "ci90_block": pc(xs), "p_ni": ps[r]}
        for r, p in holm(ps).items():
            out[r]["p_holm"] = p
        return out
    res["ni_primary"] = family(NI_RULES)
    res["ni_sequential"] = family(SEQ_NI)
    res["contrasts"] = {}
    for a, b in (("seqfull", "sample@40"), ("seqfull", "sample@80"), ("seqsample@80", "sample@40"),
                 ("seqsample@80", "sample@80"), ("full", "none")):
        if a in point and b in point:
            xs = np.array([x[a] - x[b] for x in boots])
            res["contrasts"][f"{a} - {b}"] = {"est": point[a] - point[b], "ci90_block": pc(xs)}
    # Net(10^4) of keeping each rule's choices against keeping the incumbent
    pr = _per_round(tables)
    res["net_1e4"] = {}
    for r in ("full", "sample@40", "net@40", "replaynull@40", "seqfull", "seqsample@80"):
        if r not in pr:
            continue
        x = pr[r]
        for v in (1, 10):
            rate = v * np.mean([a for a, _, _ in x]) - np.mean([d for _, d, _ in x])
            cv = np.mean([c for _, _, c in x])
            res["net_1e4"][f"{r},v={v}"] = {"net": float(1e4 * rate - cv),
                                            "breakeven_N": float(cv / rate) if rate > 0 else None}
    return res


# ------------------------------------------------------------- descriptive --
def describe(name: str, P: Path) -> dict:
    from domains.tau2.common import COMPONENT_SIGNALS
    from rrsi.components import normalize
    from .state import evaluation
    tabs = {T["name"]: T for T in json.loads((P / "verify" / GUARD).read_text())["tables"]}
    reg = json.loads((P / "verify" / "e1_analysis.json").read_text())
    rob = P / "verify" / "e1_robust_posthoc.json"
    mde = json.loads(rob.read_text())["candidate_mde80_median"] if rob.exists() else None
    comps, files, cands = {}, {"code": 0, "text": 0}, []
    drafts = rejects = 0
    for D in DOMAINS:
        k = int(A.load_domain(D).cfg.get("heldout_k", 4))
        held = lambda c: {t: x["rewards"] for t, x in json.loads(
            (P / "verify" / D / "jobs" / "heldout" / c[:12] / "eval.json").read_text())["per_task"].items()}
        rows = {row["t"]: row for row in tabs[D]["rows"]}
        for rnd in rounds(P / "rrsi" / D):
            for c in rnd.cands:
                drafts += 1
                rejects += c.gate_failure == "critic_reject"
                if not c.measured:
                    continue
                diff = subprocess.run(["git", "-C", str(ROOT), "diff", rnd.inc_commit, c.commit, "--",
                                       f"domains/{D}/harness"], capture_output=True, text=True).stdout
                # RRSI's own labelling with the corrected signals: each declared edit
                # component checked against the diff (as registered for E1R)
                pp = P / "rrsi" / D / f"r{rnd.t}" / c.variant / "proposal.json"
                decl = [e.get("component") for e in
                        (json.loads(pp.read_text()).get("edits", []) if pp.exists() else c.edits)]
                for lab in sorted({normalize(d, diff, COMPONENT_SIGNALS) for d in decl or [None]}):
                    comps[lab] = comps.get(lab, 0) + 1
                touched = {ln.split(" b/")[-1].rsplit("/", 1)[-1] for ln in diff.splitlines()
                           if ln.startswith("diff --git")}
                files["code" if touched & CODE_FILES else "text"] += 1
                rate = (evaluation(P / "rrsi" / D, c.job).extra or {}).get("harness_error_rate") or 0.0
                d = rows.get(rnd.t, {}).get("dep", {}).get(c.variant)
                if not d:
                    continue
                hc, hi, half = held(c.commit), held(rnd.inc_commit), k // 2
                cands.append({"gain": float(np.mean(list(d.values()))), "fails_check": rate > 0.02,
                              "h1": float(np.mean([np.mean(hc[t][:half]) - np.mean(hi[t][:half]) for t in hi])),
                              "h2": float(np.mean([np.mean(hc[t][half:k]) - np.mean(hi[t][half:k]) for t in hi]))})

    def split_half(sel):
        if len(sel) < 3:
            return None
        r = float(np.corrcoef([x["h1"] for x in sel], [x["h2"] for x in sel])[0, 1])
        return {"r": r, "spearman_brown": 2 * r / (1 + r) if r > -1 else None, "n": len(sel)}
    g = np.array([x["gain"] for x in cands])
    p1 = {k: {kk: vv for kk, vv in v.items() if kk != "task_diffs"} for k, v in reg["primary_P1"].items()}
    n = len(cands)
    return {"drafts": drafts, "candidates_measured": n,
            "critic_reject_rate": rejects / drafts if drafts else None,
            "mean_deployment_gain": float(g.mean()) if n else None,
            "sd_deployment_gain": float(g.std(ddof=1)) if n > 1 else None,
            "share_positive": float(np.mean(g > 0)) if n else None,
            "share_failing_error_check": float(np.mean([x["fails_check"] for x in cands])) if n else None,
            "candidate_mde80": mde,
            "share_gain_beyond_mde80": float(np.mean(np.abs(g) > mde)) if n and mde else None,
            "split_half": split_half(cands),
            "split_half_without_check_failures": split_half([x for x in cands if not x["fails_check"]]),
            "transfer": p1, "labels": comps, "files": files}


def report(trajs: dict, primary: list[str], pooled: list[str], out: Path | None) -> dict:
    res = {"trajectories": {n: str(p) for n, p in trajs.items()}}
    have = [n for n in trajs if (trajs[n] / "verify" / GUARD).exists()]
    if all(n in have for n in primary):
        res["primary"] = analyse(*_tables(trajs, primary))
    if all(n in have for n in pooled):
        res["pooled"] = analyse(*_tables(trajs, pooled))
    res["per_trajectory"] = {n: _dv(_per_round(_tables(trajs, [n])[0])) for n in have}
    res["descriptive"] = {n: describe(n, trajs[n]) for n in have}
    if out:
        out.write_text(json.dumps(res, indent=1))
    pp = lambda x: f"{100 * x:+.2f}"
    for part in ("primary", "pooled"):
        if part not in res:
            continue
        R = res[part]
        print(f"== {part}: rounds {R['n_rounds']} blocks {R['n_blocks']}")
        for r, d in R["decision_value"].items():
            print(f"  DV {r:14s} {pp(d['est'])} [{pp(d['ci90_block'][0])}, {pp(d['ci90_block'][1])}] "
                  f"episodes/round {d['episodes_per_round']:.0f}")
        for fam in ("ni_primary", "ni_sequential"):
            for r, d in R[fam].items():
                print(f"  NI {r:14s} - full {pp(d['est'])} [{pp(d['ci90_block'][0])}, {pp(d['ci90_block'][1])}] "
                      f"p {d['p_ni']:.3f} Holm {d['p_holm']:.3f}")
    for n, dv in res["per_trajectory"].items():
        print(f"  {n}: " + "  ".join(f"{r} {pp(dv[r])}" for r in ("full", "sample@40", "net@40",
                                                                  "replaynull@40", "seqfull", "none") if r in dv))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prepare", "report"))
    ap.add_argument("--traj", nargs="+", required=True, help="name=runs directory")
    ap.add_argument("--primary", default="r2,r3")
    ap.add_argument("--pooled", default="r1,r2,r3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    trajs = _trajs(args.traj)
    if args.cmd == "prepare":
        prepare(trajs)
    else:
        report(trajs, [x for x in args.primary.split(",") if x], [x for x in args.pooled.split(",") if x],
               Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
