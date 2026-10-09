"""Repair experiment, checks for the third review of paper 2 (post hoc).

  python -m attrib.phaseb_review3 [--noread] [--out <json>]

source_runs  The 8 incumbent states come from 4 loop runs (r2/r3 x retail/
             airline), two checkpoints each (t5, t15). For each contrast the
             two checkpoint differences of a run are averaged first and the
             interval uses t over the 4 run differences (df 3), next to the
             registered interval over the 8 states; plus the smallest margin
             at which the 90% interval shows equivalence, and the correlation
             of the two checkpoints' differences within a run.
pointer_gain Does a more accurate pointer give a better candidate? For each
             step group (and RRSI's earliest cited step, and the oracle step),
             the mean rescue gain R at its pointers over the state's failing
             traces (attrib.phaseb_accuracy's measure, all 81 failures), against
             the candidates' held-out gain G: Pearson and Spearman over slots,
             raw and within state (both centred on the state's mean).
noread       (--noread, after every round and deployment of the two no-read
             groups, attrib.phaseb NOREAD_GROUPS): the groups table, contrasts
             oracle_fix_noread - rrsi_noread (the pointer, diagnosis and
             correction against RRSI's analysis, neither proposer able to read
             failing traces), rrsi_noread - rrsi and oracle_fix_noread - oracle
             (the effect of removing reading), round-level gains, and how often
             the proposers tried to read a failing trace.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from attrib.phaseb import (DOMAINS, GROUPS, NOREAD_GROUPS, PB, STATES, STEP_GROUPS, parse_state, picks,
                           run_dir, state_dir)
from attrib.phaseb_accuracy import rrsi_steps
from attrib.phaseb_analyze import contrast, groups_table, round_level, signflip, slots

T95 = {7: 1.894579, 3: 2.353363}
T975 = {7: 2.364624, 3: 3.182446}
ORACLE_GROUPS = GROUPS + ("oracle",)
N_PERM = 2000


def by_run(c: dict) -> dict:
    d = np.array(c["state_diffs"]) * 100
    runs = d.reshape(4, 2).mean(1)        # STATES are run-major: (t5, t15) pairs

    def ci(x, df):
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
        return ([round(m - T975[df] * se, 2), round(m + T975[df] * se, 2)],
                [round(m - T95[df] * se, 2), round(m + T95[df] * se, 2)])
    c95_8, c90_8 = ci(d, 7)
    c95_4, c90_4 = ci(runs, 3)
    return {"g": c["g"], "h": c["h"], "estimate_pp": round(float(d.mean()), 2),
            "states": {"ci95": c95_8, "ci90": c90_8, "eq_margin": round(max(map(abs, c90_8)), 2)},
            "runs": {"ci95": c95_4, "ci90": c90_4, "eq_margin": round(max(map(abs, c90_4)), 2),
                     "diffs": [round(float(x), 2) for x in runs]},
            "checkpoint_corr": round(float(np.corrcoef(d[0::2], d[1::2])[0, 1]), 2)
            if d[0::2].std() and d[1::2].std() else None}


def pointer_scores() -> dict:
    """{(group, state): mean R at the group's pointer over the state's failing traces}."""
    out = {}
    for dom in DOMAINS:
        gt = {g["fid"]: g for g in json.loads((PB / "gt" / dom / "a" / "result.json").read_text())["failures"]}
        pk = picks(dom)
        pk["rrsi"] = rrsi_steps(dom)
        for s in STATES:
            if parse_state(s)[1] != dom:
                continue
            fids = [f["fid"] for f in json.loads((state_dir(s) / "failures.json").read_text())]
            for g in ("rrsi", "oracle") + STEP_GROUPS:
                vals = []
                for f in fids:
                    k, R = pk[g].get(f), gt[f]["R"]
                    vals.append(R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0)
                out[(g, s)] = float(np.mean(vals))
    return out


def _spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def pointer_gain(rows: list[dict]) -> dict:
    ps = pointer_scores()
    out = {}
    for name, groups in (("step_and_oracle", STEP_GROUPS + ("oracle",)),
                         ("step_oracle_rrsi", STEP_GROUPS + ("oracle", "rrsi"))):
        rs = [r for r in rows if r["group"] in groups]
        x = np.array([ps[(r["group"], r["state"])] for r in rs])
        y = np.array([r["G"] * 100 for r in rs])
        sx = {s: x[[r["state"] == s for r in rs]].mean() for s in STATES}
        sy = {s: y[[r["state"] == s for r in rs]].mean() for s in STATES}
        xc = x - np.array([sx[r["state"]] for r in rs])
        yc = y - np.array([sy[r["state"]] for r in rs])
        dep = np.array([r["gate"] is None for r in rs])
        # permutation p: pointer scores shuffled between groups within each state
        rng = np.random.default_rng(0)
        obs, hits = abs(_spearman(xc, yc)), 0
        for _ in range(N_PERM):
            perm = {}
            for s_ in STATES:
                gs = list(groups)
                sh = rng.permutation(gs)
                perm.update({(g, s_): ps[(h, s_)] for g, h in zip(gs, sh)})
            xp = np.array([perm[(r["group"], r["state"])] for r in rs])
            xpc = xp - np.array([np.mean([perm[(g, r["state"])] for g in groups]) for r in rs])
            hits += abs(_spearman(xpc, yc)) >= obs - 1e-12
        out[name] = {"slots": len(rs), "pearson": round(float(np.corrcoef(x, y)[0, 1]), 3),
                     "spearman": round(_spearman(x, y), 3),
                     "pearson_within_state": round(float(np.corrcoef(xc, yc)[0, 1]), 3),
                     "spearman_within_state": round(_spearman(xc, yc), 3),
                     "p_perm_within_state": round((hits + 1) / (N_PERM + 1), 4),
                     "deployable_slots": int(dep.sum()),
                     "spearman_within_state_deployable": round(_spearman(xc[dep], yc[dep]), 3)}
    out["pointer_score_by_group"] = {g: round(float(np.mean([ps[(g, s)] for s in STATES])), 3)
                                     for g in ("rrsi", "oracle") + STEP_GROUPS}
    return out


def read_attempts(g: str) -> dict:
    tried, drafts, succ = [], 0, []
    for s in STATES:
        t = parse_state(s)[2]
        fails = {str(f["task_id"]) for f in json.loads((state_dir(s) / "failures.json").read_text())}
        for v in "AB":
            p = run_dir(g, s) / f"r{t}" / v / "proposal.json"
            if not p.exists():
                continue
            log = json.loads(p.read_text())["log"]
            log = ast.literal_eval(log) if isinstance(log, str) else log
            reads = [a for a in log if a.get("action") == "read_trace"]
            drafts += 1
            tried.append(sum(str(a.get("task_id")) in fails for a in reads))
            succ.append(sum(str(a.get("task_id")) not in fails for a in reads))
    return {"drafts": drafts, "failing_read_attempts_per_draft": round(float(np.mean(tried)), 2) if tried else None,
            "success_reads_per_draft": round(float(np.mean(succ)), 2) if succ else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noread", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    groups = ORACLE_GROUPS + (NOREAD_GROUPS if args.noread else ())
    rows = slots(groups)
    pairs = [(g, h) for h in ("none", "rrsi") for g in STEP_GROUPS] + [("oracle", "rrsi"), ("oracle", "none")]
    res = {"source_runs": [by_run(contrast(rows, (g,), (h,))) for g, h in pairs]
           + [by_run(contrast(rows, ("oracle",), STEP_GROUPS))],
           "pointer_gain": pointer_gain(rows)}
    if args.noread:
        rl = round_level(rows)
        npairs = [("oracle_fix_noread", "rrsi_noread"), ("rrsi_noread", "rrsi"), ("oracle_fix_noread", "oracle"),
                  ("oracle_fix_noread", "rrsi"), ("rrsi_noread", "none"), ("oracle_fix_noread", "none")]
        res["noread"] = {
            "groups": groups_table(rows, groups),
            "slot_level": [{**contrast(rows, (g,), (h,)), "by_run": by_run(contrast(rows, (g,), (h,)))}
                           for g, h in npairs],
            "round_level": [(lambda c: {**c, "p_signflip": signflip(c), "p_perm": None})(contrast(rl, (g,), (h,)))
                            for g, h in npairs],
            "round_level_means": {g: float(np.mean([r["G"] for r in rl if r["group"] == g])) for g in groups},
            "reads": {g: read_attempts(g) for g in NOREAD_GROUPS + ("rrsi", "oracle")},
            "slots": [r for r in rows if r["group"] in NOREAD_GROUPS]}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "noread"}, indent=1))
    if args.noread:
        print(json.dumps({k: v for k, v in res["noread"].items() if k != "slots"}, indent=1, default=str))


if __name__ == "__main__":
    main()
