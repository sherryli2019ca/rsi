"""Analysis of Phase B (attrib/PREREGISTRATION_B.md), run once every round and
every deployment has finished.

  python -m attrib.phaseb analyze [--out <json>]   (or python -m attrib.phaseb_analyze)

Slot = (group, state, variant A/B), 80 in all. G = held-out S of the candidate
minus held-out S of the state's incumbent (both deployed in Phase B), when the
candidate passed RRSI's gates and the common error check; 0 otherwise (the
incumbent stays). A contrast g - h is the mean over the 8 states of
mean_v G(g, s, v) - mean_v G(h, s, v); p from a permutation of group labels
within each state (10,000, seed 0); 95% CI from the t distribution over the 8
state differences (df 7).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.phaseb import (DOMAINS, GROUPS, PB, STATES, STEP_GROUPS, TRAJ, candidates,  # noqa: E402
                           cell_done, heldout_dir, incumbent, parse_state, picks, run_dir, src_dir)

N_PERM = 10_000
T975_DF7, T95_DF7 = 2.364624, 1.894579      # t quantiles, df 7
EQ_MARGIN = 0.025
PRIMARY = [(g, h) for h in ("none", "rrsi") for g in STEP_GROUPS]
PHASE_A = {"first_write": "first_write", "binary_search": "binary_search_pro", "cf_search": "search@40",
           "rrsi": "rrsi_digest_pro"}


def heldout_S(dom: str, commit: str, root: Path | None = None) -> float:
    p = (root / "jobs" / "heldout" / commit[:12] / "eval.json") if root else heldout_dir(dom, commit) / "eval.json"
    if not p.exists():
        raise SystemExit(f"no held-out deployment of {commit[:12]} ({dom}): run `phaseb deploy` first")
    return json.loads(p.read_text())["S"]


def slots(groups=GROUPS) -> list[dict]:
    rows = []
    for s in STATES:
        dom = parse_state(s)[1]
        inc = incumbent(s)[0]["commit"]
        s_inc = heldout_S(dom, inc)
        for g in groups:
            if not cell_done(g, s):
                raise SystemExit(f"round not finished: {g}:{s}")
            t = parse_state(s)[2]
            fr = json.loads((run_dir(g, s) / "frontier.json").read_text())
            nxt = next(x for x in fr["trajectory"] if x["t"] == t + 1)
            for c in candidates(g, s):
                c["accepted"] = bool(c["commit"]) and nxt["commit"] != inc and \
                    nxt["commit"].startswith(c["commit"][:7])
                c["G"] = heldout_S(dom, c["commit"]) - s_inc if c["gate"] is None else 0.0
                rows.append(c)
    return rows


def _cells(rows: list[dict], key="G") -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault((r["group"], r["state"]), []).append(r[key])
    return out


def contrast(rows: list[dict], g: tuple, h: tuple, states=STATES, seed: int = 0) -> dict:
    """g, h: tuples of groups (pooled)."""
    cells = _cells(rows)
    d, pools = [], []
    for s in states:
        a = [x for gg in g for x in cells[(gg, s)]]
        b = [x for hh in h for x in cells[(hh, s)]]
        d.append(np.mean(a) - np.mean(b))
        pools.append((np.array(a + b), len(a)))
    d = np.array(d)
    est = float(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(N_PERM):
        tot = 0.0
        for vals, na in pools:
            p = rng.permutation(vals)
            tot += p[:na].mean() - p[na:].mean()
        hits += abs(tot / len(pools)) >= abs(est) - 1e-12
    n = len(states)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t975, t95 = (T975_DF7, T95_DF7) if n == 8 else (2.776445, 2.131847) if n == 4 else (float("nan"),) * 2
    ci = [est - t975 * se, est + t975 * se]
    ci90 = [est - t95 * se, est + t95 * se]
    return {"g": "+".join(g), "h": "+".join(h), "estimate": est, "ci95": ci, "p_perm": (hits + 1) / (N_PERM + 1),
            "se": se, "state_diffs": d.tolist(), "ci90": ci90,
            "equivalent_within_2.5pp": bool(ci90[0] > -EQ_MARGIN and ci90[1] < EQ_MARGIN)}


def signflip(c: dict) -> float:
    """Exact sign-flip p over the state differences: the test for S4, whose
    unit is the cell (one round outcome per group and state). contrast()'s
    slot permutation splits a cell's duplicated value across groups, so its
    p_perm is not valid for S4 (deviation logged 2026-10-09)."""
    d = np.array(c["state_diffs"])
    obs = abs(d.mean())
    flips = list(itertools.product((1, -1), repeat=len(d)))
    return sum(abs((d * np.array(f)).mean()) >= obs - 1e-12 for f in flips) / len(flips)


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def groups_table(rows: list[dict], groups=GROUPS) -> dict:
    out = {}
    for g in groups:
        rs = [r for r in rows if r["group"] == g]
        dep = [r for r in rs if r["gate"] is None]
        gates: dict = {}
        for r in rs:
            gates[r["gate"] or "deployable"] = gates.get(r["gate"] or "deployable", 0) + 1
        out[g] = {"slots": len(rs), "gates": gates, "mean_G": float(np.mean([r["G"] for r in rs])),
                  "deployable": len(dep),
                  "mean_G_deployable": float(np.mean([r["G"] for r in dep])) if dep else None,
                  "share_positive_deployable": float(np.mean([r["G"] > 0 for r in dep])) if dep else None,
                  "accepted_by_rrsi": sum(r["accepted"] for r in rs),
                  "by_domain": {d: float(np.mean([r["G"] for r in rs if parse_state(r["state"])[1] == d]))
                                for d in DOMAINS}}
    return out


def round_level(rows: list[dict]) -> list[dict]:
    """S4: the gain of the candidate RRSI's own rule accepted (0 if none, or if
    it fails the common error check)."""
    rr = [{**r, "R": r["G"] if r["accepted"] else 0.0} for r in rows]
    by = {}
    for r in rr:
        by.setdefault((r["group"], r["state"]), 0.0)
        by[(r["group"], r["state"])] += r["R"]
    # one value per cell, duplicated over its two slots so contrast() averages it
    return [{"group": g, "state": s, "G": v} for (g, s), v in by.items() for _ in (0, 1)]


def agreement() -> dict:
    out = {}
    for dom in DOMAINS:
        pk = picks(dom)
        fids = sorted(pk["first_write"])
        out[dom] = {"n": len(fids)}
        for a, b in itertools.combinations(STEP_GROUPS, 2):
            out[dom][f"{a}=={b}"] = float(np.mean([pk[a].get(f) == pk[b].get(f) for f in fids]))
        out[dom]["all_three_agree"] = float(np.mean([len({pk[g].get(f) for g in STEP_GROUPS}) == 1 for f in fids]))
    return out


def drift() -> list[dict]:
    """S9: Phase B incumbent deployment vs E1R's deployment of the same commit."""
    out = []
    for s in STATES:
        traj, dom, _ = parse_state(s)
        inc = incumbent(s)[0]["commit"]
        e1r = TRAJ[traj] / "runs" / "verify" / dom
        try:
            orig = heldout_S(dom, inc, e1r)
        except SystemExit:
            orig = None
        new = heldout_S(dom, inc)
        out.append({"state": s, "phaseB": new, "e1r": orig, "diff": None if orig is None else new - orig})
    return out


def original_round(rows_unused=None) -> dict:
    """S10: the E1R round's own two candidates at each state (RRSI's evidence,
    drafted during E1R), held-out gains from E1R's deployments, same gate rules."""
    from verify.state import evaluation, rounds
    vals = []
    for s in STATES:
        traj, dom, t = parse_state(s)
        rnd = next(r for r in rounds(src_dir(s)) if r.t == t)
        e1r = TRAJ[traj] / "runs" / "verify" / dom
        s_inc = heldout_S(dom, rnd.inc_commit, e1r)
        for c in rnd.cands:
            g = 0.0
            if c.measured:
                rate = (evaluation(src_dir(s), c.job).extra or {}).get("harness_error_rate") or 0.0
                if rate <= 0.02:
                    g = heldout_S(dom, c.commit, e1r) - s_inc
            vals.append({"state": s, "variant": c.variant, "G": g, "measured": c.measured})
    return {"slots": vals, "mean_G": float(np.mean([v["G"] for v in vals])),
            "measured": sum(v["measured"] for v in vals)}


def phase_a_accuracy() -> dict:
    out = {}
    for dom in DOMAINS:
        p = Path("/home/user/attrib_runs/main") / f"analysis_{dom}.json"
        if p.exists():
            a = json.loads(p.read_text())["methods"]
            out[dom] = {g: a[m]["R_rescuable"][0] for g, m in PHASE_A.items() if m in a}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--out", default=str(PB / "analysis.json"))
    args, _ = ap.parse_known_args()
    rows = slots()
    prim = [contrast(rows, (g,), (h,)) for g, h in PRIMARY]
    for c, p in zip(prim, holm([c["p_perm"] for c in prim])):
        c["p_holm"] = p
    res = {"n_slots": len(rows), "groups": groups_table(rows), "primary": prim,
           "secondary": {
               "S1_rrsi_vs_none": contrast(rows, ("rrsi",), ("none",)),
               "S2_step_groups_vs_none": contrast(rows, STEP_GROUPS, ("none",)),
               "S4_round_level": [{**c, "p_signflip": signflip(c), "p_perm": None}
                                  for c in (contrast(round_level(rows), (g,), (h,)) for g, h in PRIMARY)],
               "S4_round_level_means": {g: float(np.mean([r["G"] for r in round_level(rows) if r["group"] == g]))
                                        for g in GROUPS},
               "S5_by_domain": {d: [contrast(rows, (g,), (h,), states=[s for s in STATES if parse_state(s)[1] == d])
                                    for g, h in PRIMARY] for d in DOMAINS},
               "S7_phase_a_accuracy": phase_a_accuracy(),
               "S8_agreement": agreement(),
               "S9_drift": drift(),
               "S10_original_round": original_round()},
           "slots": rows}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "slots"}, indent=1))


if __name__ == "__main__":
    main()
