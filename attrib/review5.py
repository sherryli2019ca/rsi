"""Post hoc analyses for the fifth review of paper 2 (not registered).

  python -m attrib.review5 /home/user/attrib_runs/main [--out results/attrib/phaseAB_review5.json] [--only ...]

targeted   Repair experiment (attrib.phaseb): did candidates fix the training
           failures their evidence was about? For every candidate with a full
           evaluation on the training tasks (RRSI's gate; both seeds), its
           success on the tasks of the state's failing traces against the
           incumbent's success on the same tasks and seeds, on the traces with
           an oracle step (the oracle groups' evidence) and on the rest, and on
           the incumbent's passing tasks; then whether that training fix rate
           tracks the held-out gain G.
maxgain    Counterfactual search with the metric's selection rule
           (attrib.search_maxgain): rescue gain of the step each variant
           answers under the registered stopping rule (first flip) and under
           the largest estimated gain, on the rescuable failures; and, from the
           registered runs alone, the largest gain among the suspects they
           tested.
blindfull  Full profiles under the oracle without grading on a random 100
           failures (attrib.blind_full): rescuability, best rescue steps by
           kind, every method's score on the failures rescuable under this
           oracle, against the registered profile on the same failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from attrib.analyze import _boot, _mean, load


def _t(x) -> dict:
    """Mean with t intervals over units and the exact sign-flip p."""
    import itertools
    x = np.asarray(x, dtype=float)
    df = len(x) - 1
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    q95, q90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    if len(x) <= 14:
        flips = np.array(list(itertools.product((1, -1), repeat=len(x))))
    else:  # Monte Carlo sign flips
        flips = np.random.default_rng(0).choice((1, -1), size=(20000, len(x)))
    p = float(np.mean(np.abs((flips * x).mean(1)) >= abs(m) - 1e-12))
    return {"mean": round(m, 2), "ci95": [round(m - q95 * se, 2), round(m + q95 * se, 2)],
            "ci90": [round(m - q90 * se, 2), round(m + q90 * se, 2)], "p_signflip": round(p, 4)}


# ---------------------------------------------------------------- targeted --
def _succ(job: Path, task: str) -> list[int]:
    out = []
    for sd in ("s0", "s1"):
        p = job / sd / f"{task}.json"
        if p.exists():
            out.append(int(json.loads(p.read_text()).get("reward", 0) >= 1))
    return out


def targeted() -> dict:
    from attrib.phaseb import GROUPS, NOREAD_GROUPS, PB, STATES, parse_state, picks, run_dir
    from attrib.phaseb_analyze import slots
    groups = GROUPS + ("oracle",) + NOREAD_GROUPS
    rows = slots(groups)
    ora = {}
    for d in ("tau2_retail", "tau2_airline"):
        ora.update(picks(d)["oracle"])
    out_rows = []
    for r in rows:
        s, g = r["state"], r["group"]
        t = parse_state(s)[2]
        rd = run_dir(g, s)
        job = rd / "jobs" / f"r{t}{r['variant']}"
        if not (job / "s0").exists():
            continue
        inc = rd / "jobs" / json.loads((PB / "states" / s / "prep.json").read_text())["job"]
        fails = json.loads((PB / "states" / s / "failures.json").read_text())
        tasks = sorted({f["task_id"] for f in fails})
        ftask = {f["task_id"]: f["fid"] for f in fails}
        passing = sorted({p.stem for p in (inc / "s0").glob("*.json")} - set(tasks))

        def rate(ts, j):
            v = [x for tk in ts for x in _succ(j, tk)]
            return float(np.mean(v)) if v else None
        with_or = [tk for tk in tasks if isinstance(ora.get(ftask[tk]), int)]
        without = [tk for tk in tasks if tk not in with_or]
        pp = rd / f"r{t}" / r["variant"] / "proposal.json"
        pred = set()
        if pp.exists():
            for e in json.loads(pp.read_text()).get("edits") or []:
                pred |= {str(x) for x in (e.get("predicted_affected") or [])}
        pred_fail = sorted(pred & set(tasks))
        out_rows.append({**{k: r[k] for k in ("group", "state", "variant", "gate", "accepted")},
                         "G": round(r["G"] * 100, 2),
                         "fail_cand": rate(tasks, job), "fail_inc": rate(tasks, inc),
                         "oracle_cand": rate(with_or, job), "oracle_inc": rate(with_or, inc),
                         "rest_cand": rate(without, job), "rest_inc": rate(without, inc),
                         "pass_cand": rate(passing, job), "pass_inc": rate(passing, inc),
                         "n_fail": len(tasks), "n_oracle": len(with_or), "predicted": sorted(pred),
                         "pred_fail": pred_fail, "pred_oracle": sorted(pred & set(with_or)),
                         "pred_cand": rate(pred_fail, job), "pred_inc": rate(pred_fail, inc),
                         "unpred_cand": rate([tk for tk in tasks if tk not in pred], job),
                         "unpred_inc": rate([tk for tk in tasks if tk not in pred], inc)})

    def summ(rs, a, b):
        d = [x[a] - x[b] for x in rs if x[a] is not None and x[b] is not None]
        return {"n": len(d), "mean": round(float(np.mean(d)) * 100, 1) if d else None}
    per = {}
    for g in groups:
        rs = [x for x in out_rows if x["group"] == g]
        per[g] = {"evaluated": len(rs),
                  "targets_per_candidate": round(float(np.mean([len(x["pred_fail"]) for x in rs])), 2) if rs else None,
                  "oracle_traces_targeted": round(float(np.mean([len(x["pred_oracle"]) / x["n_oracle"]
                                                              for x in rs if x["n_oracle"]])), 3) if rs else None,
                  "failing_targeted": round(float(np.mean([len(x["pred_fail"]) / x["n_fail"] for x in rs])), 3) if rs else None,
                  "predicted_tasks": summ(rs, "pred_cand", "pred_inc"),
                  "unpredicted_tasks": summ(rs, "unpred_cand", "unpred_inc"),
                  "failing_tasks": summ(rs, "fail_cand", "fail_inc"),
                  "oracle_traces": summ(rs, "oracle_cand", "oracle_inc"),
                  "other_traces": summ(rs, "rest_cand", "rest_inc"),
                  "passing_tasks": summ(rs, "pass_cand", "pass_inc"),
                  "G_mean": round(float(np.mean([x["G"] for x in rs])), 2) if rs else None}
    # matched contrasts over states: group mean (over its evaluated candidates) minus reference
    def contrast(g, h, a, b):
        d = []
        for s in STATES:
            ok = lambda x: x[a] is not None and x[b] is not None  # noqa: E731
            ga = [x[a] - x[b] for x in out_rows if x["group"] in g and x["state"] == s and ok(x)]
            ha = [x[a] - x[b] for x in out_rows if x["group"] in h and x["state"] == s and ok(x)]
            if ga and ha:
                d.append((np.mean(ga) - np.mean(ha)) * 100)
        return {"states": len(d), **_t(np.array(d))} if len(d) > 2 else {"states": len(d)}
    cons = {}
    for g, h in ((("oracle_fix_noread",), ("rrsi_noread",)), (("oracle",), ("rrsi",)),
                 (("oracle", "oracle_fix_noread"), ("rrsi", "rrsi_noread")),
                 (("first_write", "binary_search", "cf_search"), ("rrsi",)), (("oracle",), ("none",))):
        key = "+".join(g) + " - " + "+".join(h)
        cons[key] = {"predicted_tasks": contrast(g, h, "pred_cand", "pred_inc"),
                     "oracle_traces": contrast(g, h, "oracle_cand", "oracle_inc"),
                     "failing_tasks": contrast(g, h, "fail_cand", "fail_inc"),
                     "passing_tasks": contrast(g, h, "pass_cand", "pass_inc")}
    def pmu(rs):
        d = [(x["pred_cand"] - x["pred_inc"]) - (x["unpred_cand"] - x["unpred_inc"]) for x in rs
             if None not in (x["pred_cand"], x["pred_inc"], x["unpred_cand"], x["unpred_inc"])]
        return {"n": len(d), **_t(np.array(d) * 100)} if len(d) > 2 else {"n": len(d)}
    pred_vs_unpred = {"all": pmu(out_rows), "oracle_groups": pmu([x for x in out_rows if x["group"] in
                                                                 ("oracle", "oracle_fix_noread")]),
                      "rrsi_groups": pmu([x for x in out_rows if x["group"] in ("rrsi", "rrsi_noread")])}

    def share(g, h):
        d = []
        for s in STATES:
            a_ = [len(x["pred_oracle"]) / x["n_oracle"] for x in out_rows if x["group"] in g and x["state"] == s and x["n_oracle"]]
            b_ = [len(x["pred_oracle"]) / x["n_oracle"] for x in out_rows if x["group"] in h and x["state"] == s and x["n_oracle"]]
            if a_ and b_:
                d.append((np.mean(a_) - np.mean(b_)) * 100)
        return {"states": len(d), **_t(np.array(d))} if len(d) > 2 else {"states": len(d)}
    targeting = {"oracle_fix_noread - rrsi_noread": share(("oracle_fix_noread",), ("rrsi_noread",)),
                 "oracle - rrsi": share(("oracle",), ("rrsi",)),
                 "oracle+oracle_fix_noread - rrsi+rrsi_noread": share(("oracle", "oracle_fix_noread"),
                                                                      ("rrsi", "rrsi_noread"))}
    dep = [x for x in out_rows if x["gate"] is None and x["fail_cand"] is not None]
    fix = np.array([x["fail_cand"] - x["fail_inc"] for x in dep])
    G = np.array([x["G"] for x in dep])
    rho, p = stats.spearmanr(fix, G)
    pas = np.array([x["pass_cand"] - x["pass_inc"] for x in dep])
    rho2, p2 = stats.spearmanr(pas, G)
    return {"rows": out_rows, "groups": per, "contrasts": cons,
            "deployable_n": len(dep),
            "spearman_fix_vs_G": [round(float(rho), 3), round(float(p), 4)],
            "spearman_passing_vs_G": [round(float(rho2), 3), round(float(p2), 4)],
            "predicted_minus_unpredicted": pred_vs_unpred, "targeting_oracle_traces": targeting}


# ----------------------------------------------------------------- maxgain --
def maxgain(main: Path, data: dict, rng) -> dict:
    out = {}
    scores = {}
    task = {}
    for d, v in data.items():
        R = {f: g["R"] for f, g in v["gt"].items() if g["decisive"] is not None}
        task.update({(d, f): v["fails"][f]["task_id"] for f in R})
        for var in ("search", "search_informed", "aligned"):
            p = main / d / "search_maxgain" / var / "result.json"
            if not p.exists():
                continue
            res = {x["fid"]: x for x in json.loads(p.read_text())["failures"]}
            for rule in ("first_flip", "maxgain"):
                key = f"{var}:{rule}"
                for f, r in R.items():
                    k = res.get(f, {}).get(rule)
                    scores.setdefault(key, {})[(d, f)] = r[k] if isinstance(k, int) and 0 <= k < len(r) else 0.0
            for f in R:
                x = res.get(f)
                if x:
                    scores.setdefault(f"{var}:replays", {})[(d, f)] = x["replays"]
        # registered runs alone: the largest gain among tested suspects
        for var in ("search", "search_informed"):
            p = main / d / var / "result.json"
            if not p.exists():
                continue
            res = {x["fid"]: x for x in json.loads(p.read_text())["failures"]}
            for f, r in R.items():
                x = res.get(f)
                if not x:
                    continue
                tested = [e for e in x["log"] if "corrected" in e]
                est = [((e["corrected"] - e.get("null", 0)) / 4, -e["i"], e["step"]) for e in tested]
                k = max(est)[2] if est else (x["suspects"][0] if x["suspects"] else None)
                scores.setdefault(f"{var}:tested_max", {})[(d, f)] = r[k] if isinstance(k, int) and 0 <= k < len(r) else 0.0
    keys = sorted(scores)
    for k in keys:
        vals = scores[k]
        out[k] = {"n": len(vals), "mean": round(float(np.mean(list(vals.values()))), 4)}
    # paired differences
    pairs = [("search:maxgain", "search:first_flip"), ("search_informed:maxgain", "search_informed:first_flip"),
             ("aligned:maxgain", "aligned:first_flip"), ("aligned:maxgain", "search_informed:first_flip"),
             ("aligned:maxgain", "search:first_flip")]
    diffs = {}
    for a, b in pairs:
        if a in scores and b in scores:
            common = sorted(set(scores[a]) & set(scores[b]))
            x = np.array([scores[a][c] - scores[b][c] for c in common])
            clusters = [f"{c[0]}/{task[c]}" for c in common]
            diffs[f"{a} - {b}"] = {"n": len(x), "mean": round(float(x.mean()), 4),
                                   "ci95": _cluster_ci(x, clusters, rng)}
    return {"scores": out, "diffs": diffs, "_raw": {k: {f"{d}/{f}": v for (d, f), v in s.items()}
                                                     for k, s in scores.items()}}


def _cluster_ci(x: np.ndarray, clusters: list[str], rng, n=10000) -> list[float]:
    ids = sorted(set(clusters))
    idx = {c: [i for i, cc in enumerate(clusters) if cc == c] for c in ids}
    sums = np.array([x[idx[c]].sum() for c in ids])
    cnts = np.array([len(idx[c]) for c in ids])
    m = []
    for _ in range(n):
        b = rng.integers(0, len(ids), len(ids))
        m.append(sums[b].sum() / cnts[b].sum())
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    main_dir = Path(a.main)
    rng = np.random.default_rng(0)
    res = {}
    if a.out and Path(a.out).exists():
        res = json.loads(Path(a.out).read_text())
    want = set(a.only or ("targeted", "maxgain", "blindfull"))
    data = None
    if want & {"maxgain", "blindfull"}:
        data = {d: load(main_dir / d) for d in ("tau2_retail", "tau2_airline", "appworld")}
    if "targeted" in want:
        res["targeted"] = targeted()
    if "maxgain" in want:
        res["maxgain"] = maxgain(main_dir, data, rng)
    if "blindfull" in want:
        res["blindfull"] = blindfull(main_dir, data, rng)
    s = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s)
    print(json.dumps({k: (v if k != "maxgain" else {kk: vv for kk, vv in v.items() if kk != "_raw"})
                      for k, v in res.items() if k != "targeted"}, indent=1, default=str)[:6000])
    if "targeted" in res:
        t = res["targeted"]
        print(json.dumps({k: t[k] for k in t if k != "rows"}, indent=1)[:6000])


def blindfull(main: Path, data: dict, rng) -> dict:
    """Full grading-blind profiles on the random sample against the registered
    profiles of the same failures."""
    import random
    from attrib.factorial import informed
    from attrib.review4 import BLIND_METHODS, REG10, _kendall, _kind_of
    prng = random.Random(0)
    rows, missing = [], []
    for d, v in data.items():
        p = main / d / "gt" / "blind" / "result_full.json"
        if not p.exists():
            missing.append(d)
            continue
        res = json.loads(p.read_text())
        v["picks"]["search_inf@40"], v["picks"]["search_inf@0"] = informed(main, d, v)[:2]
        sr = json.loads((main / d / "search" / "result.json").read_text())["failures"]
        v["picks"]["search@0"] = {x["fid"]: (x["suspects"][0] if x.get("suspects") else None) for x in sr}
        kind = _kind_of(d)
        for f in res["sample"]:
            g = v["gt"][f]
            bl = [s["R"] for s in sorted(res["failures"][f], key=lambda s: s["k"])]
            trace = {s["index"]: s for s in json.loads(Path(v["fails"][f]["trace"]).read_text())["steps"]}
            kb = max(range(len(bl)), key=lambda k: (bl[k], -k))
            r = {"domain": d, "fid": f, "cluster": f"{d}/{v['fails'][f]['task_id']}", "n_steps": len(bl),
                 "reg_rescuable": g["decisive"] is not None, "blind_rescuable": bl[kb] >= 0.5,
                 "blind_best": kb, "blind_max": bl[kb], "reg_best": g["decisive"],
                 "blind_best_kind": kind(trace[kb]) if bl[kb] >= 0.5 else None,
                 "reg_best_kind": kind(trace[g["decisive"]]) if g["decisive"] is not None else None,
                 "blind_best_pos": kb / max(1, len(bl) - 1)}
            for m in BLIND_METHODS:
                k = v["picks"][m].get(f)
                if isinstance(k, dict):
                    k = k.get("step")
                ok = isinstance(k, int) and 0 <= k < len(bl)
                r[m] = bl[k] if ok else 0.0
                r[m + "|reg"] = g["R"][k] if ok else 0.0
            rows.append(r)
    if not rows:
        return {"missing": missing}

    def means(rs, suffix=""):
        return {m: round(_mean([r[m + suffix] for r in rs]), 4) for m in BLIND_METHODS} if rs else {}
    br = [r for r in rows if r["blind_rescuable"]]
    rr = [r for r in rows if r["reg_rescuable"]]
    mb, mr = means(br), means(rr, "|reg")
    allb, allr = means(rows), means(rows, "|reg")

    def rank(mm, ms=REG10):
        return 1 + sum(mm[x] > mm["first_write"] for x in ms if x != "first_write")

    def diff(rs, a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rs],
                     lambda xs: _mean([x["d"] for x in xs]), prng) if len(rs) > 2 else None
    llm10 = [m for m in REG10 if m not in ("first_write", "last_step")]
    best_b = max(llm10, key=lambda m: mb[m]) if mb else None
    kinds = {}
    for r in br:
        kinds.setdefault(f"{r['domain']}/{r['blind_best_kind']}", 0)
        kinds[f"{r['domain']}/{r['blind_best_kind']}"] += 1
    rkinds = {}
    for r in rr:
        rkinds.setdefault(f"{r['domain']}/{r['reg_best_kind']}", 0)
        rkinds[f"{r['domain']}/{r['reg_best_kind']}"] += 1
    by_dom = {d: {"n": sum(r["domain"] == d for r in rows),
                  "reg_rescuable": sum(r["domain"] == d and r["reg_rescuable"] for r in rows),
                  "blind_rescuable": sum(r["domain"] == d and r["blind_rescuable"] for r in rows),
                  "both": sum(r["domain"] == d and r["reg_rescuable"] and r["blind_rescuable"] for r in rows)}
              for d in sorted({r["domain"] for r in rows})}
    return {"missing": missing, "n": len(rows), "by_domain": by_dom,
            "blind_rescuable": len(br), "reg_rescuable": len(rr),
            "both_rescuable": sum(r["reg_rescuable"] and r["blind_rescuable"] for r in rows),
            "blind_only": sum(r["blind_rescuable"] and not r["reg_rescuable"] for r in rows),
            "same_best_step_if_both": round(_mean([r["blind_best"] == r["reg_best"] for r in rows
                                                   if r["reg_rescuable"] and r["blind_rescuable"]]) or 0, 3),
            "blind_best_kinds": kinds, "reg_best_kinds": rkinds,
            "blind_best_pos_mean": round(_mean([r["blind_best_pos"] for r in br]), 3) if br else None,
            "scores_blind_rescuable": mb, "scores_reg_rescuable_registered": mr,
            "scores_all_blind": allb, "scores_all_registered": allr,
            "rank_first_write_blind": rank(mb) if mb else None,
            "rank_first_write_registered_sample": rank(mr) if mr else None,
            "rank_first_write_all_blind": rank(allb), "rank_first_write_all_registered": rank(allr),
            "kendall_blind_vs_registered_sample": _kendall(mb, mr, REG10) if mb and mr else None,
            "kendall_all": _kendall(allb, allr, REG10),
            "best_llm_blind": best_b,
            "first_write_minus_blind": {m: diff(br, "first_write", m) for m in
                                        ("binary_search_pro", "search@40", "binary_search_gain_pro",
                                         "search_inf@40", best_b) if m},
            "first_write_minus_all_blind": {m: diff(rows, "first_write", m) for m in
                                            ("binary_search_pro", "search@40", "binary_search_gain_pro",
                                             "search_inf@40")},
            "_rows": rows}


if __name__ == "__main__":
    main()
