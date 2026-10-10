"""Post-hoc analyses for the fourth review of paper 2 (not registered).

  python -m attrib.review4 /home/user/attrib_runs/main [--out <json>]

accepted   Repair experiment, the outcome of the round: the candidate RRSI's own
           rule accepted (its held-out gain, 0 if none). Per group the rounds
           with an accepted candidate and their mean gain; contrasts against
           RRSI's analysis and against no attribution, and pooled contrasts
           (every group with a step pointer; the two oracle-evidence groups),
           with the 8 states (t, df 7) and with the two checkpoints of each
           run averaged first (t, df 3), exact sign-flip p over states and
           over runs, and the smallest margin the 90% interval excludes.
strata     Phase A: each method's rescue score by the kind of the best rescue
           step (domain x kind), on the 157 rescuable failures; the same split
           pooled into state-changing best steps (writes, and AppWorld's
           submission) and others, with first write minus the best LLM
           variants inside each, and a kind-balanced mean (every domain x kind
           cell with at least 5 failures weighted equally).
matched    Counterfactual search against the judges at matched information:
           search@40 against the judges without grading information, and
           search with grading information against the registered judges.
admiss     The admissibility screen (attrib.admissibility) by domain and kind
           of step, among corrections at the first write's step and at the
           best rescue step of rescuable failures.
scope      AppWorld: lines and API calls of each correction against the
           observed code block, and the share of changed lines, for all
           corrections and for those at best rescue steps.
splithalf  Best-step agreement between halves of the data on all 300 failures:
           best steps chosen on two oracle samples and two null replays and
           re-chosen on the other two (all 12 splits), among failures
           rescuable in the first half and in both; the in-sample and
           out-of-sample gain of the step chosen. The same for the repair
           experiment's 81 failing traces and its oracle pointer (the step of
           largest gain, attrib.phaseb.picks); and for the correction given to
           the oracle-fix group (the sample with most successful corrected
           replays), chosen on one corrected replay and scored on the other.
blind      After attrib.review4_runs blind: every method scored by the
           grading-blind oracle's rescue gain at its named step on the 157
           rescuable failures (fixed cohort), against the registered oracle
           at the same steps: ranks, Kendall's tau, first write minus the best
           LLM variants (cluster bootstrap), gains and proposal shares at all
           scanned steps and at best rescue steps by domain and kind.
fresh      After attrib.review4_runs fresh: rescue gain of each repair group's
           pointer in the registered profile and in the fresh one (new oracle
           samples and replays), on all 81 traces and the 33 rescuable ones;
           the oracle pointer against the others on fresh data; and the
           supplied oracle-fix correction's registered gain against its gain
           in 8 fresh corrected and 8 fresh null replays.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from attrib.analyze import _boot, _mean, load

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
K = 4
METHODS = ("first_write", "last_step", "all_at_once_pro", "all_at_once_pro_think", "binary_search_pro",
           "search@40", "all_at_once_gain_pro", "binary_search_gain_pro", "search_inf@40")
LLM = METHODS[2:]
STATE_CHANGING = ("write", "submit")
T95 = {7: 1.894579, 3: 2.353363}
T975 = {7: 2.364624, 3: 3.182446}


# ---------------------------------------------------------------- accepted --
def _t(x: np.ndarray) -> dict:
    df = len(x) - 1
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
    c95 = [round(m - T975[df] * se, 2), round(m + T975[df] * se, 2)]
    c90 = [round(m - T95[df] * se, 2), round(m + T95[df] * se, 2)]
    flips = itertools.product((1, -1), repeat=len(x))
    p = float(np.mean([abs((x * np.array(f)).mean()) >= abs(m) - 1e-12 for f in flips]))
    return {"ci95": c95, "ci90": c90, "eq_margin": round(max(map(abs, c90)), 2), "p_signflip": round(p, 4)}


def accepted() -> dict:
    from attrib.phaseb import GROUPS, NOREAD_GROUPS, STATES, STEP_GROUPS
    from attrib.phaseb_analyze import contrast, round_level, slots
    groups = GROUPS + ("oracle",) + NOREAD_GROUPS
    rows = slots(groups)
    rl = round_level(rows)
    per = {}
    for g in groups:
        acc = [r for r in rows if r["group"] == g and r["accepted"]]
        per[g] = {"rounds": len(STATES), "accepted": len(acc),
                  "mean_gain_accepted": round(float(np.mean([r["G"] for r in acc])) * 100, 2) if acc else None,
                  "round_gain": round(float(np.mean([r["G"] for r in rl if r["group"] == g])) * 100, 2),
                  "accepted_gains": [round(r["G"] * 100, 2) for r in acc]}
    pairs = [((g,), (h,)) for h in ("rrsi", "none") for g in groups if g not in (h,)]
    pairs += [(STEP_GROUPS + ("oracle",), ("rrsi",)), (("oracle", "oracle_fix_noread"), ("rrsi", "rrsi_noread")),
              (("oracle_fix_noread",), ("rrsi_noread",))]
    out = []
    for g, h in pairs:
        c = contrast(rl, g, h)
        d = np.array(c["state_diffs"]) * 100
        runs = d.reshape(4, 2).mean(1)
        out.append({"g": "+".join(g), "h": "+".join(h), "estimate_pp": round(float(d.mean()), 2),
                    "states": _t(d), "runs": {**_t(runs), "diffs": [round(float(x), 2) for x in runs]}})
    return {"groups": per, "contrasts": out}


# ------------------------------------------------------------------ strata --
def _kind_of(d: str):
    from attrib.report_paper import aw_kind, tau2_kind
    return aw_kind if d == "appworld" else tau2_kind


def _picks(main: Path, data: dict) -> None:
    from attrib.factorial import informed
    for d, v in data.items():
        v["picks"]["search_inf@40"] = informed(main, d, v)[0]


def _rows(data: dict) -> list[dict]:
    rows = []
    for d, v in data.items():
        kind = _kind_of(d)
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            steps = {s["index"]: s for s in json.loads(Path(v["fails"][f]["trace"]).read_text())["steps"]}
            r = {"domain": d, "fid": f, "cluster": f"{d}/{v['fails'][f]['task_id']}",
                 "kind": kind(steps[g["decisive"]])}
            fw = v["picks"]["first_write"].get(f)
            r["fw_is_best"] = fw == g["decisive"]
            for m in METHODS:
                k = v["picks"][m].get(f)
                r[m] = g["R"][k] if isinstance(k, int) and 0 <= k < len(g["R"]) else 0.0
            rows.append(r)
    return rows


def strata(main: Path, data: dict, rng) -> dict:
    rows = _rows(data)
    cells = defaultdict(list)
    for r in rows:
        cells[(r["domain"], r["kind"])].append(r)
    by_cell = {f"{d}/{k}": {"n": len(rs), "fw_names_best": sum(r["fw_is_best"] for r in rs),
                            **{m: round(_mean([r[m] for r in rs]), 3) for m in METHODS}}
               for (d, k), rs in sorted(cells.items())}

    def diff(rs, a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rs],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    split = {}
    for name, keep in (("state_changing", lambda r: r["kind"] in STATE_CHANGING),
                       ("other", lambda r: r["kind"] not in STATE_CHANGING)):
        rs = [r for r in rows if keep(r)]
        split[name] = {"n": len(rs), "by_domain": {d: sum(r["domain"] == d for r in rs) for d in DOMAINS},
                       "mean": {m: round(_mean([r[m] for r in rs]), 3) for m in METHODS},
                       "first_write_minus": {m: diff(rs, "first_write", m)
                                             for m in ("binary_search_pro", "binary_search_gain_pro",
                                                       "all_at_once_gain_pro", "search_inf@40")}}
    big = [rs for rs in cells.values() if len(rs) >= 5]
    balanced = {m: round(_mean([_mean([r[m] for r in rs]) for rs in big]), 3) for m in METHODS}
    share_fw = sum(r["first_write"] for r in rows if r["kind"] in STATE_CHANGING) / sum(r["first_write"] for r in rows)
    return {"n": len(rows), "by_cell": by_cell, "split": split,
            "kind_balanced": {"cells": len(big), "mean": balanced},
            "first_write_score_share_from_state_changing_best_steps": round(share_fw, 3)}


def matched(data: dict, rng) -> dict:
    """Counterfactual search against the judges at matched information: neither
    reads the grading (search@40 vs the blind judges), or both do (informed
    search vs the registered judges), on the 157 rescuable failures."""
    rows = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            r = {"cluster": f"{d}/{v['fails'][f]['task_id']}"}
            for m in ("search@40", "search_inf@40", "all_at_once_blind_pro", "binary_search_blind_pro",
                      "all_at_once_pro", "binary_search_pro"):
                k = v["picks"][m].get(f)
                r[m] = g["R"][k] if isinstance(k, int) and 0 <= k < len(g["R"]) else 0.0
            rows.append(r)

    def diff(a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    return {"no_grading": {b: diff("search@40", b) for b in ("all_at_once_blind_pro", "binary_search_blind_pro")},
            "grading": {b: diff("search_inf@40", b) for b in ("all_at_once_pro", "binary_search_pro")}}


# ------------------------------------------------------------------ admiss --
def admiss(main: Path, data: dict) -> dict:
    from attrib.admissibility import _rates, collect
    out = {}
    for d, v in data.items():
        rows = collect(main, d, v)
        fw = v["picks"]["first_write"]
        at_fw = [r for r in rows if r["rescuable"] and fw.get(r["fid"]) == r["k"]]
        at_best = [r for r in rows if r["decisive"]]
        out[d] = {"first_write_step": {"all": _rates(at_fw)}, "best_rescue_step": {"all": _rates(at_best)}}
        for name, rs in (("first_write_step", at_fw), ("best_rescue_step", at_best)):
            by = defaultdict(list)
            for r in rs:
                by[r["kind"]].append(r)
            out[d][name]["by_kind"] = {k: _rates(x) for k, x in sorted(by.items())}
        if d == "appworld":
            out[d]["scope_rows"] = [{k: r[k] for k in ("fid", "k", "i", "kind", "gain", "decisive", "obs_calls",
                                                        "corr_calls", "obs_lines", "corr_lines")}
                                    | {"code": (r["action"] or {}).get("code") or ""} for r in rows]
    return out


# ------------------------------------------------------------------- scope --
def scope(main: Path, data: dict, rows: list[dict]) -> dict:
    import difflib
    v = data["appworld"]
    obs_code = {}
    for f in v["gt"]:
        obs_code[f] = {s["index"]: s.get("code") or "" for s in
                       json.loads(Path(v["fails"][f]["trace"]).read_text())["steps"]}

    def summ(rs):
        if not rs:
            return None
        ch = []
        for r in rs:
            a = [x for x in obs_code[r["fid"]][r["k"]].splitlines() if x.strip()]
            b = [x for x in r["code"].splitlines() if x.strip()]
            sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
            kept = sum(blk.size for blk in sm.get_matching_blocks())
            ch.append(1 - kept / max(len(b), 1))
        return {"n": len(rs), "obs_lines": round(_mean([r["obs_lines"] for r in rs]), 2),
                "corr_lines": round(_mean([r["corr_lines"] for r in rs]), 2),
                "obs_calls": round(_mean([r["obs_calls"] for r in rs]), 2),
                "corr_calls": round(_mean([r["corr_calls"] for r in rs]), 2),
                "corr_lines_median": float(np.median([r["corr_lines"] for r in rs])),
                "corr_lines_p90": float(np.percentile([r["corr_lines"] for r in rs], 90)),
                "share_calls_le_observed": round(_mean([r["corr_calls"] <= max(r["obs_calls"], 1) for r in rs]), 3),
                "share_calls_le_observed_plus1": round(_mean([r["corr_calls"] <= max(r["obs_calls"], 1) + 1
                                                              for r in rs]), 3),
                "share_new_lines": round(_mean(ch), 3)}
    return {"all": summ(rows), "best_rescue_step": summ([r for r in rows if r["decisive"]]),
            "best_rescue_step_by_kind": {k: summ([r for r in rows if r["decisive"] and r["kind"] == k])
                                         for k in sorted({r["kind"] for r in rows if r["decisive"]})}}


# --------------------------------------------------------------- splithalf --
def _outcomes(rdir: Path) -> dict:
    """(fid, k) -> {"null": {j: ok}, "corr": {i: {rep: ok}}} from replay files."""
    out = defaultdict(lambda: {"null": {}, "corr": defaultdict(dict)})
    for p in rdir.glob("*_k*_*.json"):
        fid, rest = p.stem.rsplit("_k", 1)
        parts = rest.split("_")
        if not parts[0].isdigit() or len(parts) < 2:
            continue
        x = json.loads(p.read_text())
        ok = int(x.get("reward", 0) >= 1)
        if parts[1].startswith("n"):
            out[(fid, int(parts[0]))]["null"][int(parts[1][1:])] = ok
        elif (x.get("replay") or {}).get("forced"):
            out[(fid, int(parts[0]))]["corr"][int(parts[1][1:])][int(parts[2])] = ok
    return out


def _profile(g: dict, oc: dict, fid: str, S: tuple, N: tuple, reps=(0, 1)) -> list[float]:
    R = []
    for st in g["steps"]:
        o = oc.get((fid, st["k"]))
        nul = [o["null"][j] for j in N if o and j in o["null"]]
        tot = 0.0
        if nul:
            null = sum(nul) / len(nul)
            for i in S:
                s = st["samples"][i]
                c = o["corr"].get(i, {})
                if s.get("verdict") == "mistake" and all(r in c for r in reps):
                    tot += sum(c[r] for r in reps) / len(reps) - null
        R.append(tot / len(S))
    return R


def _splits():
    for SA in itertools.combinations(range(K), 2):
        SB = tuple(i for i in range(K) if i not in SA)
        for NA, NB in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
            yield SA, NA, SB, NB


def _agreement(gts: dict, ocs: dict, thr: float = 0.5) -> dict:
    """gts: {domain: {fid: profile}}; the best step chosen on half A, re-chosen on half B."""
    per = []
    for SA, NA, SB, NB in _splits():
        a = {"resc_A": 0, "resc_both": 0, "same_A": 0, "same_both": 0, "within1_both": 0,
             "in": [], "out": [], "B_rescuable_given_A": 0}
        for d, gt in gts.items():
            for f, g in gt.items():
                RA = _profile(g, ocs[d], f, SA, NA)
                RB = _profile(g, ocs[d], f, SB, NB)
                if not RA or max(RA) < thr:
                    continue
                kA, kB = RA.index(max(RA)), RB.index(max(RB))
                a["resc_A"] += 1
                a["same_A"] += kA == kB
                a["in"].append(RA[kA])
                a["out"].append(RB[kA])
                if max(RB) >= thr:
                    a["resc_both"] += 1
                    a["same_both"] += kA == kB
                    a["within1_both"] += abs(kA - kB) <= 1
        per.append(a)
    m = lambda key: round(float(np.mean([p[key] for p in per])), 2)
    return {"splits": len(per), "rescuable_A": m("resc_A"), "rescuable_both": m("resc_both"),
            "same_best_step_given_A": round(float(np.mean([p["same_A"] / p["resc_A"] for p in per])), 3),
            "same_best_step_both": round(float(np.mean([p["same_both"] / p["resc_both"] for p in per])), 3),
            "within1_both": round(float(np.mean([p["within1_both"] / p["resc_both"] for p in per])), 3),
            "gain_in_sample": round(float(np.mean([np.mean(p["in"]) for p in per])), 3),
            "gain_out_of_sample": round(float(np.mean([np.mean(p["out"]) for p in per])), 3)}


def _oracle_pointer(gts: dict, ocs: dict) -> dict:
    """Phase B's oracle pointer (largest gain, any value above 0) chosen on half A, scored on half B."""
    per = []
    for SA, NA, SB, NB in _splits():
        ins, outs = [], []
        for d, gt in gts.items():
            for f, g in gt.items():
                RA = _profile(g, ocs[d], f, SA, NA)
                RB = _profile(g, ocs[d], f, SB, NB)
                if not RA or max(RA) <= 0:
                    ins.append(0.0)
                    outs.append(0.0)
                    continue
                kA = RA.index(max(RA))
                ins.append(RA[kA])
                outs.append(RB[kA])
        per.append((np.mean(ins), np.mean(outs)))
    full = [max(g["R"]) if g["R"] and max(g["R"]) > 0 else 0.0 for gt in gts.values() for g in gt.values()]
    return {"traces": len(full), "full_data_pointer_R": round(float(np.mean(full)), 3),
            "half_in_sample": round(float(np.mean([a for a, _ in per])), 3),
            "half_out_of_sample": round(float(np.mean([b for _, b in per])), 3)}


def _oracle_fix(gts: dict, ocs: dict) -> dict:
    """The oracle-fix correction: at the full-data oracle step, the sample with the most
    successful corrected replays, chosen on replay r and scored on the other replay."""
    rows = []
    for d, gt in gts.items():
        for f, g in gt.items():
            if not g["R"] or max(g["R"]) <= 0:
                continue
            k = g["R"].index(max(g["R"]))
            st, o = g["steps"][k], ocs[d].get((f, g["steps"][k]["k"]))
            if not o or not o["null"]:
                continue
            null = sum(o["null"].values()) / len(o["null"])
            cand = [i for i, s in enumerate(st["samples"]) if s.get("verdict") == "mistake"
                    and all(r in o["corr"].get(i, {}) for r in (0, 1))]
            if not cand:
                continue
            both = max(cand, key=lambda i: (sum(o["corr"][i].values()), -i))
            r = {"fid": f, "domain": d, "chosen_both": sum(o["corr"][both].values()) / 2 - null}
            for sel, ev in ((0, 1), (1, 0)):
                i = max(cand, key=lambda i: (o["corr"][i][sel], -i))
                r[f"in_{sel}"] = o["corr"][i][sel] - null
                r[f"out_{sel}"] = o["corr"][i][ev] - null
            rows.append(r)
    return {"traces": len(rows), "chosen_in_sample_gain": round(_mean([r["chosen_both"] for r in rows]), 3),
            "chosen_on_one_replay_in_sample": round(_mean([(r["in_0"] + r["in_1"]) / 2 for r in rows]), 3),
            "chosen_on_one_replay_out_of_sample": round(_mean([(r["out_0"] + r["out_1"]) / 2 for r in rows]), 3)}


def splithalf(main: Path, data: dict) -> dict:
    from attrib.phaseb import PB
    gts = {d: v["gt"] for d, v in data.items()}
    ocs = {d: _outcomes(main / d / "gt" / "a" / "replays") for d in DOMAINS}
    pb_gt, pb_oc = {}, {}
    for d in ("tau2_retail", "tau2_airline"):
        pb_gt[d] = {g["fid"]: g for g in json.loads((PB / "gt" / d / "a" / "result.json").read_text())["failures"]}
        pb_oc[d] = _outcomes(PB / "gt" / d / "a" / "replays")
    return {"phaseA": _agreement(gts, ocs),
            "phaseA_by_domain": {d: _agreement({d: gts[d]}, ocs) for d in DOMAINS},
            "phaseB_best_step": _agreement(pb_gt, pb_oc),
            "phaseB_oracle_pointer": _oracle_pointer(pb_gt, pb_oc),
            "phaseB_oracle_fix": _oracle_fix(pb_gt, pb_oc)}


# ------------------------------------------------------------------- blind --
BLIND_METHODS = ("first_write", "last_step", "all_at_once_flash", "all_at_once_pro", "all_at_once_pro_think",
                 "step_by_step_pro", "binary_search_pro", "study1_pro", "rrsi_digest_pro", "search@40",
                 "all_at_once_gain_pro", "binary_search_gain_pro", "all_at_once_blind_pro",
                 "binary_search_blind_pro", "search@0", "search_inf@0", "search_inf@40")
REG10 = BLIND_METHODS[:10]


def _kendall(a: dict, b: dict, ms) -> float:
    c = dsc = 0
    for x, y in itertools.combinations(ms, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        c += s > 0
        dsc += s < 0
    return round((c - dsc) / (c + dsc), 3) if c + dsc else 1.0


def blind(main: Path, data: dict, rng) -> dict:
    """The grading-blind oracle (attrib.review4_runs blind) at the named steps of
    the 157 rescuable failures, against the registered oracle at the same steps."""
    from attrib.factorial import informed
    rows, steps = [], []
    for d, v in data.items():
        v["picks"]["search_inf@0"] = informed(main, d, v)[1]
        sr = json.loads((main / d / "search" / "result.json").read_text())["failures"]
        v["picks"]["search@0"] = {x["fid"]: (x["suspects"][0] if x.get("suspects") else None) for x in sr}
        res = json.loads((main / d / "gt" / "blind" / "result.json").read_text())
        kind = _kind_of(d)
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            bl = {s["k"]: s for s in res["failures"][f]}
            trace = {s["index"]: s for s in json.loads(Path(v["fails"][f]["trace"]).read_text())["steps"]}
            r = {"domain": d, "fid": f, "cluster": f"{d}/{v['fails'][f]['task_id']}",
                 "kind": kind(trace[g["decisive"]])}
            for m in BLIND_METHODS:
                k = v["picks"][m].get(f)
                ok = isinstance(k, int) and 0 <= k < len(g["R"])
                r[m] = bl[k]["R"] if ok else 0.0
                r[m + "|reg"] = g["R"][k] if ok else 0.0
            kd = g["decisive"]
            r["best"], r["best|reg"] = bl[kd]["R"], g["R"][kd]
            kb = max(bl, key=lambda k: (bl[k]["R"], -k))
            r["blind_best_step"], r["blind_max"] = kb, bl[kb]["R"]
            r["blind_best_is_reg_best"] = kb == kd
            rows.append(r)
            for k, s in bl.items():
                st = g["steps"][k]
                prop_reg = sum(x.get("verdict") == "mistake" and x.get("corrected") is not None and x.get("n") == 2
                               for x in st["samples"]) / K
                prop_bl = sum(x.get("verdict") == "mistake" and x.get("corrected") is not None and x.get("n") == 2
                              for x in s["samples"]) / K
                steps.append({"domain": d, "kind": kind(trace[k]), "best": k == kd, "R_reg": g["R"][k],
                              "R_blind": s["R"], "prop_reg": prop_reg, "prop_blind": prop_bl})
    mean = {m: round(_mean([r[m] for r in rows]), 4) for m in BLIND_METHODS}
    reg = {m: round(_mean([r[m + "|reg"] for r in rows]), 4) for m in BLIND_METHODS}

    def diff(a, b, key=""):
        return _boot([{"d": r[a + key] - r[b + key], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    llm10 = [m for m in REG10 if m not in ("first_write", "last_step")]
    best10 = max(llm10, key=lambda m: mean[m])
    bestall = max([m for m in BLIND_METHODS if m not in ("first_write", "last_step")], key=lambda m: mean[m])

    def agg(ss):
        return {"steps": len(ss), "R_reg": round(_mean([s["R_reg"] for s in ss]), 3),
                "R_blind": round(_mean([s["R_blind"] for s in ss]), 3),
                "prop_reg": round(_mean([s["prop_reg"] for s in ss]), 3),
                "prop_blind": round(_mean([s["prop_blind"] for s in ss]), 3)} if ss else None
    by_dom = {d: {"n": sum(r["domain"] == d for r in rows),
                  "mean": {m: round(_mean([r[m] for r in rows if r["domain"] == d]), 3) for m in BLIND_METHODS}}
              for d in DOMAINS}
    return {"n": len(rows), "mean": mean, "registered_same_failures": reg, "by_domain": by_dom,
            "rank_first_write_reg10": 1 + sum(mean[x] > mean["first_write"] for x in REG10 if x != "first_write"),
            "rank_first_write_all": 1 + sum(mean[x] > mean["first_write"] for x in BLIND_METHODS
                                            if x != "first_write"),
            "kendall_reg10_vs_registered": _kendall(mean, reg, REG10),
            "kendall_all_vs_registered": _kendall(mean, reg, BLIND_METHODS),
            "best_llm_reg10": best10, "best_llm_all": bestall,
            "first_write_minus": {m: diff("first_write", m) for m in
                                  ("binary_search_pro", best10, "binary_search_gain_pro", "all_at_once_gain_pro",
                                   "search_inf@40", bestall)},
            "best_step": {"R_reg": round(_mean([r["best|reg"] for r in rows]), 3),
                          "R_blind": round(_mean([r["best"] for r in rows]), 3),
                          "still_rescuable_at_reg_best": sum(r["best"] >= 0.5 for r in rows),
                          "rescuable_at_any_candidate": sum(r["blind_max"] >= 0.5 for r in rows),
                          "blind_best_is_reg_best": round(_mean([r["blind_best_is_reg_best"] for r in rows]), 3)},
            "steps_all": agg(steps), "steps_best": agg([s for s in steps if s["best"]]),
            "steps_by_domain_kind": {f"{d}/{k}": agg([s for s in steps if s["domain"] == d and s["kind"] == k])
                                     for d, k in sorted({(s["domain"], s["kind"]) for s in steps})},
            "best_by_domain_kind": {f"{d}/{k}": agg([s for s in steps if s["best"] and s["domain"] == d
                                                     and s["kind"] == k])
                                    for d, k in sorted({(s["domain"], s["kind"]) for s in steps if s["best"]})}}


# ------------------------------------------------------------------- fresh --
def fresh(rng) -> dict:
    """Fresh, independent replays at the repair experiment's pointers
    (attrib.review4_runs fresh) against the registered profile."""
    from attrib.phaseb import PB, STEP_GROUPS, picks
    from attrib.phaseb_accuracy import rrsi_steps
    groups = ("oracle",) + STEP_GROUPS + ("rrsi",)
    rows, sup = [], []
    for d in ("tau2_retail", "tau2_airline"):
        pk = picks(d)
        pk["rrsi"] = rrsi_steps(d)
        reg = {g["fid"]: g for g in json.loads((PB / "gt" / d / "a" / "result.json").read_text())["failures"]}
        rp = PB / "gt" / d / "fresh" / "result.json"
        if not rp.exists():
            continue
        fr = json.loads(rp.read_text())["failures"]
        for f, row in fr.items():
            prof = {s["k"]: s for s in row["steps"]}
            g = reg[f]
            r = {"domain": d, "fid": f, "cluster": f"{d}/{f.rsplit('_', 1)[1]}",
                 "rescuable": g["decisive"] is not None}
            for grp in groups:
                k = pk[grp].get(f)
                ok = isinstance(k, int) and 0 <= k < len(g["R"])
                r[grp + "|reg"] = g["R"][k] if ok else 0.0
                r[grp + "|fresh"] = prof[k]["R"] if ok and k in prof else 0.0
            rows.append(r)
            if row.get("supplied") and row["supplied"]["fresh_gain"] is not None:
                s = row["supplied"]
                sup.append({"domain": d, "fid": f, "cluster": r["cluster"],
                            "reg_corrected": s["registered_corrected"], "reg_null": s["registered_null"],
                            "reg_gain": s["registered_corrected"] - s["registered_null"],
                            "fresh_corrected": s["fresh_corrected"], "fresh_null": s["fresh_null"],
                            "fresh_gain": s["fresh_gain"], "reg_R": s["registered_R"],
                            "fresh_R": prof[s["k"]]["R"]})

    def summ(rs):
        out = {"traces": len(rs)}
        if not rs:
            return out
        for grp in groups:
            out[grp] = {"reg": round(_mean([r[grp + "|reg"] for r in rs]), 3),
                        "fresh": round(_mean([r[grp + "|fresh"] for r in rs]), 3)}
        for grp in STEP_GROUPS + ("rrsi",):
            out[f"oracle_minus_{grp}_fresh"] = _boot(
                [{"d": r["oracle|fresh"] - r[grp + "|fresh"], "cluster": r["cluster"]} for r in rs],
                lambda xs: _mean([x["d"] for x in xs]), rng)
        out["oracle_reg_minus_fresh"] = _boot(
            [{"d": r["oracle|reg"] - r["oracle|fresh"], "cluster": r["cluster"]} for r in rs],
            lambda xs: _mean([x["d"] for x in xs]), rng)
        return out
    if not sup:
        return {"all": summ(rows), "rescuable": summ([r for r in rows if r["rescuable"]]), "supplied_correction": None}
    sm = {k: round(_mean([s[k] for s in sup]), 3) for k in
          ("reg_corrected", "reg_null", "reg_gain", "fresh_corrected", "fresh_null", "fresh_gain", "reg_R", "fresh_R")}
    return {"all": summ(rows), "rescuable": summ([r for r in rows if r["rescuable"]]),
            "supplied_correction": {"traces": len(sup), **sm,
                                    "reg_minus_fresh_gain": _boot(
                                        [{"d": s["reg_gain"] - s["fresh_gain"], "cluster": s["cluster"]}
                                         for s in sup], lambda xs: _mean([x["d"] for x in xs]), rng),
                                    "share_fresh_gain_ge_0.5": round(_mean([s["fresh_gain"] >= 0.5 for s in sup]), 3),
                                    "share_fresh_gain_gt_0": round(_mean([s["fresh_gain"] > 0 for s in sup]), 3)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    main_dir = Path(args.main)
    rng = random.Random(0)
    want = set(args.only or ("accepted", "strata", "admiss", "splithalf", "blind", "fresh"))
    data = {d: load(main_dir / d) for d in DOMAINS}
    _picks(main_dir, data)
    res = {}
    if "accepted" in want:
        res["accepted"] = accepted()
    if "strata" in want:
        res["strata"] = strata(main_dir, data, rng)
        res["matched"] = matched(data, rng)
    if "admiss" in want:
        ad = admiss(main_dir, data)
        res["scope"] = scope(main_dir, data, ad["appworld"].pop("scope_rows"))
        res["admiss"] = ad
    if "splithalf" in want:
        res["splithalf"] = splithalf(main_dir, data)
    if "blind" in want:
        res["blind"] = blind(main_dir, data, rng)
    if "fresh" in want:
        res["fresh"] = fresh(rng)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
