"""Post-hoc analyses for the ninth review of paper 1 (not registered).

  python -m verify.posthoc_r9 curves --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r9 calib  --runs DIR/runs/rrsi --out DIR/runs/verify --domains D1,D2
  python -m verify.posthoc_r9 report --traj r1=DIR r2=DIR ... [--live cl1=DIR ...] [--json FILE]
  python -m verify.posthoc_r9 tables --json FILE [--tex paper/tables]

curves: the guarded analysis of verify/posthoc_e1.py restricted to full
  evaluation, sequential full evaluation at a grid of thresholds gamma
  (GAMMAS), fixed sampling at a grid of budgets (SAMPLE_B) and full evaluation
  of one candidate drawn at random (rtop-full), so that the cost, the retention
  of full evaluation's acceptances and the decision value can be traced along
  each rule's budget. Writes <out>/e1_r9_curves.json.

calib: the predictive probabilities of sequential full evaluation (gamma 0.05,
  batches of 10, as registered) against full evaluation's own decision, on
  the recorded trajectories. For every measured candidate, M random orders of
  its full evaluation's episodes; at every interim look the predicted
  probability that the fixed-budget rule admits it, and whether it stops
  there. The outcome is RRSI's decision on the complete evaluation, the
  branch of its rule that decided it (above the noise band, inside the band,
  below the floor) and whether full evaluation selected it. Writes
  <out>/e1_r9_calib.json.

report: (a) reliability of the predicted probabilities (bins of p at interim
  looks against the share admitted by full evaluation; dropped candidates:
  share admitted or selected against the mean p at the drop), from the
  recorded trajectories and, with --live, from the live sequential loops of
  experiments CL and IL with their shadow evaluations; (b) the gamma curve and
  (c) the matched-cost comparison (block bootstrap as in posthoc_r7); (d)
  leave-one-trajectory-out on the primary and pooled sets; (e) the size of the
  independent evidence per trajectory.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

from . import analyze as A
from . import posthoc_e1 as P
from .posthoc_r7 import SETS, _load_tables, summarise
from .posthoc_r8 import _stats
from .state import rounds

GAMMAS = {"g005": 0.005, "g01": 0.01, "g02": 0.02, "": 0.05, "g10": 0.10, "g20": 0.20,
          "g30": 0.30, "g50": 0.50}
SAMPLE_B = (10, 20, 30, 40, 50, 60, 80, 100, 120, 150)
BINS = (0.0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0 + 1e-9)
GAMMA = 0.05


def _gl(g):
    return "seqfull" + (f"[{g}]" if g else "")


# ----------------------------------------------------------------- curves --
def _curves_hook(mod):
    rules = mod.rules
    label = mod.label

    def rules_():
        out = [r for r in rules() if label(*r) in ("none", "full", "net@40", "rtop-full")]
        out += [("seqfull" + (f"_{g}" if g else ""), 0, None) for g in GAMMAS]
        out += [("sample", b, None) for b in SAMPLE_B]
        return out
    mod.rules = rules_


def curves(runs: Path, out: Path, domains: str):
    saved = dict(P.SEQ_GAMMA)
    P.SEQ_GAMMA.update(GAMMAS)
    try:
        P._guard(runs, out, domains=domains, dest="e1_r9_curves.json", hook=_curves_hook)
    finally:
        P.SEQ_GAMMA.clear()
        P.SEQ_GAMMA.update(saved)


# ------------------------------------------------------------------ calib --
def _branch(text: str, admissible: bool, nu: int) -> str:
    """The branch of RRSI's rule that decided a complete evaluation."""
    t = text or ""
    if "guard" in t or "harness" in t:
        return "guard"
    if "below" in t and "floor" in t:
        return "below_floor"
    if re.search(r"gain [-+]?\d*\.\d+ > delta", t):
        return "above_band"
    if "within delta" in t:
        if not admissible:
            return "in_band_rejected"
        return "in_band_novelty" if nu > 0 else "in_band_cost"
    return "other"


def _calib_main(M: int = 400):
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    a = ap.parse_args(sys.argv[1:])
    rows = []
    for name in a.domains.split(","):
        D = A.load(name, Path(a.runs), Path(a.out))
        cfg = D["cfg"]
        cap = cfg.get("max_harness_error_rate", 0.02)
        for R in D["rounds"]:
            s_inc = R["S_inc"]
            floor_var = s_inc * (1 - s_inc)
            delta, L = R["delta"], R["S_star"] - R["delta"] - s_inc
            for v, C in R["cands"].items():
                pairs = A._pairs(R["inc"], C["full"])
                N = len(pairs)
                if N == 0:
                    continue
                dec = C["decision"]
                nu = dec.get("novelty") or 0
                adm = bool(dec.get("admissible"))
                rng = random.Random(1000 * R["t"] + sum(map(ord, v)))
                looks = [[0, 0.0, 0] for _ in range(len(BINS) - 1)]    # count, sum p, dropped here
                # diagnostic: the same looks with the cost change of the complete evaluation
                looks_dc = [[0, 0.0, 0] for _ in range(len(BINS) - 1)]
                dC_full = A._rel_cost([p[5] for p in pairs], [p[6] for p in pairs])
                by_n = {}
                drop_p, guard_drops, eps = [], 0, 0
                for _ in range(M):
                    order = list(pairs)
                    rng.shuffle(order)
                    n = 0
                    while n < N:
                        n = min(N, n + P.SEQ_BATCH)
                        pk = order[:n]
                        if sum(isinstance(p[2], P._Err) for p in pk) > cap * N:
                            guard_drops += 1
                            break
                        if n == N:
                            break
                        diff = [p[2] - p[3] for p in pk]
                        m = float(np.mean(diff))
                        sv = float(np.var(diff, ddof=1)) if n > 1 else 0.0
                        dC = A._rel_cost([p[5] for p in pk], [p[6] for p in pk])
                        p = P.p_admissible(m, max(sv, floor_var), n, N, delta, L, dC, nu, cfg)
                        b = int(np.searchsorted(BINS, p, side="right")) - 1
                        stop = p < GAMMA
                        looks[b][0] += 1
                        looks[b][1] += p
                        looks[b][2] += stop
                        q = P.p_admissible(m, max(sv, floor_var), n, N, delta, L, dC_full, nu, cfg)
                        bq = int(np.searchsorted(BINS, q, side="right")) - 1
                        looks_dc[bq][0] += 1
                        looks_dc[bq][1] += q
                        x = by_n.setdefault(n, [0, 0.0])
                        x[0] += 1
                        x[1] += p
                        if stop:
                            drop_p.append(p)
                            break
                    eps += n
                rows.append({"domain": name, "t": R["t"], "cand": v, "N": N, "nu": nu,
                             "admissible": adm, "selected": R["accepted"] == v,
                             "branch": _branch(dec.get("reason", ""), adm, nu),
                             "dS": dec.get("delta_S"), "dC": dec.get("delta_C"),
                             "M": M, "looks": looks, "looks_dc": looks_dc, "dC_full": dC_full,
                             "by_n": {str(k): x for k, x in by_n.items()},
                             "p_drop": len(drop_p) / M, "p_guard_drop": guard_drops / M,
                             "mean_p_at_drop": float(np.mean(drop_p)) if drop_p else None,
                             "episodes": eps / M})
    (Path(a.out) / "e1_analysis.json").write_text(json.dumps({"rows": rows, "M": M, "gamma": GAMMA},
                                                             indent=1))
    print(f"{len(rows)} candidates")


def calib(runs: Path, out: Path, domains: str):
    orig = A.main

    def hook(mod):
        mod.main = _calib_main
    try:
        P._guard(runs, out, domains=domains, dest="e1_r9_calib.json", hook=hook)
    finally:
        A.main = orig


def _live_rows(run: Path, domain: str = "tau2_airline") -> list[dict]:
    """Calibration rows of a live sequential loop: every interim look with its
    predicted probability, and the decision of the complete evaluation (the
    loop's own for a candidate that survived, the shadow evaluation's for one
    that was dropped)."""
    rr = Path(run) / "runs" / "rrsi" / domain
    sh = Path(run) / "runs" / "verify" / domain / "shadow.json"
    if not sh.exists():
        return []
    shadow = {r["t"]: r for r in json.loads(sh.read_text())}
    rows = []
    for t in sorted(shadow):
        sel = json.loads((rr / f"r{t}" / "selection.json").read_text())
        decs = {x["variant"]: x for x in json.loads((rr / f"r{t}" / "decisions.json").read_text())}
        for v, rec in sel["candidates"].items():
            s = shadow[t]["candidates"][v]
            nu = rec.get("nu") or 0
            if rec["dropped"]:
                adm, why = bool(s["admissible"]), s.get("why", "")
            else:
                adm, why = bool(decs[v].get("admissible")), decs[v].get("reason", "")
            looks = [[0, 0.0, 0] for _ in range(len(BINS) - 1)]
            by_n, drop_p = {}, None
            for st in rec["steps"]:
                p = st.get("p_admit")
                if p is None or st["n"] >= rec["N"]:
                    continue
                b = int(np.searchsorted(BINS, p, side="right")) - 1
                stop = rec["dropped"] and st["n"] == rec["n"] and p < sel.get("gamma", GAMMA)
                looks[b][0] += 1
                looks[b][1] += p
                looks[b][2] += stop
                by_n[str(st["n"])] = [1, p]
                if stop:
                    drop_p = p
            rows.append({"domain": domain, "t": t, "cand": v, "N": rec["N"], "nu": nu,
                         "admissible": adm, "selected": shadow[t]["full_choice"] == v,
                         "branch": _branch(why, adm, nu), "M": 1, "looks": looks, "by_n": by_n,
                         "p_drop": float(drop_p is not None),
                         "p_guard_drop": float(rec["dropped"] and drop_p is None),
                         "mean_p_at_drop": drop_p, "episodes": rec["n"]})
    return rows


def _reliability(rows: list[dict], B: int = 2000, seed: int = 4) -> dict:
    """Bins of interim p against the share admitted by the complete evaluation
    (weighted by looks), and the stopped candidates: share admitted or
    selected by full evaluation against the mean p at the stop. Intervals:
    bootstrap over candidates."""
    def stat(rs):
        out = {"bins": []}
        for b in range(len(BINS) - 1):
            w = np.array([r["looks"][b][0] / r["M"] for r in rs])
            sp = sum(r["looks"][b][1] / r["M"] for r in rs)
            n = w.sum()
            out["bins"].append({"lo": BINS[b], "hi": min(1.0, BINS[b + 1]), "looks": float(n),
                                "mean_p": sp / n if n else None,
                                "admitted": float(np.dot(w, [r["admissible"] for r in rs]) / n) if n else None})
        out["bins_dc"] = []
        for b in range(len(BINS) - 1):
            ls = [r for r in rs if "looks_dc" in r]
            w = np.array([r["looks_dc"][b][0] / r["M"] for r in ls])
            n = w.sum()
            out["bins_dc"].append({"lo": BINS[b], "looks": float(n),
                                   "mean_p": sum(r["looks_dc"][b][1] / r["M"] for r in ls) / n if n else None,
                                   "admitted": float(np.dot(w, [r["admissible"] for r in ls]) / n) if n else None})
        w = np.array([r["p_drop"] for r in rs])
        nd = w.sum()
        out["stopped"] = float(nd)
        out["stopped_admitted"] = float(np.dot(w, [r["admissible"] for r in rs]) / nd) if nd else None
        out["stopped_selected"] = float(np.dot(w, [r["selected"] for r in rs]) / nd) if nd else None
        mp = [(r["p_drop"], r["mean_p_at_drop"]) for r in rs if r["mean_p_at_drop"] is not None]
        out["stopped_mean_p"] = (sum(a * b for a, b in mp) / sum(a for a, _ in mp)) if mp else None
        adm = [r for r in rs if r["admissible"]]
        out["admissible"] = len(adm)
        out["admissible_stopped"] = float(np.mean([r["p_drop"] for r in adm])) if adm else None
        sel = [r for r in rs if r["selected"]]
        out["selected"] = len(sel)
        out["selected_stopped"] = float(np.mean([r["p_drop"] for r in sel])) if sel else None
        return out
    res = stat(rows)
    res["candidates"] = len(rows)
    rng = np.random.default_rng(seed)
    bs = [stat([rows[i] for i in rng.integers(0, len(rows), len(rows))]) for _ in range(B)] if rows else []
    for k in ("stopped_admitted", "stopped_selected", "admissible_stopped", "selected_stopped"):
        xs = [b[k] for b in bs if b[k] is not None]
        if xs:
            res[k + "_ci90"] = [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    for i, b in enumerate(res["bins"]):
        xs = [x["bins"][i]["admitted"] for x in bs if x["bins"][i]["admitted"] is not None]
        if xs and b["looks"]:
            b["admitted_ci90"] = [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    # by branch of the complete evaluation and by novelty (known before evaluation)
    res["by_branch"] = {}
    for br in sorted({r["branch"] for r in rows}):
        rs = [r for r in rows if r["branch"] == br]
        res["by_branch"][br] = {"candidates": len(rs),
                                "stopped": float(np.mean([r["p_drop"] for r in rs])),
                                "guard_stopped": float(np.mean([r["p_guard_drop"] for r in rs])),
                                "mean_p_first_look": float(np.mean([r["by_n"][min(r["by_n"], key=int)][1]
                                                                    / r["by_n"][min(r["by_n"], key=int)][0]
                                                                    for r in rs if r["by_n"]])) if any(r["by_n"] for r in rs) else None,
                                "episode_share": float(np.mean([r["episodes"] / r["N"] for r in rs]))}
    res["by_novelty"] = {}
    for k, f in (("nu0", lambda r: r["nu"] == 0), ("nu_pos", lambda r: r["nu"] > 0)):
        rs = [r for r in rows if f(r)]
        if rs:
            x = stat(rs)
            res["by_novelty"][k] = {"candidates": len(rs), "admitted": float(np.mean([r["admissible"] for r in rs])),
                                    "stopped": float(np.mean([r["p_drop"] for r in rs])),
                                    "stopped_admitted": x["stopped_admitted"],
                                    "stopped_mean_p": x["stopped_mean_p"],
                                    "bins": [(b["looks"], b["mean_p"], b["admitted"]) for b in x["bins"]]}
    return res


# ----------------------------------------------------------------- report --
CURVE_RULES = ["full", "rtop-full"] + [_gl(g) for g in GAMMAS] + [f"sample@{b}" for b in SAMPLE_B]
LOO_RULES = ("full", "seqfull", "seqsample@80", "sample@80", "sample@40", "net@40", "replaynull@40",
             "rtop-full")


def _prune(tables, keep):
    return [{**T, "rows": [{**r, "probs": {k: v for k, v in r["probs"].items() if k in keep},
                                 "cost": {k: v for k, v in r["cost"].items() if k in keep}}
                           for r in T["rows"]]} for T in tables]


def _curve(tables, block_of):
    s = summarise(tables, block_of, "full")
    out = {}
    for r in CURVE_RULES:
        if r not in s:
            continue
        st = _stats(tables, r)
        x = s[r]
        out[r] = {"episodes": x.get("episodes"), "dv": x["dv"], "diff_vs_full": x.get("diff_vs_full"),
                  "p_ni": x.get("p_ni"), "accept_rate": x.get("accept_rate"),
                  "agree_with_full": x.get("agree_with_full"), "recall": st["recall"],
                  "accept_when_full_keeps": st["accept_when_full_keeps"],
                  "forgone_gain": st["forgone_gain"], "avoided_harm": st["avoided_harm"]}
    full_ep = out["full"]["episodes"]
    for r in out:
        out[r]["episode_share"] = out[r]["episodes"] / full_ep if full_ep else None
    # fixed sampling at the episodes of sequential full evaluation (gamma 0.05) and of rtop-full,
    # interpolated linearly along the sample@b curve
    samp = sorted(((out[f"sample@{b}"]["episodes"], f"sample@{b}") for b in SAMPLE_B
                   if f"sample@{b}" in out))
    match = {}
    for ref in ("seqfull", "rtop-full"):
        e = out[ref]["episodes"]
        lo = max([x for x in samp if x[0] <= e], default=None)
        hi = min([x for x in samp if x[0] >= e], default=None)
        if lo is None or hi is None:
            continue
        w = 0.0 if hi[0] == lo[0] else (e - lo[0]) / (hi[0] - lo[0])
        it = lambda k: (1 - w) * (out[lo[1]][k] if not isinstance(out[lo[1]][k], list) else out[lo[1]][k][0]) + \
            w * (out[hi[1]][k] if not isinstance(out[hi[1]][k], list) else out[hi[1]][k][0])
        match[ref] = {"episodes": e, "between": [lo[1], hi[1]], "w": w,
                      "sample_dv": it("dv"), "sample_recall": it("recall"),
                      "sample_accept_when_full_keeps": it("accept_when_full_keeps"),
                      "ref_dv": out[ref]["dv"][0], "ref_recall": out[ref]["recall"],
                      "ref_accept_when_full_keeps": out[ref]["accept_when_full_keeps"]}
    return out, match


def _effective(trajs: dict) -> list[dict]:
    from rrsi.domain import load_domain
    rows = []
    for n, root in trajs.items():
        f = Path(root) / "verify" / "e1_analysis_guard_posthoc.json"
        if not f.exists():
            continue
        for T in json.loads(f.read_text())["tables"]:
            rr = list(rounds(Path(root) / "rrsi" / T["name"]))
            ts = {r.t for r in rr}
            rs = [r for r in T["rows"] if r["t"] in ts]
            dom = load_domain(T["name"])
            ho = sorted({t for r in rs for d in r["dep"].values() for t in d})
            fa = sum(1 for r in rs if max(r["probs"]["full"], key=r["probs"]["full"].get) not in ("null", None))
            rows.append({"traj": n, "domain": T["name"], "rounds": len(rs),
                         "candidates": sum(len(r["dep"]) for r in rs),
                         "full_accepts": fa,
                         "blocks": len({r.inc_commit for r in rr if r.t in {x["t"] for x in rs}}),
                         "evolve_tasks": len(dom.evolve_ids()), "evolve_trials": int(dom.cfg.get("k", 2)),
                         "heldout_tasks": len(ho), "heldout_trials": int(dom.cfg.get("heldout_k", 4)),
                         "cand_eval_episodes": int(round(sum(r["cost"]["full"]["episodes"] for r in rs)))})
    return rows


def report(trajs: dict, live: dict, js: Path | None):
    res = {"calibration": {}, "curves": {}, "matched": {}, "loo": {}, "effective": _effective(trajs)}
    # (a) calibration
    calrows = {}
    for n, root in trajs.items():
        f = Path(root) / "verify" / "e1_r9_calib.json"
        if f.exists():
            calrows[n] = [dict(r, traj=n) for r in json.loads(f.read_text())["rows"]]
    groups = {"primary": ["r2", "r3"], "tau2": ["r1", "r2", "r3", "r4"], "e3": ["e3"]}
    for g, names in groups.items():
        rs = [r for n in names for r in calrows.get(n, [])]
        if g == "tau2":
            rs_air = [r for r in rs if r["domain"] == "tau2_airline"]
            if rs_air:
                res["calibration"]["tau2_airline"] = _reliability(rs_air)
        if rs:
            res["calibration"][g] = _reliability(rs)
    lv = [r for n, root in live.items() for r in _live_rows(Path(root))]
    if lv:
        res["calibration"]["live"] = _reliability(lv)
        res["calibration"]["live"]["loops"] = sorted(n for n, root in live.items() if _live_rows(Path(root)))
    # (b, c) gamma curve and matched cost
    for set_name, names in SETS:
        if not all(n in trajs for n in names):
            continue
        tables, block_of = _load_tables(trajs, names, "e1_r9_curves.json")
        if tables is None:
            continue
        res["curves"][set_name], res["matched"][set_name] = _curve(tables, block_of)
    # (d) leave one trajectory (and domain) out
    for set_name, names in (("primary", ["r2", "r3"]), ("pooled", ["r1", "r2", "r3"])):
        if not all(n in trajs for n in names):
            continue
        tables, block_of = _load_tables(trajs, names, "e1_analysis_guard_posthoc.json")
        tables = _prune(tables, LOO_RULES)
        res["loo"][set_name] = {}
        for drop in [T["name"] for T in tables]:
            s = summarise([T for T in tables if T["name"] != drop], block_of, "full")
            res["loo"][set_name][drop] = {r: {"diff_vs_full": s[r]["diff_vs_full"], "p_ni": s[r]["p_ni"],
                                              "agree_with_full": s[r]["agree_with_full"]}
                                          for r in LOO_RULES if r != "full" and r in s}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1))
    _print(res)
    return res


def _print(res):
    pp = lambda x: "  -  " if x is None else f"{100 * x:+.2f}"
    for g, c in res["calibration"].items():
        print(f"== calibration {g}: {c['candidates']} candidates; stopped {c['stopped']:.1f} "
              f"admitted {c['stopped_admitted']} selected {c['stopped_selected']} mean p {c['stopped_mean_p']}")
        print(f"   admissible {c['admissible']} stopped {c['admissible_stopped']}; selected {c['selected']} "
              f"stopped {c['selected_stopped']}")
        for b in c["bins"]:
            print(f"   p in [{b['lo']:.2f},{b['hi']:.2f}): looks {b['looks']:.1f} mean p "
                  f"{b['mean_p'] if b['mean_p'] is None else round(b['mean_p'], 3)} admitted "
                  f"{b['admitted'] if b['admitted'] is None else round(b['admitted'], 3)} {b.get('admitted_ci90')}")
        for br, x in c["by_branch"].items():
            print(f"   {br}: {x}")
        for k, x in c["by_novelty"].items():
            print(f"   {k}: " + json.dumps({a: b for a, b in x.items() if a != "bins"}))
    for s, rows in res["curves"].items():
        print(f"== curve {s}")
        for r, x in rows.items():
            d = x["diff_vs_full"]
            print(f"{r:16s} ep {x['episodes']:6.1f} ({x['episode_share']:.2f}) D {pp(x['dv'][0])} "
                  f"recall {x['recall'] if x['recall'] is None else round(x['recall'], 2)} "
                  f"acc|keep {x['accept_when_full_keeps']:.2f}"
                  + (f" diff {pp(d[0])} [{pp(d[1])},{pp(d[2])}] pNI {x['p_ni']}" if d else ""))
        print("matched:", json.dumps(res["matched"][s]))
    for s, v in res["loo"].items():
        print(f"== leave one out {s}")
        for drop, rows in v.items():
            print(f"  without {drop}: " + "; ".join(
                f"{r} {pp(x['diff_vs_full'][0])} p25 {x['p_ni']['0.25']:.3f} p75 {x['p_ni']['0.75']:.3f}"
                for r, x in rows.items()))
    for r in res["effective"]:
        print(r)


# ----------------------------------------------------------------- tables --
def _f(x, d=2):
    return "--" if x is None else f"{x:.{d}f}"


def _pp(x):
    return "--" if x is None else f"{100 * x:+.2f}".replace("-", "$-$")


def tables(js: Path, out: Path):
    """paper/tables/r9_calib.tex, r9_branch.tex, r9_gamma.tex, r9_matched.tex, r9_loo.tex,
    r9_effective.tex"""
    R = json.loads(Path(js).read_text())
    out.mkdir(parents=True, exist_ok=True)
    C = R["calibration"]
    cols = [(c, k) for c, k in (("primary", "bins"), ("primary", "bins_dc"), ("tau2_airline", "bins"),
                                 ("e3", "bins"), ("live", "bins")) if c in C]
    lines = []
    for i, b in enumerate(C["primary"]["bins"]):
        cells = [f"$[{b['lo']:.2f}, {b['hi']:.2f})$" if b["hi"] < 1 else f"$[{b['lo']:.2f}, 1]$"]
        for c, k in cols:
            x = C[c][k][i]
            mp = x["mean_p"]
            cells.append(f"{_f(mp, 3 if mp is not None and mp < 0.1 else 2)} & {_f(x['admitted'])}"
                         if x["looks"] else "-- & --")
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    row = lambda name, f: lines.append(" & ".join([name] + [("\\multicolumn{2}{c}{--}" if k == "bins_dc" else f(C[c]))
                                                            for c, k in cols]) + " \\\\")
    row("Stopped: mean $p$ / admitted", lambda x: f"{_f(x['stopped_mean_p'], 3)} & {_f(x['stopped_admitted'])}")
    row("Stopped: selected by full", lambda x: f"\\multicolumn{{2}}{{c}}{{{_f(x['stopped_selected'])}}}")
    row("Admitted by full: stopped", lambda x: f"\\multicolumn{{2}}{{c}}{{{100 * x['admissible_stopped']:.0f}\\% of {x['admissible']}}}")
    (out / "r9_calib.tex").write_text("\n".join(lines) + "\n")
    names = {"above_band": "Above the band", "in_band_cost": "In band, cheaper",
             "in_band_novelty": "In band, new component", "in_band_rejected": "In band, rejected",
             "below_floor": "Below the floor"}
    lines = []
    for br, name in names.items():
        cells = [name]
        for c in ("tau2", "e3", "live"):
            x = C.get(c, {}).get("by_branch", {}).get(br)
            cells.append("-- & --" if x is None else f"{x['candidates']} & {_f(x['stopped'])}")
        lines.append(" & ".join(cells) + " \\\\")
    (out / "r9_branch.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for g in GAMMAS:
        r = _gl(g)
        cells = [f"{GAMMAS[g]:g}"]
        for s in ("primary", "r1", "e3"):
            x = R["curves"].get(s, {}).get(r)
            if x is None:
                cells.append("-- & -- & --")
                continue
            cells.append(f"{_f(x['episode_share'])} & {_f(x['recall'])} & {_pp(x['diff_vs_full'][0])}")
        x = R["curves"]["primary"][r]
        cells.append(_p(x["p_ni"]["0.25"]))
        lines.append(" & ".join(cells) + " \\\\")
    (out / "r9_gamma.tex").write_text("\n".join(lines) + "\n")
    lines = []
    rows_ = ["full", "seqfull", "rtop-full"] + [f"sample@{b}" for b in SAMPLE_B]
    lab = {"full": "Full evaluation", "seqfull": "Sequential full ($\\gamma=0.05$)",
           "rtop-full": "Full, one random candidate"}
    for r in rows_:
        x = R["curves"]["primary"].get(r)
        if x is None:
            continue
        cells = [lab.get(r, r.replace("sample@", "Sample@")), f"{x['episodes']:.0f}", _f(x["recall"]),
                 _f(x["accept_when_full_keeps"]), _pp(x["dv"][0])]
        cells.append("--" if r == "full" else f"{_pp(x['diff_vs_full'][1])}, {_pp(x['diff_vs_full'][2])}")
        cells.append("--" if r == "full" else _p(x["p_ni"]["0.25"]))
        lines.append(" & ".join(cells) + " \\\\")
    (out / "r9_matched.tex").write_text("\n".join(lines) + "\n")
    lines = []
    short = {"tau2_retail": "retail", "tau2_airline": "airline"}
    for s in ("primary", "pooled"):
        for drop, rows in R["loo"].get(s, {}).items():
            t, d = drop.split("/")
            cells = [("r2+r3" if s == "primary" else "r1--r3") + f" without {t} {short.get(d, d)}"]
            for r in ("seqfull", "sample@40", "net@40", "replaynull@40"):
                x = rows[r]
                cells.append(f"{_pp(x['diff_vs_full'][0])} & {_p(x['p_ni']['0.25'] if r == 'seqfull' else x['p_ni']['0.75'])}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    (out / "r9_loo.tex").write_text("\n".join(lines[:-1]) + "\n")
    lines = []
    dn = {"tau2_retail": "retail", "tau2_airline": "airline", "appworld": "AppWorld"}
    for x in R["effective"]:
        lines.append(" & ".join([f"{x['traj'].upper() if x['traj'] == 'e3' else x['traj']} {dn[x['domain']]}",
                                 str(x["rounds"]), str(x["candidates"]), str(x["full_accepts"]), str(x["blocks"]),
                                 f"{x['evolve_tasks']} $\\times$ {x['evolve_trials']}",
                                 f"{x['heldout_tasks']} $\\times$ {x['heldout_trials']}",
                                 f"{x['cand_eval_episodes']:,}".replace(",", "{,}")]) + " \\\\")
    (out / "r9_effective.tex").write_text("\n".join(lines) + "\n")
    il = Path(__file__).resolve().parents[1] / "results" / "il" / "analysis.json"
    if il.exists():
        # transfer at round 10 per loop for the main text's dot plot (experiment IL)
        from .il import FULL, NEW, SEQ
        res = json.loads(il.read_text())
        tr = res["transfer"]
        rows, seen = {True: ["x y"], False: ["x y"]}, {}
        for arm, names in ((1, FULL), (0, SEQ)):
            for n in names:
                x = round(100 * tr[n], 2)
                k = seen[(arm, x)] = seen.get((arm, x), -1) + 1
                rows[n in NEW].append(f"{x} {arm + 0.16 * k * (-1) ** k:.2f}")
        (out / "il_points_new.dat").write_text("\n".join(rows[True]) + "\n")
        (out / "il_points_old.dat").write_text("\n".join(rows[False]) + "\n")
        p = res["primary"]["transfer"]
        (out / "il_means.dat").write_text(f"x y\n{100 * p['seq_mean']:.2f} 0\n{100 * p['full_mean']:.2f} 1\n")
    for f in ("r9_calib", "r9_branch", "r9_gamma", "r9_matched", "r9_loo", "r9_effective"):
        print(f"== {f}\n" + (out / f"{f}.tex").read_text())


def _p(x):
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("curves", "calib", "report", "tables"))
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--domains", default="tau2_retail,tau2_airline")
    ap.add_argument("--traj", nargs="*", default=[])
    ap.add_argument("--live", nargs="*", default=[])
    ap.add_argument("--json")
    ap.add_argument("--tex", default=str(Path(__file__).resolve().parents[1] / "paper" / "tables"))
    a = ap.parse_args()
    if a.cmd == "curves":
        curves(Path(a.runs), Path(a.out), a.domains)
    elif a.cmd == "calib":
        calib(Path(a.runs), Path(a.out), a.domains)
    elif a.cmd == "tables":
        tables(Path(a.json), Path(a.tex))
    else:
        kv = lambda xs: {k: v for k, v in (x.split("=", 1) for x in xs)}
        report(kv(a.traj), kv(a.live), Path(a.json) if a.json else None)


if __name__ == "__main__":
    main()
