"""Zero-cost analyses for the fifteenth review of paper 1 (post hoc, not registered).

  python -m verify.posthoc_r15 report --traj r1=DIR/runs r2=... r3=... e3=... [--json results/r15/report.json]
  python -m verify.posthoc_r15 tables [--json results/r15/report.json] [--tex paper/tables/r15_cells.tex]

cells (question 1): decision values and lost acceptances by domain and by
early (rounds 0-9) and late (10-19) rounds, on r2+r3 (the registered primary
set), r1-r3 and AppWorld. Rules full, seqfull (registered sequential rule) and
seqcost (posthoc_r10's cost-aware rule) from posthoc_r10's merged tables; per
cell the rounds, full evaluation's acceptances, D(full), D(seqfull) - D(full),
seqfull's recall and the held-out gain of the acceptances it drops
(posthoc_r8._stats), with 90% block-bootstrap intervals (blocks of rounds
sharing an incumbent, and held-out tasks, within domain and trajectory). The
pooled D(full) is the round-weighted mean of the cells' values.

utility (question 3): for full evaluation's acceptances on each set, weighted
by the probability that seqfull drops (or keeps) them, the per-episode
deployment utility v * dp - dc_run of Eq. (net) at v = 10 (as in the paper's
Net(N) columns) and v = 1, split into its success part v * dp and its
running-cost part -dc_run (dc_run: the candidate's held-out running cost
relative to the incumbent's), overall and by the branch of RRSI's rule that
admitted the candidate (posthoc_r9's calibration rows).

calib (question 2): stopped candidates' admission rate by full evaluation
against their mean predicted probability at the stop, with 90% and 95%
intervals from a bootstrap that resamples whole candidates (all their looks
and all 400 orderings together) and, more conservatively, whole blocks of
rounds sharing an incumbent (live loops: blocks within each loop); and the
bootstrap distribution of the difference (admitted - mean p).

power (question 4): BK's lineage-model standard errors (results/bk/
analysis.json) projected to K blocks: var_K = var_4 * 4 / K and
Satterthwaite df_K = df_4 * (K - 1) / 3 (every variance term of
verify/bk._lineage_se is divided by the number of blocks, and its df are
2(K - 1) or K - 1). Power of the one-sided t test (noncentral t) at
alpha = 0.05, unadjusted and Holm-worst-case over BK's five tested contrasts
(alpha / 5), the minimum detectable effect at 80% power, and the blocks
needed for effects of 1.0, 1.6 and 2.0 points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import nct, t as tdist

from . import il
from .posthoc_r8 import _resample, _stats
from .posthoc_r9 import _live_rows
from .posthoc_r10 import _merged
from .state import rounds

BK_JSON = Path("results/bk/analysis.json")
SETS = (("r2r3", ["r2", "r3"]), ("r1r3", ["r1", "r2", "r3"]), ("e3", ["e3"]))
RULES = ("full", "seqfull", "seqcost")
B = 2000


def _cut(tables, dom=None, t0=0, t1=99):
    out = []
    for T in tables:
        if dom and not T["name"].endswith("/" + dom):
            continue
        out.append({**T, "rows": [r for r in T["rows"] if t0 <= r["t"] <= t1]})
    return [T for T in out if T["rows"]]


def _dvals(sample, rule):
    """Round values of rule over a sample [(T, idx, tasks)]."""
    from . import analyze as A
    return [A.round_value(T["rows"][i], rule, ts) for T, idx, ts in sample for i in idx]


def _cell(tables, block_of, seed: int) -> dict:
    obs = [(T, range(len(T["rows"])), None) for T in tables]
    rng = np.random.default_rng(seed)
    boots = [_resample(tables, block_of, rng) for _ in range(B)]

    def one(sample):
        f = np.mean(_dvals(sample, "full"))
        x = {"D_full": f, "seq_minus_full": np.mean(_dvals(sample, "seqfull")) - f,
             "cost_minus_full": np.mean(_dvals(sample, "seqcost")) - f}
        st = _stats(tables, "seqfull", sample)
        x.update(recall=st["recall"], gain_dropped=st["gain_dropped"], forgone=st["forgone_gain"],
                 avoided=st["avoided_harm"], full_acc_gain=st["full_acc_gain"])
        return x
    pt = one(obs)
    bs = [one(s) for s in boots]
    st = _stats(tables, "seqfull")
    out = {"rounds": sum(len(T["rows"]) for T in tables), "full_accepts": st["full_accepts"],
           "dropped": st["dropped"]}
    for k, v in pt.items():
        xs = [b[k] for b in bs if b[k] is not None]
        out[k] = [None if v is None else float(v)] + ([float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
                                                       if xs else [None, None])
    return out


def cells(trajs: dict) -> dict:
    res = {}
    for name, names in SETS:
        if not all(n in trajs for n in names):
            continue
        tables, block_of = _merged(trajs, names)
        doms = sorted({T["name"].split("/", 1)[1] for T in tables})
        spec = {"all": (None, 0, 99), "early": (None, 0, 9), "late": (None, 10, 99)}
        for d in doms if len(doms) > 1 else []:
            short = d.replace("tau2_", "")
            spec.update({short: (d, 0, 99), f"{short}_early": (d, 0, 9), f"{short}_late": (d, 10, 99)})
        res[name] = {}
        for i, (k, (d, t0, t1)) in enumerate(spec.items()):
            sub = _cut(tables, d, t0, t1)
            if sub:
                res[name][k] = _cell(sub, block_of, 100 + i)
        parts = [k for k in res[name] if "_" in k] or ["early", "late"]
        n = sum(res[name][k]["rounds"] for k in parts)
        res[name]["pooled_check"] = {"weighted_D_full": float(sum(res[name][k]["rounds"] * res[name][k]["D_full"][0]
                                                                  for k in parts) / n), "parts": parts}
    return res


def _branches(trajs: dict, names) -> dict:
    out = {}
    for n in names:
        f = Path(trajs[n]) / "verify" / "e1_r9_calib.json"
        for r in json.loads(f.read_text())["rows"]:
            out[(n, r["domain"], r["t"], r["cand"])] = (r["branch"], r.get("dC"))
    return out


def utility(trajs: dict) -> dict:
    res = {}
    for name, names in SETS:
        if not all(n in trajs for n in names):
            continue
        tables, _ = _merged(trajs, names)
        br = _branches(trajs, names)
        acc = []
        for T in tables:
            n, dom = T["name"].split("/", 1)
            for r in T["rows"]:
                f = r["probs"]["full"]
                c = max(f, key=f.get)
                if c in (None, "null"):
                    continue
                d = r["dep"].get(c) or {}
                dp = float(np.mean(list(d.values()))) if d else 0.0
                keep = r["probs"]["seqfull"].get(c, 0.0)
                b, edc = br.get((n, dom, r["t"], c), ("unknown", None))
                acc.append({"dp": dp, "dc": float(r["dcr"].get(c) or 0.0), "keep": keep, "branch": b,
                            "edc": edc})

        def summ(xs, w):
            w = np.array(w, float)
            if w.sum() <= 0:
                return None
            dp = np.array([x["dp"] for x in xs])
            dc = np.array([x["dc"] for x in xs])
            o = {"n": float(w.sum()), "dp_points": float(100 * np.dot(w, dp) / w.sum()),
                 "dc_run_pct": float(100 * np.dot(w, dc) / w.sum()),
                 "share_cheaper": float(np.dot(w, dc < 0) / w.sum()),
                 "share_dp_pos": float(np.dot(w, dp > 0) / w.sum())}
            ev = [(wi, x["edc"]) for wi, x in zip(w, xs) if x["edc"] is not None]
            if ev:
                o["evolve_dc_pct"] = float(100 * sum(a * b for a, b in ev) / sum(a for a, _ in ev))
            for v in (1, 10):
                o[f"v{v}"] = {"success": float(v * np.dot(w, dp) / w.sum()), "cost": float(-np.dot(w, dc) / w.sum()),
                              "total": float(np.dot(w, v * dp - dc) / w.sum())}
            return o
        x = {"all": summ(acc, [1.0] * len(acc)),
             "dropped": summ(acc, [1 - a["keep"] for a in acc]),
             "kept": summ(acc, [a["keep"] for a in acc]), "by_branch": {}}
        for b in sorted({a["branch"] for a in acc}):
            xs = [a for a in acc if a["branch"] == b]
            x["by_branch"][b] = {"all": summ(xs, [1.0] * len(xs)),
                                 "dropped": summ(xs, [1 - a["keep"] for a in xs])}
        res[name] = x
    return res


def _calib_stat(rows):
    w = np.array([r["p_drop"] for r in rows])
    nd = w.sum()
    if nd <= 0:
        return None, None
    adm = float(np.dot(w, [r["admissible"] for r in rows]) / nd)
    mp = [(r["p_drop"], r["mean_p_at_drop"]) for r in rows if r["mean_p_at_drop"] is not None]
    return adm, sum(a * b for a, b in mp) / sum(a for a, _ in mp)


def _calib_boot(rows, cluster, seed=31):
    groups = {}
    for r in rows:
        groups.setdefault(cluster(r), []).append(r)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    adm, diff = [], []
    for _ in range(B):
        rs = [r for k in rng.integers(0, len(keys), len(keys)) for r in groups[keys[k]]]
        a, m = _calib_stat(rs)
        if a is not None:
            adm.append(a)
            diff.append(a - m)
    q = lambda xs, lo, hi: [float(np.percentile(xs, lo)), float(np.percentile(xs, hi))]
    return {"clusters": len(keys), "admitted_ci90": q(adm, 5, 95), "admitted_ci95": q(adm, 2.5, 97.5),
            "diff_ci90": q(diff, 5, 95), "diff_ci95": q(diff, 2.5, 97.5),
            "p_diff_le_0": float(np.mean(np.array(diff) <= 0))}


def calib(trajs: dict) -> dict:
    groups = {"r2r3": (["r2", "r3"], None), "airline": (["r1", "r2", "r3", "r4"], "tau2_airline"),
              "e3": (["e3"], None)}
    inc = {}

    def inc_of(key, root, dom):
        if key not in inc:
            inc[key] = {r.t: r.inc_commit for r in rounds(Path(root) / dom)}
        return inc[key]
    res = {}
    for g, (names, dom) in groups.items():
        if not all(n in trajs for n in names):
            continue
        rows = [dict(r, traj=n) for n in names
                for r in json.loads((Path(trajs[n]) / "verify" / "e1_r9_calib.json").read_text())["rows"]
                if dom is None or r["domain"] == dom]
        a, m = _calib_stat(rows)
        blk = lambda r: (r["traj"], r["domain"],
                         inc_of((r["traj"], r["domain"]), Path(trajs[r["traj"]]) / "rrsi", r["domain"])[r["t"]])
        res[g] = {"candidates": len(rows), "admitted": a, "mean_p": m,
                  "by_candidate": _calib_boot(rows, lambda r: (r["traj"], r["domain"], r["t"], r["cand"])),
                  "by_incumbent_block": _calib_boot(rows, blk)}
    lv = [dict(r, loop=n) for n, root in il.SEQ.items() for r in _live_rows(Path(root))]
    if lv:
        a, m = _calib_stat(lv)
        blk = lambda r: (r["loop"], inc_of(("live", r["loop"]), Path(il.SEQ[r["loop"]]) / "runs" / "rrsi",
                                           "tau2_airline")[r["t"]])
        res["live"] = {"candidates": len(lv), "admitted": a, "mean_p": m,
                       "by_candidate": _calib_boot(lv, lambda r: (r["loop"], r["t"], r["cand"])),
                       "by_incumbent_block": _calib_boot(lv, blk),
                       "by_loop": _calib_boot(lv, lambda r: r["loop"])}
    return res


def power() -> dict:
    con = json.loads(BK_JSON.read_text())["contrasts"]
    out = {}
    for k in ("admission", "seq_minus_full", "cost_minus_seq", "history"):
        se4, df4 = con[k]["se"], con[k]["df"]

        def at(K):
            return se4 * np.sqrt(4 / K), df4 * (K - 1) / 3

        def pw(theta, K, alpha):
            se, df = at(K)
            return float(nct.sf(tdist.ppf(1 - alpha, df), df, theta / se))

        def mde(K, alpha, target=0.8):
            lo, hi = 0.0, 0.5
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if pw(mid, K, alpha) < target else (lo, mid)
            return hi

        def need(theta, alpha):
            for K in range(3, 200):
                if pw(theta, K, alpha) >= 0.8:
                    return K
            return None
        x = {"est": con[k]["est"], "se4": se4, "df4": df4}
        for lab, alpha in (("one_sided_05", 0.05), ("holm_worst_01", 0.01)):
            x[lab] = {"power_at_4_for_est": pw(abs(con[k]["est"]), 4, alpha),
                      "mde80": {K: mde(K, alpha) for K in (4, 6, 8, 12, 16)},
                      "blocks_for_80": {f"{th:g}": need(th / 100, alpha) for th in (1.0, 1.6, 2.0)}}
        out[k] = x
    return out


def report(trajs: dict, js: Path | None):
    res = {"cells": cells(trajs), "utility": utility(trajs), "calib": calib(trajs), "power": power()}
    if js:
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps(res, indent=1, default=float))
    pp = lambda x: "  -  " if x is None else f"{100 * x:+.2f}"
    ci = lambda v: f"{pp(v[0])} [{pp(v[1])},{pp(v[2])}]"
    for name, cs in res["cells"].items():
        print(f"== cells {name}  (pooled check D(full) {pp(cs['pooled_check']['weighted_D_full'])})")
        for k, c in cs.items():
            if k == "pooled_check":
                continue
            rec = c["recall"][0]
            print(f"  {k:15s} rounds {c['rounds']:3d} acc {c['full_accepts']:2d} D(full) {ci(c['D_full'])}"
                  f" s-f {ci(c['seq_minus_full'])} c-f {ci(c['cost_minus_full'])}"
                  f" recall {'-' if rec is None else round(rec, 2)} dropped {c['dropped']:.1f}"
                  f" gain dropped {pp(c['gain_dropped'][0])} forgone {pp(c['forgone'][0])} avoided {pp(c['avoided'][0])}")
    for name, u in res["utility"].items():
        print(f"== utility {name}")
        for k in ("all", "dropped", "kept"):
            x = u[k]
            if x:
                print(f"  {k:8s} n {x['n']:.1f} dp {x['dp_points']:+.2f} pts dc_run {x['dc_run_pct']:+.2f}% "
                      f"cheaper {x['share_cheaper']:.2f} | v10 success {x['v10']['success']:+.4f} cost "
                      f"{x['v10']['cost']:+.4f} total {x['v10']['total']:+.4f} | v1 success {x['v1']['success']:+.4f}")
        for b, y in u["by_branch"].items():
            for k in ("all", "dropped"):
                x = y[k]
                if x:
                    print(f"  {b:14s} {k:8s} n {x['n']:.1f} dp {x['dp_points']:+.2f} dc_run {x['dc_run_pct']:+.2f}% "
                          f"(evolve {x.get('evolve_dc_pct', float('nan')):+.2f}%) "
                          f"v10 success {x['v10']['success']:+.4f} cost {x['v10']['cost']:+.4f}")
    for g, c in res["calib"].items():
        print(f"== calib {g}: {c['candidates']} cands, stopped admitted {c['admitted']:.3f} vs mean p {c['mean_p']:.3f}")
        for k in ("by_candidate", "by_incumbent_block", "by_loop"):
            if k in c:
                x = c[k]
                print(f"  {k:19s} clusters {x['clusters']:3d} adm 95% [{x['admitted_ci95'][0]:.3f},{x['admitted_ci95'][1]:.3f}]"
                      f" diff 95% [{x['diff_ci95'][0]:+.3f},{x['diff_ci95'][1]:+.3f}] P(diff<=0) {x['p_diff_le_0']:.3f}")
    for k, x in res["power"].items():
        print(f"== power {k}: est {pp(x['est'])} se4 {100 * x['se4']:.2f} df4 {x['df4']:.1f}")
        for lab in ("one_sided_05", "holm_worst_01"):
            y = x[lab]
            print(f"  {lab}: power at 4 for |est| {y['power_at_4_for_est']:.2f}; MDE80 "
                  + ", ".join(f"K{K} {100 * v:.2f}" for K, v in y["mde80"].items())
                  + f"; blocks for 80%: {y['blocks_for_80']}")
    return res


def tables(js: Path, tex: Path):
    res = json.loads(js.read_text())["cells"]
    num = lambda x: f"{100 * x:+.2f}".replace("+0.00", "0.00").replace("-0.00", "0.00")
    pp = lambda x: "--" if x is None else f"${num(x)}$"
    ci = lambda v: "--" if v[0] is None else f"${num(v[0])}$ [${num(v[1])}, {num(v[2])}$]"
    lab = {"r2r3": "r2+r3 (registered set)", "r1r3": "r1--r3", "e3": "AppWorld (E3)"}
    order = {"r2r3": ["retail_early", "retail_late", "airline_early", "airline_late", "all"],
             "r1r3": ["retail_early", "retail_late", "airline_early", "airline_late", "all"],
             "e3": ["early", "late", "all"]}
    name = {"retail_early": "retail, 0--9", "retail_late": "retail, 10--19", "airline_early": "airline, 0--9",
            "airline_late": "airline, 10--19", "early": "rounds 0--9", "late": "rounds 10--19", "all": "all"}
    rows = []
    for s_, ks in order.items():
        if s_ not in res:
            continue
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{lab[s_]}}}}} \\\\")
        for k in ks:
            c = res[s_][k]
            rec = c["recall"][0]
            gd = c["gain_dropped"][0]
            rows.append(f"{name[k]} & {c['rounds']} & {c['full_accepts']} & {ci(c['D_full'])} & "
                        f"{ci(c['seq_minus_full'])} & {'--' if rec is None else f'{100 * rec:.0f}'} & "
                        f"{c['dropped']:.1f} & {'--' if gd is None or c['dropped'] < 0.5 else pp(gd)} \\\\")
    out = ("\\begin{tabular}{@{}lrrllrrr@{}}\n\\toprule\n"
           "Rounds & $n$ & Acc. & $D$(full) [90\\%] & seq.$-$full [90\\%] & Recall & Drop. & Gain \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    tex.write_text(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "tables"])
    ap.add_argument("--traj", nargs="+")
    ap.add_argument("--json", default="results/r15/report.json")
    ap.add_argument("--tex", default="paper/tables/r15_cells.tex")
    a = ap.parse_args()
    if a.cmd == "tables":
        tables(Path(a.json), Path(a.tex))
    else:
        report(dict(x.split("=", 1) for x in a.traj), Path(a.json))


if __name__ == "__main__":
    main()
