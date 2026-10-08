"""Post-hoc analyses of experiment E1 (not registered), as reported in the paper.

  python -m verify.posthoc_e1 guard   [--runs runs/rrsi] [--out runs/verify]
  python -m verify.posthoc_e1 summary [--runs runs/rrsi] [--out runs/verify]
  python -m verify.posthoc_e1 robust  [--runs runs/rrsi] [--out runs/verify] [--src FILE]

guard: the registered analysis (verify/analyze.py, unchanged) with RRSI's
  domain guard applied to every rule's own evidence episodes. RRSI rejects a
  harness that raises errors in more than max_harness_error_rate (0.02) of its
  trials; the registered analysis applies that guard only through RRSI's own
  decision (rule `full`). Here a sample/replay/net draw also drops a candidate
  when more than 2% of that candidate's episodes in the draw ended in a harness
  error. Writes <out>/e1_analysis_guard_posthoc.json and leaves the primary
  <out>/e1_analysis.json as it was.

summary: from <out>/e1_analysis.json, prints the decision values without the
  rounds that contain a candidate whose deployment gain is below -10 points,
  the split-half reliability of candidates' deployment gains with and without
  those candidates, the correlation of each estimate with deployment gain,
  and replay start shifts for text and code changes (a candidate is a code
  change if its diff touches the harness's agent, tool or check modules).

guard also adds two baselines with a cheap execution check: nonecheck@10 and
  judgecheck@10 run 10 episodes of each candidate and drop it on a harness
  error, then accept a passing candidate at random or by the judge.

robust: from the guarded analysis (or --src), pairwise contrasts among full,
  sample@40, net@40, replaynull@40 and nonecheck@10 with the registered
  bootstrap (rounds and tasks), a bootstrap over blocks of rounds that share an
  incumbent, leave-one-round-out and leave-one-incumbent-out ranges, minimum
  detectable differences (80% power, one-sided 5%), Net(10^4) with its interval
  and break-even horizon, and the resolution of a single candidate's gain.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from . import analyze as A
from .state import ROOT, rounds

OUTLIER = -0.10
CHECK_RULES = ("nonecheck", "judgecheck")   # no verification / judge after a cheap execution check
CHECK_B = (10,)
CODE_FILES = {"agent.py", "tools.py", "checks.py"}


# ------------------------------------------------------------------ guard --
class _Err(float):
    """A reward of an episode that ended in a harness error."""


def _guard(runs: Path, out: Path):
    log = []
    orig_replays, orig_ev, orig_fix, orig_est = A._replays, A._ev, A._fix, A.estimate

    def replays(d):
        res = orig_replays(d)
        if not Path(d).is_dir():
            return res
        for p in sorted(Path(d).glob("*__*.json")):
            r = json.loads(p.read_text())
            if "harness_error" in r:
                t = str(r["task_id"])
                i = sorted(Path(d).glob(f"{t}__*.json")).index(p)
                res[t][i] = (_Err(res[t][i][0]),) + tuple(res[t][i][1:])
        return res

    def ev(path):
        e = orig_ev(path)
        if e is None:
            return e
        for sdir in Path(path).parent.glob("s[0-9]*"):
            s = int(sdir.name[1:])
            for p in sdir.glob("*.json"):
                r = json.loads(p.read_text())
                tr = e.per_task.get(str(r.get("task_id")))
                if "harness_error" in r and tr is not None and s < len(tr.rewards):
                    tr.rewards[s] = _Err(tr.rewards[s])
        return e

    def fix(rep, picks, rng, used):
        rows = orig_fix(rep, picks, rng, used)
        log.extend(r[0] for r in rows)
        return rows

    class Rng(random.Random):
        def sample(self, pop, k, **kw):
            res = super().sample(pop, k, **kw)
            if res and isinstance(res[0], tuple) and len(res[0]) == 7:   # evolve pairs
                log.extend(p[2] for p in res)
            return res

    class RandomNS:
        Random = Rng

    def estimate(rule, b, R, v, rng, mu0=None, null_vals=None, ep_tok=1.0):
        log.clear()
        e = orig_est(rule, b, R, v, rng, mu0, null_vals, ep_tok)
        if e is None or rule in ("judge", "full") or b >= 10 ** 6:
            return e
        if log and sum(isinstance(x, _Err) for x in log) / len(log) > 0.02:
            return (None,) + tuple(e[1:])
        return e

    orig_rules, orig_probs = A.rules, A.choice_probs

    def rules_():
        return orig_rules() + [(r, b, None) for r in CHECK_RULES for b in CHECK_B]

    def choice_probs(rule, b, R, cfg, seed, mu0=None, ep_tok=1.0):
        """nonecheck@b / judgecheck@b: run b episodes of each candidate (drawn from
        its full evaluation, as sample@b does), drop it if more than 2% of them
        ended in a harness error, then accept a passing candidate at random
        (nonecheck) or decide with the judge's estimate (judgecheck)."""
        if rule not in CHECK_RULES:
            return orig_probs(rule, b, R, cfg, seed, mu0, ep_tok)
        vs = list(R["cands"])
        if not vs:
            return {None: 1.0}, 0.0, 0.0
        rng, counts, eps = Rng(seed), {}, 0.0
        for _ in range(A.M):
            ok = []
            for v in vs:
                log.clear()
                e = orig_est("sample", b, R, v, rng, None, None, ep_tok)
                eps += 0 if e is None else e[3]
                if not (log and sum(isinstance(x, _Err) for x in log) / len(log) > 0.02):
                    ok.append(v)
            if rule == "nonecheck":
                for v in ok:
                    counts[v] = counts.get(v, 0) + 1 / len(ok)
                if not ok:
                    counts[None] = counts.get(None, 0) + 1
            else:
                ests = {v: (R["cands"][v]["judge"], 0.0, 0.0, R["cands"][v]["novelty"]) for v in ok if R["cands"][v]["judge"] is not None}
                w = A.decide(ests, cfg, 0.0)
                counts[w] = counts.get(w, 0) + 1
        return {k: n / A.M for k, n in counts.items()}, eps / A.M, eps / A.M

    A._replays, A._ev, A._fix, A.estimate, A.random = replays, ev, fix, estimate, RandomNS
    A.rules, A.choice_probs = rules_, choice_probs
    primary, backup = out / "e1_analysis.json", out / "e1_analysis.primary.bak"
    shutil.copy2(primary, backup)
    try:
        sys.argv = ["analyze", "--runs", str(runs), "--out", str(out)]
        A.main()
        shutil.move(primary, out / "e1_analysis_guard_posthoc.json")
    finally:
        shutil.copy2(backup, primary)
        backup.unlink()


# ---------------------------------------------------------------- summary --
def _mean_dep(d):
    return float(np.mean(list(d.values())))


def _summary(runs: Path, out: Path):
    r = json.loads((out / "e1_analysis.json").read_text())
    rows = [(T["name"], row) for T in r["tables"] for row in T["rows"]]
    bad = {(n, row["t"]) for n, row in rows if any(d and _mean_dep(d) < OUTLIER for d in row["dep"].values())}
    print("rounds with a candidate below", OUTLIER, ":", sorted(bad))
    keep = [(n, row) for n, row in rows if (n, row["t"]) not in bad]
    rv = lambda row, rule: sum(p * _mean_dep(row["dep"][c]) for c, p in row["probs"][rule].items()
                               if c != "null" and row["dep"].get(c))
    print(f"decision value without those rounds ({len(keep)} rounds), pp:")
    for rule in r["decision_value"]:
        print(f"  {rule:24s} {100 * np.mean([rv(row, rule) for _, row in keep]):+.2f}")

    k_h = {T["name"]: A.load_domain(T["name"]).cfg.get("heldout_k", 4) for T in r["tables"]}
    cands = []
    for name, row in rows:
        inc_commit = None
        for rnd in rounds(runs / name):
            if rnd.t == row["t"]:
                inc_commit, commits = rnd.inc_commit, {c.variant: c.commit for c in rnd.cands if c.measured}
        held = lambda c: {t: x["rewards"] for t, x in json.loads(
            (out / name / "jobs" / "heldout" / c[:12] / "eval.json").read_text())["per_task"].items()}
        h = held(inc_commit)
        for v, d in row["dep"].items():
            if not d:
                continue
            c, k = held(commits[v]), k_h[name]
            half = k // 2
            diff = (runs / name / f"r{row['t']}" / v / "diff.patch").read_text()
            files = {f.split("/harness/")[-1] for f in re.findall(r"^\+\+\+ b/(\S+)", diff, re.M)}
            cands.append({"dep": _mean_dep(d),
                          "h1": np.mean([np.mean(c[t][:half]) - np.mean(h[t][:half]) for t in h]),
                          "h2": np.mean([np.mean(c[t][half:k]) - np.mean(h[t][half:k]) for t in h]),
                          "evolve": row["evolve_dS"].get(v), "judge": row["judge"].get(v),
                          "replay": (row["replay_full"].get(v) or [None])[0],
                          "code": bool(files & CODE_FILES), "starts": row["starts"][v]})
    dep = np.array([x["dep"] for x in cands])
    ok = dep >= OUTLIER
    print(f"candidates {len(cands)}: mean dep {100 * dep.mean():+.2f} pp, positive {int((dep > 0).sum())}; "
          f"without outliers mean {100 * dep[ok].mean():+.2f} pp")
    for lab, m in (("all", np.ones(len(cands), bool)), ("without outliers", ok)):
        h1 = np.array([x["h1"] for x in cands])[m]
        h2 = np.array([x["h2"] for x in cands])[m]
        rhh = np.corrcoef(h1, h2)[0, 1]
        print(f"split-half r ({lab}) {rhh:+.2f}, Spearman-Brown {2 * rhh / (1 + rhh):+.2f}")
    from scipy.stats import spearmanr
    for key in ("evolve", "judge", "replay"):
        x = np.array([np.nan if c[key] is None else c[key] for c in cands], float)
        m = ~np.isnan(x)
        print(f"corr(dep, {key}): pearson {np.corrcoef(x[m], dep[m])[0, 1]:+.2f} "
              f"spearman {spearmanr(x[m], dep[m]).correlation:+.2f} "
              f"| without outliers pearson {np.corrcoef(x[m & ok], dep[m & ok])[0, 1]:+.2f}")
    for lab, code in (("text", False), ("code", True)):
        sel = [c for c in cands if c["code"] == code]
        st = [(e, q) for c in sel for e, q in c["starts"] if e is not None and q is not None]
        moved = sum(1 for c in sel if any(e < q for e, q in c["starts"] if e is not None and q is not None))
        print(f"{lab} changes: {len(sel)} candidates, replays live before the requested step "
              f"{sum(e < q for e, q in st)}/{len(st)}, candidates with any {moved}")


# ----------------------------------------------------------------- robust --
ROBUST_RULES = ("full", "sample@40", "net@40", "replaynull@40", "nonecheck@10")
Z80 = 1.645 + 0.8416          # one-sided alpha 0.05, power 0.80


def _per_round(tables, rng=None, block_of=None):
    """{rule: [(round value, dc_run share, C_verify), ...]} over the rounds of both
    domains. rng=None: the observed rounds. With rng: rounds (or, with block_of,
    blocks of rounds sharing an incumbent) and held-out tasks resampled within
    domain, as in verify/analyze.py's bootstrap."""
    out = {}
    for T in tables:
        rows = T["rows"]
        if rng is None:
            idx, tasks = range(len(rows)), None
        else:
            if block_of is None:
                idx = rng.integers(0, len(rows), len(rows))
            else:
                groups = {}
                for i, row in enumerate(rows):
                    groups.setdefault(block_of[T["name"]][row["t"]], []).append(i)
                keys = list(groups)
                idx = [i for k in rng.integers(0, len(keys), len(keys)) for i in groups[keys[k]]]
            all_tasks = sorted({t for row in rows for d in row["dep"].values() for t in d})
            tasks = list(rng.choice(all_tasks, len(all_tasks)))
        for i in idx:
            row = rows[i]
            for rule, probs in row["probs"].items():
                drun = sum(p * (row["dcr"].get(c) or 0.0) for c, p in probs.items() if c not in (None, "null"))
                out.setdefault(rule, []).append((A.round_value(row, rule, tasks), drun,
                                                 row["cost"][rule]["episode_equivalents"]))
    return out


def _dv(pr):
    return {k: float(np.mean([x[0] for x in v])) for k, v in pr.items()}


def _robust(runs: Path, out: Path, src: str = "e1_analysis_guard_posthoc.json"):
    r = json.loads((out / src).read_text())
    tables = r["tables"]
    block_of = {T["name"]: {rnd.t: rnd.inc_commit for rnd in rounds(runs / T["name"])} for T in tables}
    point = _dv(_per_round(tables))
    rules = [k for k in ROBUST_RULES if k in point] + [k for k in ("none", "judge", "judgecheck@10") if k in point]
    rng1, rng2 = np.random.default_rng(1), np.random.default_rng(2)
    boots = [_per_round(tables, rng1) for _ in range(A.B)]
    bboots = [_per_round(tables, rng2, block_of) for _ in range(A.B)]
    res = {"source": src, "decision_value": {}, "contrasts": {}, "net": {}}
    pc = lambda xs: [float(np.percentile(xs, 5)), float(np.percentile(xs, 95))]
    for k in rules:
        res["decision_value"][k] = {"est": point[k], "ci90": pc([_dv(b)[k] for b in boots]),
                                    "ci90_block": pc([_dv(b)[k] for b in bboots])}
    # leave one round out / one incumbent block out
    rows_all = [(T["name"], i) for T in tables for i in range(len(T["rows"]))]

    def dv_without(drop):
        sub = [{"name": T["name"], "rows": [row for i, row in enumerate(T["rows"]) if not drop(T["name"], i, row)]}
               for T in tables]
        return _dv(_per_round(sub))
    loo = [dv_without(lambda n, i, row, a=a, b=b: (n, i) == (a, b)) for a, b in rows_all]
    blocks = sorted({(n, c) for n, d in block_of.items() for c in d.values()})
    lbo = [dv_without(lambda n, i, row, bl=bl: (n, block_of[n][row["t"]]) == bl) for bl in blocks]
    for a in ROBUST_RULES:
        for b in ROBUST_RULES:
            if a >= b or a not in point or b not in point:
                continue
            xs = np.array([_dv(x)[a] - _dv(x)[b] for x in boots])
            xb = np.array([_dv(x)[a] - _dv(x)[b] for x in bboots])
            res["contrasts"][f"{a} - {b}"] = {
                "est": point[a] - point[b], "ci90": pc(xs), "p_le0": float(np.mean(xs <= 0)),
                "ci90_block": pc(xb), "sd": float(xs.std()), "sd_block": float(xb.std()),
                "mde80": float(Z80 * xs.std()), "mde80_block": float(Z80 * xb.std()),
                "loo_range": [float(min(d[a] - d[b] for d in loo)), float(max(d[a] - d[b] for d in loo))],
                "lbo_range": [float(min(d[a] - d[b] for d in lbo)), float(max(d[a] - d[b] for d in lbo))]}
    # Net(N) of full evaluation, with uncertainty, and its break-even horizon
    for k in ("full", "sample@40", "net@40"):
        for v in (1, 10):
            def net_and_be(pr, N=10 ** 4):
                x = pr[k]
                rate = v * np.mean([a for a, _, _ in x]) - np.mean([d for _, d, _ in x])
                cv = np.mean([c for _, _, c in x])
                return N * rate - cv, (cv / rate if rate > 0 else np.inf)
            pts = net_and_be(_per_round(tables))
            bs = np.array([net_and_be(b) for b in boots])
            fin = np.isfinite(bs[:, 1])
            be = np.where(fin, bs[:, 1], 1e12)     # 1e12: never breaks even
            res["net"][f"{k},v={v}"] = {
                "net_1e4": float(pts[0]), "net_1e4_ci90": pc(bs[:, 0]), "p_net_1e4_pos": float(np.mean(bs[:, 0] > 0)),
                "breakeven_N": float(pts[1]), "breakeven_p_finite": float(np.mean(fin)),
                "breakeven_q05": float(np.percentile(be, 5)), "breakeven_q50": float(np.percentile(be, 50))}
    # resolution of a single candidate's deployment gain
    se = [np.std(list(d.values()), ddof=1) / np.sqrt(len(d)) for T in tables for row in T["rows"]
          for d in row["dep"].values() if d]
    res["candidate_se_median"] = float(np.median(se))
    res["candidate_mde80_median"] = float(Z80 * np.median(se))
    res["n_blocks"] = {n: len(set(d.values())) for n, d in block_of.items()}
    (out / "e1_robust_posthoc.json").write_text(json.dumps(res, indent=1))
    pp = lambda x: f"{100 * x:+.2f}"
    print("blocks (incumbents) per domain:", res["n_blocks"])
    for k, d in res["decision_value"].items():
        print(f"DV {k:14s} {pp(d['est'])} [{pp(d['ci90'][0])}, {pp(d['ci90'][1])}]  "
              f"block [{pp(d['ci90_block'][0])}, {pp(d['ci90_block'][1])}]")
    for k, d in res["contrasts"].items():
        print(f"{k:28s} {pp(d['est'])} [{pp(d['ci90'][0])}, {pp(d['ci90'][1])}] P<=0 {d['p_le0']:.3f} "
              f"| block [{pp(d['ci90_block'][0])}, {pp(d['ci90_block'][1])}] | MDE80 {100 * d['mde80']:.2f} "
              f"(block {100 * d['mde80_block']:.2f}) | LOO [{pp(d['loo_range'][0])}, {pp(d['loo_range'][1])}] "
              f"LBO [{pp(d['lbo_range'][0])}, {pp(d['lbo_range'][1])}]")
    for k, d in res["net"].items():
        print(f"Net {k:16s} Net(1e4) {d['net_1e4']:8.0f} [{d['net_1e4_ci90'][0]:.0f}, {d['net_1e4_ci90'][1]:.0f}] "
              f"P>0 {d['p_net_1e4_pos']:.2f} | break-even N {d['breakeven_N']:.0f}, finite in "
              f"{d['breakeven_p_finite']:.2f} of resamples, q05 {d['breakeven_q05']:.0f}, median {d['breakeven_q50']:.0f}")
    print(f"single candidate: median SE {100 * res['candidate_se_median']:.2f} pp, "
          f"MDE80 {100 * res['candidate_mde80_median']:.2f} pp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("guard", "summary", "robust"))
    ap.add_argument("--src", default="e1_analysis_guard_posthoc.json",
                    help="robust: analysis file in --out to read (e.g. e1_analysis.json)")
    ap.add_argument("--runs", default=str(ROOT / "runs" / "rrsi"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "verify"))
    args = ap.parse_args()
    if args.cmd == "robust":
        _robust(Path(args.runs), Path(args.out), args.src)
    else:
        (_guard if args.cmd == "guard" else _summary)(Path(args.runs), Path(args.out))


if __name__ == "__main__":
    main()
