"""Post-hoc analyses of experiment E1 (not registered), as reported in the paper.

  python -m verify.posthoc_e1 guard   [--runs runs/rrsi] [--out runs/verify]
  python -m verify.posthoc_e1 summary [--runs runs/rrsi] [--out runs/verify]

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

    A._replays, A._ev, A._fix, A.estimate, A.random = replays, ev, fix, estimate, RandomNS
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("guard", "summary"))
    ap.add_argument("--runs", default=str(ROOT / "runs" / "rrsi"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "verify"))
    args = ap.parse_args()
    (_guard if args.cmd == "guard" else _summary)(Path(args.runs), Path(args.out))


if __name__ == "__main__":
    main()
