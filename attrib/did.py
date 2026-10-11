"""How much each evaluation choice moves the first write's lead (post hoc).

Table 2 reports, under each evaluation choice, the first write's difference from
the best LLM method, (FW - best)_choice. Here the change from the registered
comparison, (FW - best)_choice - (FW - best)_registered, gets a 95% interval
from one cluster bootstrap over tasks (10,000 draws, seed 0) shared by every
choice and cohort, so the two differences in each change are paired: a draw
takes tasks with replacement and recomputes both on the failures of those
tasks that fall in each cohort. Profiles stay fixed, as in the main text.

Choices (cohorts: failures rescuable under the choice, "reselected"; and the
registered 157, "fixed"):
  registered      binary search is the best LLM method
  judges_gain     judges asked for the gain over null (best: binary search asked
                  for the gain; fixed only)
  aligned_search  counterfactual search aligned with the metric (fixed only; it
                  ran on the 157)
  no_null         profiles without null replays
  blind           the oracle without the grading at the named steps (fixed only)
  tau2_confirmed, tau2_replaced, all_writes
                  attrib.audit_rescore's filters
The best method of a choice is fixed at its point estimate (as in Table 2), not
re-chosen in each draw.

  python -m attrib.did /home/user/attrib_runs/main --release <release>/data/phaseA \
      --blind <dir with <domain>/gt/blind/result.json> --out results/attrib/did.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attrib.analyze import _mean, load
from attrib.audit_rescore import flags, profiles
from attrib.robustness import LLM
from attrib.robustness import profile as rprofile

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
THRESH = 0.5
DRAWS = 10_000
RULES = ("tau2_confirmed", "tau2_replaced", "all_writes")


def _score(R, k) -> float:
    if isinstance(k, dict):
        k = k.get("step")
    return R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0


def build(main: Path, release: Path, blind: Path) -> tuple[list, dict]:
    data = {d: load(main / d) for d in DOMAINS}
    for d, v in data.items():
        for m in ("binary_search_gain_pro", "all_at_once_gain_pro"):
            v["picks"][m] = {r["fid"]: r.get("step") for r in (json.loads(p.read_text()) for p in
                             (release / d / "methods" / m).glob("*.json"))}
        res = json.loads((release / d / "search_maxgain" / "aligned" / "result.json").read_text())
        v["picks"]["aligned_search"] = {x["fid"]: x["maxgain"] for x in res["failures"]}
    traces = release
    prof = {"registered": {d: {f: rprofile(g) for f, g in v["gt"].items()} for d, v in data.items()},
            "no_null": {d: {f: rprofile(g, use_null=False) for f, g in v["gt"].items()} for d, v in data.items()}}
    for d, v in data.items():
        for f, g in v["gt"].items():
            assert np.allclose(prof["registered"][d][f], g["R"], atol=1e-3), (d, f)
    fl = {d: flags(main, d, data[d], traces) for d in DOMAINS}
    for rule in RULES:
        prof[rule] = {d: profiles(data[d], d, fl[d], rule)[0] for d in DOMAINS}
    bl = {d: {f: {s["k"]: s["R"] for s in x}
              for f, x in json.loads((blind / d / "gt" / "blind" / "result.json").read_text())["failures"].items()}
          for d in DOMAINS}
    methods = list(LLM) + ["first_write", "binary_search_gain_pro", "aligned_search"]
    rows = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            r = {"domain": d, "fid": f, "cluster": f"{d}/{v['fails'][f]['task_id']}",
                 "reg157": g["decisive"] is not None}
            for c, P in prof.items():
                R = P[d][f]
                r[f"{c}|resc"] = bool(R) and max(R) >= THRESH
                for m in methods:
                    if m in ("binary_search_gain_pro", "aligned_search") and c != "registered":
                        continue
                    r[f"{c}|{m}"] = _score(R, v["picks"][m].get(f))
            if r["reg157"]:
                for m in list(LLM) + ["first_write"]:
                    k = v["picks"][m].get(f)
                    ok = isinstance(k, int) and 0 <= k < len(g["R"])
                    r[f"blind|{m}"] = bl[d][f][k] if ok else 0.0
            rows.append(r)
    return rows, data


def _best(rows: list, c: str, cohort: str) -> str:
    sel = [r for r in rows if r[cohort]]
    return max(LLM, key=lambda m: _mean([r[f"{c}|{m}"] for r in sel]))


def spec(rows: list) -> list[dict]:
    """(name, profile key, cohort column, best method) for each row of Table 2."""
    out = [{"choice": "registered", "cohort": "fixed", "prof": "registered", "col": "reg157",
            "best": _best(rows, "registered", "reg157")},
           {"choice": "judges_gain", "cohort": "fixed", "prof": "registered", "col": "reg157",
            "best": "binary_search_gain_pro"},
           {"choice": "aligned_search", "cohort": "fixed", "prof": "registered", "col": "reg157",
            "best": "aligned_search"}]
    for c in ("no_null",) + RULES:
        out.append({"choice": c, "cohort": "reselected", "prof": c, "col": f"{c}|resc",
                    "best": _best(rows, c, f"{c}|resc")})
        out.append({"choice": c, "cohort": "fixed", "prof": c, "col": "reg157", "best": _best(rows, c, "reg157")})
    out.insert(5, {"choice": "blind", "cohort": "fixed", "prof": "blind", "col": "reg157",
                   "best": _best([r for r in rows if r["reg157"]], "blind", "reg157")})
    return out


def analyse(rows: list) -> dict:
    clusters = sorted({r["cluster"] for r in rows})
    ci = np.array([clusters.index(r["cluster"]) for r in rows])
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
                  for _ in range(DRAWS)])[:, ci].astype(float)            # draws x failures
    S = spec(rows)
    est, boot = [], []
    for s in S:
        mask = np.array([bool(r[s["col"]]) for r in rows], float)
        diff = np.array([(r[f"{s['prof']}|first_write"] - r[f"{s['prof']}|{s['best']}"]) if r[s["col"]] else 0.0
                         for r in rows])
        est.append(float((mask * diff).sum() / mask.sum()))
        boot.append((W * mask * diff).sum(1) / (W * mask).sum(1))
    ref = boot[0]

    def iv(x):
        return [round(float(np.quantile(x, .025)), 4), round(float(np.quantile(x, .975)), 4)]
    out = []
    for s, e, b in zip(S, est, boot):
        out.append({**{k: s[k] for k in ("choice", "cohort", "best")},
                    "n": int(sum(bool(r[s["col"]]) for r in rows)),
                    "first_write_minus_best": [round(e, 4)] + iv(b),
                    "change_vs_registered": [round(e - est[0], 4)] + iv(b - ref)})
    return {"draws": DRAWS, "rows": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--release", required=True, help="release data/phaseA: gain-asked picks, aligned search, traces")
    ap.add_argument("--blind", required=True, help="dir with <domain>/gt/blind/result.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows, _ = build(Path(a.main), Path(a.release), Path(a.blind))
    res = analyse(rows)
    for r in res["rows"]:
        print(f"{r['choice']:15s} {r['cohort']:10s} n={r['n']:3d} best={r['best']:22s} "
              f"FW-best {r['first_write_minus_best']}  change {r['change_vs_registered']}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
