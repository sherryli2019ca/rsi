"""Re-score the methods without the oracle's state-changing tau2 corrections
(post hoc, after the human audit of 60 corrections).

Both auditors judged most audited tau2 corrections that change state invalid
(they replaced the item, payment or baggage count the user had last confirmed,
or broke the policy). No mechanical rule can find exactly those, so this
re-scoring drops every tau2 correction that calls a state-changing tool
(any call outside domains.tau2.common.READ_PREFIXES and transfers), counting
it as not proposed: R'_k = (1/K) sum_i c_ki g_ki with c_ki = 0 for dropped
corrections; rescuable = max R' >= 0.5; AppWorld profiles are unchanged.
Methods are scored as in attrib.second_oracle.

  python -m attrib.audit_rescore /home/user/attrib_runs/main --out results/attrib/audit_rescore.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from attrib.admissibility import DOMAINS, EXTRA, K
from attrib.analyze import _mean, load
from attrib.robustness import METHODS, kendall
from attrib.second_oracle import _ranking
from domains.tau2.common import READ_PREFIXES


def _writes(action: dict) -> bool:
    return any(not c["name"].startswith(READ_PREFIXES) and not c["name"].startswith("transfer")
               for c in action.get("tool_calls") or [])


def profiles(main: Path, d: str, v: dict) -> tuple[dict, dict]:
    odir = main / d / "gt" / "a" / "oracle"
    prof, n = {}, {"effective": 0, "dropped": 0}
    for fid, g in v["gt"].items():
        R = []
        for st in g["steps"]:
            tot = 0.0
            for i, s in enumerate(st["samples"]):
                if not (s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2
                        and st["null"] is not None):
                    continue
                n["effective"] += 1
                if d != "appworld":
                    o = json.loads((odir / f"{fid}_k{st['k']}_o{i}.json").read_text())
                    if _writes(o.get("action") or {}):
                        n["dropped"] += 1
                        continue
                tot += s["corrected"] - st["null"]
            R.append(tot / K)
        prof[fid] = R
    return prof, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--out")
    a = ap.parse_args()
    main_dir = Path(a.main)
    data = {d: load(main_dir / d) for d in DOMAINS}
    methods = [m for m in METHODS + EXTRA if all(m in v["picks"] for v in data.values())]
    reg = {m: _mean([v["gt"][f]["R"][k] if isinstance(k := v["picks"][m].get(f), int) and 0 <= k < len(v["gt"][f]["R"])
                     else 0.0 for v in data.values() for f in v["gt"] if v["gt"][f]["decisive"] is not None])
           for m in METHODS}
    prof, counts = {}, {}
    for d in DOMAINS:
        prof[d], counts[d] = profiles(main_dir, d, data[d])
    res = {"counts": counts}
    r = _ranking(data, prof, random.Random(0), methods)
    r["kendall_tau_vs_registered"] = round(kendall({m: r["mean"][m] for m in METHODS}, reg), 3)
    res["pooled"] = r
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(counts))
    for nm, x in (("pooled", r),):
        print(nm, x["n_rescuable"], "fw", x["mean"]["first_write"], "rank", x["first_write_rank"], "best",
              x["best_llm"], x["mean"][x["best_llm"]], "fw-bs", x["first_write_minus_binary_search"],
              "fw-bsgain", x.get("first_write_minus_binary_search_gain_pro"), "tau", x.get("kendall_tau_vs_registered"))
        print("  ", {m: x["mean"][m] for m in methods})


if __name__ == "__main__":
    main()
