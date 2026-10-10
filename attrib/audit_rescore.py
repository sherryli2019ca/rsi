"""Re-score the methods without the oracle's state-changing corrections
(post hoc, after the human audit of 60 corrections).

Both auditors judged most audited tau2 corrections that change state invalid
(they replaced the item, payment or baggage count the user had last confirmed,
made several calls in one turn, or broke the policy otherwise). Three rules
drop corrections, counting them as not proposed: R'_k = (1/K) sum_i c_ki g_ki
with c_ki = 0 for dropped corrections; rescuable = max R' >= 0.5. Methods are
scored as in attrib.second_oracle.

  tau2_writes      every tau2 correction that calls a state-changing tool (any
                   call outside domains.tau2.common.READ_PREFIXES and
                   transfers); an upper bound, as it also drops valid ones
  tau2_conflicts   only tau2 corrections that make two or more tool calls in
                   one turn (the policy allows one at a time), or that change
                   state differently from the write the agent made at that step
                   (another tool or other arguments; the policy has the agent
                   confirm a write with the user before making it)
  all_writes       tau2_writes plus every AppWorld correction that calls a
                   state-changing API (attrib.aw.is_write, the submission
                   included)

The audited tau2 corrections that change state are checked against each rule.

  python -m attrib.audit_rescore /home/user/attrib_runs/main [--traces <release>/data/phaseA] \
      [--key <release>/data/audit/audit_key_mechanical.json] --out results/attrib/audit_rescore.json
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

RULES = ("tau2_writes", "tau2_conflicts", "all_writes")
AUDIT = Path("/mnt/project-files/reviews/paper2-audit/audit_key_mechanical.json")


def _is_write(name: str) -> bool:
    return not name.startswith(READ_PREFIXES) and not name.startswith("transfer")


def tau2_flags(action: dict, observed: dict) -> dict:
    calls = [(c["name"], c.get("arguments")) for c in action.get("tool_calls") or []]
    writes = [c for c in calls if _is_write(c[0])]
    obs = [(b["name"], b.get("input")) for b in observed["assistant"]
           if b["type"] == "tool_use" and _is_write(b["name"])]
    return {"write": bool(writes), "multi": len(calls) >= 2,
            "conflict": bool(obs) and any(w not in obs for w in writes)}


def aw_write(o: dict) -> bool:
    from attrib.aw import _CALL, is_write
    text = json.dumps(o.get("action")) + str(o.get("force") or "")
    return any(is_write(a, b) for a, b in _CALL.findall(text.replace("\\n", "\n")))


def _trace(v: dict, d: str, fid: str, traces: Path | None) -> dict:
    p = Path(v["fails"][fid]["trace"])
    if not p.exists() and traces is not None:
        p = traces / d / "traces" / f"{fid}.json"
    return json.loads(p.read_text())


def flags(main: Path, d: str, v: dict, traces: Path | None) -> dict:
    """(fid, k, i) -> flags for every effective correction of the registered profile."""
    odir = main / d / "gt" / "a" / "oracle"
    out = {}
    for fid, g in v["gt"].items():
        rec = None if d == "appworld" else _trace(v, d, fid, traces)
        for st in g["steps"]:
            for i, s in enumerate(st["samples"]):
                if not (s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2
                        and st["null"] is not None):
                    continue
                o = json.loads((odir / f"{fid}_k{st['k']}_o{i}.json").read_text())
                if d == "appworld":
                    out[(fid, st["k"], i)] = {"write": aw_write(o), "multi": False, "conflict": False}
                else:
                    out[(fid, st["k"], i)] = tau2_flags(o.get("action") or {}, rec["steps"][st["k"]])
    return out


def dropped(d: str, f: dict, rule: str) -> bool:
    if rule == "all_writes":
        return f["write"]
    if d == "appworld":
        return False
    if rule == "tau2_writes":
        return f["write"]
    return f["write"] and (f["multi"] or f["conflict"])


def profiles(v: dict, d: str, fl: dict, rule: str) -> tuple[dict, dict]:
    prof, n = {}, {"effective": 0, "dropped": 0}
    for fid, g in v["gt"].items():
        R = []
        for st in g["steps"]:
            tot = 0.0
            for i, s in enumerate(st["samples"]):
                f = fl.get((fid, st["k"], i))
                if f is None:
                    continue
                n["effective"] += 1
                if dropped(d, f, rule):
                    n["dropped"] += 1
                    continue
                tot += s["corrected"] - st["null"]
            R.append(tot / K)
        prof[fid] = R
    return prof, n


def audit_check(fl: dict, sheets: dict, key_path: Path) -> dict:
    """The rules against the auditors on the audited tau2 corrections that change state."""
    key = json.loads(key_path.read_text())
    labs = {name: {r["id"]: r["Q3_valid"] == "yes" for r in rows} for name, rows in sheets.items()}
    out = {}
    for rule in ("tau2_writes", "tau2_conflicts"):
        rows = []
        for x in key:
            if x["domain"] == "appworld":
                continue
            f = fl[x["domain"]].get((x["fid"], x["k"], x["i"]))
            if f is None or not f["write"]:
                continue
            rows.append({"item": x["item"], "dropped": dropped(x["domain"], f, rule),
                         **{n: lab[x["item"]] for n, lab in labs.items()}})
        res = {"n": len(rows), "dropped": sum(r["dropped"] for r in rows),
               "dropped_items": [r["item"] for r in rows if r["dropped"]]}
        for n in labs:
            res[n] = {"kept_valid": sum(not r["dropped"] and r[n] for r in rows),
                      "kept_invalid": sum(not r["dropped"] and not r[n] for r in rows),
                      "dropped_valid": sum(r["dropped"] and r[n] for r in rows),
                      "dropped_invalid": sum(r["dropped"] and not r[n] for r in rows)}
        out[rule] = res
    return out


def main():
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--traces", help="release data/phaseA, for traces not at their recorded path")
    ap.add_argument("--sheets", nargs="*", default=["auditor1=attrib/audit/auditor1.csv",
                                                    "auditor2=attrib/audit/auditor2.csv"])
    ap.add_argument("--key", default=str(AUDIT), help="the audit sample's mechanical key (item -> correction)")
    ap.add_argument("--out")
    a = ap.parse_args()
    main_dir = Path(a.main)
    traces = Path(a.traces) if a.traces else None
    data = {d: load(main_dir / d) for d in DOMAINS}
    methods = [m for m in METHODS + EXTRA if all(m in v["picks"] for v in data.values())]
    reg = {m: _mean([v["gt"][f]["R"][k] if isinstance(k := v["picks"][m].get(f), int) and 0 <= k < len(v["gt"][f]["R"])
                     else 0.0 for v in data.values() for f in v["gt"] if v["gt"][f]["decisive"] is not None])
           for m in METHODS}
    fl = {d: flags(main_dir, d, data[d], traces) for d in DOMAINS}
    res = {}
    for rule in RULES:
        prof, counts = {}, {}
        for d in DOMAINS:
            prof[d], counts[d] = profiles(data[d], d, fl[d], rule)
        r = _ranking(data, prof, random.Random(0), methods)
        r["kendall_tau_vs_registered"] = round(kendall({m: r["mean"][m] for m in METHODS}, reg), 3)
        res[rule] = {"counts": counts, "pooled": r}
        print(rule, json.dumps(counts))
        print("  ", r["n_rescuable"], "fw", r["mean"]["first_write"], "rank", r["first_write_rank"], "best",
              r["best_llm"], r["mean"][r["best_llm"]], "fw-best", r.get("first_write_minus_best_llm"),
              "fw-bs", r["first_write_minus_binary_search"], "tau", r["kendall_tau_vs_registered"])
        print("  ", {m: r["mean"][m] for m in methods})
    sheets = {}
    for s in a.sheets:
        name, path = s.split("=", 1)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            sheets[name] = list(csv.DictReader(fh))
    res["audit_check"] = audit_check(fl, sheets, Path(a.key))
    print(json.dumps(res["audit_check"], indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
