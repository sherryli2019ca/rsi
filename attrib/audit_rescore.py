"""Re-score the methods without the oracle's state-changing corrections
(post hoc, after the human audit of 60 corrections).

Both auditors rejected most audited tau2 corrections that replaced the item,
option or reason the user had last confirmed, and both judged several calls in
one turn by their content. Five rules drop corrections, counting them as not
proposed: R'_k = (1/K) sum_i c_ki g_ki with c_ki = 0 for dropped corrections;
rescuable = max R' >= 0.5. Methods are scored as in attrib.second_oracle, on the
failures rescuable under each rule and on the registered rescuable failures kept
as a fixed cohort (as are the profiles without null replays).

  tau2_confirmed   only tau2 corrections that change a write the agent made at
                   that step (another tool or other arguments) and drop a value
                   of it (an identifier, option or reason, as text) that the
                   user wrote or the agent showed in the exchange the user
                   answered last before the step: the direct reading of
                   overriding the user's confirmed choice
  tau2_writes      every tau2 correction that calls a state-changing tool (any
                   call outside domains.tau2.common.READ_PREFIXES and
                   transfers); an upper bound, as it also drops valid ones
  tau2_conflicts   only tau2 corrections that make two or more tool calls in
                   one turn (the policy allows one at a time), or that change
                   state differently from the write the agent made at that step
                   (another tool or other arguments; the policy has the agent
                   confirm a write with the user before making it)
  tau2_replaced    only tau2 corrections that change state differently from the
                   write the agent made at that step (the lenient reading:
                   several calls in one turn are kept)
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
from attrib.analyze import _boot, _mean, load
from attrib.robustness import LLM, METHODS, kendall
from attrib.robustness import profile as rprofile
from attrib.second_oracle import _ranking
from domains.tau2.common import READ_PREFIXES

RULES = ("tau2_confirmed", "tau2_replaced", "tau2_conflicts", "tau2_writes", "all_writes")
AUDIT = Path("/mnt/project-files/reviews/paper2-audit/audit_key_mechanical.json")


def _is_write(name: str) -> bool:
    return not name.startswith(READ_PREFIXES) and not name.startswith("transfer")


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


def exchange(rec: dict, k: int) -> str:
    """The user's last message before step k and the agent's text since the user's previous one."""
    us = [j for j in range(k) if rec["steps"][j].get("user")]
    if not us:
        return str(rec.get("opening") or "")
    j, i0 = us[-1], (us[-2] + 1 if len(us) > 1 else 0)
    agent = " ".join(b.get("text", "") for s in rec["steps"][i0:j + 1] for b in s["assistant"] if b["type"] == "text")
    return rec["steps"][j]["user"] + " " + agent


def tau2_flags(action: dict, observed: dict, shown: str = "") -> dict:
    calls = [(c["name"], c.get("arguments")) for c in action.get("tool_calls") or []]
    writes = [c for c in calls if _is_write(c[0])]
    obs = [(b["name"], b.get("input")) for b in observed["assistant"]
           if b["type"] == "tool_use" and _is_write(b["name"])]
    conflict = bool(obs) and any(w not in obs for w in writes)
    kept = {json.dumps(x) for w in writes for x in _leaves(w[1])}
    lost = [x for o in obs for x in _leaves(o[1]) if isinstance(x, str) and len(x) >= 3 and json.dumps(x) not in kept]
    return {"write": bool(writes), "multi": len(calls) >= 2, "conflict": conflict,
            "confirmed": conflict and bool(writes) and any(x.lower() in shown.lower() for x in lost)}


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
                    out[(fid, st["k"], i)] = {"write": aw_write(o), "multi": False, "conflict": False,
                                              "confirmed": False}
                else:
                    out[(fid, st["k"], i)] = tau2_flags(o.get("action") or {}, rec["steps"][st["k"]],
                                                        exchange(rec, st["k"]))
    return out


def dropped(d: str, f: dict, rule: str) -> bool:
    if rule == "all_writes":
        return f["write"]
    if d == "appworld":
        return False
    if rule == "tau2_writes":
        return f["write"]
    if rule == "tau2_confirmed":
        return f["write"] and f["confirmed"]
    if rule == "tau2_replaced":
        return f["write"] and f["conflict"]
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


def audit_check(fl: dict, sheets: dict, key_path: Path, main: Path | None = None) -> dict:
    """The rules against the auditors on the audited tau2 corrections that change state (with
    main, also over distinct corrections: samples of one step can propose the same action)."""
    key = json.loads(key_path.read_text())
    seen, dup = {}, set()
    if main is not None:
        for x in key:
            o = json.loads((main / x["domain"] / "gt" / "a" / "oracle" / f"{x['fid']}_k{x['k']}_o{x['i']}.json").read_text())
            sig = (x["domain"], x["fid"], x["k"], json.dumps(o.get("action"), sort_keys=True))
            if sig in seen:
                dup.add(x["item"])
            seen.setdefault(sig, x["item"])
    labs = {name: {r["id"]: r["Q3_valid"] == "yes" for r in rows} for name, rows in sheets.items()}
    out = {}
    for rule in ("tau2_confirmed", "tau2_replaced", "tau2_conflicts", "tau2_writes"):
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
        if main is not None:
            dr = [r for r in rows if r["item"] not in dup]
            res["distinct"] = {"n": len(dr), "dropped": sum(r["dropped"] for r in dr),
                               **{n: {"agree": sum(r["dropped"] != r[n] for r in dr)} for n in labs}}
        out[rule] = res
    return out


def fixed_cohort(data: dict, prof: dict, methods: list, reg: dict, rng) -> dict:
    """Score on the registered rescuable failures, whatever the profile says is rescuable."""
    rows = []
    for d, v in data.items():
        for f, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            R = prof[d][f]
            r = {"cluster": f"{d}/{v['fails'][f]['task_id']}"}
            for m in methods:
                k = v["picks"][m].get(f)
                r[m] = R[k] if isinstance(k, int) and 0 <= k < len(R) else 0.0
            rows.append(r)
    mean = {m: round(_mean([r[m] for r in rows]), 4) for m in methods}
    best = max(LLM, key=lambda m: mean[m])

    def diff(a, b):
        return _boot([{"d": r[a] - r[b], "cluster": r["cluster"]} for r in rows],
                     lambda xs: _mean([x["d"] for x in xs]), rng)
    return {"n": len(rows), "mean": mean, "best_llm": best,
            "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
            "first_write_minus_best_llm": diff("first_write", best),
            "first_write_minus_binary_search": diff("first_write", "binary_search_pro"),
            "kendall_tau_vs_registered": round(kendall({m: mean[m] for m in METHODS}, reg), 3)}


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
    res = {"fixed_cohort": {}}
    for name, use_null in (("registered", True), ("no_null", False)):
        prof = {d: {f: rprofile(g, use_null=use_null) for f, g in v["gt"].items()} for d, v in data.items()}
        res["fixed_cohort"][name] = fixed_cohort(data, prof, methods, reg, random.Random(0))
        if not use_null:
            r = _ranking(data, prof, random.Random(0), methods)
            r["kendall_tau_vs_registered"] = round(kendall({m: r["mean"][m] for m in METHODS}, reg), 3)
            res["no_null"] = {"pooled": r}
    for rule in RULES:
        prof, counts = {}, {}
        for d in DOMAINS:
            prof[d], counts[d] = profiles(data[d], d, fl[d], rule)
        r = _ranking(data, prof, random.Random(0), methods)
        r["kendall_tau_vs_registered"] = round(kendall({m: r["mean"][m] for m in METHODS}, reg), 3)
        res[rule] = {"counts": counts, "pooled": r}
        res["fixed_cohort"][rule] = fixed_cohort(data, prof, methods, reg, random.Random(0))
        print(rule, json.dumps(counts))
        print("  ", r["n_rescuable"], "fw", r["mean"]["first_write"], "rank", r["first_write_rank"], "best",
              r["best_llm"], r["mean"][r["best_llm"]], "fw-best", r.get("first_write_minus_best_llm"),
              "fw-bs", r["first_write_minus_binary_search"], "tau", r["kendall_tau_vs_registered"])
        print("  ", {m: r["mean"][m] for m in methods})
    # share of tau2 corrections at the first write's steps (registered rescuable failures) each rule drops
    at_fw = [(d, f) for d in DOMAINS if d != "appworld"
             for (fid, k, _), f in fl[d].items()
             if data[d]["gt"][fid]["decisive"] is not None and data[d]["picks"]["first_write"].get(fid) == k]
    res["first_write_steps"] = {"tau2_corrections": len(at_fw),
                                **{rule: sum(dropped(d, f, rule) for d, f in at_fw) for rule in RULES}}
    print("first-write steps", res["first_write_steps"])
    sheets = {}
    for s in a.sheets:
        name, path = s.split("=", 1)
        with open(path, newline="", encoding="utf-8-sig") as fh:
            sheets[name] = list(csv.DictReader(fh))
    res["audit_check"] = audit_check(fl, sheets, Path(a.key), main_dir)
    print(json.dumps(res["audit_check"], indent=1))
    for name, r in res["fixed_cohort"].items():
        print("fixed", name, r["n"], "fw", r["mean"]["first_write"], "rank", r["first_write_rank"], "best",
              r["best_llm"], r["mean"][r["best_llm"]], r["first_write_minus_best_llm"], "tau", r["kendall_tau_vs_registered"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
