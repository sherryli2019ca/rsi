"""Screen the oracle fixes the repair experiment gave its proposers (post hoc).

The oracle-fix group (no reading) gave the proposer, for 60 failing traces, the
step of largest rescue gain, the oracle's diagnosis and the correction that
rescued most often there. This applies the mechanical flags of
attrib.audit_rescore to those corrections: whether they change state, make
several calls in one turn, replace the write the agent made at that step, or
drop a value of it the user wrote or was shown just before (a confirmed choice).

  python -m attrib.oraclefix_screen <release>/data/phaseB --out results/attrib/phaseB_oraclefix_screen.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from attrib.audit_rescore import exchange, tau2_flags

CALL = re.compile(r"^call (\w+)\((.*)\)\s*$", re.M)


def screen(pb: Path) -> dict:
    rows = []
    for p in sorted(pb.glob("rounds/oracle_fix_noread/*/r*/analysis/digests/*.json")):
        state = p.parts[-5]
        dg = json.loads(p.read_text())
        if "corrected_action" not in dg:
            continue
        dom = "tau2_retail" if "retail" in state else "tau2_airline"
        rec = json.loads((pb / dom / "traces" / f"{state}_{dg['task_id']}.json").read_text())
        calls = []
        for m in CALL.finditer(dg["corrected_action"]):
            try:
                calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
            except json.JSONDecodeError:
                calls.append({"name": m.group(1), "arguments": m.group(2)})
        k = dg["decisive_step"]
        f = tau2_flags({"tool_calls": calls}, rec["steps"][k], exchange(rec, k))
        rows.append({"state": state, "task": dg["task_id"], "k": k, "message": not calls, **f})
    n = {"fixes": len(rows), "messages": sum(r["message"] for r in rows),
         "writes": sum(r["write"] for r in rows),
         "writes_replacing_agent_write": sum(r["write"] and r["conflict"] for r in rows),
         "writes_changing_confirmed_choice": sum(r["write"] and r["confirmed"] for r in rows),
         "several_calls": sum(r["multi"] for r in rows),
         "replaced_or_several_calls": sum(r["write"] and (r["conflict"] or r["multi"]) for r in rows)}
    return {"counts": n, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phaseb", help="release data/phaseB (rounds/, <domain>/traces/)")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = screen(Path(a.phaseb))
    print(json.dumps(res["counts"]))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
