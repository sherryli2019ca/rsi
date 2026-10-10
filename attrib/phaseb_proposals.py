"""Repair experiment without the slots that drafted nothing (post hoc).

The registered analysis (attrib.phaseb_analyze) gives a slot G = 0 when its
round produced no proposal, as when a candidate fails a gate, since the loop
then keeps its incumbent. This recomputes the six primary contrasts over the
slots whose round produced a proposal: per state, each group's mean G over its
proposing slots; a state enters a contrast only if both groups proposed there;
95% interval from the t distribution over those states.

  python -m attrib.phaseb_proposals [--analysis results/attrib/phaseB_analysis.json] [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}   # t quantiles by df
PRIMARY = [(g, h) for h in ("none", "rrsi") for g in ("first_write", "binary_search", "cf_search")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="results/attrib/phaseB_analysis.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    slots = json.loads(Path(a.analysis).read_text())["slots"]
    no_prop = {}
    by = {}
    for s in slots:
        no_prop[s["group"]] = no_prop.get(s["group"], 0) + (s["gate"] == "no_proposal")
        if s["gate"] != "no_proposal":
            by.setdefault((s["group"], s["state"]), []).append(s["G"])
    states = sorted({s["state"] for s in slots})
    out = {"slots": len(slots), "no_proposal": no_prop, "drafts": len(slots) - sum(no_prop.values()),
           "contrasts": {}}
    for g, h in PRIMARY:
        d = [statistics.mean(by[g, st]) - statistics.mean(by[h, st]) for st in states
             if (g, st) in by and (h, st) in by]
        m = statistics.mean(d)
        se = statistics.stdev(d) / len(d) ** .5
        t = T975[len(d) - 1]
        out["contrasts"][f"{g}-{h}"] = {"states": len(d), "estimate_pp": round(100 * m, 2),
                                        "ci95_pp": [round(100 * (m - t * se), 2), round(100 * (m + t * se), 2)]}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
