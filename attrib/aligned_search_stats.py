"""Exact agreement and cost of the aligned counterfactual search (post hoc).

The aligned search (attrib.search_maxgain, variant "aligned") reads the grading,
asks for the step whose correction most raises success over letting the agent
act again, tests every suspect within 40 replays and answers the largest
estimated gain. This adds what Table 1 needs besides its score: the share of
rescuable failures where it names the best rescue step, and its list-price cost
per failure (suspect call plus every corrected and null replay), averaged over
domains as for the other methods.

  python -m attrib.aligned_search_stats /home/user/attrib_runs/main --runs <dir with <domain>/search_maxgain/aligned> \
      --out results/attrib/aligned_search_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from attrib.admissibility import DOMAINS
from attrib.analyze import _usage_dollars, load
from attrib.pilot_report import replay_dollars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main")
    ap.add_argument("--runs", help="directory holding <domain>/search_maxgain/aligned (default: main)")
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = Path(a.runs or a.main)
    hit = n = 0
    dollars = {}
    for d in DOMAINS:
        v = load(Path(a.main) / d)
        sd = runs / d / "search_maxgain" / "aligned"
        res = {x["fid"]: x for x in json.loads((sd / "result.json").read_text())["failures"]}
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            n += 1
            hit += res[fid]["maxgain"] == g["decisive"]
        rep = 0.0
        for p in (sd / "replays").glob("*.json"):
            x = json.loads(p.read_text())
            if isinstance(x, dict):      # refs_*.json files list replay requests, not replays
                rep += replay_dollars(x)
        dollars[d] = round((sum(_usage_dollars(sd / "usage.jsonl").values()) + rep) / len(res), 5)
    out = {"n": n, "exact": round(hit / n, 4), "dollars_per_failure": round(sum(dollars.values()) / 3, 5),
           "dollars_by_domain": dollars}
    print(json.dumps(out))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
