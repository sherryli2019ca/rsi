"""Phase B with an oracle step pointer (post hoc, second review of paper 2, W4).

The registered groups pointed the proposer at the step named by first write,
binary search or counterfactual search. The oracle group (attrib.phaseb,
POSTHOC_GROUPS) points at the step with the largest rescue gain in each
trace's own profile (attrib.phaseb_accuracy), the best pointer the protocol can
define, with the same digest format and the same rounds, gates and held-out
deployment. Run only after every oracle-group deployment has finished.

  python -m attrib.phaseb_oracle [--out <json>]

Slot-level gains G and the same contrasts as the registered analysis
(attrib.phaseb_analyze.contrast: permutation p within states, t interval over
the 8 states), oracle against each registered group and against the three
step groups pooled; round-level gains (RRSI's own acceptance); the groups
table with the oracle group added.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from attrib.phaseb import GROUPS, POSTHOC_GROUPS, STEP_GROUPS
from attrib.phaseb_analyze import contrast, groups_table, round_level, signflip, slots

ALL = GROUPS + POSTHOC_GROUPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = slots(ALL)
    rl = round_level(rows)
    res = {"n_slots": len(rows), "groups": groups_table(rows, ALL),
           "slot_level": {h: contrast(rows, ("oracle",), (h,)) for h in GROUPS},
           "slot_level_vs_step_groups": contrast(rows, ("oracle",), STEP_GROUPS),
           "round_level": {h: (lambda c: {**c, "p_signflip": signflip(c), "p_perm": None})(
               contrast(rl, ("oracle",), (h,))) for h in GROUPS},
           "round_level_means": {g: float(np.mean([r["G"] for r in rl if r["group"] == g])) for g in ALL},
           "slots": [r for r in rows if r["group"] == "oracle"]}
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "slots"}, indent=1))


if __name__ == "__main__":
    main()
