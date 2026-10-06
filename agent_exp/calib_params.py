"""Write results/calibrated_params.json from the measured real-agent runs:
check sensitivity / false-positive rate, spurious recovery b, analyst accuracy r
and the single-step / full-replay cost ratio.

  python -m agent_exp.calib_params
"""
from __future__ import annotations

import json

import numpy as np


def main():
    P = {}
    real = json.load(open("results/real.json"))
    cal = [s["calibration"] for s in real.values()]
    pf = sum(c["p_pass_given_fail"][0] * c["p_pass_given_fail"][1] for c in cal) / \
        sum(c["p_pass_given_fail"][1] for c in cal)
    ps = sum(c["p_pass_given_success"][0] * c["p_pass_given_success"][1] for c in cal) / \
        sum(c["p_pass_given_success"][1] for c in cal)
    P["shopdesk"] = {"sens": ps, "fpr": pf,
                     "b": 0.05,  # no null replays on ShopDesk: simulator default, not measured
                     "b_measured": False,
                     "r": float(np.mean([s["attr_acc"] for s in real.values()])),
                     "cost_ratio": float(np.mean([c["tok_single"] / c["tok_full"] for c in cal]))}
    for env, fn in [("tau2_retail", "runs/tau2_retail/evaluation_rep20.json"),
                    ("tau2_airline", "runs/tau2_airline/evaluation_heldout.json")]:
        e = json.load(open(fn))
        P[env] = {"sens": e["calibration"]["sens"], "fpr": e["calibration"]["fpr"],
                  "b": e["b_hat"], "b_measured": True, "r": e["attr_acc_injected"],
                  "cost_ratio": e["costs"][1]}
    with open("results/calibrated_params.json", "w") as fh:
        json.dump(P, fh, indent=1)
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
