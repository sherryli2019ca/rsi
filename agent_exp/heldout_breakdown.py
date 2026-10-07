"""Held-out success on retail split by episode type: clean episodes (natural
failures only), fault-injected episodes, and all episodes except those with the
injected step-limit fault. Re-uses the runs in heldout.json (no new episodes).

  python -m agent_exp.heldout_breakdown --out runs/tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from agent_exp.bank import Domain, _load, _save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    args = ap.parse_args()
    D = Domain(args.domain)
    res = _load(os.path.join(args.out, "heldout.json"))
    n = len(res["runs"]["none"])
    trials = n // len(D.split["test"])
    # the same episode list as bank_heldout.py
    rng = random.Random(7)
    mix = [None, None] + list(D.faults)
    test = [(tid, rng.choice(mix), t) for tid in D.split["test"] for t in range(trials)]
    var = np.array([str(v) for _, v, _ in test])
    groups = {"all": var == var, "clean": var == "None", "injected": var != "None",
              "no_step_fault": var != "F_steps"}
    sets = {}
    for k, key in res["sets"].items():
        m, B, s = k.rsplit("|", 2)
        sets.setdefault(f"{m}|{B}", []).append(key)
    out = {"n_episodes": {g: int(v.sum()) for g, v in groups.items()}}

    def stats(keys):
        r = {}
        for g, sel in groups.items():
            ys = np.array([np.array(res["runs"][k])[sel].mean() for k in keys])
            # s.e. over seeds of the patch-set choice plus episode noise of one set
            ep = np.mean([np.array(res["runs"][k])[sel].std() / np.sqrt(sel.sum()) for k in keys])
            r[g] = [float(ys.mean()), float(np.sqrt(ys.var() / len(ys) + ep ** 2))]
        ps = [json.loads(k) if k != "none" else {} for k in keys]
        r["has_step_patch"] = float(np.mean(["CFG.max_steps" in p for p in ps]))
        r["n_patches"] = float(np.mean([len(p) for p in ps]))
        return r

    tasks = np.array([t for t, _, _ in test])
    none = np.array(res["runs"]["none"], float)

    def paired(keys):
        """Mean over seeds of (patched - unpatched) on the same episodes, with a
        standard error clustered by task."""
        y = np.mean([np.array(res["runs"][k], float) for k in keys], 0)
        r = {}
        for g, sel in groups.items():
            d, t = y[sel] - none[sel], tasks[sel]
            sums = np.array([d[t == u].sum() for u in np.unique(t)])
            cnt = np.array([(t == u).sum() for u in np.unique(t)])
            r[g] = [float(d.mean()), float(np.sqrt(((sums - d.mean() * cnt) ** 2).sum()) / len(d))]
        return r

    out["unpatched"] = stats(["none"])
    for g, keys in sets.items():
        out[g] = stats(keys)
        out[g]["diff"] = paired(keys)
    _save(os.path.join(args.out, "heldout_breakdown.json"), out)
    for g, r in out.items():
        if g != "n_episodes":
            print(g, {k: (round(v[0], 3) if isinstance(v, list) else v if isinstance(v, dict) else round(v, 2)) for k, v in r.items()})
    print(out["n_episodes"])


if __name__ == "__main__":
    main()
