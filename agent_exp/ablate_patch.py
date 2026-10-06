"""Leave-one-out check of an applied patch set: which accepted patch causes a
held-out regression? Usage:
  python -m agent_exp.ablate_patch runs/real_s1/verify_CARVE-full-only_60.json --seed 1
"""
import argparse
import json
import random

from agent_exp.agent import components_with, run_agent
from agent_exp.env import ShopDB, check, make_tasks
from agent_exp.llm import LLM
from agent_exp.pipeline import VARIANTS, _pmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verify")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=120)
    args = ap.parse_args()
    res = json.load(open(args.verify))
    llm, db = LLM(), ShopDB(args.seed)
    test = make_tasks(db, args.n_test, seed=args.seed + 99)
    rng = random.Random(args.seed + 2)
    variants = [rng.choice(VARIANTS) for _ in test]
    out = {}
    for drop in [None] + list(res["apply"]):
        apply = {k: v for k, v in res["apply"].items() if k != drop}

        def one(n):
            f = [variants[n]] if variants[n] else []
            tr = run_agent(llm, db, test[n], components_with(f, apply), f)
            return int(check(tr.answer, test[n].answer)), tr.answer

        r = _pmap(one, range(len(test)), 24)
        out[str(drop)] = {"success": sum(x[0] for x in r) / len(test),
                          "answers": [x[1] for x in r]}
        print("drop", drop, out[str(drop)]["success"], flush=True)
    json.dump(out, open(args.verify.replace(".json", "_ablate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
