"""Full rescue profiles under the oracle without grading, on a random subset of
failures (post hoc, not registered).

attrib.review4_runs blind ran the grading-blind oracle only at steps some
method named on the 157 failures the registered oracle found rescuable. Here it
runs at every step of a seeded random sample of failures drawn from all 300,
whatever their registered rescuability: 34 retail, 33 airline, 33 AppWorld
(seed SEED). Same oracle, K = 4 samples, 2 corrected replays per sample that
calls the step a mistake, the registered profile's 4 null replays where they
exist (linked) and 4 new ones elsewhere. Oracle outputs and replays share
<main>/<domain>/gt/blind/{oracle,replays} with the earlier run (identical
settings, so steps already done are reused); the full profiles go to
<main>/<domain>/gt/blind/result_full.json.

  python -m attrib.blind_full run --domain tau2_retail|tau2_airline|appworld \
      [--main /home/user/attrib_runs/main] [--workers 8] [--replay-workers N] [--limit N]
  python -m attrib.blind_full sample        # print the sample
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.review4_runs import (BLIND_SYSTEM_AW, BLIND_SYSTEM_TAU2, K, MAIN, R_C, _samples, _setup,  # noqa: E402
                                 blind_prompt_aw, blind_prompt_tau2, drive_all, run_oracle, step_profile)

SEED = 20261010
N = {"tau2_retail": 34, "tau2_airline": 33, "appworld": 33}


def sample(main: Path, domain: str) -> list[str]:
    fids = sorted(f["fid"] for f in json.loads((main / domain / "failures.json").read_text()))
    return sorted(random.Random(f"{SEED}-{domain}").sample(fids, N[domain]))


def run(args) -> None:
    main = Path(args.main)
    dom, m, ctx, force_of, drive = _setup(args.domain)
    rep = main / args.domain / "gt" / "blind"
    odir, rdir = rep / "oracle", rep / "replays"
    rdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    fails = {f["fid"]: f for f in json.loads((main / args.domain / "failures.json").read_text())}
    fids = sample(main, args.domain)
    if args.limit:
        fids = fids[:args.limit]
    recs = {f: json.loads(Path(fails[f]["trace"]).read_text()) for f in fids}
    steps = {f: list(range(recs[f]["n_steps"])) for f in fids}
    system, mk = (BLIND_SYSTEM_AW, blind_prompt_aw) if dom == "appworld" else (BLIND_SYSTEM_TAU2, blind_prompt_tau2)
    jobs = [(f, recs[f], k) for f in fids for k in steps[f]]
    print(f"{args.domain}: {len(fids)} failures, {len(jobs)} steps", flush=True)
    run_oracle(jobs, odir, system, mk, ctx, force_of, args.workers)
    reg = main / args.domain / "gt" / "a" / "replays"
    refs = []
    for f in fids:
        base = {"task_id": fails[f]["task_id"], "trace": fails[f]["trace"]}
        for k in steps[f]:
            sm = _samples(odir, f, k)
            mist = [i for i, x in enumerate(sm) if x.get("verdict") == "mistake"]
            for i in mist:
                refs += [(fails[f]["harness"], {**base, "start": k, "force": sm[i]["force"],
                                                "key": f"{f}_k{k}_o{i}_{j}"}) for j in range(R_C)]
            if mist:
                for j in range(4):
                    a, b = reg / f"{f}_k{k}_n{j}.json", rdir / f"{f}_k{k}_n{j}.json"
                    if a.exists() and not b.exists():
                        b.symlink_to(a.resolve())
                    elif not b.exists():
                        refs.append((fails[f]["harness"], {**base, "start": k, "key": f"{f}_k{k}_n{j}"}))
    todo = [x for x in refs if not (rdir / f"{x[1]['key']}.json").exists()]
    print(f"{args.domain}: {len(todo)} replays to run", flush=True)
    missing = drive_all(todo, rdir, dom, drive, args.replay_workers or args.workers) if todo else 0
    res = {"K": K, "R_C": R_C, "seed": SEED, "missing_replays": missing, "sample": fids,
           "failures": {f: [step_profile(f, k, odir, rdir, 4) for k in steps[f]] for f in fids}}
    (rep / "result_full.json").write_text(json.dumps(res, indent=1))
    print(f"{args.domain}: done, {missing} replays missing", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "sample"))
    ap.add_argument("--domain", choices=("tau2_retail", "tau2_airline", "appworld"))
    ap.add_argument("--main", default=str(MAIN))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "sample":
        for d in N:
            print(d, sample(Path(a.main), d))
        return
    run(a)


if __name__ == "__main__":
    main()
