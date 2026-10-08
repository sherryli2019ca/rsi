"""The main set of failed tau2 episodes for the attribution study.

Pool: every failed trial (reward < 1, no harness error, at least one step) of
every incumbent of an RRSI run, from its evolve evaluation (runs/rrsi/<domain>/
jobs/<job>) and its held-out deployment (runs/verify/<domain>/jobs/heldout/
<commit>), minus the pilot's failures. Sample: at most CAP trials per
(task, harness); tasks in a seeded order, round-robin, one trial per task per
pass, until n trials.

  python -m attrib.pool --domain tau2_retail --runs <root with runs/> \
      --exclude <pilot failures.json> --n 100 --out <failures.json>
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import _incumbents  # noqa: E402

CAP = 2


def pool(domain: str, root: Path) -> list[dict]:
    rows = []
    wt = root / "runs" / "verify" / domain / "wt"
    for commit, job in _incumbents(root / "runs" / "rrsi" / domain).items():
        srcs = [("evolve", root / "runs" / "rrsi" / domain / "jobs" / job),
                ("heldout", root / "runs" / "verify" / domain / "jobs" / "heldout" / commit)]
        for src, d in srcs:
            for p in sorted(d.glob("s[0-9]*/*.json")):
                r = json.loads(p.read_text())
                if r.get("reward", 0) < 1 and "harness_error" not in r and r.get("steps"):
                    rows.append({"commit": commit, "source": src, "task_id": str(r["task_id"]),
                                 "trial": p.parent.name, "trace": str(p), "n_steps": r["n_steps"],
                                 "harness": str(wt / commit / "domains" / domain / "harness"),
                                 "fid": f"{commit}_{src[0]}_{r['task_id']}_{p.parent.name}"})
    return rows


def sample(rows: list[dict], n: int, seed: int, exclude: set) -> list[dict]:
    rng = random.Random(seed)
    rows = [r for r in rows if r["trace"] not in exclude]
    rng.shuffle(rows)
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    out, per_combo = [], {}
    while len(out) < n and any(by_task.values()):
        for t in tasks:
            while by_task[t]:
                r = by_task[t].pop()
                key = (t, r["commit"])
                if per_combo.get(key, 0) < CAP:
                    per_combo[key] = per_combo.get(key, 0) + 1
                    out.append(r)
                    break
            if len(out) >= n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline"))
    ap.add_argument("--runs", required=True, help="the run root (holds runs/rrsi and runs/verify)")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exclude = {f["trace"] for p in args.exclude for f in json.loads(Path(p).read_text())}
    rows = pool(args.domain, Path(args.runs))
    out = sample(rows, args.n, args.seed, exclude)
    for r in out:
        if not Path(r["harness"]).is_dir():
            raise SystemExit(f"no harness worktree at {r['harness']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    tasks = {r["task_id"] for r in out}
    print(f"{args.domain}: pool {len(rows)}, sampled {len(out)} over {len(tasks)} tasks "
          f"({sum(r['source'] == 'heldout' for r in out)} held-out)")


if __name__ == "__main__":
    main()
