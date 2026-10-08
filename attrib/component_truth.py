"""Component ground truth for failed evolve trials of E1 (attrib/PREREGISTRATION.md).

E1's targeted replays (verify/replays.py) re-ran each round's reference failures
(one failed trial per task of the incumbent) 4 times with every candidate of
the round and 4 times with the incumbent (null). A candidate fixes a reference
failure if its replays succeed at least 2 more times than the null replays;
the failure's component truth is the set of components of the candidates that
fix it, each by its diff (rrsi.components.classify_diff with the tau2 signals,
which label code edits by the files and functions they touch).

  python -m attrib.component_truth --run-root /home/user/e1 --domain tau2_retail --out <json>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N, MARGIN = 4, 2


def _wins(d: Path, task: str) -> int | None:
    rs = [d / f"{task}__{j}.json" for j in range(N)]
    if not all(p.exists() for p in rs):
        return None
    return sum(int(json.loads(p.read_text()).get("reward", 0) >= 1) for p in rs)


def truth(run_root: Path, domain: str) -> dict:
    from domains.tau2.common import COMPONENT_SIGNALS
    from rrsi.components import classify_diff
    from verify.state import rounds
    out = {}
    for rnd in rounds(run_root / "runs" / "rrsi" / domain):
        rdir = run_root / "runs" / "verify" / domain / "replays" / f"r{rnd.t}"
        if not (rdir / "refs.json").exists():
            continue
        for ref in json.loads((rdir / "refs.json").read_text()):
            task = str(ref["task_id"])
            null = _wins(rdir / "null", task)
            if null is None:
                continue
            fixes = []
            for c in rnd.cands:
                if not c.commit:
                    continue
                w = _wins(rdir / c.variant, task)
                if w is None or w - null < MARGIN:
                    continue
                diff = subprocess.run(["git", "-C", str(ROOT), "diff", rnd.inc_commit, c.commit, "--",
                                       f"domains/{domain}/harness"], capture_output=True, text=True).stdout
                fixes.append({"round": rnd.t, "variant": c.variant, "commit": c.commit,
                              "component": classify_diff(diff, COMPONENT_SIGNALS), "wins": w, "null": null})
            out[str(Path(ref["trace"]).resolve())] = {"task_id": task, "round": rnd.t, "fixes": fixes,
                                                      "components": sorted({f["component"] for f in fixes})}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = truth(Path(args.run_root), args.domain)
    Path(args.out).write_text(json.dumps(res, indent=1))
    n_fix = sum(bool(v["fixes"]) for v in res.values())
    print(f"{args.domain}: {len(res)} reference failures, {n_fix} fixed by some candidate")


if __name__ == "__main__":
    main()
