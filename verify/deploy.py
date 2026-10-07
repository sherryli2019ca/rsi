"""Deployment ground truth: a harness commit run from the first step on the
held-out tasks, k trials each (resume-safe; raising k later adds trials).
Output: runs/verify/<domain>/jobs/heldout/<commit12>/s<trial>/<task>.json and
eval.json next to it."""
from __future__ import annotations

from pathlib import Path

from rrsi.evaluate import aggregate

from .state import worktree


def run(domain, repo: Path, out_root: Path, commit: str, k: int):
    wt = worktree(repo, Path(out_root) / "wt", commit)
    job = f"heldout/{commit[:12]}"
    ids = domain.heldout_ids()
    domain.run(wt, out_root, job, ids, k, log_prefix=job)
    per, extra = domain.score(out_root, job, ids, k)
    ev = aggregate(job, k, per, extra)
    ev.save(Path(out_root) / "jobs" / job / "eval.json")
    return ev


def main():
    """Deploy one harness (commit or branch) on the held-out tasks, e.g. the
    final incumbents of a live comparison (verify/run_compare.sh)."""
    import argparse
    import json

    from rrsi import gitops as G
    from rrsi.domain import load_domain

    from .state import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--ref", required=True, help="commit or branch, e.g. evolve/sel_net40/tau2_retail")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "runs" / "compare" / "deploy"))
    args = ap.parse_args()
    dom = load_domain(args.domain)
    commit = G.git(ROOT, "rev-parse", args.ref).stdout.strip()
    ev = run(dom, ROOT, Path(args.out) / dom.name, commit, args.k)
    print(json.dumps({"ref": args.ref, "commit": commit[:12], "S": ev.S, "C": ev.C,
                      "trials": ev.n_expected, "missing": ev.missing}))


if __name__ == "__main__":
    main()
