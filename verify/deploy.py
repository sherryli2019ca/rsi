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
