"""Targeted mid-step replays of a round's failed evolve trials.

Replay set of round t: for every evolve task the incumbent failed in RRSI's
full evaluation, the trial RRSI itself digested (its worst trial), up to
MAX_TRIALS tasks. Start step: the earliest step the round's failure digest
cites; without a digest, the first state-changing call that is not a gold
action; otherwise 0 (a rerun from the first step with the recorded opening).
Each reference is replayed R_CAND times with each candidate harness and R_NULL
times with the incumbent harness (null replays). A code-level change may move the
effective start earlier (the driver records it).

Output: runs/verify/<domain>/replays/r<t>/<label>/<task>__<j>.json with
label = the candidate variant or "null"; refs in .../r<t>/refs.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .state import Round, digest_step, evaluation, first_bad_write, trial_path, worktree

MAX_TRIALS = 20
R_CAND = 4          # replays per reference with each candidate (replay@80 saturates at 4 per reference)
R_NULL = 4          # null replays per reference (shared by the round's candidates)


def references(run_dir: Path, rnd: Round, seed: int = 0) -> list[dict]:
    ev = evaluation(run_dir, rnd.inc_job)
    refs = []
    for tid, tr in ev.per_task.items():
        if not tr.rewards or min(tr.rewards) >= 1:
            continue
        s = tr.rewards.index(min(tr.rewards))
        p = trial_path(run_dir, rnd.inc_job, tid, s)
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        if rec.get("harness_error") or not rec.get("steps"):
            continue
        start, source = digest_step(Path(run_dir) / f"r{rnd.t}" / "analysis" / "digests"
                                    / f"{tid}_failure.json"), "digest"
        if start is None:
            start, source = first_bad_write(rec), "first_bad_write"
        if start is None:
            start, source = 0, "start"
        start = max(0, min(int(start), len(rec["steps"]) - 1))
        refs.append({"task_id": tid, "trial": s, "trace": str(p), "start": start,
                     "start_source": source, "n_failed": sum(1 for r in tr.rewards if r < 1),
                     "k": len(tr.rewards)})
    refs.sort(key=lambda r: r["task_id"])
    if len(refs) > MAX_TRIALS:
        refs = sorted(random.Random(seed + rnd.t).sample(refs, MAX_TRIALS),
                      key=lambda r: r["task_id"])
    return refs


def run(domain, repo: Path, run_dir: Path, out_root: Path, rnd: Round) -> dict:
    """Null replays with the incumbent and replays with every measured
    candidate; resume-safe. Returns {label: dir}."""
    rdir = Path(out_root) / "replays" / f"r{rnd.t}"
    rdir.mkdir(parents=True, exist_ok=True)
    rp = rdir / "refs.json"
    if rp.exists():
        refs = json.loads(rp.read_text())
    else:
        refs = references(run_dir, rnd)
        rp.write_text(json.dumps(refs, indent=1))
    jobs = [("null", rnd.inc_commit)] + [(c.variant, c.commit) for c in rnd.cands if c.measured]
    out = {}
    for label, commit in jobs:
        wt = worktree(repo, Path(out_root) / "wt", commit)
        n = R_NULL if label == "null" else R_CAND
        items = [{"key": f"{r['task_id']}__{j}", "task_id": r["task_id"], "trace": r["trace"],
                  "start": r["start"]} for r in refs for j in range(n)]
        domain.replay(wt, out_root, f"r{rnd.t}/{label}", items)
        out[label] = Path(out_root) / "replays" / f"r{rnd.t}" / label
    return out
