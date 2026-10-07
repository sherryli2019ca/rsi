"""LLM-judge evidence: predict a candidate's effect without running it.

The judge (the search model, thinking off) reads the candidate's diff and
declared edits, the incumbent's failed trials of the round (the same traces
the proposer saw, compact rendering) and one line per solved task, and
predicts for each failed task the probability that the change makes it pass,
plus the expected number of currently solved evolve trials it would break.
Estimate: dS = (sum_task n_failed * p_fix - expected_breaks) / n_trials.
Output: runs/verify/<domain>/judge/r<t><variant>.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rrsi.llm import generate

from .state import Round, evaluation

MAX_DIFF = 14000
MAX_TRACE = 5000
MAX_TRACES = 14

SYSTEM = """You review a proposed change to the harness (scaffold code and prompts) of
an LLM customer-service agent. The policy model is frozen; only the harness
changed. Predict, as a careful engineer who has to bet on it, what the change
will do when the agent is run again from the first step on the same tasks.

You get: the diff, the proposer's declared edits, the trials the CURRENT harness
FAILED (with the grading block), and one line per task it SOLVED.

For each failed task, give the probability (0 to 1) that the changed harness
solves it on a fresh run. Then estimate how many of the currently solved
trials the change would break (a non-negative number, at most the number of
solved trials). Be calibrated: most harness changes flip few tasks, and
reruns of an unchanged harness also differ by chance.

Return STRICT JSON:
{"fix": {"<task_id exactly as after FAILED TASK>": <probability>, ...},
 "expected_breaks": <number>,
 "rationale": "two or three sentences"}"""


def run(domain, run_dir: Path, out_root: Path, rnd: Round, model: str | None = None) -> dict:
    ev = evaluation(run_dir, rnd.inc_job)
    n_trials = sum(len(tr.rewards) for tr in ev.per_task.values())
    failed = {t: tr for t, tr in ev.per_task.items() if tr.rewards and min(tr.rewards) < 1}
    solved = [t for t, tr in ev.per_task.items() if tr.rewards and min(tr.rewards) >= 1]
    n_solved = sum(len(ev.per_task[t].rewards) for t in solved)
    traces = []
    for tid in sorted(failed, key=lambda t: failed[t].mean)[:MAX_TRACES]:
        tr = failed[tid]
        rec = domain.load_trial(run_dir, rnd.inc_job, tid, tr.rewards.index(min(tr.rewards)))
        if rec is not None:
            traces.append(f"--- FAILED TASK {tid} (failed {sum(r < 1 for r in tr.rewards)} of "
                          f"{len(tr.rewards)} trials) ---\n" + domain.render_trace(rec)[-MAX_TRACE:])
    rows = []
    for tid in solved:
        rec = domain.load_trial(run_dir, rnd.inc_job, tid, 0)
        if rec is not None:
            rows.append(domain.task_row(tid, rec, ev.per_task[tid]))
    out = {}
    jdir = Path(out_root) / "judge"
    jdir.mkdir(parents=True, exist_ok=True)
    for c in rnd.cands:
        if not c.measured:
            continue
        path = jdir / f"r{rnd.t}{c.variant}.json"
        if path.exists():
            out[c.variant] = json.loads(path.read_text())
            continue
        diff_p = Path(run_dir) / f"r{rnd.t}" / c.variant / "diff.patch"
        diff = diff_p.read_text()[:MAX_DIFF] if diff_p.exists() else ""
        edits = [{k: e.get(k) for k in ("id", "component", "hypothesis", "targets_mode",
                                        "trigger_condition")} for e in c.edits]
        prompt = ("=== DIFF ===\n" + diff + "\n\n=== DECLARED EDITS ===\n" +
                  json.dumps(edits, indent=1) + "\n\n=== FAILED TRIALS ===\n" +
                  "\n\n".join(traces) + f"\n\n=== SOLVED TASKS ({n_solved} trials) ===\n" +
                  "\n".join(rows))
        res = {}
        for _ in range(3):
            try:
                res = json.loads(generate(prompt, system=SYSTEM, json_only=True, model=model,
                                          role="judge"))
                break
            except Exception as e:  # noqa: BLE001 - malformed JSON or API error: retry
                res = {"error": repr(e)[:300]}
        fix = {}
        for k, v in (res.get("fix") or {}).items():
            k = re.sub(r"(?i)^\s*(failed\s+)?task\s*", "", str(k)).strip()
            if k in failed:
                try:
                    fix[k] = min(1.0, max(0.0, float(v)))
                except (TypeError, ValueError):
                    pass
        breaks = min(float(n_solved), max(0.0, float(res.get("expected_breaks") or 0.0)))
        gain = sum(sum(r < 1 for r in failed[t].rewards) * p for t, p in fix.items())
        res.update({"dS": (gain - breaks) / max(1, n_trials), "n_trials": n_trials,
                    "n_failed_tasks": len(failed), "n_solved_trials": n_solved,
                    "n_traces_shown": len(traces)})
        path.write_text(json.dumps(res, indent=1))
        out[c.variant] = res
    return out
