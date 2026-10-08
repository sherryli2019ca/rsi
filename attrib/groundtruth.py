"""Ground truth for failure attribution by counterfactual replay with null controls.

For a failed tau2 episode with agent steps 0..n-1, step k is tested as follows.
An oracle (deepseek-v4-pro, thinking off) sees the domain policy, the tools,
the task's grading criteria (gold actions and the information the agent had to
say) and the conversation up to step k, and judges whether the agent's action
at k is a mistake; if so it gives the action a careful agent could have taken
at k with what it knew then. The episode is then replayed from k with that
action forced (corrected replays) and from k without it (null replays, the
agent acting again), n times each, with the harness that produced the episode
(domains/tau2/driver.py, "force"). Step k flips if the corrected replays
succeed at least FLIP more times than the null replays; the decisive step is
the earliest step that flips.

  python -m attrib.groundtruth run --domain tau2_retail --runs <rrsi runs> \
      --wt <dir of harness worktrees by commit> --out <dir> --n-fail 10 --rep a

Stages are resume-safe (existing files are kept): select failures (once per
domain), oracle calls, replays, summary (<out>/<domain>/<rep>/result.json).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_REPLAYS = 4          # corrected and null replays per tested step
FLIP = 2               # corrected successes minus null successes needed to flip
RESULT_CAP = 1200      # characters of a tool result shown to the oracle
ORACLE_MODEL = "deepseek-v4-pro"

ORACLE_SYSTEM = """You construct ground truth for failure attribution. A customer-service agent \
failed a task of tau2-bench. You see the domain policy, the agent's tools, the task's grading \
criteria (the database must end as if exactly the gold actions were executed, and the agent \
must tell the user every required piece of information), and the conversation up to one step \
of the agent.

Decide whether the agent's action at that step is a mistake: an action a careful agent \
following the policy would not take on the way to the gold outcome. Actions that are correct, \
or harmless and do not lead away from the gold outcome, are ok.

If it is a mistake, give the single action a careful agent would take at that step instead: \
one assistant turn, either a message to the user or one or more tool calls. Correct only this \
step; do not do more than a careful agent would do at this point. The corrected action must be \
one the agent could have taken with what it knew then (the conversation and tool results so \
far, and the policy); use the grading criteria only to know what the right outcome is, never \
to supply ids, amounts or facts the agent had not yet learned.

Answer with JSON:
{"verdict": "ok" | "mistake", "why": "<one sentence>",
 "action": {"message": "<text to the user>"} | {"tool_calls": [{"name": "<tool>", "arguments": {...}}]}}
("action" only when the verdict is mistake.)"""


# ------------------------------------------------------------------ select --
def _incumbents(run_dir: Path):
    from verify.state import rounds
    seen = {}
    for r in rounds(run_dir):
        seen.setdefault(r.inc_commit, r.inc_job)
    return seen


def select(domain: str, runs: Path, wt: Path, n: int, seed: int) -> list[dict]:
    """n failed evolve trials of the run's incumbents, one per task, spread over
    incumbents in turn (seeded)."""
    run_dir = runs / domain
    incs = _incumbents(run_dir)
    pools = {}
    for commit, job in incs.items():
        rows = []
        for p in sorted((run_dir / "jobs" / job).glob("s[0-9]*/*.json")):
            r = json.loads(p.read_text())
            if r.get("reward", 0) < 1 and "harness_error" not in r and r.get("steps"):
                rows.append({"commit": commit, "job": job, "task_id": str(r["task_id"]),
                             "trial": p.parent.name, "trace": str(p), "n_steps": r["n_steps"]})
        random.Random(f"{seed}:{commit}").shuffle(rows)
        pools[commit] = rows
    out, used = [], set()
    order = sorted(pools)
    while len(out) < n and any(pools.values()):
        for c in order:
            while pools[c]:
                row = pools[c].pop()
                if row["task_id"] not in used:
                    used.add(row["task_id"])
                    harness = wt / c / "domains" / domain / "harness"
                    if not harness.is_dir():
                        raise SystemExit(f"no harness worktree for {c} at {harness}")
                    row["harness"] = str(harness)
                    row["fid"] = f"{c}_{row['task_id']}_{row['trial']}"
                    out.append(row)
                    break
            if len(out) >= n:
                break
    return out


# ------------------------------------------------------------------ oracle --
def _render_step(st: dict) -> str:
    lines = []
    for b in st["assistant"]:
        if b["type"] == "text":
            lines.append(f"  agent says: {b['text']}")
        else:
            lines.append(f"  agent calls {b['name']}({json.dumps(b['input'], ensure_ascii=False)})")
    for r in st.get("tool_results") or []:
        c = str(r["content"])
        lines.append("  tool result: " + (c if len(c) <= RESULT_CAP else c[:RESULT_CAP] + " ...[truncated]"))
    if st.get("user") is not None:
        lines.append(f"  user replies: {st['user']}")
    return "\n".join(lines)


def _context(m) -> str:
    from agent_exp import tau2_env  # noqa: F401  (m is that module)
    pol = m.system_prompt(m.BASE_COMPONENTS)
    tools = "\n".join(f"- {t['name']}({', '.join(t['input_schema'].get('properties', {}))}): "
                      f"{t['description'].strip().splitlines()[0][:300]}"
                      for t in m.tool_defs(m.BASE_COMPONENTS))
    return f"=== DOMAIN POLICY ===\n{pol}\n\n=== TOOLS ===\n{tools}"


def oracle_prompt(rec: dict, k: int) -> str:
    gold = "\n".join(f"- {a['name']}({json.dumps(a['arguments'], ensure_ascii=False)})"
                     for a in rec.get("gold_actions") or []) or "(none: the database must stay unchanged)"
    info = "\n".join(f"- {x}" for x in rec.get("communicate_info") or []) or "(none)"
    conv = [f"user (opening): {rec['opening']}"]
    for st in rec["steps"][:k]:
        conv.append(f"step {st['index']}:\n{_render_step(st)}")
    target = rec["steps"][k]
    act = "\n".join(f"  agent says: {b['text']}" if b["type"] == "text" else
                    f"  agent calls {b['name']}({json.dumps(b['input'], ensure_ascii=False)})"
                    for b in target["assistant"])
    return (f"=== GRADING CRITERIA ===\nGold actions:\n{gold}\nInformation the agent must say:\n{info}\n"
            f"Outcome of this episode: database matches gold = {rec.get('db_match')}, "
            f"missing information = {rec.get('missing_info')}\n\n"
            f"=== CONVERSATION BEFORE STEP {k} ===\n" + "\n".join(conv) +
            f"\n\n=== AGENT ACTION AT STEP {k} (judge this) ===\n{act}\n")


def to_blocks(action: dict, tool_names: set) -> list | None:
    if not isinstance(action, dict):
        return None
    if isinstance(action.get("message"), str) and action["message"].strip():
        return [{"type": "text", "text": action["message"]}]
    calls = action.get("tool_calls")
    if isinstance(calls, list) and calls:
        out = []
        for i, c in enumerate(calls):
            if not isinstance(c, dict) or c.get("name") not in tool_names:
                return None
            args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
            out.append({"type": "tool_use", "id": f"fix_{i}", "name": c["name"], "input": args})
        return out
    return None


def run_oracle(fails: list, odir: Path, m, workers: int = 8) -> None:
    from rrsi.llm import generate
    ctx = _context(m)
    names = {t["name"] for t in m.tool_defs(m.BASE_COMPONENTS)}
    jobs = []
    for f in fails:
        rec = json.loads(Path(f["trace"]).read_text())
        for k in range(rec["n_steps"]):
            p = odir / f"{f['fid']}_k{k}.json"
            if not p.exists():
                jobs.append((p, rec, k))

    def one(job):
        p, rec, k = job
        raw = generate(oracle_prompt(rec, k), system=ORACLE_SYSTEM, json_only=True,
                       model=ORACLE_MODEL, cache_prefix=ctx, role="oracle")
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            v = {"verdict": "unparseable", "raw": raw[:2000]}
        if not isinstance(v, dict):
            v = {"verdict": "unparseable", "raw": str(v)[:2000]}
        if v.get("verdict") == "mistake":
            v["force"] = to_blocks(v.get("action"), names)
            if v["force"] is None:
                v["verdict"] = "invalid_action"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v, ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


# ----------------------------------------------------------------- replays --
def run_replays(fails: list, odir: Path, rdir: Path, dom: str, workers: int) -> None:
    by_harness = {}
    for f in fails:
        rec_n = json.loads(Path(f["trace"]).read_text())["n_steps"]
        for k in range(rec_n):
            v = json.loads((odir / f"{f['fid']}_k{k}.json").read_text())
            if v.get("verdict") != "mistake":
                continue
            for j in range(N_REPLAYS):
                base = {"task_id": f["task_id"], "trace": f["trace"], "start": k}
                by_harness.setdefault(f["harness"], []).extend([
                    {**base, "key": f"{f['fid']}_k{k}_c{j}", "force": v["force"]},
                    {**base, "key": f"{f['fid']}_k{k}_n{j}"}])
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for harness, refs in by_harness.items():
        todo = [r for r in refs if not (rdir / f"{r['key']}.json").exists()]
        if not todo:
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        rf = rdir / f"refs_{abs(hash(harness)) % 10**8}.json"
        rf.write_text(json.dumps(todo))
        subprocess.run([sys.executable, "-m", "domains.tau2.driver", "--harness", harness,
                        "--domain", dom, "--out", str(rdir), "--replay", str(rf),
                        "--workers", str(workers), "--temperature", "0"],
                       cwd=ROOT, env=env, check=False)


# ----------------------------------------------------------------- summary --
def summarize(fails: list, odir: Path, rdir: Path) -> dict:
    out = []
    for f in fails:
        n = json.loads(Path(f["trace"]).read_text())["n_steps"]
        steps = []
        for k in range(n):
            v = json.loads((odir / f"{f['fid']}_k{k}.json").read_text())
            row = {"k": k, "verdict": v.get("verdict")}
            if v.get("verdict") == "mistake":
                c = [json.loads((rdir / f"{f['fid']}_k{k}_c{j}.json").read_text()) for j in range(N_REPLAYS)
                     if (rdir / f"{f['fid']}_k{k}_c{j}.json").exists()]
                nl = [json.loads((rdir / f"{f['fid']}_k{k}_n{j}.json").read_text()) for j in range(N_REPLAYS)
                      if (rdir / f"{f['fid']}_k{k}_n{j}.json").exists()]
                ok_c = [x for x in c if (x.get("replay") or {}).get("forced")]
                row.update({"corrected": sum(x.get("reward", 0) for x in ok_c), "n_corrected": len(ok_c),
                            "null": sum(x.get("reward", 0) for x in nl), "n_null": len(nl),
                            "harness_errors": sum("harness_error" in x for x in c + nl)})
                row["flip"] = (len(ok_c) == N_REPLAYS and len(nl) == N_REPLAYS
                               and row["corrected"] - row["null"] >= FLIP)
            steps.append(row)
        flips = [s["k"] for s in steps if s.get("flip")]
        mistakes = [s["k"] for s in steps if s["verdict"] == "mistake"]
        out.append({**{k: f[k] for k in ("fid", "commit", "task_id", "trial", "n_steps")},
                    "decisive": flips[0] if flips else None, "flips": flips,
                    "first_mistake": mistakes[0] if mistakes else None, "mistakes": mistakes,
                    "steps": steps})
    return {"n_replays": N_REPLAYS, "flip": FLIP, "failures": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-fail", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", default="a")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dom = args.domain.split("_", 1)[1]
    os.environ["TAU2_DOMAIN"] = dom
    from agent_exp import tau2_env as m
    base = Path(args.out) / args.domain
    base.mkdir(parents=True, exist_ok=True)
    fp = base / "failures.json"
    if not fp.exists():
        fp.write_text(json.dumps(select(args.domain, Path(args.runs), Path(args.wt), args.n_fail, args.seed),
                                 indent=1))
    fails = json.loads(fp.read_text())
    odir, rdir = base / args.rep / "oracle", base / args.rep / "replays"
    (base / args.rep).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(base / args.rep / "usage.jsonl"))
    run_oracle(fails, odir, m, args.workers)
    run_replays(fails, odir, rdir, dom, args.workers)
    res = summarize(fails, odir, rdir)
    (base / args.rep / "result.json").write_text(json.dumps(res, indent=1))
    for x in res["failures"]:
        print(f"{x['fid']:28s} steps {x['n_steps']:2d} first mistake {x['first_mistake']} "
              f"decisive {x['decisive']} flips {x['flips']}")


if __name__ == "__main__":
    main()
