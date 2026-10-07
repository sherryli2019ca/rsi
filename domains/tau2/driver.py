"""Frozen episode driver for the tau2-bench RRSI domains (tau2_retail, tau2_airline).

Runs an evolvable harness (domains/tau2_<d>/harness in some worktree) against
tau2-bench's own environment, tools, policy, tasks and user-simulator
guidelines, with the reward used throughout this project: the final database
must equal the one the task's gold actions produce and every communicate_info
string must have been said (agent_exp/tau2_env.py). The harness sees only the
contract documented in domains/tau2/base_harness/agent.py.

Two modes, both resume-safe (an existing output file is never re-run):

  fresh   every task in --ids x --k trials, written to <out>/s<trial>/<task>.json
  replay  every reference in --replay (a JSON list of {"key", "task_id",
          "trace", "start"}): the recorded episode `trace` (a trial file of this
          driver or a trace of agent_exp) is re-executed up to step `start` and
          continued live with the harness, written to <out>/<key>.json

Replay with code-level harness changes. Recorded agent actions in the prefix
are kept and their tool calls re-executed on a fresh database, but every call
goes through the NEW harness's before_tool / render_result. If the new harness
blocks a recorded call or renders a different observation at step i, the
recorded actions after step i were conditioned on something this harness would
not have shown, so the episode goes live from step i + 1 instead of `start`
(recorded as start_effective, with the steps that diverged). Changes to the
system prompt or tool descriptions do not move the start: as in all replays of
this project, they act from the replayed step on. observe_prefix lets the
harness rebuild per-episode state from the recorded steps.

  python -m domains.tau2.driver --harness <dir> --domain retail --out <dir> \
      --ids 0,1,2 --k 2 [--replay refs.json] [--workers 8] [--temperature 0]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAX_STEPS = 40          # agent steps per episode, as in every run of this project


class HarnessError(Exception):
    pass


def load_agent_class(harness_dir: Path):
    """Import the harness package found at harness_dir under a unique name."""
    harness_dir = Path(harness_dir).resolve()
    name = "harness_" + hashlib.md5(str(harness_dir).encode()).hexdigest()[:10]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, harness_dir / "__init__.py", submodule_search_locations=[str(harness_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(name + ".agent").Agent


class PolicyModel:
    """The frozen policy model handed to the harness as `llm`, and the user
    simulator. Model, temperature and thinking settings are fixed here."""

    def __init__(self, temperature=None):
        from agent_exp import llm as L
        self.L = L
        self.llm = L.LLM(temperature=temperature)

    def make(self):
        acc = {"in": 0, "out": 0, "cache_read": 0, "calls": 0}
        L, llm = self.L, self.llm

        def call(system, tools, messages, max_tokens=4000):
            kw = dict(model=L.AGENT_MODEL, max_tokens=int(max_tokens), system=system,
                      messages=messages, **llm.temp)
            if L.BACKEND == "deepseek":
                kw["thinking"] = {"type": "disabled"}
            if tools:
                kw["tools"] = tools
            resp = llm._create(**kw)
            u = resp.usage
            acc["in"] += u.input_tokens
            acc["out"] += u.output_tokens
            acc["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            acc["calls"] += 1
            blocks = []
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.id, "name": b.name,
                                   "input": b.input})
            return blocks, {"in": u.input_tokens, "out": u.output_tokens}
        return call, acc


def _normalize(blocks):
    """Validate what the harness returned; anything malformed is the harness's fault."""
    if not isinstance(blocks, list):
        raise HarnessError(f"next_action returned {type(blocks).__name__}, not a list")
    out, seen = [], set()
    for i, b in enumerate(blocks):
        if not isinstance(b, dict) or b.get("type") not in ("text", "tool_use"):
            raise HarnessError(f"bad content block {str(b)[:200]}")
        if b["type"] == "text":
            if str(b.get("text", "")).strip():
                out.append({"type": "text", "text": str(b["text"])})
        else:
            bid = str(b.get("id") or f"call_{i}")
            while bid in seen:
                bid += "_"
            seen.add(bid)
            out.append({"type": "tool_use", "id": bid, "name": str(b.get("name")),
                        "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
    return out or [{"type": "text", "text": "..."}]


def _h(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001 - a harness bug fails the episode
        raise HarnessError(f"{getattr(fn, '__name__', fn)}: {e!r}\n{traceback.format_exc()[-1500:]}")


def episode(m, Agent, pm: PolicyModel, task, prefix: dict | None = None, start: int = 0) -> dict:
    t0 = time.time()
    env = m.get_environment()
    call, acc = pm.make()
    tools = m.tool_defs(m.BASE_COMPONENTS)
    policy = open(m.POLICY_PATH).read()
    agent = _h(Agent, call, policy, copy.deepcopy(tools))
    user_sys = m._guidelines() + f"\n\n<scenario>\n{m.scenario_text(task)}\n</scenario>"
    umsgs = [{"role": "user", "content": m.GREETING}]
    pm.llm.meter(reset=True)
    opening = prefix.get("opening") if prefix else None
    if not opening:
        opening = pm.llm.chat(user_sys, umsgs, purpose="user")
    umsgs.append({"role": "assistant", "content": opening})
    msgs = [{"role": "user", "content": "(conversation start)"},
            {"role": "assistant", "content": m.GREETING},
            {"role": "user", "content": opening}]
    steps, texts, stopped = [], [], ""
    live_from = start if prefix else 0
    diverged = []

    def say(text, reply):
        texts.append(text)
        umsgs.append({"role": "user", "content": text or "..."})
        umsgs.append({"role": "assistant", "content": reply})
        msgs.append({"role": "user", "content": reply})

    # ---- recorded prefix ----------------------------------------------------
    for i, st in enumerate((prefix or {}).get("steps", [])[:start]):
        if i >= live_from:
            break
        blocks = copy.deepcopy(st["assistant"])
        _h(agent.observe_prefix, copy.deepcopy(msgs), copy.deepcopy(blocks))
        msgs.append({"role": "assistant", "content": blocks})
        rec = {"index": i, "assistant": blocks, "tool_results": [], "raw": [],
               "user": None, "prefix": True}
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if uses:
            old = {r["tool_use_id"]: str(r["content"]) for r in st.get("tool_results") or []}
            for b in uses:
                g = _h(agent.before_tool, copy.deepcopy(b))
                raw = None if g is not None else m._call(env, b["name"], b["input"])
                obs = str(g) if g is not None else str(_h(agent.render_result, copy.deepcopy(b), raw))
                if g is not None or obs != old.get(b["id"]):
                    diverged.append({"step": i, "call": b["name"],
                                     "kind": "blocked" if g is not None else "rendered"})
                    live_from = min(live_from, i + 1)
                rec["tool_results"].append({"type": "tool_result", "tool_use_id": b["id"],
                                            "content": obs})
                rec["raw"].append(raw)
            msgs.append({"role": "user", "content": rec["tool_results"]})
        else:
            text = " ".join(b["text"] for b in blocks if b["type"] == "text")
            rec["user"] = st.get("user") or "..."
            say(text, rec["user"])
        steps.append(rec)
        if rec["user"] is not None and any(s in rec["user"] for s in m.STOP_TOKENS):
            stopped = "user_stop"
            break

    # ---- live ------------------------------------------------------------------
    while not stopped and len(steps) < MAX_STEPS:
        blocks = _normalize(_h(agent.next_action, copy.deepcopy(msgs)))
        msgs.append({"role": "assistant", "content": blocks})
        rec = {"index": len(steps), "assistant": blocks, "tool_results": [], "raw": [],
               "user": None}
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if uses:
            for b in uses:
                g = _h(agent.before_tool, copy.deepcopy(b))
                raw = None if g is not None else m._call(env, b["name"], b["input"])
                obs = str(g) if g is not None else str(_h(agent.render_result, copy.deepcopy(b), raw))
                rec["tool_results"].append({"type": "tool_result", "tool_use_id": b["id"],
                                            "content": obs})
                rec["raw"].append(raw)
            msgs.append({"role": "user", "content": rec["tool_results"]})
            steps.append(rec)
            continue
        text = " ".join(b["text"] for b in blocks if b["type"] == "text")
        umsgs_view = umsgs + [{"role": "user", "content": text or "..."}]
        reply = pm.llm.chat(user_sys, umsgs_view, purpose="user") or "..."
        rec["user"] = reply
        say(text, reply)
        steps.append(rec)
        if any(s in reply for s in m.STOP_TOKENS):
            stopped = "user_stop"
    if not stopped:
        stopped = "max_steps"

    # ---- grade -----------------------------------------------------------------
    db_match = env.get_db_hash() == m.gold_hash(task)
    said = " ".join(texts).lower().replace(",", "")
    infos = [str(x) for x in (task.evaluation_criteria.communicate_info or [])]
    missing = [x for x in infos if x.lower().replace(",", "") not in said]
    user_tok = pm.llm.meter(reset=True)
    gold = [{"name": a.name, "arguments": a.arguments}
            for a in (task.evaluation_criteria.actions or [])]
    return {"task_id": task.id, "reward": int(db_match and not missing), "db_match": db_match,
            "missing_info": missing, "communicate_info": infos, "gold_actions": gold,
            "stopped": stopped, "opening": opening, "steps": steps, "n_steps": len(steps),
            "tokens": {"agent_in": acc["in"], "agent_out": acc["out"],
                       "agent_cache_read": acc["cache_read"],
                       "agent_total": acc["in"] + acc["out"] + acc["cache_read"],
                       "user": user_tok},
            "calls": {"agent": acc["calls"]}, "seconds": round(time.time() - t0, 2),
            "replay": ({"start_requested": start, "start_effective": live_from,
                        "diverged": diverged} if prefix else None)}


def run_jobs(m, Agent, pm, jobs, workers):
    """jobs: [(out_path, task, prefix, start)]; resume-safe; infra errors retried."""
    def one(job):
        out, task, prefix, start = job
        if out.exists():
            return "skip"
        err = ""
        for attempt in range(3):
            try:
                rec = episode(m, Agent, pm, task, prefix, start)
            except HarnessError as e:
                rec = {"task_id": task.id, "reward": 0, "harness_error": str(e)[:3000],
                       "steps": [], "n_steps": 0, "tokens": {}, "stopped": "harness_error"}
            except Exception as e:  # noqa: BLE001 - API or environment failure: retry
                err = f"{e!r}"[:500]
                time.sleep(5 * (attempt + 1))
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, default=str))
            os.replace(tmp, out)
            return "ok" if "harness_error" not in rec else "harness_error"
        return "infra: " + err
    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--domain", required=True, choices=["retail", "airline"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()
    os.environ["TAU2_DOMAIN"] = args.domain
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    from agent_exp import tau2_env as m
    tasks = {t.id: t for t in m.get_tasks("base")}
    Agent = load_agent_class(Path(args.harness))
    pm = PolicyModel(args.temperature)
    out = Path(args.out)
    if args.replay:
        refs = json.loads(Path(args.replay).read_text())
        jobs = []
        for r in refs:
            tr = r["trace"] if isinstance(r["trace"], dict) else json.loads(Path(r["trace"]).read_text())
            jobs.append((out / f"{r['key']}.json", tasks[str(r["task_id"])], tr, int(r["start"])))
    else:
        ids = [x for x in args.ids.split(",") if x] if not os.path.exists(args.ids) else \
            json.loads(Path(args.ids).read_text())
        jobs = [(out / f"s{s}" / f"{tid}.json", tasks[str(tid)], None, 0)
                for s in range(args.k) for tid in ids]
    res = run_jobs(m, Agent, pm, jobs, args.workers)
    summary = {s: res.count(s) for s in set(res)}
    print(json.dumps({"jobs": len(jobs), **summary}), flush=True)


if __name__ == "__main__":
    main()
