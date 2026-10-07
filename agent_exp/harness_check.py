"""Harness check (P0): our replayable loop against tau2-bench's own orchestrator.

On the clean held-out episodes (every test task x 3 trials) four arms run at
temperature 0:
  ours_t0|none    our loop (agent_exp.tau2_env), unpatched
  ours_t0|apply   our loop with the Apply-attributed patch set (seed 0)
  off_t0|none     tau2's Orchestrator + LLMAgent + UserSimulator, unpatched
  off_t0|apply    the same with the patched policy sections and tool descriptions
Airline also gets ours_def|none and ours_def|apply at the provider's default
temperature (retail has these in heldout_clean.json). The step limit is the
orchestrator's own (tau2: 200 messages in total; ours: 40 agent steps, raised by
the patch set): clean episodes never come near either.

Every episode is scored three ways:
  ours      final database equals the gold one and every communicate_info string
            was said (the reward used throughout the paper)
  official  tau2's evaluator with each task's reward basis (DB, communicate,
            NL assertions), on the trajectory converted to tau2 messages
  parts     db / communicate / nl_assertions separately (NL assertions judged by
            the analyst model at temperature 0; tau2 uses GPT-4.1)

Null replays: the first 10 null replays of every category in the bank are re-run
from the same source trace and step at temperature 0.

Both models are called through DeepSeek's Anthropic-compatible endpoint with
thinking disabled (litellm 'anthropic/' provider for tau2's side).

  python -m agent_exp.harness_check --domain tau2_retail --out runs/tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid

import numpy as np

from agent_exp.bank import Domain, _load, _pmap, _save
from agent_exp.llm import LLM

ENDPOINT = "https://api.deepseek.com/anthropic"


def litellm_args(model_temp=0.0, max_tokens=4000):
    return {"temperature": model_temp, "api_base": ENDPOINT, "api_key": "proxy",
            "thinking": {"type": "disabled"}, "allowed_openai_params": ["thinking"],
            "max_tokens": max_tokens, "num_retries": 6}


def setup_tau2():
    """Point tau2's NL-assertion judge at the analyst model and make it tolerate
    fenced JSON."""
    import sys

    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    import tau2.evaluator.evaluator_nl_assertions as nl
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from tau2.utils import llm_utils

    # litellm tags function tools as type "custom", which DeepSeek's endpoint rejects
    orig = AnthropicConfig._map_tool_helper

    def map_tool(self, tool):
        t, mcp = orig(self, tool)
        if t is not None and t.get("type") == "custom":
            t.pop("type")
        return t, mcp
    AnthropicConfig._map_tool_helper = map_tool
    nl.DEFAULT_LLM_NL_ASSERTIONS = "anthropic/deepseek-v4-pro"
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = litellm_args(0.0, 8000)

    def gen(*a, **kw):
        m = llm_utils.generate(*a, **kw)
        if m.content:
            t = m.content.strip()
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
            i, j = t.find("{"), t.rfind("}")
            m.content = t[i:j + 1] if i >= 0 else t
        return m
    nl.generate = gen


def to_messages(trace, task):
    """Our trace as tau2 messages, for tau2's evaluator."""
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
    msgs = [AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
            UserMessage(role="user", content=trace["opening"])]
    for st in trace["steps"]:
        text = " ".join(b["text"] for b in st["assistant"] if b["type"] == "text").strip()
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b["input"], requestor="assistant")
                 for b in st["assistant"] if b["type"] == "tool_use"]
        msgs.append(AssistantMessage(role="assistant", content=None if calls else (text or "..."),
                                     tool_calls=calls or None))
        for r in st["tool_results"]:
            msgs.append(ToolMessage(id=r["tool_use_id"], role="tool", content=str(r["content"]),
                                    requestor="assistant",
                                    error=str(r["content"]).startswith("Error")))
        if st.get("user") is not None:
            msgs.append(UserMessage(role="user", content=st["user"]))
    return msgs


def score(sim_messages, task, domain, stopped_ok):
    """Official reward plus its parts on a list of tau2 messages."""
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    sim = SimulationRun(id=str(uuid.uuid4()), task_id=task.id, start_time=now, end_time=now,
                        duration=0.0, messages=sim_messages,
                        termination_reason=TerminationReason.USER_STOP if stopped_ok
                        else TerminationReason.MAX_STEPS)
    out = {}
    for lab, et in (("official", EvaluationType.ALL), ("db", EvaluationType.ENV),
                    ("communicate", EvaluationType.COMMUNICATE),
                    ("nl", EvaluationType.NL_ASSERTIONS)):
        try:
            r = evaluate_simulation(sim, task, et, False, domain, strict_replay=False)
            out[lab] = float(r.reward)
        except Exception as e:                      # judge or replay failure
            out[lab] = None
            out[f"{lab}_error"] = str(e)[:200]
    basis = task.evaluation_criteria.reward_basis if task.evaluation_criteria else []
    out["basis"] = [str(getattr(b, "value", b)) for b in basis or []]
    out["has_nl"] = bool(task.evaluation_criteria and task.evaluation_criteria.nl_assertions)
    return out


def run_official(D, task, patches, trial):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner import build_text_orchestrator
    dom = D.m.DOMAIN
    cfg = TextRunConfig(domain=dom, agent="llm_agent", user="user_simulator",
                        llm_agent="anthropic/deepseek-v4-flash", llm_args_agent=litellm_args(0.0, 4000),
                        llm_user="anthropic/deepseek-v4-flash", llm_args_user=litellm_args(0.0, 2000),
                        seed=300 + trial)
    orch = build_text_orchestrator(cfg, task, seed=300 + trial)
    if patches:
        comps = D.m.components_with([], patches)
        orch.agent.domain_policy = "\n\n".join(comps[k] for k in D.m.BASE_COMPONENTS
                                               if k.startswith("POL."))
        for t in orch.agent.tools:
            if f"TOOL.{t.name}" in patches:
                t.short_desc, t.long_desc = patches[f"TOOL.{t.name}"], ""
    t0 = time.time()
    sim = orch.run()
    dur = time.time() - t0
    from tau2.data_model.simulation import TerminationReason
    ok = sim.termination_reason in (TerminationReason.AGENT_STOP, TerminationReason.USER_STOP)
    res = score(sim.messages, task, dom, ok)
    tok = {"agent": 0, "user": 0}
    calls = {"agent": 0, "user": 0}
    for m in sim.messages:
        if getattr(m, "usage", None) and m.role in ("assistant", "user"):
            who = "agent" if m.role == "assistant" else "user"
            tok[who] += int(m.usage.get("prompt_tokens", 0) or 0) + int(m.usage.get("completion_tokens", 0) or 0)
            calls[who] += 1
    # our reward on the same run: DB match and communicate_info said
    res["ours"] = float((res["db"] or 0) * (res["communicate"] if res["communicate"] is not None else 1))
    res.update(termination=str(sim.termination_reason.value), seconds=dur, tokens=tok, calls=calls,
               n_messages=len(sim.messages))
    return res


def run_ours(D, llm, task, patches):
    llm.meter(reset=True)
    t0 = time.time()
    tr = D.m.run_agent(llm, task, D.m.components_with([], patches))
    dur = time.time() - t0
    d = json.loads(json.dumps(tr, default=lambda o: o.__dict__))
    res = score(to_messages(d, task), task, D.m.DOMAIN, tr.stopped == "user_stop")
    res.update(ours=float(tr.reward), termination=tr.stopped, seconds=dur,
               tokens={"all": llm.meter(reset=True)}, n_steps=len(tr.steps))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n_null", type=int, default=10)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--max_tasks", type=int, default=None)
    args = ap.parse_args()
    D = Domain(args.domain, split="test")
    setup_tau2()
    llm0 = LLM(temperature=0.0)
    llmd = LLM()
    held = _load(os.path.join(args.out, "heldout.json"))
    apply_set = json.loads(held["sets"]["LLM-only|10|0"])
    path = os.path.join(args.out, "harness_check.json")
    res = _load(path) or {"episodes": [], "arms": {}, "null": {}, "apply_set": apply_set}
    eps = [(tid, t) for tid in D.split["test"][:args.max_tasks] for t in range(args.trials)]
    res["episodes"] = eps
    arms = ["ours_t0|none", "ours_t0|apply", "off_t0|none", "off_t0|apply"]
    if args.domain == "tau2_airline":
        arms += ["ours_def|none", "ours_def|apply"]
    arms = [a for a in arms if not args.only or a in args.only]
    for arm in arms:
        have = res["arms"].setdefault(arm, {})
        todo = [e for e in range(len(eps)) if str(e) not in have]
        kind, ps = arm.split("|")
        patches = apply_set if ps == "apply" else {}

        def one(e):
            tid, trial = eps[e]
            task = D.tasks[tid]
            for attempt in range(3):
                try:
                    if kind == "off_t0":
                        return e, run_official(D, task, patches, trial)
                    return e, run_ours(D, llm0 if kind == "ours_t0" else llmd, task, patches)
                except Exception as ex:             # transient API failure
                    err = str(ex)[:300]
                    time.sleep(5 * (attempt + 1))
            return e, {"error": err}
        for s in range(0, len(todo), 2 * args.workers):
            for e, r in _pmap(one, todo[s:s + 2 * args.workers], args.workers):
                have[str(e)] = r
            _save(path, res)
        ok = [v for v in have.values() if "error" not in v]
        print(arm, len(ok), "ours %.3f" % np.mean([v["ours"] for v in ok]),
              "official %.3f" % np.mean([v["official"] or 0 for v in ok]),
              "errors", len(have) - len(ok), flush=True)

    if not args.only or "null" in args.only:
        # null replays at temperature 0 from the bank's source traces and steps
        bank = _load(os.path.join(args.out, "bank.json"))
        attrs = _load(os.path.join(args.out, "attributions.json"))
        Dtr = Domain(args.domain, split="train")
        by_uid = {d["uid"]: d for d in _load(os.path.join(args.out, "traces.json"))}
        keys = [k for k, v in bank["null"].items() if int(k.split("|")[1]) < args.n_null
                and k not in res["null"]][:args.max_tasks]

        def nul(k):
            v = bank["null"][k]
            d = by_uid[v["trace"]]
            tr = Dtr.trace_from(d)
            step = max(0, min(attrs[v["trace"]]["step"], len(tr.steps) - 1))
            llm0.meter(reset=True)
            for attempt in range(3):
                try:
                    new = Dtr.m.run_agent(llm0, Dtr.tasks[d["task_id"]],
                                          Dtr.m.components_with(tr.faults, {}), tr.faults,
                                          prefix=tr, start_step=step)
                    return k, {"y": new.reward, "tok": llm0.meter(reset=True), "cat": v["cat"],
                               "y_default": v["y"], "trace": v["trace"]}
                except Exception as ex:
                    err = str(ex)[:300]
                    time.sleep(5 * (attempt + 1))
            return k, {"error": err}
        for s in range(0, len(keys), 2 * args.workers):
            for k, r in _pmap(nul, keys[s:s + 2 * args.workers], args.workers):
                res["null"][k] = r
            _save(path, res)
        ok = [v for v in res["null"].values() if "error" not in v]
        print("null T0 %.3f vs default %.3f (n=%d)" % (np.mean([v["y"] for v in ok]),
                                                       np.mean([v["y_default"] for v in ok]), len(ok)))
    print("tokens", {m: v for m, v in llm0.io.items()}, {m: v for m, v in llmd.io.items()})


if __name__ == "__main__":
    main()
