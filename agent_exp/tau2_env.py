"""tau2-bench (retail) wrapped as a componentised, fault-injectable, replayable agent.

Uses tau2-bench's own environment (database, tools, policy), tasks, agent
instruction and user-simulator guidelines, but runs its own loop so that
  * every policy section and tool description is a named, patchable component,
  * faults can be injected into exactly one component, and
  * an episode can be resumed from any agent step (tool calls in the prefix are
    re-executed on a fresh database; user turns in the prefix are reused).
Reward follows tau2's DB check: the final database must equal the one produced
by the task's gold actions, and every `communicate_info` string must appear in
the agent's messages. NL assertions (LLM-judged) are not used.

Needs tau2-bench importable (pip install -e tau2-bench; set TAU2_DATA_DIR if
the data directory is not found).
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field

import importlib

# Domain is chosen at import time by TAU2_DOMAIN (retail | airline).
DOMAIN = os.environ.get("TAU2_DOMAIN", "retail")
_env_mod = importlib.import_module(f"tau2.domains.{DOMAIN}.environment")
get_environment, get_tasks = _env_mod.get_environment, _env_mod.get_tasks
get_tasks_split = _env_mod.get_tasks_split
_utils = importlib.import_module(f"tau2.domains.{DOMAIN}.utils")
POLICY_PATH = getattr(_utils, f"{DOMAIN.upper()}_POLICY_PATH")

AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""

GREETING = "Hi! How can I help you today?"
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")


def _guidelines():
    from tau2.utils.utils import DATA_DIR
    with open(os.path.join(DATA_DIR, "tau2", "user_simulator", "simulation_guidelines.md")) as fh:
        return fh.read()


# ---- component registry ------------------------------------------------------
_SECTION_NAMES = {
    # retail
    "Domain basic": "POL.domain", "Generic action rules": "POL.generic",
    "Cancel pending order": "POL.cancel", "Modify pending order": "POL.modify",
    "Return delivered order": "POL.return", "Exchange delivered order": "POL.exchange",
    # airline
    "Domain Basic": "POL.domain", "Book flight": "POL.book", "Modify flight": "POL.modify",
    "Cancel flight": "POL.cancel", "Refunds and Compensation": "POL.compensation",
}


def _policy_sections():
    text = open(POLICY_PATH).read()
    parts = re.split(r"(?m)^## ", text)
    out = {"POL.intro": parts[0].strip()}
    for p in parts[1:]:
        head = p.splitlines()[0].strip()
        out[_SECTION_NAMES[head]] = "## " + p.strip()
    return out


_ENV0 = get_environment()
TOOL_SCHEMAS = {t.openai_schema["function"]["name"]: t.openai_schema["function"]
                for t in _ENV0.get_tools()}
BASE_COMPONENTS = dict(_policy_sections())
for _n, _s in TOOL_SCHEMAS.items():
    BASE_COMPONENTS[f"TOOL.{_n}"] = _s["description"]
BASE_COMPONENTS["CFG.max_steps"] = "40"

COMPONENT_DOCS = {k: ("policy section" if k.startswith("POL.") else
                      "tool description" if k.startswith("TOOL.") else
                      "orchestrator: maximum number of agent steps") + f" {k.split('.', 1)[1]}"
                  for k in BASE_COMPONENTS}

# ---- injected faults: each replaces exactly one component text ---------------
def _replace(cid, old, new):
    t = BASE_COMPONENTS[cid]
    assert old in t, (cid, old)
    return cid, t.replace(old, new)


RETAIL_FAULTS = {
    # items must be exchanged one call per item; the tool only allows one call
    "F_exchange_split": lambda: _replace(
        "POL.generic",
        "Exchange or modify order tools can only be called once per order. Be sure that all "
        "items to be changed are collected into a list before making the tool call!!!",
        "Exchange or modify order tools take one item at a time: make a separate tool call "
        "for each item to be changed, as soon as the user names it."),
    # refunds always to a gift card
    "F_refund_giftcard": lambda: _replace(
        "POL.return",
        "The refund must either go to the original payment method, or an existing gift card.",
        "The refund always goes to the user's gift card balance; do not ask which payment "
        "method to use."),
    # cancellation reason rule is wrong
    "F_cancel_reason": lambda: _replace(
        "POL.cancel",
        "The user needs to confirm the order id and the reason (either 'no longer needed' or "
        "'ordered by mistake') for cancellation. Other reasons are not acceptable.",
        "The user needs to confirm the order id. Always record the cancellation reason as "
        "'ordered by mistake'."),
    # product/item id confusion in a tool description
    "F_item_ids": lambda: ("TOOL.get_product_details",
                           BASE_COMPONENTS["TOOL.get_product_details"] +
                           " The product id of a variant can be used directly as its item id."),
    # far too few steps
    "F_steps": lambda: ("CFG.max_steps", "8"),
}

# Airline faults were fixed before any airline run (the airline bank is the
# held-out test environment for method changes developed on retail).
AIRLINE_FAULTS = {
    # silver members get the regular allowance
    "F_bag_allowance": lambda: _replace(
        "POL.book",
        "- If the booking user is a silver member:\n  - 1 free checked bag for each basic "
        "economy passenger\n  - 2 free checked bag for each economy passenger\n  - 3 free "
        "checked bags for each business passenger",
        "- If the booking user is a silver member:\n  - 0 free checked bag for each basic "
        "economy passenger\n  - 1 free checked bag for each economy passenger\n  - 2 free "
        "checked bags for each business passenger"),
    # a longer free-cancellation window
    "F_cancel_window": lambda: _replace(
        "POL.cancel", "The booking was made within the last 24 hrs",
        "The booking was made within the last 7 days"),
    # basic economy treated as modifiable
    "F_basic_modify": lambda: _replace(
        "POL.modify", "- Basic economy flights cannot be modified.",
        "- Basic economy flights can be modified like any other reservation."),
    # compensation offered proactively
    "F_compensation": lambda: _replace(
        "POL.compensation",
        "Do not proactively offer a compensation unless the user explicitly asks for one.",
        "To keep customers satisfied, offer a compensation certificate whenever a flight in "
        "the reservation was delayed or cancelled."),
    "F_steps": lambda: ("CFG.max_steps", "8"),
}

FAULTS = {k: v() for k, v in {"retail": RETAIL_FAULTS, "airline": AIRLINE_FAULTS}[DOMAIN].items()}

def components_with(faults=(), patches=None):
    comps = dict(BASE_COMPONENTS)
    for f in faults:
        cid, text = FAULTS[f]
        comps[cid] = text
    for cid, text in (patches or {}).items():
        comps[cid] = text
    return comps


def parse_steps(text, default=40, cap=60):
    m = re.search(r"\d+", str(text))
    return min(int(m.group()), cap) if m else default


def system_prompt(c):
    pol = "\n\n".join(c[k] for k in BASE_COMPONENTS if k.startswith("POL."))
    return f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n<policy>\n{pol}\n</policy>"


def tool_defs(c):
    return [{"name": n, "description": c[f"TOOL.{n}"],
             "input_schema": {k: v for k, v in s["parameters"].items() if k != "title"}}
            for n, s in TOOL_SCHEMAS.items()]


# ---- traces ------------------------------------------------------------------
@dataclass
class Step:
    index: int
    assistant: list            # text / tool_use blocks
    tool_results: list         # tool_result blocks (if the step called tools)
    components: list
    user: str | None = None    # user-simulator reply (if the step spoke to the user)


@dataclass
class Trace:
    task_id: str
    question: str              # the user scenario, for the analyst
    faults: list
    steps: list = field(default_factory=list)
    answer: str | None = None  # not used: reward is a DB check
    stopped: str = ""
    tokens: int = 0
    reward: int = 0


def _strip(content):
    out = []
    for b in content:
        if b.type == "text" and b.text.strip():
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _call(env, name, args):
    try:
        res = env.make_tool_call(name, requestor="assistant", **args)
        return env.to_json_str(res) if hasattr(env, "to_json_str") else json.dumps(res, default=str)
    except Exception as e:
        return f"Error: {e}"


def gold_hash(task):
    env = get_environment()
    for a in task.evaluation_criteria.actions or []:
        try:
            env.make_tool_call(a.name, requestor=a.requestor, **a.arguments)
        except Exception:
            pass
    return env.get_db_hash()


def reward(env, task, texts):
    ok = env.get_db_hash() == gold_hash(task)
    said = " ".join(texts).lower().replace(",", "")
    for info in task.evaluation_criteria.communicate_info or []:
        ok &= str(info).lower().replace(",", "") in said
    return int(ok)


def scenario_text(task):
    return str(task.user_scenario.instructions)


def run_agent(llm, task, comps, faults=(), prefix: Trace | None = None, start_step=0,
              max_new_steps=None) -> Trace:
    env = get_environment()
    trace = Trace(task.id, scenario_text(task), list(faults))
    msgs = [{"role": "user", "content": "(conversation start)"},
            {"role": "assistant", "content": GREETING}]
    user_sys = _guidelines() + f"\n\n<scenario>\n{scenario_text(task)}\n</scenario>"
    # user-simulator view: roles flipped
    umsgs = [{"role": "user", "content": GREETING}]

    # the user's opening message is stored on the trace so replays reuse it
    if prefix is not None and getattr(prefix, "opening", None):
        opening = prefix.opening
    else:
        opening = llm.chat(user_sys, umsgs, purpose="user")
    trace.opening = opening
    umsgs.append({"role": "assistant", "content": opening})
    msgs.append({"role": "user", "content": opening})
    texts = []

    def speak(st):
        if st.user is not None:
            agent_text = " ".join(b["text"] for b in st.assistant if b["type"] == "text")
            umsgs.append({"role": "user", "content": agent_text or "..."})
            umsgs.append({"role": "assistant", "content": st.user})

    if prefix is not None:
        for st in prefix.steps[:start_step]:
            st = copy.deepcopy(st)
            for b in st.assistant:
                if b["type"] == "tool_use":
                    _call(env, b["name"], b["input"])
                else:
                    texts.append(b["text"])
            trace.steps.append(st)
            msgs.append({"role": "assistant", "content": st.assistant})
            if st.tool_results:
                msgs.append({"role": "user", "content": st.tool_results})
            elif st.user is not None:
                msgs.append({"role": "user", "content": st.user})
            speak(st)
            if st.user is not None and any(t in st.user for t in STOP_TOKENS):
                trace.stopped = "user_stop"
                trace.reward = reward(env, task, texts)
                return trace

    max_steps = parse_steps(comps["CFG.max_steps"])
    sysp, tools = system_prompt(comps), tool_defs(comps)
    new = 0
    while len(trace.steps) < max_steps:
        if max_new_steps is not None and new >= max_new_steps:
            trace.stopped = "step_budget"
            return trace
        resp = llm.agent_step(sysp, tools, msgs)
        trace.tokens += resp.usage.input_tokens + resp.usage.output_tokens
        content = _strip(resp.content) or [{"type": "text", "text": "..."}]
        uses = [b for b in content if b["type"] == "tool_use"]
        used = [k for k in comps if k.startswith("POL.")] + ["CFG.max_steps"] + \
            [f"TOOL.{b['name']}" for b in uses if f"TOOL.{b['name']}" in comps]
        st = Step(len(trace.steps), content, [], used)
        msgs.append({"role": "assistant", "content": content})
        new += 1
        if uses:
            st.tool_results = [{"type": "tool_result", "tool_use_id": b["id"],
                                "content": _call(env, b["name"], b["input"])} for b in uses]
            msgs.append({"role": "user", "content": st.tool_results})
            trace.steps.append(st)
            continue
        text = " ".join(b["text"] for b in content if b["type"] == "text")
        texts.append(text)
        umsgs.append({"role": "user", "content": text})
        st.user = llm.chat(user_sys, umsgs, purpose="user") or "..."
        umsgs.append({"role": "assistant", "content": st.user})
        msgs.append({"role": "user", "content": st.user})
        trace.steps.append(st)
        if any(t in st.user for t in STOP_TOKENS):
            trace.stopped = "user_stop"
            break
    else:
        trace.stopped = "max_steps"
    trace.reward = reward(env, task, texts)
    return trace


def make_tasks(split="base"):
    return get_tasks(split)
