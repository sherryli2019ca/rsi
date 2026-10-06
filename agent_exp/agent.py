"""A ReAct-style tool-use agent assembled from named, patchable components.

Every step of a trace records which components shaped it, so an attribution
that points at a step can be mapped to the component registry.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from agent_exp.env import ShopDB, Tools

# ---- component registry ------------------------------------------------------
BASE_COMPONENTS = {
    "SYS.role": "You are a customer-service agent for an online shop. Use the tools to look "
                "up facts; never guess database values.",
    "SYS.plan": "Before acting, decide which records you need. Read the relevant policy "
                "before applying any rule.",
    "SYS.verify": "Before answering, re-check every number you use against tool outputs and "
                  "recompute totals with the calculate tool.",
    "SYS.format": "When you are done, reply with a single line 'FINAL: <answer>'. Dollar "
                  "amounts use two decimals without a currency sign; counts are integers.",
    "TOOL.get_customer": "Look up a customer by id. Returns name and tier.",
    "TOOL.list_orders": "List the order ids of a customer. Results are paginated: pass "
                        "page=2, 3, ... while has_more is true.",
    "TOOL.get_order": "Get an order: items (sku, qty), status and days since delivery.",
    "TOOL.get_product": "Get a product: name, category and price_cents (price in cents).",
    "TOOL.get_policy": "Get a shop policy by name: 'refund' or 'cancellation'.",
    "TOOL.calculate": "Evaluate an arithmetic expression.",
    "CFG.max_steps": "14",
}

COMPONENT_DOCS = {
    "SYS.role": "system prompt: role and grounding rule",
    "SYS.plan": "system prompt: planning instructions",
    "SYS.verify": "system prompt: verification instructions",
    "SYS.format": "system prompt: answer format",
    "TOOL.get_customer": "tool description of get_customer",
    "TOOL.list_orders": "tool description of list_orders",
    "TOOL.get_order": "tool description of get_order",
    "TOOL.get_product": "tool description of get_product",
    "TOOL.get_policy": "tool description of get_policy",
    "TOOL.calculate": "tool description of calculate",
    "CFG.max_steps": "orchestrator: maximum number of agent steps",
}

TOOL_SCHEMAS = {
    "get_customer": {"customer_id": {"type": "string"}},
    "list_orders": {"customer_id": {"type": "string"}, "page": {"type": "integer"}},
    "get_order": {"order_id": {"type": "string"}},
    "get_product": {"sku": {"type": "string"}},
    "get_policy": {"name": {"type": "string"}},
    "calculate": {"expression": {"type": "string"}},
}
REQUIRED = {"list_orders": ["customer_id"]}

# ---- injected faults (ground truth) ------------------------------------------
FAULTS = {
    # Each fault replaces exactly one component text. The first versions were
    # subtler (e.g. "price_cents (price in dollars)"); with deepseek-v4-flash as
    # the agent they almost never changed the outcome (smoke run, 0/13 failures),
    # so the misleading statements were made explicit.
    "F_price_unit": ("TOOL.get_product",
                     "Get a product: name, category and price_cents. Despite its legacy "
                     "name, price_cents is already in dollars: use it directly as the "
                     "dollar price."),
    "F_pagination": ("TOOL.list_orders",
                     "List all order ids of a customer in a single call. The result is "
                     "always complete: ignore has_more and never request further pages."),
    "F_format": ("SYS.format", "When you are done, reply with 'FINAL: <answer>'. Round dollar "
                               "amounts to whole dollars."),
    "F_policy": ("SYS.plan", "Before acting, decide which records you need. Shop policies are "
                             "standard and well known: every delivered order is refundable "
                             "within 30 days and electronics always carry a 10% restocking "
                             "fee, so do not call get_policy."),
    "F_steps": ("CFG.max_steps", "3"),
}


def components_with(faults=(), patches=None):
    comps = dict(BASE_COMPONENTS)
    for f in faults:
        cid, text = FAULTS[f]
        comps[cid] = text
    for cid, text in (patches or {}).items():
        comps[cid] = text
    return comps


def parse_steps(text, default=14, cap=40):
    """CFG.max_steps patches are free text from the analyst; take the first integer."""
    import re

    m = re.search(r"\d+", str(text))
    return min(int(m.group()), cap) if m else default


def system_prompt(c):
    return "\n\n".join(c[k] for k in ("SYS.role", "SYS.plan", "SYS.verify", "SYS.format"))


def tool_defs(c):
    return [{"name": n, "description": c[f"TOOL.{n}"],
             "input_schema": {"type": "object", "properties": props,
                              "required": REQUIRED.get(n, list(props))}}
            for n, props in TOOL_SCHEMAS.items()]


# ---- traces ------------------------------------------------------------------
@dataclass
class Step:
    index: int
    assistant: list            # content blocks (text / tool_use), thinking stripped
    tool_results: list         # tool_result blocks returned for this step
    components: list           # component ids that shaped this step


@dataclass
class Trace:
    task_id: str
    question: str
    faults: list
    steps: list = field(default_factory=list)
    answer: str | None = None
    stopped: str = ""
    tokens: int = 0

    def to_json(self):
        return json.dumps(self.__dict__, default=lambda o: o.__dict__)


def _strip(content):
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def _final(content):
    for b in content:
        if b.get("type") == "text" and "FINAL:" in b["text"]:
            return b["text"].split("FINAL:")[-1].strip().splitlines()[0].strip()
    return None


def run_agent(llm, db: ShopDB, task, comps, faults=(), prefix: Trace | None = None,
              start_step: int = 0, max_new_steps: int | None = None) -> Trace:
    """Run (or resume) an episode.

    With `prefix`, steps [0, start_step) are replayed verbatim from the recorded
    trace (no model calls; tool results are deterministic) and generation
    resumes at start_step under the given components.
    """
    tools = Tools(db)
    trace = Trace(task.tid, task.question, list(faults))
    messages = [{"role": "user", "content": task.question}]
    if prefix is not None:
        for st in prefix.steps[:start_step]:
            trace.steps.append(copy.deepcopy(st))
            messages.append({"role": "assistant", "content": st.assistant})
            if st.tool_results:
                messages.append({"role": "user", "content": st.tool_results})
    max_steps = parse_steps(comps["CFG.max_steps"])
    new = 0
    while len(trace.steps) < max_steps:
        if max_new_steps is not None and new >= max_new_steps:
            trace.stopped = "step_budget"
            return trace
        resp = llm.agent_step(system_prompt(comps), tool_defs(comps), messages)
        trace.tokens += resp.usage.input_tokens + resp.usage.output_tokens
        content = _strip(resp.content)
        uses = [b for b in content if b["type"] == "tool_use"]
        used = ["SYS.role", "SYS.plan", "SYS.verify", "SYS.format", "CFG.max_steps"] + \
            [f"TOOL.{b['name']}" for b in uses if f"TOOL.{b['name']}" in comps]
        results = [{"type": "tool_result", "tool_use_id": b["id"],
                    "content": tools.call(b["name"], b["input"])} for b in uses]
        trace.steps.append(Step(len(trace.steps), content, results, used))
        new += 1
        messages.append({"role": "assistant", "content": content})
        if not uses:
            trace.answer = _final(content)
            trace.stopped = "answered"
            return trace
        messages.append({"role": "user", "content": results})
    trace.stopped = "max_steps"
    return trace
