"""LLM analyst: failure attribution, taxonomy induction, labelling, patch
generation and the single-step judge."""
from __future__ import annotations

import json

from agent_exp.agent import COMPONENT_DOCS, Trace


def render_trace(t: Trace, max_chars=12000) -> str:
    lines = [f"QUESTION: {t.question}"]
    for st in t.steps:
        for b in st.assistant:
            if b["type"] == "text":
                lines.append(f"[step {st.index}] AGENT: {b['text']}")
            else:
                lines.append(f"[step {st.index}] CALL {b['name']}({json.dumps(b['input'])})")
        for r in st.tool_results:
            lines.append(f"[step {st.index}] RESULT: {r['content']}")
    lines.append(f"FINAL ANSWER: {t.answer}   (episode ended: {t.stopped})")
    s = "\n".join(lines)
    return s if len(s) <= max_chars else s[:max_chars // 2] + "\n...\n" + s[-max_chars // 2:]


def registry_text(comps) -> str:
    return "\n".join(f"- {cid} ({COMPONENT_DOCS[cid]}): {comps[cid]!r}" for cid in COMPONENT_DOCS)


ATTR_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["root_cause", "step", "component"],
    "properties": {
        "root_cause": {"type": "string"},
        "step": {"type": "integer"},
        "component": {"type": "string", "enum": list(COMPONENT_DOCS)},
    },
}


def attribute(llm, trace: Trace, gold: str, comps) -> dict:
    prompt = f"""An LLM agent failed a task. Find the root cause of the failure.

Agent components (id, role, current text):
{registry_text(comps)}

Trace:
{render_trace(trace)}

The correct answer was: {gold}

Give (1) a one-sentence root cause describing the cause, not the symptom;
(2) the index of the earliest step where the agent went wrong;
(3) the single component whose change would most plausibly prevent this failure."""
    return llm.ask_json(prompt, ATTR_SCHEMA, "attribution")


TAX_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["categories"],
    "properties": {"categories": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["name", "definition"],
        "properties": {"name": {"type": "string"}, "definition": {"type": "string"}}}}},
}


def induce_taxonomy(llm, root_causes: list[str], batch=40, max_categories=12) -> list[dict]:
    """TnT-LLM-style iterative induction over batches of root-cause sentences."""
    cats: list[dict] = []
    for s in range(0, len(root_causes), batch):
        chunk = "\n".join(f"- {r}" for r in root_causes[s:s + batch])
        prompt = f"""You maintain a taxonomy of LLM-agent failure causes.
Current taxonomy (may be empty): {json.dumps(cats)}

New root-cause descriptions:
{chunk}

Update the taxonomy so that it covers these descriptions: add, merge or split
categories as needed. Categories must describe causes, be mutually exclusive,
and number at most {max_categories}. Give each a short name and a one-sentence
definition with its boundary against similar categories."""
        cats = llm.ask_json(prompt, TAX_SCHEMA, "taxonomy")["categories"]
    return cats


def label(llm, root_cause: str, cats: list[dict]) -> int:
    names = [c["name"] for c in cats] + ["other"]
    schema = {"type": "object", "additionalProperties": False, "required": ["category"],
              "properties": {"category": {"type": "string", "enum": names}}}
    prompt = (f"Taxonomy: {json.dumps(cats)}\n\nRoot cause: {root_cause}\n\n"
              "Which category fits best? Use 'other' if none fits.")
    out = llm.ask_json(prompt, schema, "labelling", max_tokens=2000)["category"]
    return names.index(out)


PATCH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["patches"],
    "properties": {"patches": {"type": "array", "items": {"type": "string"}}},
}


def make_patches(llm, cid: str, comps, category: dict, examples: list[str], k=3) -> list[str]:
    prompt = f"""You improve one component of an LLM agent.
Component {cid} ({COMPONENT_DOCS[cid]}) currently reads:
{comps[cid]!r}

It is suspected of causing this failure category:
{category['name']}: {category['definition']}
Example root causes:
""" + "\n".join(f"- {e}" for e in examples[:5]) + f"""

Write {k} alternative replacement texts for this component only, each a
different plausible fix. For CFG.max_steps reply with integers as strings."""
    patches = llm.ask_json(prompt, PATCH_SCHEMA, "patches")["patches"][:k]
    while len(patches) < k:
        patches.append(patches[-1])
    return patches


JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fixed"],
                "properties": {"fixed": {"type": "boolean"}}}


def judge_step(llm, trace: Trace, step: int, root_cause: str, new_step_blocks) -> int:
    prompt = f"""An agent failed; the diagnosed error is at step {step}: {root_cause}

Original trace:
{render_trace(trace)}

A modified agent produced this action at step {step} instead:
{json.dumps(new_step_blocks)}

Does the new action avoid the diagnosed error, so that the episode is back on a
correct path?"""
    return int(llm.ask_json(prompt, JUDGE_SCHEMA, "judge", max_tokens=2000)["fixed"])
