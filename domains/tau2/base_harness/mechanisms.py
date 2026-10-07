"""Optional building blocks for structural edits. The starting harness uses none
of them; wire one into agent.py to use it.

  subcall(llm, brief, context)        one bounded extra text-only policy call
  ToolRegistry, run_with_client_tools harness-side ("client") tools the policy
                                      can call; they run here, not in the
                                      environment, and never touch its database
  load_skills(dir), skill_catalog()   procedure files under skills/

Everything is per episode. The driver builds a new Agent for every episode and
runs episodes in parallel threads, so nothing may persist across episodes (no
module-level mutable state, no files written): the harness is evaluated on
the tasks it evolves on, and a replayed episode must not depend on which
episodes ran before it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _text(blocks) -> str:
    return " ".join(b["text"] for b in blocks if b.get("type") == "text").strip()


def transcript_text(messages: list[dict], max_chars: int = 12000) -> str:
    """The conversation as plain text (most recent part if it is long)."""
    lines = []
    for m in messages:
        role = "CUSTOMER" if m["role"] == "user" else "AGENT"
        if isinstance(m["content"], str):
            lines.append(f"{role}: {m['content']}")
            continue
        for b in m["content"]:
            if b.get("type") == "text":
                lines.append(f"{role}: {b['text']}")
            elif b.get("type") == "tool_use":
                lines.append(f"AGENT CALLS {b['name']}({json.dumps(b.get('input'))})")
            elif b.get("type") == "tool_result":
                lines.append(f"TOOL RESULT: {b.get('content')}")
    out = "\n".join(lines)
    return out[-max_chars:]


def subcall(llm, brief: str, context: str, max_tokens: int = 800) -> str:
    """One extra call to the frozen policy without tools; returns its text.
    Keep the brief tight ("return only X") and the call bounded."""
    blocks, _ = llm(brief, None, [{"role": "user", "content": context}], max_tokens=max_tokens)
    return _text(blocks)


class ToolRegistry:
    """Harness-side tools: fn(**input) -> str runs locally."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str, input_schema: dict, fn) -> None:
        self._tools[name] = {"spec": {"name": name, "description": description,
                                      "input_schema": input_schema}, "fn": fn}

    def specs(self) -> list[dict]:
        return [dict(t["spec"]) for t in self._tools.values()]

    def handles(self, name: str) -> bool:
        return name in self._tools

    def run(self, call: dict) -> str:
        try:
            return str(self._tools[call["name"]]["fn"](**(call.get("input") or {})))
        except Exception as e:  # noqa: BLE001 - a tool error goes back to the policy
            return f"Error: {e}"


def run_with_client_tools(llm, system: str, tools: list[dict], messages: list[dict],
                          registry: ToolRegistry, max_rounds: int = 3,
                          max_tokens: int = 4000) -> list[dict]:
    """Call the policy with environment + client tools. While it calls only
    client tools, run them here and call again (at most max_rounds extra
    calls). Returns the first turn that speaks to the customer or calls an
    environment tool, with any client-tool calls removed. The client-tool
    exchanges are not added to the episode transcript."""
    msgs = list(messages)
    all_tools = list(tools) + registry.specs()
    blocks: list[dict] = []
    for _ in range(max_rounds + 1):
        blocks, _ = llm(system, all_tools, msgs, max_tokens=max_tokens)
        local = [b for b in blocks if b["type"] == "tool_use" and registry.handles(b["name"])]
        rest = [b for b in blocks if not (b["type"] == "tool_use" and registry.handles(b["name"]))]
        if not local or any(b["type"] == "tool_use" for b in rest):
            return rest
        msgs = msgs + [{"role": "assistant", "content": blocks},
                       {"role": "user", "content": [
                           {"type": "tool_result", "tool_use_id": b["id"], "content": registry.run(b)}
                           for b in local]}]
    return [b for b in blocks if not (b["type"] == "tool_use" and registry.handles(b["name"]))]


_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


def load_skills(skills_dir: Path | str) -> list[dict]:
    """skills/<name>/SKILL.md (or skills/<name>.md) with YAML-like frontmatter
    `name:` and `description:`; returns [{name, description, body}]."""
    d = Path(skills_dir)
    out = []
    if not d.is_dir():
        return out
    for p in sorted(list(d.glob("*/SKILL.md")) + list(d.glob("*.md"))):
        text = p.read_text()
        meta, body = {}, text
        m = _FRONT.match(text)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = text[m.end():]
        out.append({"name": meta.get("name") or (p.parent.name if p.name == "SKILL.md" else p.stem),
                    "description": meta.get("description", ""), "body": body.strip()})
    return out


def skill_catalog(skills: list[dict]) -> str:
    return "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
