"""Optional building blocks for structural edits. The starting harness uses none
of them; wire one into agent.py to use it.

  subcall(llm, brief, context)          one bounded extra text-only policy call
  ToolRegistry, run_with_client_tools   harness-side ("client") tools the policy
                                        can call with a ```tool block; they run
                                        here, never in the AppWorld REPL
  prepend_to_code(text, source)         put harness-provided Python (helper
                                        functions for the REPL) in front of the
                                        code block of an assistant turn
  load_skills(dir), skill_catalog()     procedure files under skills/
  code_of(text)                         the code the driver will execute for a
                                        turn (a copy of its frozen rule)

Everything is per episode. The driver builds a new Agent for every episode and
runs episodes in separate processes, so nothing may persist across episodes
(no module-level mutable state, no files written): the harness is evaluated on
the tasks it evolves on, and a replayed episode must not depend on which
episodes ran before it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_FULL_CODE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
_PARTIAL_CODE = re.compile(r".*```python[ \t]*\n(.*)", re.S)


def code_of(text: str) -> str:
    """The code the driver executes for an assistant turn (a copy of the
    driver's rule; changing it here does not change what the driver runs)."""
    m = _FULL_CODE.search(text)
    if m:
        return m.group(1).strip()
    m = _PARTIAL_CODE.match(text)
    return m.group(1).strip() if m else ""


def transcript_text(messages: list[dict], max_chars: int = 12000) -> str:
    """The episode as plain text (most recent part if it is long)."""
    lines = []
    for m in messages:
        role = "AGENT" if m["role"] == "assistant" else "ENVIRONMENT"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)[-max_chars:]


def subcall(llm, brief: str, context: str, max_tokens: int = 800) -> str:
    """One extra call to the frozen policy; returns its text. Keep the brief
    tight ("return only X") and the call bounded."""
    text, _ = llm(brief, [{"role": "user", "content": context}], max_tokens=max_tokens)
    return text.strip()


def prepend_to_code(text: str, source: str) -> str:
    """Insert `source` (Python) at the start of the first python block of an
    assistant turn, e.g. helper functions the policy may then call in the
    REPL. The transcript keeps the modified text, so replays re-execute it.
    Returns `text` unchanged when it has no python block."""
    m = re.search(r"```python[ \t]*\n", text)
    if not m or not source.strip():
        return text
    return text[:m.end()] + source.rstrip() + "\n\n" + text[m.end():]


class ToolRegistry:
    """Harness-side tools: fn(**input) -> str runs locally."""

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, description: str, input_schema: dict, fn) -> None:
        self._tools[name] = {"description": description, "input_schema": input_schema, "fn": fn}

    def catalog(self) -> str:
        """Tool descriptions for the prompt, with the calling convention."""
        rows = [f"- {n}: {t['description']} input: {json.dumps(t['input_schema'])}"
                for n, t in self._tools.items()]
        return ("Harness tools (answered by the harness, not executed in the REPL). Call one "
                "INSTEAD of writing python code, as\n```tool\n{\"name\": \"<tool>\", \"input\": "
                "{...}}\n```\n" + "\n".join(rows))

    def handles(self, name: str) -> bool:
        return name in self._tools

    def run(self, name: str, inp: dict) -> str:
        try:
            return str(self._tools[name]["fn"](**(inp or {})))
        except Exception as e:  # noqa: BLE001 - a tool error goes back to the policy
            return f"Error: {e}"


_TOOL_BLOCK = re.compile(r"```tool[ \t]*\n(.*?)```", re.S)


def run_with_client_tools(llm, system: str, messages: list[dict], registry: ToolRegistry,
                          max_rounds: int = 3, max_tokens: int = 4000) -> str:
    """Call the policy; while its reply holds a ```tool block and no python
    block, answer the tool here and call again (at most max_rounds extra
    calls). Returns the first reply with python code (or the last reply).
    The tool exchanges are not added to the episode transcript."""
    msgs = list(messages)
    text = ""
    for _ in range(max_rounds + 1):
        text, _ = llm(system, msgs, max_tokens=max_tokens)
        m = _TOOL_BLOCK.search(text)
        if m is None or _FULL_CODE.search(text[:m.start()]):
            return text
        try:
            call = json.loads(m.group(1))
            name, inp = str(call.get("name")), call.get("input") or {}
            result = (registry.run(name, inp) if registry.handles(name)
                      else f"Error: unknown tool {name!r}")
        except (ValueError, AttributeError) as e:
            result = f"Error: tool block is not valid JSON ({e})"
        msgs = msgs + [{"role": "assistant", "content": text[:m.end()]},
                       {"role": "user", "content": f"Tool result:\n{result}"}]
    return text


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
