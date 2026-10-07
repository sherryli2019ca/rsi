"""Code-agent harness for AppWorld (the evolvable scaffold).

The driver (domains/appworld/driver.py, frozen) owns the AppWorld world, the
step limit, code execution and the grader. It talks to this harness only
through the contract below; keep these signatures and return types.

  Agent(llm, task)
      task  {"instruction": str,
             "supervisor": {"first_name", "last_name", "email", "phone_number"},
             "app_descriptions": {app_name: description}}
            (the supervisor is the person the agent works for; the apps are
            the ones the task's world exposes through `apis`)
  next_action(messages) -> str
      The agent's next turn: text containing ONE Python code block. The
      driver executes the first complete ```python ... ``` block of the text
      (else a trailing unterminated ```python block; else nothing, and the
      observation is "No code available to execute."), in a persistent REPL
      where `apis.<app>.<api>(...)` calls the apps and variables survive
      between steps. The returned text is what the transcript keeps.
  before_execute(code) -> str | None
      None executes the code; a string is shown to the model as the
      observation instead and the code is NOT executed (the step is used).
  render_output(code, raw) -> str
      What the model sees as the observation of executed code (`raw` is the
      REPL output: printed text, "Execution successful." or "Execution
      failed. Traceback: ...").
  observe_prefix(messages, assistant_text) -> None
      Called instead of next_action for every recorded step when an episode
      is replayed from a later step, so per-episode state can be rebuilt. Any
      state next_action keeps across turns must be reconstructible here.

  llm(system, messages, max_tokens=4000, stop=None) -> (text, usage)
      The frozen policy model (injected by the driver: model, temperature and
      thinking settings are fixed there). `system` may be "" or None; `stop`
      is an optional list of stop sequences. Every call counts towards the
      episode's token cost; more than 200 calls in an episode raise.

messages is the episode so far in Anthropic format, starting with the agent's
first turn: alternating {"role": "assistant", "content": <text next_action
returned>} and {"role": "user", "content": <the observation>} with string
contents; it is empty on the first step. Everything before it (system prompt,
demonstrations, the task statement) is the harness's own. The episode ends
when the code calls apis.supervisor.complete_task(...) or after 40 steps.
Turns must be non-empty strings (empty ones are replaced by "...").

Harness code runs in the same process as the REPL: do not use the global
`random` module (the REPL's random state is seeded and shared) and never
touch AppWorld itself.
"""
from __future__ import annotations

import re

from . import prompts

MAX_TOKENS = 4000

_FIRST_BLOCK = re.compile(r"```python[ \t]*\n.*?```", re.S)


def keep_first_code_block(text: str) -> str:
    """Drop everything after the first complete python block (the official
    ReAct agent's ignore_multiple_calls) and close an unterminated one."""
    m = _FIRST_BLOCK.search(text)
    if m:
        return text[:m.end()]
    if re.search(r"```python[ \t]*\n", text):
        return text + ("" if text.endswith("\n") else "\n") + "```"
    return text


class Agent:
    def __init__(self, llm, task: dict):
        self.llm = llm
        self.system = ""
        self.preamble = prompts.prompt_messages(task)

    def next_action(self, messages: list[dict]) -> str:
        text, _ = self.llm(self.system, self.preamble + messages, max_tokens=MAX_TOKENS)
        return keep_first_code_block(text)

    def before_execute(self, code: str) -> str | None:
        return None

    def render_output(self, code: str, raw: str) -> str:
        return prompts.format_output(raw)

    def observe_prefix(self, messages: list[dict], assistant_text: str) -> None:
        return None
