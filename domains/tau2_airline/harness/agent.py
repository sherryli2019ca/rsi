"""Customer-service agent harness for tau2-bench (the evolvable scaffold).

The driver (domains/tau2/driver.py, frozen) owns the environment, the user
simulator, the step limit and the grader. It talks to this harness only
through the contract below; keep these signatures and return types.

  Agent(llm, policy, tools)
      policy  the domain policy text (markdown)
      tools   [{"name", "description", "input_schema"}] as the environment
              defines them
  next_action(messages) -> list[dict]
      The agent's next turn: content blocks {"type": "text", "text": ...} or
      {"type": "tool_use", "id", "name", "input"}. A turn either calls tools
      or speaks to the user, not both (text blocks next to a tool call are
      kept in the transcript but not sent to the user).
  before_tool(call) -> str | None
      None executes the call against the environment; a string is returned to
      the model as the tool result instead and the call is NOT executed.
  render_result(call, raw) -> str
      What the model sees as the result of an executed call (`raw` is the
      environment's output, or "Error: ..." if the call failed).
  observe_prefix(messages, blocks) -> None
      Called instead of next_action for every recorded step when an episode is
      replayed from a later step, so per-episode state can be rebuilt. Any
      state next_action keeps across turns must be reconstructible here.

  llm(system, tools, messages, max_tokens=4000) -> (blocks, usage)
      The frozen policy model (injected by the driver: model, temperature and
      thinking settings are fixed there). `tools` may be None for a text-only
      call. Every call counts towards the episode's token cost.

messages follow the Anthropic format: alternating user / assistant turns;
tool results are user turns made of {"type": "tool_result", "tool_use_id",
"content"} blocks.
"""
from __future__ import annotations

from . import prompts, tools as tool_layer


class Agent:
    def __init__(self, llm, policy: str, tools: list[dict]):
        self.llm = llm
        self.system = prompts.system_prompt(policy)
        self.tools = tool_layer.describe(tools)

    def next_action(self, messages: list[dict]) -> list[dict]:
        blocks, _ = self.llm(self.system, self.tools, messages)
        return blocks

    def before_tool(self, call: dict) -> str | None:
        return None

    def render_result(self, call: dict, raw: str) -> str:
        return raw

    def observe_prefix(self, messages: list[dict], blocks: list[dict]) -> None:
        return None
