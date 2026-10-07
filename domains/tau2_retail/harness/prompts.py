"""Model-facing text of the harness. The starting values are tau2-bench's own
LLMAgent instruction and system-prompt layout, verbatim."""
from __future__ import annotations

AGENT_INSTRUCTION = """You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only."""


def system_prompt(policy: str) -> str:
    return (f"<instructions>\n{AGENT_INSTRUCTION}\n</instructions>\n"
            f"<policy>\n{policy}\n</policy>")
