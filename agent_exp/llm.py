"""Thin wrapper over the Anthropic SDK with token accounting.

Models default to claude-opus-5-5; set AGENT_MODEL / ANALYST_MODEL to change
them. The analyst (attribution, taxonomy, patches, single-step judge) and the
agent are kept separate so their token costs can be reported separately.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

AGENT_MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5-5")
ANALYST_MODEL = os.environ.get("ANALYST_MODEL", "claude-opus-5-5")


class LLM:
    def __init__(self, agent_effort="low", analyst_effort="medium"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.agent_effort, self.analyst_effort = agent_effort, analyst_effort
        self.tokens = defaultdict(int)          # purpose -> tokens

    def _count(self, purpose, resp):
        u = resp.usage
        self.tokens[purpose] += u.input_tokens + u.output_tokens

    def agent_step(self, system, tools, messages, purpose="agent"):
        resp = self.client.messages.create(
            model=AGENT_MODEL, max_tokens=4000, system=system, tools=tools,
            messages=messages, output_config={"effort": self.agent_effort})
        self._count(purpose, resp)
        return resp

    def ask_json(self, prompt, schema, purpose, max_tokens=8000):
        """One analyst call constrained to a JSON schema."""
        resp = self.client.messages.create(
            model=ANALYST_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.analyst_effort,
                           "format": {"type": "json_schema", "schema": schema}})
        self._count(purpose, resp)
        if resp.stop_reason == "refusal":
            raise RuntimeError("analyst call refused")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)
