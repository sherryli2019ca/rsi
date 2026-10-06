"""Thin wrapper over the Anthropic SDK with token accounting.

Two backends:
  anthropic  Claude models (default claude-opus-5-5) via ANTHROPIC_API_KEY.
  deepseek   DeepSeek models through DeepSeek's Anthropic-compatible endpoint
             (https://api.deepseek.com/anthropic). Selected with
             LLM_BACKEND=deepseek, or automatically when ANTHROPIC_API_KEY is
             unset. Defaults: agent deepseek-v4-flash, analyst deepseek-v4-pro.
             The endpoint rejects forced tool_choice in thinking mode, so both
             roles run with thinking disabled and analyst JSON comes from a
             forced tool call validated against the schema.

Set AGENT_MODEL / ANALYST_MODEL to change models. The analyst (attribution,
taxonomy, patches, single-step judge) and the agent are kept separate so their
token costs can be reported separately.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict

BACKEND = os.environ.get(
    "LLM_BACKEND", "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "deepseek")
_DEFAULTS = {"anthropic": ("claude-opus-5-5", "claude-opus-5-5"),
             "deepseek": ("deepseek-v4-flash", "deepseek-v4-pro")}[BACKEND]
AGENT_MODEL = os.environ.get("AGENT_MODEL", _DEFAULTS[0])
ANALYST_MODEL = os.environ.get("ANALYST_MODEL", _DEFAULTS[1])


def _valid(obj, schema):
    """Check required keys and enums (the DeepSeek endpoint does not enforce them)."""
    if not isinstance(obj, dict):
        return False
    for k in schema.get("required", []):
        if k not in obj:
            return False
    for k, p in schema.get("properties", {}).items():
        if k in obj and "enum" in p and obj[k] not in p["enum"]:
            return False
    return True


class LLM:
    def __init__(self, agent_effort="low", analyst_effort="medium"):
        import anthropic

        if BACKEND == "deepseek":
            # The key is injected by the environment's egress proxy.
            self.client = anthropic.Anthropic(base_url="https://api.deepseek.com/anthropic",
                                              api_key=os.environ.get("DEEPSEEK_API_KEY", "proxy"),
                                              max_retries=6, timeout=300)
        else:
            self.client = anthropic.Anthropic()
        self.agent_effort, self.analyst_effort = agent_effort, analyst_effort
        self.tokens = defaultdict(int)          # purpose -> tokens
        self.io = defaultdict(lambda: [0, 0, 0])  # model -> [uncached in, out, cache-read in]
        self._lock = threading.Lock()
        self.local = threading.local()          # per-thread token meter, see meter()

    def _count(self, purpose, model, resp):
        u = resp.usage
        self.local.acc = getattr(self.local, "acc", 0) + u.input_tokens + u.output_tokens
        with self._lock:
            self.tokens[purpose] += u.input_tokens + u.output_tokens
            self.io[model][0] += u.input_tokens
            self.io[model][1] += u.output_tokens
            self.io[model][2] += getattr(u, "cache_read_input_tokens", 0) or 0

    def meter(self, reset=False):
        """Tokens used by the calling thread since the last reset."""
        v = getattr(self.local, "acc", 0)
        if reset:
            self.local.acc = 0
        return v

    def _create(self, **kw):
        for attempt in range(5):
            try:
                return self.client.messages.create(**kw)
            except Exception as e:                       # transient server errors
                if attempt == 4 or getattr(e, "status_code", 500) in (400, 401, 403):
                    raise
                time.sleep(2 ** attempt)

    def agent_step(self, system, tools, messages, purpose="agent"):
        if BACKEND == "deepseek":
            resp = self._create(model=AGENT_MODEL, max_tokens=4000, system=system,
                                tools=tools, messages=messages, thinking={"type": "disabled"})
        else:
            resp = self._create(model=AGENT_MODEL, max_tokens=4000, system=system, tools=tools,
                                messages=messages, output_config={"effort": self.agent_effort})
        self._count(purpose, AGENT_MODEL, resp)
        return resp

    def chat(self, system, messages, purpose="user", model=None):
        """Plain text turn (used for the tau2 user simulator)."""
        model = model or AGENT_MODEL
        kw = {"thinking": {"type": "disabled"}} if BACKEND == "deepseek" else {}
        resp = self._create(model=model, max_tokens=2000, system=system,
                            messages=messages, **kw)
        self._count(purpose, model, resp)
        return "".join(b.text for b in resp.content if b.type == "text")

    def ask_json(self, prompt, schema, purpose, max_tokens=8000):
        """One analyst call constrained to a JSON schema."""
        if BACKEND == "deepseek":
            return self._ask_json_tool(prompt, schema, purpose, max_tokens)
        resp = self._create(
            model=ANALYST_MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.analyst_effort,
                           "format": {"type": "json_schema", "schema": schema}})
        self._count(purpose, ANALYST_MODEL, resp)
        if resp.stop_reason == "refusal":
            raise RuntimeError("analyst call refused")
        text = next(b.text for b in resp.content if b.type == "text")
        return json.loads(text)

    def _ask_json_tool(self, prompt, schema, purpose, max_tokens):
        tool = {"name": "submit", "description": "Submit your answer.", "input_schema": schema}
        for _ in range(3):
            resp = self._create(model=ANALYST_MODEL, max_tokens=max_tokens, tools=[tool],
                                tool_choice={"type": "tool", "name": "submit"},
                                thinking={"type": "disabled"},
                                messages=[{"role": "user", "content": prompt}])
            self._count(purpose, ANALYST_MODEL, resp)
            out = next((b.input for b in resp.content if b.type == "tool_use"), None)
            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except ValueError:
                    out = None
            if _valid(out, schema):
                return out
        raise RuntimeError(f"analyst output failed schema validation: {out!r}")
