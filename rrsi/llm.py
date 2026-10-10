# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""LLM client for the three search roles (proposer, analyst, critic).

Two backends:
  deepseek  (default when RRSI_VERTEX_PROJECTS is unset) DeepSeek models through
            DeepSeek's Anthropic-compatible endpoint (https://api.deepseek.com/
            anthropic) with thinking disabled; RRSI_SEARCH_MODEL defaults to
            deepseek-v4-pro, and a Claude model name in a config is mapped to it.
  vertex    the original: Claude Opus 4.8 via AnthropicVertex, round-robin over
            the configured GCP projects with retry-and-rotate on failure.
`cache_prefix` sends a large stable leading block (constitution + harness
source) as its own ephemeral-cached content block so repeated turns pay to read
it once. Token usage is accumulated per model (usage()) and, when RRSI_USAGE_LOG
is set, appended to that JSONL file one line per call (with the round when
RRSI_ROUND is set).

Modified from google-research/rrsi (be50316): DeepSeek backend, usage
accounting, model-name mapping.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import threading
import time

_PROJECTS = [
    {"project_id": p.strip(), "region": os.environ.get("RRSI_VERTEX_REGION", "global")}
    for p in os.environ.get("RRSI_VERTEX_PROJECTS", "").split(",")
    if p.strip()
]
BACKEND = os.environ.get("RRSI_LLM_BACKEND", "vertex" if _PROJECTS else "deepseek")
MODEL = os.environ.get("RRSI_SEARCH_MODEL",
                       "claude-opus-4-8" if BACKEND == "vertex" else "deepseek-v4-pro")
DEEPSEEK_URL = os.environ.get("RRSI_DEEPSEEK_URL", "https://api.deepseek.com/anthropic")
MAX_TOKENS = 20_000
# Thinking for some roles (an addition of this repository, deepseek backend):
# RRSI_THINKING_ROLES=proposer,analyst,digester,critic and RRSI_THINKING_BUDGET
# (tokens, default 8000) enable thinking for calls tagged with those roles;
# every other call keeps thinking disabled.
THINKING_ROLES = {r.strip() for r in os.environ.get("RRSI_THINKING_ROLES", "").split(",") if r.strip()}
THINKING_BUDGET = int(os.environ.get("RRSI_THINKING_BUDGET") or 8000)

_clients: dict = {}
_clients_lock = threading.Lock()
_rr = itertools.count()
_usage: dict = {}
_usage_lock = threading.Lock()
_JSON_SUFFIX = ("\n\nOutput ONLY a single valid JSON object. No prose before or "
                "after, no markdown fences.")


def _client_for(idx: int):
    with _clients_lock:
        c = _clients.get(idx)
        if c is not None:
            return c
        if BACKEND == "deepseek":
            from anthropic import Anthropic
            # the key is injected by the environment's egress proxy when unset
            c = Anthropic(base_url=DEEPSEEK_URL,
                          api_key=os.environ.get("DEEPSEEK_API_KEY", "proxy"),
                          max_retries=2, timeout=600)
        else:
            from anthropic import AnthropicVertex
            if not _PROJECTS:
                raise RuntimeError("set RRSI_VERTEX_PROJECTS to a comma-separated list of GCP "
                                   "projects with Claude on Vertex AI enabled")
            c = AnthropicVertex(**_PROJECTS[idx])
        _clients[idx] = c
        return c


def resolve_model(model: str | None) -> str:
    m = model or MODEL
    if BACKEND == "deepseek" and m.startswith("claude"):
        return MODEL
    return m


def _record(model: str, resp, role: str | None) -> None:
    u = resp.usage
    row = {"model": model, "in": u.input_tokens, "out": u.output_tokens,
           "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
    with _usage_lock:
        acc = _usage.setdefault(model, {"calls": 0, "in": 0, "out": 0, "cache_read": 0})
        acc["calls"] += 1
        for k in ("in", "out", "cache_read"):
            acc[k] += row[k]
        path = os.environ.get("RRSI_USAGE_LOG")
        if path:
            with open(path, "a") as f:
                tag = ({"round": int(os.environ["RRSI_ROUND"])}
                       if os.environ.get("RRSI_ROUND", "").isdigit() else {})
                f.write(json.dumps({**row, "role": role, "t": time.time(), **tag}) + "\n")


def usage() -> dict:
    with _usage_lock:
        return json.loads(json.dumps(_usage))


def extract_json(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t.startswith("{") and not t.startswith("["):
        start = min([i for i in (t.find("{"), t.find("[")) if i != -1], default=-1)
        if start != -1:
            t = t[start:]
    if t and t[0] == "{" and not t.endswith("}"):
        end = t.rfind("}")
        if end != -1:
            t = t[:end + 1]
    if t and t[0] == "[" and not t.endswith("]"):
        end = t.rfind("]")
        if end != -1:
            t = t[:end + 1]
    return t


def generate(prompt: str, system: str | None = None, max_retries: int = 6,
             json_only: bool = False, model: str | None = None,
             max_tokens: int = MAX_TOKENS, cache_prefix: str | None = None,
             role: str | None = None) -> str:
    mdl = resolve_model(model)
    sys_prompt = (system or "") + (_JSON_SUFFIX if json_only else "")
    content = ([{"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": prompt}] if cache_prefix else prompt)
    n = max(1, len(_PROJECTS))
    start = next(_rr)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        idx = (start + attempt) % n
        try:
            client = _client_for(idx)
            kwargs = {"model": mdl, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": content}]}
            if sys_prompt:
                kwargs["system"] = sys_prompt
            if BACKEND == "deepseek":
                if role in THINKING_ROLES:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
                    kwargs["max_tokens"] = max_tokens + THINKING_BUDGET
                else:
                    kwargs["thinking"] = {"type": "disabled"}
            resp = client.messages.create(**kwargs)
            _record(mdl, resp, role)
            text = "".join(b.text for b in resp.content
                           if getattr(b, "type", "") == "text")
            if text:
                return extract_json(text) if json_only else text
            last_err = RuntimeError("empty response")
        except Exception as e:  # noqa: BLE001 - rotate to the other project
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"generate failed after {max_retries} tries: {last_err}")


if __name__ == "__main__":
    print(generate("Reply with exactly: OK", max_tokens=16), usage())
