"""How the environment's tools are presented to the model. The starting
harness passes them through unchanged."""
from __future__ import annotations


def describe(tools: list[dict]) -> list[dict]:
    return [dict(t) for t in tools]
