"""A deterministic stand-in for the LLM, used only to test the plumbing."""
import random
from collections import defaultdict
from types import SimpleNamespace as NS


class MockLLM:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.tokens = defaultdict(int)
        self.n = 0

    def agent_step(self, system, tools, messages, purpose="agent"):
        self.tokens[purpose] += 100
        self.n += 1
        steps = sum(1 for m in messages if m["role"] == "assistant")
        if steps < 2:
            content = [NS(type="tool_use", id=f"tu{self.n}", name="get_policy",
                          input={"name": "refund"})]
        else:
            ans = "0.00" if self.rng.random() < 0.5 else "1"
            content = [NS(type="text", text=f"FINAL: {ans}")]
        return NS(content=content, usage=NS(input_tokens=80, output_tokens=20))

    def ask_json(self, prompt, schema, purpose, max_tokens=8000):
        self.tokens[purpose] += 50
        props = schema["properties"]
        if "root_cause" in props:
            return {"root_cause": self.rng.choice(["wrong unit", "missing pages", "rounding"]),
                    "step": 1, "component": self.rng.choice(props["component"]["enum"])}
        if "categories" in props:
            return {"categories": [{"name": "unit", "definition": "unit errors"},
                                   {"name": "paging", "definition": "pagination"}]}
        if "category" in props:
            return {"category": self.rng.choice(props["category"]["enum"])}
        if "patches" in props:
            return {"patches": ["patched A", "patched B", "patched C"]}
        if "fixed" in props:
            return {"fixed": self.rng.random() < 0.5}
        raise ValueError(schema)
