# Origin of this code

`rrsi/`, `rrsi.py` and `tests/test_rrsi_core.py` are taken from
google-research/rrsi (Regularized Recursive Self-Improvement), commit
be50316e1db05914068a973f322770ef08ed7ba1, licensed under the Apache License,
Version 2.0 (`rrsi/LICENSE`). The upstream copyright headers are kept in every
file.

Modifications made in this repository:

- `rrsi/llm.py`: rewritten backend. Calls go to DeepSeek's Anthropic-compatible
  endpoint by default (`RRSI_LLM_BACKEND=deepseek`; the Vertex backend is kept
  for `RRSI_LLM_BACKEND=vertex`); upstream Claude model names in configs are
  mapped to `RRSI_MODEL` (default `deepseek-v4-pro`); thinking is disabled
  explicitly; per-model and per-role token usage is accounted (`usage()`,
  optional JSONL log at `RRSI_USAGE_LOG`).
- `rrsi/analyst.py`, `rrsi/digester.py`, `rrsi/propose.py`, `rrsi/critic.py`:
  every `generate(...)` call is tagged with its role for the cost accounting.
- `rrsi.py`: the default `--runs` directory is `runs/rrsi`.
- Upstream domains (coding, eng, workspace) are not included. This repository
  adds `domains/tau2/` (shared tau2-bench adapter, frozen episode driver with
  mid-step replay, starting harness) and `domains/tau2_retail/`,
  `domains/tau2_airline/` (configs, constitutions, starting harnesses), written
  against RRSI's `Domain` interface.
