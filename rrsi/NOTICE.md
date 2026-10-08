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
  explicitly, except for the roles listed in `RRSI_THINKING_ROLES` (budget
  `RRSI_THINKING_BUDGET`, default 8000 tokens); per-model and per-role token usage is accounted (`usage()`,
  optional JSONL log at `RRSI_USAGE_LOG`).
- `rrsi/analyst.py`, `rrsi/digester.py`, `rrsi/propose.py`, `rrsi/critic.py`:
  every `generate(...)` call is tagged with its role for the cost accounting.
- `rrsi/propose.py`: a `done()` with no file changes and no edits is bounced like
  an `abort` (at most 3 times) instead of ending the draft with no candidate.
- `rrsi.py`: the default `--runs` directory is `runs/rrsi`; options
  `--selection` and `--branch-ns` (below).
- `rrsi/config.py`, `rrsi/loop.py`: two configuration keys. `selection`
  (default `full`, RRSI's own steps 5-6, unchanged) can instead select each
  round's winner with one verification evidence type at a fixed episode budget
  (`none`, `judge`, `sample@b`, `replay@b`, `replaynull@b`, `net@b`;
  implemented in `verify/live.py`, which then evaluates only the winner on the
  full evolve set); `readjudicate` and `reevaluate` apply to `full` only.
  `branch_ns` (default empty, upstream branch names) namespaces the git
  branches so that several runs can share one repository. In every mode a round
  writes its selection cost to `r<t>/selection.json` in the same units.
- Upstream domains (coding, eng, workspace) are not included. This repository
  adds `domains/tau2/` (shared tau2-bench adapter, frozen episode driver with
  mid-step replay, starting harness) and `domains/tau2_retail/`,
  `domains/tau2_airline/` (configs, constitutions, starting harnesses), written
  against RRSI's `Domain` interface, and `domains/appworld/` (frozen AppWorld
  driver with mid-step replay, adapter, starting harness, configuration).
