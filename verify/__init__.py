"""Verification-evidence study on top of an RRSI run (experiment E1).

RRSI drafts candidate harness edits and selects them by a full evolve-set
evaluation. For every candidate of a finished RRSI round this package collects
cheaper verification evidence (targeted mid-step replays of the incumbent's
failed trials, null replays with the incumbent, an LLM judge; random and
regression samples are subsampled from RRSI's own full evaluation), measures
the deployment gain of every candidate and incumbent on held-out tasks, and
scores each evidence type by the deployment value of the decisions it would
have made at a given episode budget.

  python -m verify.follow --domain tau2_retail [--runs runs/rrsi]
"""
