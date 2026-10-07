# Pre-registration: verification evidence on RRSI rounds (experiment E1)

Registered before any held-out episode of E1 was run. The analysis code is
`verify/analyze.py` at the commit that adds this file; it is not changed after
the first held-out episode except to fix a bug, and any such fix is reported.

## Question

When a self-improvement loop proposes a change to an agent harness, which
verification evidence predicts the change's deployment gain on independent
tasks, and is that evidence worth its cost?

## Setting

- Self-improvement loop: RRSI (google-research/rrsi, vendored in `rrsi/`),
  T = 20 rounds, m = 2 candidates per round, k = 2 trials per evolve task,
  proposer / analyst / critic / judge: deepseek-v4-pro (thinking off).
- Agent: deepseek-v4-flash (thinking off), temperature 0, user simulator the
  same model at temperature 0; starting harness = tau2-bench's LLM agent prompt.
- Domains: tau2-bench retail (evolve = train split, 74 tasks; held-out = test
  split, 40 tasks) and airline (30 / 20). All failures are natural (no fault
  injection).
- Reward: final database equals the gold one and every communicate_info
  string was said (tau2's DB and COMMUNICATE checks; NL assertions changed no
  outcome in the harness check P0).
- RRSI's noise band delta is calibrated from three evaluations of the base
  harness on the evolve set.

## Evidence per candidate (budget b = episodes per candidate)

none; LLM judge (0 episodes); sample@b; replay@b; replaynull@b; net@b; full
(RRSI's own decision). Definitions, the decision rule and the scoring are in
the docstring of `verify/analyze.py`. b in {10, 40, 80}. Replay references: one
failed trial per failed evolve task of the incumbent (at most 20), start step
from the round's failure digest (else the first non-gold state-changing call,
else 0); 4 replays per reference per candidate and 4 null replays per
reference with the incumbent.

## Ground truth

Every incumbent and every measured candidate runs from the first step on the
held-out tasks: 4 trials per task (retail) and 8 (airline). The base and the
final incumbent get 10 trials per task. dep(c) = held-out success of c minus
that of its incumbent, paired by task.

## Primary endpoints

1. Transfer: held-out success of the final incumbent minus the base harness,
   pooled over the 60 held-out tasks (90% bootstrap interval over tasks),
   reported next to the same difference on the evolve set.
2. Decision value at b = 40: the per-round deployment gain of the choice each
   rule makes (0 when it keeps the incumbent), averaged over all rounds of both
   domains. Contrasts: net@40 - full (non-inferiority, margin 0.75 percentage
   points per round), net@40 - none and full - none (superiority). One-sided
   bootstrap p-values (B = 2000; rounds and held-out tasks resampled within
   domain), Holm-adjusted over the three, alpha = 0.05.

Power (planning assumptions): within-task variance 0.07 (measured at
temperature 0 in P0), 160 held-out episodes per harness, rules disagreeing in
40% of rounds: standard error of the mean round difference about 0.30 points,
power about 80% at the 0.75-point margin. If the observed disagreement rate
(known before the held-out data) exceeds 40%, held-out trials per harness are
increased so that this standard error stays at or below 0.30 points, before
any held-out result is looked at.

## Secondary

Correlation and calibration of each evidence estimate with dep (corrected
for held-out measurement reliability by split halves); decision value at
budgets 10 / 40 / 80; cost per round in episodes and token equivalents;
Net(N) = N(v dp - dc_run) - C_verify - dC_change; net@40 under Beta priors on
the candidate fix rate with mean 0.41 / 0.29 (audited analyst accuracy,
retail / airline), 0.7 and 0.2; how often a candidate's code moves the replay
start earlier, by component.

## Not changed after registration

Domains, splits, models and temperatures, T, m, k, budgets, evidence
definitions, the decision rule, z = 2, the held-out trial counts (except the
pre-specified increase above), the margin and the endpoints.

## Changes after registration

- 2026-10-07, before any held-out episode of E1 had run: in
  `verify/analyze.py`, the failure share f of the replay estimators was
  computed from the replay references only, which are capped at 20 tasks, so
  it fell short of the registered definition (failed trials / trials of the
  incumbent's evaluation) whenever more than 20 evolve tasks failed. It now
  follows the definition.
- Same day, also before any held-out episode: the registered secondary
  Net(N) was listed but not yet computed; `net_values` in `verify/analyze.py`
  now computes it per rule as the mean over rounds of N (v dep - dc_run) -
  C_verify, in episodes of running cost (dc_run: the chosen harness's relative
  change in policy tokens per held-out episode; C_verify: the rule's episode
  equivalents; dC_change = 0, since candidate generation is the same for every
  rule), for N in {100, 1000, 10000} and v in {1, 10}. Nothing else changed.
- 15:40Z, power rule: the evolve-side disagreement rate between net@40 and
  full on the six rounds with collected evidence (retail 2, airline 4) is 0.58,
  above the registered 0.40, which calls for about 230 held-out episodes per
  harness instead of 160. The follower's progress log prints each deployed
  harness's held-out success rate, and a few of these lines were seen while
  monitoring the run; no held-out analysis was run, and the decision uses only
  the registered formula on evolve-side evidence. The increase is put to the
  project owner because it adds cost to the approved run; the outcome will be
  recorded here.
