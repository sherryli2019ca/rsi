# Pre-registration: two further RRSI trajectories per domain (experiment E1R)

Approved by Dr Cao on 2026-10-08 after the sixth review, and registered before
any episode of E1R was run. E1R repeats experiment E1 (`verify/PREREGISTRATION.md`)
to answer three questions E1 could not: whether its decision values hold on
independently generated trajectories, how much they depend on rounds that share
an incumbent, and whether RRSI's component labels, which in E1 called most code
changes prompt edits, changed the candidates the loop proposed.

## Design

- Two new trajectories per domain (tau2 retail and airline), named r2 and r3;
  E1's trajectory is r1. Each runs `verify/run_e1.sh` from its own worktree
  pinned to the commit that adds this file, with RRSI's branches under
  `evolve/<name>/`.
- Everything is as registered for E1 (loop, models, temperatures, tasks,
  evidence, budgets, held-out deployment of every measured candidate and
  incumbent with 240 episodes per harness, base and final harnesses on 400
  (retail) and 240 (airline)), with one change: the corrected component
  signals. The domains' signals now see git's python diff driver
  (`domains/.gitattributes`) and label code in the agent module or in a new
  harness module as control flow instead of letting it fall through to
  prompt. Offline, on E1's 47 measured candidates, this labels 24 as control
  flow and 18 as prompt (E1: 2 and 38), and no candidate that edits code is
  labelled prompt only (E1: 21 of 31).
- Trajectories differ because the proposer, analyst and critic sample at the
  endpoint's default temperature; nothing else is reseeded.

## Primary analysis (the 80 new rounds)

Decision values D under the common error check (a rule that runs episodes
drops a candidate when over 2% of that candidate's episodes in its draw ended
in a harness error; `verify/posthoc_e1.py guard`, as reported in the paper),
for full evaluation, sample@40, net@40 and replay-null@40. Hypotheses: each
@40 rule is non-inferior to full evaluation with a margin of 0.75 points per
round, one-sided alpha 0.05, Holm-adjusted over the three rules. Intervals
come from a bootstrap that resamples, within domain and trajectory, blocks of
rounds that share an incumbent, and held-out tasks (B = 2000, M = 400 evidence
subsamples, as in E1).

## Secondary

- The registered E1 analysis (`verify/analyze.py`, unchanged) on each new
  trajectory.
- All three trajectories pooled (120 rounds), with the same block bootstrap,
  and each trajectory's own D, so that variation between trajectories is
  reported next to the pooled estimate.
- The check-only baselines (nonecheck@10, judgecheck@10), the other budgets,
  pairwise contrasts, leave-one-block-out ranges, minimum detectable
  differences at 80% power, and Net(10^4) with break-even horizons, as in
  `verify/posthoc_e1.py robust`.
- Component labels (descriptive, no test): for r1 against r2 and r3, the mix
  of measured candidates by corrected label and by files touched, the standard
  deviation of candidates' deployment gains, the share whose gain exceeds the
  single-candidate detectable effect, and the split-half reliability of gains
  without candidates that fail the error check.

## Operations

- No held-out success rate is examined before all rounds of a trajectory are
  deployed, apart from completion and error checks; anything seen earlier is
  reported.
- Quote about 42,000 episodes and 75 US dollars at third-party prices. If
  spending passes 110 dollars we stop and report.
