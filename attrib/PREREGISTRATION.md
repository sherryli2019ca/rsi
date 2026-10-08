# Pre-registration: evaluating failure attribution (Phase A, tau2)

Written 2026-10-08, before any attribution method was run on the main set and
before any ground truth was computed on it. Plan approved by Dr Cao (Phase A,
02:44Z; cost-saving version, 03:28Z). Code: attrib/ at the commit that adds
this file.

## Question

Which attribution method best finds the decisive step of a failed agent
episode, and at what cost? (Q1 accuracy, Q2 cost. Q3, the downstream value of
attributions for repair, is Phase B and is registered separately.)

## Failures

- tau2 retail and airline, from E1 (RRSI reference trajectories, commit
  c2ffaa3): failed trials of every incumbent, in its evolve evaluation and in
  its held-out deployment (attrib/pool.py: reward < 1, no harness error, at
  least one step; at most 2 trials per task and harness; tasks round-robin in
  a seeded order; seed 0).
- Pilot failures (attrib_runs/pilot, 10 per domain) are excluded; they were
  used to choose the ground-truth protocol.
- Airline: 100 failures (24 tasks). Retail: the E1 pool yields only 86 under
  the cap (32 tasks); the remaining 14 come from the E1R incumbents' failures
  with the same rule once E1R ends. Analyses are run on the 86 + 14.
- AppWorld (100 failures from E3) follows the same protocol after E3 ends;
  registered in an addendum before it starts.

## Methods (attrib/methods.py, attrib/search.py; prompts and models fixed by this commit)

Every method reads the failure as RRSI's analyst reads it (customer's hidden
instructions, conversation, grading section), except counterfactual search,
which sees no grading. Each outputs one step (and, where it can, a component
from RRSI's vocabulary).

| Method | Model | Calls / replays |
|---|---|---|
| last step, first write (baselines) | none | 0 |
| all-at-once judge (Who&When) | flash; pro; pro with thinking | 1 call |
| step-by-step judge (Who&When) | pro | 1 call per step until "decisive" |
| binary-search judge (Who&When) | pro | about log2(steps) calls |
| Study 1 attributor | pro | 1 call |
| RRSI trace digester (failure lens), earliest cited step | pro | multi-turn |
| counterfactual search | pro + replays | 1 call + up to 40 replays |

Counterfactual search answers under budgets 8/16/24/32/40 are read off the
same run (attrib.search.answer_at). A method that returns no valid step is
scored wrong.

## Ground truth

Counterfactual replay with null control (attrib/groundtruth.py, protocol 1;
attrib/groundtruth2.py, protocol 2). The protocol for the main set is chosen on
the pilot by test-retest agreement of the decisive step and fixed in
Addendum 1 below before the main ground truth is run. Scan: curtailed (steps in
order, stop at the first flip), with a seeded 20% of failures scanned fully.
A further seeded 10% of failures get a second, independent ground truth run;
their agreement is reported as the ceiling for every method's accuracy.

## Metrics and analysis

- Primary metric: step accuracy, the share of failures whose method step equals
  the decisive step, over failures with a decisive step. The share of failures
  without one is reported per domain.
- Secondary: within one step; hits any flipping step (fully scanned failures
  only); component accuracy on failures with a component ground truth (an E1
  candidate whose targeted replays of this trial beat the null replays by 2 or
  more; its component by files touched); cost per attribution in dollars (list
  prices as in verify/posthoc_e1.py) and the accuracy-cost frontier.
- Primary comparison: counterfactual search (budget 40) vs the all-at-once
  judge (pro), exact step accuracy, McNemar exact test, two-sided alpha .05,
  retail and airline pooled.
- Everything else: estimates with 95% cluster-bootstrap intervals (resampling
  tasks; 10,000 draws), no further tests.

## Budget

Phase A stop line: 100 US dollars (all parts). Expected for tau2: methods about
8, counterfactual search about 5, ground truth about 10.

## Operations log

- 2026-10-08 03:35Z: registered; methods start on the main tau2 set.
