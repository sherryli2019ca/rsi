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

## Addendum 1 (2026-10-08 ~04:25Z, before any ground truth was computed on the main set): ground truth protocol 3 and the metrics that follow from it

The methods had started on the main set (03:35Z); no ground truth had been run
on it, and none of the methods sees ground truth.

**Pilot evidence** (attrib_runs/pilot, the 20 pilot failures; attrib/pilot_report.py):

- Protocol 1 (earliest step whose single oracle correction beats the null
  replays by 2 of 4), four independent repetitions: two repetitions named the
  same decisive step in 42% of failure pairs (50/120), and in 23% of the pairs
  where either found one. Most disagreements came from the oracle (a step
  judged ok in one repetition, or a correction that worked in one repetition
  and not in the other), not from the null replays: a failure usually has
  several steps whose correction rescues it, and which of them is found first
  depends on the corrections the oracle happens to write.
- Protocol 2 (three oracle samples per step plus a confirmation stage) was
  stopped part-way to free memory and is not analysed.
- Protocol 3 (below), two independent repetitions: step-level rescue gains
  correlate .74 (retail) and .64 (airline) between repetitions, and .75/.67
  with protocol 1's gains pooled over its four repetitions. The decisive step
  agreed in 15 of 20 failures; where both repetitions found one, in 7 of 8
  (the eighth differed by one step); the other disagreements were one
  repetition finding no step with a gain of at least 0.5. Cost: 0.054 dollars
  per failure.

**Ground truth (replaces the protocol choice and the curtailed scan above):**
protocol 3, attrib/groundtruth3.py at the commit adding this addendum. For
every agent step k: K = 4 oracle samples (the oracle of protocol 1); 2
corrected replays per sample that calls the step a mistake; 4 null replays if
any sample does. Rescue gain R_k = mean over the 4 samples of (corrected
success rate - null success rate), 0 for a sample judging the step ok or
giving an action the replay cannot apply. Every step is scanned. Decisive step:
the step with the largest R_k if it is at least 0.5 (ties: earliest), else
none; a failure with a decisive step is "rescuable". The earliest step with
R_k >= 0.5 is also recorded. Retest: a seeded 10% of the failures
(Random("retest:0"); retail 9, airline 10) get a second, independent run;
their agreement is reported.

**Metrics (replace the metrics above where they differ):**

- Primary metric: the rescue gain at the step a method names, R(k-hat),
  averaged over rescuable failures. A method naming no valid step scores 0.
- Secondary: R(k-hat) over all failures; the share of the largest gain it
  achieves (R(k-hat) / max R) on rescuable failures; exact accuracy against the
  decisive step and within one step; accuracy against the earliest step with
  R >= 0.5; component accuracy and cost as above.
- Primary comparison: counterfactual search (budget 40) vs the all-at-once
  judge (pro), the paired difference in R(k-hat) over rescuable failures,
  retail and airline pooled: 95% cluster-bootstrap interval (tasks, 10,000
  draws) and a paired permutation test flipping signs by task cluster (10,000
  draws), two-sided alpha .05. McNemar on exact accuracy is secondary.

Budget: the tau2 ground truth is now about 11 dollars (205 runs of 0.054).

## Operations log

- 2026-10-08 03:35Z: registered; methods start on the main tau2 set.
- 03:52Z: a memory watchdog (stops attribution replays when available memory
  falls below 1.7 GB; they are resume-safe) stopped the pilot's protocol-3
  replays once; the gaps were filled by resuming before the pilot analysis.
