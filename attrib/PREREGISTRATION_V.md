# Pre-registration: are rescuing corrections valid? (fresh tau2 failures)

Written 2026-10-11, before any failure of this set was drawn, before any
method, oracle or replay was run on it, and before any correction was
labelled. Approved by Dr Cao on the decision card of 01:42Z (option "只标改变
结果的": label only the corrections that change the outcome). Budget about 7
dollars, stop at 15. Code: attrib/validity.py at the commit that adds this
file; methods, prompts, oracle and replays are the code of the main set
(attrib/methods.py, attrib/search_maxgain.py, attrib/groundtruth3.py,
attrib/review4_runs.py) at this commit. Phase A is registered in
attrib/PREREGISTRATION.md; everything about the main set that this file
refers to as "aligned" was found post hoc on the main set and is fixed here
unchanged.

## Questions

V1. Of the corrections with which the grading-aware oracle changes the outcome
at the steps attribution methods name, how many are valid under criteria
fixed in advance?

V2. Scored by valid rescue gain on fresh failures, does the first-write rule
keep an advantage over the judges and search aligned with the metric?

No direction is predicted. A reversal, a tie or a kept lead are all reported
as found.

## Failures

Fresh failed tau2 episodes of agents built by RRSI, from the E1R runs r2 and
r3 (archive/e1r2-tau2-runs.tar.gz, e1r3-tau2-runs.tar.gz), on tasks not in
the main set (retail: none of its 36 tasks; airline: none of its 24).

Pool: every failed trial (reward < 1, no harness error, at least one step) of
(a) every candidate harness's full evaluation on training tasks (jobs r<t>A,
r<t>B; not smoke tests), (b) the base harness's evaluation, and (c) every
held-out deployment of an incumbent; minus the evolve jobs of the eight
repair-experiment states (attrib/PREREGISTRATION_B.md), whose failing traces
were analysed post hoc. Using incumbents only, as the main set did, leaves 53
retail and 24 airline failures on 25 tasks, too few tasks; candidates are
also agents the loop built, but this population differs from the main set's
and the paper says so.

Sample: at most 2 trials per (task, harness); tasks in a seeded order
(seed 0), round-robin, one trial per task per pass (attrib.pool.sample), until
60 retail and 20 airline failures. The harness of each failure is restored
from archive/rsi-local-branches.bundle at its commit.

## Methods (fixed; as in the main set's post hoc analyses)

- first write: the first step calling a state-changing tool, else the last step.
- second-to-last step: step n - 2 (position baseline).
- all-at-once judge asked for the gain (`all_at_once_gain:pro`).
- binary search asked for the gain (`binary_search_gain:pro`).
- aligned counterfactual search (`attrib.search_maxgain`, variant aligned):
  suspects from the informed prompt asked for the gain, every suspect tested
  with 4 corrected and 4 null replays, answer = largest estimated gain.

Each method names one step per failure. A method that returns no valid step
scores 0 on that failure.

## Interventions

At every step named by any of the five methods (the union), the registered
oracle of the main set (DeepSeek-V4-Pro, thinking off, reads the grading;
attrib.groundtruth3 prompt) gives K = 4 samples; each sample that calls the
step a mistake with an applicable action gets 2 corrected replays, and each
step with at least one such sample gets 4 null replays. Raw gain of sample i
at step k: c_{k,i} (p_corr_{k,i} - p_null_k); R_k = (1/K) sum_i of it. No full
profiles are computed and no failure is selected by its gains.

## Validity labels

Items: the distinct corrections that change the outcome, i.e. sample i at step
k with c_{k,i} = 1 and p_corr_{k,i} != p_null_k; corrections with the same
failure, step and action (normalised JSON of the forced turn) are one item.
Corrections with p_corr = p_null contribute 0 to every score whatever their
validity and are not labelled.

Each item shows: the domain policy, the conversation up to step k (customer
messages, agent turns, tool calls and results), the agent's original action
at step k, and the corrected action. Hidden: the customer's hidden
instructions, the grading and gold actions, the oracle's reason, the replay
outcomes and gains, which methods named the step, and the item's failure id.
Items are shuffled (seed 0).

A correction is valid only if all four hold (answered yes / no each):

1. Information. Every value and every choice in the corrected action is
   supported by what the agent had at that step: the conversation, the tool
   results so far and the policy. Identifiers, amounts or facts that appear
   only later, and choices the customer has not made and the policy does not
   determine (for example which of two matching items, when the customer has
   not said), are not supported. Asking the customer is supported.
2. Policy. The action follows the domain policy and the tools' constraints
   (for example it does not change the database without the explicit
   confirmation the policy requires, and does not transfer when the policy
   says not to).
3. Confirmation. The action does not override the customer's latest valid
   confirmation: it does not change an item, option, quantity, reason,
   address, payment method or other choice the customer most recently
   confirmed, unless the customer later withdrew it or the policy forbids it.
4. Scope. The action is one assistant turn that replaces the action at step k
   and does only what a careful agent would do at that point. A turn with
   several tool calls is in scope only if every call's arguments were known
   before the turn (no call needs another call's result); it may not skip a
   confirmation or lookup the policy requires between them.

Valid: v = 1 if 1-4 are all yes, else 0. These criteria are not changed during
the audit; a case the auditors find unclear is recorded with a note and
decided in adjudication under the same wording.

Two auditors label every item independently, without seeing each other's
labels. Afterwards they adjudicate every item on which their valid labels
differ, by discussion; if they cannot agree, Dr Cao decides. Both independent
label sets and the adjudicated set are kept and released.

## Metrics

R^valid_k = (1/K) sum_i c_{k,i} v_{k,i} (p_corr_{k,i} - p_null_k), with K = 4
kept as the denominator (an invalid correction counts as no correction);
v from the adjudicated labels.

Per method m, over all 80 failures (primary; no selection):
- raw score: mean of R at the named step;
- valid score: mean of R^valid at the named step;
- validity rate: among the labelled corrections at m's named steps (each
  failure, step and sample once), the share valid;
- dollars per failure (methods; the oracle's cost is the reference's, shared).

Secondary: the same on the failures whose largest raw R over the union steps
is at least 0.5; retail and airline separately; the scores under each
auditor's independent labels; agreement (percent and Cohen's kappa) per
criterion and for v before adjudication.

## Analysis

Paired differences first write minus each other method (valid score; also
raw score), with 95% intervals from a cluster bootstrap over tasks (10,000
draws, seed 0). No hypothesis test; nothing is called equivalent. The paper
reports V1 and V2 as a test of the main set's post hoc findings on fresh
failures, with whatever result.

## Deviations

Any change after this commit is logged below with its date and reason.

## Operations log
- 2026-10-11 02:00Z, before this file was committed: a dry run of `sample`
  (deterministic, seed 0; it reads only the traces' rewards and step counts)
  found a pool of 450 retail failures on 55 tasks and 103 airline failures on
  12 tasks; the sample is 60 retail failures on 55 tasks (33 harnesses, 20
  held-out) and 20 airline failures on 12 tasks (16 harnesses, 8 held-out).
  An infrastructure smoke test ran every step (methods, aligned search,
  oracle, replays, sheet, analyze with dummy labels) on a scratch copy with
  one retail and one airline pool failure that are not in the sample
  (r2_4101495_e_2_s0, r2_c69d804_e_11_s1); cost 0.10 dollars; its outputs are
  not used. One null replay on a sampled failure (r3_cf3d60d_e_6_s0, step 9)
  checked the replay driver; its output is not used.
- 2026-10-11 02:03Z: `sample` drew the same 80 failures as the dry run
  (results/attrib/validity/failures_tau2_*.json) and `harness` restored their
  49 harnesses. The three methods ran 02:03-02:05Z (0.46 dollars).
- 02:14Z: to speed up replays, the remaining aligned-search and oracle replays
  run with several harness groups at once (the same refs, keys and driver as
  attrib.groundtruth._drive, which runs the groups one after another);
  nothing about what is run changes. Replays interrupted by the switch had
  not been written and are run again.
