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

- 2026-10-11 05:55Z, after the labels arrived and before any score was
  computed with them: the audit sheet (`sheet`) rendered the conversation
  with the oracle's view, which cuts every tool result at 1,200 characters;
  this registration says the items show the conversation as the agent had
  it. 98 of the 159 items had at least one cut tool result. A mechanical
  check found one correction (V096) that uses a value visible only in the
  full output (both auditors judged it invalid on confirmation); auditors
  noted the cut on 8 items. `fullsheet` writes the 98 items with full tool
  results (audit/items_full_98.md). How the labels are corrected is for Dr
  Cao to decide; the labels as first returned are kept.
- 2026-10-11 08:23Z, Dr Cao: no recheck; the labels are used as first
  returned (nothing relabelled with the full tool results). The 10 items
  whose valid labels differ are not adjudicated, against "v from the
  adjudicated labels" above. Every score is instead reported twice, with
  those items counted invalid (strict: valid only if both files accept) and
  valid (lenient: either accepts); any adjudication lies between the two.
  Results: results/attrib/validity_analysis.json (`analyze` once per bound;
  each file's own labels as auditor1 and auditor2).

## Addendum 1 (2026-10-11, before any rerun): replays with a faithful restoration

Approved by Dr Cao at 13:41Z ("重跑"), after external review 9 and the
restoration diagnosis in the operations log (~13:30Z entry). Written and
committed before anything below is run.

Cause. Restoring a failure's prefix re-runs each recorded call through the
harness that produced it. Ten harnesses ask an LLM to check writes in
before_tool; asked again, the check can decide differently from the
recording, and the driver then went live before the replayed step. In this
study the harness is always the recording's own, so the recorded decision is
the right one.

Fix. domains/tau2/driver.py, opt-in environment flag
TAU2_REPLAY_RECORDED_CHECKS=1 (off by default, so paper 1's replays under
changed harnesses are unaffected). For each recorded prefix call, before_tool
still runs (harness state), but the recorded decision is kept: a call the
recording blocked is not executed; any other call is executed on the fresh
database (render_result runs) and the recorded observation is shown. The
episode goes live early only if an executed call returns a different raw
result than the recording ("state" divergence, kept in replay.diverged).
Replays whose restoration did not diverge are unchanged by construction
(same prefix, same observations, same live start) and are not rerun.

Scope (retail only; airline and the main set had no or negligible
divergence and are not rerun):
- the gt replays whose replay.diverged is non-empty: 130 corrected and 80
  null replays (tau2_retail/gt/replays);
- aligned search's test replays whose replay.diverged is non-empty: 328 of
  1,800 (tau2_retail/search_maxgain/aligned/replays).
Before the rerun the whole run directory is archived
(archive/attrib-validity-runs-run1.tar.gz) and the result files of the first
run are kept beside the new ones; the diverged replays are moved, not
deleted, to a superseded/ folder next to each replays folder. The same
commands then run with the flag set: `search_maxgain.run` (aligned, retail)
re-runs the moved tests and recomputes its picks; `validity.oracle` (retail)
runs the oracle at any union step that is new, and every missing replay
(the moved ones and those at new steps), then rewrites gt/result.json.
Nothing else changes: same failures, methods, oracle prompt, K, R_C, N_NULL,
seeds and temperature.

Labels. An outcome-changing correction whose item (domain, failure, step,
normalized action; item_key in attrib/validity.py) was already labelled
keeps both auditors' labels. New items go to the same two auditors, shuffled
and with the same blinding, with every tool result in full, as a separate
sheet (audit/items_add.md, labels_template_add.csv; key_add.json withheld).
Until those labels return, each score is reported with the new items counted
invalid and counted valid, within the strict and lenient bounds above; after,
the strict and lenient bounds as above.

Reporting. The rerun is the primary result for the validity check; the first
run's numbers are kept and reported beside it. Same analysis as above, and the
post hoc analyses of attrib/validity_posthoc.py, whichever way they come out.

Budget. About 1 to 3 dollars; stop at 5.

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
- 2026-10-11 02:41Z: aligned search, oracle and replays finished with no
  missing replays and no harness errors; spend 3.87 dollars (plus 0.10 for
  the smoke test). Oracle: 237 union steps (retail 173, airline 64). 116
  retail corrected replays were not applied because the harness blocked an
  earlier write while the state was restored, which moved the start earlier;
  as in the main set, a sample without both corrected replays applied counts
  as no correction. `sheet`: 215 outcome-changing corrections, 159 distinct
  items (114 retail, 45 airline), sent to the auditors with the criteria in
  English and Chinese; the key (audit/key.json) is withheld from them. Runs
  archived as archive/attrib-validity-runs.tar.gz.
- 2026-10-11 05:43Z: Dr Cao sent two label files as the two independent
  auditors' results: auditor_codex_lenient_A.csv (md5 19c034b8...) and
  labels_C.csv (md5 88600d8d...), kept as received in
  attrib/validity_labels/. Both cover all 159 items with yes/no answers.
  Agreement before adjudication: valid 149 of 159 (kappa .79);
  information .93 (.75), policy .89 (.57), confirmation .99 (.94), scope .89
  (0; one file answers no 17 times, the other never). Valid: 128 and 132.
  10 items to adjudicate (audit/adjudication.csv). No method score has been
  computed with these labels.
- 2026-10-11 08:23Z onward: `analyze` failed on the returned files' byte
  order mark; `_labels` now reads them as utf-8-sig (no value changes).
  Scores computed for both bounds (deviation above), then written into the
  paper (Table tab:validity, abstract, section 5, conclusion, limitations,
  registration appendix). Over all 80 failures: first write raw 0.241,
  valid 0.128 (strict) and 0.134 (lenient); validity rate of its
  outcome-changing corrections .57 and .59, other methods .59 to .90; first
  write minus aligned search on valid score -0.027 [-0.093, +0.037] (strict)
  and -0.025 [-0.089, +0.036] (lenient).
- 2026-10-11 ~12:55Z, after a self-review of the paper: post hoc analyses,
  not registered, in attrib/validity_posthoc.py (results/attrib/
  validity_posthoc.json): the paired change of (first write - m) from raw to
  valid score, validity-rate differences, valid scores minus the
  second-to-last step's, validity by kind of the observed step, where the 58
  unapplied samples fall (only at steps after a write: second-to-last 45 of
  188 samples, all-at-once 21 of 180, binary search 12 of 175, first write
  and aligned search none) with scores without the 13 failures concerned,
  and scores without the scope criterion (unchanged). Written into the
  paper (v22) and marked post hoc there.
- 2026-10-11 ~13:30Z, after external review 9 asked why the 116 retail
  corrected replays were not applied: the cause is the replay driver's
  restoration, not the corrections. Nine of the 33 retail harnesses (24 of
  the 60 retail failures) and one airline harness ask an LLM to check each
  write in before_tool. Restoring a prefix re-runs this check on every
  recorded write; domains/tau2/driver.py treats any block or observation that
  differs from the recording as a divergence and goes live from there. On
  130 of 736 retail corrected replays and 80 of 488 null replays the check,
  which had blocked the recorded write, blocked it again with other wording
  (100 of the 130) or let it through (30); 116 of the corrected replays
  diverged before their step and were not applied. Aligned search's own
  tests: 328 of 1,800 retail replays diverged. In the main set: 10 of 3,118
  retail corrected replays not applied, 12 of 2,516 null replays diverged.
  Post hoc, without the 14 failures with such a replay at a named step, no
  conclusion changes (attrib/validity_posthoc.py, "restoration" and
  "*_without_diverged"). Nothing was rerun; a rerun with the recorded check
  decisions replayed would need Dr Cao's approval and labels for any new
  outcome-changing corrections.
- 2026-10-11 13:47-13:57Z, Addendum 1 (commit ffbfb79 before any rerun):
  the first run was archived (archive/attrib-validity-runs-run1.tar.gz) and
  its 130 corrected and 80 null gt replays and 328 aligned-search tests whose
  restoration diverged moved to superseded/. Smoke test with the flag (one
  corrected, one null replay of r2_600d0fc_e_103_s1 at step 11, output not
  used): both restored without divergence, the correction applied. Aligned
  search, rerun with the flag: all 328 tests restored without divergence;
  its pick changed on 6 of 60 retail failures. Oracle at the 2 new union
  steps (173 retail union steps, as before) and every missing replay: 224
  gt replays rerun, none diverged; no sample is left unapplied (58 in the
  first run). Gains changed for 24 samples. Outcome-changing corrections:
  234 samples, 170 distinct items, of which 157 were labelled in the first
  sheet and 13 (all retail) are new; 2 labelled items no longer change the
  outcome. The 13 went to the two auditors (audit/add/items_add.md, every
  tool result in full; key audit/key_add.json withheld). Spend 0.49 dollars
  (552 replays; oracle 0.001). Results with the 13 counted invalid and
  valid: results/attrib/validity_rerun.json (attrib/validity_rerun.py). Raw
  R over all 80 failures: first write 0.241 (unchanged), second-to-last
  0.245 (0.220 in the first run), all-at-once 0.180 (0.144), binary search
  0.216 (0.191), aligned search 0.231 (0.207). Valid R, first write minus
  aligned search: -0.024 [-0.091, +0.039] to -0.051 [-0.125, +0.019] across
  the bounds.
