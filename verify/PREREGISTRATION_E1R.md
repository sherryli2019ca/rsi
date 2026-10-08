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

## Addendum (2026-10-08, before any r4 episode): a stronger proposer

Approved by Dr Cao on 2026-10-08 to test whether most candidates hurt because
the search model is weak. One more trajectory per domain, r4, identical to r2
and r3 except that the proposer, analyst, digester and critic run deepseek-v4-pro
with thinking enabled (`RRSI_THINKING_ROLES=proposer,analyst,digester,critic`,
budget 8000 tokens); the judge, which is evidence rather than part of the loop,
keeps thinking disabled. r4 is excluded from the primary analysis above, which
stays on r2 and r3. Comparison (descriptive, r4 against r1-r3, whose spread
gives the variation between trajectories): candidates' mean deployment gain and
share with a positive gain, the critic's rejection rate, the share of measured
candidates that fail the error check, transfer of the final harness, and the
decision values of the rules. Quote about 45-50 US dollars; stop and report
past 75.

## Operations log

- 2026-10-08 01:20-01:24Z: round 2 of r2 (retail) ended with an exception in
  RRSI's digester, which received a digest that was a string, and the driver
  stopped after three rounds that could not settle. Commit 47ef920 sends such a
  reply back to the digester as an error and drops report entries that are not
  objects; it changes nothing else and was applied to every run worktree.
- 2026-10-08 about 01:26Z: the container restarted and stopped every run. All
  runs resumed at 01:31Z from their last settled round. A resumed round reuses
  the analysis report, drafted candidates and finished evaluations it had
  already written, so no completed step was sampled again. No held-out success
  rate was examined.

## Addendum 2 (2026-10-08, before any E1R or E3 decision value was computed): sequential verification

Proposed by Dr Cao on 2026-10-08: find a suitable candidate with as few
episodes as possible instead of a fixed budget. Two rules are added to the
analyses of r2 and r3 (r4 descriptively) and of E3. Both are defined in
`verify/posthoc_e1.py` (`SEQ_RULES`, `p_admissible`) and run under the common
error check.

- seqfull: RRSI's full evaluation of each candidate, run in batches of 10
  episodes in random order. After each batch a candidate is dropped when its
  harness errors already exceed 2% of its planned episodes (the guard can no
  longer pass) or when the predictive probability that RRSI's own rule admits
  it after the full evaluation is below gamma = 0.05. A candidate that is not
  dropped completes the evaluation and is decided by RRSI's rule; seqfull
  admits only candidates that RRSI's rule admits.
- seqsample@80: the same, planned at 80 paired episodes, decided at the end by
  the sample rule.
- Predictive probability: the final dS is normal with mean the running paired
  difference m and variance ((N-n)/N)^2 s^2 (1/n + 1/(N-n)) after n of N
  episodes, where s^2 is the larger of the running variance of the paired
  differences and S_t(1-S_t); dC is held at its running estimate and the
  novelty count is known; the floor on dS is RRSI's (S* - delta - S_t) for
  seqfull and minus twice the final standard error for seqsample@80.
- gamma = 0.01 and 0.10 are sensitivity analyses.

Hypotheses (one family, Holm over two, one-sided alpha 0.05, the block
bootstrap of the primary analysis, the 80 new rounds): seqfull and
seqsample@80 are each non-inferior to full evaluation in decision value, with
a margin of 0.75 points per round. Reported with them: episodes per round and
their ratio to full evaluation's, the contrasts with sample@40 and sample@80,
and Net(10^4). E3 uses the same hypotheses on its 20 rounds.

These rules were developed on E1, whose results were known, so E1 is their
exploratory set. On E1 (40 rounds, common error check): seqfull +0.15 points
per round [-0.05, +0.42] at 65 episodes per round, seqsample@80 +0.12
[-0.19, +0.51] at 43, against full evaluation's +0.22 at 123; full minus
seqfull +0.07 [-0.03, +0.24]. With gamma = 0 seqfull reproduces every decision
of full evaluation.

## Operations log, continued

- 2026-10-08 between about 06:31Z and 07:10Z: the container stopped again
  (likely reclaimed while the session was idle). Every run resumed at 07:12Z
  from its last settled step, as at 01:26Z; follower logs were kept
  (follow.before_0712Z.log).
- 2026-10-08 07:12-07:25Z, progress checks: the last lines of the r2, r3 and
  r4 follower logs were displayed while checking the restart, and they include
  the held-out success rates of a few rounds' candidates and incumbents
  (r2 retail r10, r2 airline r12, r3 retail r9, r3 airline r15, r4 r1).
  Nothing was decided from them; the analysis was fixed before (this file,
  verify/e1r.py at commit 42731dd, tested on E1 only). Later checks filter
  these values out.
