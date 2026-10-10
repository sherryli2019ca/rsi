# Pre-registration: blocked loops with shared calibration: the cost-aware stopping rule and what sequential evaluation changes in an RRSI loop (experiment BK)

DRAFT, not yet approved: Dr Cao has been offered this experiment (option 1,
"five arms", of `/mnt/project-files/reviews/review11-plan.md`, which replaces
option B of review10-plan.md; option 2, "rule_fix", keeps arms f, s and c in
six blocks) and has not chosen yet. If it is approved, `DESIGN` in
`verify/bk.py` is set to the chosen design, this file is committed with the
code below before any episode of BK runs, and the approval is recorded here.

The eleventh review asks for a comparison with shared or balanced
calibration and for a live test of the cost-aware stopping rule that the
tenth review's revision tested only offline ("Running it in independent loops
would help establish whether correcting the identified approximation also
reduces the deployment loss").

The tenth review of paper 1 accepts the gap between recorded decisions and
live loops (experiment IL: sequential loops 4.4 points below full-evaluation
loops at round 10) but finds its explanation only partly established. Two
things change when the sequential rule drives a loop, and the reviewer asks
for "a targeted intervention separating admission changes from
proposal-history changes":

- admission: a candidate the rule stops cannot be accepted, even when full
  evaluation would have accepted it;
- history: a stopped candidate enters the proposer's history with its
  early-stopped estimate (which was 2.2 points below the full evaluation on
  average in IL), without attribution, and with the rule's stop reason; it
  also feeds the prune set.

A post hoc analysis found a second problem in IL: the noise band delta,
calibrated per loop before round 0, happened to be larger in the sequential
arm (mean 9.8 vs 6.7 points), and delta correlates with transfer (about
-0.6 in each arm). Adjusted for delta, the 6 v 6 difference is -2.1 points
[-6.2, +1.9]. BK calibrates the noise band once per block and gives it to all
four arms of the block, so the comparison is balanced on delta by design.

## Known before registration

- IL and CL results, including the per-loop transfers, the shadow
  evaluations, the delta imbalance and every analysis in
  `results/il/` and `results/r10/`.
- Replaying the sequential rule on the full-evaluation IL loops' own episodes
  (`verify/bk.py replay`, no new episodes): in their states the rule would
  have changed the decision in 3 of 40 rounds (f2 r1, f3 r3, f4 r9).
- The cost-aware rule replayed on the same loops changes the decision in 1 of
  40 rounds and uses 82% of the candidate-evaluation episodes (the registered
  rule 73%).
- The new selection modes were checked on the finished IL loops without
  running any episode: on the full loops seqhist and seqadm reproduce every
  recorded full decision and winner (40 rounds), seqcost's choices equal the
  replay's in all 40 rounds, and replaying the registered rule on the seqfull
  loops' own episodes reproduces all 54 recorded stopping records exactly.

## Design

- Domain tau2 airline; rounds t = 0..9 with T = 20; loop, models,
  temperatures, tasks, prompts, m = 2, k = 2 and the noise-band calibration
  (baseline plus two repeated base evaluations, z = 2) are those of IL.
- Four blocks b1..b4 (design five_arms). In each block a calibration
  checkout `/home/user/bk_<block>_cal` calibrates the noise band once;
  `verify/bk.py seed` then copies the frontier, the calibration, the history
  and the three base evaluations into the block's five loop checkouts and puts
  each loop's branch at the same base commit. Every checkout is detached at the commit
  that adds this file; RRSI's branches live under `evolve/bk_*/` and are
  never pushed.
- Arms (selection modes in `verify/live.py`):

  | arm | selection | admission | history of stopped candidates |
  |---|---|---|---|
  | f | full | full evaluation | full evaluation |
  | s | seqfull (gamma 0.05, batches of 10, as in IL) | sequential | early-stopped |
  | c | seqcost (cost-aware rule, gamma 0.05) | sequential, cost-aware | early-stopped |
  | h | seqhist | full evaluation | early-stopped (seqfull's) |
  | a | seqadm | sequential (seqfull's) | full evaluation |

  seqcost is seqfull with the predictive probability of
  `verify/posthoc_r10.p_cost`: it integrates over the final cost change,
  normal around the running estimate with variance
  ((N-n)/N)^2 s_c^2 (1/n + 1/(N-n)), instead of holding the cost change at
  its running estimate (offline: recall of full evaluation's acceptances 88%
  instead of 74% on r2+r3).

  In h and a every screened candidate is evaluated in full and the
  sequential rule is replayed on its episodes in seqfull's order to find
  where seqfull would have stopped it. h accepts what full evaluation accepts;
  a stopped candidate that does not win is recorded exactly as seqfull
  records it (early-stopped estimate, seqfull's reason, no attribution). a
  accepts what seqfull would accept; every candidate is recorded with its full
  evaluation, and a stopped candidate that full evaluation admits is recorded
  as "not admitted: the sequential rule stopped it" with its full scores.
- The loops of a block run concurrently. Blocks run in two waves (b1 and
  b2, then b3 and b4), ten loops at a time
  (`verify/run_bk_all.sh` -> `verify/run_bk.sh` -> `verify/run_bk_loop.sh`).
- Held-out deployment, 20 test tasks x 12 trials, with IL's deployment code:
  each block's base once (in its calibration checkout) and every loop's
  incumbent after round 9.
- What full evaluation and the sequential rule would have chosen in every
  round: s and c loops by the shadow evaluation (`verify/cl.py shadow`, as in
  IL), f loops by the replay (`verify/bk.py replay`, no episodes; both rules),
  h and a loops from their own round records.
- IL's missed acceptances (option A of the plan, `verify/bk.py missed`):
  held-out deployments of the four stopped candidates that full evaluation
  would have chosen in IL's sequential loops (s2 r5 A, s3 r9 B, s4 r0 A,
  s4 r1 B), each compared with the incumbent it would have replaced (all
  four incumbents are already deployed). CL's two were deployed earlier
  (+3.75 and +4.2 points).

## Outcomes (`verify/bk.py analyze`)

Transfer of a loop = held-out success of its incumbent after round 9 minus
its block's base, mean over the 20 tasks (paired by task); 0 when the loop
accepted nothing.

Primary, each with a 90% interval from the two-way additive model (block +
arm, residual df 9, t intervals); a bootstrap over blocks and held-out tasks
(B = 4000) is reported alongside:

1. Replication: s minus f, averaged over blocks (the IL comparison with the
   noise band balanced and both arms concurrent).
2. The corrected rule: c minus f, and c minus s.
3. Admission main effect: (s + a - f - h) / 2.
4. History main effect: (s + h - f - a) / 2.

The residual variance comes from the additive model over all five arms
(df 12). Secondary: the interaction (s - a) - (h - f); s minus f pooled with IL (s:
cl1, cl2, s1..s4 and the four BK s loops; f: r2, r3, f1..f4 and the four BK
f loops), OLS with arm, centred delta and a BK indicator; the held-out gains
of IL's four missed acceptances (and CL's two) over their incumbents.

Descriptive, per loop: accepted changes; rounds in which full evaluation
would have accepted; rounds in which seqfull and full evaluation would choose
differently; stopped candidates; stopped candidates full evaluation would
have chosen; delta; the loop's dollar cost and candidate-evaluation episodes
(IL's accounting; the block's calibration counted in each of its loops).

## How the results will be read (decided now)

- The IL loss is "replicated with a balanced noise band" if the 90% interval
  of s - f lies below 0, and "not replicated" otherwise; the paper reports the
  point estimate and interval either way, with the pooled estimate.
- The corrected rule "reduces the loss" if the 90% interval of c - s lies
  above 0; "loses to full evaluation" if that of c - f lies below 0. If
  neither, the paper reports both estimates and says the loops cannot tell.
  Its candidate-evaluation episodes and loop cost are reported with it.
- A mechanism is "supported" if its main effect's 90% interval lies below 0.
  If neither interval excludes 0, the paper says that the experiment could not
  separate the two mechanisms and keeps the present (softened) wording.
- If the interaction's interval excludes 0, the main effects are reported
  only together with the simple effects (s - a, h - f, s - h, a - f).
- A non-significant effect is not read as the absence of an effect.

Power, stated before the runs: IL's between-loop SD was 4.6 (sequential) and
3.1 (full) points, including the base deployment's noise and the delta
differences, both of which cancel within a block. For a residual SD of 3 to
4.5 points, a main effect has a standard error of 1.5 to 2.3 points (80%
power at two-sided 10% for effects of about 4 to 6 points) and a difference
of two arms (s - f, c - f, c - s) one of 2.1 to 3.2 points (about 6 to 8
points). The experiment can therefore show a
mechanism that carries most of a 4-point loss only if the residual SD is near
the low end; if the two mechanisms carry about 2 points each, the most likely
outcome is that neither interval excludes 0.

## Cost and stopping

Quote: about 80 dollars (20 ten-round loops at 2.9 to 3.4 dollars, four
block calibrations, 24 held-out deployments at about 0.3 dollars, the shadow
evaluations, and about 1.2 dollars for IL's missed acceptances). Stop line 120
dollars: `verify/bk.py spend --limit 120` runs before every round of every loop, and a
loop that would start a round above the limit stops; Dr Cao is then asked
before anything else runs.

## Blinding

No held-out success rate of a BK deployment or of IL's missed acceptances is
printed or inspected until every BK deployment has finished; progress is
monitored from counts and spend only (`verify/bk.py status`). The analysis
code (`verify/bk.py analyze`, `tables`) is committed with this file, before
any episode.

## Operations log

(changes after registration are recorded here)
