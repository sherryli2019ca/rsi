# Pre-registration: blocked loops with shared calibration: the cost-aware stopping rule and what sequential evaluation changes in an RRSI loop (experiment BK)

Approved. Dr Cao chose the five-arm experiment (option 1 of
`/mnt/project-files/reviews/review11-plan.md`, decision card, 2026-10-10
00:49Z) and then asked whether an equivalent experiment could cost less. The
coupled design below was offered in its place (decision card 00:58Z,
recommended; about 60 instead of 80 dollars) together with the original
design of independent loops. On 2026-10-10 at 10:00:53Z Dr Cao chose the
coupled design (decision card re-posted after the twelfth review, whose
questions 1-3 ask for the same comparisons). `DESIGN = "five_arms_coupled"`
in `verify/bk.py`; this file is committed with the code before any episode
of BK runs, and that commit is the registered one.

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
arms of the block, so the comparison is balanced on delta by design.

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
- The coupled design's code was checked without any model call or episode
  (the network cut off; `python -m verify.bk_checks offline` and `power`
  reproduce the first and last of these checks):
  - On the four full IL loops, the group round's record for arm f equals
    the recorded IL round in all 40 rounds: every history row, including the
    reason text, the attribution and the winner.
  - In those states, a's round differs from f's in 4 of 40 rounds (a
    stopped candidate that full evaluation admits), and s's from h's in 3
    (the full winner stopped). These are the rounds the replay identifies.
    In 4 rounds no candidate was evaluated and all five arms stay together.
  - Two test blocks were built from IL states (f2 round 1, where all five
    arms differ; f1 round 1, where the arms form the groups {f, a}, {h, s}
    and {c}). Every forked lineage reran the round in its own mode and
    recorded exactly the state the group had predicted for its arms, with no
    new episode and no model call.
  - A forked lineage's loop script started and read its mode correctly.
  - Under a simulated null, the lineage model's 90% intervals covered 0 in
    90% (pairwise contrasts) to 94% (admission, history, interaction) of
    4000 replications.
- The twelfth review of paper 1 (2026-10-10 09:41Z) and its zero-cost
  analysis (`verify/posthoc_r12.py`, `results/r12/report.json`): offline,
  letting the final success and cost changes correlate moves the cost-aware
  rule's recall of full evaluation's acceptances by at most two points.
  Nothing in BK changes because of it.

## Design

- Domain tau2 airline; rounds t = 0..9 with T = 20; loop, models,
  temperatures, tasks, prompts, m = 2, k = 2 and the noise-band calibration
  (baseline plus two repeated base evaluations, z = 2) are those of IL.
- Four blocks b1..b4. In each block a calibration checkout
  `/home/user/bk_<block>_cal` calibrates the noise band once. Every checkout
  is detached at the commit that adds this file. RRSI's branches live under
  `evolve/bk_*/` and `bk_*/`; they are never pushed.
- Arms (selection modes in `verify/live.py`):

  | arm | selection | admission | history of stopped candidates |
  |---|---|---|---|
  | f | full | full evaluation | full evaluation |
  | h | seqhist | full evaluation | early-stopped (seqfull's) |
  | c | seqcost (cost-aware rule, gamma 0.05) | sequential, cost-aware | early-stopped |
  | a | seqadm | sequential (seqfull's) | full evaluation |
  | s | seqfull (gamma 0.05, batches of 10, as in IL) | sequential | early-stopped |

  seqcost is seqfull with the predictive probability of
  `verify/posthoc_r10.p_cost`. It integrates over the final cost change,
  normal around the running estimate with variance
  ((N-n)/N)^2 s_c^2 (1/n + 1/(N-n)), instead of holding the cost change at
  its running estimate. Offline, it recalls 88% of full evaluation's
  acceptances on r2+r3, against 74% for seqfull.

  In h and a, every screened candidate is evaluated in full. The sequential
  rule is then replayed on each candidate's episodes, in seqfull's order, to
  find where seqfull would have stopped it.
  - h accepts what full evaluation accepts. A stopped candidate that does
    not win is recorded exactly as seqfull records it: early-stopped
    estimate, seqfull's reason, no attribution.
  - a accepts what seqfull would accept. Every candidate is recorded with
    its full evaluation. A stopped candidate that full evaluation admits is
    recorded as "not admitted: the sequential rule stopped it", with its
    full scores.
- Coupling (design five_arms_coupled):
  - Arms that are in the same state share their rounds. A block starts one
    lineage, `/home/user/bk_<block>_root`, which holds all five arms and is
    seeded with the block's calibration (`verify/bk.py seed`).
  - In a lineage holding several arms, a round (selection `group:<arms>`,
    `verify/live.py select_group`) drafts the candidates once and evaluates
    every screened candidate in full. It then computes each arm's round from
    those episodes exactly as the arm's own mode records it: full evaluation
    with RRSI's rule; seqfull and seqcost with the rule replayed in its
    order; seqhist and seqadm as above.
  - Arms whose rounds add the same thing to the loop's state stay together.
    That state is: the history rows (values as stored, reason text included),
    which candidates are attributed, and the winner.
  - Each other group of arms is forked into its own lineage,
    `/home/user/bk_<block>_<arms>` (`verify/bk.py fork_lineage`). The fork
    gets the state as of the start of the round, the round's checkpoints
    (analysis, candidates, evaluations; episodes as hard links) and its own
    branches. It reruns the round in its own mode and then continues alone.
  - The forking lineage keeps the group of its first arm, in the order
    f, h, c, a, s.
  - A lineage holding one arm runs that arm's own mode, as an independent
    loop would.
  - Each arm's loop therefore has exactly the distribution of an
    independent loop of its mode. Arms are coupled only while their states
    are identical.
  - Because the reason texts differ, h, c and the pair {f, a} always
    separate in the first round in which a candidate is evaluated. a stays
    with f until the sequential rule stops a candidate that full evaluation
    admits. s stays with h until the rule stops the candidate full evaluation
    would choose.
- Blocks run in two waves (b1 and b2, then b3 and b4). The next wave starts
  when every lineage of the current one has finished its rounds
  (`verify/run_bk_all.sh` -> `verify/run_bk.sh` -> `verify/run_bk_loop.sh`).
- Held-out deployment, 20 test tasks x 12 trials, with IL's deployment code:
  - each block's base once;
  - every lineage's final incumbent after round 9.

  Deployments are made once per block and harness, in the calibration
  checkout. Arms that end with the same harness share its deployment.
- What full evaluation and the sequential rule would have chosen in every
  round of every arm comes from three sources:
  - shared rounds: `r<t>/group.json`;
  - rounds of a lineage that ends as s or c alone: the shadow evaluation
    (`verify/cl.py shadow`, as in IL);
  - a lineage holding f: the replay (`verify/bk.py replay`, no episodes);
  - h and a alone: their own round records.
- IL's missed acceptances (option A of the plan, `verify/bk.py missed`):
  held-out deployments of the four stopped candidates that full evaluation
  would have chosen in IL's sequential loops (s2 r5 A, s3 r9 B, s4 r0 A,
  s4 r1 B). Each is compared with the incumbent it would have replaced; all
  four incumbents are already deployed. CL's two were deployed earlier
  (+3.75 and +4.2 points).

## Outcomes (`verify/bk.py analyze`)

Transfer of an arm = held-out success of its final incumbent after round 9
minus its block's base, mean over the 20 tasks (paired by task). It is 0 when
the arm accepted nothing.

Primary estimands:

1. Replication: s minus f, averaged over blocks. This is the IL comparison
   with the noise band balanced and both arms concurrent.
2. The corrected rule: c minus f, and c minus s.
3. Admission main effect: (s + a - f - h) / 2.
4. History main effect: (s + h - f - a) / 2.

Each primary estimand gets a 90% interval from the lineage model
(`verify/bk.py _lineage_se`). The model keeps the additive model's own
assumption: every arm's loop has the same residual variance s^2 around
block + arm.
- Loops of different lineages are independent.
- a shares f's rounds until their states first differ, so
  cov(a, f) = s^2 - v_a / 2 with v_a = var(a - f). The same holds for s and h.
- s^2 comes from the additive model over f, h and c, each of which always
  has a lineage of its own (df 6).
- v_a and v_s come from the spread of a - f and s - h over blocks (df 3).
- Intervals use t quantiles with Satterthwaite df.
- If a contrast's estimated variance is not positive, its own spread over
  blocks is used (df 3).

A bootstrap over blocks and held-out tasks (B = 4000) and the per-block values
are reported alongside. Secondary analyses:
- the interaction (s - a) - (h - f);
- s minus f pooled with IL, by OLS with arm, centred delta and a BK
  indicator (s: cl1, cl2, s1..s4 and the four BK s arms; f: r2, r3,
  f1..f4 and the four BK f arms);
- the held-out gains of IL's four missed acceptances (and CL's two) over
  their incumbents.

Descriptive, per arm:
- accepted changes;
- rounds in which full evaluation would have accepted;
- rounds in which seqfull (seqcost for c) and full evaluation would choose
  differently;
- stopped candidates, and stopped candidates full evaluation would have
  chosen;
- delta;
- the round in which a and s forked, if they did;
- the arm's dollar cost as if it had run alone, with IL's accounting:
  - model calls of each of its rounds (tagged with the round);
  - smoke tests;
  - the candidate episodes its own mode uses (all of them for f, h and a;
    the first n in the rule's order for s and c);
  - the block's calibration;
- its candidate-evaluation episodes;
- the experiment's actual spend.

## How the results will be read (decided now)

- The IL loss is "replicated with a balanced noise band" if the 90% interval
  of s - f lies below 0, and "not replicated" otherwise. The paper reports the
  point estimate and interval either way, with the pooled estimate.
- The corrected rule "reduces the loss" if the 90% interval of c - s lies
  above 0. It "loses to full evaluation" if that of c - f lies below 0. If
  neither, the paper reports both estimates and says the loops cannot tell.
  Its candidate-evaluation episodes and loop cost are reported with it.
- A mechanism is "supported" if its main effect's 90% interval lies below 0.
  If neither interval excludes 0, the paper says that the experiment could not
  separate the two mechanisms and keeps the present (softened) wording.
- If the interaction's interval excludes 0, the main effects are reported
  only together with the simple effects (s - a, h - f, s - h, a - f).
- A non-significant effect is not read as the absence of an effect.

Power, stated before the runs. IL's between-loop SD was 4.6 (sequential) and
3.1 (full) points. That SD includes the base deployment's noise and the delta
differences, both of which cancel within a block. The 90% interval widths
below come from simulating the lineage model with a residual SD of 3.5 points,
a fork in 60% of blocks, and the shared share of the loop uniform on 0 to 0.9.
The independent design is shown for comparison.

| contrast | coupled, 4 blocks | independent, 4 blocks |
|---|---|---|
| a difference of two arms (s - f, c - f, c - s) | 9.2 | 8.8 |
| admission main effect | 3.6 | 6.2 |
| history main effect | 10.0 | 6.2 |
| interaction | 7.3 | 12.5 |

For a residual SD of 3 to 4.5 points, 80% power at two-sided 10% applies to
effects of about:
- 6 to 8 points for a difference of two arms;
- 2 to 4 points for the admission effect (it depends on how often a and s
  fork);
- 6 to 9 points for the history effect.

The experiment can therefore show an admission effect of the size of IL's
missed acceptances. If most of a 4-point loss comes through history, the most
likely outcome for the history effect is an interval that includes 0.

## Cost and stopping

Quote: about 60 dollars.
- About 37 lineage-rounds per block instead of 50, at 0.31 to 0.34 dollars
  a round.
- Four block calibrations.
- About 21 held-out deployments at about 0.3 dollars.
- The shadow evaluations.
- About 1.2 dollars for IL's missed acceptances.

The design of independent loops is quoted at about 80 dollars. Stop line: 85
dollars. `verify/bk.py spend --limit 85` runs before every round of every
lineage. It counts each episode once even where forked lineages hold hard
links to it. A lineage that would start a round above the limit stops, and
Dr Cao is then asked before anything else runs.

## Blinding

No held-out success rate of a BK deployment or of IL's missed acceptances is
printed or inspected until every BK deployment has finished. Progress is
monitored from counts and spend only (`verify/bk.py status`). The analysis
code (`verify/bk.py analyze`, `tables`) is committed with this file, before
any episode.

## Operations log

(changes after registration are recorded here)

- 2026-10-10 10:02Z: blocks b1 and b2 and IL's missed acceptances launched
  from the registered commit 7b2aae2.
- 2026-10-10, between 12:36Z and 14:42Z: the container restarted. All ten
  lineages of b1 and b2 had finished their rounds; the held-out deployments of
  three (b1_c, b2_c, b2_s) and the calibration of b3 and b4 were interrupted.
  At 14:43Z they were resumed with the same scripts (`run_bk_loop.sh` in each
  unfinished lineage, `run_bk_all.sh`), which skip finished work. No result
  was inspected.
