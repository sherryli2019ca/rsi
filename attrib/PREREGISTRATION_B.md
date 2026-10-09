# Pre-registration: evaluating failure attribution, Phase B (repairs, tau2)

Written 2026-10-09, before any Phase B attribution was run and before any
Phase B candidate was drafted. Approved by Dr Cao on the decision card of
02:55Z (option "跑 B 阶段": 8 tau2 incumbent states, 5 attribution groups, 2
candidates per group and state, each deployed on 240 held-out episodes; about
50 dollars, stop above 100). Code: attrib/phaseb.py and
attrib/phaseb_analyze.py at the commit that adds this file. Phase A is
registered in attrib/PREREGISTRATION.md.

## Question

Q3: does the attribution a proposer is given change the deployment value of
the repairs it drafts? In particular, do the methods that located the
decisive step better in Phase A lead to better repairs?

## States

The incumbent of the E1R trajectories r2 and r3 (pinned code 1b69724 and
37a9b49, identical trees) at the start of rounds 5 and 15, in both tau2
domains: 8 states, all distinct non-base harnesses.

| State | Incumbent | Its evolve job | Traces RRSI builds | Failing (reward < 1) |
|---|---|---|---|---|
| r2 retail t5 | fdbf43f | r0A | 28 | 10 |
| r2 retail t15 | 024abfd | r10B | 28 | 11 |
| r2 airline t5 | 53cab6a | r4A | 18 | 8 |
| r2 airline t15 | 6ec1725 | r12A | 18 | 7 |
| r3 retail t5 | d80f7de | r2B | 28 | 12 |
| r3 retail t15 | 1144db9 | r10A | 28 | 9 |
| r3 airline t5 | 75ca952 | r2A | 18 | 12 |
| r3 airline t15 | 92ff955 | r11A | 18 | 12 |

Traces are RRSI's own `build_traces` on the incumbent's evolve evaluation
(worst trial of the 22 lowest tasks, best trial of the 6 highest), so every
group's proposer sees the same traces the original round's proposer saw. Only
evolve-set failures reach the proposer; gains are measured on held-out tasks.

## Groups (the evidence F_t the proposer gets)

| Group | Analysis report | Per-task digests |
|---|---|---|
| none | a note: no analysis; read the traces yourself | none |
| rrsi | RRSI's own report and digests of that round, unchanged | RRSI's (it digested 8 of 9 failing traces in r3 retail t15 and 8 of 12 in r3 airline t5; all failing traces elsewhere) |
| first_write | the same note plus one sentence on the digests | one per failing trace: the first state-changing call (rule, zero cost) |
| binary_search | as above | the binary-search judge (Pro, thinking off; Phase A prompt) |
| cf_search | as above | counterfactual search with 40 replays (Pro suspects; Phase A protocol, no grading seen) |

The notes (attrib/phaseb.py, `NOTE_NONE`, `NOTE_STEP`) are fixed by this
commit. A step-attribution digest is
`{"task_id", "lens": "failure", "decisive_step": k, "step_excerpt": <the step
rendered, at most 600 characters>, "blocker": "failure decided at step k"}`.
It carries the step only: no reason, no component, no method name. Phase A
measured step localisation, so this isolates that signal; the rrsi group is
RRSI's practical reference and none is the floor. Every group's proposer can
read every trace, including its grading section, as in RRSI.

Attribution methods run once on the 81 failing traces (42 retail, 39
airline), against the incumbent's harness, with the Phase A code (attrib.methods,
attrib.search at this commit). A method that returns no valid step for a
trace gives no digest for it.

## Procedure

1. For each (group, state), an isolated copy of the RRSI run directory at
   that state (attrib/phaseb.py prep): the frontier with the trajectory
   truncated to the state (incumbent = trajectory entry t, S* = max S so far),
   history and attribution scoreboard truncated to rounds before t, the
   noise band (calibration.json), the incumbent's evolve job (link), and
   r<t>/analysis_report.json plus digests as in the table. The evolve branch
   evolve/pb/<group>/<state>/<domain> starts at the incumbent commit.
   Check, done before this registration: RRSI's directives recomputed from
   all 40 copies (edit budget, stall flag, tried components, exploration
   directives, prune set, noise band, S*) equal the original round's
   directives.json.
2. One unmodified RRSI round per (group, state) from the pinned E1R code
   (worktree /home/user/phaseb/code at 1b69724): proposer drafts variants A
   and B, critic with repair, smoke test, RRSI's full evolve evaluation (k =
   2) and RRSI's own selection rule. Models as in E1R (proposer, critic
   deepseek-v4-pro thinking off; policy and user simulator flash,
   temperature 0).
3. A candidate is deployable when it passed the proposer (a proposal was
   made), the critic, the smoke test, RRSI's evaluation (valid), and the
   common error check of paper 1 (harness error rate at most 2% on its
   evolve evaluation).
4. Held-out deployment (verify.deploy from the pinned code) of every
   deployable candidate and of each state's incumbent, at heldout_k (retail
   6 x 40 tasks, airline 12 x 20 tasks: 240 episodes each).
5. Order: state by state; the five groups of a state run side by side, so
   groups are not confounded with time. The first state (r2 retail t5) is
   the pilot; its cells count in the analysis unless the pilot exposes a bug
   in the Phase B code that changes what a proposer sees, in which case the
   pilot cells are discarded, the bug fixed, the cells re-run, and the change
   logged below.
6. Blinding: no held-out success rate of Phase B is printed or inspected
   before every deployment of every cell has finished. Progress reports
   show counts and spend only (attrib/phaseb.py status). RRSI's own evolve
   evaluations are part of the loop and appear in its logs; they are not an
   endpoint.

## Endpoints

Slot = (group, state, variant A or B): 80 slots, 16 per group.

- G(slot) = held-out S of the candidate minus held-out S of the state's
  incumbent (both deployed in Phase B), when the candidate is deployable;
  0 otherwise (the round keeps the incumbent).
- Contrast g − h = mean over the 8 states of [mean of g's two slots − mean of
  h's two slots].

## Primary analysis

Six contrasts, Holm-adjusted together: each step-attribution group
(first_write, binary_search, cf_search) against none and against rrsi.
Two-sided, alpha .05. p from a permutation of group labels within each state
(the 4 slots of the two groups; 10,000 permutations, seed 0); 95% CI from the
t distribution over the 8 state differences (df 7).

## Secondary analyses (not adjusted; reported as such)

- S1 rrsi vs none. S2 the three step groups pooled vs none.
- S3 per group: deployable count, gate outcomes, mean G among deployable
  candidates, share positive, candidates RRSI's rule accepted.
- S4 round level: per cell, G of the candidate RRSI's own rule accepted (0 if
  none, or if it fails the error check); the six primary contrasts on it.
- S5 the primary contrasts within each domain (4 states, df 3).
- S6 equivalence: a primary contrast whose 90% CI lies within ±2.5 pp is
  called practically equivalent (2.5 pp is about one SD of a single
  candidate's G in E1R).
- S7 Phase A accuracy (rescue magnitude of the named step, per domain) next
  to each group's mean G, descriptive. Directional expectation if step
  accuracy carries into repairs: retail first_write (.53) > binary_search
  (.33) > cf_search (.22); airline binary_search (.34) ≈ cf_search (.31) >
  first_write (.27).
- S8 how often the three methods name the same step on these traces.
- S9 drift: each incumbent's Phase B held-out S against E1R's deployment of
  the same commit.
- S10 the original E1R round's two candidates at each state (RRSI's
  evidence, drafted during E1R; E1R's deployments), descriptive, as a check
  on the rrsi group.

## Power

From E1R r2 and r3 (160 slots, same G definition): 67% of slots deployable,
no error-check failure; SD of G 2.6 pp, within-round SD 2.5 pp (states
explain little). Standard error of a contrast ≈ 2.6 × sqrt(2/16) = 0.92 pp;
minimum detectable difference at 80% power ≈ 2.6 pp unadjusted, ≈ 3.2 pp at
Holm's smallest threshold (.05/6). Mean candidate gains in E1R were near zero
(measured candidates −0.6 pp, SD 3.2), so only large differences between
groups can be detected; a null result is likely and will be reported with its
CI and S6.

## Cost and stopping

Prices from E1R and Phase A: drafting about 0.15 dollars per round (40
rounds, about 6); RRSI's evolve evaluations of about 54 candidates (about 9);
held-out deployment of about 54 candidates and 8 incumbents at 240 episodes
(about 21); attribution (binary search and counterfactual search on 81
traces, about 4). Total about 40 dollars; stop and report above 100
(attrib/phaseb.py status tracks spend from token logs).

## Not run (would need approval)

- An oracle group (the ground-truth decisive step from Phase A's protocol,
  about 14 dollars more).
- Phase B on AppWorld.

## Operations log and deviations

(Entries are appended below with UTC times.)

- 03:17Z 2026-10-09. Before any round: the copies' frontier now records the
  round in which the incumbent was accepted as `incumbent.t` (the prep first
  wrote the state's round; the field only appears in RRSI's log line). A dry
  run of RRSI's round on copies of two states passed its harness-tree check
  and reused the report. Attribution started 03:13Z; restarted at 03:15Z
  with 24 workers instead of 8 (resume-safe, nothing lost).
- 03:54Z. Attribution done: every method gave a valid step on all 81 failing
  traces; counterfactual search confirmed a suspect on 28 of 81 (fell back to
  its first suspect on the rest), 17.8 replays per trace on average.
- 04:19Z. Pilot (r2 retail t5, five groups side by side) ran end to end: 10
  drafts, 6 deployable, 2 no proposal (proposer turn limit), 2 critic
  rejections; step-group proposers read the traces around the digests' steps.
  No Phase B code change; the pilot cells stay in the analysis. Spend so far
  about 4 dollars. Main run started 04:21Z (/home/user/phaseb/run_main.sh:
  rounds two states at a time, deployments of finished cells alongside).
- 05:03Z. All 40 rounds done (the main run's 35 exited cleanly): 80 drafts,
  54 deployable (none 10, rrsi 13, first_write 10, binary_search 9, cf_search
  12); no evaluation was invalid and no candidate failed the error check.
- 05:20Z. Bookkeeping bug, no effect on data: phaseb looked up held-out jobs
  by the short incumbent hash while verify.deploy names them by the full
  hash, so finished deployments were re-invoked (every trial skipped as
  already on disk) and their worktrees were not removed. Fixed (commit
  4554d65); stale worktrees removed. No held-out number was looked at.
- ~05:55Z. The container restarted (uptime 0 at 06:01Z) with 23 of 62
  deployments finished; the deployments resumed at 06:03Z
  (/home/user/phaseb/run_deploy.sh; trials already on disk are kept).
- 07:08Z. All 62 held-out deployments complete (8 incumbents, 54
  candidates; no missing trial). The registered analysis
  (attrib/phaseb_analyze.py, unchanged since fa3a2b7 except the import of
  the job lookup fix) is run next; no held-out number was looked at before.
