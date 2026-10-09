# Pre-registration: independent loops driven by sequential or full evaluation (experiment IL)

Approved by Dr Cao on 2026-10-09 (14:41Z, decision card "4+4 loops") after the
ninth review of paper 1, and registered before any episode of IL was run.
The reviewer asked for "several independent live loops per condition, with a
prespecified comparison of final performance and total cost". Experiment CL
ran two loops with sequential full evaluation; IL adds four more, and four
loops with RRSI's own full evaluation run from the same code at the same time.

## Design

- Domain tau2 airline; rounds t = 0..9 with T = 20, as in CL and in the first
  ten airline rounds of r1 to r3. Loop, models, temperatures, tasks, prompts,
  m = 2, k = 2 and the noise-band calibration are those of CL.
- Eight new loops, each in its own worktree pinned to the commit that adds
  this file, launched together and run concurrently by `verify/run_il.sh`:
  il_s1..il_s4 with `--selection seqfull` (gamma = 0.05, batches of 10, as in
  CL) and il_f1..il_f4 with `--selection full` (RRSI's own selection). RRSI's
  branches live under `evolve/il_*/` and are never pushed.
- Held-out deployment (`verify/il.py deploy`): the base harness and the
  incumbent after round 9 of every loop, 20 test tasks x 12 trials = 240
  episodes each, with the deployment code of CL and r1 to r3.
- Shadow evaluation (seqfull loops only, `verify/cl.py shadow`, as in CL).

## Arms

- Sequential: cl1, cl2 (experiment CL) and s1..s4 (6 loops).
- Full: r2, r3 (their first ten rounds and the deployment of their incumbent
  after round 9, already run) and f1..f4 (6 loops). r1 is left out of the
  primary comparison because its component labels were not corrected; it is
  added in a sensitivity analysis.

## Outcomes (`verify/il.py analyze`)

Primary, estimated and reported with 90% intervals (bootstrap over loops
within arm and, for transfer, over held-out tasks; B = 2000; Welch standard
errors alongside):

1. Transfer at round 10 (held-out success of the incumbent after round 9
   minus the base, paired by task): mean of the sequential arm minus mean of
   the full arm. We also report the one-sided bootstrap probability that the
   difference is below -5 points (a non-inferiority margin of 5 points). With
   the between-loop SD seen so far (about 4.8 points), six loops per arm give
   about 55% power at that margin, so failing it is expected to be
   uninformative; passing it would rule out a loss above 5 points.
2. The loop's dollar cost over rounds 0..9 at list prices (model calls by
   role, candidate evaluations, smoke tests and calibration; as in Table 27 of
   the paper), sequential minus full.
3. Candidate-evaluation episodes over rounds 0..9, sequential minus full.

Sensitivity: the same comparisons on the eight new loops only (4 vs 4), and
with r1 added to the full arm. Secondary, descriptive: accepted changes per
loop; for the sequential loops, the shadow evaluation (rounds in which full
evaluation would have decided the same, full evaluation's acceptances the
sequential loop made, dropped candidates that full evaluation would have
admitted or selected).

## Cost and stopping

Quote: about 35 dollars (a ten-round loop with base and final deployments
costs about 4 to 4.5 dollars). Stop and report to Dr Cao if the measured spend
of IL exceeds 60 dollars.

## Blinding

No held-out success rate of an IL deployment is printed or inspected until
every deployment of the eight loops has finished; progress is monitored from
counts and spend only. The analysis code (`verify/il.py analyze`) is committed
with this file, before any episode.

## Operations log

(changes after registration are recorded here)
