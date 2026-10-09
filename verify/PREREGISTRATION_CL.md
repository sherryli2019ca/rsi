# Pre-registration: a closed loop driven by sequential full evaluation (experiment CL)

Approved by Dr Cao on 2026-10-09 (08:24Z) after the eighth review of paper 1,
and registered before any episode of CL was run. The paper's sequential rule
was scored offline on RRSI's own trajectories: it made full evaluation's
decision in 93% of rounds and 74% of its acceptances, with nearly the same
decision value. Offline scoring cannot show what happens when the rule drives
the loop: RRSI shows its proposer the measured gains of rejected candidates,
which become early-stopped estimates, and every acceptance the rule drops
changes the states that follow. CL runs the loop with the rule in place.

## Design

- Domain: tau2 airline only, the domain on which RRSI's evolve gains
  transferred to held-out tasks (r1, r2); retail never transferred, so it has
  no gain for the rule to lose.
- Two new trajectories, cl1 and cl2, each run by `verify/run_cl.sh` from its
  own worktree pinned to the commit that adds this file, with RRSI's branches
  under `evolve/<name>/`. Loop, models, temperatures, tasks, prompts, m = 2,
  k = 2 and the calibration of the noise band from three base evaluations are
  as for r2 and r3 (the code differs from theirs only in the selection step
  and in parts the loop does not use: thinking roles, off unless an
  environment variable is set, and forced replays).
- Rounds t = 0..9 (ten rounds), as for r4 and the first ten airline rounds of
  r1 to r3. T stays 20, so the edit-budget schedule of these rounds is the one
  r1 to r3 had.
- Selection step: `--selection seqfull` (verify/live.py). Each screened
  candidate's 60 evolve episodes run in batches of 10 in a random order over
  (task, trial) pairs. After each batch the candidate is dropped when its
  harness errors exceed 2% of the 60 planned episodes, or when the predictive
  probability that RRSI's Algorithm 2 admits it after all 60 episodes is below
  gamma = 0.05 (`verify/posthoc_e1.p_admissible`, the rule registered offline
  in PREREGISTRATION_E1R.md, addendum 2). Candidates that are not dropped
  finish their evaluation and are decided by RRSI's own rule. A dropped
  candidate enters RRSI's history as rejected with its early-stopped
  estimate, which the proposer sees in later rounds.
- Held-out deployment (`verify/cl.py deploy`): the base harness, every
  incumbent the loop accepts, and the incumbent after round 9, each on the 20
  test tasks x 12 trials = 240 episodes, with the deployment code of r1 to r3.
  Candidates that are not accepted are not deployed.
- Shadow evaluation (`verify/cl.py shadow`, secondary): after round 9, every
  dropped candidate's evaluation is completed to all 60 episodes and RRSI's
  rule is applied with that round's S*, noise band and novelty, which gives
  the decision full evaluation would have made in the states the
  sequential loop visited. The loop never sees these episodes.

## Reference

The full-evaluation loops r1, r2 and r3 on airline, first ten rounds, already
run and deployed: transfer at round 10 (held-out success of the incumbent
after round 9 minus the base, paired by task, 240 episodes each) +8.75, +8.75
and +2.92 points; 3, 3 and 4 acceptances.

## Outcomes (all descriptive; two trajectories cannot establish equivalence)

1. Transfer at round 10 of cl1 and cl2, each with a 90% task-bootstrap
   interval, and the difference between their mean and the mean of r1 to r3
   (trajectories fixed, held-out tasks resampled, B = 2000).
2. Candidate-evaluation episodes per round and their share of the planned
   episodes, against the 60 per screened candidate of full evaluation, and
   the loop's dollar cost per round (model calls by role and episodes, list
   prices as in Table 27 of the paper) against r1 to r3's first ten rounds.
3. Acceptances (above the noise band or within it), and the evolve success
   of the incumbent after each round.
4. Shadow: the share of rounds in which the sequential loop made the decision
   full evaluation would have made, and the share of full evaluation's
   acceptances it made.
5. The held-out gain of each accepted change over the incumbent it replaced.

We will report CL as consistent with the offline analysis if both
trajectories' transfer at round 10 lies within the range of r1 to r3 widened
by their 90% intervals, and as inconsistent otherwise; either way it is one
paragraph of descriptive evidence, not a test.

## Cost and stopping

Quote: about 12 to 15 dollars (two loops of ten rounds at r2's measured cost
per round, about three dollars each including evaluation, plus at most five
deployments of 240 episodes per trajectory and the shadow episodes). Stop and
report to Dr Cao if the measured spend exceeds 25 dollars.

## Blinding

No held-out success rate of a CL deployment is printed or inspected until
every deployment of both trajectories has finished; progress is monitored
from counts and spend only. The analysis code (`verify/cl.py analyze`) is
committed before the deployments finish.

## Operations log

(changes after registration are recorded here)
