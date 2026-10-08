# Pre-registration: the E1 design on AppWorld (experiment E3)

Approved by Dr Cao on 2026-10-08 as a check of the E1 findings on a second kind
of agent, and registered before any episode of the loop (the trial below
excepted). Not a test of larger effects: the trial showed as little headroom as
tau2.

## Trial (run before this registration, base harness only)

Evolve set 60 x 2: success 0.883 (RRSI noise band delta 0.041). First 60
test_normal tasks x 1: 0.90. 60 random test_challenge tasks x 1: 0.83. No
harness errors; about 0.002 US dollars per episode. The 60 test_normal episodes
are reused as the base harness's first held-out trial on those tasks.

## Design

- Domain `domains/appworld`: AppWorld 0.1.3; evolve = the first 20 train
  scenarios (60 tasks), held-out = test_normal (168 tasks, scenarios disjoint
  from train). Agent deepseek-v4-flash at temperature 0, thinking off; search
  roles and judge deepseek-v4-pro, thinking off.
- RRSI loop and evidence as in E1 and E1R (corrected component signals), T = 20,
  m = 2, k = 2. Held-out: 2 trials per task for every measured candidate and
  incumbent (336 episodes), 4 for the base and final harnesses (672).
- The judge's prompt names the agent as a code agent instead of a
  customer-service agent; nothing else differs.

## Analysis

As registered for E1R (primary: decision values under the common error check
for full evaluation, sample@40, net@40 and replay-null@40, non-inferiority of
each @40 rule with a margin of 0.75 points per round, Holm over the three,
block bootstrap over rounds sharing an incumbent and held-out tasks), on this
one trajectory, reported next to the tau2 results; net_prior with the audited
analyst accuracy is omitted (no audit on AppWorld). Secondary as in E1R.

## Operations

Quote 50-60 US dollars; stop and report past 90. Held-out success rates are not
examined before the run ends, apart from completion and error checks.

## Operations log

- 2026-10-08 about 01:26Z: the container restarted during round 0 and stopped
  the run; it resumed at 01:31Z from the last settled step, with commit 47ef920
  (the digester fix logged in PREREGISTRATION_E1R.md) applied. No held-out
  success rate of a candidate was examined.
- 2026-10-08, before any E3 decision value was computed: the sequential rules
  and hypotheses of addendum 2 of PREREGISTRATION_E1R.md are added to the E3
  analysis.
- 2026-10-08 02:20Z: with six tau2 trajectories running next to E3, the
  container had under 2 GB of memory left, so AppWorld evaluations now run 4
  episodes in parallel (was 6) and one candidate at a time (commits 121083b and
  the next); the E3 follower was stopped and resumed for this. Scheduling only;
  no episode changes.
- 2026-10-08 between about 06:31Z and 07:10Z: the container stopped again; the
  run and its follower resumed at 07:12Z from the last settled step. At 07:15Z
  the follower was restarted with 8 AppWorld episodes in parallel (the loop
  keeps 4). Scheduling only; no episode changes.
- 2026-10-08 07:14Z and 08:44Z, progress checks displayed the follower's
  round-0 and round-1 lines, which include the held-out success rates of the
  base, incumbent and candidates. Nothing was decided from them; later checks
  filter these values out.
