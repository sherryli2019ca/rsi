# Pattern Library (reference, not an allowlist)

Known mechanism families for tool-using customer-service agents, with generic
traps. You may adopt, adapt, ignore, or invent formats not listed here.

## 1. Gates before irreversible writes

Most failed trials contain one wrong or premature write. Mechanisms: in
before_tool, hold a write call until the latest customer message is an
explicit yes to a read-back of exactly these arguments; check that ids in the
arguments appeared in earlier tool results; check counts (all items the
customer named are included). Return a short, specific message to the policy
explaining what is missing.

Generic trap: a gate that never opens. Bound it (let the call through after
one or two holds), never block read-only tools, and make the condition
checkable from the transcript so replays reproduce it.

## 2. Tool-result plumbing

Tool results are raw JSON. Mechanisms: render the fields a decision needs
first (status, payment methods, item options and prices); append values the
policy otherwise computes by hand (totals, price differences, refund amounts)
computed exactly in code; mark which items are eligible for the requested
action according to fields in the result.

Generic trap: hiding or rewriting fields the policy needs later. Add, do not
remove, unless a field is clearly noise.

## 3. Exact arithmetic

The frozen policy miscalculates sums, differences and refunds. Mechanisms: a
calculator client tool; code-computed totals in rendered results.

Generic trap: the policy ignores a tool it is not told when to use; say when
in its description.

## 4. Policy structure and timing

Mechanisms: a short procedure checklist derived from the policy's own
sections, shown with the policy; a reminder of the confirmation and
one-action-at-a-time rules injected when a write tool is about to be called.

Generic trap: long restatements of the policy add tokens on every call and
rarely change behaviour. Inject only where the evidence shows the rule was
missed.

## 5. Bounded self-checks (sub-calls)

Example: before executing a write, one sub-call that reads the conversation
and the call and answers only OK or the specific inconsistency. Fire only on
write tools; at most once per call.

Cost note: sub-calls run on every trial where they fire and count towards C.

## 6. Context management

Long episodes repeat large lookups. Mechanisms: shorten old tool results that
were superseded, keeping ids and amounts exact.

Generic trap: dropping the one result a later write depends on.
