"""Domain paragraphs for the RRSI search roles on tau2-bench ({domain} is
retail or airline, {policy} the frozen policy model's label; both are filled in
by domains/tau2/common.py)."""

ANALYST = """The agent is a customer-service agent on tau2-bench ({domain}): a
frozen policy LLM ({policy}) driven by a small harness talks to a simulated
customer (another LLM following hidden instructions) and calls the airline /
retail back-office tools (look up users, orders or reservations; cancel,
modify, exchange, return, book; transfer to a human) under a written domain
policy. A trial succeeds only if the database ends exactly as the task's gold
actions would leave it AND every required piece of information was told to
the customer. Each rendered trace shows the customer's hidden instructions,
the dialogue with every tool call and result, and a GRADING block listing the
gold actions (marked MADE / NOT MADE / same tool with different arguments)
and any required information that was never said. Common failure families:
a write action with a wrong argument (wrong item, payment method, cabin,
passenger count), an action the policy forbids or one taken without the
customer's explicit confirmation, a missing action (the agent stopped,
transferred, or the customer left), a wrong calculation (prices, refunds,
compensation), and required information never stated. Rank modes by trials
lost. Where a failure digest hints the agent was blocked by harness
mechanics (what tool results look like, missing computed context, no check
before an irreversible write) rather than by judgment, use the
capability_gap lens."""

DIGESTER = """The trajectory comes from a customer-service agent on
tau2-bench ({domain}): a frozen policy LLM talks to a simulated customer and
calls back-office tools under a written domain policy. Each <task_id>.txt
holds the customer's hidden instructions, every step ([step i] AGENT text, AGENT
CALLS tool(arguments), RESULT, CUSTOMER reply), run metadata, and at the end
the GRADING block: reward, whether the final database matches the gold one,
the gold actions marked MADE / NOT MADE / same tool with different arguments,
and required information that was never said. Read the GRADING block first,
then find the step where the trajectory left the gold path (the first wrong
or missing write, the misread result, the policy rule misapplied) and quote
it by step number."""

PROPOSER = """The benchmark is tau2-bench ({domain}): customer-service
dialogues in which the agent must resolve a simulated customer's request by
calling back-office tools under a written domain policy. The harness is a
small Python package: agent.py (the Agent class: next_action, before_tool,
render_result, observe_prefix), prompts.py (the system prompt; the domain
policy text is passed in, verbatim from the benchmark), tools.py (how tool
descriptions are presented) and mechanisms.py (optional building blocks:
bounded extra policy calls, harness-side client tools, skill files). The
frozen driver owns the environment, the customer simulator, the 40-step limit
and the grader; the harness sees only the contract in agent.py's docstring.
The frozen policy is {policy}; do not assume it shares your habits or
judgment, read the trajectories for how it actually behaves.

Success needs the final database to equal the gold one exactly and every
required fact to be told to the customer, so one wrong argument on one write
call fails the task. Irreversible writes, explicit customer confirmation
before a write, exact arithmetic (prices, refunds, balances) and following
the policy's eligibility rules are where most trials are lost.

Out of bounds, enforced before measurement: importing or reading the
benchmark (tau2 package, task files, user-simulator guidelines, the grader),
writing files or keeping any state across episodes, and task-specific content
(user ids, order or reservation ids, names, e-mails, item ids, amounts that
belong to particular tasks). The run's outcome is judged on held-out tasks of
the same domain that the search never sees, so memorized task knowledge is
worthless. Litmus test: would this help a competent human agent handling
MANY unfamiliar customers under this policy?"""

CRITIC = """The harness drives a frozen policy LLM through tau2-bench
({domain}) customer-service dialogues: it talks to a simulated customer and
calls back-office tools under a written domain policy; a trial succeeds when
the final database equals the gold one and the required information was told
to the customer. Domain-general practice is legitimate and NOT leakage ("read
back every argument of a write call and get an explicit yes first", "compute
refunds with a calculator tool", "re-check the policy's eligibility rules
before an exchange"). The line is crossed at task-specific facts: particular
user, order, reservation or item ids, customer names, amounts or answers, or
branching on features that identify one task. Also REJECT anything that reads
the benchmark's files or the grader, writes files, or keeps state across
episodes (module-level caches, globals that outlive an Agent), and anything
that weakens the harness contract the driver relies on (the Agent signatures,
returning content blocks, observe_prefix rebuilding per-episode state)."""
