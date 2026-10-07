"""Domain paragraphs for the RRSI search roles on AppWorld ({policy} is the
frozen policy model's label, filled in by domains/appworld/adapter.py)."""

ANALYST = """The agent is an autonomous code agent on AppWorld: a frozen policy
LLM ({policy}) driven by a small harness does a supervisor's day-to-day task
(music, payments, shopping, e-mail, messages, notes, to-dos, files, ...) by
writing Python in a persistent REPL that calls the APIs of simulated apps
(spotify, venmo, amazon, gmail, phone, simple_note, splitwise, todoist,
file_system, plus supervisor for the supervisor's credentials and details and
api_docs for the API documentation). Each step the agent writes text with one
python code block; the environment executes it and the output is the next
observation. The episode ends when the code calls
apis.supervisor.complete_task (with the answer, for a question) or after 40
steps. A trial succeeds only if EVERY unit test of the task's evaluator
passes: the returned answer matches exactly, the app databases end in the
expected state, and nothing else was changed. Each rendered trace shows the
instruction and the supervisor, every step ([step i] AGENT text, CODE,
OUTPUT), and a GRADING block with the failed tests (requirement, test code
and assertion message) and the passed ones. Common failure families:
paginated APIs read only partly (first page only), an API called without
reading its documentation (wrong name, parameter or response field), wrong
people (relationships such as friends, roommates or coworkers come from the
phone contacts), wrong time windows (today, yesterday, this year, last N
days relative to the environment's date), wrong answer format (a sentence
instead of the bare value, wrong type or rounding), collateral or missing
writes (changing items the task did not ask about, stopping half-way), loops
on the same error until the step limit, and never calling complete_task. Rank
modes by trials lost. Where a failure digest hints the agent was blocked by
harness mechanics (how outputs are shown, overwhelming or truncated outputs,
missing information at the step it was needed, no check before
complete_task or before an irreversible write) rather than by judgment, use
the capability_gap lens."""

DIGESTER = """The trajectory comes from an autonomous code agent on AppWorld: a
frozen policy LLM writes Python in a REPL that calls simulated apps' APIs to
do a supervisor's task, and finishes by calling
apis.supervisor.complete_task. Each <task_id>.txt holds the instruction and
the supervisor, every step ([step i] AGENT text, CODE, OUTPUT; steps marked
BLOCKED were not executed), run metadata, and at the end the GRADING block:
reward, tests passed / total, each failed test with its requirement, test
code and assertion message (expected vs. produced values), and the passed
requirements. Read the GRADING block first, then find the step where the
trajectory went wrong (the first misread output, the skipped page, the wrong
API or argument, the wrong person or date window, the write that should not
have happened, the answer in the wrong form) and quote it by step number."""

PROPOSER = """The benchmark is AppWorld: an autonomous code agent does a
supervisor's day-to-day task by writing Python in a persistent REPL that calls
the APIs of simulated apps (spotify, venmo, amazon, gmail, phone,
simple_note, splitwise, todoist, file_system, supervisor, api_docs). The
harness is a small Python package: agent.py (the Agent class: next_action,
before_execute, render_output, observe_prefix), prompts.py (the official
AppWorld ReAct prompt: demonstrations, key instructions and the task; the
output format) and mechanisms.py (optional building blocks: bounded extra
policy calls, harness-side client tools, helper code prepended to the REPL
code, skill files). The frozen driver owns the AppWorld world, code
extraction (the first ```python block of a turn), execution, the 40-step limit
and the grader; the harness sees only the contract in agent.py's docstring.
The frozen policy is {policy}; do not assume it shares your habits or
judgment, read the trajectories for how it actually behaves.

Success needs every unit test of the task to pass: the exact answer for
questions (bare value, right type), the expected final state of the app
databases, and no collateral changes. Reading every page of paginated APIs,
reading an API's documentation before using it, resolving people and date
windows correctly, verifying writes, and calling complete_task correctly are
where most trials are lost.

Out of bounds, enforced before measurement: importing or reading AppWorld
itself (the appworld package, its data, task files, the evaluator, ground
truth or experiment outputs), writing files or keeping any state across
episodes, and task-specific content (task ids, person names, e-mail
addresses, phone numbers, song / playlist / file names, amounts or answers
that belong to particular tasks). The run's outcome is judged on held-out
tasks (AppWorld test_normal, other scenarios) that the search never sees, so
memorized task knowledge is worthless. Litmus test: would this help a
competent assistant doing MANY unfamiliar tasks with these apps?"""

CRITIC = """The harness drives a frozen policy LLM through AppWorld tasks: the
agent writes Python in a REPL that calls simulated apps' APIs (spotify,
venmo, amazon, gmail, phone, simple_note, splitwise, todoist, file_system,
supervisor, api_docs) and finishes with apis.supervisor.complete_task; a
trial succeeds when every unit test of the task's evaluator passes.
Domain-general practice is legitimate and NOT leakage ("loop over page_index
until an empty page", "read show_api_doc before the first call of an API",
"return only the bare value as the answer", "re-read the state after a write
to verify it", a generic pagination helper prepended to the REPL code,
compact rendering of API docs). The line is crossed at task-specific facts:
particular task or scenario ids, person names, e-mails, phone numbers, song,
playlist, note or file names, amounts or answers, or branching on features
that identify one task or scenario. Also REJECT anything that imports or
reads AppWorld itself (the appworld package, its data directory, task specs,
the evaluator, ground truth, experiment outputs) or the frozen driver, code
injected into the REPL that touches AppWorld internals (the requester,
private attributes of apis), anything that writes files or keeps state
across episodes (module-level caches, globals that outlive an Agent), and
anything that weakens the harness contract the driver relies on (the Agent
signatures, next_action returning text, observe_prefix rebuilding
per-episode state)."""
