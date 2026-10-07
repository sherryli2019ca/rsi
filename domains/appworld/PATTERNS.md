# Pattern Library (reference, not an allowlist)

Known mechanism families for code agents that act through app APIs in a
REPL, with generic traps. You may adopt, adapt, ignore, or invent formats not
listed here.

## 1. Complete reads (pagination)

Many failed trials read only the first page of a paginated API, or only one
of several sources the task names (several libraries, both sent and received
items). Mechanisms: a generic paginator helper prepended to the REPL code
(call any `apis.<app>.<api>` with page_index 0, 1, ... until an empty page,
with a hard page cap) and announced in the prompt; in render_output, mark an
output whose list has exactly the page size as "possibly more pages".

Generic trap: a helper the policy does not know about is never used; say
when to use it, and keep it free of task assumptions (no app or field names
beyond the page parameters).

## 2. API documentation discipline

The policy guesses API names, parameters or response fields. Mechanisms:
render show_api_doc outputs compactly (name, required / optional parameters
with types and defaults, response fields) so reading docs is cheap; when a
call fails with an unexpected or missing parameter, append the relevant
parameter list or a pointer to show_api_doc.

Generic trap: forcing a doc read before every call costs steps; gate only
the first call of an API, or only after a failure, and bound it.

## 3. Error recovery and loops

The agent repeats the same failing code or drifts after a traceback.
Mechanisms: render tracebacks with the failing line and a one-line likely
cause; detect the same error twice in a row and inject a short instruction to
change approach; remind the agent that variables persist across steps.

Generic trap: rewriting tracebacks so the real message is lost. Add, do not
replace.

## 4. Answer and completion discipline

Trials are lost at the last step: an answer in a sentence instead of the bare
value, a number as text, completing before all requested changes were made,
or never calling complete_task. Mechanisms: before_execute holds the first
complete_task call once and returns a short checklist (bare value, right type,
every requested change made and verified); a budget-aware reminder in the
observations near the step limit to finish and call complete_task.

Generic trap: a gate that never opens. Let the call through on the second
attempt, never block other code, and make the condition checkable from the
code and transcript so replays reproduce it.

## 5. People, relations and time windows

Relations ("friends", "coworkers", "family") come from the phone contacts;
"today", "yesterday", "this month", "last N days" are relative to the
environment's date (datetime.now() in the REPL), with inclusive day bounds.
Mechanisms: a short procedure in the prompt or a skill file for resolving
relations and date windows; a helper that returns [start, end] datetimes for
a named window.

Generic trap: hard-coding a date or a person. Everything must come from the
environment at runtime.

## 6. Bounded self-checks (sub-calls)

Example: before complete_task, one sub-call reads the task and the last few
steps and answers only OK or the specific problem (unverified write, answer
form, missed source). Fire only on completion, at most once per episode.

Cost note: sub-calls run on every trial where they fire and count towards C.

## 7. Context management

Long episodes carry large API docs and listings. Mechanisms: shorten old
outputs that were superseded (keep ids, names and numbers exact); cap a
single huge output with a note on how to print a narrower slice.

Generic trap: dropping the one output a later step depends on; prefer
eliding docs over eliding data.
