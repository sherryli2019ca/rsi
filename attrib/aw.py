"""AppWorld side of the attribution study (attrib/PREREGISTRATION.md, Addendum 2).

The tau2 modules (groundtruth3, methods, search, pool) run on AppWorld
failures with these domain pieces swapped in; their tau2 code paths are
unchanged.

  view(rec, upto)       the failure as RRSI's AppWorld analyst reads it
                        (domains/appworld/adapter.render), optionally cut after
                        step `upto` with the grading block kept
  context()             the oracle's and the search's shared prefix: what the
                        agent is, the key instructions of its prompt (the
                        official AppWorld ReAct prompt) and every app API with
                        its HTTP method
  ORACLE_SYSTEM,        protocol 3's oracle for a code agent: the grading block
  oracle_prompt         (failed tests with expected values) plays the part of
                        tau2's gold actions; the corrected action is a
                        thought and python code (kept out of markdown fences
                        inside the JSON reply)
  to_force              an oracle or search action -> the forced assistant turn
                        (thought, then one python block; None without code)
  drive                 forced and null replays with the frozen AppWorld driver
  first_write           baseline: the first step whose executed code calls an
                        app API with an HTTP method other than GET (login and
                        logout excepted; api_docs never counts)
  SEARCH_SYSTEM,        counterfactual search's suspect call (no grading block)
  search_prompt
"""
from __future__ import annotations

import functools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.appworld.adapter import AW_PY, AW_ROOT, render  # noqa: E402

AUTH = {"login", "logout"}
_CODE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
_CALL = re.compile(r"\bapis\.(\w+)\.(\w+)\s*\(")


# ------------------------------------------------------------------- views --
def view(rec: dict, upto: int | None = None) -> str:
    """The failure as RRSI's analyst reads it; with `upto`, the steps stop after
    that step (the grading block still describes the whole episode)."""
    full = render(rec)
    if upto is None:
        return full
    part = {**rec, "steps": [s for s in rec["steps"] if s["index"] <= upto]}
    head = render(part).split("\n\nRUN: ")[0]
    grading = full[full.index("=== GRADING"):]
    return f"{head}\n\n(The episode continues after step {upto}.)\n\n{grading}"


def _methods() -> dict:
    """(app, api) -> (HTTP method, description), from AppWorld's API docs."""
    out = {}
    for p in sorted((Path(AW_ROOT) / "data" / "api_docs" / "standard").glob("*.json")):
        d = json.loads(p.read_text())
        items = d.items() if isinstance(d, dict) else enumerate(d)
        for k, a in items:
            if isinstance(a, dict):
                out[(p.stem, str(a.get("api_name", k)))] = (a.get("method"), a.get("description", ""))
    return out


@functools.lru_cache(maxsize=1)
def api_methods() -> dict:
    return _methods()


def _key_instructions() -> str:
    from domains.appworld.base_harness.prompts import REACT_PROMPT
    i = REACT_PROMPT.index("**Key instructions**")
    return REACT_PROMPT[i:].split("\nASSISTANT:")[0].strip()


@functools.lru_cache(maxsize=1)
def context() -> str:
    apis = "\n".join(f"- {app}.{name} [{m}]: {str(desc).strip().splitlines()[0][:160] if desc else ''}"
                     for (app, name), (m, desc) in sorted(api_methods().items()) if app != "api_docs")
    return ("=== THE AGENT ===\nAn autonomous code agent does a supervisor's day-to-day task on AppWorld "
            "by writing Python in a persistent REPL that calls the APIs of simulated apps through "
            "`apis.<app>.<api>(...)`; each step is one assistant turn whose first ```python block is "
            "executed, and its output is shown back. It finishes by calling "
            "apis.supervisor.complete_task (with answer=... when the task asks a question). Its "
            "prompt's rules:\n\n" + _key_instructions() +
            "\n\n=== APP APIS (app.api [HTTP method]: description) ===\n" + apis)


# ------------------------------------------------------------------ oracle --
ORACLE_SYSTEM = """You construct ground truth for failure attribution. An autonomous code \
agent failed a task of AppWorld. You see what the agent is and the rules of its prompt, the \
apps' APIs, the task's grading (every unit test of the task must pass; the failed tests show \
the expected values), and the episode up to one step of the agent.

Decide whether the agent's action at that step is a mistake: an action a careful agent \
following its rules would not take on the way to passing every test. Actions that are \
correct, or harmless and do not lead away from the right outcome (for example reading API \
documentation), are ok.

If it is a mistake, give the single action a careful agent would take at that step instead: \
one assistant turn, a short thought and the python code to run. Correct only this step; do not do more than a careful agent would do at this point. The corrected action \
must be one the agent could have taken with what it knew then (the outputs it had seen and \
the variables its earlier code defined); use the grading only to know what the right outcome \
is, never to supply ids, values or answers the agent had not yet learned.

Answer with JSON:
{"verdict": "ok" | "mistake", "why": "<one sentence>",
 "action": {"thought": "<one or two sentences>", "code": "<the python code, without markdown fences>"}}
("action" only when the verdict is mistake.)"""


def _grading(rec: dict) -> str:
    full = render(rec)
    return full[full.index("=== GRADING"):]


def _steps_text(rec: dict, k: int) -> str:
    """Header and steps 0..k-1, as the analyst's view renders them."""
    part = {**rec, "steps": rec["steps"][:k]}
    return render(part).split("\n\nRUN: ")[0]


def oracle_prompt(rec: dict, k: int) -> str:
    """Grading first and the step number last, so a failure's prompts share
    their prefix."""
    st = rec["steps"][k]
    return (f"{_grading(rec)}\n\n=== EPISODE ===\n{_steps_text(rec, k)}\n\n"
            f"=== AGENT ACTION AT STEP {k} (judge this step; the episode above is everything "
            f"before it) ===\n{st.get('assistant')}\n")


def to_force(action) -> str | None:
    """{"thought", "code"} -> the assistant turn the driver executes (thought,
    then one python block); None without code. Code travels outside markdown
    fences because rrsi.llm.extract_json would take a fence inside the JSON for
    the JSON."""
    if not isinstance(action, dict):
        return None
    code = action.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    code = _CODE.search(code).group(1) if _CODE.search(code) else code
    if not code.strip() or "```" in code:
        return None
    thought = str(action.get("thought") or "").strip()
    return f"{thought}\n\n```python\n{code.strip()}\n```\n".lstrip()


# ----------------------------------------------------------------- replays --
def drive(refs: list[tuple[str, dict]], rdir: Path, dom: str, workers: int) -> None:
    """Run (harness, ref) replays whose output does not exist yet, by harness
    (attrib.groundtruth._drive for the AppWorld driver)."""
    by_harness = {}
    for harness, r in refs:
        if not (rdir / f"{r['key']}.json").exists():
            by_harness.setdefault(harness, []).append(r)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "APPWORLD_ROOT": AW_ROOT}
    for harness, todo in by_harness.items():
        rdir.mkdir(parents=True, exist_ok=True)
        rf = rdir / f"refs_{abs(hash(harness)) % 10**8}.json"
        rf.write_text(json.dumps(todo))
        with open(rdir / "driver.log", "a") as lf:
            subprocess.run([AW_PY, "-m", "domains.appworld.driver", "--harness", harness,
                            "--out", str(rdir), "--replay", str(rf), "--workers", str(workers),
                            "--temperature", "0"], cwd=ROOT, env=env, stdout=lf,
                           stderr=subprocess.STDOUT, check=False)


# ---------------------------------------------------------------- baseline --
def is_write(app: str, api: str) -> bool:
    m = (api_methods().get((app, api)) or (None,))[0]
    return app != "api_docs" and api not in AUTH and m is not None and m != "GET"


def first_write(rec, model=None):
    for st in rec["steps"]:
        if st.get("blocked"):
            continue
        if any(is_write(a, b) for a, b in _CALL.findall(st.get("code") or "")):
            return {"step": st["index"], "top3": [st["index"]], "component": None, "calls": 0}
    n = rec["n_steps"]
    return {"step": n - 1, "top3": [n - 1, n - 2, n - 3][:n], "component": None, "calls": 0,
            "fallback": True}


# ------------------------------------------------------------------ search --
def search_system(n_suspects: int, comp_text: str, informed: bool = False) -> str:
    seen = ("the apps' APIs, the agent's steps and the task's grading (every unit test must pass; "
            "the failed tests show the expected values), so you know what the correct outcome was"
            if informed else "the apps' APIs and the agent's steps; you do not know what the correct "
            "outcome was")
    rule = ("using only what the agent knew at that point: use the grading only to know what the "
            "right outcome is, never to supply ids, values or answers the agent had not yet learned"
            if informed else "using only what the agent knew at that point")
    return (("""An autonomous code agent on AppWorld failed its task: the supervisor's task was \
not done as its unit tests require. You see what the agent is, the rules of its prompt, """ + seen + """.

Name the steps where the agent most likely went wrong, most suspect first (at most %d). For \
each, give the single action a careful agent would have taken at that step instead (one \
assistant turn: a short thought and the python code to run, """ + rule + """) and the harness \
component most likely at fault.

Harness components:
%s

Answer with JSON:
{"suspects": [{"step": <int>, "why": "<one sentence>", "component": "<component>",
  "action": {"thought": "<one or two sentences>", "code": "<the python code, without markdown fences>"}}]}""") % (
        n_suspects, comp_text))


def search_prompt(rec: dict) -> str:
    return "=== FAILED EPISODE ===\n" + _steps_text(rec, rec["n_steps"])


INTRO_HEAD = """An autonomous code agent (an LLM inside a harness) failed a task of AppWorld: \
it writes Python that calls simulated apps' APIs to do a supervisor's task. Below is the \
failure as the developers' analysis tools see it: the supervisor and the instruction, the \
agent's steps (each numbered, with its code and the output), and the grading (the task's \
unit tests).

The DECISIVE MISTAKE is the earliest agent step such that, had the agent acted correctly at \
that step and continued normally, the task would most likely have succeeded.

Harness components (a fix to the harness changes one of these):
"""

COMPONENTS = {
    "prompt": "the prompt text the model reads (instructions, demonstrations, output format)",
    "control_flow": "the agent loop code: how the next action is produced, checks or gates "
                    "applied to code before it runs, retries, enforced verification steps",
    "config": "numeric settings such as temperature, token limits or step limits",
    "output_plumbing": "how execution outputs and API documentation are rendered to the model",
    "context_mgmt": "which parts of the episode history the model sees",
    "client_tool": "extra tools or helper code the harness gives the model",
    "skill": "instruction files loaded for specific situations",
    "memory": "structured state the harness keeps across steps",
    "subagent": "an extra model call, e.g. a second model reviewing the code before it runs",
}
