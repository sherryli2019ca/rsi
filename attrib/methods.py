"""Attribution methods compared against the counterfactual ground truth
(attrib/groundtruth.py), on failed tau2 episodes.

Every method reads a failure as RRSI's analyst reads it (domains/tau2/common.
render: the customer's hidden instructions, the conversation with tool results,
and the grading section) and returns

  {"step": int | None, "top3": [int], "component": str | None,
   "components3": [str], "calls": int, "fallback": bool, ...}

with `component` from RRSI's vocabulary (rrsi.components.K). Token usage goes
to RRSI_USAGE_LOG, one line per call, tagged with the method as its role.

  last_step       baseline: the last agent step
  first_write     baseline: the first step calling a state-changing tool (else
                  the last step)
  all_at_once     Who&When's all-at-once judge: one call on the whole failure
  step_by_step    Who&When's step-by-step judge: steps in order, one call each
                  on the episode up to that step, until one is called decisive
                  (none: the last step, fallback)
  binary_search   Who&When's binary search: halve the step range, one call per
                  halving, on the episode up to the end of the range
  study1          the single-call attributor of Study 1 (agent_exp/analyst.
                  attribute), with RRSI's components
  rrsi_digest     RRSI's trace digester (failure lens): the earliest step its
                  evidence cites (none: the last step, fallback); no component

  python -m attrib.methods run --domain tau2_retail --failures <failures.json> \
      --out <dir> --methods all_at_once:pro,step_by_step:pro,... [--workers 8]

Model names: flash (deepseek-v4-flash), pro (deepseek-v4-pro, thinking off),
pro_think (deepseek-v4-pro, thinking on). Results: <out>/<method>/<fid>.json,
resume-safe.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rrsi.components import K  # noqa: E402

MODELS = {"flash": ("deepseek-v4-flash", False), "pro": ("deepseek-v4-pro", False),
          "pro_think": ("deepseek-v4-pro", True)}

COMPONENTS = {
    "prompt": "the system prompt text (instructions and policy wording the model reads)",
    "control_flow": "the agent loop code: how the next action is produced, checks or gates "
                    "applied to tool calls before they run, retries, enforced confirmations",
    "config": "numeric settings such as temperature, token limits or step limits",
    "output_plumbing": "how tool results and tool descriptions are rendered to the model",
    "context_mgmt": "which parts of the conversation history the model sees",
    "client_tool": "extra tools the harness gives the model (e.g. a checker or calculator)",
    "skill": "instruction files loaded for specific situations",
    "memory": "structured state the harness keeps across steps",
    "subagent": "an extra model call, e.g. a second model reviewing an action",
}
assert set(COMPONENTS) == set(K)
COMP_TEXT = "\n".join(f"- {k}: {v}" for k, v in COMPONENTS.items())

INTRO = """A customer-service agent (an LLM inside a harness) failed a task of tau2-bench. \
Below is the failure as the developers' analysis tools see it: the customer's hidden \
instructions, the conversation (each agent step is numbered), and the grading.

The DECISIVE MISTAKE is the earliest agent step such that, had the agent acted correctly at \
that step and continued normally, the task would most likely have succeeded.

Harness components (a fix to the harness changes one of these):
""" + COMP_TEXT


# ------------------------------------------------------------------- views --
_SCEN: dict = {}


def _scenario(task_id) -> str:
    """The user simulator's instructions for the task (what RRSI's analyst sees
    as the customer's hidden instructions)."""
    if not _SCEN:
        from agent_exp import tau2_env as m
        _SCEN.update({str(t.id): m.scenario_text(t) for t in m.get_tasks("base")})
    return _SCEN.get(str(task_id), "")


def view(rec: dict, upto: int | None = None) -> str:
    """The failure as RRSI's analyst reads it; with `upto`, the conversation
    stops after that step (the grading section still describes the whole
    episode)."""
    from domains.tau2.common import render
    rec = {**rec, "scenario": rec.get("scenario") or _scenario(rec["task_id"])}
    full = render(rec)
    if upto is None:
        return full
    part = copy.deepcopy(rec)
    part["steps"] = [s for s in rec["steps"] if s["index"] <= upto]
    head = render(part).split("\n\nRUN: ")[0]
    grading = full[full.index("=== GRADING"):]
    return f"{head}\n\n(The conversation continues after step {upto}.)\n\n{grading}"


def _call(prompt: str, method: str, model: str) -> dict:
    from rrsi.llm import generate
    mdl, think = MODELS[model]
    role = f"{method}:{model}"
    if think:
        from rrsi import llm
        llm.THINKING_ROLES.add(role)
    # thinking runs past its nominal budget on this endpoint: room for it, few retries
    raw = generate(prompt, json_only=True, model=mdl, role=role,
                   max_tokens=24000 if think else 2000, max_retries=3 if think else 6)
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        v = {}
    return v if isinstance(v, dict) else {}


def _int(x, n: int):
    try:
        k = int(x)
    except (TypeError, ValueError):
        return None
    return k if 0 <= k < n else None


def _comp(x):
    return x if x in COMPONENTS else None


# ----------------------------------------------------------------- methods --
def last_step(rec, model=None):
    n = rec["n_steps"]
    return {"step": n - 1, "top3": [n - 1, n - 2, n - 3][:n], "component": None, "calls": 0}


def first_write(rec, model=None):
    from domains.tau2.common import READ_PREFIXES
    for st in rec["steps"]:
        if any(b["type"] == "tool_use" and not b["name"].startswith(READ_PREFIXES)
               for b in st["assistant"]):
            return {"step": st["index"], "top3": [st["index"]], "component": None, "calls": 0}
    return {**last_step(rec), "fallback": True}


def all_at_once(rec, model="pro"):
    p = (INTRO + "\n\n=== FAILURE ===\n" + view(rec) + """

Find the decisive mistake and the harness component whose change would most likely have \
prevented it. Answer with JSON:
{"step": <step number>, "top3": [<the three most likely step numbers, best first>],
 "component": "<component>", "components3": ["<three most likely components, best first>"],
 "reason": "<one sentence>"}""")
    v = _call(p, "all_at_once", model)
    n = rec["n_steps"]
    top3 = [k for k in (_int(x, n) for x in v.get("top3") or []) if k is not None][:3]
    step = _int(v.get("step"), n)
    return {"step": step if step is not None else (top3[0] if top3 else None), "top3": top3,
            "component": _comp(v.get("component")),
            "components3": [c for c in (v.get("components3") or []) if c in COMPONENTS][:3],
            "calls": 1, "reason": v.get("reason")}


def step_by_step(rec, model="pro"):
    n = rec["n_steps"]
    for k in range(n):
        p = (INTRO + f"\n\n=== FAILURE, UP TO STEP {k} ===\n" + view(rec, upto=k) + f"""

Judge step {k} only, given what happened before it. Is the agent's action at step {k} the \
decisive mistake (the earliest step whose correction would most likely have made the task \
succeed)? Answer with JSON:
{{"decisive": true | false, "component": "<component, if decisive>", "reason": "<one sentence>"}}""")
        v = _call(p, "step_by_step", model)
        if v.get("decisive") is True:
            return {"step": k, "top3": [k], "component": _comp(v.get("component")), "calls": k + 1,
                    "reason": v.get("reason")}
    return {**last_step(rec), "calls": n, "fallback": True}


def binary_search(rec, model="pro"):
    n = rec["n_steps"]
    lo, hi, calls = 0, n - 1, 0
    comp = None
    while lo < hi:
        mid = (lo + hi) // 2
        p = (INTRO + f"\n\n=== FAILURE, UP TO STEP {hi} ===\n" + view(rec, upto=hi) + f"""

The decisive mistake is assumed to lie between step {lo} and step {hi}. Is it in the first \
half (steps {lo} to {mid}) or in the second half (steps {mid + 1} to {hi})? Answer with JSON:
{{"half": "first" | "second", "component": "<the component most likely at fault>",
 "reason": "<one sentence>"}}""")
        v = _call(p, "binary_search", model)
        calls += 1
        comp = _comp(v.get("component")) or comp
        if v.get("half") == "second":
            lo = mid + 1
        else:
            hi = mid
    return {"step": lo, "top3": [lo], "component": comp, "calls": calls}


def study1(rec, model="pro"):
    p = f"""An LLM agent failed a task. Find the root cause of the failure.

Agent components (id: role):
{COMP_TEXT}

Trace:
{view(rec)}

Give (1) a one-sentence root cause describing the cause, not the symptom;
(2) the index of the earliest step where the agent went wrong;
(3) the single component whose change would most plausibly prevent this failure.
Answer with JSON: {{"root_cause": "...", "step": <int>, "component": "<component id>"}}"""
    v = _call(p, "study1", model)
    step = _int(v.get("step"), rec["n_steps"])
    return {"step": step, "top3": [step] if step is not None else [], "component": _comp(v.get("component")),
            "calls": 1, "reason": v.get("root_cause")}


def rrsi_digest(rec, model="pro"):
    import tempfile
    from rrsi import llm
    from rrsi.digester import digest_task
    from domains.tau2.briefs import DIGESTER
    brief = DIGESTER.replace("{domain}", os.environ["TAU2_DOMAIN"])
    mdl, think = MODELS[model]
    with tempfile.TemporaryDirectory() as d:
        tid = f"t{rec['task_id']}"
        (Path(d) / f"{tid}.txt").write_text(view(rec))
        if think:
            llm.THINKING_ROLES.add("digester")
        dg = digest_task(Path(d), tid, "failure", brief, model=mdl)
    steps = []
    for e in (dg or {}).get("evidence") or []:
        m = re.search(r"step\s+(\d+)", str((e or {}).get("where", "")))
        if m and int(m.group(1)) < rec["n_steps"]:
            steps.append(int(m.group(1)))
    steps = sorted(set(steps))
    if not steps:
        return {**last_step(rec), "calls": 1, "fallback": True, "digest": dg}
    return {"step": steps[0], "top3": steps[:3], "component": None, "calls": 1, "digest": dg}


METHODS = {"last_step": last_step, "first_write": first_write, "all_at_once": all_at_once,
           "step_by_step": step_by_step, "binary_search": binary_search, "study1": study1,
           "rrsi_digest": rrsi_digest}


def run(fails: list, out: Path, specs: list[str], workers: int) -> None:
    jobs = []
    for spec in specs:
        name, _, model = spec.partition(":")
        if name not in METHODS or (model and model not in MODELS):
            raise SystemExit(f"unknown method {spec}")
        for f in fails:
            p = out / spec.replace(":", "_") / f"{f['fid']}.json"
            if not p.exists():
                jobs.append((p, name, model or None, f))

    def one(job):
        p, name, model, f = job
        rec = json.loads(Path(f["trace"]).read_text())
        res = METHODS[name](rec, model) if model else METHODS[name](rec)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"fid": f["fid"], "method": name, "model": model, **res},
                                ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(one, jobs):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline"))
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.environ["TAU2_DOMAIN"] = args.domain.split("_", 1)[1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(out / "usage.jsonl"))
    run(json.loads(Path(args.failures).read_text()), out, args.methods.split(","), args.workers)


if __name__ == "__main__":
    main()
