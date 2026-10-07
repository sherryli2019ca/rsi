"""Frozen episode driver for the AppWorld RRSI domain (domains/appworld).

Runs an evolvable harness (domains/appworld/harness in some worktree) on
AppWorld tasks (Trivedi et al., ACL 2024; package appworld 0.1.3) with the
benchmark's own environment and evaluator. The harness sees only the contract
documented in domains/appworld/base_harness/agent.py.

One episode: a fresh AppWorld world for the task (time frozen at the task's
date, databases loaded from the task's snapshot; random_seed 100 and unknown
API parameters raise, as in the official ReAct baseline config). The harness
gets the task instruction, the supervisor and the app descriptions. Up to
MAX_STEPS = 40 agent steps, each: next_action -> assistant text; the driver
extracts the code (extract_code: the first ```python block, as the official
ReAct agent does) and passes it through before_execute; world.execute runs it
unless blocked; render_output turns the raw output into the next user turn.
The episode stops when the task is completed (the agent called
apis.supervisor.complete_task) or at the step limit, and is graded by
world.evaluate(): reward 1 iff every test of the task passes.

Process model. AppWorld keeps process-global state (opening a world closes
every other world of the process, time is frozen process-wide by freezegun,
execution timeouts use SIGALRM and need the main thread), so every episode runs
in its own child process (`--episode`, job spec on stdin) and --workers
threads of the parent feed them. AppWorld's own outputs go to
<out>/_appworld/experiments/outputs/<experiment>/tasks/<task_id>
(APPWORLD_ROOT=<out>/_appworld, whose `data` is a symlink to the installed
data under $APPWORLD_ROOT or /tmp/claude-0/aw_root); experiment names are
unique per episode (s<trial> for fresh runs, replay_<key> for replays).

Two modes, both resume-safe (an existing output file is never re-run):

  fresh   every task in --ids x --k trials, written to <out>/s<trial>/<task>.json
  replay  every reference in --replay (a JSON list of {"key", "task_id",
          "trace", "start"}): the recorded episode `trace` (a trial file of this
          driver, path or record) is re-executed up to step `start` and
          continued live with the harness, written to <out>/<key>.json

Replay with code-level harness changes. Recorded assistant turns in the prefix
are kept and their code re-executed in a fresh world (time and databases are
the task's snapshot, so re-execution is deterministic), but every step goes
through the NEW harness's before_execute / render_output. If the new harness
blocks a recorded step's code or renders a different observation at step i,
the recorded turns after step i were conditioned on something this harness
would not have shown, so the episode goes live from step i + 1 instead of
`start` (recorded as start_effective, with the steps that diverged). Changes
to the prompt do not move the start: as in all replays of this project, they
act from the replayed step on. observe_prefix lets the harness rebuild
per-episode state from the recorded steps.

Failures: an exception raised by harness code (including a harness that cannot
be imported, a request the endpoint rejects with HTTP 400, e.g. malformed
messages or a context that no longer fits, and more than MAX_POLICY_CALLS
policy calls) is recorded as reward 0 with "harness_error"; a policy-endpoint
(network, rate limit, server) or AppWorld failure is infrastructure: the
episode is retried (3 attempts) and otherwise left missing (the adapter scores
a missing trial as 0).

Policy tokens: agent_in (uncached input), agent_cache_read (cached input; the
endpoint caches repeated prefixes automatically), agent_out, and agent_total =
their sum, over every policy call of the episode (live steps only in a replay).

  /home/user/venv-aw/bin/python -m domains.appworld.driver --harness <dir> \
      --out <dir> --ids 82e2fac_1,692c77d_1 --k 2 [--replay refs.json] \
      [--workers 8] [--temperature 0]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import importlib
import importlib.util
import json
import os
import re
import ssl
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

MAX_STEPS = 40              # agent steps per episode, as in every run of this project
MAX_POLICY_CALLS = 200      # policy calls per episode (cost guard; exceeding it is a harness error)
POLICY_MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/anthropic/v1/messages"
APPWORLD_CONFIG = {"random_seed": 100, "raise_on_extra_parameters": True}   # official ReAct config
EPISODE_TIMEOUT = 3600      # seconds per child process
SOURCE_ROOT = "/tmp/claude-0/aw_root"   # default AppWorld root holding data/
# freezegun (started by AppWorld) patches module-level time functions; a tuple
# member keeps the real clock for the episode's wall time.
_CLOCK = (time.monotonic,)


class HarnessError(Exception):
    pass


class PolicyAPIError(Exception):
    """The policy endpoint failed (network, rate limit, server): infrastructure."""


class PolicyRequestError(ValueError):
    """The endpoint rejected the request (HTTP 400/422: malformed messages,
    context too long): the harness's fault."""


# ---- harness loading --------------------------------------------------------

def load_agent_class(harness_dir: Path):
    """Import the harness package found at harness_dir under a unique name."""
    harness_dir = Path(harness_dir).resolve()
    name = "harness_" + hashlib.md5(str(harness_dir).encode()).hexdigest()[:10]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, harness_dir / "__init__.py", submodule_search_locations=[str(harness_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(name + ".agent").Agent


# ---- policy model -------------------------------------------------------------

def _cafile():
    for c in (os.environ.get("SSL_CERT_FILE"), os.environ.get("REQUESTS_CA_BUNDLE"),
              "/root/.ccr/ca-bundle.crt"):
        if c and os.path.exists(c):
            return c
    return None


class PolicyModel:
    """The frozen policy model handed to the harness as `llm`: DeepSeek's
    Anthropic-compatible Messages endpoint over a stdlib HTTP client (the
    outbound proxy injects the key). Model, temperature and thinking (off) are
    fixed here."""

    def __init__(self, temperature=None):
        self.temperature = temperature
        ctx = ssl.create_default_context(cafile=_cafile())
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(urllib.request.getproxies()),
            urllib.request.HTTPSHandler(context=ctx))
        self.key = os.environ.get("DEEPSEEK_API_KEY", "proxy")

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        err = ""
        for attempt in range(6):
            req = urllib.request.Request(ENDPOINT, data=data, method="POST", headers={
                "x-api-key": self.key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"})
            try:
                with self.opener.open(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:600]
                if e.code in (400, 413, 422):
                    raise PolicyRequestError(f"policy endpoint rejected the request: HTTP {e.code}: {msg}")
                if e.code in (401, 403, 404):
                    raise PolicyAPIError(f"HTTP {e.code}: {msg}")
                err = f"HTTP {e.code}: {msg}"
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
                err = repr(e)[:400]
            time.sleep(min(60, 3 * 2 ** attempt))     # no jitter: the global `random` belongs to the REPL
        raise PolicyAPIError(err)

    def make(self):
        acc = {"in": 0, "out": 0, "cache_read": 0, "calls": 0, "attempts": 0, "api_error": None}

        def llm(system, messages, max_tokens=4000, stop=None):
            if acc["attempts"] >= MAX_POLICY_CALLS:
                raise RuntimeError(f"policy call budget exceeded ({MAX_POLICY_CALLS} per episode)")
            acc["attempts"] += 1
            body = {"model": POLICY_MODEL, "max_tokens": int(max_tokens), "messages": messages,
                    "thinking": {"type": "disabled"}}
            if system:
                body["system"] = system
            if self.temperature is not None:
                body["temperature"] = self.temperature
            if stop:
                body["stop_sequences"] = [str(s) for s in stop]
            try:
                resp = self._post(body)
            except PolicyAPIError as e:
                acc["api_error"] = str(e)
                raise
            u = resp.get("usage") or {}
            i, o, c = (int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                       int(u.get("cache_read_input_tokens") or 0))
            acc["in"] += i
            acc["out"] += o
            acc["cache_read"] += c
            acc["calls"] += 1
            text = "".join(str(b.get("text") or "") for b in resp.get("content") or []
                           if b.get("type") == "text")
            return text, {"in": i, "out": o, "cache_read": c, "stop_reason": resp.get("stop_reason")}
        return llm, acc


# ---- episode --------------------------------------------------------------------

_FULL_CODE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)
_PARTIAL_CODE = re.compile(r".*```python[ \t]*\n(.*)", re.S)


def extract_code(text: str) -> str:
    """The code the driver executes for an assistant turn: the first complete
    ```python block, else a trailing unterminated one (a reply cut off at
    max_tokens), else "" (AppWorld answers "No code available to execute.").
    This is the official ReAct agent's rule (ignore_multiple_calls=True)."""
    m = _FULL_CODE.search(text)
    if m:
        return m.group(1).strip()
    m = _PARTIAL_CODE.match(text)
    return m.group(1).strip() if m else ""


def _h(acc, fn, *a):
    """Call harness code; anything it raises is the harness's fault, unless the
    policy endpoint failed underneath it (infrastructure)."""
    acc["api_error"] = None
    try:
        return fn(*a)
    except PolicyAPIError:
        raise
    except (Exception, SystemExit) as e:  # noqa: BLE001 - a harness bug fails the episode
        if acc["api_error"]:
            raise PolicyAPIError(acc["api_error"]) from e
        raise HarnessError(f"{getattr(fn, '__name__', fn)}: {e!r}\n{traceback.format_exc()[-1500:]}")


def _text(x, what: str) -> str:
    """Turns the model sees must be non-empty strings (the endpoint rejects empty content)."""
    if not isinstance(x, str):
        raise HarnessError(f"{what} returned {type(x).__name__}, not str")
    return x if x.strip() else "..."


def _task_view(world) -> dict:
    s = world.task.supervisor
    sup = {k: str(getattr(s, k, None) if not isinstance(s, dict) else s.get(k))
           for k in ("first_name", "last_name", "email", "phone_number")}
    return {"instruction": str(world.task.instruction), "supervisor": sup,
            "app_descriptions": {str(k): str(v) for k, v in world.task.app_descriptions.items()}}


def _step(acc, agent, world, index: int, text: str, code: str) -> dict:
    g = _h(acc, agent.before_execute, code)
    if g is not None:
        return {"index": index, "assistant": text, "code": code, "blocked": True, "raw": None,
                "output": _text(str(g), "before_execute")}
    raw = world.execute(code)
    obs = _h(acc, agent.render_output, code, raw)
    return {"index": index, "assistant": text, "code": code, "blocked": False, "raw": raw,
            "output": _text(obs if isinstance(obs, str) else str(obs), "render_output")}


def episode(Agent, pm: PolicyModel, task_id: str, experiment: str,
            prefix: dict | None = None, start: int = 0) -> dict:
    from appworld import AppWorld
    t0 = _CLOCK[0]()
    llm, acc = pm.make()
    steps, diverged, msgs = [], [], []
    live_from = start if prefix else 0
    completed, err = False, None
    rec: dict = {"task_id": task_id, "experiment": experiment}
    with AppWorld(task_id=task_id, experiment_name=experiment, **APPWORLD_CONFIG) as world:
        task = _task_view(world)
        rec["instruction"], rec["supervisor"] = task["instruction"], task["supervisor"]
        try:
            agent = _h(acc, Agent, llm, copy.deepcopy(task))
            # ---- recorded prefix ------------------------------------------------
            for i, st in enumerate((prefix or {}).get("steps", [])[:start]):
                if i >= live_from:
                    break
                text = str(st["assistant"])
                _h(acc, agent.observe_prefix, copy.deepcopy(msgs), text)
                code = st["code"] if isinstance(st.get("code"), str) else extract_code(text)
                step = _step(acc, agent, world, i, text, code)
                step["prefix"] = True
                if step["blocked"] or step["output"] != st.get("output"):
                    diverged.append({"step": i, "kind": "blocked" if step["blocked"] else "rendered"})
                    live_from = min(live_from, i + 1)
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": step["output"]}]
                steps.append(step)
                if world.task_completed():
                    completed = True
                    break
            if prefix:      # a recording shorter than `start` goes live where it ends
                live_from = min(live_from, len(steps))
            # ---- live -------------------------------------------------------------
            while not completed and len(steps) < MAX_STEPS:
                text = _text(_h(acc, agent.next_action, copy.deepcopy(msgs)), "next_action")
                step = _step(acc, agent, world, len(steps), text, extract_code(text))
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": step["output"]}]
                steps.append(step)
                completed = world.task_completed()
        except HarnessError as e:
            err = str(e)[:3000]
        # ---- grade ---------------------------------------------------------------
        tests = None
        if err is None:
            d = world.evaluate().to_dict()
            tests = {"passed": len(d["passes"]), "total": int(d["num_tests"]),
                     "difficulty": d.get("difficulty"),
                     "passes": [p["requirement"] for p in d["passes"]],
                     "failures": [{"requirement": f["requirement"], "trace": f["trace"]}
                                  for f in d["failures"]]}
    rec.update({
        "reward": int(err is None and tests["passed"] == tests["total"]),
        "tests": tests, "completed": completed,
        "stopped": "harness_error" if err else ("completed" if completed else "max_steps"),
        "steps": steps, "n_steps": len(steps),
        "tokens": {"agent_in": acc["in"], "agent_out": acc["out"],
                   "agent_cache_read": acc["cache_read"],
                   "agent_total": acc["in"] + acc["out"] + acc["cache_read"]},
        "calls": {"agent": acc["calls"]}, "seconds": round(_CLOCK[0]() - t0, 2),
        "replay": ({"start_requested": start, "start_effective": live_from,
                    "diverged": diverged} if prefix else None)})
    if err:
        rec["harness_error"] = err
    return rec


def child_main() -> None:
    """One episode in this process; spec on stdin, record to spec["part"].
    Exit 0 = record written (also for harness errors), 3 = infrastructure."""
    spec = json.loads(sys.stdin.read())
    part = Path(spec["part"])
    try:
        pm = PolicyModel(spec.get("temperature"))
        prefix = spec.get("trace")
        if prefix is not None and not isinstance(prefix, dict):
            prefix = json.loads(Path(prefix).read_text())
        try:
            Agent = load_agent_class(Path(spec["harness"]))
        except Exception as e:  # noqa: BLE001 - an unimportable harness fails the episode
            rec = {"task_id": spec["task_id"], "reward": 0, "tests": None, "completed": False,
                   "stopped": "harness_error", "steps": [], "n_steps": 0, "tokens": {},
                   "calls": {"agent": 0}, "seconds": 0.0, "replay": None,
                   "harness_error": f"import: {e!r}\n{traceback.format_exc()[-1500:]}"}
        else:
            rec = episode(Agent, pm, spec["task_id"], spec["experiment"], prefix,
                          int(spec.get("start") or 0))
    except Exception as e:  # noqa: BLE001 - API or environment failure: the parent retries
        sys.stderr.write(traceback.format_exc()[-3000:] + f"\nINFRA: {e!r}"[:600] + "\n")
        sys.stderr.flush()
        os._exit(3)
    part.write_text(json.dumps(rec, default=str))
    sys.stdout.flush()
    os._exit(0)     # skip interpreter teardown of AppWorld's global state


# ---- job runner -------------------------------------------------------------------

def run_jobs(jobs, harness: Path, temperature, aw_root: Path, workers: int):
    """jobs: [(out_path, task_id, trace (path, record or None), start, experiment)].
    Resume-safe; infrastructure failures retried 3x; writes are atomic."""
    env = {**os.environ, "APPWORLD_ROOT": str(aw_root),
           "PYTHONPATH": os.pathsep.join([str(REPO)] + [p for p in
                                         os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}

    def one(job):
        out, task_id, trace, start, experiment = job
        if out.exists():
            return "skip"
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        spec = {"harness": str(harness), "task_id": task_id, "experiment": experiment,
                "part": str(part), "temperature": temperature, "trace": trace, "start": start}
        err = ""
        for attempt in range(3):
            part.unlink(missing_ok=True)
            try:
                p = subprocess.run([sys.executable, "-m", "domains.appworld.driver", "--episode"],
                                   input=json.dumps(spec, default=str), capture_output=True,
                                   text=True, timeout=EPISODE_TIMEOUT, cwd=str(REPO), env=env)
                ok = p.returncode == 0 and part.exists()
                err = "" if ok else f"exit {p.returncode}: {(p.stderr or p.stdout or '')[-1500:]}"
            except subprocess.TimeoutExpired:
                ok, err = False, f"episode exceeded {EPISODE_TIMEOUT}s"
            if ok:
                try:
                    rec = json.loads(part.read_text())
                except ValueError as e:
                    err = f"unreadable record: {e!r}"
                else:
                    os.replace(part, out)
                    tok = (rec.get("tokens") or {}).get("agent_total")
                    print(f"{out.relative_to(out.parents[1])}: reward={rec.get('reward')} "
                          f"steps={rec.get('n_steps')} tokens={tok} calls="
                          f"{(rec.get('calls') or {}).get('agent')} s={rec.get('seconds')}"
                          + (" HARNESS_ERROR" if rec.get("harness_error") else ""), flush=True)
                    return "harness_error" if rec.get("harness_error") else "ok"
            time.sleep(5 * (attempt + 1))
        part.unlink(missing_ok=True)
        print(f"[infra] {out}: {err[-1200:]}", flush=True)
        return "infra"

    with ThreadPoolExecutor(max(1, workers)) as ex:
        return list(ex.map(one, jobs))


def _safe(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--harness")
    ap.add_argument("--out")
    ap.add_argument("--ids", default="")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=None)
    args = ap.parse_args()
    if args.episode:
        return child_main()
    if not args.harness or not args.out:
        ap.error("--harness and --out are required")
    src = Path(os.environ.get("APPWORLD_ROOT") or SOURCE_ROOT)
    data = (src / "data").resolve()
    if not (data / "tasks").is_dir():
        raise SystemExit(f"no AppWorld data under {src} (set APPWORLD_ROOT)")
    out = Path(args.out).resolve()
    aw_root = out / "_appworld"
    aw_root.mkdir(parents=True, exist_ok=True)
    try:
        (aw_root / "data").symlink_to(data, target_is_directory=True)
    except FileExistsError:
        pass
    if args.replay:
        refs = json.loads(Path(args.replay).read_text())
        jobs = [(out / f"{r['key']}.json", str(r["task_id"]), r["trace"], int(r["start"]),
                 f"replay_{_safe(r['key'])}") for r in refs]
    else:
        ids = [x for x in args.ids.split(",") if x] if not os.path.exists(args.ids) else \
            json.loads(Path(args.ids).read_text())
        jobs = [(out / f"s{s}" / f"{tid}.json", str(tid), None, 0, f"s{s}")
                for s in range(args.k) for tid in ids]
    unknown = sorted({j[1] for j in jobs if not (data / "tasks" / j[1] / "specs.json").exists()})
    if unknown:
        raise SystemExit(f"unknown AppWorld task ids: {unknown[:10]}")
    res = run_jobs(jobs, Path(args.harness).resolve(), args.temperature, aw_root, args.workers)
    summary = {s: res.count(s) for s in set(res)}
    print(json.dumps({"jobs": len(jobs), **summary}), flush=True)


if __name__ == "__main__":
    main()
