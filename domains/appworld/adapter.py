"""AppWorld as an RRSI domain.

Evaluate = the frozen driver (domains/appworld/driver.py, run from the MAIN
checkout with the AppWorld virtualenv) on the candidate's harness
(<worktree>/domains/appworld/harness). Trial s of job J for task t lives at
runs/appworld/jobs/J/s<s>/<t>.json; AppWorld's own outputs for the job are
under runs/appworld/jobs/J/_appworld. A trial's reward is 1 when every unit
test of the task's AppWorld evaluator passes (the benchmark's task goal
completion); C is the policy's tokens per trial (input + output + cache
reads, every policy call the harness makes). A missing trial counts as a
failure; a trial the harness crashed counts as a failure and is reported in
extra["harness_errors"].

Evolve = the first 20 scenarios of AppWorld's train split (60 tasks),
held-out = test_normal (168 tasks); smoke = two short train tasks outside the
evolve set (split.json).

AppWorld's task data may only be redistributed in encrypted form, so nothing
derived from it is stored here: the task-specific leakage patterns (names,
quoted values and answers of the evolve tasks) are built at load time from
the local data, and run outputs (trajectories) must not be published in the
clear.

Modified for this project; the Domain interface is from google-research/rrsi.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # MAIN checkout (frozen driver)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
from domains.appworld import briefs as B    # noqa: E402

AW_PY = os.environ.get("APPWORLD_PYTHON", "/home/user/venv-aw/bin/python")
AW_ROOT = os.environ.get("APPWORLD_ROOT", "/tmp/claude-0/aw_root")
MAX_OBS = 1200       # chars of an execution output shown in a compact trace
MAX_FAIL = 1500      # chars of a failed test's trace shown in a compact trace


def _added(pat: str) -> str:
    """Match only lines the diff ADDS (context and removed lines are ignored)."""
    return r"(?m)^\+(?!\+\+).*(?:" + pat + ")"


STATIC_CRITIC = [
    (r"\b[0-9a-f]{7}_\d{1,2}\b", "AppWorld task id in diff"),
    (r"\b(evaluator|evaluate_task|evaluate_tasks|evaluate_dataset|TestTracker|ground_truth|"
     r"GroundTruth|compiled_solution|private_data|public_data|test_data|required_apps|"
     r"task_completed)\b|\b(evaluation|solution|generator)\.py\b|"
     r"\b(answer|api_calls|metadata|specs)\.json\b",
     "evaluator / ground-truth reference in diff"),
    (r"\b(import|from)\s+(appworld\w*|domains|rrsi|agent_exp|freezegun)\b|__import__|\bimportlib\b",
     "imports the benchmark, the frozen driver or the search code"),
    (r"APPWORLD_|aw_root|_appworld\b|base_dbs|experiments/outputs|data/tasks|datasets/|"
     r"\btest_normal\b|\btest_challenge\b|\b(train|dev)\.txt\b|\.jsonl\b|README_BEFORE_SHARING",
     "AppWorld data / output path in diff"),
    (r"\.write_text\(|\.write_bytes\(|\bopen\([^)\n]*[\"'][wax]b?\+?[\"']|"
     r"\bos\.(remove|unlink|makedirs|mkdir|rename|replace|rmdir)\(|\bshutil\.|"
     r"\bjson\.dump\(|\bpickle\.dump|\bsqlite3\b",
     "writes files (no state may persist across episodes)"),
    (r"\bos\.environ\b|\bgetenv\(", "reads the process environment"),
    (r"\b(import|from)\s+(subprocess|socket|urllib|requests|httpx|http)\b|"
     r"\b(subprocess|socket|urllib|httpx)\.", "spawns processes or uses the network"),
    (r"\brequester\s*\.|\bapis\._|\brequest_tracker\b|\bsafety_guard\b",
     "touches AppWorld internals from the REPL"),
    (r"[\w.+-]+@(?!example\.)[\w-]+\.[a-z][\w.-]*", "e-mail address in diff"),
    (r"\b\d{10}\b", "phone number in diff"),
]

# Capitalized words of the evolve instructions that are not task-specific.
_COMMON = set("""I A An The This That These Those It Its If Then Here There What Which Who Whom
How When Where Why Give Like Export Add Name Follow Arrange Remove Start Keep Compress Move
Mark Accept Reject Play Send Make Reply Some All Any Each Every Terminate Delete Create Find
Set Update Buy Order Return Pay Request Share Download Upload Rate Unfollow Unlike Repeat
Spotify Venmo Amazon Gmail Splitwise Todoist Phone Simple Note SimpleNote File System Supervisor
January February March April May June July August September October November December
Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec Monday Tuesday Wednesday Thursday Friday
Saturday Sunday Title Artists R&B""".split())


def _value_patterns(task_ids: list[str], data_root: Path) -> list[tuple[str, str]]:
    """Patterns for the evolve / smoke tasks' own values, from the local
    AppWorld data (never written to the repository): supervisor names,
    proper nouns and quoted strings of the instructions, ground-truth answers."""
    names, quoted, answers = set(), set(), set()
    for t in task_ids:
        d = data_root / "tasks" / t
        try:
            spec = json.loads((d / "specs.json").read_text())
        except (OSError, ValueError):
            continue
        sup = spec.get("supervisor") or spec.get("main_user") or {}
        if sup.get("first_name") and sup.get("last_name"):
            names.add(f"{sup['first_name']} {sup['last_name']}")
        ins = str(spec.get("instruction") or "")
        for q in re.findall(r'"([^"\n]{3,})"', ins):
            if len(q) >= 8 or re.search(r"[\s/~.!?&|<>]", q):
                quoted.add(q)
        # runs of capitalized words outside quotes, common words stripped at both ends
        for run in re.findall(r"\b[A-Z][\w&'-]*(?:[ \t]+[A-Z][\w&'-]*)*", re.sub(r'"[^"\n]*"', " , ", ins)):
            words = run.split()
            while words and (words[0] in _COMMON or words[0].startswith("I'")):
                words.pop(0)
            while words and (words[-1] in _COMMON or words[-1].startswith("I'")):
                words.pop()
            if words and len(" ".join(words)) >= 3:
                names.add(" ".join(words))
        try:
            ans = json.loads((d / "ground_truth" / "answer.json").read_text())
        except (OSError, ValueError):
            ans = None
        for a in (ans if isinstance(ans, list) else [ans]):
            if isinstance(a, bool) or a is None:
                continue
            if isinstance(a, (int, float)):
                if abs(a) >= 1000:
                    answers.add(rf"\b{int(a)}(?:\.0+)?\b" if float(a).is_integer()
                                else rf"\b{re.escape(str(a))}\b")
            elif isinstance(a, str) and len(a.strip()) >= 4 and re.search(r"[A-Za-z]", a):
                answers.add(r"(?i:" + r"\s+".join(map(re.escape, a.split())) + ")")

    def alt(xs):
        return "|".join(sorted(xs, key=lambda x: (-len(x), x)))
    out = []
    if names:
        out.append((r"\b(?:" + alt(r"\s+".join(map(re.escape, n.split())) for n in names) + r")\b",
                    "person / place name from an evolve task in diff"))
    if quoted:
        out.append((r"(?i:" + alt(re.escape(q) for q in quoted) + ")",
                    "quoted value from an evolve task's instruction in diff"))
    if answers:
        out.append((alt(answers), "ground-truth answer of an evolve task in diff"))
    return out


COMPONENT_SIGNALS = [
    ("subagent",        [r"\bsubcall\(", r"(?m)^\+.*\bsub_?agent"]),
    ("client_tool",     [r"ToolRegistry", r"run_with_client_tools", r"(?m)^\+.*\.register\(",
                         r"\bprepend_to_code\(", r"(?m)^\+.*\bREPL_HELPERS\b"]),
    ("skill",           [r"skills/", r"load_skills", r"skill_catalog"]),
    ("context_mgmt",    [r"(?m)^\+.*\b(summar\w*|compact\w*|trim\w*|truncat\w*|keep_last|"
                         r"history_window|elide\w*)\b"]),
    ("output_plumbing", [r"(?m)^(@@.*def (render_output|format_output)|[+-].*\b(render_output|"
                         r"format_output)\b)"]),
    ("control_flow",    [r"(?m)^(@@.*def (next_action|before_execute|observe_prefix)|"
                         r"[+-].*\b(next_action|before_execute|observe_prefix)\b)"]),
    ("config",          [r"(?m)^\+\s*[A-Z][A-Z0-9_]{2,}\s*=\s*[\d.]+",
                         r"(?m)^\+.*max_tokens\s*=\s*\d+"]),
    # Code in agent.py or another module except prompts.py that no signal
    # above names is control flow (same correction as domains/tau2/common.py).
    ("control_flow",    [r"(?m)^\+\+\+ b/\S*harness/(?!prompts\.py$)\S+\.py$"]),
    ("prompt",          [r"(?m)^(\+\+\+|---) .*prompts\.py", r"REACT_PROMPT", r"prompt_messages",
                         r"render_prompt"]),
]


def _lines(tag: str, label: str, text: str) -> list[str]:
    """A multi-line block with every line prefixed by the step tag, so that
    step-range slicing of the rendered trace keeps or drops it whole."""
    body = str(text).rstrip("\n").splitlines() or [""]
    return [f"{tag} {label}"] + [f"{tag}   {l}" for l in body]


def _prose(text: str) -> str:
    """Assistant text without its first python block (shown as CODE)."""
    return re.sub(r"```python[ \t]*\n.*?(```|$)", "", str(text), count=1, flags=re.S).strip()


def render(rec: dict, detail: bool = False) -> str:
    """A trial as the analyst / digester / proposer read it."""
    sup = rec.get("supervisor") or {}
    out = [f"TASK {rec.get('task_id')}  (trial {rec.get('trial')})",
           f"SUPERVISOR: {sup.get('first_name')} {sup.get('last_name')}, {sup.get('email')}, "
           f"phone {sup.get('phone_number')}",
           f"INSTRUCTION: {rec.get('instruction')}", ""]
    for st in rec.get("steps") or []:
        tag = f"[step {st['index']}]"
        flag = " [replayed prefix]" if st.get("prefix") else ""
        prose = _prose(st.get("assistant") or "")
        if prose:
            out += _lines(tag, f"AGENT{flag}:", prose)
        out += _lines(tag, f"CODE{flag}:", st.get("code") or "(no python block: nothing executed)")
        obs = str(st.get("output") or "")
        if not detail and len(obs) > MAX_OBS:
            obs = obs[:MAX_OBS] + f" ...[{len(obs) - MAX_OBS} more chars]"
        out += _lines(tag, "BLOCKED BY HARNESS (not executed), shown to the agent:"
                      if st.get("blocked") else "OUTPUT:", obs)
    tok = rec.get("tokens") or {}
    out += ["", "RUN: " + f"stopped={rec.get('stopped')} steps={rec.get('n_steps')} "
            f"completed={rec.get('completed')} policy_tokens={tok.get('agent_total')} "
            f"calls={(rec.get('calls') or {}).get('agent')}"]
    if rec.get("replay"):
        out.append(f"REPLAY: {json.dumps(rec['replay'])}")
    if rec.get("harness_error"):
        out += ["HARNESS ERROR (the harness code raised; the episode was scored 0):",
                str(rec["harness_error"])[-1500:]]
    tests = rec.get("tests") or {}
    out += ["", "=== GRADING (AppWorld evaluator; the agent never sees it) ===",
            f"reward={rec.get('reward')}  tests passed {tests.get('passed')}/{tests.get('total')}  "
            f"difficulty={tests.get('difficulty')}  complete_task called={rec.get('completed')}"]
    fails = tests.get("failures") or []
    if fails:
        out.append("FAILED tests (requirement, test code, assertion):")
        for f in fails:
            tr = str(f.get("trace") or "")
            if not detail and len(tr) > MAX_FAIL:
                tr = tr[:MAX_FAIL] + f" ...[{len(tr) - MAX_FAIL} more chars]"
            out.append(f"  - {f.get('requirement')}")
            out += [f"      {l}" for l in tr.splitlines()]
    if tests.get("passes"):
        out.append("passed tests:")
        out += [f"  - {p}" for p in tests["passes"]]
    return "\n".join(out)


class AppWorldDomain(Domain):
    name = "appworld"
    harness_path = "harness"
    component_signals = COMPONENT_SIGNALS

    def __init__(self):
        self.here = REPO / "domains" / self.name
        self.cfg = json.loads((self.here / "rrsi.json").read_text())
        self.split = json.loads((self.here / "split.json").read_text())
        scen = sorted({t.split("_")[0] for k in ("evolve", "heldout", "smoke")
                       for t in self.split.get(k, [])})
        pats = STATIC_CRITIC + [(r"\b(?:" + "|".join(scen) + r")\b", "AppWorld scenario id in diff")]
        pats += _value_patterns(list(self.split["evolve"]) + list(self.split.get("smoke", [])),
                                Path(AW_ROOT) / "data")
        self.critic_patterns = [(_added(p), why) for p, why in pats]
        label = self.cfg.get("policy_label", "the frozen policy model")
        self.briefs = {k: v.replace("{policy}", label)
                       for k, v in {"analyst": B.ANALYST, "digester": B.DIGESTER,
                                    "proposer": B.PROPOSER, "critic": B.CRITIC}.items()}

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self):
        # RRSI_APPWORLD_EVOLVE_LIMIT: first N evolve tasks only, for smoke runs of the loop
        n = int(os.environ.get("RRSI_APPWORLD_EVOLVE_LIMIT") or 0)
        ids = list(self.split["evolve"])
        return ids[:n] if n else ids

    def heldout_ids(self):
        n = int(os.environ.get("RRSI_APPWORLD_HELDOUT_LIMIT") or 0)      # smoke runs only
        ids = list(self.split["heldout"])
        return ids[:n] if n else ids

    def smoke_ids(self, incumbent_per_task=None):
        return list(self.split["smoke"])

    # ---- Evaluate ----------------------------------------------------------
    def _driver(self, root: Path, out: Path, log: Path, extra: list[str]) -> None:
        hdir = self.harness_dir(Path(root))
        cmd = [AW_PY, "-m", "domains.appworld.driver", "--harness", str(hdir),
               "--out", str(out), "--workers", str(self.cfg.get("concurrency", 6))]
        t = self.cfg.get("policy_temperature")
        if t is not None:
            cmd += ["--temperature", str(t)]
        env = {**os.environ, "PYTHONPATH": str(REPO), "APPWORLD_ROOT": AW_ROOT}
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as lf:
            subprocess.run(cmd + extra, cwd=str(REPO), env=env, stdout=lf, stderr=subprocess.STDOUT)

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        runs_dir = Path(runs_dir)
        out = runs_dir / "jobs" / job
        out.mkdir(parents=True, exist_ok=True)
        idf = out / "_ids.json"
        idf.write_text(json.dumps(list(ids)))
        for attempt in range(3):
            self._driver(root, out, runs_dir / "logs" / f"{job.replace('/', '_')}.log",
                         ["--ids", str(idf), "--k", str(k)])
            miss = sum(1 for s in range(k) for t in ids if not (out / f"s{s}" / f"{t}.json").exists())
            if not miss:
                return
            print(f"[{self.name}] {log_prefix}{job}: {miss} trials missing after attempt "
                  f"{attempt + 1}", flush=True)

    def replay(self, root, runs_dir, job, refs: list[dict]) -> Path:
        """Mid-step replays with the harness under `root`: refs are
        {"key", "task_id", "trace" (path or record), "start"}; output in
        runs_dir/replays/<job>/<key>.json."""
        out = Path(runs_dir) / "replays" / job
        out.mkdir(parents=True, exist_ok=True)
        rf = out / "_refs.json"
        rf.write_text(json.dumps(refs, default=str))
        self._driver(root, out, Path(runs_dir) / "logs" / f"replay_{job.replace('/', '_')}.log",
                     ["--replay", str(rf)])
        return out

    def _trial(self, runs_dir, job, task_id, s):
        p = Path(runs_dir) / "jobs" / job / f"s{s}" / f"{task_id}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return None

    def score(self, runs_dir, job, ids, k):
        per, n = {}, {"trials": 0, "harness_errors": 0, "completed": 0, "max_steps": 0,
                      "steps": 0, "tests_passed": 0, "tests_total": 0}
        for t in ids:
            rewards, toks, missing = [], [], 0
            for s in range(k):
                rec = self._trial(runs_dir, job, t, s)
                if rec is None:
                    rewards.append(0.0)
                    toks.append(None)
                    missing += 1
                    continue
                rewards.append(float(rec.get("reward") or 0))
                toks.append((rec.get("tokens") or {}).get("agent_total") or None)
                n["trials"] += 1
                n["harness_errors"] += int("harness_error" in rec)
                n["completed"] += int(bool(rec.get("completed")))
                n["max_steps"] += int(rec.get("stopped") == "max_steps")
                n["steps"] += int(rec.get("n_steps") or 0)
                tests = rec.get("tests") or {}
                n["tests_passed"] += int(tests.get("passed") or 0)
                n["tests_total"] += int(tests.get("total") or 0)
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing)
        d = max(1, n["trials"])
        extra = {"trials": n["trials"], "harness_errors": n["harness_errors"],
                 "harness_error_rate": n["harness_errors"] / d,
                 "completed_rate": n["completed"] / d, "max_steps_rate": n["max_steps"] / d,
                 "mean_steps": n["steps"] / d,
                 "test_pass_rate": n["tests_passed"] / max(1, n["tests_total"])}
        return per, extra

    def guards(self, incumbent, candidate):
        g = []
        rate = (candidate.extra or {}).get("harness_error_rate", 0.0)
        if rate > self.cfg.get("max_harness_error_rate", 0.02):
            g.append(f"harness raised in {rate:.1%} of trials")
        return g

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        rec = self._trial(runs_dir, job, task_id, trial)
        if rec is None:
            return None
        rec["trial"] = trial
        return rec

    def render_trace(self, rec, detail=False):
        return render(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        tests = rec.get("tests") or {}
        return (f"{task_id} | reward {rec.get('reward')} (mean over trials {tr.mean:.2f}) | "
                f"tests {tests.get('passed')}/{tests.get('total')} | "
                f"complete_task={rec.get('completed')} | steps={rec.get('n_steps')} | "
                f"stopped={rec.get('stopped')}"
                + (" | HARNESS ERROR" if rec.get("harness_error") else ""))

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        hdir = self.harness_dir(root)
        comp = subprocess.run([AW_PY, "-m", "compileall", "-q", str(hdir)],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "detail": (comp.stdout + comp.stderr)[-1500:]}
        shutil.rmtree(runs_dir / "jobs" / job, ignore_errors=True)
        self.run(root, runs_dir, job, ids, 1)
        errs, done = [], 0
        for t in ids:
            rec = self._trial(runs_dir, job, t, 0)
            if rec is None:
                continue
            done += 1
            if rec.get("harness_error"):
                errs.append(str(rec["harness_error"])[-1200:])
        if errs:
            return False, {"stage": "smoke_run", "detail": errs[0]}
        if done < len(ids):
            return False, {"stage": "smoke_run", "detail": f"{done}/{len(ids)} episodes finished"}
        return True, {"stage": "smoke_run", "n": done}


DOMAIN = AppWorldDomain()
