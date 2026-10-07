"""Shared RRSI domain adapter for tau2-bench (tau2_retail, tau2_airline).

Evaluate = the frozen driver (domains/tau2/driver.py, run from the MAIN
checkout with the tau2 virtualenv) on the candidate's harness
(<worktree>/domains/tau2_<d>/harness). Trial s of job J for task t lives at
runs/<domain>/jobs/J/s<s>/<t>.json. A trial's reward is 1 when the final
database equals the gold one and every communicate_info string was said (the
reward used throughout agent_exp); C is the policy's tokens per trial
(input + output + cache reads, every policy call the harness makes). A missing
trial counts as a failure; a trial the harness crashed counts as a failure and
is reported in extra["harness_errors"].

Evolve / held-out are tau2-bench's own train / test splits (split.json).

Modified for this project; the Domain interface is from google-research/rrsi.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # MAIN checkout (frozen driver)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
from domains.tau2 import briefs as B    # noqa: E402

TAU2_PY = os.environ.get("TAU2_PYTHON", "/home/user/venv-tau2/bin/python")
MAX_OBS = 600        # chars of a tool result shown in a compact trace
READ_PREFIXES = ("get_", "find_", "list_", "search_", "calculate", "think")


def _added(pat: str) -> str:
    """Match only lines the diff ADDS (context and removed lines are ignored)."""
    return r"(?m)^\+(?!\+\+).*" + pat


COMMON_CRITIC = [
    (_added(r"\b(gold_hash|get_db_hash|db_hash|evaluation_criteria|communicate_info|"
            r"nl_assertions|gold_actions|get_tasks)\b"), "grader / task-data reference in diff"),
    (_added(r"\b(import|from)\s+(tau2|agent_exp|domains|rrsi)\b"),
     "imports the benchmark, the frozen driver or the search code"),
    (_added(r"(DATA_DIR|tasks\.json|split_tasks|simulation_guidelines|user_simulator)"),
     "benchmark data access"),
    (_added(r"(\.write_text\(|\.write_bytes\(|open\([^)]*[\"'][wa]b?\+?[\"'])"),
     "writes files (no state may persist across episodes)"),
    (_added(r"@example\.com"), "customer e-mail address in diff"),
    (_added(r"\b[a-z]+_[a-z]+_\d{4}\b"), "customer user id in diff"),
]

COMPONENT_SIGNALS = [
    ("subagent",        [r"\bsubcall\(", r"(?m)^\+.*self\.llm\([^)\n]*\bNone\b"]),
    ("client_tool",     [r"ToolRegistry", r"run_with_client_tools", r"(?m)^\+.*\.register\("]),
    ("skill",           [r"skills/", r"load_skills", r"skill_catalog"]),
    ("context_mgmt",    [r"(?m)^\+.*\b(summar\w*|compact\w*|trim\w*|keep_last|history_window)\b"]),
    ("output_plumbing", [r"(?m)^(@@.*def render_result|[+-].*\brender_result\b)",
                         r"(?m)^(\+\+\+|---) .*tools\.py", r"(?m)^@@.*def describe"]),
    ("control_flow",    [r"(?m)^(@@.*def (next_action|before_tool|observe_prefix)|"
                         r"[+-].*\b(next_action|before_tool|observe_prefix)\b)"]),
    ("config",          [r"(?m)^\+\s*[A-Z][A-Z0-9_]{2,}\s*=\s*[\d.]+", r"(?m)^\+.*max_tokens\s*=\s*\d+"]),
    ("prompt",          [r"(?m)^(\+\+\+|---) .*prompts\.py", r"AGENT_INSTRUCTION", r"system_prompt"]),
]


def _fmt_args(a) -> str:
    try:
        return json.dumps(a, sort_keys=True)
    except Exception:  # noqa: BLE001
        return str(a)


def render(rec: dict, detail: bool = False) -> str:
    """A trial as the analyst / digester / proposer read it."""
    out = [f"TASK {rec.get('task_id')}  (trial {rec.get('trial')})",
           "CUSTOMER'S HIDDEN INSTRUCTIONS (the user simulator follows these; the agent never "
           "sees them):", str(rec.get("scenario") or "").strip(), "",
           f"CUSTOMER (opening): {rec.get('opening')}"]
    calls = []
    for st in rec.get("steps") or []:
        tag = " [replayed prefix]" if st.get("prefix") else ""
        for b in st.get("assistant") or []:
            if b["type"] == "text":
                out.append(f"[step {st['index']}] AGENT{tag}: {b['text']}")
            else:
                calls.append((b["name"], b.get("input")))
                out.append(f"[step {st['index']}] AGENT CALLS{tag} {b['name']}({_fmt_args(b.get('input'))})")
        for r in st.get("tool_results") or []:
            c = str(r.get("content"))
            if not detail and len(c) > MAX_OBS:
                c = c[:MAX_OBS] + f" ...[{len(c) - MAX_OBS} more chars]"
            out.append(f"[step {st['index']}] RESULT: {c}")
        if st.get("user") is not None:
            out.append(f"[step {st['index']}] CUSTOMER: {st['user']}")
    tok = rec.get("tokens") or {}
    out += ["", "RUN: " + f"stopped={rec.get('stopped')} steps={rec.get('n_steps')} "
            f"policy_tokens={tok.get('agent_total')} calls={(rec.get('calls') or {}).get('agent')}"]
    if rec.get("harness_error"):
        out += ["HARNESS ERROR (the harness code raised; the episode was scored 0):",
                str(rec["harness_error"])[-1500:]]
    out += ["", "=== GRADING (ground truth; the agent never sees it) ===",
            f"reward={rec.get('reward')}  database_matches_gold={rec.get('db_match')}  "
            f"missing_required_info={rec.get('missing_info')}",
            "gold actions (the database must end as if exactly these were executed):"]
    made = {(n, _fmt_args(a)) for n, a in calls}
    names = {n for n, _ in calls}
    for g in rec.get("gold_actions") or []:
        key = (g["name"], _fmt_args(g.get("arguments")))
        mark = "MADE" if key in made else ("same tool, different arguments" if g["name"] in names
                                           else "NOT MADE")
        out.append(f"  - {g['name']}({_fmt_args(g.get('arguments'))})  -> {mark}")
    golds = {(g["name"], _fmt_args(g.get("arguments"))) for g in rec.get("gold_actions") or []}
    extra = [(n, a) for n, a in calls if (n, _fmt_args(a)) not in golds
             and not n.startswith(READ_PREFIXES)]
    if extra:
        out.append("agent calls that change state and are NOT gold actions:")
        out += [f"  - {n}({_fmt_args(a)})" for n, a in extra]
    if rec.get("communicate_info"):
        out.append(f"required info to tell the customer: {rec['communicate_info']}")
    return "\n".join(out)


class Tau2Domain(Domain):
    harness_path = "harness"
    tau2: str = "retail"
    critic_patterns: list = []
    component_signals = COMPONENT_SIGNALS

    def __init__(self, name: str, tau2: str, extra_patterns: list):
        self.name, self.tau2 = name, tau2
        self.here = REPO / "domains" / name
        self.cfg = json.loads((self.here / "rrsi.json").read_text())
        self.split = json.loads((self.here / "split.json").read_text())
        self.critic_patterns = COMMON_CRITIC + [(_added(p), why) for p, why in extra_patterns]
        label = self.cfg.get("policy_label", "the frozen policy model")
        self.briefs = {k: v.replace("{policy}", label).replace("{domain}", tau2)
                       for k, v in {"analyst": B.ANALYST, "digester": B.DIGESTER,
                                    "proposer": B.PROPOSER, "critic": B.CRITIC}.items()}
        self._scen: dict | None = None

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self):
        # RRSI_TAU2_EVOLVE_LIMIT: first N evolve tasks only, for smoke runs of the loop
        n = int(os.environ.get("RRSI_TAU2_EVOLVE_LIMIT") or 0)
        ids = list(self.split["evolve"])
        return ids[:n] if n else ids

    def heldout_ids(self):
        return list(self.split["heldout"])

    def smoke_ids(self, incumbent_per_task=None):
        return list(self.split["smoke"])

    # ---- Evaluate ----------------------------------------------------------
    def _driver(self, root: Path, out: Path, log: Path, extra: list[str]) -> None:
        hdir = self.harness_dir(Path(root))
        cmd = [TAU2_PY, "-m", "domains.tau2.driver", "--harness", str(hdir),
               "--domain", self.tau2, "--out", str(out),
               "--workers", str(self.cfg.get("concurrency", 8))]
        t = self.cfg.get("policy_temperature")
        if t is not None:
            cmd += ["--temperature", str(t)]
        env = {**os.environ, "PYTHONPATH": str(REPO), "TAU2_DOMAIN": self.tau2}
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
        per, n = {}, {"trials": 0, "harness_errors": 0, "db_match": 0, "info_ok": 0,
                      "max_steps": 0, "steps": 0}
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
                n["db_match"] += int(bool(rec.get("db_match")))
                n["info_ok"] += int(rec.get("db_match") is not None and not rec.get("missing_info"))
                n["max_steps"] += int(rec.get("stopped") == "max_steps")
                n["steps"] += int(rec.get("n_steps") or 0)
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing)
        d = max(1, n["trials"])
        extra = {"trials": n["trials"], "harness_errors": n["harness_errors"],
                 "harness_error_rate": n["harness_errors"] / d,
                 "db_match_rate": n["db_match"] / d, "info_rate": n["info_ok"] / d,
                 "max_steps_rate": n["max_steps"] / d, "mean_steps": n["steps"] / d}
        return per, extra

    def guards(self, incumbent, candidate):
        g = []
        rate = (candidate.extra or {}).get("harness_error_rate", 0.0)
        if rate > self.cfg.get("max_harness_error_rate", 0.02):
            g.append(f"harness raised in {rate:.1%} of trials")
        return g

    # ---- evidence ----------------------------------------------------------
    def _scenario(self, task_id):
        """The user simulator's instructions for a task, cached (not committed)
        in domains/<name>/scenarios.json on first use."""
        if self._scen is None:
            p = self.here / "scenarios.json"
            if not p.exists():
                code = ("import json,sys; from agent_exp import tau2_env as m; "
                        "json.dump({t.id: m.scenario_text(t) for t in m.get_tasks('base')}, "
                        "open(sys.argv[1], 'w'), indent=1)")
                subprocess.run([TAU2_PY, "-c", code, str(p)], cwd=str(REPO),
                               env={**os.environ, "PYTHONPATH": str(REPO), "TAU2_DOMAIN": self.tau2},
                               capture_output=True)
            self._scen = json.loads(p.read_text()) if p.exists() else {}
        return self._scen.get(str(task_id), "")

    def load_trial(self, runs_dir, job, task_id, trial):
        rec = self._trial(runs_dir, job, task_id, trial)
        if rec is None:
            return None
        rec["trial"] = trial
        rec["scenario"] = self._scenario(task_id)
        return rec

    def render_trace(self, rec, detail=False):
        return render(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        miss = rec.get("missing_info") or []
        return (f"{task_id} | reward {rec.get('reward')} (mean over trials {tr.mean:.2f}) | "
                f"db_matches_gold={rec.get('db_match')} | missing_info={len(miss)} | "
                f"steps={rec.get('n_steps')} | stopped={rec.get('stopped')}"
                + (" | HARNESS ERROR" if rec.get("harness_error") else ""))

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        hdir = self.harness_dir(root)
        comp = subprocess.run([TAU2_PY, "-m", "compileall", "-q", str(hdir)],
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
