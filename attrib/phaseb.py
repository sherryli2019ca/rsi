"""Phase B of the attribution study (attrib/PREREGISTRATION_B.md): does the
attribution a proposer is given change the deployment value of the repairs it
drafts?

Eight incumbent states of the E1R trajectories (r2, r3 x retail, airline x the
incumbent at the start of rounds 5 and 15) x five groups of evidence F_t:
  none           no analysis; the proposer reads the traces itself
  rrsi           RRSI's own analysis report and digests of that round (unchanged)
  first_write    the zero-cost rule "first state-changing call"
  binary_search  the binary-search judge (Pro)
  cf_search      counterfactual search, 40 replays (Pro + replays)
The three step-attribution groups get the same short note and, per failing
trace, one digest naming the attributed step only (no reason, no component, no
method name). Each (group, state) runs ONE unmodified RRSI round from the code
E1R ran (pinned worktree) on an isolated copy of the state; its candidates
that pass RRSI's gates and the common error check are deployed on the held-out
tasks at the domain's heldout_k, as is each state's incumbent.

  python -m attrib.phaseb prep                  # states, failing traces, group dirs, directives check
  python -m attrib.phaseb attribute             # the three methods on the failing traces; digests
  python -m attrib.phaseb rounds [--cells g:s,..] [--parallel 10]
  python -m attrib.phaseb deploy [--parallel 6]
  python -m attrib.phaseb status                # progress and spend (no held-out numbers)
  python -m attrib.phaseb analyze               # endpoints, after every deployment

Everything lives under $PHASEB_ROOT (default /home/user/phaseb).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import _render_step  # noqa: E402
from attrib.pilot_report import PRICES, replay_dollars  # noqa: E402

PB = Path(os.environ.get("PHASEB_ROOT", "/home/user/phaseb"))
CODE = PB / "code"                      # worktree pinned at the commit E1R ran
PIN = "1b69724"
PY = "/home/user/venv-tau2/bin/python"
TRAJ = {"r2": Path("/home/user/e1r2"), "r3": Path("/home/user/e1r3")}
DOMAINS = ("tau2_retail", "tau2_airline")
TS = (5, 15)
STATES = [f"{r}_{d}_t{t}" for r in TRAJ for d in DOMAINS for t in TS]
GROUPS = ("none", "rrsi", "first_write", "binary_search", "cf_search")
STEP_GROUPS = ("first_write", "binary_search", "cf_search")
MAX_ERR = 0.02                          # common error check (paper 1)
SEARCH_BUDGET = 40
EXCERPT = 600

NOTE_NONE = ("No analysis report this round. Read the traces yourself (list_traces, then "
             "read_trace) to find what went wrong; the failing traces are listed first.")
NOTE_STEP = (NOTE_NONE + " For each failing trace, PER-TASK DIGESTS give decisive_step: the "
             "step at which an automatic failure-attribution method located the failure. Read "
             "the trace around that step (read_trace with from_step/to_step) to see what went "
             "wrong there.")


def log(msg: str) -> None:
    print(f"[phaseb] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def parse_state(s: str) -> tuple[str, str, int]:
    traj, rest = s.split("_", 1)
    dom, t = rest.rsplit("_t", 1)
    return traj, dom, int(t)


def src_dir(state: str) -> Path:
    traj, dom, _ = parse_state(state)
    return TRAJ[traj] / "runs" / "rrsi" / dom


def state_dir(state: str) -> Path:
    return PB / "states" / state


def runs_root(group: str, state: str) -> Path:
    return PB / "runs" / group / state


def run_dir(group: str, state: str) -> Path:
    return runs_root(group, state) / parse_state(state)[1]


def branch_ns(group: str, state: str) -> str:
    return f"pb/{group}/{state}"


def git(*args, cwd=ROOT, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def _cfg(dom: str) -> dict:
    return json.loads((CODE / "domains" / dom / "rrsi.json").read_text())


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# ------------------------------------------------------------------- prep --
def incumbent(state: str) -> tuple[dict, list, dict]:
    """(trajectory entry t, trajectory up to t, incumbent evaluation)."""
    _, _, t = parse_state(state)
    fr = json.loads((src_dir(state) / "frontier.json").read_text())
    traj = [x for x in fr["trajectory"] if x["t"] <= t]
    inc = next(x for x in traj if x["t"] == t)
    ev = json.loads((src_dir(state) / "jobs" / inc["job"] / "eval.json").read_text())
    return inc, traj, ev


def traces(state: str) -> dict:
    """RRSI's build_traces on the incumbent's evolve evaluation: worst trial of
    the n_fail_traces lowest tasks, best trial of the n_success_traces highest.
    Returns {task_id: (trial file, reward, lens)}."""
    inc, _, ev = incumbent(state)
    cfg = _cfg(parse_state(state)[1])
    per = {tid: v["rewards"] for tid, v in ev["per_task"].items()}
    ranked = sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1]) if kv[1] else 0.0)
    fails = [t for t, _ in ranked[:cfg["n_fail_traces"]]]
    wins = [t for t, _ in ranked[-cfg["n_success_traces"]:] if t not in fails]
    out = {}
    for tid in fails + wins:
        rw = per[tid]
        if not rw:
            continue
        idx = rw.index(min(rw)) if tid in fails else rw.index(max(rw))
        p = src_dir(state) / "jobs" / inc["job"] / f"s{idx}" / f"{tid}.json"
        if p.exists():
            out[tid] = (str(p), json.loads(p.read_text()).get("reward", 0), "fail" if tid in fails else "win")
    return out


def harness_path(state: str, commit: str) -> Path:
    traj, dom, _ = parse_state(state)
    wt = TRAJ[traj] / "runs" / "verify" / dom / "wt" / commit[:12]
    if not (wt / ".git").exists():
        from verify.state import worktree
        worktree(ROOT, wt.parent, commit)
    return wt / "domains" / dom / "harness"


def prep_state(state: str) -> dict:
    traj, dom, t = parse_state(state)
    inc, trajectory, ev = incumbent(state)
    sd = state_dir(state)
    sd.mkdir(parents=True, exist_ok=True)
    tr = traces(state)
    harness = str(harness_path(state, inc["commit"]))
    fails = [{"fid": f"{state}_{tid}", "task_id": tid, "trace": p, "harness": harness,
              "n_steps": json.loads(Path(p).read_text())["n_steps"]}
             for tid, (p, r, _) in tr.items() if r < 1]
    (sd / "failures.json").write_text(json.dumps(fails, indent=1))
    orig = src_dir(state) / f"r{t}"
    orig_fail_digests = sorted(p.name.split("_")[0] for p in (orig / "analysis" / "digests").glob("*_failure.json"))
    harness_rel = f"domains/{dom}/harness"
    tree = git("rev-parse", f"{inc['commit']}:{harness_rel}")[:12]
    fr_src = json.loads((src_dir(state) / "frontier.json").read_text())
    for g in GROUPS:
        rd = run_dir(g, state)
        (rd / "jobs").mkdir(parents=True, exist_ok=True)
        if not (rd / "frontier.json").exists():
            fr = {"domain": fr_src.get("domain"),
                  "incumbent": {"t": t, "commit": inc["commit"], "harness_tree": tree, "job": inc["job"],
                                "S": inc["S"], "C": inc["C"], "extra": ev.get("extra"),
                                "variant": inc["job"][-1] if inc["job"] != "base" else "-"},
                  "S_star": max(x["S"] for x in trajectory), "trajectory": trajectory,
                  "config": fr_src.get("config")}
            (rd / "frontier.json").write_text(json.dumps(fr, indent=1))
            shutil.copy(src_dir(state) / "calibration.json", rd / "calibration.json")
            for name in ("history.jsonl", "attribution.jsonl"):
                rows = [r for r in _jsonl(src_dir(state) / name) if r["t"] < t]
                (rd / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        link = rd / "jobs" / inc["job"]
        if not link.exists():
            link.symlink_to(src_dir(state) / "jobs" / inc["job"])
        rdir = rd / f"r{t}"
        (rdir / "analysis" / "digests").mkdir(parents=True, exist_ok=True)
        if g == "rrsi":
            shutil.copy(orig / "analysis_report.json", rdir / "analysis_report.json")
            for p in (orig / "analysis" / "digests").glob("*.json"):
                shutil.copy(p, rdir / "analysis" / "digests" / p.name)
        else:
            (rdir / "analysis_report.json").write_text(json.dumps(
                {"note": NOTE_NONE if g == "none" else NOTE_STEP}, indent=1))
        br = f"evolve/{branch_ns(g, state)}/{dom}"
        if git("rev-parse", "--verify", "--quiet", f"refs/heads/{br}", check=False) == "":
            git("branch", br, inc["commit"])
    check = directives_check(state)
    res = {"state": state, "incumbent": inc["commit"], "job": inc["job"], "traces": len(tr),
           "failing": len(fails), "rrsi_failure_digests": len(orig_fail_digests),
           "same_failing_set": sorted(f["task_id"] for f in fails) == orig_fail_digests,
           "directives_match": check}
    (sd / "prep.json").write_text(json.dumps(res, indent=1))
    return res


def directives_check(state: str) -> dict:
    """Recompute RRSI's directives (b_t, sigma_t, tried, E_t, prune set, delta,
    S*) from a group dir with the pinned code, compare with the original round."""
    _, dom, t = parse_state(state)
    code = f"""
import json, sys
from pathlib import Path
from rrsi.config import RRSIConfig
from rrsi.history import History, exploration, stall_flag
from rrsi.schedule import edit_budget
rd = Path(sys.argv[1]); t = int(sys.argv[2]); cfg = RRSIConfig.load(sys.argv[3])
fr = json.loads((rd / 'frontier.json').read_text())
delta = float(json.loads((rd / 'calibration.json').read_text())['delta']) if cfg.delta is None else cfg.delta
h = History(rd / 'history.jsonl')
sigma = stall_flag([x['S'] for x in fr['trajectory']], t, cfg.w, delta)
tried = h.tried()
print(json.dumps({{'t': t, 'b_t': edit_budget(t, cfg.T, cfg.b_min, cfg.b_max), 'sigma_t': sigma,
    'tried': sorted(tried), 'explore': exploration(t, sigma, tried, cfg.m_draft),
    'prune_set': h.prune_set(t, cfg.n_prune), 'delta': delta, 'S_star': fr['S_star']}}))
"""
    out = {}
    orig = json.loads((src_dir(state) / f"r{t}" / "directives.json").read_text())
    for g in GROUPS:
        r = subprocess.run([PY, "-c", code, str(run_dir(g, state)), str(t),
                            str(CODE / "domains" / dom / "rrsi.json")],
                           cwd=CODE, env={**os.environ, "PYTHONPATH": str(CODE)},
                           capture_output=True, text=True)
        if r.returncode != 0:
            out[g] = f"error: {r.stderr[-400:]}"
            continue
        new = json.loads(r.stdout)
        diff = [k for k in orig if json.dumps(orig[k], sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]
        out[g] = diff or True
    return out


def prep() -> None:
    if not (CODE / ".git").exists():
        CODE.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(CODE), PIN)
    rows = [prep_state(s) for s in STATES]
    (PB / "prep.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(json.dumps(r))


# -------------------------------------------------------------- attribute --
def attribute(workers: int) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for dom in DOMAINS:
        fails = [f for s in STATES if parse_state(s)[1] == dom
                 for f in json.loads((state_dir(s) / "failures.json").read_text())]
        ad = PB / "attrib" / dom
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "failures.json").write_text(json.dumps(fails, indent=1))
        log(f"{dom}: {len(fails)} failing traces; first write + binary search")
        subprocess.run([PY, "-m", "attrib.methods", "run", "--domain", dom, "--failures",
                        str(ad / "failures.json"), "--out", str(ad / "methods"),
                        "--methods", "first_write,binary_search:pro", "--workers", str(workers)],
                       cwd=ROOT, env=env, check=True)
        log(f"{dom}: counterfactual search@{SEARCH_BUDGET}")
        subprocess.run([PY, "-m", "attrib.search", "run", "--domain", dom, "--failures",
                        str(ad / "failures.json"), "--out", str(ad / "search"),
                        "--budget", str(SEARCH_BUDGET), "--workers", str(workers)],
                       cwd=ROOT, env=env, check=True)
    write_digests()


def picks(dom: str) -> dict:
    """{group: {fid: step}} of the three step-attribution groups."""
    ad = PB / "attrib" / dom
    out = {"first_write": {}, "binary_search": {}, "cf_search": {}}
    for g, sub in (("first_write", "first_write"), ("binary_search", "binary_search_pro")):
        for p in (ad / "methods" / sub).glob("*.json"):
            r = json.loads(p.read_text())
            out[g][r["fid"]] = r.get("step")
    res = ad / "search" / "result.json"
    if res.exists():
        for r in json.loads(res.read_text())["failures"]:
            out["cf_search"][r["fid"]] = r[f"at{SEARCH_BUDGET}"]["step"]
    return out


def write_digests() -> None:
    for dom in DOMAINS:
        pk = picks(dom)
        for s in STATES:
            if parse_state(s)[1] != dom:
                continue
            t = parse_state(s)[2]
            fails = json.loads((state_dir(s) / "failures.json").read_text())
            for g in STEP_GROUPS:
                dg = run_dir(g, s) / f"r{t}" / "analysis" / "digests"
                dg.mkdir(parents=True, exist_ok=True)
                for f in fails:
                    if f["fid"] not in pk[g]:
                        raise SystemExit(f"{g}: no attribution for {f['fid']}")
                    k = pk[g][f["fid"]]
                    rec = json.loads(Path(f["trace"]).read_text())
                    st = next((x for x in rec["steps"] if x["index"] == k), None) if k is not None else None
                    if st is None:
                        continue
                    ex = _render_step(st)
                    (dg / f"{f['task_id']}_failure.json").write_text(json.dumps(
                        {"task_id": f["task_id"], "lens": "failure", "decisive_step": k,
                         "step_excerpt": ex[:EXCERPT] + (" ...[truncated]" if len(ex) > EXCERPT else ""),
                         "blocker": f"failure decided at step {k}"}, ensure_ascii=False, indent=1))
    log("digests written")


# ----------------------------------------------------------------- rounds --
def cell_done(g: str, s: str) -> bool:
    t = parse_state(s)[2]
    rd = run_dir(g, s)
    if not (rd / f"r{t}" / "decisions.json").exists():
        return False
    fr = json.loads((rd / "frontier.json").read_text())
    return any(x["t"] == t + 1 for x in fr["trajectory"])


def run_cell(g: str, s: str) -> str:
    _, dom, t = parse_state(s)
    rd = run_dir(g, s)
    if cell_done(g, s):
        return f"{g}:{s} done"
    if g in STEP_GROUPS and not list((rd / f"r{t}" / "analysis" / "digests").glob("*.json")):
        return f"{g}:{s} no digests (run attribute first)"
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(CODE), "RRSI_USAGE_LOG": str(rd / "usage.jsonl")}
    with open(rd / "logs" / f"round{t}.log", "a") as fh:
        r = subprocess.run([PY, "rrsi.py", "--domain", dom, "--runs", str(runs_root(g, s)),
                            "--branch-ns", branch_ns(g, s), "round", "--t", str(t)],
                           cwd=CODE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return f"{g}:{s} exit {r.returncode}" + (" done" if cell_done(g, s) else "")


def all_cells() -> list[tuple[str, str]]:
    """State-major order: the five groups of a state run side by side."""
    return [(g, s) for s in STATES for g in GROUPS]


def rounds(cells: list[tuple[str, str]], parallel: int) -> None:
    with ThreadPoolExecutor(parallel) as ex:
        for msg in ex.map(lambda c: run_cell(*c), cells):
            log(msg)


# ----------------------------------------------------------------- deploy --
def candidates(g: str, s: str) -> list[dict]:
    """The round's slots A and B with their gate outcome."""
    t = parse_state(s)[2]
    rdir = run_dir(g, s) / f"r{t}"
    out = []
    for v in ("A", "B"):
        vd = rdir / v
        prep = json.loads((vd / "prep.json").read_text()) if (vd / "prep.json").exists() else {}
        row = {"group": g, "state": s, "variant": v, "commit": prep.get("commit"),
               "gate": prep.get("gate_failure") or (None if prep else "not_drafted")}
        if row["gate"] is None:
            if not (vd / "eval.json").exists():
                row["gate"] = "eval_invalid"
            else:
                ev = json.loads((vd / "eval.json").read_text())
                row["evolve_S"], row["evolve_C"] = ev["S"], ev["C"]
                row["error_rate"] = (ev.get("extra") or {}).get("harness_error_rate") or 0.0
                if row["error_rate"] > MAX_ERR:
                    row["gate"] = "error_check"
        out.append(row)
    return out


def heldout_dir(dom: str, commit: str) -> Path:
    return PB / "verify" / dom / "jobs" / "heldout" / commit[:12]


def deploy_one(dom: str, commit: str) -> str:
    k = int(_cfg(dom)["heldout_k"])
    hd = heldout_dir(dom, commit)
    if (hd / "eval.json").exists() and json.loads((hd / "eval.json").read_text()).get("n_expected") == \
            k * len(json.loads((CODE / "domains" / dom / "split.json").read_text())["heldout"]):
        return f"{dom} {commit[:12]} done"
    (PB / "verify" / dom / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    with open(PB / "verify" / dom / "logs" / f"{commit[:12]}.log", "a") as fh:
        r = subprocess.run([PY, "-m", "verify.deploy", "--domain", dom, "--ref", commit, "--k", str(k),
                            "--out", str(PB / "verify")], cwd=CODE, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    wt = PB / "verify" / dom / "wt" / commit[:12]
    if wt.exists():
        git("worktree", "remove", "--force", str(wt), check=False)
        shutil.rmtree(wt, ignore_errors=True)
    ev = json.loads((hd / "eval.json").read_text()) if (hd / "eval.json").exists() else {}
    return f"{dom} {commit[:12]} exit {r.returncode} trials {ev.get('n_expected')} missing {ev.get('missing')}"


def deploy_jobs() -> list[tuple[str, str]]:
    jobs = []
    for s in STATES:
        dom = parse_state(s)[1]
        jobs.append((dom, incumbent(s)[0]["commit"]))
        for g in GROUPS:
            if cell_done(g, s):
                jobs += [(dom, c["commit"]) for c in candidates(g, s) if c["gate"] is None]
    return list(dict.fromkeys(jobs))


def deploy(parallel: int) -> None:
    with ThreadPoolExecutor(parallel) as ex:
        for msg in ex.map(lambda j: deploy_one(*j), deploy_jobs()):
            log(msg)


# ----------------------------------------------------------------- status --
def _usage(p: Path) -> float:
    d = 0.0
    for u in _jsonl(p):
        pi, pc, po = PRICES.get(u.get("model"), PRICES["deepseek-v4-pro"])
        d += ((u.get("in") or 0) * pi + (u.get("cache_read") or 0) * pc + (u.get("out") or 0) * po) / 1e6
    return d


def _episodes(root: Path, pattern: str) -> tuple[float, int]:
    d, n = 0.0, 0
    for p in root.glob(pattern):
        if p.name in ("eval.json",) or p.is_symlink() or any(x.is_symlink() for x in p.parents):
            continue
        try:
            d += replay_dollars(json.loads(p.read_text()))
            n += 1
        except Exception:  # noqa: BLE001
            pass
    return d, n


def spend() -> dict:
    s = {"model_rounds": sum(_usage(p) for p in (PB / "runs").glob("*/*/*/usage.jsonl"))}
    s["attribution_calls"] = sum(_usage(p) for p in (PB / "attrib").glob("*/*/usage.jsonl"))
    s["search_replays"], _ = _episodes(PB / "attrib", "*/search/replays/*.json")
    ev, n_ev = 0.0, 0
    for g in GROUPS:
        for st in STATES:
            jd = run_dir(g, st) / "jobs"
            for job in jd.iterdir() if jd.exists() else []:
                if job.is_symlink():
                    continue
                d, n = _episodes(job, "s*/*.json")
                ev, n_ev = ev + d, n_ev + n
    s["evolve_and_smoke_episodes"], s["n_evolve_and_smoke"] = ev, n_ev
    s["heldout_episodes"], s["n_heldout"] = _episodes(PB / "verify", "*/jobs/heldout/*/s*/*.json")
    s["total"] = sum(v for k, v in s.items() if not k.startswith("n_"))
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in s.items()}


def status() -> None:
    cells = all_cells()
    done = [c for c in cells if cell_done(*c)]
    gates: dict = {}
    for g, s in done:
        for c in candidates(g, s):
            gates.setdefault(g, {}).setdefault(c["gate"] or "deployable", 0)
            gates[g][c["gate"] or "deployable"] += 1
    jobs = deploy_jobs()
    k = {d: int(_cfg(d)["heldout_k"]) for d in DOMAINS}
    deployed = sum((heldout_dir(d, c) / "eval.json").exists() for d, c in jobs)
    print(json.dumps({"rounds_done": f"{len(done)}/{len(cells)}", "gates": gates,
                      "deployments_done": f"{deployed}/{len(jobs)}", "heldout_k": k,
                      "spend": spend()}, indent=1))


# ---------------------------------------------------------------- analyze --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "attribute", "digests", "rounds", "deploy", "status", "analyze"))
    ap.add_argument("--cells", default="", help="group:state,... (default: all)")
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep()
    elif args.cmd == "attribute":
        attribute(args.workers)
    elif args.cmd == "digests":
        write_digests()
    elif args.cmd == "rounds":
        cells = [tuple(c.split(":")) for c in args.cells.split(",") if c] or all_cells()
        rounds(cells, args.parallel)
    elif args.cmd == "deploy":
        deploy(args.parallel)
    elif args.cmd == "status":
        status()
    elif args.cmd == "analyze":
        from attrib.phaseb_analyze import main as analyze_main
        analyze_main()


if __name__ == "__main__":
    main()
