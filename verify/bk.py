"""Experiment BK (verify/PREREGISTRATION_BK.md): RRSI loops on tau2 airline in
blocks that share one noise-band calibration, comparing selection rules and
separating the two things sequential evaluation changes in a loop: which
candidates can be accepted (admission) and what the proposer's history records
for the candidates the rule stops (history).

  arm  selection  admission                   history of stopped candidates
  f    full       full evaluation             full evaluation
  s    seqfull    sequential (registered)     early-stopped estimates
  c    seqcost    sequential (cost-aware)     early-stopped estimates
  h    seqhist    full evaluation             early-stopped estimates (seqfull's)
  a    seqadm     sequential (seqfull's)      full evaluation

DESIGN picks the blocks and arms (fixed at registration). Each block
calibrates the noise band once (baseline plus two repeated base evaluations,
in /home/user/bk_<block>_cal) and seeds its loops with that calibration, so
the arms of a block start from the same state and the same delta. The base is
deployed on the held-out tasks once per block, and every final incumbent is
deployed once per block (in the calibration checkout's verify directory).

Coupled design (five_arms_coupled): arms that are in the same state share
their rounds. A block starts one lineage, /home/user/bk_<block>_root, holding
all five arms (selection group:f,h,c,a,s, verify/live.py select_group). When
the arms' rounds differ, each other group of arms is forked into its own
lineage /home/user/bk_<block>_<arms> (fork_lineage), which reruns that round
from its checkpoints in its own mode (no model call, no episode) and continues
alone. Every arm's marginal distribution is that of an independent loop of
its mode; arms are coupled only while their states are identical.

  python -m verify.bk arms                    the design's arms as "letter=selection" words
  python -m verify.bk starts                  "checkout arms" of the lineages each block starts
  python -m verify.bk seed    --cal DIR --run DIR --arms f,h,c,a,s
  python -m verify.bk selection --run DIR     the lineage's selection mode now
  python -m verify.bk deploy  --run DIR       final incumbent on the held-out tasks
  python -m verify.bk deploy-base --run CAL   the block's base on the held-out tasks
  python -m verify.bk post    --run DIR       after the deployment: shadow (a lineage that ends
                                              as seqfull or seqcost alone), replay (one with f)
  python -m verify.bk replay  --run DIR       (full arm) where seqfull and seqcost would have
                                              stopped each candidate, and what they would choose
  python -m verify.bk missed  [--run DIR ...] (IL seqfull loops) held-out deployments of the
                                              stopped candidates full evaluation would have chosen
  python -m verify.bk spend   [--limit USD]   spend of every BK checkout (exit 3 above the limit)
  python -m verify.bk status                  progress (no held-out S)
  python -m verify.bk analyze [--json results/bk/analysis.json]
  python -m verify.bk tables  [--json ...] [--tex paper/tables/bk.tex]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rrsi import gitops as G
from rrsi.config import RRSIConfig
from rrsi.domain import load_domain
from rrsi.evaluate import EvalResult
from rrsi.history import History
from rrsi.selection import Candidate

from . import cl, deploy, il
from .state import trial_path, worktree

D = cl.D
T_LAST = cl.T_LAST
K_HELDOUT = cl.K_HELDOUT
HOME = Path("/home/user")
SELECTION = {"f": "full", "s": "seqfull", "c": "seqcost", "h": "seqhist", "a": "seqadm"}
DESIGNS = {
    "five_arms": {"blocks": ("b1", "b2", "b3", "b4"), "arms": ("f", "s", "c", "h", "a")},
    "rule_fix": {"blocks": ("b1", "b2", "b3", "b4", "b5", "b6"), "arms": ("f", "s", "c")},
    "five_arms_coupled": {"blocks": ("b1", "b2", "b3", "b4"), "arms": ("f", "h", "c", "a", "s"),
                          "coupled": True},
}
DESIGN = "five_arms_coupled"                      # set when the experiment is registered
BLOCKS = DESIGNS[DESIGN]["blocks"]
ARMS = {a: SELECTION[a] for a in DESIGNS[DESIGN]["arms"]}
COUPLED = bool(DESIGNS[DESIGN].get("coupled"))
ORDER = ("f", "h", "c", "a", "s")                 # a lineage keeps the group of its first arm
IND = ("f", "h", "c")                             # coupled: arms that always have their own lineage
PAIRED = {"a": "f", "s": "h"}                     # coupled: arm -> the arm it shares rounds with
ADM = {"f": 0, "h": 0, "s": 1, "a": 1}          # 1 = sequential admission
HIST = {"f": 0, "a": 0, "s": 1, "h": 1}         # 1 = early-stopped history
CAL_JOBS = ("base", "heldout_base2", "heldout_base3")
# contrasts over the arm means; those whose arms the design lacks are skipped
CONTRASTS = {
    "seq_minus_full": {"s": 1, "f": -1},
    "cost_minus_full": {"c": 1, "f": -1},
    "cost_minus_seq": {"c": 1, "s": -1},
    "admission": {"s": .5, "a": .5, "f": -.5, "h": -.5},
    "history": {"s": .5, "h": .5, "f": -.5, "a": -.5},
    "interaction": {"s": 1, "a": -1, "h": -1, "f": 1},   # (s - a) - (h - f)
}
MISSED_IL = {"s2": "/home/user/il_s2", "s3": "/home/user/il_s3", "s4": "/home/user/il_s4"}


def loop_dir(b: str, arm: str) -> Path:
    """The checkout that holds arm's loop at the end (coupled: its last lineage)."""
    if not COUPLED:
        return HOME / f"bk_{b}_{arm}"
    for p in lineage_dirs(b):
        if arm in members(p):
            return p
    raise FileNotFoundError(f"block {b}: no lineage holds arm {arm}")


def cal_dir(b: str) -> Path:
    return HOME / f"bk_{b}_cal"


def mode_for(arms: list) -> str:
    return SELECTION[arms[0]] if len(arms) == 1 else "group:" + ",".join(arms)


def lineage_dirs(b: str) -> list:
    return sorted(p for p in HOME.glob(f"bk_{b}_*") if (p / "runs" / "bk_lineage.json").exists())


def lineage(run: Path) -> dict:
    return json.loads((Path(run) / "runs" / "bk_lineage.json").read_text())


def members(run: Path) -> list:
    return (Path(run) / "runs" / "bk_members").read_text().split()


def set_members(run: Path, arms: list) -> None:
    (Path(run) / "runs" / "bk_members").write_text(" ".join(arms) + "\n")


def starts(b: str) -> list:
    """(checkout, arms) of the lineages a block starts with."""
    arms = list(ARMS)
    if COUPLED:
        return [(HOME / f"bk_{b}_root", sorted(arms, key=ORDER.index))]
    return [(HOME / f"bk_{b}_{a}", [a]) for a in arms]


# ------------------------------------------------------------------ setup --
def _link(src, dst) -> None:
    """copytree's copy function for episode files, which are written once:
    a hard link (an existing target is kept, so a repeated copy resumes)."""
    if not os.path.exists(dst):
        os.link(src, dst)


def cmd_seed(cal: Path, run: Path, arms: list) -> None:
    """Copy the block's calibration (frontier, noise band, history, and the three
    base evaluations the round-0 analysis reads) into a loop checkout, and put
    the loop's branch evolve/<checkout name>/<D> at the same base commit."""
    src, dst = cl._rrsi(cal), cl._rrsi(run)
    if (Path(run) / "runs" / "bk_lineage.json").exists():
        print(f"{run.name}: already seeded")
        return
    calib = json.loads((src / "calibration.json").read_text())
    if list(calib.get("jobs", [])) != list(CAL_JOBS):
        sys.exit(f"{cal}: calibration not finished (jobs {calib.get('jobs')})")
    fr = json.loads((src / "frontier.json").read_text())
    if len(fr["trajectory"]) != 1:
        sys.exit(f"{cal}: the calibration checkout ran rounds; it must stay at the base")
    ns = Path(run).name
    branch = f"evolve/{ns}/{D}"
    G.ensure_branch(run, branch, "HEAD")
    head = G.rev(run, branch)
    if not (head.startswith(fr["incumbent"]["commit"]) or fr["incumbent"]["commit"].startswith(head)):
        sys.exit(f"{run}: {branch} at {head}, block base is {fr['incumbent']['commit']}")
    if G.tree_hash(run, branch, f"domains/{D}") != G.tree_hash(cal, "HEAD", f"domains/{D}"):
        sys.exit(f"{run}: domain tree differs from the calibration checkout")
    dst.mkdir(parents=True, exist_ok=True)
    for j in CAL_JOBS:
        shutil.copytree(src / "jobs" / j, dst / "jobs" / j, copy_function=_link, dirs_exist_ok=True)
    shutil.copy2(src / "calibration.json", dst / "calibration.json")
    shutil.copy2(src / "history.jsonl", dst / "history.jsonl")
    fr["config"].update(selection=mode_for(arms), branch_ns=ns)
    fr["seeded_from"] = str(cal)
    (dst / "frontier.json").write_text(json.dumps(fr, indent=1))
    set_members(run, arms)
    (Path(run) / "runs" / "bk_lineage.json").write_text(json.dumps(          # last: marks the seed done
        {"block": Path(cal).name.split("_")[1], "parent": None, "fork_round": 0, "arms": arms}, indent=1))
    print(f"{run.name}: seeded from {cal.name} (delta {calib['delta']:.4f}), {branch} at {head}, "
          f"selection {mode_for(arms)}")


def fork_lineage(run, t: int, arms: list) -> Path:
    """Coupled design: the lineage of `arms`, forked from `run` (an rrsi.loop.Run
    inside round t's selection step, before the round is recorded). The new
    checkout gets the state as it was when round t started (frontier, history
    and attribution of rounds before t, noise band, the incumbent's branch), round
    t's checkpoints (analysis, candidates and their commits, evaluations; the
    episodes as hard links) and its own branches; it then reruns round t in
    its own mode, which needs no model call and no episode, and continues."""
    parent = Path(run.repo)
    b = lineage(parent)["block"]
    child = HOME / f"bk_{b}_{''.join(arms)}"
    if (child / "runs" / "bk_lineage.json").exists():
        return child                              # a resumed round forked it already
    if not (child / ".git").exists():
        commit = G.git(parent, "rev-parse", "HEAD").stdout.strip()
        G.git(parent, "worktree", "add", "--detach", str(child), commit, check=True)
    src, dst = cl._rrsi(parent), cl._rrsi(child)
    (dst / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "jobs", dst / "jobs", copy_function=_link, dirs_exist_ok=True)
    rd = src / f"r{t}"
    shutil.copytree(rd, dst / f"r{t}", dirs_exist_ok=True, ignore=shutil.ignore_patterns(
        "sequential", "group_*", "selection.json", "decisions.json", "group.json"))
    for name in ("calibration.json", "global_analysis.json"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
    keep = lambda r: r.get("t", 0) < t or r.get("outcome") == "BASELINE"
    for name in ("history.jsonl", "attribution.jsonl"):
        if (src / name).exists():
            rows = [l for l in (src / name).read_text().splitlines() if l.strip() and keep(json.loads(l))]
            (dst / name).write_text("".join(l + "\n" for l in rows))
    fr = json.loads((src / "frontier.json").read_text())
    if len(fr["trajectory"]) != t + 1:
        raise RuntimeError(f"{parent.name}: frontier has {len(fr['trajectory'])} entries in round {t}")
    ns = child.name
    G.ensure_branch(child, f"evolve/{ns}/{D}", fr["incumbent"]["commit"])
    for v in sorted(x.name for x in rd.iterdir() if (x / "prep.json").exists()):
        prep = json.loads((rd / v / "prep.json").read_text())
        if prep.get("commit"):
            G.ensure_branch(child, f"{ns}/{D}/r{t}{v}", prep["commit"])
    fr["config"].update(selection=mode_for(arms), branch_ns=ns)
    fr["forked_from"] = {"lineage": parent.name, "round": t}
    (dst / "frontier.json").write_text(json.dumps(fr, indent=1))
    set_members(child, arms)
    (child / "runs" / "bk_lineage.json").write_text(json.dumps(              # last: marks the fork done
        {"block": b, "parent": parent.name, "fork_round": t, "arms": arms}, indent=1))
    if os.environ.get("BK_NO_LAUNCH") != "1":
        with open(child / "runs" / "bk.out", "a") as log:
            subprocess.Popen(["bash", "verify/run_bk_loop.sh"], cwd=child, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"[bk] round {t}: forked {ns} ({mode_for(arms)}) from {parent.name}", flush=True)
    return child


def _block(run: Path) -> str:
    return lineage(run)["block"]


def cmd_deploy(run: Path) -> None:
    """The final incumbent on the held-out tasks, deployed once per block in the
    calibration checkout's verify directory (the base is deployed there too), so
    lineages that end with the same harness share its deployment. Resume-safe."""
    dom = load_domain(D)
    tr = cl._traj(run)
    out = cal_dir(_block(run)) / "runs" / "verify" / D
    (out / "locks").mkdir(parents=True, exist_ok=True)
    if tr["final"] != tr["base"]:
        with open(out / "locks" / f"{tr['final'][:12]}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not (out / "jobs" / "heldout" / tr["final"][:12] / "eval.json").exists():
                deploy.run(dom, Path(run), out, tr["final"], K_HELDOUT)
        print(f"deployed {tr['final'][:12]} (k={K_HELDOUT})", flush=True)   # S deliberately not printed
    mine = Path(run) / "runs" / "verify" / D
    mine.mkdir(parents=True, exist_ok=True)
    (mine / "deploy.done").write_text(tr["final"] + "\n")


def cmd_post(run: Path) -> None:
    """After the deployment: the shadow evaluation for a lineage that ends as
    seqfull or seqcost alone (what full evaluation would have chosen in its own
    rounds), the replay for one that holds f (where the sequential rules would
    have stopped its candidates). Group rounds record both in group.json."""
    arms = members(run)
    if arms in (["s"], ["c"]):
        cl.cmd_shadow(Path(run))
    if "f" in arms:
        cmd_replay(Path(run))


def cmd_deploy_base(cal: Path) -> None:
    dom = load_domain(D)
    out = Path(cal) / "runs" / "verify" / D
    out.mkdir(parents=True, exist_ok=True)
    base = cl._traj(cal)["base"]
    deploy.run(dom, Path(cal), out, base, K_HELDOUT)
    (out / "deploy_base.done").write_text(base + "\n")
    print(f"deployed base {base[:12]} (k={K_HELDOUT})", flush=True)


# ----------------------------------------------------------------- replay --
def _state(run: Path, t: int, dom, cfg):
    """What round t of a finished loop started from: incumbent evaluation, S*,
    delta, candidates (with their full evaluations) and component counts."""
    rr = cl._rrsi(run)
    fr = json.loads((rr / "frontier.json").read_text())
    tr = fr["trajectory"][:t + 1]
    inc_ev = EvalResult.load(rr / "jobs" / tr[-1]["job"] / "eval.json")
    S_star = max(x["S"] for x in tr)
    delta = float(json.loads((rr / "calibration.json").read_text())["delta"])
    tmp = rr.parent.parent / "verify" / D / "replay" / f"history_r{t}.jsonl"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text("".join(l + "\n" for l in (rr / "history.jsonl").read_text().splitlines()
                           if l.strip() and json.loads(l)["t"] < t
                           or (l.strip() and json.loads(l).get("outcome") == "BASELINE")))
    counts = History(tmp).incumbent_component_counts()
    decs = json.loads((rr / f"r{t}" / "decisions.json").read_text())
    cands = []
    for x in decs:
        ep = rr / f"r{t}" / x["variant"] / "eval.json"
        if not ep.exists():
            continue
        prep = json.loads((rr / f"r{t}" / x["variant"] / "prep.json").read_text())
        c = Candidate(x["variant"], prep.get("edits") or [], commit=prep.get("commit"))
        c.ev = EvalResult.load(ep)
        cands.append((c, x))
    return inc_ev, S_star, delta, cands, counts


def cmd_replay(run: Path) -> None:
    """Full arm: for each round, where seqfull (and the cost-aware seqcost) would
    have stopped each candidate (the rule replayed on the candidate's own
    episodes, in seqfull's order) and which candidate each would then have
    chosen. Runs no episode."""
    from rrsi.components import novelty

    from .live import _seq_candidate
    dom = load_domain(D)

    def no_run(*a, **k):
        raise RuntimeError("replay must not run episodes")
    dom.run = no_run
    cfg = RRSIConfig.load(dom.root / "rrsi.json")
    rr = cl._rrsi(run)
    fake = SimpleNamespace(domain=dom, cfg=cfg, runs=rr, wt_root=rr / "wt")
    rows = []
    for t in range(T_LAST + 1):
        inc_ev, S_star, delta, cands, counts = _state(run, t, dom, cfg)
        full = {c.variant: {"S": x["S"], "admissible": bool(x["admissible"])} for c, x in cands}
        adm = {v: f for v, f in full.items() if f["admissible"]}
        row = {"t": t, "full_choice": max(adm, key=lambda v: adm[v]["S"]) if adm else None,
               "candidates": {v: dict(f) for v, f in full.items()}}
        for rule, key in (("seqfull", "seq"), ("seqcost", "seqcost")):
            sdir = rr.parent.parent / "verify" / D / "replay" / f"r{t}" / rule
            sdir.mkdir(parents=True, exist_ok=True)
            recs = {c.variant: _seq_candidate(fake, t, c, inc_ev, S_star, delta,
                                              novelty(c.components, counts), sdir,
                                              execute=False, rule=rule) for c, _ in cands}
            alive = {v: f for v, f in adm.items() if not recs[v]["dropped"]}
            row[f"{key}_choice"] = max(alive, key=lambda v: alive[v]["S"]) if alive else None
            row["stopped" if rule == "seqfull" else "stopped_seqcost"] = sorted(
                v for v, r in recs.items() if r["dropped"])
            for v, r in recs.items():
                row["candidates"][v].update({f"n_{key}": r["n"], f"m_{key}": r["m"],
                                             f"dropped_{key}": r["dropped"]})
        rows.append(row)
    out = rr.parent.parent / "verify" / D / "replay.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{run.name}: seqfull would differ in "
          f"{sum(r['full_choice'] != r['seq_choice'] for r in rows)} of {len(rows)} rounds, seqcost in "
          f"{sum(r['full_choice'] != r['seqcost_choice'] for r in rows)}")


def _native_rows(arm: str, run: Path) -> dict:
    """Per round t of a checkout that ran `arm` in its own mode: full evaluation's
    choice, the sequential rule's choice and the stopped set, from the source the
    mode has (shadow, replay or the arm's own record)."""
    rr = cl._rrsi(run)
    v = rr.parent.parent / "verify" / D
    if arm in ("s", "c"):
        rows = json.loads((v / "shadow.json").read_text())
        return {r["t"]: {"t": r["t"], "full_choice": r["full_choice"], "seq_choice": r["seq_choice"],
                         "stopped": sorted(k for k, c in r["candidates"].items() if c.get("completed"))}
                for r in rows}
    if arm == "f":
        return {r["t"]: r for r in json.loads((v / "replay.json").read_text())}
    out = {}
    for t in range(T_LAST + 1):
        sp = rr / f"r{t}" / "selection.json"
        if sp.exists():
            s = json.loads(sp.read_text())
            if s.get("rule") in ("seqhist", "seqadm"):
                out[t] = {"t": t, "full_choice": s["full_choice"], "seq_choice": s["seq_choice"],
                          "stopped": s["stopped"]}
    return out


def _rows(arm: str, b: str) -> list:
    """Per round of arm's loop in block b: full evaluation's choice, the
    sequential rule's choice (seqcost's for c, seqfull's otherwise) and the
    stopped set; from r<t>/group.json for a round the arm shared, else from
    the lineage's own records."""
    chain = _chain(b, arm)
    native = {}
    out = []
    for t in range(T_LAST + 1):
        run = chain[_where(chain, t)][0]
        g = cl._rrsi(run) / f"r{t}" / "group.json"
        if g.exists() and arm in json.loads(g.read_text())["members"]:
            x = json.loads(g.read_text())["arms"][arm]
            out.append({"t": t, "full_choice": x["full_choice"], "seq_choice": x["seq_choice"],
                        "stopped": x["stopped"]})
            continue
        if run not in native:
            native[run] = _native_rows(arm, run)
        out.append(native[run][t])
    return out


def _seq_order(inc_ev: EvalResult, t: int, variant: str, n: int) -> list:
    """The first n (task, trial) episodes of a candidate in seqfull's order
    (verify/live.py _seq_candidate)."""
    import random
    from collections import Counter
    pairs = [(tid, s) for tid, tr in sorted(inc_ev.per_task.items()) for s in range(len(tr.rewards))]
    random.Random(f"{D}:{t}:{variant}:seq").shuffle(pairs)
    slot, out = Counter(), []
    for tid, _ in pairs[:n]:
        out.append((tid, slot[tid]))
        slot[tid] += 1
    return out


def _arm_cost(b: str, arm: str) -> dict:
    """What arm's loop cost as if it had run alone (IL's accounting): model calls
    of each of its rounds (where the round was drafted, plus a forked lineage's
    rerun of it), the smoke tests, the candidate episodes its own mode uses
    (all of them for f, h and a; the first n in the rule's order for s and c),
    and the block's calibration."""
    from .posthoc_e1 import _episode_dollars
    chain = _chain(b, arm)
    llm = ev = sm = 0.0
    n_ev = 0
    dollars = lambda f: _episode_dollars(json.loads(f.read_text()).get("tokens") or {}) if f.exists() else 0.0
    for t in range(T_LAST + 1):
        j = _where(chain, t)
        run, t0 = chain[j]
        drafted = chain[j - 1][0] if (j > 0 and t == t0) else run
        rr = cl._rrsi(run)
        llm += sum(_usage_usd(x, lambda u, t=t: u.get("round") == t and u.get("role") != "judge")
                   for x in {run, drafted})
        dr = cl._rrsi(drafted)
        sm += sum(dollars(f) for jd in dr.joinpath("jobs").glob(f"r{t}[A-Z]_smoke")
                  for f in jd.glob("s[0-9]*/*.json"))
        g = rr / f"r{t}" / "group.json"
        grp = json.loads(g.read_text()) if g.exists() else None
        if grp and arm not in grp["members"]:
            grp = None
        if arm in ("s", "c"):
            fr = json.loads((rr / "frontier.json").read_text())
            inc_ev = EvalResult.load(rr / "jobs" / fr["trajectory"][t]["job"] / "eval.json")
            if grp:
                ns = grp["arms"][arm]["episodes"]
            else:
                ns = {p.stem: json.loads(p.read_text())["n"]
                      for p in (rr / f"r{t}" / "sequential").glob("*.json")}
            for v, n in ns.items():
                files = [trial_path(rr, f"r{t}{v}", tid, k) for tid, k in _seq_order(inc_ev, t, v, n)]
                ev += sum(dollars(f) for f in files)
                n_ev += n
        else:
            for jd in rr.joinpath("jobs").glob(f"r{t}[A-Z]"):
                files = list(jd.glob("s[0-9]*/*.json"))
                ev += sum(dollars(f) for f in files)
                n_ev += len(files)
    cal = sum(dollars(f) for j in CAL_JOBS for f in (cl._rrsi(cal_dir(b)) / "jobs" / j).glob("s[0-9]*/*.json"))
    return {"llm": llm, "eval": ev, "smoke": sm, "calibration": cal, "total": llm + ev + sm + cal,
            "eval_episodes": n_ev}


# ----------------------------------------------------------------- missed --
def cmd_missed(runs: list[Path]) -> None:
    """Held-out deployments of the stopped candidates that full evaluation would
    have chosen in a seqfull loop's state (from its shadow evaluation), and of
    the incumbent each would have replaced."""
    dom = load_domain(D)
    for run in runs:
        rr = cl._rrsi(run)
        out = Path(run) / "runs" / "verify" / D
        traj = json.loads((rr / "frontier.json").read_text())["trajectory"]
        todo = []
        for r in json.loads((out / "shadow.json").read_text()):
            v = r["full_choice"]
            if v is None or v == r["seq_choice"] or not r["candidates"][v].get("completed"):
                continue
            cand = json.loads((rr / f"r{r['t']}" / v / "prep.json").read_text())["commit"]
            todo += [cand, traj[r["t"]]["commit"]]
        todo = list(dict.fromkeys(todo))
        new = out / "missed_new.json"            # deployments this command added (for the spend)
        if not new.exists():
            new.write_text(json.dumps([c for c in todo if not
                                       (out / "jobs" / "heldout" / c[:12] / "eval.json").exists()]))
        for c in todo:
            worktree(Path(run), out / "wt", c)
        for c in todo:
            deploy.run(dom, Path(run), out, c, K_HELDOUT)
            print(f"{Path(run).name}: deployed {c[:12]} (k={K_HELDOUT})", flush=True)
        (out / "missed.done").write_text("\n".join(todo) + "\n")


def _missed_gains(runs: dict) -> list:
    out = []
    for n, run in runs.items():
        rr = cl._rrsi(Path(run))
        traj = json.loads((rr / "frontier.json").read_text())["trajectory"]
        sh = Path(run) / "runs" / "verify" / D / "shadow.json"
        for r in json.loads(sh.read_text()):
            v = r["full_choice"]
            if v is None or v == r["seq_choice"] or not r["candidates"][v].get("completed"):
                continue
            cand = json.loads((rr / f"r{r['t']}" / v / "prep.json").read_text())["commit"]
            inc = traj[r["t"]]["commit"]
            try:
                a, b = cl._ho(Path(run), cand), cl._ho(Path(run), inc)
            except FileNotFoundError:
                continue
            out.append({"loop": n, "t": r["t"], "variant": v,
                        "gain": float(np.mean([a[x] - b[x] for x in b]))})
    return out


# ------------------------------------------------------------ spend/status --
def _all_dirs() -> list[tuple[str, str, Path]]:
    """(block, "cal" or the checkout's arms, checkout) of every BK checkout."""
    out = []
    for b in BLOCKS:
        if cal_dir(b).exists():
            out.append((b, "cal", cal_dir(b)))
        out += [(b, "".join(members(p)), p) for p in lineage_dirs(b)]
    return out


def _usage_usd(run: Path, keep=lambda x: True) -> float:
    from .posthoc_e1 import PRICES
    usd = 0.0
    usage = Path(run) / "runs" / "rrsi" / f"{D}.usage.jsonl"
    if usage.exists():
        for line in usage.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                if not keep(x):
                    continue
                pr = PRICES.get(x.get("model"), PRICES["deepseek-v4-pro"])
                usd += (x.get("in", 0) * pr[0] + x.get("cache_read", 0) * pr[1]
                        + x.get("out", 0) * pr[2]) / 1e6
    return usd


def _episode_files(run: Path, seeded: bool) -> list:
    """Every episode on disk in a checkout (evolve side and held-out), without
    the calibration copied into a seeded loop and without the episodes a shadow
    job copied from the loop's own job."""
    rr = cl._rrsi(run)
    files = []
    if (rr / "jobs").exists():
        for job in (rr / "jobs").iterdir():
            if seeded and job.name in CAL_JOBS:
                continue
            for f in job.rglob("s*/*.json"):
                if job.name.endswith("_shadow") and \
                        (rr / "jobs" / job.name[:-7] / f.parent.name / f.name).exists():
                    continue
                files.append(f)
    ho = Path(run) / "runs" / "verify" / D / "jobs" / "heldout"
    if ho.exists():
        files += list(ho.rglob("s*/*.json"))
    return files


def _files_usd(files, seen: set) -> float:
    """Dollars of episode files, each counted once (forked lineages hold hard
    links to their parent's episodes)."""
    from .posthoc_e1 import _episode_dollars
    usd = 0.0
    for f in files:
        try:
            st = f.stat()
            if (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            usd += _episode_dollars(json.loads(f.read_text()).get("tokens") or {})
        except Exception:  # noqa: BLE001 - a file being written
            pass
    return usd


def _spend(run: Path, seeded: bool, seen: set | None = None) -> float:
    """Dollars at list prices of one checkout: model calls and its episodes."""
    return _usage_usd(run) + _files_usd(_episode_files(run, seeded), set() if seen is None else seen)


def _new_deploy_usd(run: Path) -> float:
    from .posthoc_e1 import _episode_dollars
    f = Path(run) / "runs" / "verify" / D / "missed_new.json"
    if not f.exists():
        return 0.0
    usd = 0.0
    for c in json.loads(f.read_text()):
        for g in (f.parent / "jobs" / "heldout" / c[:12]).rglob("s*/*.json"):
            try:
                usd += _episode_dollars(json.loads(g.read_text()).get("tokens") or {})
            except Exception:  # noqa: BLE001 - a file being written
                pass
    return usd


def cmd_spend(limit: float | None) -> float:
    seen = set()
    tot = sum(_spend(p, a != "cal", seen) for _, a, p in _all_dirs())
    tot += sum(_new_deploy_usd(Path(p)) for p in MISSED_IL.values())
    print(f"BK spend ${tot:.2f}" + (f" (limit ${limit:.0f})" if limit else ""))
    if limit is not None and tot > limit:
        sys.exit(3)
    return tot


def cmd_status() -> None:
    for b, a, p in _all_dirs():
        rr = cl._rrsi(p)
        fp = rr / "frontier.json"
        if not fp.exists():
            print(f"{p.name}: not started")
            continue
        fr = json.loads(fp.read_text())
        settled = len(fr["trajectory"]) - 1
        acc = len(dict.fromkeys(x["commit"] for x in fr["trajectory"])) - 1
        delta = (json.loads((rr / "calibration.json").read_text())["delta"]
                 if (rr / "calibration.json").exists() else None)
        done = [f for f in ("bk_loop.done", "bk.done", "cal.done") if (p / "runs" / f).exists()]
        what = "calibration" if a == "cal" else mode_for(members(p))
        if a != "cal" and lineage(p)["parent"]:
            what += f" (forked from {lineage(p)['parent']} in round {lineage(p)['fork_round']})"
        print(f"{p.name}: {what}, settled {settled}/10, accepted {acc}, "
              f"delta {delta}, ${_spend(p, a != 'cal'):.2f} {' '.join(done)}")
    cmd_spend(None)


# --------------------------------------------------------------- analysis --
def _chain(b: str, arm: str) -> list:
    """[(checkout, first round)] of the lineages that ran arm's rounds, oldest first."""
    out, p = [], loop_dir(b, arm)
    while p is not None:
        lin = lineage(p) if (p / "runs" / "bk_lineage.json").exists() else {"parent": None, "fork_round": 0}
        out.append((p, lin["fork_round"]))
        p = HOME / lin["parent"] if lin.get("parent") else None
    return out[::-1]


def _where(chain: list, t: int) -> int:
    return max(j for j, (_, t0) in enumerate(chain) if t0 <= t)


def _ho_tasks(b: str, a: str) -> tuple[dict, dict]:
    tr = cl._traj(loop_dir(b, a))
    base = cl._ho(cal_dir(b), tr["base"])
    final = base if tr["final"] == tr["base"] else cl._ho(cal_dir(b), tr["final"])
    return final, base


def _contrast(means: dict, c: dict) -> float:
    return float(sum(w * means[a] for a, w in c.items()))


def _additive(Y: dict, arms) -> tuple[dict, float, int]:
    """Two-way additive model (block + arm) over `arms`: arm means, residual
    variance, residual df (blocks - 1)(arms - 1)."""
    means = {a: float(np.mean([Y[(b, a)] for b in BLOCKS])) for a in arms}
    grand = float(np.mean([Y[(b, a)] for b in BLOCKS for a in arms]))
    bm = {b: float(np.mean([Y[(b, a)] for a in arms])) for b in BLOCKS}
    resid = [Y[(b, a)] - bm[b] - means[a] + grand for b in BLOCKS for a in arms]
    df = (len(BLOCKS) - 1) * (len(arms) - 1)
    return means, float(np.sum(np.square(resid)) / df), df


def _lineage_se(Y: dict, c: dict) -> tuple[float, float]:
    """Coupled design: standard error and Satterthwaite df of a contrast over the
    arm means. a and s share their rounds with f and h until their states first
    differ, so arm a = f + d_a and s = h + d_s per block. The contrast is split
    into a part over the means of f, h and c, whose residual variance comes from
    the additive model over those three arms (df 2(B - 1)), and a part over the
    mean differences d_a and d_s, whose variances come from their spread over
    blocks (df B - 1 each); the two parts are treated as independent."""
    nb = len(BLOCKS)
    w = {a: 0.0 for a in IND}
    wd = {}
    for a, x in c.items():
        if a in PAIRED:
            w[PAIRED[a]] += x
            wd[a] = wd.get(a, 0.0) + x
        else:
            w[a] += x
    _, s2, df = _additive(Y, IND)
    parts = [(s2 * sum(x * x for x in w.values()) / nb, df)]
    for a, x in wd.items():
        d = [Y[(b, a)] - Y[(b, PAIRED[a])] for b in BLOCKS]
        parts.append((x * x * float(np.var(d, ddof=1)) / nb, nb - 1))
    var = sum(v for v, _ in parts)
    den = sum(v * v / k for v, k in parts if v > 0)
    return math.sqrt(var), (var * var / den if den > 0 else float("inf"))


def cmd_analyze(js: Path, B: int = 4000, seed: int = 13) -> dict:
    ho = {(b, a): _ho_tasks(b, a) for b in BLOCKS for a in ARMS}
    tasks = sorted(next(iter(ho.values()))[1])

    def y(b, a, ts):
        f, base = ho[(b, a)]
        return float(np.mean([f[t] - base[t] for t in ts]))
    Y = {(b, a): y(b, a, tasks) for b in BLOCKS for a in ARMS}
    means = {a: float(np.mean([Y[(b, a)] for b in BLOCKS])) for a in ARMS}
    from scipy.stats import t as tdist
    if COUPLED:
        _, s2, df = _additive(Y, IND)
        model = "lineage model: additive over f, h, c plus the coupled differences a - f, s - h"
    else:
        _, s2, df = _additive(Y, ARMS)
        model = "two-way additive model (block + arm) over all arms"
    rng = np.random.default_rng(seed)
    contrasts = {k: c for k, c in CONTRASTS.items() if all(a in ARMS for a in c)}
    boots = {k: [] for k in contrasts}
    for _ in range(B):
        bs = list(rng.choice(BLOCKS, len(BLOCKS)))
        ts = list(rng.choice(tasks, len(tasks)))
        m = {a: float(np.mean([y(b, a, ts) for b in bs])) for a in ARMS}
        for k, c in contrasts.items():
            boots[k].append(_contrast(m, c))
    res = {"design": DESIGN, "model": model,
           "transfer": {f"{b}_{a}": Y[(b, a)] for b in BLOCKS for a in ARMS},
           "arm_means": means, "resid_sd": math.sqrt(s2), "df": df, "contrasts": {}}
    for k, c in contrasts.items():
        est = _contrast(means, c)
        if COUPLED:
            se, dfk = _lineage_se(Y, c)
        else:
            se, dfk = math.sqrt(s2 * sum(w * w for w in c.values()) / len(BLOCKS)), df
        q = float(tdist.ppf(0.95, dfk)) if math.isfinite(dfk) else 1.6449
        p = (float(2 * tdist.sf(abs(est) / se, dfk)) if math.isfinite(dfk) else None) if se > 0 else None
        res["contrasts"][k] = {"est": est, "se": se, "df": dfk, "ci90_t": [est - q * se, est + q * se],
                               "p": p,
                               "ci90_boot": [float(np.percentile(boots[k], 5)),
                                             float(np.percentile(boots[k], 95))],
                               "per_block": [_contrast({a: Y[(b, a)] for a in c}, c) for b in BLOCKS]}
    if COUPLED:
        res["forks"] = {f"{b}_{a}": (lineage(loop_dir(b, a))["fork_round"]
                                     if lineage(loop_dir(b, a))["parent"] else None)
                        for b in BLOCKS for a in ARMS}
    # what each arm did (descriptive)
    desc = {}
    for b in BLOCKS:
        for a in ARMS:
            run = loop_dir(b, a)
            tr = cl._traj(run)
            rows = _rows(a, b)
            full_acc = [r for r in rows if r["full_choice"] is not None]
            desc[f"{b}_{a}"] = {
                "accepted": len(dict.fromkeys(tr["commits"])) - 1,
                "rounds_full_would_accept": len(full_acc),
                "rounds_seq_differs": sum(r["full_choice"] != r["seq_choice"] for r in rows),
                "stopped_full_choice": sum(1 for r in full_acc if r["full_choice"] in r["stopped"]),
                "stopped": sum(len(r["stopped"]) for r in rows),
                "delta": json.loads((cl._rrsi(run) / "calibration.json").read_text())["delta"],
                "cost": _arm_cost(b, a) if COUPLED else cl._loop_cost(run)}
    res["loops"] = desc
    # secondary: pooled with IL (seqfull vs full), noise band as a covariate
    pool = []
    for n, p in {**il.SEQ, **il.FULL}.items():
        a, b0 = il._transfer(Path(p))
        d = json.loads((cl._rrsi(Path(p)) / "calibration.json").read_text())["delta"]
        pool.append((1.0 if n in il.SEQ else 0.0, d, 0.0, float(np.mean([a[t] - b0[t] for t in tasks]))))
    for b in BLOCKS:
        for a, s in (("s", 1.0), ("f", 0.0)):
            pool.append((s, desc[f"{b}_{a}"]["delta"], 1.0, Y[(b, a)]))
    X = np.array([[1.0, s, d - np.mean([p[1] for p in pool]), st] for s, d, st, _ in pool])
    yy = np.array([p[3] for p in pool])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    r = yy - X @ beta
    dfp = len(yy) - X.shape[1]
    cov = (r @ r / dfp) * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[1, 1])
    qp = float(tdist.ppf(0.95, dfp))
    res["pooled_with_il"] = {"n_seq": int(sum(p[0] for p in pool)), "n_full": int(sum(1 - p[0] for p in pool)),
                             "seq_minus_full_delta_adjusted": float(beta[1]),
                             "ci90": [float(beta[1] - qp * se), float(beta[1] + qp * se)], "df": dfp}
    try:
        res["missed_il"] = _missed_gains(MISSED_IL)
    except FileNotFoundError:
        res["missed_il"] = None
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1))
    pp = lambda x: f"{100 * x:+.2f}"
    for a in ARMS:
        print(f"{a} ({ARMS[a]}): mean transfer@10 {pp(means[a])}  "
              + " ".join(pp(Y[(b, a)]) for b in BLOCKS))
    for k, x in res["contrasts"].items():
        pv = "n/a" if x["p"] is None else f"{x['p']:.3f}"
        print(f"{k}: {pp(x['est'])} 90% t [{pp(x['ci90_t'][0])}, {pp(x['ci90_t'][1])}] p {pv}; "
              f"bootstrap [{pp(x['ci90_boot'][0])}, {pp(x['ci90_boot'][1])}]")
    x = res["pooled_with_il"]
    print(f"pooled with IL ({x['n_seq']} v {x['n_full']}), delta-adjusted: "
          f"{pp(x['seq_minus_full_delta_adjusted'])} [{pp(x['ci90'][0])}, {pp(x['ci90'][1])}]")
    return res


def cmd_tables(js: Path, out: Path) -> None:
    res = json.loads(Path(js).read_text())
    pp = lambda x: f"{100 * x:+.1f}"
    names = {"f": ("Full", "full", "full"), "s": ("Sequential", "seq.", "early"),
             "c": ("Cost-aware seq.", "cost-aware", "early"), "h": ("History only", "full", "early"),
             "a": ("Admission only", "seq.", "full")}
    rows = []
    for a in ARMS:
        loops = [res["loops"][f"{b}_{a}"] for b in BLOCKS]
        n, adm, hist = names[a]
        rows.append(f"{n} & {adm} & {hist} & "
                    + " & ".join(f"${pp(res['transfer'][f'{b}_{a}'])}$" for b in BLOCKS)
                    + f" & ${pp(res['arm_means'][a])}$ & {sum(x['accepted'] for x in loops)} \\\\")
    rows.append("\\midrule")
    lab = {"seq_minus_full": "Sequential $-$ full", "cost_minus_full": "Cost-aware $-$ full",
           "cost_minus_seq": "Cost-aware $-$ sequential", "admission": "Admission (main effect)",
           "history": "History (main effect)", "interaction": "Interaction"}
    nb = len(BLOCKS)
    for k, x in res["contrasts"].items():
        rows.append(f"\\multicolumn{{3}}{{l}}{{{lab[k]}}} & \\multicolumn{{{nb}}}{{l}}"
                    f"{{90\\% $[{pp(x['ci90_t'][0])},{pp(x['ci90_t'][1])}]$}} & ${pp(x['est'])}$ & \\\\")
    tex = ("\\begin{tabular}{lll" + "r" * (nb + 2) + "}\n\\toprule\n"
           "Arm & Admission & History & " + " & ".join(b.upper() for b in BLOCKS)
           + " & Mean & Accepted \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("arms", "starts", "seed", "selection", "deploy", "deploy-base", "post",
                                    "replay", "missed", "spend", "status", "analyze", "tables"))
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--cal")
    ap.add_argument("--arms")
    ap.add_argument("--block")
    ap.add_argument("--limit", type=float)
    ap.add_argument("--json", default="results/bk/analysis.json")
    ap.add_argument("--tex", default="paper/tables/bk.tex")
    a = ap.parse_args()
    if a.cmd == "arms":
        print(" ".join(f"{k}={v}" for k, v in ARMS.items()))
    elif a.cmd == "starts":
        for p, arms in starts(a.block):
            print(p, ",".join(arms))
    elif a.cmd == "seed":
        cmd_seed(Path(a.cal), Path(a.run[0]), a.arms.split(","))
    elif a.cmd == "selection":
        print(mode_for(members(Path(a.run[0]))))
    elif a.cmd == "deploy":
        cmd_deploy(Path(a.run[0]))
    elif a.cmd == "deploy-base":
        cmd_deploy_base(Path(a.run[0]))
    elif a.cmd == "post":
        cmd_post(Path(a.run[0]))
    elif a.cmd == "replay":
        cmd_replay(Path(a.run[0]))
    elif a.cmd == "missed":
        cmd_missed([Path(x) for x in (a.run or MISSED_IL.values())])
    elif a.cmd == "spend":
        cmd_spend(a.limit)
    elif a.cmd == "status":
        cmd_status()
    elif a.cmd == "analyze":
        cmd_analyze(Path(a.json))
    else:
        cmd_tables(Path(a.json), Path(a.tex))


if __name__ == "__main__":
    main()
