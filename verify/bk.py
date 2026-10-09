"""Experiment BK (verify/PREREGISTRATION_BK.md): a blocked 2x2 that separates
the two things sequential evaluation changes in an RRSI loop on tau2 airline,
which candidates can be accepted (admission) and what the proposer's history
records for the candidates the rule stops (history).

  arm  selection  admission    history
  f    full       full         full evaluation
  s    seqfull    sequential   early-stopped estimates
  h    seqhist    full         early-stopped estimates
  a    seqadm     sequential   full evaluation

Each block calibrates the noise band once (baseline plus two repeated base
evaluations, in /home/user/bk_<block>_cal) and seeds its four loops with that
calibration, so the arms of a block start from the same state and the same
delta. The base is deployed on the held-out tasks once per block.

  python -m verify.bk seed    --cal DIR --run DIR --ns NS --selection SEL
  python -m verify.bk deploy  --run DIR       final incumbent on the held-out tasks
  python -m verify.bk deploy-base --run CAL   the block's base on the held-out tasks
  python -m verify.bk replay  --run DIR       (full arm) where seqfull would have stopped
                                              each candidate, and the choice it would make
  python -m verify.bk missed  --run DIR ...   (IL seqfull loops) held-out deployments of the
                                              stopped candidates full evaluation would have chosen
  python -m verify.bk spend   [--limit USD]   spend of every BK loop (exit 3 above the limit)
  python -m verify.bk status                  progress (no held-out S)
  python -m verify.bk analyze [--json results/bk/analysis.json]
  python -m verify.bk tables  [--json ...] [--tex paper/tables/bk.tex]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
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
from .state import worktree

D = cl.D
T_LAST = cl.T_LAST
K_HELDOUT = cl.K_HELDOUT
HOME = Path("/home/user")
BLOCKS = ("b1", "b2", "b3", "b4")
ARMS = {"f": "full", "s": "seqfull", "h": "seqhist", "a": "seqadm"}
ADM = {"f": 0, "h": 0, "s": 1, "a": 1}          # 1 = sequential admission
HIST = {"f": 0, "a": 0, "s": 1, "h": 1}         # 1 = early-stopped history
CAL_JOBS = ("base", "heldout_base2", "heldout_base3")
# contrasts over the arm means (f, s, h, a)
CONTRASTS = {
    "seq_minus_full": {"s": 1, "f": -1},
    "admission": {"s": .5, "a": .5, "f": -.5, "h": -.5},
    "history": {"s": .5, "h": .5, "f": -.5, "a": -.5},
    "interaction": {"s": 1, "a": -1, "h": -1, "f": 1},   # (s - a) - (h - f)
}
MISSED_IL = {"s2": "/home/user/il_s2", "s3": "/home/user/il_s3", "s4": "/home/user/il_s4"}


def loop_dir(b: str, arm: str) -> Path:
    return HOME / f"bk_{b}_{arm}"


def cal_dir(b: str) -> Path:
    return HOME / f"bk_{b}_cal"


# ------------------------------------------------------------------ setup --
def cmd_seed(cal: Path, run: Path, ns: str, sel: str) -> None:
    """Copy the block's calibration (frontier, noise band, history, and the three
    base evaluations the round-0 analysis reads) into a loop checkout, and put
    the loop's branch evolve/<ns>/<D> at the same base commit."""
    src, dst = cl._rrsi(cal), cl._rrsi(run)
    if (dst / "frontier.json").exists():
        print(f"{run.name}: already seeded")
        return
    calib = json.loads((src / "calibration.json").read_text())
    if list(calib.get("jobs", [])) != list(CAL_JOBS):
        sys.exit(f"{cal}: calibration not finished (jobs {calib.get('jobs')})")
    fr = json.loads((src / "frontier.json").read_text())
    if len(fr["trajectory"]) != 1:
        sys.exit(f"{cal}: the calibration checkout ran rounds; it must stay at the base")
    branch = f"evolve/{ns}/{D}"
    G.ensure_branch(run, branch, "HEAD")
    head = G.rev(run, branch)
    if not (head.startswith(fr["incumbent"]["commit"]) or fr["incumbent"]["commit"].startswith(head)):
        sys.exit(f"{run}: {branch} at {head}, block base is {fr['incumbent']['commit']}")
    if G.tree_hash(run, branch, f"domains/{D}") != G.tree_hash(cal, "HEAD", f"domains/{D}"):
        sys.exit(f"{run}: domain tree differs from the calibration checkout")
    dst.mkdir(parents=True, exist_ok=True)
    for j in CAL_JOBS:
        shutil.copytree(src / "jobs" / j, dst / "jobs" / j, dirs_exist_ok=True)
    shutil.copy2(src / "calibration.json", dst / "calibration.json")
    shutil.copy2(src / "history.jsonl", dst / "history.jsonl")
    fr["config"].update(selection=sel, branch_ns=ns)
    fr["seeded_from"] = str(cal)
    (dst / "frontier.json").write_text(json.dumps(fr, indent=1))   # last: marks the seed done
    print(f"{run.name}: seeded from {cal.name} (delta {calib['delta']:.4f}), {branch} at {head}")


def cmd_deploy(run: Path) -> None:
    """The final incumbent on the held-out tasks (the base is deployed once per
    block, in the calibration checkout). Resume-safe."""
    dom = load_domain(D)
    out = Path(run) / "runs" / "verify" / D
    out.mkdir(parents=True, exist_ok=True)
    tr = cl._traj(run)
    if tr["final"] != tr["base"]:
        deploy.run(dom, Path(run), out, tr["final"], K_HELDOUT)
        print(f"deployed {tr['final'][:12]} (k={K_HELDOUT})", flush=True)   # S deliberately not printed
    (out / "deploy.done").write_text(tr["final"] + "\n")


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
    """Full arm: for each round, where seqfull would have stopped each candidate
    (the rule replayed on the candidate's own episodes, in seqfull's order) and
    which candidate seqfull would then have chosen. Runs no episode."""
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
        sdir = rr.parent.parent / "verify" / D / "replay" / f"r{t}"
        sdir.mkdir(parents=True, exist_ok=True)
        recs, full = {}, {}
        for c, x in cands:
            recs[c.variant] = _seq_candidate(fake, t, c, inc_ev, S_star, delta,
                                             novelty(c.components, counts), sdir, execute=False)
            full[c.variant] = {"S": x["S"], "admissible": bool(x["admissible"])}
        adm = {v: f for v, f in full.items() if f["admissible"]}
        alive = {v: f for v, f in adm.items() if not recs[v]["dropped"]}
        rows.append({"t": t, "full_choice": max(adm, key=lambda v: adm[v]["S"]) if adm else None,
                     "seq_choice": max(alive, key=lambda v: alive[v]["S"]) if alive else None,
                     "stopped": sorted(v for v, r in recs.items() if r["dropped"]),
                     "candidates": {v: {**full[v], "n_seq": recs[v]["n"], "m_seq": recs[v]["m"],
                                        "dropped": recs[v]["dropped"]} for v in recs}})
    out = rr.parent.parent / "verify" / D / "replay.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"{run.name}: seqfull would differ in "
          f"{sum(r['full_choice'] != r['seq_choice'] for r in rows)} of {len(rows)} rounds")


def _rows(arm: str, run: Path) -> list:
    """Per round: full evaluation's choice, seqfull's choice, and the stopped set,
    from the source each arm has (shadow, replay or the arm's own record)."""
    rr = cl._rrsi(run)
    v = rr.parent.parent / "verify" / D
    if arm == "s":
        rows = json.loads((v / "shadow.json").read_text())
        return [{"t": r["t"], "full_choice": r["full_choice"], "seq_choice": r["seq_choice"],
                 "stopped": sorted(k for k, c in r["candidates"].items() if c.get("completed"))}
                for r in rows]
    if arm == "f":
        return json.loads((v / "replay.json").read_text())
    out = []
    for t in range(T_LAST + 1):
        s = json.loads((rr / f"r{t}" / "selection.json").read_text())
        out.append({"t": t, "full_choice": s["full_choice"], "seq_choice": s["seq_choice"],
                    "stopped": s["stopped"]})
    return out


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
    out = []
    for b in BLOCKS:
        out.append((b, "cal", cal_dir(b)))
        out += [(b, a, loop_dir(b, a)) for a in ARMS]
    return [x for x in out if x[2].exists()]


def _spend(run: Path, seeded: bool) -> float:
    """Dollars at list prices of one checkout: model calls, every episode on disk
    (evolve side and held-out), without the calibration copied into a seeded loop."""
    from .posthoc_e1 import PRICES, _episode_dollars
    usd = 0.0
    usage = Path(run) / "runs" / "rrsi" / f"{D}.usage.jsonl"
    if usage.exists():
        for line in usage.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                pr = PRICES.get(x.get("model"), PRICES["deepseek-v4-pro"])
                usd += (x.get("in", 0) * pr[0] + x.get("cache_read", 0) * pr[1]
                        + x.get("out", 0) * pr[2]) / 1e6
    rr = cl._rrsi(run)
    files = []
    if (rr / "jobs").exists():
        for job in (rr / "jobs").iterdir():
            if seeded and job.name in CAL_JOBS:
                continue
            files += list(job.rglob("s*/*.json"))
    ho = Path(run) / "runs" / "verify" / D / "jobs" / "heldout"
    if ho.exists():
        files += list(ho.rglob("s*/*.json"))
    for f in files:
        try:
            usd += _episode_dollars(json.loads(f.read_text()).get("tokens") or {})
        except Exception:  # noqa: BLE001 - a file being written
            pass
    return usd


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
    tot = sum(_spend(p, a != "cal") for _, a, p in _all_dirs())
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
        print(f"{p.name}: {ARMS.get(a, 'calibration')}, settled {settled}/10, accepted {acc}, "
              f"delta {delta}, ${_spend(p, a != 'cal'):.2f} {' '.join(done)}")
    cmd_spend(None)


# --------------------------------------------------------------- analysis --
def _ho_tasks(b: str, a: str) -> tuple[dict, dict]:
    run = loop_dir(b, a)
    tr = cl._traj(run)
    base = cl._ho(cal_dir(b), tr["base"])
    final = base if tr["final"] == tr["base"] else cl._ho(run, tr["final"])
    return final, base


def _contrast(means: dict, c: dict) -> float:
    return float(sum(w * means[a] for a, w in c.items()))


def cmd_analyze(js: Path, B: int = 4000, seed: int = 13) -> dict:
    ho = {(b, a): _ho_tasks(b, a) for b in BLOCKS for a in ARMS}
    tasks = sorted(next(iter(ho.values()))[1])

    def y(b, a, ts):
        f, base = ho[(b, a)]
        return float(np.mean([f[t] - base[t] for t in ts]))
    Y = {(b, a): y(b, a, tasks) for b in BLOCKS for a in ARMS}
    means = {a: float(np.mean([Y[(b, a)] for b in BLOCKS])) for a in ARMS}
    # two-way additive model (block + arm): residual variance with (4-1)(4-1) = 9 df
    grand = float(np.mean(list(Y.values())))
    bm = {b: float(np.mean([Y[(b, a)] for a in ARMS])) for b in BLOCKS}
    resid = [Y[(b, a)] - bm[b] - means[a] + grand for b in BLOCKS for a in ARMS]
    df = (len(BLOCKS) - 1) * (len(ARMS) - 1)
    s2 = float(np.sum(np.square(resid)) / df)
    from scipy.stats import t as tdist
    q = float(tdist.ppf(0.95, df))
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in CONTRASTS}
    for _ in range(B):
        bs = list(rng.choice(BLOCKS, len(BLOCKS)))
        ts = list(rng.choice(tasks, len(tasks)))
        m = {a: float(np.mean([y(b, a, ts) for b in bs])) for a in ARMS}
        for k, c in CONTRASTS.items():
            boots[k].append(_contrast(m, c))
    res = {"transfer": {f"{b}_{a}": Y[(b, a)] for b in BLOCKS for a in ARMS},
           "arm_means": means, "resid_sd": math.sqrt(s2), "df": df, "contrasts": {}}
    for k, c in CONTRASTS.items():
        est = _contrast(means, c)
        se = math.sqrt(s2 * sum(w * w for w in c.values()) / len(BLOCKS))
        p = float(2 * tdist.sf(abs(est) / se, df)) if se > 0 else None
        res["contrasts"][k] = {"est": est, "se": se, "ci90_t": [est - q * se, est + q * se], "p": p,
                               "ci90_boot": [float(np.percentile(boots[k], 5)),
                                             float(np.percentile(boots[k], 95))],
                               "per_block": ([Y[(b, "s")] - Y[(b, "f")] for b in BLOCKS]
                                             if k == "seq_minus_full" else None)}
    # what each arm did (descriptive)
    desc = {}
    for b in BLOCKS:
        for a in ARMS:
            run = loop_dir(b, a)
            tr = cl._traj(run)
            rows = _rows(a, run)
            full_acc = [r for r in rows if r["full_choice"] is not None]
            desc[f"{b}_{a}"] = {
                "accepted": len(dict.fromkeys(tr["commits"])) - 1,
                "rounds_full_would_accept": len(full_acc),
                "rounds_seq_differs": sum(r["full_choice"] != r["seq_choice"] for r in rows),
                "stopped_full_choice": sum(1 for r in full_acc if r["full_choice"] in r["stopped"]),
                "stopped": sum(len(r["stopped"]) for r in rows),
                "delta": json.loads((cl._rrsi(run) / "calibration.json").read_text())["delta"],
                "cost": cl._loop_cost(run)}
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
        print(f"{k}: {pp(x['est'])} 90% t [{pp(x['ci90_t'][0])}, {pp(x['ci90_t'][1])}] p {x['p']:.3f}; "
              f"bootstrap [{pp(x['ci90_boot'][0])}, {pp(x['ci90_boot'][1])}]")
    x = res["pooled_with_il"]
    print(f"pooled with IL ({x['n_seq']} v {x['n_full']}), delta-adjusted: "
          f"{pp(x['seq_minus_full_delta_adjusted'])} [{pp(x['ci90'][0])}, {pp(x['ci90'][1])}]")
    return res


def cmd_tables(js: Path, out: Path) -> None:
    res = json.loads(Path(js).read_text())
    pp = lambda x: f"{100 * x:+.1f}"
    names = {"f": "Full", "s": "Sequential", "h": "History only", "a": "Admission only"}
    rows = []
    for a in ("f", "s", "h", "a"):
        loops = [res["loops"][f"{b}_{a}"] for b in BLOCKS]
        rows.append(f"{names[a]} & {'seq.' if ADM[a] else 'full'} & {'early' if HIST[a] else 'full'} & "
                    + " & ".join(f"${pp(res['transfer'][f'{b}_{a}'])}$" for b in BLOCKS)
                    + f" & ${pp(res['arm_means'][a])}$ & {sum(x['accepted'] for x in loops)} \\\\")
    rows.append("\\midrule")
    lab = {"seq_minus_full": "Sequential $-$ full", "admission": "Admission (main effect)",
           "history": "History (main effect)", "interaction": "Interaction"}
    for k, x in res["contrasts"].items():
        rows.append(f"\\multicolumn{{3}}{{l}}{{{lab[k]}}} & \\multicolumn{{4}}{{l}}"
                    f"{{90\\% $[{pp(x['ci90_t'][0])},{pp(x['ci90_t'][1])}]$}} & ${pp(x['est'])}$ & \\\\")
    tex = ("\\begin{tabular}{lllrrrrrr}\n\\toprule\n"
           "Arm & Admission & History & B1 & B2 & B3 & B4 & Mean & Accepted \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("seed", "deploy", "deploy-base", "replay", "missed", "spend",
                                    "status", "analyze", "tables"))
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--cal")
    ap.add_argument("--ns")
    ap.add_argument("--selection")
    ap.add_argument("--limit", type=float)
    ap.add_argument("--json", default="results/bk/analysis.json")
    ap.add_argument("--tex", default="paper/tables/bk.tex")
    a = ap.parse_args()
    if a.cmd == "seed":
        cmd_seed(Path(a.cal), Path(a.run[0]), a.ns, a.selection)
    elif a.cmd == "deploy":
        cmd_deploy(Path(a.run[0]))
    elif a.cmd == "deploy-base":
        cmd_deploy_base(Path(a.run[0]))
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
