"""Experiment CL (verify/PREREGISTRATION_CL.md): RRSI on tau2 airline driven by
sequential full evaluation.

  python -m verify.cl deploy  --run DIR      held-out deployments of the base, every
                                             accepted incumbent and the final incumbent
  python -m verify.cl shadow  --run DIR      complete every dropped candidate's evolve
                                             evaluation and apply RRSI's rule to it
  python -m verify.cl status  --run DIR ...  progress, episodes and spend (no held-out S)
  python -m verify.cl analyze --run CL1 CL2 [--json results/cl/analysis.json]
                                             the registered outcomes, with r1 to r3 as reference

DIR is a trajectory's checkout (its runs/ holds rrsi/ and verify/). Deployments
go to DIR/runs/verify/tau2_airline/jobs/heldout/<commit12>/, as for r1 to r3.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from rrsi.domain import load_domain
from rrsi.evaluate import aggregate, relative_cost_change
from rrsi.selection import cost_rule

from . import deploy
from .state import worktree

D = "tau2_airline"
K_HELDOUT = 12
T_LAST = 9


def _rrsi(run: Path) -> Path:
    return Path(run) / "runs" / "rrsi" / D


def _commits(run: Path) -> list[str]:
    """Base, every accepted incumbent, in order (the last is the final one)."""
    fr = json.loads((_rrsi(run) / "frontier.json").read_text())
    out = []
    for x in fr["trajectory"][:T_LAST + 2]:
        if not out or out[-1] != x["commit"]:
            out.append(x["commit"])
    return out


def cmd_deploy(run: Path) -> None:
    dom = load_domain(D)
    out = Path(run) / "runs" / "verify" / D
    out.mkdir(parents=True, exist_ok=True)
    for c in _commits(run):
        deploy.run(dom, Path(run), out, c, K_HELDOUT)
        print(f"deployed {c[:12]} (k={K_HELDOUT})", flush=True)     # S deliberately not printed
    (out / "deploy.done").write_text("\n".join(_commits(run)) + "\n")


def cmd_shadow(run: Path) -> None:
    """Full evaluation of every dropped candidate, decided by RRSI's rule with the
    round's S*, noise band and the candidate's novelty; the decision full
    evaluation would have made in that round. The loop's episodes are copied
    into a separate job r<t><v>_shadow and completed there, so the loop's own
    job directories keep exactly the episodes the loop ran."""
    import shutil
    dom = load_domain(D)
    rr = _rrsi(run)
    ids = dom.evolve_ids()
    cfg = json.loads((dom.root / "rrsi.json").read_text())
    k = int(cfg["k"])
    rows = []
    for t in range(T_LAST + 1):
        sp = rr / f"r{t}" / "selection.json"
        if not sp.exists():
            continue
        sel = json.loads(sp.read_text())
        decs = {x["variant"]: x for x in json.loads((rr / f"r{t}" / "decisions.json").read_text())}
        S_inc, S_star, delta = sel["S_inc"], sel["S_star"], sel["delta"]
        full = {}
        for v, rec in sel["candidates"].items():
            if not rec["dropped"]:
                d = decs[v]
                full[v] = {"S": d.get("S"), "admissible": bool(d.get("admissible")), "completed": False}
                continue
            job, sjob = f"r{t}{v}", f"r{t}{v}_shadow"
            for f in (rr / "jobs" / job).glob("s*/*.json"):
                g = rr / "jobs" / sjob / f.parent.name / f.name
                if not g.exists():
                    g.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, g)
            prep = json.loads((rr / f"r{t}" / v / "prep.json").read_text())
            wt = worktree(Path(run), rr / "wt_shadow", prep["commit"])
            dom.run(wt, rr, sjob, ids, k, log_prefix=f"shadow {sjob}")
            per, extra = dom.score(rr, sjob, ids, k)
            ev = aggregate(sjob, k, per, extra)
            dS, dC = ev.S - S_inc, relative_cost_change(ev.C, _inc_C(rr, t))
            ok, why = ev.S >= S_star - delta, "below floor"
            if ok:
                ok, why = cost_rule(dS, dC, rec["nu"], delta, _Cfg(cfg))
            g = dom.guards(None, ev)
            if ok and g:
                ok, why = False, "domain guard violated: " + "; ".join(g)
            full[v] = {"S": ev.S, "dS": dS, "dC": dC, "admissible": ok, "why": why,
                       "completed": True, "n_seq": rec["n"], "m_seq": rec["m"]}
        adm = {v: x for v, x in full.items() if x["admissible"]}
        full_choice = max(adm, key=lambda v: adm[v]["S"]) if adm else None
        rows.append({"t": t, "seq_choice": sel["winner"], "full_choice": full_choice,
                     "candidates": full})
        print(f"r{t}: seq {sel['winner']} full {full_choice}", flush=True)
    out = Path(run) / "runs" / "verify" / D
    out.mkdir(parents=True, exist_ok=True)
    (out / "shadow.json").write_text(json.dumps(rows, indent=1))


class _Cfg:
    def __init__(self, d):
        self.beta0, self.beta1 = d["beta0"], d["beta1"]
        self.w_s, self.w_c, self.w_n = d["w_s"], d["w_c"], d["w_n"]


def _inc_C(rr: Path, t: int):
    """Mean policy tokens of the incumbent the round started from."""
    return json.loads((rr / "frontier.json").read_text())["trajectory"][t]["C"]


def cmd_status(runs: list[Path]) -> None:
    from .posthoc_e1 import PRICES
    for run in runs:
        rr = _rrsi(run)
        fr = json.loads((rr / "frontier.json").read_text()) if (rr / "frontier.json").exists() else None
        settled = len(fr["trajectory"]) - 1 if fr else -1
        ep = planned = 0
        for t in range(T_LAST + 1):
            sp = rr / f"r{t}" / "selection.json"
            if sp.exists():
                c = json.loads(sp.read_text())["cost"]
                ep += c["selection_episodes"]
                planned += c.get("planned_episodes", 0)
        usd = 0.0
        usage = Path(run) / "runs" / "rrsi" / f"{D}.usage.jsonl"
        if usage.exists():
            for line in usage.read_text().splitlines():
                if line.strip():
                    x = json.loads(line)
                    pr = PRICES.get(x.get("model"), PRICES["deepseek-v4-pro"])
                    usd += (x.get("in", 0) * pr[0] + x.get("cache_read", 0) * pr[1] + x.get("out", 0) * pr[2]) / 1e6
        n_ep = sum(1 for _ in (rr / "jobs").rglob("s*/*.json")) if (rr / "jobs").exists() else 0
        vdir = Path(run) / "runs" / "verify" / D / "jobs" / "heldout"
        n_ho = sum(1 for _ in vdir.rglob("s*/*.json")) if vdir.exists() else 0
        print(f"{Path(run).name}: settled rounds {settled + 1 if settled >= 0 else 0}/10, "
              f"accepted {len(_commits(run)) - 1 if fr else 0}, candidate episodes {ep}/{planned} planned, "
              f"evolve-side episodes on disk {n_ep}, held-out episodes {n_ho}, model calls ${usd:.2f}")


# --------------------------------------------------------------- analysis --
REFS = {"r1": "/home/user/e1", "r2": "/home/user/e1r2", "r3": "/home/user/e1r3"}


def _ho(run: Path, commit: str) -> dict:
    e = json.loads((Path(run) / "runs" / "verify" / D / "jobs" / "heldout" / commit[:12] /
                    "eval.json").read_text())
    return {t: float(np.mean(x["rewards"])) for t, x in e["per_task"].items()}


def _gain(a: dict, b: dict, tasks) -> float:
    return float(np.mean([a[t] - b[t] for t in tasks]))


def _traj(run: Path) -> dict:
    fr = json.loads((_rrsi(run) / "frontier.json").read_text())
    tr = fr["trajectory"][:T_LAST + 2]
    return {"base": tr[0]["commit"], "final": tr[-1]["commit"], "S": [x["S"] for x in tr],
            "commits": [x["commit"] for x in tr]}


def _cutoff(run: Path) -> float:
    """End of round 9: when its decisions were written."""
    return (_rrsi(run) / f"r{T_LAST}" / "decisions.json").stat().st_mtime


def _loop_cost(run: Path) -> dict:
    """Dollars of rounds 0..9 at list prices: model calls by role (the judge, study
    evidence, excluded) up to the end of round 9, candidate evaluations, smoke
    tests and the base calibration; and candidate-evaluation episodes."""
    from .posthoc_e1 import PRICES, _episode_dollars
    rr = _rrsi(run)
    cut = _cutoff(run) + 1
    llm = 0.0
    for line in (Path(run) / "runs" / "rrsi" / f"{D}.usage.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        x = json.loads(line)
        if x.get("role") == "judge" or x.get("t", 0) > cut:
            continue
        pr = PRICES.get(x.get("model"), PRICES["deepseek-v4-pro"])
        llm += (x.get("in", 0) * pr[0] + x.get("cache_read", 0) * pr[1] + x.get("out", 0) * pr[2]) / 1e6
    ev = sm = cal = 0.0
    n_ev = 0
    for job in (rr / "jobs").iterdir():
        m = re.fullmatch(r"r(\d+)[A-Z](_smoke)?", job.name)
        if m and int(m.group(1)) > T_LAST:
            continue
        if not m and job.name not in ("base", "heldout_base2", "heldout_base3"):
            continue                                   # shadow jobs and anything else
        recs = [json.loads(f.read_text()) for f in job.glob("s[0-9]*/*.json")]
        usd = sum(_episode_dollars(r.get("tokens") or {}) for r in recs)
        if not m:
            cal += usd
        elif m.group(2):
            sm += usd
        else:
            ev += usd
            n_ev += len(recs)
    return {"llm": llm, "eval": ev, "smoke": sm, "calibration": cal, "total": llm + ev + sm + cal,
            "eval_episodes": n_ev}


def _accepts(run: Path) -> list:
    rr = _rrsi(run)
    out = []
    for t in range(T_LAST + 1):
        decs = json.loads((rr / f"r{t}" / "decisions.json").read_text())
        fr_t = _traj(run)["commits"]
        if fr_t[t + 1] != fr_t[t]:
            w = [x for x in decs if x.get("admissible")]
            w = max(w, key=lambda x: x.get("S") or 0) if w else None
            out.append({"t": t, "variant": w and w["variant"],
                        "within_band": bool(w and "within delta" in w["reason"])})
    return out


def cmd_analyze(cl: list[Path], js: Path, B: int = 2000, seed: int = 5) -> dict:
    runs = {Path(r).name: Path(r) for r in cl}
    runs.update({k: Path(v) for k, v in REFS.items()})
    tr = {n: _traj(r) for n, r in runs.items()}
    ho = {n: {c: _ho(r, c) for c in dict.fromkeys(tr[n]["commits"])} for n, r in runs.items()}
    tasks = sorted(ho["r1"][tr["r1"]["base"]])
    cl_n, ref_n = [Path(r).name for r in cl], list(REFS)

    def transfer(n, ts):
        return _gain(ho[n][tr[n]["final"]], ho[n][tr[n]["base"]], ts)
    point = {n: transfer(n, tasks) for n in runs}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        ts = list(rng.choice(tasks, len(tasks)))
        boots.append({n: transfer(n, ts) for n in runs})
    ci = {n: [float(np.percentile([b[n] for b in boots], q)) for q in (5, 95)] for n in runs}
    diff = [np.mean([b[n] for n in cl_n]) - np.mean([b[n] for n in ref_n]) for b in boots]
    lo = min(ci[n][0] for n in ref_n)
    hi = max(ci[n][1] for n in ref_n)
    res = {"transfer": {n: {"point": point[n], "ci90": ci[n]} for n in runs},
           "diff_cl_minus_ref": [float(np.mean([point[n] for n in cl_n]) - np.mean([point[n] for n in ref_n])),
                                 float(np.percentile(diff, 5)), float(np.percentile(diff, 95))],
           "ref_band": [lo, hi],
           "consistent": all(lo <= point[n] <= hi for n in cl_n)}
    # accepted changes' held-out gains (incumbent over the one it replaced)
    res["accepted_gains"] = {}
    for n in runs:
        cs = list(dict.fromkeys(tr[n]["commits"]))
        res["accepted_gains"][n] = [_gain(ho[n][b], ho[n][a], tasks) for a, b in zip(cs, cs[1:])]
    res["accepts"] = {n: _accepts(r) for n, r in runs.items()}
    res["evolve_S"] = {n: tr[n]["S"] for n in runs}
    res["cost"] = {n: _loop_cost(r) for n, r in runs.items()}
    seq = {}
    for n in cl_n:
        rr = _rrsi(runs[n])
        sels = [json.loads((rr / f"r{t}" / "selection.json").read_text()) for t in range(T_LAST + 1)]
        seq[n] = {"episodes": sum(x["cost"]["selection_episodes"] for x in sels),
                  "planned": sum(x["cost"].get("planned_episodes", 0) for x in sels),
                  "dropped": sum(1 for x in sels for c in x["candidates"].values() if c["dropped"]),
                  "measured": sum(len(x["candidates"]) for x in sels)}
        sh = Path(runs[n]) / "runs" / "verify" / D / "shadow.json"
        if sh.exists():
            rows = json.loads(sh.read_text())
            acc = [r for r in rows if r["full_choice"] is not None]
            seq[n]["shadow"] = {"rounds": len(rows),
                                "agree": float(np.mean([r["seq_choice"] == r["full_choice"] for r in rows])),
                                "full_accepts": len(acc),
                                "recall": (float(np.mean([r["seq_choice"] == r["full_choice"] for r in acc]))
                                           if acc else None),
                                "seq_accepts": sum(1 for r in rows if r["seq_choice"] is not None)}
    res["sequential"] = seq
    js.write_text(json.dumps(res, indent=1))
    pp = lambda x: f"{100 * x:+.2f}"
    for n in runs:
        c = res["cost"][n]
        print(f"{n}: transfer@10 {pp(point[n])} [{pp(ci[n][0])}, {pp(ci[n][1])}]  accepts "
              f"{len(res['accepts'][n])}  loop ${c['total']:.2f} (llm {c['llm']:.2f}, eval {c['eval']:.2f}, "
              f"eval episodes {c['eval_episodes']})")
    print("cl - ref", [pp(x) for x in res["diff_cl_minus_ref"]], "consistent", res["consistent"])
    print(json.dumps(seq, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("deploy", "shadow", "status", "analyze"))
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--json", default="results/cl/analysis.json")
    a = ap.parse_args()
    if a.cmd == "deploy":
        cmd_deploy(Path(a.run[0]))
    elif a.cmd == "shadow":
        cmd_shadow(Path(a.run[0]))
    elif a.cmd == "analyze":
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        cmd_analyze([Path(x) for x in a.run], Path(a.json))
    else:
        cmd_status([Path(x) for x in a.run])


if __name__ == "__main__":
    main()
