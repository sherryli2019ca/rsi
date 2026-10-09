"""Experiment CL (verify/PREREGISTRATION_CL.md): RRSI on tau2 airline driven by
sequential full evaluation.

  python -m verify.cl deploy  --run DIR      held-out deployments of the base, every
                                             accepted incumbent and the final incumbent
  python -m verify.cl shadow  --run DIR      complete every dropped candidate's evolve
                                             evaluation and apply RRSI's rule to it
  python -m verify.cl status  --run DIR ...  progress, episodes and spend (no held-out S)

DIR is a trajectory's checkout (its runs/ holds rrsi/ and verify/). Deployments
go to DIR/runs/verify/tau2_airline/jobs/heldout/<commit12>/, as for r1 to r3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    evaluation would have made in that round."""
    dom = load_domain(D)
    rr = _rrsi(run)
    ids = dom.evolve_ids()
    cfg = json.loads((dom.root / "rrsi.json").read_text())
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
            job = f"r{t}{v}"
            if not rec["dropped"]:
                d = decs[v]
                full[v] = {"S": d.get("S"), "admissible": bool(d.get("admissible")), "completed": False}
                continue
            prep = json.loads((rr / f"r{t}" / v / "prep.json").read_text())
            wt = worktree(Path(run), rr / "wt_shadow", prep["commit"])
            dom.run(wt, rr, job, ids, int(cfg["k"]), log_prefix=f"shadow {job}")
            per, extra = dom.score(rr, job, ids, int(cfg["k"]))
            ev = aggregate(job, int(cfg["k"]), per, extra)
            inc_C = _inc_C(rr, t)
            dS, dC = ev.S - S_inc, relative_cost_change(ev.C, inc_C)
            ok, why = (ev.S >= S_star - delta), ""
            if ok:
                ok, why = cost_rule(dS, dC, rec["nu"], delta, _Cfg(cfg))
            if ok and extra.get("harness_error_rate", 0.0) > cfg.get("max_harness_error_rate", 0.02):
                ok, why = False, "harness errors"
            full[v] = {"S": ev.S, "dS": dS, "dC": dC, "admissible": ok, "why": why,
                       "completed": True, "n_seq": rec["n"], "m_seq": rec["m"]}
        adm = {v: x for v, x in full.items() if x["admissible"]}
        full_choice = max(adm, key=lambda v: adm[v]["S"]) if adm else None
        rows.append({"t": t, "seq_choice": sel["winner"], "full_choice": full_choice,
                     "candidates": full})
        print(f"r{t}: seq {sel['winner']} full {full_choice}", flush=True)
    (Path(run) / "runs" / "verify" / D).mkdir(parents=True, exist_ok=True)
    (Path(run) / "runs" / "verify" / D / "shadow.json").write_text(json.dumps(rows, indent=1))


class _Cfg:
    def __init__(self, d):
        self.beta0, self.beta1 = d["beta0"], d["beta1"]
        self.w_s, self.w_c, self.w_n = d["w_s"], d["w_c"], d["w_n"]


def _inc_C(rr: Path, t: int):
    """Mean policy tokens of the incumbent the round started from."""
    fr = json.loads((rr / "frontier.json").read_text())
    job = fr["trajectory"][t]["job"]
    return json.loads((rr / "jobs" / job / "eval.json").read_text()).get("C")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("deploy", "shadow", "status"))
    ap.add_argument("--run", nargs="+", required=True)
    a = ap.parse_args()
    if a.cmd == "deploy":
        cmd_deploy(Path(a.run[0]))
    elif a.cmd == "shadow":
        cmd_shadow(Path(a.run[0]))
    else:
        cmd_status([Path(x) for x in a.run])


if __name__ == "__main__":
    main()
