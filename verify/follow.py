"""Follow an RRSI run and collect the E1 evidence for each settled round:
targeted replays (candidates + null), the LLM judge, and held-out deployment
of the round's incumbent and measured candidates. When the run has finished
all T rounds, the base and final incumbents get the larger endpoint k.

  python -m verify.follow --domain tau2_retail [--runs runs/rrsi] [--out runs/verify] [--once]

Resume-safe: a round is marked done in <out>/<domain>/done_r<t> only after all
of its pieces exist. Stops when <out>/<domain>/STOP exists.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from rrsi.domain import load_domain

from . import deploy, judge, replays
from .state import ROOT, rounds


def log(name, msg):
    print(f"[verify:{name}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def process(dom, run_dir: Path, out: Path, rnd, k_heldout: int, model) -> None:
    replays.run(dom, ROOT, run_dir, out, rnd)
    j = judge.run(dom, run_dir, out, rnd, model=model)
    ev = deploy.run(dom, ROOT, out, rnd.inc_commit, k_heldout)
    msg = [f"inc {rnd.inc_commit[:7]} heldout S={ev.S:.3f}"]
    for c in rnd.cands:
        if c.measured:
            e = deploy.run(dom, ROOT, out, c.commit, k_heldout)
            msg.append(f"{c.variant} {c.commit[:7]} dS_evolve={c.decision.get('delta_S')} "
                       f"judge={j.get(c.variant, {}).get('dS')} heldout S={e.S:.3f}")
    log(dom.name, f"r{rnd.t} done: " + "; ".join(msg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--runs", default=str(ROOT / "runs" / "rrsi"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "verify"))
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--k-heldout", type=int, default=None, help="override heldout_k (smoke runs)")
    args = ap.parse_args()
    dom = load_domain(args.domain)
    cfg = dom.cfg
    run_dir = Path(args.runs) / dom.name
    out = Path(args.out) / dom.name
    out.mkdir(parents=True, exist_ok=True)
    k_h, k_end = int(cfg.get("heldout_k", 4)), int(cfg.get("heldout_k_endpoints", 10))
    if args.k_heldout:
        k_h = args.k_heldout
    T = int(cfg.get("T", 20))
    model = cfg.get("judge_model") or cfg.get("analyst_model")
    while True:
        if (out / "STOP").exists():
            log(dom.name, "STOP present; exiting")
            return
        try:
            rs = rounds(run_dir) if (run_dir / "frontier.json").exists() else []
            for rnd in rs:
                if (out / f"done_r{rnd.t}").exists():
                    continue
                log(dom.name, f"r{rnd.t}: replays, judge, held-out")
                process(dom, run_dir, out, rnd, k_h, model)
                (out / f"done_r{rnd.t}").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            fr = json.loads((run_dir / "frontier.json").read_text()) if rs else {}
            traj = fr.get("trajectory") or []
            if len(traj) >= T + 1 and all((out / f"done_r{t}").exists() for t in range(T)):
                for commit in {traj[0]["commit"], traj[-1]["commit"]}:
                    ev = deploy.run(dom, ROOT, out, commit, k_end)
                    log(dom.name, f"endpoint {commit[:7]} heldout S={ev.S:.3f} (k={k_end})")
                (out / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
                log(dom.name, "all rounds and endpoints done")
                return
        except Exception:  # noqa: BLE001 - keep following; the error is logged
            log(dom.name, "error:\n" + traceback.format_exc()[-2000:])
        if args.once:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
