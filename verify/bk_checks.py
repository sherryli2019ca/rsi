"""Zero-cost checks of experiment BK's coupled design (verify/PREREGISTRATION_BK.md).

  python -m verify.bk_checks offline DIR   group rounds recomputed on the finished IL full
                                           loops (no episode, no model call; scratch files
                                           in DIR): arm f's round equals the recorded IL
                                           round, and where a and s would fork
  python -m verify.bk_checks power         coverage and width of the lineage model's 90%
                                           intervals (verify/bk.py _lineage_se) under a
                                           simulated null
"""
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import t as tdist

from rrsi.config import RRSIConfig
from rrsi.domain import load_domain

from . import bk, cl
from .live import group_outcomes

IL_FULL = ("il_f1", "il_f2", "il_f3", "il_f4")


def offline(scratch: Path) -> None:
    dom = load_domain(bk.D)

    def no_run(*a, **k):
        raise RuntimeError("must not run episodes")
    dom.run = no_run
    cfg = RRSIConfig.load(dom.root / "rrsi.json")
    tot = same = 0
    split = {"a": 0, "s": 0}
    parts_seen = {}
    for n in IL_FULL:
        run = bk.HOME / n
        rr = cl._rrsi(run)
        hist = [json.loads(l) for l in (rr / "history.jsonl").read_text().splitlines() if l.strip()]
        attr = [json.loads(l) for l in (rr / "attribution.jsonl").read_text().splitlines() if l.strip()]
        fake = SimpleNamespace(domain=dom, cfg=cfg, runs=rr, wt_root=rr / "wt")
        rep = {r["t"]: r for r in json.loads((run / "runs" / "verify" / bk.D / "replay.json").read_text())}
        for t in range(bk.T_LAST + 1):
            inc_ev, S_star, delta, cands, counts = bk._state(run, t, dom, cfg)
            cs = [c for c, _ in cands]
            rdir = scratch / n / f"r{t}"
            shutil.rmtree(rdir, ignore_errors=True)
            out, parts, rec, _ = group_outcomes(fake, t, cs, len(cs), inc_ev, S_star, delta, counts,
                                                list(bk.ORDER), rdir)
            got = []
            for c in cs:
                r = next(x for x in hist if x["t"] == t and x["variant"] == c.variant
                         and x.get("edit_id") is not None)
                att = any(a["t"] == t and a["variant"] == c.variant for a in attr)
                got.append((c.variant, r["outcome"], r["delta_S"], r["delta_C"], r["S"], r["C"],
                            r["detail"], att))
            acc = [x for x in hist if x["t"] == t and x.get("accepted") and x.get("edit_id") is not None]
            key = out["f"]["key"]
            tot += 1
            if [x for x in key[1] if len(x) > 2] == got and key[0] == (acc[0]["variant"] if acc else None):
                same += 1
            else:
                print(f"DIFFERENT {n} r{t}")
            split["a"] += "a" not in next(p for p in parts if "f" in p)
            split["s"] += "s" not in next(p for p in parts if "h" in p)
            assert (rec["full_choice"], rec["seqfull_choice"], rec["seqcost_choice"]) == \
                (rep[t]["full_choice"], rep[t]["seq_choice"], rep[t]["seqcost_choice"]), (n, t)
            parts_seen.setdefault(str(parts), []).append(f"{n[-2:]} r{t}")
    print(f"arm f's group round equals the recorded IL round: {same}/{tot}")
    print(f"rounds in which a's round differs from f's: {split['a']}; s's from h's: {split['s']}")
    for k, v in parts_seen.items():
        print(f"groups {k}: {len(v)} rounds ({', '.join(v[:4])}{', ...' if len(v) > 4 else ''})")


def _simulate(nb: int, sd: float = 3.5, p_fork: float = 0.6, R: int = 4000, seed: int = 1):
    """Null data in which a coupled arm keeps a loop's marginal variance: it
    equals its partner unless it forks, and then shares a fraction r of the
    loop (uniform on 0 to 0.9) with it."""
    bk.BLOCKS = tuple(f"b{i}" for i in range(nb))
    rng = np.random.default_rng(seed)
    cover = {k: 0 for k in bk.CONTRASTS}
    width = {k: [] for k in bk.CONTRASTS}
    for _ in range(R):
        Y = {}
        for b in bk.BLOCKS:
            mu = rng.normal(0, 3)
            e = {a: rng.normal(0, sd) for a in bk.IND}
            for a, p in bk.PAIRED.items():
                if rng.random() < p_fork:
                    r = rng.uniform(0, 0.9)
                    e[a] = r * e[p] + math.sqrt(1 - r * r) * rng.normal(0, sd)
                else:
                    e[a] = e[p]
            for a in bk.ARMS:
                Y[(b, a)] = mu + e[a]
        means = {a: float(np.mean([Y[(b, a)] for b in bk.BLOCKS])) for a in bk.ARMS}
        for k, c in bk.CONTRASTS.items():
            est = bk._contrast(means, c)
            se, df = bk._lineage_se(Y, c)
            q = tdist.ppf(0.95, df) if math.isfinite(df) else 1.645
            cover[k] += abs(est) <= q * se
            width[k].append(2 * q * se)
    return {k: (cover[k] / R, float(np.median(width[k]))) for k in bk.CONTRASTS}


def power() -> None:
    sd = 3.5
    for nb in (4, 5):
        print(f"coupled design, {nb} blocks (residual SD {sd}, fork in 60% of blocks):")
        for k, (cov, w) in _simulate(nb, sd).items():
            print(f"  {k:16s} coverage {cov:.3f}, median 90% width {w:.1f}")
    q = tdist.ppf(0.95, 12)
    print(f"independent design, 4 blocks (df 12): difference of two arms {2 * q * sd * math.sqrt(2 / 4):.1f}, "
          f"main effect {2 * q * sd * math.sqrt(1 / 4):.1f}, interaction {2 * q * sd:.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "offline":
        offline(Path(sys.argv[2]))
    else:
        power()
