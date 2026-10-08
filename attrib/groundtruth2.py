"""Ground truth for failure attribution, protocol 2: several oracle samples per
step and a confirmation stage (attrib/groundtruth.py is protocol 1).

The pilot of protocol 1 (one oracle sample per step, 2026-10-08) agreed with
itself on the decisive step in 12 of 20 failures, and most disagreements came
from the oracle (a step judged ok in one repetition and mistaken in the other,
or a correction that worked in one repetition and not in the other), not from
the replays. Protocol 2 searches more and guards the extra search:

  1. K oracle samples per step (same prompt as protocol 1); the step is tested
     if any sample calls it a mistake, with each distinct correction.
  2. Discovery: N_REPLAYS corrected replays per correction; the best correction
     (most successes, earliest sample on ties) goes on only if it reaches FLIP
     successes; then N_REPLAYS null replays; the step passes discovery if best
     minus null >= FLIP.
  3. Confirmation: N_REPLAYS fresh corrected replays of the best correction and
     N_REPLAYS fresh null replays; the step flips if the fresh corrected
     successes exceed the fresh null successes by at least CONFIRM.

Steps are scanned in order and a failure stops at its first flip (curtailed),
except a seeded share scanned fully. Replay keys: <fid>_k<k>_c<i>_<j> (correction
i), _n<j> (null), _v<j> (confirmation, corrected), _m<j> (confirmation, null).

  python -m attrib.groundtruth2 run --domain tau2_retail --runs <rrsi runs> \
      --wt <dir of harness worktrees by commit> --out <dir> --rep a \
      [--failures <failures.json>] [--samples 2] [--full-frac 0.2]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import (FLIP, N_REPLAYS, ORACLE_MODEL, ORACLE_SYSTEM,  # noqa: E402
                                _context, _drive, oracle_prompt, select, to_blocks)

CONFIRM = 1


# ------------------------------------------------------------------ oracle --
def run_oracle(fails: list, odir: Path, m, samples: int, workers: int) -> None:
    from rrsi.llm import generate
    ctx = _context(m)
    names = {t["name"] for t in m.tool_defs(m.BASE_COMPONENTS)}
    jobs = []
    for f in fails:
        rec = json.loads(Path(f["trace"]).read_text())
        for k in range(rec["n_steps"]):
            for i in range(samples):
                p = odir / f"{f['fid']}_k{k}_o{i}.json"
                if not p.exists():
                    jobs.append((p, rec, k))

    def one(job):
        p, rec, k = job
        raw = generate(oracle_prompt(rec, k), system=ORACLE_SYSTEM, json_only=True,
                       model=ORACLE_MODEL, cache_prefix=ctx, role="oracle")
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            v = {"verdict": "unparseable", "raw": raw[:2000]}
        if not isinstance(v, dict):
            v = {"verdict": "unparseable", "raw": str(v)[:2000]}
        if v.get("verdict") == "mistake":
            v["force"] = to_blocks(v.get("action"), names)
            if v["force"] is None:
                v["verdict"] = "invalid_action"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(v, ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


def corrections(f: dict, k: int, odir: Path, samples: int) -> tuple[list, list]:
    """(verdicts of the samples, distinct corrections in sample order)."""
    verdicts, out, seen = [], [], set()
    for i in range(samples):
        v = json.loads((odir / f"{f['fid']}_k{k}_o{i}.json").read_text())
        verdicts.append(v.get("verdict"))
        if v.get("verdict") == "mistake":
            key = json.dumps(v["force"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(v["force"])
    return verdicts, out


# ----------------------------------------------------------------- replays --
def _ref(f: dict, k: int, key: str, force=None) -> dict:
    r = {"task_id": f["task_id"], "trace": f["trace"], "start": k, "key": f"{f['fid']}_k{k}_{key}"}
    if force is not None:
        r["force"] = force
    return r


def _count(rdir: Path, f: dict, k: int, prefix: str, forced: bool):
    """(successes, valid, missing) over the N_REPLAYS replays <fid>_k<k>_<prefix><j>."""
    s = n = miss = 0
    for j in range(N_REPLAYS):
        p = rdir / f"{f['fid']}_k{k}_{prefix}{j}.json"
        if not p.exists():
            miss += 1
            continue
        x = json.loads(p.read_text())
        if forced and not (x.get("replay") or {}).get("forced"):
            continue
        n += 1
        s += int(x.get("reward", 0) >= 1)
    return s, n, miss


def step_state(f: dict, k: int, odir: Path, rdir: Path, samples: int) -> dict:
    """What is known about step k, and the replays still needed to settle it."""
    verdicts, corrs = corrections(f, k, odir, samples)
    st = {"k": k, "verdicts": verdicts, "n_corrections": len(corrs)}
    if not corrs:
        return {**st, "outcome": "ok", "need": []}
    need, succ = [], []
    for i, force in enumerate(corrs):
        s, n, miss = _count(rdir, f, k, f"c{i}_", True)
        if miss:
            need += [_ref(f, k, f"c{i}_{j}", force) for j in range(N_REPLAYS)]
        succ.append(s if n == N_REPLAYS else -1)   # a correction not applied every time is not tested
    st["corrected"] = succ
    if need:
        return {**st, "outcome": None, "need": need}
    best = max(range(len(corrs)), key=lambda i: (succ[i], -i))
    st["best"] = best
    if succ[best] < FLIP:
        return {**st, "outcome": "no_flip", "need": []}
    z, _, miss = _count(rdir, f, k, "n", False)
    if miss:
        return {**st, "outcome": None, "need": [_ref(f, k, f"n{j}") for j in range(N_REPLAYS)]}
    st["null"] = z
    if succ[best] - z < FLIP:
        return {**st, "outcome": "no_flip", "need": []}
    vc, vn, vmiss = _count(rdir, f, k, "v", True)
    mz, _, mmiss = _count(rdir, f, k, "m", False)
    if vmiss or mmiss:
        return {**st, "outcome": None,
                "need": [_ref(f, k, f"v{j}", corrs[best]) for j in range(N_REPLAYS)]
                + [_ref(f, k, f"m{j}") for j in range(N_REPLAYS)]}
    st.update({"confirm_corrected": vc if vn == N_REPLAYS else None, "confirm_null": mz})
    ok = vn == N_REPLAYS and vc - mz >= CONFIRM
    return {**st, "outcome": "flip" if ok else "unconfirmed", "need": []}


def _n_steps(f: dict) -> int:
    return json.loads(Path(f["trace"]).read_text())["n_steps"]


def run_replays(fails: list, odir: Path, rdir: Path, dom: str, samples: int, workers: int,
                full: set) -> None:
    """In waves across failures: each failure contributes the replays its first
    unsettled step needs (every unsettled step, for a fully scanned failure)."""
    longest = max((_n_steps(f) for f in fails), default=0)
    for _ in range(4 * longest + 4):
        wave = []
        for f in fails:
            for k in range(_n_steps(f)):
                st = step_state(f, k, odir, rdir, samples)
                if st["need"]:
                    wave += [(f["harness"], r) for r in st["need"]]
                    if f["fid"] not in full:
                        break
                elif st["outcome"] == "flip" and f["fid"] not in full:
                    break
        if not wave:
            return
        before = sum(1 for _ in rdir.glob("*.json")) if rdir.exists() else 0
        _drive(wave, rdir, dom, workers)
        if sum(1 for _ in rdir.glob("*.json")) == before:
            raise SystemExit("a replay wave wrote nothing; see the driver's output")


def summarize(fails: list, odir: Path, rdir: Path, samples: int, full: set) -> dict:
    out = []
    for f in fails:
        steps, stop = [], False
        for k in range(_n_steps(f)):
            if stop:
                steps.append({"k": k, "outcome": "not_scanned"})
                continue
            st = step_state(f, k, odir, rdir, samples)
            st.pop("need")
            steps.append(st)
            stop = st["outcome"] == "flip" and f["fid"] not in full
        flips = [s["k"] for s in steps if s["outcome"] == "flip"]
        out.append({**{k: f[k] for k in ("fid", "commit", "task_id", "trial", "n_steps")},
                    "scan": "full" if f["fid"] in full else "curtailed",
                    "decisive": flips[0] if flips else None, "flips": flips,
                    "discovered": [s["k"] for s in steps if s["outcome"] in ("flip", "unconfirmed")],
                    "steps": steps})
    return {"protocol": 2, "samples": samples, "n_replays": N_REPLAYS, "flip": FLIP,
            "confirm": CONFIRM, "failures": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline"))
    ap.add_argument("--runs", required=True)
    ap.add_argument("--wt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--failures", default=None, help="a failures.json to use instead of selecting")
    ap.add_argument("--n-fail", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rep", default="a")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--full-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dom = args.domain.split("_", 1)[1]
    os.environ["TAU2_DOMAIN"] = dom
    from agent_exp import tau2_env as m
    base = Path(args.out) / args.domain
    base.mkdir(parents=True, exist_ok=True)
    fp = base / "failures.json"
    if not fp.exists():
        fails = (json.loads(Path(args.failures).read_text()) if args.failures else
                 select(args.domain, Path(args.runs), Path(args.wt), args.n_fail, args.seed))
        fp.write_text(json.dumps(fails, indent=1))
    fails = json.loads(fp.read_text())
    rep = base / args.rep
    rep.mkdir(parents=True, exist_ok=True)
    odir, rdir = rep / "oracle", rep / "replays"
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    fids = sorted(f["fid"] for f in fails)
    full = set(random.Random(f"full:{args.seed}").sample(fids, round(args.full_frac * len(fids))))
    run_oracle(fails, odir, m, args.samples, args.workers)
    run_replays(fails, odir, rdir, dom, args.samples, args.workers, full)
    res = summarize(fails, odir, rdir, args.samples, full)
    (rep / "result.json").write_text(json.dumps(res, indent=1))
    for x in res["failures"]:
        print(f"{x['fid']:28s} steps {x['n_steps']:2d} decisive {x['decisive']} flips {x['flips']} "
              f"discovered {x['discovered']} ({x['scan']})")


if __name__ == "__main__":
    main()
