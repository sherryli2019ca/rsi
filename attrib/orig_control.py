"""An alternative null control (post hoc, second review of paper 2): replays
that force the agent's observed action at step k, instead of letting the agent
act again at k.

R_k (attrib.groundtruth3) compares forcing the oracle's correction with null
replays that keep steps 0..k-1 and let the agent act again at k. Here every
step of a rescuable failure that has null replays also gets R_O replays with
the observed action forced at k (then live, like a corrected replay), and

  R'_k = (1/K) sum_i c_ki (p_corr_ki - p_obs_k)

replaces the null rate by their success rate. Steps whose observed action
cannot be forced (an AppWorld turn without a python block) keep the null rate.

  python -m attrib.orig_control run /home/user/attrib_runs/main [--domain D] [--workers 8]
  python -m attrib.orig_control analyze /home/user/attrib_runs/main [--out <json>]

Replays: <main>/<domain>/gt/orig/replays/<fid>_k<k>_f<j>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.analyze import _boot, _mean, load  # noqa: E402
from attrib.robustness import METHODS, LLM, kendall  # noqa: E402

DOMAINS = ("tau2_retail", "tau2_airline", "appworld")
R_O = 4
K = 4


def _force(st: dict, appworld: bool):
    if appworld:
        a = st.get("assistant") or ""
        return a if "```python" in a else None
    return st["assistant"] or None


def run(main: Path, workers: int, domains=DOMAINS) -> None:
    for d in domains:
        v = load(main / d)
        aw = d == "appworld"
        refs = []
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            f = v["fails"][fid]
            steps = {s["index"]: s for s in json.loads(Path(f["trace"]).read_text())["steps"]}
            for st in g["steps"]:
                if st["null"] is None:
                    continue
                force = _force(steps[st["k"]], aw)
                if force is None:
                    continue
                refs += [(f["harness"], {"task_id": f["task_id"], "trace": f["trace"], "start": st["k"],
                                         "force": force, "key": f"{fid}_k{st['k']}_f{j}"}) for j in range(R_O)]
        rdir = main / d / "gt" / "orig" / "replays"
        rdir.mkdir(parents=True, exist_ok=True)
        print(d, len(refs), "replays", flush=True)
        if aw:
            from attrib.aw import drive
            drive(refs, rdir, "appworld", workers)
        else:
            from attrib.groundtruth import _drive
            os.environ["TAU2_DOMAIN"] = d.split("_", 1)[1]
            _drive(refs, rdir, d.split("_", 1)[1], workers)


def _obs_rate(rdir: Path, fid: str, k: int):
    s = n = 0
    for j in range(R_O):
        p = rdir / f"{fid}_k{k}_f{j}.json"
        if not p.exists():
            continue
        x = json.loads(p.read_text())
        if not (x.get("replay") or {}).get("forced"):
            continue
        n += 1
        s += int(x.get("reward", 0) >= 1)
    return (s / n) if n else None


def analyze(main: Path) -> dict:
    rng = random.Random(0)
    rows, base_rows = {m: [] for m in METHODS}, {m: [] for m in METHODS}
    pairs, kept = [], 0
    n_new = 0
    for d in DOMAINS:
        v = load(main / d)
        rdir = main / d / "gt" / "orig" / "replays"
        for fid, g in v["gt"].items():
            if g["decisive"] is None:
                continue
            Rp = []
            for st in g["steps"]:
                ctrl = st["null"]
                if ctrl is not None:
                    o = _obs_rate(rdir, fid, st["k"])
                    if o is not None:
                        pairs.append((st["null"], o))
                        ctrl = o
                tot = 0.0
                for s in st["samples"]:
                    if s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2 \
                            and ctrl is not None:
                        tot += s["corrected"] - ctrl
                Rp.append(tot / K)
            still = max(Rp) >= 0.5
            kept += still
            n_new += 1
            cl = f"{d}/{v['fails'][fid]['task_id']}"
            for m in METHODS:
                k = v["picks"][m].get(fid)
                ok = isinstance(k, int) and 0 <= k < len(Rp)
                rows[m].append({"R": Rp[k] if ok else 0.0, "cluster": cl, "fid": fid})
                base_rows[m].append({"R": g["R"][k] if ok else 0.0, "cluster": cl, "fid": fid})
    mean = {m: round(_mean([r["R"] for r in rows[m]]), 4) for m in METHODS}
    base = {m: _mean([r["R"] for r in base_rows[m]]) for m in METHODS}
    best = max(LLM, key=lambda m: mean[m])
    fw = {r["fid"]: r["R"] for r in rows["first_write"]}
    return {"n_failures": n_new, "still_rescuable": kept, "steps_compared": len(pairs),
            "mean_null": round(_mean([a for a, _ in pairs]), 4), "mean_observed": round(_mean([b for _, b in pairs]), 4),
            "mean": mean, "kendall_tau_vs_main": round(kendall(mean, base), 3),
            "first_write_rank": 1 + sum(mean[x] > mean["first_write"] for x in METHODS if x != "first_write"),
            "best_llm": best,
            "first_write_minus_binary_search": _boot(
                [{"d": fw[r["fid"]] - r["R"], "cluster": r["cluster"]} for r in rows["binary_search_pro"]],
                lambda xs: _mean([x["d"] for x in xs]), rng),
            "first_write_minus_best_llm": _boot(
                [{"d": fw[r["fid"]] - r["R"], "cluster": r["cluster"]} for r in rows[best]],
                lambda xs: _mean([x["d"] for x in xs]), rng)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "analyze"))
    ap.add_argument("main")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--domain", choices=DOMAINS, help="run one domain only")
    args = ap.parse_args()
    if args.cmd == "run":
        run(Path(args.main), args.workers, (args.domain,) if args.domain else DOMAINS)
        return
    res = analyze(Path(args.main))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
