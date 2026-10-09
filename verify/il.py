"""Experiment IL (verify/PREREGISTRATION_IL.md): independent RRSI loops on tau2
airline driven by sequential full evaluation or by RRSI's own full evaluation.

  python -m verify.il deploy  --run DIR      held-out deployments of the base and the
                                             incumbent after round 9 (resume-safe)
  python -m verify.il status  --run DIR ...  progress and spend (no held-out S)
  python -m verify.il analyze [--json results/il/analysis.json]
                                             the registered comparison over all loops
  python -m verify.il tables  [--json ...] [--tex paper/tables/il.tex]

DIR is a loop's checkout (its runs/ holds rrsi/ and verify/).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from rrsi.domain import load_domain

from . import cl, deploy
from .state import worktree

D = cl.D
T_LAST = cl.T_LAST
# registered arms: the new loops, and the earlier loops of the same design
SEQ = {"cl1": "/home/user/cl1", "cl2": "/home/user/cl2",
       **{f"s{i}": f"/home/user/il_s{i}" for i in range(1, 5)}}
FULL = {"r2": "/home/user/e1r2", "r3": "/home/user/e1r3",
        **{f"f{i}": f"/home/user/il_f{i}" for i in range(1, 5)}}
NEW = {"s1", "s2", "s3", "s4", "f1", "f2", "f3", "f4"}
R1 = {"r1": "/home/user/e1"}          # sensitivity only (component labels not corrected)
MARGIN = 0.05                          # non-inferiority margin on transfer, reported descriptively


def _endpoints(run: Path) -> list[str]:
    tr = cl._traj(Path(run))
    return list(dict.fromkeys([tr["base"], tr["final"]]))


def cmd_deploy(run: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    dom = load_domain(D)
    out = Path(run) / "runs" / "verify" / D
    out.mkdir(parents=True, exist_ok=True)
    cs = _endpoints(run)
    for c in cs:
        worktree(Path(run), out / "wt", c)

    def one(c):
        deploy.run(dom, Path(run), out, c, cl.K_HELDOUT)
        print(f"deployed {c[:12]} (k={cl.K_HELDOUT})", flush=True)     # S deliberately not printed
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(one, cs))
    (out / "deploy.done").write_text("\n".join(cs) + "\n")


def cmd_status(runs: list[Path]) -> None:
    from .posthoc_e1 import PRICES, _episode_dollars
    tot = 0.0
    for run in runs:
        rr = cl._rrsi(run)
        fr = json.loads((rr / "frontier.json").read_text()) if (rr / "frontier.json").exists() else None
        settled = len(fr["trajectory"]) - 1 if fr else 0
        usd = 0.0
        usage = Path(run) / "runs" / "rrsi" / f"{D}.usage.jsonl"
        if usage.exists():
            for line in usage.read_text().splitlines():
                if line.strip():
                    x = json.loads(line)
                    pr = PRICES.get(x.get("model"), PRICES["deepseek-v4-pro"])
                    usd += (x.get("in", 0) * pr[0] + x.get("cache_read", 0) * pr[1] + x.get("out", 0) * pr[2]) / 1e6
        eps = list((rr / "jobs").rglob("s*/*.json")) if (rr / "jobs").exists() else []
        vdir = Path(run) / "runs" / "verify" / D / "jobs" / "heldout"
        ho = list(vdir.rglob("s*/*.json")) if vdir.exists() else []
        ep_usd = sum(_episode_dollars(json.loads(p.read_text()).get("tokens") or {}) for p in eps + ho)
        tot += usd + ep_usd
        acc = len(set(x["commit"] for x in fr["trajectory"])) - 1 if fr else 0
        print(f"{Path(run).name}: settled {settled}/10, accepted {acc}, evolve episodes {len(eps)}, "
              f"held-out episodes {len(ho)}, ${usd + ep_usd:.2f}")
    print(f"total ${tot:.2f}")


def _transfer(run: Path, tasks=None) -> dict:
    tr = cl._traj(Path(run))
    a, b = cl._ho(Path(run), tr["final"]), cl._ho(Path(run), tr["base"])
    return a, b


def cmd_analyze(js: Path, B: int = 2000, seed: int = 9) -> dict:
    arms = {"seq": {k: Path(v) for k, v in SEQ.items()}, "full": {k: Path(v) for k, v in FULL.items()}}
    allr = {**arms["seq"], **arms["full"], **{k: Path(v) for k, v in R1.items()}}
    ho = {n: _transfer(r) for n, r in allr.items()}
    tasks = sorted(ho["r2"][1])
    tf = lambda n, ts: float(np.mean([ho[n][0][t] - ho[n][1][t] for t in ts]))
    cost = {n: cl._loop_cost(r) for n, r in allr.items()}
    point = {n: tf(n, tasks) for n in allr}
    rng = np.random.default_rng(seed)

    def compare(seq_names, full_names, key):
        """seq - full on transfer (key='transfer') or loop dollars ('usd') or
        candidate-evaluation episodes ('episodes'); bootstrap over loops within
        arm and, for transfer, held-out tasks."""
        val = lambda n, ts: (tf(n, ts) if key == "transfer" else
                             cost[n]["total"] if key == "usd" else cost[n]["eval_episodes"])
        pt = np.mean([val(n, tasks) for n in seq_names]) - np.mean([val(n, tasks) for n in full_names])
        bs = []
        for _ in range(B):
            ts = list(rng.choice(tasks, len(tasks))) if key == "transfer" else tasks
            s = rng.choice(seq_names, len(seq_names))
            f = rng.choice(full_names, len(full_names))
            bs.append(np.mean([val(n, ts) for n in s]) - np.mean([val(n, ts) for n in f]))
        bs = np.array(bs)
        xs = [val(n, tasks) for n in seq_names]
        ys = [val(n, tasks) for n in full_names]
        se = math.sqrt(np.var(xs, ddof=1) / len(xs) + np.var(ys, ddof=1) / len(ys))
        out = {"seq": len(seq_names), "full": len(full_names), "diff": float(pt),
               "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
               "welch_se": se, "seq_mean": float(np.mean(xs)), "full_mean": float(np.mean(ys)),
               "seq_sd": float(np.std(xs, ddof=1)), "full_sd": float(np.std(ys, ddof=1))}
        if key == "transfer":
            out["p_ni_margin5"] = float(np.mean(bs <= -MARGIN))
        return out

    S, F = list(arms["seq"]), list(arms["full"])
    Sn, Fn = [n for n in S if n in NEW], [n for n in F if n in NEW]
    res = {"transfer": {n: point[n] for n in allr},
           "cost": cost,
           "primary": {k: compare(S, F, k) for k in ("transfer", "usd", "episodes")},
           "new_only": {k: compare(Sn, Fn, k) for k in ("transfer", "usd", "episodes")},
           "with_r1": {k: compare(S, F + ["r1"], k) for k in ("transfer", "usd", "episodes")}}
    acc = {}
    for n, r in allr.items():
        tr = cl._traj(r)
        acc[n] = len(dict.fromkeys(tr["commits"])) - 1
    res["accepted_changes"] = acc
    shadow = {}
    for n in S:
        f = arms["seq"][n] / "runs" / "verify" / D / "shadow.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())
        drops = [c for r in rows for c in r["candidates"].values() if c.get("completed")]
        accs = [r for r in rows if r["full_choice"] is not None]
        shadow[n] = {"rounds": len(rows), "agree": sum(r["seq_choice"] == r["full_choice"] for r in rows),
                     "full_accepts": len(accs),
                     "seq_made": sum(r["seq_choice"] == r["full_choice"] for r in accs),
                     "dropped": len(drops), "dropped_admissible": sum(c["admissible"] for c in drops),
                     "dropped_selected": sum(1 for r in rows for v, c in r["candidates"].items()
                                             if c.get("completed") and r["full_choice"] == v)}
    res["shadow"] = shadow
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps(res, indent=1))
    pp = lambda x: f"{100 * x:+.2f}"
    for n in allr:
        print(f"{n}: transfer@10 {pp(point[n])}  accepted {acc[n]}  loop ${cost[n]['total']:.2f}  "
              f"eval episodes {cost[n]['eval_episodes']}")
    for part in ("primary", "new_only", "with_r1"):
        x = res[part]["transfer"]
        print(f"{part}: transfer seq-full {pp(x['diff'])} [{pp(x['ci90'][0])}, {pp(x['ci90'][1])}] "
              f"p(NI 5pt) {x.get('p_ni_margin5'):.3f}; usd {res[part]['usd']['diff']:+.2f} "
              f"[{res[part]['usd']['ci90'][0]:+.2f}, {res[part]['usd']['ci90'][1]:+.2f}]; episodes "
              f"{res[part]['episodes']['diff']:+.0f}")
    print(json.dumps(shadow, indent=1))
    return res


def cmd_tables(js: Path, out: Path) -> None:
    res = json.loads(Path(js).read_text())
    pp = lambda x: f"{100 * x:+.2f}"
    rows = []
    for arm, names in (("full", FULL), ("seq.", SEQ)):
        for n in names:
            c = res["cost"][n]
            rows.append(f"{n} & {arm} & {res['accepted_changes'][n]} & {c['eval_episodes']} & "
                        f"\\${c['total']:.2f} & ${pp(res['transfer'][n])}$ \\\\")
        rows.append("\\midrule")
    p = res["primary"]
    rows.append(f"\\multicolumn{{3}}{{l}}{{Seq.$-$full (6 vs 6)}} & ${p['episodes']['diff']:+.0f}$ & "
                f"${p['usd']['diff']:+.2f}$ & ${pp(p['transfer']['diff'])}$ \\\\")
    rows.append(f"\\multicolumn{{3}}{{l}}{{\\quad 90\\% interval}} & "
                f"$[{p['episodes']['ci90'][0]:+.0f},{p['episodes']['ci90'][1]:+.0f}]$ & "
                f"$[{p['usd']['ci90'][0]:+.2f},{p['usd']['ci90'][1]:+.2f}]$ & "
                f"$[{pp(p['transfer']['ci90'][0])},{pp(p['transfer']['ci90'][1])}]$ \\\\")
    tex = ("\\begin{tabular}{llrrrr}\n\\toprule\n"
           "Loop & Selection & Accepted & Cand.\\ episodes & Loop \\$ & Transfer \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(tex)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("deploy", "status", "analyze", "tables"))
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--json", default="results/il/analysis.json")
    ap.add_argument("--tex", default="paper/tables/il.tex")
    a = ap.parse_args()
    if a.cmd == "deploy":
        cmd_deploy(Path(a.run[0]))
    elif a.cmd == "status":
        cmd_status([Path(x) for x in a.run])
    elif a.cmd == "analyze":
        cmd_analyze(Path(a.json))
    else:
        cmd_tables(Path(a.json), Path(a.tex))


if __name__ == "__main__":
    main()
