"""Ground truth for failure attribution, protocol 3: the rescue profile.

The pilot (20 tau2 failures, four independent repetitions of protocol 1,
2026-10-08) showed that the earliest step whose correction flips the outcome
depends on which corrections the oracle happens to propose: two repetitions
named the same earliest step in 42% of failure pairs. The size of the gain from
correcting each step was far more repeatable. Protocol 3 therefore measures,
for every agent step k of a failed episode, its rescue gain

  R_k = mean over K oracle samples of [sample calls step k a mistake] x
        (success rate of replays with that sample's correction forced at k
         - success rate of null replays from k)

i.e. the expected gain in success from letting an oracle that knows the
grading criteria correct step k (a sample that judges the step ok, or gives an
action that cannot be applied, contributes 0). Every step is scanned. Per step:
K oracle samples (attrib.groundtruth.ORACLE_SYSTEM, the same oracle as
protocols 1 and 2), R_C corrected replays per mistaken sample, R_N null
replays if any sample called the step a mistake.

Labels derived from the profile: the decisive step is the step with the
largest R_k if that is at least THRESHOLD (ties: earliest), else none; the
earliest step with R_k >= THRESHOLD is reported too.

Cost: the K samples of a step share one prompt, and a failure's steps share
their prefix, so most oracle input is read from the endpoint's cache; samples
run one after another per step for that reason.

  python -m attrib.groundtruth3 run --domain tau2_retail --failures <failures.json> \
      --out <dir> --rep a [--only <fids file>] [--workers 8]

Replay keys: <fid>_k<k>_o<i>_<j> (sample i corrected), <fid>_k<k>_n<j> (null).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import (ORACLE_MODEL, ORACLE_SYSTEM, _context, _drive,  # noqa: E402
                                _render_step, to_blocks)

K = 4            # oracle samples per step
R_C = 2          # corrected replays per mistaken sample
R_N = 4          # null replays per step
THRESHOLD = 0.5


def oracle_prompt(rec: dict, k: int) -> str:
    """Protocol 1's oracle prompt with the step number moved to the end, so the
    prompts of one failure share their prefix."""
    gold = "\n".join(f"- {a['name']}({json.dumps(a['arguments'], ensure_ascii=False)})"
                     for a in rec.get("gold_actions") or []) or "(none: the database must stay unchanged)"
    info = "\n".join(f"- {x}" for x in rec.get("communicate_info") or []) or "(none)"
    conv = [f"user (opening): {rec['opening']}"]
    for st in rec["steps"][:k]:
        conv.append(f"step {st['index']}:\n{_render_step(st)}")
    act = "\n".join(f"  agent says: {b['text']}" if b["type"] == "text" else
                    f"  agent calls {b['name']}({json.dumps(b['input'], ensure_ascii=False)})"
                    for b in rec["steps"][k]["assistant"])
    return (f"=== GRADING CRITERIA ===\nGold actions:\n{gold}\nInformation the agent must say:\n{info}\n"
            f"Outcome of this episode: database matches gold = {rec.get('db_match')}, "
            f"missing information = {rec.get('missing_info')}\n\n"
            f"=== CONVERSATION ===\n" + "\n".join(conv) +
            f"\n\n=== AGENT ACTION AT STEP {k} (judge this step; the conversation above is "
            f"everything before it) ===\n{act}\n")


def run_oracle(fails: list, odir: Path, m, workers: int) -> None:
    from rrsi.llm import generate
    ctx = _context(m)
    names = {t["name"] for t in m.tool_defs(m.BASE_COMPONENTS)}
    jobs = []
    for f in fails:
        rec = json.loads(Path(f["trace"]).read_text())
        for k in range(rec["n_steps"]):
            if any(not (odir / f"{f['fid']}_k{k}_o{i}.json").exists() for i in range(K)):
                jobs.append((f, rec, k))
    jobs.sort(key=lambda j: j[2])          # earlier steps first: their prefixes are cached for later ones

    def one(job):
        f, rec, k = job
        prompt = oracle_prompt(rec, k)
        for i in range(K):
            p = odir / f"{f['fid']}_k{k}_o{i}.json"
            if p.exists():
                continue
            raw = generate(prompt, system=ORACLE_SYSTEM, json_only=True, model=ORACLE_MODEL,
                           cache_prefix=ctx, role="oracle")
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


def _samples(f: dict, k: int, odir: Path) -> list[dict]:
    return [json.loads((odir / f"{f['fid']}_k{k}_o{i}.json").read_text()) for i in range(K)]


def run_replays(fails: list, odir: Path, rdir: Path, dom: str, workers: int) -> None:
    refs = []
    for f in fails:
        base = {"task_id": f["task_id"], "trace": f["trace"]}
        for k in range(f["n_steps"]):
            sm = _samples(f, k, odir)
            mistaken = [i for i, v in enumerate(sm) if v.get("verdict") == "mistake"]
            for i in mistaken:
                refs += [(f["harness"], {**base, "start": k, "force": sm[i]["force"],
                                         "key": f"{f['fid']}_k{k}_o{i}_{j}"}) for j in range(R_C)]
            if mistaken:
                refs += [(f["harness"], {**base, "start": k, "key": f"{f['fid']}_k{k}_n{j}"})
                         for j in range(R_N)]
    for _ in range(3):                    # a driver stopped part-way is resumed
        _drive(refs, rdir, dom, workers)
        if all((rdir / f"{r['key']}.json").exists() for _, r in refs):
            return
    missing = sum(not (rdir / f"{r['key']}.json").exists() for _, r in refs)
    raise SystemExit(f"{missing} replays still missing; rerun to resume")


def _rate(rdir: Path, keys: list[str], forced: bool):
    s = n = 0
    for key in keys:
        p = rdir / f"{key}.json"
        if not p.exists():
            continue
        x = json.loads(p.read_text())
        if forced and not (x.get("replay") or {}).get("forced"):
            continue
        n += 1
        s += int(x.get("reward", 0) >= 1)
    return (s / n if n else None), n


def profile(f: dict, odir: Path, rdir: Path) -> dict:
    steps = []
    for k in range(f["n_steps"]):
        sm = _samples(f, k, odir)
        null, nn = _rate(rdir, [f"{f['fid']}_k{k}_n{j}" for j in range(R_N)], False)
        gains, rows = [], []
        for i, v in enumerate(sm):
            if v.get("verdict") != "mistake":
                gains.append(0.0)
                rows.append({"verdict": v.get("verdict")})
                continue
            c, nc = _rate(rdir, [f"{f['fid']}_k{k}_o{i}_{j}" for j in range(R_C)], True)
            # a correction the replay could not apply, or no null replay: no measured gain
            g = (c - null) if c is not None and null is not None and nc == R_C else 0.0
            gains.append(g)
            rows.append({"verdict": "mistake", "corrected": c, "n": nc})
        steps.append({"k": k, "R": round(sum(gains) / K, 4), "null": null, "n_null": nn, "samples": rows})
    Rs = [s["R"] for s in steps]
    best = max(Rs) if Rs else 0.0
    return {**{x: f[x] for x in ("fid", "commit", "task_id", "trial", "n_steps") if x in f},
            "R": Rs, "max_R": best,
            "decisive": Rs.index(best) if best >= THRESHOLD else None,
            "earliest": next((k for k, r in enumerate(Rs) if r >= THRESHOLD), None),
            "steps": steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline"))
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", default="a")
    ap.add_argument("--only", default=None, help="a JSON list of fids to restrict to (the retest)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dom = args.domain.split("_", 1)[1]
    os.environ["TAU2_DOMAIN"] = dom
    from agent_exp import tau2_env as m
    fails = json.loads(Path(args.failures).read_text())
    if args.only:
        keep = set(json.loads(Path(args.only).read_text()))
        fails = [f for f in fails if f["fid"] in keep]
    rep = Path(args.out) / args.rep
    rep.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    odir, rdir = rep / "oracle", rep / "replays"
    run_oracle(fails, odir, m, args.workers)
    run_replays(fails, odir, rdir, dom, args.workers)
    res = {"protocol": 3, "K": K, "R_C": R_C, "R_N": R_N, "threshold": THRESHOLD,
           "failures": [profile(f, odir, rdir) for f in fails]}
    (rep / "result.json").write_text(json.dumps(res, indent=1))
    for x in res["failures"]:
        print(f"{x['fid']:30s} max R {x['max_R']:.2f} decisive {x['decisive']} earliest {x['earliest']}")


if __name__ == "__main__":
    main()
