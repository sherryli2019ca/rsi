"""Post hoc runs for the fourth review of paper 2 (not registered).

  python -m attrib.review4_runs blind --domain tau2_retail|tau2_airline|appworld \
      [--main /home/user/attrib_runs/main] [--workers 8] [--replay-workers N] [--limit N]
  python -m attrib.review4_runs fresh --domain tau2_retail|tau2_airline [--workers 8] [--limit N]

blind  A grading-blind oracle. The registered oracle reads the task's grading
       (tau2: gold actions, required information and the outcome; AppWorld: the
       failed tests with expected values) and is told to use it only to know
       the right outcome. The blind oracle gets the same instructions and
       context (policy and tools, or the agent's rules and APIs) and the same
       episode up to step k, but no grading: only that the episode failed,
       as counterfactual search and the blind judges see it. Run at every
       step named on the 157 rescuable failures by any method of the main
       table and its post hoc variants, and at the best rescue step and the
       earliest step with a gain of at least 0.5: K = 4 samples, 2 corrected
       replays per sample that calls the step a mistake, and the registered
       profile's 4 null replays (linked; new ones where the registered oracle
       never corrected the step). Output <main>/<domain>/gt/blind/.
fresh  Fresh, independent replays for the repair experiment's pointers
       (attrib.phaseb): at each of the 81 failing traces' oracle step (the
       step of largest gain, the pointer of the oracle groups) and at the
       steps the first write, binary search, counterfactual search and RRSI's
       digests pointed to, K = 4 new samples of the registered oracle, 2
       corrected replays each, 4 new null replays (8 at oracle steps); and at
       each oracle step the exact correction the oracle-fix group was given
       (attrib.phaseb.oracle_fix), replayed 8 more times against the 8 new
       nulls. Output <PHASEB_ROOT>/gt/<domain>/fresh/.

Replay keys: <fid>_k<k>_o<i>_<j> (sample i corrected), <fid>_k<k>_n<j> (null),
<fid>_k<k>_x<j> (the oracle-fix correction). AppWorld's per-replay experiment
folders are deleted after each batch (rewards and traces stay in the JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attrib.groundtruth import ORACLE_MODEL, ORACLE_SYSTEM, _context, _drive, _render_step, to_blocks  # noqa: E402

K, R_C = 4, 2
MAIN = Path("/home/user/attrib_runs/main")

BLIND_SYSTEM_TAU2 = """You review a failed episode of tau2-bench. A customer-service agent \
served a customer and the task failed: the customer's request was not resolved as the domain \
policy requires. You see the domain policy, the agent's tools and the conversation up to one \
step of the agent. You do not see the customer's instructions or the task's grading, so you do \
not know what the correct outcome was.

Decide whether the agent's action at that step is a mistake: an action a careful agent \
following the policy would not take. Actions that are correct, or harmless and do not lead \
away from resolving the request as the policy requires, are ok.

If it is a mistake, give the single action a careful agent would take at that step instead: \
one assistant turn, either a message to the user or one or more tool calls. Correct only this \
step; do not do more than a careful agent would do at this point. The corrected action must be \
one the agent could have taken with what it knew then (the conversation and tool results so \
far, and the policy).

Answer with JSON:
{"verdict": "ok" | "mistake", "why": "<one sentence>",
 "action": {"message": "<text to the user>"} | {"tool_calls": [{"name": "<tool>", "arguments": {...}}]}}
("action" only when the verdict is mistake.)"""

BLIND_SYSTEM_AW = """You review a failed episode of AppWorld. An autonomous code agent worked \
on a supervisor's task and failed: the task was not done as its unit tests require. You see \
what the agent is and the rules of its prompt, the apps' APIs, and the episode up to one step of \
the agent. You do not see the task's grading, so you do not know what the correct outcome was.

Decide whether the agent's action at that step is a mistake: an action a careful agent \
following its rules would not take on the way to doing the task. Actions that are correct, or \
harmless and do not lead away from doing the task (for example reading API documentation), are \
ok.

If it is a mistake, give the single action a careful agent would take at that step instead: \
one assistant turn, a short thought and the python code to run. Correct only this step; do not \
do more than a careful agent would do at this point. The corrected action must be one the agent \
could have taken with what it knew then (the outputs it had seen and the variables its earlier \
code defined).

Answer with JSON:
{"verdict": "ok" | "mistake", "why": "<one sentence>",
 "action": {"thought": "<one or two sentences>", "code": "<the python code, without markdown fences>"}}
("action" only when the verdict is mistake.)"""


def blind_prompt_tau2(rec: dict, k: int) -> str:
    """attrib.groundtruth3.oracle_prompt without the grading block."""
    conv = [f"user (opening): {rec['opening']}"]
    for st in rec["steps"][:k]:
        conv.append(f"step {st['index']}:\n{_render_step(st)}")
    act = "\n".join(f"  agent says: {b['text']}" if b["type"] == "text" else
                    f"  agent calls {b['name']}({json.dumps(b['input'], ensure_ascii=False)})"
                    for b in rec["steps"][k]["assistant"])
    return ("=== CONVERSATION ===\n" + "\n".join(conv) +
            f"\n\n=== AGENT ACTION AT STEP {k} (judge this step; the conversation above is "
            f"everything before it) ===\n{act}\n")


def blind_prompt_aw(rec: dict, k: int) -> str:
    """attrib.aw.oracle_prompt without the grading block."""
    from attrib.aw import _steps_text
    st = rec["steps"][k]
    return (f"=== EPISODE ===\n{_steps_text(rec, k)}\n\n"
            f"=== AGENT ACTION AT STEP {k} (judge this step; the episode above is everything "
            f"before it) ===\n{st.get('assistant')}\n")


# ------------------------------------------------------------------ common --
def _setup(domain: str):
    """(dom for the driver, tau2 module or None, context, force_of, drive)."""
    if domain == "appworld":
        from attrib import aw
        return "appworld", None, aw.context(), aw.to_force, aw.drive
    dom = domain.split("_", 1)[1]
    os.environ["TAU2_DOMAIN"] = dom
    from agent_exp import tau2_env as m
    names = {t["name"] for t in m.tool_defs(m.BASE_COMPONENTS)}
    return dom, m, _context(m), (lambda a: to_blocks(a, names)), _drive


def run_oracle(jobs: list, odir: Path, system: str, make_prompt, ctx: str, force_of, workers: int) -> None:
    """jobs: (fid, rec, k); K samples per job, one after another (shared prefix)."""
    from rrsi.llm import generate

    def one(job):
        fid, rec, k = job
        prompt = make_prompt(rec, k)
        for i in range(K):
            p = odir / f"{fid}_k{k}_o{i}.json"
            if p.exists():
                continue
            raw = generate(prompt, system=system, json_only=True, model=ORACLE_MODEL, cache_prefix=ctx,
                           role="oracle")
            try:
                v = json.loads(raw)
            except json.JSONDecodeError:
                v = {"verdict": "unparseable", "raw": raw[:2000]}
            if not isinstance(v, dict):
                v = {"verdict": "unparseable", "raw": str(v)[:2000]}
            if v.get("verdict") == "mistake":
                v["force"] = force_of(v.get("action"))
                if v["force"] is None:
                    v["verdict"] = "invalid_action"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(v, ensure_ascii=False, indent=1))
    jobs = sorted(jobs, key=lambda j: j[2])
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


def _samples(odir: Path, fid: str, k: int) -> list[dict]:
    return [json.loads((odir / f"{fid}_k{k}_o{i}.json").read_text()) for i in range(K)]


def drive_all(refs: list, rdir: Path, dom: str, drive, workers: int) -> int:
    for _ in range(3):
        drive(refs, rdir, dom, workers)
        if dom == "appworld":
            shutil.rmtree(rdir / "_appworld" / "experiments", ignore_errors=True)
        missing = sum(not (rdir / f"{r['key']}.json").exists() for _, r in refs)
        if not missing:
            return 0
    return missing


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


def step_profile(fid: str, k: int, odir: Path, rdir: Path, n_null: int) -> dict:
    sm = _samples(odir, fid, k)
    null, nn = _rate(rdir, [f"{fid}_k{k}_n{j}" for j in range(n_null)], False)
    gains, rows = [], []
    for i, v in enumerate(sm):
        if v.get("verdict") != "mistake":
            gains.append(0.0)
            rows.append({"verdict": v.get("verdict")})
            continue
        c, nc = _rate(rdir, [f"{fid}_k{k}_o{i}_{j}" for j in range(R_C)], True)
        g = (c - null) if c is not None and null is not None and nc == R_C else 0.0
        gains.append(g)
        rows.append({"verdict": "mistake", "corrected": c, "n": nc})
    return {"k": k, "R": round(sum(gains) / K, 4), "null": null, "n_null": nn, "samples": rows}


# ------------------------------------------------------------------- blind --
def blind_steps(main: Path, domain: str) -> dict:
    """{fid: sorted steps} named on the rescuable failures."""
    from attrib.analyze import load
    from attrib.factorial import informed
    v = load(main / domain)
    at40, at0, _ = informed(main, domain, v)
    v["picks"]["search_inf@40"], v["picks"]["search_inf@0"] = at40, at0
    sr = json.loads((main / domain / "search" / "result.json").read_text())["failures"]
    v["picks"]["search@0"] = {x["fid"]: (x["suspects"][0] if x.get("suspects") else None) for x in sr}
    out = {}
    for f, g in v["gt"].items():
        if g["decisive"] is None:
            continue
        S = {g["decisive"]} | ({g["earliest"]} if g["earliest"] is not None else set())
        for m, pk in v["picks"].items():
            k = pk.get(f)
            if isinstance(k, dict):
                k = k.get("step")
            if isinstance(k, int) and 0 <= k < len(g["R"]):
                S.add(k)
        out[f] = sorted(S)
    return out


def blind(args) -> None:
    main = Path(args.main)
    dom, m, ctx, force_of, drive = _setup(args.domain)
    rep = main / args.domain / "gt" / "blind"
    odir, rdir = rep / "oracle", rep / "replays"
    rep.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    fails = {f["fid"]: f for f in json.loads((main / args.domain / "failures.json").read_text())}
    steps = blind_steps(main, args.domain)
    fids = sorted(steps)[:args.limit] if args.limit else sorted(steps)
    recs = {f: json.loads(Path(fails[f]["trace"]).read_text()) for f in fids}
    system, mk = (BLIND_SYSTEM_AW, blind_prompt_aw) if dom == "appworld" else (BLIND_SYSTEM_TAU2, blind_prompt_tau2)
    run_oracle([(f, recs[f], k) for f in fids for k in steps[f]], odir, system, mk, ctx, force_of, args.workers)
    reg = main / args.domain / "gt" / "a" / "replays"
    refs = []
    rdir.mkdir(parents=True, exist_ok=True)
    for f in fids:
        base = {"task_id": fails[f]["task_id"], "trace": fails[f]["trace"]}
        for k in steps[f]:
            sm = _samples(odir, f, k)
            mist = [i for i, x in enumerate(sm) if x.get("verdict") == "mistake"]
            for i in mist:
                refs += [(fails[f]["harness"], {**base, "start": k, "force": sm[i]["force"],
                                                "key": f"{f}_k{k}_o{i}_{j}"}) for j in range(R_C)]
            if mist:
                for j in range(4):
                    a, b = reg / f"{f}_k{k}_n{j}.json", rdir / f"{f}_k{k}_n{j}.json"
                    if a.exists() and not b.exists():
                        b.symlink_to(a.resolve())
                    elif not b.exists():
                        refs.append((fails[f]["harness"], {**base, "start": k, "key": f"{f}_k{k}_n{j}"}))
    missing = drive_all(refs, rdir, dom, drive, args.replay_workers or args.workers)
    res = {"K": K, "R_C": R_C, "missing_replays": missing,
           "failures": {f: [step_profile(f, k, odir, rdir, 4) for k in steps[f]] for f in fids}}
    (rep / "result.json").write_text(json.dumps(res, indent=1))
    print(f"{args.domain}: {len(fids)} failures, {sum(len(steps[f]) for f in fids)} steps, "
          f"{len(refs)} replays requested, {missing} missing", flush=True)


# ------------------------------------------------------------------- fresh --
def fresh_steps(domain: str) -> dict:
    """{fid: {"steps": sorted steps, "oracle": oracle step or None}}."""
    from attrib.phaseb import STEP_GROUPS, picks
    from attrib.phaseb_accuracy import rrsi_steps
    pk = picks(domain)
    pk["rrsi"] = rrsi_steps(domain)
    out = {}
    for f in pk["oracle"]:
        S = set()
        for g in STEP_GROUPS + ("rrsi", "oracle"):
            k = pk[g].get(f)
            if isinstance(k, int) and k >= 0:
                S.add(k)
        out[f] = {"steps": sorted(S), "oracle": pk["oracle"].get(f)}
    return out


def fresh(args) -> None:
    from attrib.groundtruth3 import oracle_prompt
    from attrib.phaseb import PB
    dom, m, ctx, force_of, drive = _setup(args.domain)
    rep = PB / "gt" / args.domain / "fresh"
    odir, rdir = rep / "oracle", rep / "replays"
    rep.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(rep / "usage.jsonl"))
    fails = {f["fid"]: f for f in json.loads((PB / "attrib" / args.domain / "failures.json").read_text())}
    st = fresh_steps(args.domain)
    fids = sorted(st)[:args.limit] if args.limit else sorted(st)
    recs = {f: json.loads(Path(fails[f]["trace"]).read_text()) for f in fids}
    run_oracle([(f, recs[f], k) for f in fids for k in st[f]["steps"] if k < recs[f]["n_steps"]],
               odir, ORACLE_SYSTEM, oracle_prompt, ctx, force_of, args.workers)
    reg_odir = PB / "gt" / args.domain / "a" / "oracle"
    reg = {g["fid"]: g for g in json.loads((PB / "gt" / args.domain / "a" / "result.json").read_text())["failures"]}
    refs, supplied = [], {}
    for f in fids:
        base = {"task_id": fails[f]["task_id"], "trace": fails[f]["trace"]}
        ko = st[f]["oracle"]
        for k in st[f]["steps"]:
            if k >= recs[f]["n_steps"]:
                continue
            sm = _samples(odir, f, k)
            mist = [i for i, x in enumerate(sm) if x.get("verdict") == "mistake"]
            for i in mist:
                refs += [(fails[f]["harness"], {**base, "start": k, "force": sm[i]["force"],
                                                "key": f"{f}_k{k}_o{i}_{j}"}) for j in range(R_C)]
            n_null = 8 if k == ko else (4 if mist else 0)
            refs += [(fails[f]["harness"], {**base, "start": k, "key": f"{f}_k{k}_n{j}"}) for j in range(n_null)]
        if ko is None:
            continue
        # the correction the oracle-fix group was given (attrib.phaseb.oracle_fix's choice)
        best = None
        for i, s in enumerate(reg[f]["steps"][ko]["samples"]):
            if s.get("verdict") == "mistake" and s.get("corrected") is not None and s.get("n") == 2 \
                    and (best is None or s["corrected"] > best[1]):
                best = (i, s["corrected"])
        if best is None:
            continue
        o = json.loads((reg_odir / f"{f}_k{ko}_o{best[0]}.json").read_text())
        supplied[f] = {"k": ko, "sample": best[0], "registered_corrected": best[1],
                       "registered_null": reg[f]["steps"][ko]["null"], "registered_R": reg[f]["R"][ko]}
        refs += [(fails[f]["harness"], {**base, "start": ko, "force": o["force"], "key": f"{f}_k{ko}_x{j}"})
                 for j in range(8)]
    missing = drive_all(refs, rdir, dom, drive, args.replay_workers or args.workers)
    res = {"K": K, "R_C": R_C, "missing_replays": missing, "failures": {}}
    for f in fids:
        ko = st[f]["oracle"]
        prof = {k: step_profile(f, k, odir, rdir, 8 if k == ko else 4) for k in st[f]["steps"]
                if k < recs[f]["n_steps"]}
        row = {"oracle_step": ko, "steps": list(prof.values())}
        if f in supplied:
            c, nc = _rate(rdir, [f"{f}_k{ko}_x{j}" for j in range(8)], True)
            null, nn = _rate(rdir, [f"{f}_k{ko}_n{j}" for j in range(8)], False)
            row["supplied"] = {**supplied[f], "fresh_corrected": c, "fresh_n": nc, "fresh_null": null,
                               "fresh_n_null": nn,
                               "fresh_gain": (c - null) if c is not None and null is not None else None}
        res["failures"][f] = row
    (rep / "result.json").write_text(json.dumps(res, indent=1))
    print(f"{args.domain}: {len(fids)} traces, {len(refs)} replays requested, {missing} missing", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("blind", "fresh"))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline", "appworld"))
    ap.add_argument("--main", default=str(MAIN))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N failures only (smoke test)")
    args = ap.parse_args()
    (blind if args.cmd == "blind" else fresh)(args)


if __name__ == "__main__":
    main()
