"""Counterfactual search: an attribution method that replays (AgenTracer-like,
without the grading criteria the ground truth's oracle sees).

Per failed tau2 episode:
  1. One call (deepseek-v4-pro, thinking off) sees the domain policy, the tools
     and the failed conversation (not the gold actions, the required
     information or the customer's hidden instructions) and names up to
     SUSPECTS steps, most suspect first, each with a corrected action and the
     harness component at fault.
  2. Suspects are tested in that order with the ground truth's replay test:
     N_REPLAYS corrected replays; if they reach FLIP successes, N_REPLAYS null
     replays; the suspect flips if corrected minus null >= FLIP. The search
     stops at the first flip or when the next suspect would exceed BUDGET
     replays.
  3. Answer: the flipped suspect, else the first suspect (fallback).

The record keeps the replays spent after each suspect, so the answer under a
smaller budget is read off the same run (attrib.search.answer_at).

  python -m attrib.search run --domain tau2_retail --failures <failures.json> \
      --out <dir> [--budget 40] [--workers 8] [--replay-workers N]

With --domain appworld (Addendum 2 of attrib/PREREGISTRATION.md): the suspect
call sees attrib.aw's context and the failed episode without its grading, an
action is one assistant turn with a python block, replays use the AppWorld
driver; the search itself is unchanged.
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

from attrib.groundtruth import FLIP, N_REPLAYS, _context, _drive, _render_step, to_blocks  # noqa: E402
from attrib.methods import COMP_TEXT, COMPONENTS  # noqa: E402

SUSPECTS = 5
BUDGET = 40
MODEL = "deepseek-v4-pro"

SYSTEM = """A customer-service agent on tau2-bench failed its task: the customer's request was \
not resolved as the domain policy requires. You see the domain policy, the agent's tools and \
the conversation; you do not know what the correct outcome was.

Name the steps where the agent most likely went wrong, most suspect first (at most %d). For \
each, give the single action a careful agent would have taken at that step instead (one \
assistant turn: a message to the user or one or more tool calls, using only what the agent \
knew at that point) and the harness component most likely at fault.

Harness components:
%s

Answer with JSON:
{"suspects": [{"step": <int>, "why": "<one sentence>", "component": "<component>",
  "action": {"message": "<text>"} | {"tool_calls": [{"name": "<tool>", "arguments": {...}}]}}]}""" % (
    SUSPECTS, COMP_TEXT)


def prompt(rec: dict) -> str:
    conv = [f"user (opening): {rec['opening']}"]
    for st in rec["steps"]:
        conv.append(f"step {st['index']}:\n{_render_step(st)}")
    return "=== FAILED CONVERSATION ===\n" + "\n".join(conv)


def suspects(fails: list, sdir: Path, m, workers: int) -> None:
    """m: the tau2 environment module, or None for AppWorld (attrib.aw)."""
    from rrsi.llm import generate
    if m is None:
        from attrib import aw
        ctx, comps, make_prompt, force_of = aw.context(), aw.COMPONENTS, aw.search_prompt, aw.to_force
        system = aw.search_system(SUSPECTS, "\n".join(f"- {k}: {v}" for k, v in comps.items()))
    else:
        ctx, comps, make_prompt, system = _context(m), COMPONENTS, prompt, SYSTEM
        names = {t["name"] for t in m.tool_defs(m.BASE_COMPONENTS)}
        force_of = lambda action: to_blocks(action, names)  # noqa: E731

    def one(f):
        p = sdir / f"{f['fid']}.json"
        if p.exists():
            return
        rec = json.loads(Path(f["trace"]).read_text())
        raw = generate(make_prompt(rec), system=system, json_only=True, model=MODEL, cache_prefix=ctx,
                       role="search", max_tokens=4000)
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            v = {}
        out, seen = [], set()
        for s in (v.get("suspects") if isinstance(v, dict) else None) or []:
            if not isinstance(s, dict):
                continue
            try:
                k = int(s.get("step"))
            except (TypeError, ValueError):
                continue
            force = force_of(s.get("action"))
            if 0 <= k < rec["n_steps"] and k not in seen:
                seen.add(k)
                out.append({"step": k, "force": force, "why": s.get("why"),
                            "component": s.get("component") if s.get("component") in comps else None})
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"suspects": out[:SUSPECTS]}, ensure_ascii=False, indent=1))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, fails))


def _count(rdir: Path, fid: str, i: int, kind: str):
    s = n = miss = 0
    for j in range(N_REPLAYS):
        p = rdir / f"{fid}_s{i}_{kind}{j}.json"
        if not p.exists():
            miss += 1
            continue
        x = json.loads(p.read_text())
        if kind == "c" and not (x.get("replay") or {}).get("forced"):
            continue
        n += 1
        s += int(x.get("reward", 0) >= 1)
    return s, n, miss


def state(f: dict, sus: list, rdir: Path, budget: int) -> dict:
    """Replays still needed, and the search's record so far."""
    spent, log = 0, []
    base = {"task_id": f["task_id"], "trace": f["trace"]}
    for i, s in enumerate(sus):
        if s["force"] is None:
            log.append({"i": i, "step": s["step"], "result": "invalid_action", "spent": spent})
            continue
        if spent + N_REPLAYS > budget:
            break
        c, nc, miss = _count(rdir, f["fid"], i, "c")
        if miss:
            return {"need": [{**base, "start": s["step"], "force": s["force"],
                              "key": f"{f['fid']}_s{i}_c{j}"} for j in range(N_REPLAYS)], "log": log}
        spent += N_REPLAYS
        if nc < N_REPLAYS or c < FLIP:
            log.append({"i": i, "step": s["step"], "corrected": c, "result": "no_flip", "spent": spent})
            continue
        if spent + N_REPLAYS > budget:
            log.append({"i": i, "step": s["step"], "corrected": c, "result": "out_of_budget",
                        "spent": spent})
            break
        z, _, miss = _count(rdir, f["fid"], i, "n")
        if miss:
            return {"need": [{**base, "start": s["step"], "key": f"{f['fid']}_s{i}_n{j}"}
                             for j in range(N_REPLAYS)], "log": log}
        spent += N_REPLAYS
        flip = c - z >= FLIP
        log.append({"i": i, "step": s["step"], "corrected": c, "null": z,
                    "result": "flip" if flip else "no_flip", "spent": spent})
        if flip:
            break
    return {"need": [], "log": log}


def answer_at(sus: list, log: list, budget: int) -> dict:
    """The search's answer had it been given `budget` replays (the log of a run
    with a budget at least as large)."""
    for e in log:
        if e["spent"] > budget:
            break
        if e["result"] == "flip":
            return {"step": e["step"], "component": sus[e["i"]]["component"], "fallback": False,
                    "replays": e["spent"]}
    spent = max([e["spent"] for e in log if e["spent"] <= budget], default=0)
    top = sus[0] if sus else {"step": None, "component": None}
    return {"step": top["step"], "component": top["component"], "fallback": True, "replays": spent}


def run(fails: list, out: Path, dom: str, m, budget: int, workers: int,
        replay_workers: int | None = None) -> None:
    sdir, rdir = out / "suspects", out / "replays"
    suspects(fails, sdir, m, workers)
    if dom == "appworld":
        from attrib.aw import drive
    else:
        drive = _drive
    sus = {f["fid"]: json.loads((sdir / f"{f['fid']}.json").read_text())["suspects"] for f in fails}
    for _ in range(2 * (budget // N_REPLAYS) + 2):
        wave = [(f["harness"], r) for f in fails for r in state(f, sus[f["fid"]], rdir, budget)["need"]]
        if not wave:
            break
        drive(wave, rdir, dom, replay_workers or workers)
    res = []
    for f in fails:
        log = state(f, sus[f["fid"]], rdir, budget)["log"]
        res.append({"fid": f["fid"], "suspects": [s["step"] for s in sus[f["fid"]]], "log": log,
                    **{f"at{b}": answer_at(sus[f["fid"]], log, b) for b in (8, 16, 24, 32, 40) if b <= budget}})
    (out / "result.json").write_text(json.dumps({"budget": budget, "failures": res}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run",))
    ap.add_argument("--domain", required=True, choices=("tau2_retail", "tau2_airline", "appworld"))
    ap.add_argument("--failures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay-workers", type=int, default=None)
    args = ap.parse_args()
    if args.domain == "appworld":
        dom, m = "appworld", None
    else:
        dom = args.domain.split("_", 1)[1]
        os.environ["TAU2_DOMAIN"] = dom
        from agent_exp import tau2_env as m
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RRSI_USAGE_LOG", str(out / "usage.jsonl"))
    run(json.loads(Path(args.failures).read_text()), out, dom, m, args.budget, args.workers,
        args.replay_workers)


if __name__ == "__main__":
    main()
