"""Second judge configurations (third review, experiment J).

The bank's single-step check regenerates the failing step and asks the analyst
model, which reads the whole trajectory, whether the diagnosed error is gone.
Here we test two cheaper judges on the same bank pairs, whose full-replay
outcome y is already known:

  short_pro    analyst model, sees only the task, the root cause, the last
               observation before the failing step, and the old and new action
  full_flash   the agent model as judge, same prompt as the bank's judge
               (on a random subsample of --n_full pairs, as it costs ~0.7 replay)
  short_flash  both changes

For each sampled (patch, replay) pair the failing step is regenerated once and
judged by all three; tokens are metered separately for the regeneration and
for each judge. Pairs are stratified on y (half repaired, half not), so
sensitivity and false-positive rate are estimated directly.

  python -m agent_exp.judge_configs --out runs/tau2_retail --domain tau2_retail --n 360
  python -m agent_exp.judge_configs --out runs/tau2_retail --domain tau2_retail --report
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading

import numpy as np

from agent_exp.analyst import JUDGE_SCHEMA, render_trace
from agent_exp.bank import Domain, _load, _pmap, _save, ground_truth

CONFIGS = ["short_pro", "full_flash", "short_flash"]
_lock = threading.Lock()


def _blocks(blocks):
    out = []
    for b in blocks:
        out.append(b["text"] if b["type"] == "text" else f"CALL {b['name']}({json.dumps(b['input'])})")
    return "\n".join(out)


def judge_short(llm, tr, step, root_cause, new_blocks, model=None):
    prev = ""
    if step > 0:
        p = tr.steps[step - 1]
        prev = "\n".join(str(r["content"])[:1500] for r in p.tool_results)
        if getattr(p, "user", None):
            prev += f"\nUSER: {p.user}"
    if step == 0 and getattr(tr, "opening", None):
        prev = f"USER: {tr.opening}"
    prompt = f"""An agent failed. Task: {tr.question[:1500]}

Diagnosed error at step {step}: {root_cause}

What the agent saw just before step {step}:
{prev or '(nothing)'}

Original action at step {step}:
{_blocks(tr.steps[step].assistant)}

A modified agent produced this action at step {step} instead:
{_blocks(new_blocks)}

Does the new action avoid the diagnosed error, so that the episode is back on a
correct path?"""
    return int(llm.ask_json(prompt, JUDGE_SCHEMA, "judge", max_tokens=2000, model=model)["fixed"])


def sample_pairs(bank, n, seed):
    rng = random.Random(seed)
    pos, neg = [], []
    for key, c in sorted(bank["cells"].items()):
        for r, y in enumerate(c["y"]):
            (pos if y else neg).append((key, r))
    rng.shuffle(pos)
    rng.shuffle(neg)
    return sorted(pos[:n // 2] + neg[:n - n // 2])


def run(args):
    from agent_exp.llm import AGENT_MODEL, LLM

    llm, D = LLM(), Domain(args.domain)
    bank = _load(os.path.join(args.out, "bank.json"))
    data = {d["uid"]: d for d in _load(os.path.join(args.out, "traces.json"))}
    attrs = _load(os.path.join(args.out, "attributions.json"))
    patches = _load(os.path.join(args.out, "patches.json"))
    path = os.path.join(args.out, "judge_configs.json")
    res = _load(path) or {"rows": {}}
    allp = sample_pairs(bank, args.n, args.seed)
    # the full-trace flash judge costs ~0.7 replay per check: run it on a subsample
    sub = set(random.Random(args.seed + 1).sample(allp, min(args.n_full, len(allp))))
    pairs = [p for p in allp if f"{p[0]}|{p[1]}" not in res["rows"]]
    print("pairs to run", len(pairs), flush=True)
    meter = llm.meter

    def one(pair):
        key, r = pair
        c = bank["cells"][key]
        i, j, k = map(int, key.split(","))
        uid = c["trace"][r]
        d = data[uid]
        tr = D.trace_from(d)
        a = attrs[uid]
        step = max(0, min(a["step"], len(tr.steps) - 1))
        cid = D.comp_ids[j]
        comps = D.m.components_with(tr.faults, {cid: patches[f"{i},{j}"][k]})
        meter(reset=True)
        new = D.m.run_agent(llm, D.tasks[d["task_id"]], comps, tr.faults, prefix=tr,
                            start_step=step, max_new_steps=1)
        row = {"y": c["y"][r], "z_bank": c["z"][r], "tok_full": c["tok_full"][r],
               "tok_single_bank": c["tok_single"][r], "tok_regen": meter(reset=True)}
        ok = len(new.steps) > step
        blocks = new.steps[step].assistant if ok else None
        for name in CONFIGS:
            if name == "full_flash" and pair not in sub:
                continue
            model = AGENT_MODEL if name.endswith("flash") else None
            z = 0
            if ok:
                if name.startswith("short"):
                    z = judge_short(llm, tr, step, a["root_cause"], blocks, model)
                else:
                    z = int(llm.ask_json(*_full_prompt(tr, step, a["root_cause"], blocks),
                                         max_tokens=2000, model=model)["fixed"])
            row[f"z_{name}"] = z
            row[f"tok_{name}"] = meter(reset=True)
        with _lock:
            res["rows"][f"{key}|{r}"] = row
        return row

    for s in range(0, len(pairs), 4 * args.workers):
        _pmap(one, pairs[s:s + 4 * args.workers], args.workers)
        with _lock:
            _save(path, res)
        print("done", len(res["rows"]), flush=True)
    print(json.dumps({k: dict(llm.io)[k] for k in llm.io}))


def _full_prompt(tr, step, root_cause, blocks):
    """The bank judge's prompt (analyst.judge_step), for use with another model."""
    prompt = f"""An agent failed; the diagnosed error is at step {step}: {root_cause}

Original trace:
{render_trace(tr)}

A modified agent produced this action at step {step} instead:
{json.dumps(blocks)}

Does the new action avoid the diagnosed error, so that the episode is back on a
correct path?"""
    return prompt, JUDGE_SCHEMA, "judge"


# ---- report ------------------------------------------------------------------
def kl(a, b):
    a, b = np.clip(a, 1e-6, 1 - 1e-6), np.clip(b, 1e-6, 1 - 1e-6)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def break_even(s, e, b, q):
    """Cost ratio c_S/c_F below which single-step checks decide an edge with q
    and spurious recovery b at lower expected cost than replays (Prop. cfid),
    averaged over true and false edges."""
    p1, p0 = b + (1 - b) * q, b
    z1, z0 = s * p1 + e * (1 - p1), s * p0 + e * (1 - p0)
    return float(0.5 * (kl(z1, z0) / kl(p1, p0) + kl(z0, z1) / kl(p0, p1)))


def wilson(k, n):
    if n == 0:
        return [None, None]
    p, z = k / n, 1.645
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def report(args):
    bank = _load(os.path.join(args.out, "bank.json"))
    R = _load(os.path.join(args.out, "judge_configs.json"))["rows"]
    rows = list(R.values())
    cells = list(bank["cells"].values())
    tf = float(np.mean([t for c in cells for t in c["tok_full"]]))
    I = 1 + max(int(k.split(",")[0]) for k in bank["cells"])
    J = 1 + max(int(k.split(",")[1]) for k in bank["cells"])
    _, E, Q, b = ground_truth(bank, I, J)
    q = float(np.median(Q.max(-1)[E])) if E.any() else 0.5
    bbar = float(np.mean([v["y"] for v in bank["null"].values()]))
    y = np.array([r["y"] for r in rows])
    regen = float(np.mean([r["tok_regen"] for r in rows]))
    out = {"n": len(rows), "n_pos": int(y.sum()), "tok_full": tf, "b": bbar, "q": q,
           "regen_cost": regen / tf, "configs": {}}
    # the bank judge on the whole bank, for reference
    ys = np.array([v for c in cells for v in c["y"]])
    zs = np.array([v for c in cells for v in c["z"]])
    ts = float(np.mean([t for c in cells for t in c["tok_single"]]))
    names = ["bank_all", "bank"] + CONFIGS
    for name in names:
        if name == "bank_all":
            z, yy, cost = zs, ys, ts / tf
        elif name == "bank":       # bank judge on the sampled pairs (its own regeneration)
            z, yy = np.array([r["z_bank"] for r in rows]), y
            cost = float(np.mean([r["tok_single_bank"] for r in rows])) / tf
        else:
            rs = [r for r in rows if f"z_{name}" in r]
            z, yy = np.array([r[f"z_{name}"] for r in rs]), np.array([r["y"] for r in rs])
            cost = (regen + float(np.mean([r[f"tok_{name}"] for r in rs]))) / tf
        s, e = float(z[yy == 1].mean()), float(z[yy == 0].mean())
        be = break_even(s, e, bbar, q)
        out["configs"][name] = {
            "sens": s, "sens_ci": wilson(int(z[yy == 1].sum()), int((yy == 1).sum())),
            "fpr": e, "fpr_ci": wilson(int(z[yy == 0].sum()), int((yy == 0).sum())),
            "cost": cost, "break_even": be, "efficiency": be / cost}
        out["configs"][name]["n"] = int(len(z))
        if name in CONFIGS:
            out["configs"][name]["judge_cost"] = float(np.mean([r[f"tok_{name}"] for r in rs])) / tf
            out["configs"][name]["agree_bank"] = float(np.mean(z == np.array([r["z_bank"] for r in rs])))
    _save(os.path.join(args.out, "judge_configs_report.json"), out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default="tau2_retail")
    ap.add_argument("--n", type=int, default=360)
    ap.add_argument("--n_full", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else run(args)


if __name__ == "__main__":
    main()
