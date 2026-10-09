"""Live selection with verification evidence inside RRSI (config key `selection`).

RRSI's own selection (selection = "full", the default) evaluates every screened
candidate on the whole evolve set, k trials per task, and applies Algorithm 2
to those measurements. With any other value, steps 5-6 of a round are replaced:
each screened candidate gets ONE evidence type at a fixed episode budget b, the
evidence gives an estimate of dS = S(candidate) - S(H_t) with a standard error,
and the same rule verify/analyze.py applies offline picks the winner (RRSI's
Algorithm 2 with the evidence's own noise band delta_e = z * se, z = delta_z;
the S* floor is replaced by dS >= -delta_e; the domain guard is checked on the
evidence episodes). Only the winner is then evaluated on the full evolve set,
because RRSI's next round analyses the incumbent's own trials; that refresh
evaluation is not used for the decision and its cost is reported separately.

  none          a screened candidate drawn at random (no evidence)
  judge         the LLM judge of verify/judge.py (0 episodes)
  sample@b      b fresh candidate episodes on random (task, trial) pairs of
                H_t's evaluation: mean(c - h)
  replay@b      b candidate replays of H_t's failed trials, spread evenly over
                the replay references of verify/replays.py: f * fix_c
  replaynull@b  b/2 candidate replays and b/2 null replays with H_t (shared by
                the round's candidates): f * (fix_c - fix_null)
  net@b         b/4 candidate and b/4 null replays and b/2 fresh candidate
                episodes on pairs H_t solved, compared with H_t's other trial
                of the task: f * (fix_c - fix_null) + (1 - f) * mean(c - h_other)

seqfull       RRSI's own full evaluation and rule, run sequentially: each
              screened candidate's evolve episodes in batches of SEQ_BATCH
              (random order over (task, trial) pairs); after each batch the
              candidate is dropped when its harness errors already exceed the
              guard's share of the planned episodes or when the predictive
              probability that RRSI's Algorithm 2 admits it after all episodes
              falls below SEQ_GAMMA (verify/posthoc_e1.p_admissible, the rule
              registered offline in PREREGISTRATION_E1R.md, addendum 2).
              Candidates that are not dropped complete their evaluation and
              are decided by RRSI's select_round; a dropped candidate is
              recorded as REJECTED with its early-stopped estimate, which is
              what the proposer sees in the next rounds' history.

f = failed trials / trials of H_t's evaluation. A missing replay (infrastructure)
is left out, as in the offline analysis; a missing fresh episode scores 0, as in
RRSI's Evaluate.

Cost accounting, the same for every mode including "full" (account_full):
r<t>/selection.json holds the episodes run from the first step, the replays,
their tokens (policy + user simulator), episode equivalents (tokens divided by
the mean tokens of one of H_t's evolve episodes; a fresh episode counts as
one), the judge's LLM tokens, and the refresh evaluation of the winner. The
search-side LLM tokens (analyst, proposer, critic) are logged per role by
rrsi/llm.py (RRSI_USAGE_LOG) and are the same machinery in every mode.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from rrsi import llm
from rrsi.components import novelty
from rrsi.evaluate import EvalResult
from rrsi.selection import Decision

from . import judge as judge_mod
from .replays import references
from .state import Cand, Round, trial_path

RULES = ("none", "judge", "sample", "replay", "replaynull", "net", "seqfull")
SEQ_BATCH = 10
SEQ_GAMMA = 0.05
MIN_B = {"sample": 1, "replay": 1, "replaynull": 2, "net": 4}


def parse_mode(mode: str) -> tuple[str, int]:
    """'net@40' -> ('net', 40); 'judge' -> ('judge', 0)."""
    rule, _, b = str(mode).strip().partition("@")
    if rule not in RULES:
        raise SystemExit(f"unknown selection {mode!r}: use full, none, judge, sample@b, "
                         f"replay@b, replaynull@b, net@b or seqfull")
    if rule in ("none", "judge", "seqfull"):
        if b:
            raise SystemExit(f"selection {rule} takes no episode budget")
        return rule, 0
    try:
        n = int(b)
    except ValueError:
        raise SystemExit(f"selection {mode!r} needs an episode budget, e.g. {rule}@40") from None
    if n < MIN_B[rule]:
        raise SystemExit(f"selection {rule} needs a budget of at least {MIN_B[rule]}")
    return rule, n


# ---------------------------------------------------------------- records --
def _load(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001 - missing or partial file
        return None


def _reward(rec) -> float:
    return float(rec.get("reward") or 0) if rec else 0.0


def _tokens(rec) -> int:
    """Policy + user-simulator tokens of one episode or replay."""
    tk = (rec or {}).get("tokens") or {}
    return int(tk.get("agent_total") or 0) + int(tk.get("user") or 0)


def _agent_tokens(rec):
    return ((rec or {}).get("tokens") or {}).get("agent_total") or None


def _job_records(run, job: str) -> list:
    d = run.runs / "jobs" / job
    return [_load(p) for p in sorted(d.glob("s*/*.json"))] if d.is_dir() else []


def _cost(recs: list, n_attempted: int | None = None) -> dict:
    present = [r for r in recs if r is not None]
    return {"n": len(recs) if n_attempted is None else n_attempted,
            "missing": sum(1 for r in recs if r is None),
            "tokens": sum(_tokens(r) for r in present),
            "harness_errors": sum(1 for r in present if "harness_error" in r)}


def _ep_tokens(run, job: str) -> float:
    toks = [_tokens(r) for r in _job_records(run, job) if r and _tokens(r)]
    return float(np.mean(toks)) if toks else 1.0


# --------------------------------------------------------------- evidence --
def _pairs(inc: EvalResult) -> list[tuple]:
    """[(task, s, h_s, h_other, agent_tokens_s)] over H_t's evaluation."""
    out = []
    for tid, tr in inc.per_task.items():
        for s, h in enumerate(tr.rewards):
            other = [x for j, x in enumerate(tr.rewards) if j != s]
            out.append((tid, s, h, float(np.mean(other)) if other else h,
                        tr.tokens[s] if s < len(tr.tokens) else None))
    return out


def _spread(refs: list, n: int, rng: random.Random) -> list:
    """n draws spread evenly over refs (each ref once before any ref twice)."""
    order = list(range(len(refs)))
    rng.shuffle(order)
    return [refs[order[i % len(order)]] for i in range(n)] if refs else []


def _replay(run, wt: Path, t: int, label: str, picks: list) -> list:
    items, seen = [], Counter()
    for r in picks:
        j = seen[r["task_id"]]
        seen[r["task_id"]] += 1
        items.append({"key": f"{r['task_id']}__{j}", "task_id": r["task_id"],
                      "trace": r["trace"], "start": r["start"]})
    if items:
        run.domain.replay(wt, run.runs, f"r{t}/{label}", items)
    d = run.runs / "replays" / f"r{t}" / label
    return [_load(d / f"{it['key']}.json") for it in items]


def _fresh(run, wt: Path, job: str, pairs: list) -> list:
    """One fresh episode of the candidate per pair; the j-th pair of a task
    gets the candidate's trial j-1 (resume-safe: the runner fills gaps)."""
    n = Counter(p[0] for p in pairs)
    for j in range(1, max(n.values(), default=0) + 1):
        ids = sorted(t for t in n if n[t] >= j)
        run.domain.run(wt, run.runs, job, ids, j, log_prefix=job)
    seen, out = Counter(), []
    for p in pairs:
        s = seen[p[0]]
        seen[p[0]] += 1
        out.append(_load(trial_path(run.runs, job, p[0], s)))
    return out


def _collect(run, t: int, c, rule: str, b: int, refs: list, inc: EvalResult,
             seed: str) -> dict:
    wt = run.wt_root / f"r{t}{c.variant}"
    rng = random.Random(f"{seed}:{c.variant}")
    nc = {"replay": b, "replaynull": b // 2, "net": b // 4}.get(rule, 0)
    ev = {"replays": _replay(run, wt, t, c.variant, _spread(refs, nc, rng)) if nc else [],
          "pairs": [], "fresh": []}
    if rule in ("sample", "net"):
        pool = _pairs(inc)
        n = b
        if rule == "net":
            pool = [p for p in pool if p[2] >= 1]
            n = b - 2 * nc
        ev["pairs"] = rng.sample(pool, min(n, len(pool)))
        ev["fresh"] = _fresh(run, wt, f"r{t}{c.variant}_ev", ev["pairs"])
    return ev


def _rel_cost(tc: list, th: list) -> float:
    tc = [x for x in tc if x]
    th = [x for x in th if x]
    if not tc or not th:
        return 0.0
    return float((np.mean(tc) - np.mean(th)) / np.mean(th))


def estimate(rule: str, ev: dict, null: list, f: float):
    """-> (dS, se, dC) or None when the evidence is empty; the estimators of
    verify/analyze.py applied to live evidence."""
    if rule == "sample":
        if not ev["pairs"]:
            return None
        diff = [_reward(r) - p[2] for r, p in zip(ev["fresh"], ev["pairs"])]
        se = float(np.std(diff, ddof=1)) / math.sqrt(len(diff)) if len(diff) > 1 else 1.0
        return (float(np.mean(diff)), se,
                _rel_cost([_agent_tokens(r) for r in ev["fresh"]], [p[4] for p in ev["pairs"]]))
    fc = [_reward(r) for r in ev["replays"] if r is not None]
    if not fc:
        return None
    fix_c = float(np.mean(fc))
    var_c = fix_c * (1 - fix_c) / len(fc)
    if rule == "replay":
        return (f * fix_c, f * math.sqrt(var_c) or 1e-9, 0.0)
    fn = [_reward(r) for r in null if r is not None]
    if not fn:
        return None
    fix_n = float(np.mean(fn))
    se2 = var_c + fix_n * (1 - fix_n) / len(fn)
    gain = f * (fix_c - fix_n)
    if rule == "replaynull":
        return (gain, f * math.sqrt(se2) or 1e-9, 0.0)
    harm = [_reward(r) - p[3] for r, p in zip(ev["fresh"], ev["pairs"])]
    h = float(np.mean(harm)) if harm else 0.0
    se_h = float(np.std(harm, ddof=1)) / math.sqrt(len(harm)) if len(harm) > 1 else 0.0
    se = math.sqrt((f ** 2) * se2 + ((1 - f) * se_h) ** 2) or 1e-9
    return (gain + (1 - f) * h, se,
            _rel_cost([_agent_tokens(r) for r in ev["fresh"]], [p[4] for p in ev["pairs"]]))


def admissible(est, nu: int, cfg, z: float) -> tuple[bool, str, float]:
    """RRSI's Algorithm 2 on an estimate, with delta_e = z * se."""
    if est is None:
        return False, "no evidence (nothing to replay or sample)", 0.0
    dS, se, dC = est
    delta = z * (se or 0.0)
    if dS > delta:
        budget = cfg.beta0 + cfg.beta1 * dS
        ok = (dC or 0.0) <= budget
        return ok, (f"estimate {dS:+.4f} > delta_e {delta:.4f}; cost change {dC:+.3f} "
                    f"{'<=' if ok else '>'} budget {budget:.3f}"), delta
    if dS >= -delta and delta > 0:
        shaped = cfg.w_s * dS - cfg.w_c * (dC or 0.0) + cfg.w_n * nu
        ok = shaped > 0
        return ok, (f"estimate {dS:+.4f} within delta_e {delta:.4f}; shaped "
                    f"{cfg.w_s}*dS - {cfg.w_c}*dC + {cfg.w_n}*nu = {shaped:+.4f} "
                    f"{'>' if ok else '<='} 0 (nu={nu})"), delta
    return False, f"estimate {dS:+.4f} not above -delta_e {-delta:.4f}", delta


def _usage_diff(a: dict, b: dict) -> dict:
    out = {}
    for m, u in b.items():
        d = {k: u.get(k, 0) - (a.get(m) or {}).get(k, 0) for k in ("calls", "in", "out", "cache_read")}
        if d["calls"]:
            out[m] = d
    return out


# ------------------------------------------------------------------- main --
def select_with_evidence(run, t: int, cands: list, inc_ev: EvalResult, rdir: Path,
                         ids: list, counts: dict):
    """Steps 5-6 of a round with one evidence type; returns (winner, decisions)
    like rrsi.selection.select_round. The winner (if any) leaves with its full
    evolve evaluation in winner.ev."""
    d, cfg = run.domain, run.cfg
    rule, b = parse_mode(cfg.selection)
    if rule == "seqfull":
        return select_sequential(run, t, cands, inc_ev, rdir, ids, counts)
    z = float(cfg.delta_z)
    edir = rdir / "evidence"
    edir.mkdir(exist_ok=True)
    fr = run.frontier()
    inc_commit, inc_job = fr["incumbent"]["commit"], fr["incumbent"]["job"]
    n_trials = sum(len(tr.rewards) for tr in inc_ev.per_task.values())
    f = sum(1 for tr in inc_ev.per_task.values() for r in tr.rewards if r < 1) / max(1, n_trials)
    live = [c for c in cands if c.gate_failure is None]
    seed = f"{d.name}:{t}:{cfg.selection}"
    rnd = Round(t, inc_commit, inc_job, [], None)

    refs = []
    if rule in ("replay", "replaynull", "net") and live:
        rp = edir / "refs.json"
        if rp.exists():
            refs = json.loads(rp.read_text())
        else:
            refs = references(run.runs, rnd)
            rp.write_text(json.dumps(refs, indent=1))
    null = []
    nn = {"replaynull": b // 2, "net": b // 4}.get(rule, 0)
    if nn and refs and live:
        wt = run.checkout("incumbent", run.branch)
        null = _replay(run, wt, t, "null", _spread(refs, nn, random.Random(f"{seed}:null")))
    u0 = llm.usage()
    judged = {}
    if rule == "judge" and live:
        rnd.cands = [Cand(t, c.variant, c.commit, "live", None, {}, c.edits) for c in live]
        judged = judge_mod.run(d, run.runs, edir, rnd,
                               model=cfg.notes.get("judge_model") or cfg.analyst_model)
    judge_llm = _usage_diff(u0, llm.usage())
    evid = {}
    if rule in ("sample", "replay", "replaynull", "net") and live:
        with ThreadPoolExecutor(max_workers=max(1, cfg.eval_parallel)) as ex:
            res = list(ex.map(lambda c: _collect(run, t, c, rule, b, refs, inc_ev, seed), live))
        evid = {c.variant: e for c, e in zip(live, res)}

    pick = random.Random(f"{seed}:none").choice(live).variant if rule == "none" and live else None
    decisions, rows = [], {}
    for c in cands:
        if c.gate_failure is not None:
            decisions.append(Decision(c.variant, False, c.gate_failure))
            continue
        nu = novelty(c.components, counts)
        e = evid.get(c.variant) or {"replays": [], "pairs": [], "fresh": []}
        recs = e["replays"] + e["fresh"]
        cost = _cost(recs)
        g = []
        if recs:
            pseudo = EvalResult("evidence", 1, {}, 0.0, None, len(recs), cost["missing"],
                                {"harness_error_rate": cost["harness_errors"] /
                                 max(1, len(recs) - cost["missing"])})
            g = d.guards(inc_ev, pseudo)
        if rule == "none":
            est, delta = None, 0.0
            ok, why = True, ("no evidence: drawn at random among the screened candidates"
                             if c.variant == pick else "no evidence: another candidate was drawn")
        else:
            if rule == "judge":
                j = judged.get(c.variant) or {}
                est = (float(j["dS"]), 0.0, 0.0) if j.get("dS") is not None else None
            else:
                est = estimate(rule, e, null, f)
            ok, why, delta = admissible(est, nu, cfg, z)
            if ok and g:
                ok, why = False, "domain guard violated on the evidence episodes: " + "; ".join(g)
        dec = Decision(c.variant, ok, f"[{cfg.selection}] {why}",
                       delta_S=None if est is None else est[0],
                       delta_C=None if est is None else est[2], novelty=nu, guards=g)
        decisions.append(dec)
        rows[c.variant] = {"dS": dec.delta_S, "se": None if est is None else est[1],
                           "dC": dec.delta_C, "delta_e": delta, "novelty": nu,
                           "admissible": ok, "reason": why,
                           "replays": _cost(e["replays"]), "fresh": _cost(e["fresh"]),
                           "fix_c": (float(np.mean([_reward(r) for r in e["replays"] if r]))
                                     if any(r is not None for r in e["replays"]) else None)}

    winner = None
    adm = [(c, x) for c, x in zip(cands, decisions) if x.admissible]
    if rule == "none":
        winner = next((c for c, _ in adm if c.variant == pick), None)
    elif adm:
        winner, _ = max(adm, key=lambda cx: cx[1].delta_S)

    refresh = {"n": 0, "missing": 0, "tokens": 0, "harness_errors": 0}
    if winner is not None:
        dec = decisions[cands.index(winner)]
        run._evaluate(t, winner, rdir, ids)
        refresh = _cost(_job_records(run, f"r{t}{winner.variant}"))
        if winner.ev is None:
            dec.admissible = False
            dec.reason += (f"; selected, but its full evolve evaluation (needed for the next "
                           f"round's analysis) failed: {winner.detail}; H_t kept")
            winner.gate_failure = None
            winner = None
        else:
            dec.S, dec.C = winner.ev.S, winner.ev.C

    ep_tok = _ep_tokens(run, inc_job)
    null_cost = _cost(null)
    fresh_n = sum(r["fresh"]["n"] for r in rows.values())
    rep_n = null_cost["n"] + sum(r["replays"]["n"] for r in rows.values())
    rep_tok = null_cost["tokens"] + sum(r["replays"]["tokens"] for r in rows.values())
    out = {"t": t, "selection": cfg.selection, "rule": rule, "b": b, "f": f,
           "n_refs": len(refs), "ep_tokens": ep_tok,
           "null": {**null_cost, "fix_null": (float(np.mean([_reward(r) for r in null if r]))
                                              if any(r is not None for r in null) else None)},
           "candidates": rows, "winner": None if winner is None else winner.variant,
           "cost": {"selection_episodes": fresh_n, "selection_replays": rep_n,
                    "selection_tokens": rep_tok + sum(r["fresh"]["tokens"] for r in rows.values()),
                    "selection_episode_equivalents": fresh_n + rep_tok / ep_tok,
                    "judge_llm": judge_llm,
                    "refresh_episodes": refresh["n"], "refresh_tokens": refresh["tokens"]}}
    (rdir / "selection.json").write_text(json.dumps(out, indent=1))
    return winner, decisions


def account_full(run, t: int, rdir: Path, cands: list) -> None:
    """RRSI's own selection in the units of select_with_evidence: every
    screened candidate's full evolve evaluation is selection cost."""
    fr = run.frontier()
    ep_tok = _ep_tokens(run, fr["incumbent"]["job"])
    rows, n, tok = {}, 0, 0
    for c in cands:
        if not c.commit:
            continue
        cost = _cost(_job_records(run, f"r{t}{c.variant}"))
        rows[c.variant] = {"S": None if c.ev is None else c.ev.S,
                           "dS": None if c.ev is None else c.ev.S - fr["incumbent"]["S"],
                           "fresh": cost}
        n += cost["n"]
        tok += cost["tokens"]
    out = {"t": t, "selection": "full", "rule": "full", "b": None, "ep_tokens": ep_tok,
           "candidates": rows,
           "cost": {"selection_episodes": n, "selection_replays": 0, "selection_tokens": tok,
                    "selection_episode_equivalents": float(n), "judge_llm": {},
                    "refresh_episodes": 0, "refresh_tokens": 0}}
    (rdir / "selection.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------- sequential --
def _seq_candidate(run, t: int, c, inc_ev: EvalResult, S_star: float, delta: float,
                   nu: int, sdir: Path) -> dict:
    """Run candidate c's evolve episodes in batches with the predictive stop.
    Resume-safe: episodes already on disk are reused, and the stopping
    sequence is replayed from the same order. Returns the candidate's record."""
    from .posthoc_e1 import p_admissible
    d, cfg = run.domain, run.cfg
    rp = sdir / f"{c.variant}.json"
    if rp.exists():
        rec = json.loads(rp.read_text())
        if rec.get("finished"):
            return rec
    job = f"r{t}{c.variant}"
    wt = run.wt_root / job
    pairs = [(tid, s) for tid, tr in sorted(inc_ev.per_task.items()) for s in range(len(tr.rewards))]
    random.Random(f"{d.name}:{t}:{c.variant}:seq").shuffle(pairs)
    N = len(pairs)
    cap = float(cfg.notes.get("max_harness_error_rate", 0.02))
    s_inc = inc_ev.S
    floor_var = s_inc * (1 - s_inc)
    L = S_star - delta - s_inc
    cfgd = {"beta0": cfg.beta0, "beta1": cfg.beta1, "w_s": cfg.w_s, "w_c": cfg.w_c, "w_n": cfg.w_n}
    slot, drawn, steps = Counter(), [], []
    n, dropped, why, m, dC, errors = 0, False, "", 0.0, 0.0, 0
    while n < N:
        batch = pairs[n:n + SEQ_BATCH]
        for tid, s in batch:
            drawn.append((tid, s, slot[tid]))
            slot[tid] += 1
        need = {}
        for tid, _, _ in drawn[n:]:
            need[tid] = slot[tid]
        for k in sorted(set(need.values())):
            d.run(wt, run.runs, job, sorted(x for x in need if need[x] == k), k, log_prefix=job)
        n = len(drawn)
        recs = [_load(trial_path(run.runs, job, tid, j)) for tid, _, j in drawn]
        diff = [_reward(r) - inc_ev.per_task[tid].rewards[s] for r, (tid, s, _) in zip(recs, drawn)]
        errors = sum(1 for r in recs if r is not None and "harness_error" in r)
        m = float(np.mean(diff))
        sv = float(np.var(diff, ddof=1)) if n > 1 else 0.0
        dC = _rel_cost([_agent_tokens(r) for r in recs],
                       [inc_ev.per_task[tid].tokens[s] if s < len(inc_ev.per_task[tid].tokens) else None
                        for tid, s, _ in drawn])
        step = {"n": n, "m": m, "sd": math.sqrt(sv), "dC": dC, "errors": errors,
                "missing": sum(1 for r in recs if r is None)}
        if errors > cap * N:
            dropped, why = True, (f"harness errors {errors} already exceed {cap:.0%} of the "
                                  f"{N} planned episodes")
        elif n < N:
            p = p_admissible(m, max(sv, floor_var), n, N, delta, L, dC, nu, cfgd)
            step["p_admit"] = p
            if p < SEQ_GAMMA:
                dropped, why = True, (f"P(admitted after all {N} episodes) = {p:.3f} < {SEQ_GAMMA}")
        steps.append(step)
        if dropped:
            break
    rec = {"variant": c.variant, "N": N, "n": n, "dropped": dropped, "why": why, "m": m,
           "dC": dC, "errors": errors, "nu": nu, "steps": steps, "finished": True}
    rp.write_text(json.dumps(rec, indent=1))
    return rec


def select_sequential(run, t: int, cands: list, inc_ev: EvalResult, rdir: Path, ids: list,
                      counts: dict):
    """Steps 5-6 with sequential full evaluation (selection = seqfull)."""
    from rrsi.selection import select_round
    d, cfg = run.domain, run.cfg
    fr = run.frontier()
    S_star, delta = fr["S_star"], run.delta()
    sdir = rdir / "sequential"
    sdir.mkdir(exist_ok=True)
    live = [c for c in cands if c.gate_failure is None]
    nus = {c.variant: novelty(c.components, counts) for c in live}
    with ThreadPoolExecutor(max_workers=max(1, cfg.eval_parallel)) as ex:
        recs = list(ex.map(lambda c: _seq_candidate(run, t, c, inc_ev, S_star, delta,
                                                    nus[c.variant], sdir), live))
    seq = {c.variant: r for c, r in zip(live, recs)}
    alive = [c for c in live if not seq[c.variant]["dropped"]]
    for c in alive:
        run._evaluate(t, c, rdir, ids)        # every episode is on disk: scores only
    winner, dec_alive = select_round(alive, inc_ev, S_star, delta, cfg, counts,
                                     guard_fn=d.guards)
    by_v = {c.variant: x for c, x in zip(alive, dec_alive)}
    decisions = []
    for c in cands:
        if c.gate_failure is not None:
            decisions.append(Decision(c.variant, False, c.gate_failure))
        elif c.variant in by_v:
            x = by_v[c.variant]
            x.reason = f"[seqfull: all {seq[c.variant]['N']} episodes] {x.reason}"
            decisions.append(x)
        else:
            r = seq[c.variant]
            decisions.append(Decision(
                c.variant, False,
                f"[seqfull] stopped after {r['n']} of {r['N']} evolve episodes: {r['why']}; "
                f"early-stopped estimate dS {r['m']:+.4f}",
                S=inc_ev.S + r["m"], C=None, delta_S=r["m"], delta_C=r["dC"], novelty=r["nu"]))
    ep = sum(r["n"] for r in seq.values())
    out = {"t": t, "selection": cfg.selection, "rule": "seqfull", "gamma": SEQ_GAMMA,
           "batch": SEQ_BATCH, "S_star": S_star, "delta": delta, "S_inc": inc_ev.S,
           "candidates": seq, "winner": None if winner is None else winner.variant,
           "cost": {"selection_episodes": ep, "planned_episodes": sum(r["N"] for r in seq.values()),
                    "selection_replays": 0, "selection_episode_equivalents": float(ep),
                    "judge_llm": {}, "refresh_episodes": 0, "refresh_tokens": 0}}
    (rdir / "selection.json").write_text(json.dumps(out, indent=1))
    return winner, decisions
