"""Pre-registered analysis of experiment E1 (verification evidence on RRSI rounds).

For every settled RRSI round t of a domain we have: the incumbent H_t, its full
evolve evaluation (k trials per task), up to m candidates with their full
evolve evaluations (RRSI's own selection evidence), targeted replays of H_t's
failed trials with every candidate and with H_t itself (null), an LLM-judge
estimate per candidate, and held-out deployment trials of H_t and of every
candidate. This script turns each evidence type at each episode budget into a
selection decision with ONE rule and scores the decision by its held-out
deployment gain.

Evidence estimators of dS = S(candidate) - S(H_t) on the evolve set (b = episode
budget per candidate; null replays are shared by the round's candidates):

  none           accept a measured candidate (each equally likely)
  judge          the judge's dS (no standard error, no cost estimate)
  sample@b       b random (task, trial) pairs of the candidate's full evaluation,
                 paired with H_t's result on the same pair: mean(c - h)
  replay@b       b candidate replays of H_t's failed trials, spread evenly over
                 the replay references: f * fix_c, with f = failed trials / trials
  replaynull@b   b/2 candidate + b/2 null replays: f * (fix_c - fix_null)
  net@b          b/4 candidate + b/4 null replays and b/2 regression pairs (pairs
                 where H_t succeeded; the null is H_t's other trial of the same
                 task): f * (fix_c - fix_null) + (1 - f) * mean(c - h_other)
  net_prior@b    net@b with fix_c shrunk to a Beta prior (mean mu0, 2 pseudo
                 trials), mu0 in {audited, 0.7, 0.2}: sensitivity only
  full           RRSI's own decision on the full evolve evaluation (reference)

Decision rule (RRSI's Algorithm 2 with the evidence's own noise band
delta_e = z * se, z = delta_z of the domain config; the S* floor is replaced
by dS >= -delta_e): a candidate with dS > delta_e is admissible iff
dC <= beta0 + beta1 * dS; one with |dS| <= delta_e iff w_s*dS - w_c*dC + w_n*nu
> 0; one with dS < -delta_e never. The admissible candidate with the largest dS
wins; none admissible keeps H_t. judge: delta_e = 0, dC = 0, nu = 0.

Scores (per round, expectation over M random subsamples of the evidence):
  D(rule) = sum_choice P(choice) * dep(choice), dep(c) = held-out success of c
  minus that of H_t (task-paired, first heldout_k trials of each), dep(none) = 0.

Primary endpoints (pooled over domains, rounds weighted equally):
  P1 transfer: held-out success of the final incumbent minus base (all trials,
     tasks pooled), next to the same difference on the evolve set.
  P2 decision value at b = 40: net@40 - full (non-inferiority, margin 0.75 pp),
     net@40 - none and full - none (superiority); one-sided bootstrap p-values,
     Holm over the three. Bootstrap: rounds and held-out tasks resampled within
     domain, B = 2000.

  python -m verify.analyze --domains tau2_retail,tau2_airline [--out runs/verify]
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from rrsi.components import K_STR
from rrsi.domain import load_domain
from rrsi.evaluate import EvalResult

from .state import ROOT, rounds

BUDGETS = (10, 40, 80)
M = 400
B = 2000
MARGIN = 0.0075
PRIORS = {"audited": {"tau2_retail": 0.41, "tau2_airline": 0.29}, "orig": 0.7, "weak": 0.2}
NET_N = (100, 1000, 10000)       # deployment horizons (episodes) of the secondary Net(N)
NET_V = (1, 10)                  # value of one success, in episodes of running cost


# ----------------------------------------------------------------- loading --
def _ev(path: Path) -> EvalResult | None:
    return EvalResult.load(path) if path.exists() else None


def _replays(d: Path) -> dict:
    """{task_id: [(reward, tokens), ...]} from a replay directory."""
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*__*.json")):
        r = json.loads(p.read_text())
        tok = (r.get("tokens") or {})
        out.setdefault(str(r["task_id"]), []).append(
            (float(r.get("reward") or 0), (tok.get("agent_total") or 0) + (tok.get("user") or 0),
             (r.get("replay") or {}).get("start_effective"), (r.get("replay") or {}).get("start_requested")))
    return out


def _accepted_components(run_dir: Path, t: int) -> dict:
    counts = {}
    hp = run_dir / "history.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("t", 0) < t and row.get("accepted") and row.get("component"):
                counts[row["component"]] = counts.get(row["component"], 0) + 1
    return counts


def load(name: str, runs: Path, out: Path) -> dict:
    dom = load_domain(name)
    run_dir, vdir = Path(runs) / name, Path(out) / name
    k_h = int(dom.cfg.get("heldout_k", 4))
    data = {"name": name, "cfg": dom.cfg, "rounds": [], "heldout": {}, "heldout_C": {}}

    def heldout(commit):
        key = commit[:12]
        if key not in data["heldout"]:
            ev = _ev(vdir / "jobs" / "heldout" / key / "eval.json")
            data["heldout_C"][key] = None if ev is None else ev.C
            data["heldout"][key] = None if ev is None else {
                t: tr.rewards for t, tr in ev.per_task.items()}
        return data["heldout"][key]

    for rnd in rounds(run_dir):
        if not (vdir / f"done_r{rnd.t}").exists():
            continue
        inc = _ev(run_dir / "jobs" / rnd.inc_job / "eval.json")
        refs = json.loads((vdir / "replays" / f"r{rnd.t}" / "refs.json").read_text())
        counts = _accepted_components(run_dir, rnd.t)
        R = {"t": rnd.t, "inc_commit": rnd.inc_commit, "inc": inc, "refs": refs,
             "null": _replays(vdir / "replays" / f"r{rnd.t}" / "null"),
             "accepted": rnd.accepted, "inc_heldout": heldout(rnd.inc_commit),
             "inc_heldout_C": data["heldout_C"].get(rnd.inc_commit[:12]), "cands": {}}
        for c in rnd.cands:
            if not c.measured:
                continue
            jp = vdir / "judge" / f"r{rnd.t}{c.variant}.json"
            comps = {str(e.get("component", "")).lower() for e in c.edits}
            R["cands"][c.variant] = {
                "commit": c.commit, "full": _ev(run_dir / "jobs" / c.job / "eval.json"),
                "replays": _replays(vdir / "replays" / f"r{rnd.t}" / c.variant),
                "judge": json.loads(jp.read_text()).get("dS") if jp.exists() else None,
                "novelty": sum(1 for x in comps if x in K_STR and counts.get(x, 0) == 0),
                "components": sorted(comps), "decision": c.decision,
                "heldout": heldout(c.commit),
                "heldout_C": data["heldout_C"].get(c.commit[:12])}
        data["rounds"].append(R)
    data["k_h"] = k_h
    traj = json.loads((run_dir / "frontier.json").read_text())["trajectory"]
    toks = []
    for p in (run_dir / "jobs" / traj[0]["job"]).glob("s*/*.json"):
        tk = json.loads(p.read_text()).get("tokens") or {}
        if tk.get("agent_total"):
            toks.append(tk["agent_total"] + (tk.get("user") or 0))
    data["ep_tokens"] = float(np.mean(toks)) if toks else 1.0
    data["base"], data["final"] = traj[0], traj[-1]
    return data


# --------------------------------------------------------------- evidence --
def _pairs(inc: EvalResult, cand: EvalResult):
    """[(task, s, c, h, h_other, tok_c, tok_h)] over the evolve set."""
    out = []
    for t, tr in inc.per_task.items():
        ct = cand.per_task.get(t)
        if ct is None:
            continue
        k = min(len(tr.rewards), len(ct.rewards))
        for s in range(k):
            other = [tr.rewards[j] for j in range(len(tr.rewards)) if j != s]
            out.append((t, s, ct.rewards[s], tr.rewards[s],
                        float(np.mean(other)) if other else tr.rewards[s],
                        ct.tokens[s] if s < len(ct.tokens) else None,
                        tr.tokens[s] if s < len(tr.tokens) else None))
    return out


def _rel_cost(tc, th):
    tc = [x for x in tc if x]
    th = [x for x in th if x]
    if not tc or not th:
        return 0.0
    return (np.mean(tc) - np.mean(th)) / np.mean(th)


def _spread(refs, n, rng):
    """n draws spread evenly over refs (each ref once before any ref twice)."""
    order = list(range(len(refs)))
    rng.shuffle(order)
    return [refs[order[i % len(order)]] for i in range(n)] if refs else []


def _fix(replays, picks, rng, used):
    """len(picks) replays drawn in a random order per reference, without
    replacement: [(reward, tokens), ...] (fewer when a reference runs out)."""
    rows = []
    for r in picks:
        pool = replays.get(r["task_id"]) or []
        if r["task_id"] not in used:
            used[r["task_id"]] = rng.sample(range(len(pool)), len(pool))
        order = used[r["task_id"]]
        i = sum(1 for x in rows if x[2] == r["task_id"])
        if i < len(order):
            rows.append((pool[order[i]][0], pool[order[i]][1], r["task_id"]))
    return [(a, b) for a, b, _ in rows]


def estimate(rule: str, b: int, R: dict, v: str, rng, mu0: float | None = None,
             null_vals: list | None = None, ep_tok: float = 1.0):
    """-> (dS, se, dC, episodes, episode_equivalents) for candidate v of round R
    (the shared null replays are not included), or None if not computable."""
    C = R["cands"][v]
    inc = R["inc"]
    n_trials = sum(len(tr.rewards) for tr in inc.per_task.values())
    f = sum(1 for tr in inc.per_task.values() for r in tr.rewards if r < 1) / max(1, n_trials)
    if rule == "judge":
        return None if C["judge"] is None else (C["judge"], 0.0, 0.0, 0, 0.0)
    pairs = _pairs(inc, C["full"])
    if rule == "full":
        d = C["decision"]
        return (d.get("delta_S"), None, d.get("delta_C"), len(pairs), float(len(pairs)))
    if rule == "sample":
        pk = rng.sample(pairs, min(b, len(pairs)))
        diff = [p[2] - p[3] for p in pk]
        se = np.std(diff, ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else 1.0
        return (float(np.mean(diff)), float(se), _rel_cost([p[5] for p in pk], [p[6] for p in pk]),
                len(pk), float(len(pk)))
    # replay-based
    nc = {"replay": b, "replaynull": b // 2, "net": b // 4, "net_prior": b // 4}[rule]
    fc_rows = _fix(C["replays"], _spread(R["refs"], nc, rng), rng, {})
    if not fc_rows:
        return None
    fc = [x[0] for x in fc_rows]
    eq = sum(x[1] for x in fc_rows) / ep_tok
    fix_c = float(np.mean(fc))
    if mu0 is not None:
        fix_c = (sum(fc) + 2 * mu0) / (len(fc) + 2)
    var_c = fix_c * (1 - fix_c) / max(1, len(fc))
    if rule == "replay":
        return (f * fix_c, f * math.sqrt(var_c) or 1e-9, 0.0, len(fc), eq)
    if not null_vals:
        return None
    fn = [x[0] for x in null_vals]
    fix_n = float(np.mean(fn))
    se2 = var_c + fix_n * (1 - fix_n) / len(fn)
    gain = f * (fix_c - fix_n)
    if rule == "replaynull":
        return (gain, f * math.sqrt(se2) or 1e-9, 0.0, len(fc), eq)
    solved = [p for p in pairs if p[3] >= 1]
    pk = rng.sample(solved, min(max(0, b - 2 * nc), len(solved)))
    harm = [p[2] - p[4] for p in pk]
    h = float(np.mean(harm)) if harm else 0.0
    se_h = np.std(harm, ddof=1) / math.sqrt(len(harm)) if len(harm) > 1 else 0.0
    dS = gain + (1 - f) * h
    se = math.sqrt((f ** 2) * se2 + ((1 - f) * se_h) ** 2) or 1e-9
    return (dS, se, _rel_cost([p[5] for p in pk], [p[6] for p in pk]), len(fc) + len(pk),
            eq + len(pk))


def decide(ests: dict, cfg: dict, z: float):
    """RRSI Algorithm 2 on estimates {v: (dS, se, dC, nu)} -> winner or None."""
    best = None
    for v, (dS, se, dC, nu) in ests.items():
        if dS is None:
            continue
        delta = z * (se or 0.0)
        if dS > delta:
            ok = (dC or 0.0) <= cfg["beta0"] + cfg["beta1"] * dS
        elif dS >= -delta and delta > 0:
            ok = cfg["w_s"] * dS - cfg["w_c"] * (dC or 0.0) + cfg["w_n"] * nu > 0
        else:
            ok = False
        if ok and (best is None or dS > ests[best][0]):
            best = v
    return best


def choice_probs(rule: str, b: int, R: dict, cfg: dict, seed: int, mu0=None,
                 ep_tok: float = 1.0) -> tuple[dict, float, float]:
    """-> ({choice: probability}, mean episodes, mean episode-equivalents) per round."""
    vs = list(R["cands"])
    if not vs:
        return {None: 1.0}, 0.0, 0.0
    if rule == "none":
        return {v: 1.0 / len(vs) for v in vs}, 0.0, 0.0
    z = float(cfg.get("delta_z", 2.0))
    draws = 1 if rule in ("judge", "full") else M
    rng = random.Random(seed)
    counts, eps, eqs = {}, 0.0, 0.0
    for _ in range(draws):
        null_vals = None
        if rule in ("replaynull", "net", "net_prior"):
            nn = b // 2 if rule == "replaynull" else b // 4
            null_vals = _fix(R["null"], _spread(R["refs"], nn, rng), rng, {})
            eps += len(null_vals)
            eqs += sum(x[1] for x in null_vals) / ep_tok
        ests = {}
        for v in vs:
            e = estimate(rule, b, R, v, rng, mu0, null_vals, ep_tok)
            if e is not None:
                ests[v] = (e[0], e[1], e[2], R["cands"][v]["novelty"])
                eps += e[3]
                eqs += e[4]
        if rule == "full":
            w = R["accepted"] if R["accepted"] in vs else None
        else:
            w = decide(ests, cfg, z)
        counts[w] = counts.get(w, 0) + 1
    return {k: n / draws for k, n in counts.items()}, eps / draws, eqs / draws


# ---------------------------------------------------------------- scoring --
def task_diffs(a: dict | None, h: dict | None, k: int) -> dict:
    """{task: mean(a) - mean(h)} over the first k trials of each."""
    if not a or not h:
        return {}
    return {t: float(np.mean(a[t][:k]) - np.mean(h[t][:k])) for t in h if t in a}


def rules():
    out = [("none", 0, None), ("judge", 0, None), ("full", 0, None)]
    for b in BUDGETS:
        for r in ("sample", "replay", "replaynull", "net"):
            out.append((r, b, None))
    for p in PRIORS:
        out.append(("net_prior", 40, p))
    return out


def label(rule, b, p):
    return rule if rule in ("none", "judge", "full") else (f"{rule}@{b}" + (f"[{p}]" if p else ""))


def domain_tables(D: dict) -> dict:
    cfg = D["cfg"]
    k = D["k_h"]
    rows = []
    for R in D["rounds"]:
        dep = {v: task_diffs(C["heldout"], R["inc_heldout"], k) for v, C in R["cands"].items()}
        probs, cost = {}, {}
        for rule, b, p in rules():
            mu0 = None
            if p:
                mu0 = PRIORS[p].get(D["name"]) if isinstance(PRIORS[p], dict) else PRIORS[p]
                if mu0 is None:
                    continue      # no audited analyst accuracy for this domain (AppWorld)
            lab = label(rule, b, p)
            probs[lab], ep, eq = choice_probs(rule, b, R, cfg, seed=1000 * R["t"] + b, mu0=mu0,
                                              ep_tok=D["ep_tokens"])
            cost[lab] = {"episodes": ep, "episode_equivalents": eq}
        dcr = {v: (C["heldout_C"] - R["inc_heldout_C"]) / R["inc_heldout_C"]
               if C["heldout_C"] and R["inc_heldout_C"] else 0.0 for v, C in R["cands"].items()}
        rows.append({"t": R["t"], "dep": dep, "dcr": dcr, "probs": probs, "cost": cost,
                     "evolve_dS": {v: C["decision"].get("delta_S") for v, C in R["cands"].items()},
                     "judge": {v: C["judge"] for v, C in R["cands"].items()},
                     "replay_full": {v: estimate("replay", 10 ** 6, R, v, random.Random(0),
                                                 ep_tok=D["ep_tokens"]) for v in R["cands"]},
                     "starts": {v: [(x[2], x[3]) for xs in C["replays"].values() for x in xs]
                                for v, C in R["cands"].items()},
                     "components": {v: C["components"] for v, C in R["cands"].items()}})
    return {"name": D["name"], "rows": rows}


def round_value(row: dict, rule: str, tasks=None) -> float:
    val = 0.0
    for v, p in row["probs"][rule].items():
        if v is None:
            continue
        d = row["dep"].get(v) or {}
        ts = tasks if tasks is not None else list(d)
        vals = [d[t] for t in ts if t in d]
        val += p * (float(np.mean(vals)) if vals else 0.0)
    return val


def decision_values(tables: list[dict], rng: np.random.Generator, boot: bool) -> dict:
    """Mean over all rounds of all domains of each rule's round value."""
    per_rule = {}
    for T in tables:
        rows = T["rows"]
        if not rows:
            continue
        idx = rng.integers(0, len(rows), len(rows)) if boot else range(len(rows))
        all_tasks = sorted({t for r in rows for d in r["dep"].values() for t in d})
        tasks = list(rng.choice(all_tasks, len(all_tasks))) if boot and all_tasks else None
        for i in idx:
            for rule in rows[i]["probs"]:
                per_rule.setdefault(rule, []).append(round_value(rows[i], rule, tasks))
    return {r: float(np.mean(v)) for r, v in per_rule.items()}


def net_values(tables: list[dict]) -> dict:
    """Secondary: Net(N) = N (v dep - dc_run) - C_verify per round, in episodes of
    running cost, mean over the rounds of all domains. dc_run is the chosen
    harness's relative change in policy tokens per held-out episode; C_verify the
    rule's episode equivalents in that round; dC_change = 0 (candidate generation
    is the same for every rule)."""
    out = {}
    for T in tables:
        for row in T["rows"]:
            for rule, probs in row["probs"].items():
                dp = sum(p * float(np.mean(list(row["dep"][c].values())))
                         for c, p in probs.items() if c is not None and row["dep"].get(c))
                drun = sum(p * (row["dcr"].get(c) or 0.0) for c, p in probs.items() if c is not None)
                cv = row["cost"][rule]["episode_equivalents"]
                for N in NET_N:
                    for v in NET_V:
                        out.setdefault(rule, {}).setdefault(f"N={N},v={v}", []).append(
                            N * (v * dp - drun) - cv)
    return {r: {k: float(np.mean(x)) for k, x in d.items()} for r, d in out.items()}


def holm(ps: dict) -> dict:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[k]))
        out[k] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="tau2_retail,tau2_airline")
    ap.add_argument("--runs", default=str(ROOT / "runs" / "rrsi"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "verify"))
    args = ap.parse_args()
    datas = [load(n, Path(args.runs), Path(args.out)) for n in args.domains.split(",") if n]
    tables = [domain_tables(D) for D in datas]
    point = decision_values(tables, np.random.default_rng(0), boot=False)
    rng = np.random.default_rng(1)
    boots = [decision_values(tables, rng, boot=True) for _ in range(B)]

    def contrast(a, b):
        xs = np.array([x[a] - x[b] for x in boots])
        return {"est": point[a] - point[b], "ci90": [float(np.percentile(xs, 5)),
                                                     float(np.percentile(xs, 95))], "boot": xs}

    c1, c2, c3 = contrast("net@40", "full"), contrast("net@40", "none"), contrast("full", "none")
    ps = {"net@40 - full (NI)": float(np.mean(c1["boot"] <= -MARGIN)),
          "net@40 - none": float(np.mean(c2["boot"] <= 0)),
          "full - none": float(np.mean(c3["boot"] <= 0))}
    adj = holm(ps)
    res = {"decision_value": point,
           "decision_value_ci90": {r: [float(np.percentile([x[r] for x in boots], 5)),
                                       float(np.percentile([x[r] for x in boots], 95))]
                                   for r in point},
           "primary_P2": {name: {"est": c["est"], "ci90": c["ci90"], "p_one_sided": ps[name],
                                 "p_holm": adj[name]}
                          for name, c in zip(ps, (c1, c2, c3))},
           "margin": MARGIN, "n_rounds": {T["name"]: len(T["rows"]) for T in tables}}
    # P1 transfer
    p1 = {}
    for D in datas:
        vdir = Path(args.out) / D["name"]
        b_h = _ev(vdir / "jobs" / "heldout" / D["base"]["commit"][:12] / "eval.json")
        f_h = _ev(vdir / "jobs" / "heldout" / D["final"]["commit"][:12] / "eval.json")
        b_e = _ev(Path(args.runs) / D["name"] / "jobs" / D["base"]["job"] / "eval.json")
        f_e = _ev(Path(args.runs) / D["name"] / "jobs" / D["final"]["job"] / "eval.json")
        if b_h and f_h:
            d = task_diffs({t: x.rewards for t, x in f_h.per_task.items()},
                           {t: x.rewards for t, x in b_h.per_task.items()}, 10 ** 6)
            arr = np.array(list(d.values()))
            bs = [np.mean(np.random.default_rng(i).choice(arr, len(arr))) for i in range(B)]
            p1[D["name"]] = {"heldout_gain": float(arr.mean()),
                             "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))],
                             "evolve_gain": (f_e.S - b_e.S) if (b_e and f_e) else None,
                             "task_diffs": d}
    if p1:
        allv = np.concatenate([np.array(list(v["task_diffs"].values())) for v in p1.values()])
        bs = [np.mean(np.random.default_rng(i).choice(allv, len(allv))) for i in range(B)]
        p1["pooled"] = {"heldout_gain": float(allv.mean()),
                        "ci90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}
    res["primary_P1"] = p1
    res["cost_per_round"] = {r: {k: float(np.mean([row["cost"][r][k] for T in tables
                                                    for row in T["rows"]]))
                                 for k in ("episodes", "episode_equivalents")}
                             for r in point if tables and tables[0]["rows"]}
    res["net_N"] = net_values(tables)
    res["tables"] = tables
    dest = Path(args.out) / "e1_analysis.json"
    dest.write_text(json.dumps(res, indent=1, default=lambda o: None))
    print(json.dumps({k: res[k] for k in ("primary_P2", "n_rounds")}, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "task_diffs"}
                      for k, v in p1.items()}, indent=1))


if __name__ == "__main__":
    main()
