"""Fifth-review reanalysis of the tau2 banks (no new agent runs).

  binom     primary harm model: task-level binomial hierarchical model with task
            effects shared by the unpatched baseline and every patch, and
            uncertainty in the between-patch spread (agent_exp/harm_binom.py);
            two chains, R-hat, posterior predictive check
  scoring   every policy's deployable sets scored under it: on the full bank,
            on the outcome-split halves (split{t}), on the source-split halves
            (clean{t}: no shared trajectory or regression task, audited analyst
            accuracy, costs and judge calibration from the policy's half), and
            on the full bank with the audited analyst accuracy (audit)
  vpi       value of perfect information over Apply attributed and the Bayes
            default under the binomial model
  payback   probability that verification pays back, under several gain models
            (normal with flat prior, Student t, sceptical priors, task bootstrap)
            and change-cost distributions
  judges    break-even ratios with calibration uncertainty (bootstrap over
            source trajectories and null replays), per category, with judge
            accuracy defined against a patch's replay success rate instead of
            single replays, and under relative prices of the two models; call
            counts as a proxy for latency

  python -m agent_exp.reanalysis5 --out runs/tau2_retail --domain tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.special import expit

from agent_exp import harm_binom as HB
from agent_exp import reanalysis as RA
from agent_exp.reanalysis import CTX, boot_bank, split_bank, split_bank_grouped, truth, yardstick
from agent_exp.reanalysis3 import SHOW, bayes_default, deploy, groups_of, value, vpi3

BUDGETS = (10, 40, 80)


# ---- binomial harm model -----------------------------------------------------------
def tables(bank):
    """(P, T) runs and failures per patch and regression task, P = I*J*3."""
    tasks = CTX["base_tasks"]
    col = {t: n for n, t in enumerate(tasks)}
    I, J = CTX["I"], CTX["J"]
    N = np.zeros((I * J * 3, len(tasks)))
    NF = np.zeros_like(N)
    for key, runs in bank["reg"].items():
        i, j, k = map(int, key.split(","))
        p = (i * J + j) * 3 + k
        for r in runs:
            N[p, col[r["task"]]] += 1
            NF[p, col[r["task"]]] += 1 - r["y"]
    return N, NF


def solved_weights():
    """Weight of each regression task: the number of solved clean training
    episodes of that task, i.e. its share of the frame regression tasks were
    drawn from."""
    n = {}
    for d in CTX["data"]:
        if d["reward"] and d["variant"] is None:
            n[str(d["task_id"])] = n.get(str(d["task_id"]), 0) + 1
    return np.array([n.get(str(t), 0) for t in CTX["base_tasks"]], float)


class BinomHarm:
    """Posterior draws of rho (I, J, 3) and of the mean unpatched failure rate."""

    def __init__(self, bank, seeds=(0, 1, 2, 3), iters=20000, burn=5000, thin=25,
                 slope=True, weighted=True):
        N, NF = tables(bank)
        self.has = N.sum(1) > 0
        n0, f0 = CTX["base_n"], CTX["base_nf"]
        chains = [HB.sample(N[self.has], NF[self.has], n0, f0, iters, burn, thin, seed=s,
                            slope=slope)
                  for s in seeds]
        self.chains = chains
        self.wt = solved_weights() if weighted else None
        self.draws = {k: np.concatenate([c[k] for c in chains]) for k in chains[0]}
        rng = np.random.default_rng(7)
        rho, base = HB.rho_draws(self.draws, self.wt)
        S = len(base)
        full = np.zeros((S, len(self.has)))
        full[:, self.has] = rho
        for p in np.nonzero(~self.has)[0]:
            full[:, p] = HB.new_patch_rho(self.draws, rng, self.wt)
        self.rho = full.reshape(S, CTX["I"], CTX["J"], 3)
        self.base = base
        self.N, self.NF = N, NF

    def rhat(self):
        """Split-free Gelman-Rubin R-hat over the chains for mu, sd and pooled rho."""
        out = {}
        for name, f in (("mu", lambda c: c["mu"]), ("sd", lambda c: c["sd"]),
                        ("beta", lambda c: c["beta"]),
                        ("pooled_rho", lambda c: HB.rho_draws(c, self.wt)[0].mean(1))):
            x = np.array([f(c) for c in self.chains])
            m, n = x.shape
            B = n * x.mean(1).var(ddof=1)
            W = x.var(1, ddof=1).mean()
            out[name] = float(np.sqrt(((n - 1) / n * W + B / n) / W)) if W > 0 else 1.0
        return out

    def summary(self):
        q = lambda v: [float(np.mean(v)), float(np.quantile(v, .05)), float(np.quantile(v, .95))]
        pooled = self.rho[:, self.has.reshape(self.rho.shape[1:])].mean(1)
        spread = self.rho[:, self.has.reshape(self.rho.shape[1:])].std(1)
        return {"mu_logit": q(self.draws["mu"]), "sd_logit": q(self.draws["sd"]),
                "beta": q(self.draws["beta"]),
                "pooled_rho": q(pooled), "spread_rho": q(spread), "base": q(self.base),
                "rhat": self.rhat(), "n_draws": len(self.base)}


def score(dec, name, bank_truth, harm, w, o, n_draw=300, n_seed=40, with_vpi=False,
          seed=1):
    """Posterior value of each policy's deployable sets (fractions of the fixed
    yardstick o), harm drawn from `harm`, edges and repair rates from a
    bootstrap of `bank_truth`'s replays."""
    f, inb = CTX["f"], CTX["inb"]
    I, J = inb.shape
    rng = np.random.default_rng(seed)
    groups = groups_of(dec, name, BUDGETS, n_seed, f, True)
    if not groups:
        return None
    defaults = {}
    if with_vpi:
        for c in (0.0, 0.02):
            defaults[c] = {"apply": groups_of(dec, "main", (10,), 1, f, True)["LLM-only|10"][0],
                           "bayes": deploy(bayes_default(c, "wrong"), f)}
            groups[f"Bayes-default-{c}|0"] = [defaults[c]["bayes"]]
    vals = {c: {g: [] for g in groups} for c in (0.0, 0.02)}
    vp = {}
    S = len(harm.base)
    for _ in range(n_draw):
        s = int(rng.integers(S))
        R, cap = w * harm.rho[s], w * float(harm.base[s])
        Tb = truth(boot_bank(bank_truth, rng), I, J, CTX["ev"]["r0"], w)
        for c in vals:
            for g, sets in groups.items():
                vals[c][g].append(np.mean([value(cs, Tb, f, R, c, "wrong", cap) for cs in sets]) / o)
            for dn, dc in defaults.get(c, {}).items():
                for k, v in vpi3(dc, Tb, f, R, c, "wrong", inb, cap).items():
                    vp.setdefault(f"{c}|{dn}", {}).setdefault(k, []).append(v / o)
    q = lambda v: [float(np.mean(v)), float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]
    out = {"vpi": {k: {kk: q(vv) for kk, vv in v.items()} for k, v in vp.items()}}
    for c, vs in vals.items():
        ref = np.array(vs["LLM-only|10"])
        res = {}
        for g, v in vs.items():
            v = np.array(v)
            res[g] = {"mean": float(v.mean()), "lo": float(np.quantile(v, .05)),
                      "hi": float(np.quantile(v, .95)), "p_better": float((v - ref > 0).mean())}
        for B in BUDGETS:
            gs = [g for g in vs if g.endswith(f"|{B}") and not g.startswith("Bayes")]
            M = np.array([vs[g] for g in gs])
            best = np.bincount(M.argmax(0), minlength=len(gs)) / M.shape[1]
            res[f"p_best|{B}"] = {g.split("|")[0]: float(b) for g, b in zip(gs, best)}
        out[f"{c}"] = res
    return out


def pool_scores(parts):
    """Average split results over halvings (means of means, pooled quantiles are
    not kept: report the mean interval)."""
    parts = [p for p in parts if p]
    out = {}
    for c in ("0.0", "0.02"):
        res = {}
        for g in parts[0][c]:
            if g.startswith("p_best"):
                res[g] = {m: float(np.mean([p[c][g].get(m, 0) for p in parts])) for m in parts[0][c][g]}
            else:
                res[g] = {k: float(np.mean([p[c][g][k] for p in parts]))
                          for k in ("mean", "lo", "hi", "p_better")}
        out[c] = res
    return out


# ---- payback -------------------------------------------------------------------------
def heldout_pairs(domain):
    """Per method: per-task mean paired difference (verified - Apply attributed)
    over episodes and seeds, the number of patches, and the episodes spent."""
    from agent_exp.review2_runs import heldout_episodes
    H = RA._load(f"runs/{domain}/heldout.json")
    D = CTX["D"]
    trials = len(H["runs"]["none"]) // len(D.split["test"])
    tasks = np.array([t for t, _, _ in heldout_episodes(D, trials)])
    sets = {}
    for k, key in H["sets"].items():
        m, B, s = k.rsplit("|", 2)
        if B == "10":
            sets.setdefault(m, []).append(key)
    run = lambda keys: np.mean([np.array(H["runs"][k], float) for k in keys], 0)
    apply = run(sets["LLM-only"])
    nA = np.mean([len(json.loads(k)) if k != "none" else 0 for k in sets["LLM-only"]])
    out = {}
    for m, keys in sets.items():
        if m == "LLM-only":
            continue
        d = run(keys) - apply
        ut = np.unique(tasks)
        out[m] = {"task_diff": np.array([d[tasks == u].mean() for u in ut]),
                  "cnt": np.array([(tasks == u).sum() for u in ut]),
                  "dn": nA - np.mean([len(json.loads(k)) if k != "none" else 0 for k in keys])}
    return out


def payback(domain, dec, F_abs, rng, n=20000, horizons=(300, 1000, 10000)):
    pairs = heldout_pairs(domain)
    gain_models = ("normal", "t", "sceptic03", "sceptic05", "bootstrap")
    cost_models = {"none": lambda: np.zeros(n), "U02": lambda: rng.uniform(0, .02, n),
                   "U05": lambda: rng.uniform(0, .05, n), "fixed05": lambda: np.full(n, .05)}
    out = {}
    for m, r in pairs.items():
        td, cnt = r["task_diff"], r["cnt"]
        T = len(td)
        mean = float((td * cnt).sum() / cnt.sum())
        se = float(np.sqrt((((td - mean) * cnt) ** 2).sum()) / cnt.sum())
        spent = [v[1] for k, v in dec.items() if k.startswith(f"main|{m}|10|")]
        C = float(np.mean(spent)) if spent else 10.0
        res = {"mean": mean, "se": se, "C": C, "dn": float(r["dn"])}
        for gm in gain_models:
            if gm == "normal":
                g = mean + se * rng.standard_normal(n)
            elif gm == "t":
                g = mean + se * rng.standard_t(T - 1, n)
            elif gm.startswith("sceptic"):
                tau = {"sceptic03": 0.03, "sceptic05": 0.05}[gm]
                v = 1 / (1 / se ** 2 + 1 / tau ** 2)
                g = v * mean / se ** 2 + np.sqrt(v) * rng.standard_normal(n)
            else:
                idx = rng.integers(T, size=(n, T))
                g = (td[idx] * cnt[idx]).sum(1) / cnt[idx].sum(1)
            for cm, draw in cost_models.items():
                gg = g + draw() * F_abs * r["dn"]
                res[f"{gm}|{cm}"] = {"ever": float((gg > 0).mean()),
                                     **{f"N{N}": float((N * gg > C).mean()) for N in horizons}}
        out[m] = res
    return out


# ---- judges --------------------------------------------------------------------------
def judges(out_dir, rng, n_boot=2000):
    from agent_exp.judge_configs import break_even
    J = RA._load(os.path.join(out_dir, "judge_configs.json"))
    if not J:
        return None
    bank, attrs = CTX["bank"], CTX["attrs"]
    rows = []
    for key, r in J["rows"].items():
        ck, rep = key.rsplit("|", 1)
        c = bank["cells"][ck]
        rows.append(dict(r, cell=ck, cat=int(ck.split(",")[0]), trace=c["trace"][int(rep)]))
    tf = float(np.mean([t for c in bank["cells"].values() for t in c["tok_full"]]))
    I, J_ = CTX["I"], CTX["J"]
    _, E, Q, b = RA.ground_truth(bank, I, J_)
    qmed = float(np.median(Q.max(-1)[E])) if E.any() else 0.5
    nulls = [v for v in bank["null"].values()]
    cfgs = {"bank": ("z_bank", None), "short_pro": ("z_short_pro", "tok_short_pro"),
            "short_flash": ("z_short_flash", "tok_short_flash"),
            "full_flash": ("z_full_flash", "tok_full_flash")}
    regen = np.array([r["tok_regen"] for r in rows])
    out = {"q": qmed}

    def be_of(rs, bb, name):
        zk, _ = cfgs[name]
        rs = [r for r in rs if zk in r]
        y = np.array([r["y"] for r in rs])
        z = np.array([r[zk] for r in rs])
        if (y == 1).sum() == 0 or (y == 0).sum() == 0:
            return None
        return break_even(z[y == 1].mean(), z[y == 0].mean(), bb, qmed)

    def cost_of(rs, name, lam=1.0):
        zk, tk = cfgs[name]
        rs = [r for r in rs if zk in r]
        if name == "bank":
            # the bank judge's tokens include its own regeneration; the judge part
            # is billed at the analyst (pro) price
            reg = np.mean([r["tok_regen"] for r in rs])
            judge = np.mean([r["tok_single_bank"] for r in rs]) - reg
            return (reg + lam * judge) / tf
        reg = np.mean([r["tok_regen"] for r in rs])
        return (reg + (lam if name.endswith("pro") else 1.0) * np.mean([r[tk] for r in rs])) / tf

    bbar = float(np.mean([v["y"] for v in nulls]))
    traces = sorted({r["trace"] for r in rows})
    bytrace = {t: [r for r in rows if r["trace"] == t] for t in traces}
    for name in cfgs:
        be0 = be_of(rows, bbar, name)
        c0 = cost_of(rows, name)
        boots = []
        for _ in range(n_boot):
            tt = rng.choice(len(traces), len(traces))
            rs = [r for x in tt for r in bytrace[traces[x]]]
            nb = [nulls[x] for x in rng.integers(len(nulls), size=len(nulls))]
            bb = float(np.mean([v["y"] for v in nb]))
            be = be_of(rs, bb, name)
            if be is not None:
                boots.append(be / cost_of(rs, name))
        boots = np.array(boots)
        res = {"break_even": be0, "cost": c0, "ratio": be0 / c0,
               "ratio_90": [float(np.quantile(boots, .05)), float(np.quantile(boots, .95))],
               "p_ratio_gt1": float((boots > 1).mean()),
               "cost_by_price": {str(l): cost_of(rows, name, l) for l in (1, 2, 4)}}
        # per category: own spurious recovery and judge accuracy (>= 20 pairs, both outcomes)
        per = {}
        for i in sorted({r["cat"] for r in rows}):
            rs = [r for r in rows if r["cat"] == i and cfgs[name][0] in r]
            bi = [v["y"] for v in nulls if v["cat"] == i]
            if len(rs) < 20 or not bi:
                continue
            be = be_of(rs, float(np.mean(bi)), name)
            if be is not None:
                per[str(i)] = {"n": len(rs), "b": float(np.mean(bi)), "ratio": be / cost_of(rs, name)}
        res["per_category"] = per
        out[name] = res
    # bank judge accuracy against each patch's replay success rate (20 + 20 per patch):
    # mean z_k = eps + (s - eps) * p_k; method of moments corrects for the binomial
    # noise in the estimated p_k
    ybar = np.array([np.mean(c["y"]) for c in bank["cells"].values()])
    zbar = np.array([np.mean(c["z"]) for c in bank["cells"].values()])
    n = np.array([len(c["y"]) for c in bank["cells"].values()])

    def mom(idx):
        yb, zb = ybar[idx], zbar[idx]
        noise = np.mean(yb * (1 - yb) / np.maximum(n[idx] - 1, 1))
        slope = np.cov(yb, zb)[0, 1] / max(1e-9, yb.var(ddof=1) - noise)
        e = zb.mean() - slope * yb.mean()
        return e + slope, e

    s_, eps = mom(np.arange(len(ybar)))
    tsing = float(np.mean([t for c in bank["cells"].values() for t in c["tok_single"]]) / tf)
    bs = []
    for _ in range(n_boot):
        ss, ee = mom(rng.integers(len(ybar), size=len(ybar)))
        nb = [nulls[x] for x in rng.integers(len(nulls), size=len(nulls))]
        bs.append(break_even(np.clip(ss, 0, 1), np.clip(ee, 0, 1),
                             float(np.mean([v["y"] for v in nb])), qmed) / tsing)
    out["patch_level"] = {"sens": float(s_), "fpr": float(eps),
                          "break_even": break_even(s_, eps, bbar, qmed),
                          "cost": tsing}
    out["patch_level"]["ratio"] = out["patch_level"]["break_even"] / tsing
    out["patch_level"]["ratio_90"] = [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))]
    out["patch_level"]["p_ratio_gt1"] = float((np.array(bs) > 1).mean())
    # latency proxy: sequential model calls. A check is one regeneration and one
    # judge call; a replay from the failing step runs the rest of the episode
    # (one agent call per step plus one user-simulator call when it speaks)
    data = {d["uid"]: d for d in CTX["data"]}
    calls = []
    for r in rows:
        d = data[r["trace"]]
        st = max(0, min(attrs[r["trace"]]["step"], len(d["steps"]) - 1))
        rest = d["steps"][st:]
        calls.append(len(rest) + sum(1 for x in rest if x.get("user")))
    # every judge's accuracy against the patch's replay success rate p_k (from its
    # 20 bank replays): z = eps + (s - eps) p_k, errors-in-variables corrected,
    # bootstrapped over patches
    pk = {k: (np.mean(c["y"]), len(c["y"])) for k, c in bank["cells"].items()}

    def mom_pairs(rs, zk):
        x = np.array([pk[r["cell"]][0] for r in rs])
        nn = np.array([pk[r["cell"]][1] for r in rs])
        z = np.array([r[zk] for r in rs], float)
        noise = np.mean(x * (1 - x) / np.maximum(nn - 1, 1))
        slope = np.cov(x, z)[0, 1] / max(1e-9, x.var(ddof=1) - noise)
        e = z.mean() - slope * x.mean()
        return float(np.clip(e + slope, 0, 1)), float(np.clip(e, 0, 1))

    for name, (zk, _) in cfgs.items():
        rs = [r for r in rows if zk in r]
        s1, e1 = mom_pairs(rs, zk)
        c1 = cost_of(rs, name)
        cellsk = sorted({r["cell"] for r in rs})
        bycell = {c: [r for r in rs if r["cell"] == c] for c in cellsk}
        bs = []
        for _ in range(n_boot):
            pick = rng.integers(len(cellsk), size=len(cellsk))
            rr = [r for x in pick for r in bycell[cellsk[x]]]
            if len(rr) < 10:
                continue
            ss, ee = mom_pairs(rr, zk)
            nb = [nulls[x] for x in rng.integers(len(nulls), size=len(nulls))]
            bs.append(break_even(ss, ee, float(np.mean([v["y"] for v in nb])), qmed) / cost_of(rr, name))
        be1 = break_even(s1, e1, bbar, qmed)
        out[name]["patch_level"] = {"sens": s1, "fpr": e1, "break_even": be1, "ratio": be1 / c1,
                                    "ratio_90": [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                                    "p_ratio_gt1": float((np.array(bs) > 1).mean())}
    out["latency_calls"] = {"replay_median": float(np.median(calls)),
                            "replay_iqr": [float(np.quantile(calls, .25)), float(np.quantile(calls, .75))],
                            "check": 2}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--split_draws", type=int, default=100)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    CTX.update(RA.load(args.out, args.domain))
    dec = RA._load(os.path.join(args.out, "reanalysis_decisions.json"))
    path = os.path.join(args.out, "reanalysis5.json")
    rep = RA._load(path) or {}
    ev = CTX["ev"]
    w = ev["w"]
    T = truth(CTX["bank"], CTX["I"], CTX["J"], ev["r0"], w)
    CTX["T"] = T
    o = yardstick(T, CTX["f"])
    rep["yardstick_abs"] = o / (1 + w)
    todo = set(args.only or ["binom", "split", "clean", "audit", "weak", "payback", "judges"])
    H = None
    if "binom" in todo:
        H = BinomHarm(CTX["bank"])
        rep["binom"] = H.summary()
        sel = H.N[H.has].sum(1) >= 20
        rep["binom"]["ppc"] = HB.ppc({k: v for k, v in H.draws.items()},
                                     H.N[H.has], H.NF[H.has], CTX["base_n"],
                                     np.random.default_rng(3), sel, f0=CTX["base_nf"])
        # sensitivity: logit-additive (no slope), and tasks weighted equally
        Ha = BinomHarm(CTX["bank"], slope=False)
        rep["binom_additive"] = Ha.summary()
        rep["binom_additive"]["ppc"] = HB.ppc(Ha.draws, Ha.N[Ha.has], Ha.NF[Ha.has],
                                              CTX["base_n"], np.random.default_rng(3), sel,
                                              f0=CTX["base_nf"])
        rho_u, _ = HB.rho_draws(H.draws, None)
        q = lambda v: [float(np.mean(v)), float(np.quantile(v, .05)), float(np.quantile(v, .95))]
        rep["binom"]["pooled_rho_unweighted"] = q(rho_u.mean(1))
        rep["primary"] = score(dec, "main", CTX["bank"], H, w, o, args.draws, with_vpi=True)
        rep["primary_additive"] = score(dec, "main", CTX["bank"], Ha, w, o, args.draws, with_vpi=True)
        RA._save(path, rep)
        print("binom", json.dumps(rep["binom"]), flush=True)
    for name in ("audit", "weak"):
        if name in todo and any(k.startswith(f"{name}|") for k in dec):
            H = H or BinomHarm(CTX["bank"])
            rep[name] = score(dec, name, CTX["bank"], H, w, o, args.draws)
            RA._save(path, rep)
            print(name, "done", flush=True)
    for kind, splitter, n_seed in (("split", split_bank, 40), ("clean", split_bank_grouped, 20)):
        if kind not in todo:
            continue
        parts = []
        for t in range(RA.N_SPLITS):
            if not any(k.startswith(f"{kind}{t}|") for k in dec):
                continue
            _, Bh = splitter(CTX["bank"], t)
            HBh = BinomHarm(Bh, seeds=(0, 1), iters=12000, burn=3000, thin=30)
            parts.append(score(dec, f"{kind}{t}", Bh, HBh, w, o, args.split_draws, n_seed=n_seed,
                               seed=10 + t))
        if parts:
            rep[kind] = pool_scores(parts)
            rep[kind]["n_halvings"] = len(parts)
            RA._save(path, rep)
            print(kind, "done", len(parts), flush=True)
    if "payback" in todo:
        Hb = RA._load(os.path.join(args.out, "heldout_breakdown.json"))
        F_abs = 1 - Hb["unpatched"]["all"][0]
        rep["payback"] = payback(os.path.basename(args.out.rstrip("/")), dec, F_abs,
                                 np.random.default_rng(9))
        RA._save(path, rep)
    if "judges" in todo:
        rep["judges"] = judges(args.out, np.random.default_rng(4))
        RA._save(path, rep)
    print(json.dumps({k: v for k, v in rep.items() if k in ("binom", "judges")}, indent=1,
                     default=float)[:6000])


if __name__ == "__main__":
    main()
