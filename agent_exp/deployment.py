"""Deployment gains of whole patch sets, their running cost, and what the
verification signals predict about them (restructured analysis; no new agent
runs).

  sets      held-out success of every patch set (budget 10, 5 policy seeds) in
            absolute terms, paired differences from Apply attributed and from
            the unpatched agent, task-clustered 90% intervals; variance split
            into policy seeds, tasks and run-to-run noise
  cost      running cost of the step-limit patches and of every patch on solved
            tasks (task-matched regression-run tokens), episodes truncated by
            the step-limit fault (training traces), and each set's estimated
            change in running cost per deployed episode
  payback   Net(N) = N (v dp - dc_run) - C_verify - dC_change relative to Apply
            attributed: the horizon N* after which verification has paid back,
            at the point estimate and the upper 90% bound of dp
  transfer  each set's deployment gain predicted from its patches' bank replays
            (net of spurious recovery) and from judge checks, against the
            observed held-out gain; leave-one-out contributions of single
            patches against their predicted marginal gain

  python -m agent_exp.deployment --out runs/tau2_retail --domain tau2_retail
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from agent_exp import reanalysis as RA
from agent_exp.reanalysis import CTX
from agent_exp.review2_runs import heldout_episodes

Z90 = 1.645


def clustered(d, tasks):
    """Mean of per-episode differences d and its task-clustered standard error."""
    m = float(d.mean())
    sums = np.array([d[tasks == u].sum() for u in np.unique(tasks)])
    cnt = np.array([(tasks == u).sum() for u in np.unique(tasks)])
    return m, float(np.sqrt(((sums - m * cnt) ** 2).sum()) / len(d))


def cells_of(ps, patches, comp_ids):
    """Bank cells (i, j, k) whose patch text a deployed set uses."""
    out = []
    for comp, text in ps.items():
        j = comp_ids.index(comp)
        for cell, texts in patches.items():
            i, jj = map(int, cell.split(","))
            if jj == j and text in texts:
                out.append((i, j, texts.index(text)))
    return out


def predicted(cells, rate, f, F):
    """Failure mass a set repairs if every cell's rate is its repair rate:
    F sum_i f_i [1 - prod_k (1 - q_k)]."""
    by = {}
    for i, j, k in cells:
        by.setdefault(i, []).append(rate[i, j, k])
    return float(F * sum(f[i] * (1 - np.prod([1 - q for q in qs])) for i, qs in by.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", required=True)
    args = ap.parse_args()
    CTX.update(RA.load(args.out, args.domain))
    D, bank, f = CTX["D"], CTX["bank"], CTX["f"]
    I, J = CTX["I"], CTX["J"]
    H = RA._load(os.path.join(args.out, "heldout.json"))
    patches = RA._load(os.path.join(args.out, "patches.json"))
    dec = RA._load(os.path.join(args.out, "reanalysis_decisions.json"))
    trials = len(H["runs"]["none"]) // len(D.split["test"])
    eps = heldout_episodes(D, trials)
    tasks = np.array([t for t, _, _ in eps])
    var = np.array([str(v) for _, v, _ in eps])
    groups = {"all": var == var, "clean": var == "None", "no_step_fault": var != "F_steps",
              "step_fault": var == "F_steps"}
    run = lambda key: np.array(H["runs"][key], float)
    U = run("none")
    sets = {}
    for k, key in H["sets"].items():
        m, B, s = k.rsplit("|", 2)
        if B == "10":
            sets.setdefault(m, []).append(key)
    A = np.mean([run(k) for k in sets["LLM-only"]], 0)
    F_ho = 1 - U.mean()
    out = {"n": {g: int(v.sum()) for g, v in groups.items()}, "F_heldout": float(F_ho),
           "arms": {}}

    # ---- running cost ---------------------------------------------------------------
    data = CTX["data"]
    tok_ep = float(np.mean([t["tokens"] for t in data]))
    by_var = {}
    for t in data:
        by_var.setdefault(str(t["variant"]), []).append(t)
    trace_stats = {v: {"n": len(ts), "success": float(np.mean([t["reward"] for t in ts])),
                       "tokens": float(np.mean([t["tokens"] for t in ts])),
                       "steps": float(np.mean([len(t["steps"]) for t in ts])),
                       "calls": float(np.mean([len(t["steps"]) + sum(1 for s in t["steps"] if s.get("user"))
                                               for t in ts])),
                       "hit_cap": float(np.mean([t["stopped"] == "max_steps" for t in ts])),
                       "max_steps": int(max(len(t["steps"]) for t in ts))}
                   for v, ts in by_var.items()}
    step = CTX["step"]
    # token effect of every patch on solved tasks: its regression runs against the
    # runs of all other patches on the same task
    bytask = {}
    for key, runs in bank["reg"].items():
        for r in runs:
            bytask.setdefault(r["task"], []).append((key, r["tok"]))
    tok_reg = float(np.mean([r["tok"] for v in bank["reg"].values() for r in v]))
    eff = {}
    for key, runs in bank["reg"].items():
        dd = [r["tok"] - np.mean([t for kk, t in bytask[r["task"]] if kk != key])
              for r in runs if len(bytask[r["task"]]) > 1]
        if dd:
            eff[key] = (float(np.mean(dd)) / tok_reg, float(np.std(dd) / np.sqrt(len(dd))) / tok_reg)
    step_eff = [eff[k] for k in eff if int(k.split(",")[1]) == step]
    pooled_step = [r["tok"] - np.mean([t for kk, t in bytask[r["task"]] if int(kk.split(",")[1]) != step])
                   for key, runs in bank["reg"].items() if int(key.split(",")[1]) == step
                   for r in runs]
    clean = trace_stats["None"]
    sf = trace_stats.get("F_steps")
    out["cost"] = {
        "traces": trace_stats, "tok_episode": tok_ep, "tok_regression_run": tok_reg,
        "step_patch_on_solved": [float(np.mean(pooled_step) / tok_reg),
                                 float(np.std(pooled_step) / np.sqrt(len(pooled_step)) / tok_reg)],
        "all_patches_on_solved_sd": float(np.std([e[0] for e in eff.values()])),
        # restoring a truncated step-fault episode to ordinary length
        "step_restore_per_fault_episode": (clean["tokens"] - sf["tokens"]) / tok_ep if sf else 0.0,
        "step_fault_share_heldout": float(groups["step_fault"].mean()),
    }

    def run_cost(ps_list):
        """Estimated change in running cost per deployed episode against the
        unpatched agent, in episodes: restored step-fault episodes plus the token
        effects of the set's patches on the other episodes."""
        vals, ses = [], []
        sh = out["cost"]["step_fault_share_heldout"]
        for ps in ps_list:
            c = cells_of(ps, patches, D.comp_ids)
            has_step = "CFG.max_steps" in ps
            e = [eff.get(f"{i},{j},{k}", (0.0, 0.0)) for i, j, k in c if j != step]
            vals.append(sh * out["cost"]["step_restore_per_fault_episode"] * has_step +
                        (1 - sh) * sum(x[0] for x in e) * tok_reg / tok_ep)
            ses.append((1 - sh) * np.sqrt(sum(x[1] ** 2 for x in e)) * tok_reg / tok_ep)
        return float(np.mean(vals)), float(np.mean(ses))

    # ---- arms -----------------------------------------------------------------------
    def arm(keys, name):
        Y = np.array([run(k) for k in keys])
        y = Y.mean(0)
        ps = [json.loads(k) if k != "none" else {} for k in keys]
        r = {"n_patches": float(np.mean([len(p) for p in ps])),
             "has_step": float(np.mean(["CFG.max_steps" in p for p in ps])),
             "n_distinct_sets": len(set(keys))}
        r["run_cost"], r["run_cost_se"] = run_cost(ps)
        spent = [v[1] for kk, v in dec.items() if kk.startswith(f"main|{name}|10|")]
        r["C_verify"] = float(np.mean(spent)) if spent else 0.0
        for g, sel in groups.items():
            r[g] = float(y[sel].mean())
            for lab, base in (("vs_apply", A), ("vs_unpatched", U)):
                m, se = clustered(y[sel] - base[sel], tasks[sel])
                r[f"{lab}|{g}"] = [m, se, m - Z90 * se, m + Z90 * se]
        return r

    out["arms"]["unpatched"] = arm(["none"], "none")
    for m, keys in sets.items():
        out["arms"][m] = arm(keys, m)

    # ---- variance components ----------------------------------------------------------
    # per-episode differences of every verified seed set from Apply attributed:
    # run-to-run (within task), between tasks, between policy seeds
    sw, dfw, tm, seedm = 0.0, 0, [], []
    for m, keys in sets.items():
        if m == "LLM-only":
            continue
        for key in keys:
            d = run(key) - A
            seedm.append((m, d.mean()))
            for u in np.unique(tasks):
                x = d[tasks == u]
                sw += ((x - x.mean()) ** 2).sum()
                dfw += len(x) - 1
                tm.append(x.mean())
    sw2 = sw / dfw
    sb2 = max(0.0, float(np.var(tm, ddof=1)) - sw2 / trials)
    sv = []
    for m in {mm for mm, _ in seedm}:
        v = [x for mm, x in seedm if mm == m]
        if len(v) > 1:
            sv.append(np.var(v, ddof=1))
    noise = (sw2 / trials + sb2) / len(np.unique(tasks))
    out["variance"] = {"run": float(sw2), "task": sb2,
                       "seed": float(max(0.0, np.mean(sv) - noise)), "seed_raw": float(np.mean(sv)),
                       "seed_noise_expected": float(noise)}
    # clean episodes only
    sw, dfw, tm = 0.0, 0, []
    sel = groups["clean"]
    for m, keys in sets.items():
        if m == "LLM-only":
            continue
        for key in keys:
            d = (run(key) - A)[sel]
            for u in np.unique(tasks[sel]):
                x = d[tasks[sel] == u]
                if len(x) > 1:
                    sw += ((x - x.mean()) ** 2).sum()
                    dfw += len(x) - 1
                tm.append((x.mean(), len(x)))
    sw2c = sw / max(dfw, 1)
    out["variance"]["run_clean"] = float(sw2c)
    out["variance"]["task_clean"] = float(max(0.0, np.var([x for x, _ in tm], ddof=1)
                                              - sw2c * np.mean([1 / n for _, n in tm])))

    # ---- payback ------------------------------------------------------------------------
    a = out["arms"]["LLM-only"]
    pay = {}
    for m, r in out["arms"].items():
        if m in ("LLM-only", "unpatched"):
            continue
        dp, _, lo, hi = r["vs_apply|all"]
        dc = r["run_cost"] - a["run_cost"]
        dn = r["n_patches"] - a["n_patches"]
        res = {"dp": dp, "lo": lo, "hi": hi, "C": r["C_verify"], "dc_run": dc, "dn": dn}
        for v in (1, 10, 100):
            for lab, g in (("point", dp), ("hi", hi)):
                for kappa in (0.0, 1.0):
                    rate = v * g - dc
                    cost = r["C_verify"] + kappa * dn
                    res[f"N|v{v}|{lab}|k{kappa:g}"] = float(cost / rate) if rate > 0 else None
        pay[m] = res
    out["payback"] = pay

    # ---- transfer -----------------------------------------------------------------------
    _, E, Q, b = RA.ground_truth(bank, I, J)
    nul = {}
    for v in bank["null"].values():
        nul.setdefault(v["cat"], []).append(v["y"])
    rate_rep = np.zeros((I, J, 3))
    rate_judge = np.zeros((I, J, 3))
    zs = np.array([np.mean(c["z"]) for c in bank["cells"].values()])
    ys = np.array([np.mean(c["y"]) for c in bank["cells"].values()])
    # judge-implied repair: invert z = eps + (s - eps) p with the patch-level fit
    slope = np.cov(ys, zs)[0, 1] / max(1e-9, ys.var(ddof=1) - np.mean(ys * (1 - ys) / 19))
    eps_ = zs.mean() - slope * ys.mean()
    for key, c in bank["cells"].items():
        i, j, k = map(int, key.split(","))
        p = np.mean(c["y"])
        rate_rep[i, j, k] = max(0.0, (p - b[i]) / max(1e-9, 1 - b[i]))
        pz = (np.mean(c["z"]) - eps_) / max(1e-9, slope)
        rate_judge[i, j, k] = float(np.clip((pz - b[i]) / max(1e-9, 1 - b[i]), 0, 1))
    points = []
    for m, keys in sets.items():
        for s, key in enumerate(keys):
            ps = json.loads(key) if key != "none" else {}
            c = cells_of(ps, patches, D.comp_ids)
            obs, se = clustered(run(key) - U, tasks)
            obs_ns, se_ns = clustered((run(key) - U)[groups["no_step_fault"]], tasks[groups["no_step_fault"]])
            points.append({"method": m, "seed": s, "has_step": "CFG.max_steps" in ps,
                           "n_patches": len(ps), "obs": obs, "se": se,
                           "obs_nostepfault": obs_ns, "se_nostepfault": se_ns,
                           "pred_replay": predicted(c, rate_rep, f, F_ho),
                           "pred_judge": predicted(c, rate_judge, f, F_ho),
                           "pred_edges": predicted([x for x in c if E[x[0], x[1]]], Q, f, F_ho)})
    out["transfer"] = {"points": points, "judge_fit": [float(eps_ + slope), float(eps_)]}
    xs = np.array([p["pred_replay"] for p in points])
    yo = np.array([p["obs"] for p in points])
    out["transfer"]["corr_replay"] = float(np.corrcoef(xs, yo)[0, 1]) if xs.std() > 0 else None
    xj = np.array([p["pred_judge"] for p in points])
    out["transfer"]["corr_judge"] = float(np.corrcoef(xj, yo)[0, 1]) if xj.std() > 0 else None
    # single patches: leave-one-out contribution on held-out against the predicted
    # marginal gain of the patch within the attributed set
    if H.get("loo"):
        aset = json.loads(sets["LLM-only"][0])
        ca = cells_of(aset, patches, D.comp_ids)
        full = predicted(ca, rate_rep, f, F_ho)
        loo = []
        for lab, key in H["loo"].items():
            comp = lab[1:]
            rest = [x for x in ca if D.comp_ids[x[1]] != comp]
            m_, se_ = clustered(A - run(key), tasks)
            loo.append({"component": comp, "obs": m_, "se": se_,
                        "pred_replay": full - predicted(rest, rate_rep, f, F_ho),
                        "pred_judge": predicted(ca, rate_judge, f, F_ho) -
                        predicted(rest, rate_judge, f, F_ho)})
        out["transfer"]["loo"] = loo
    RA._save(os.path.join(args.out, "deployment.json"), out)
    for m, r in out["arms"].items():
        print(f"{m:20s} all {r['all']:.3f} clean {r['clean']:.3f} noSF {r['no_step_fault']:.3f} "
              f"vsA {r['vs_apply|all'][0]:+.3f} [{r['vs_apply|all'][2]:+.3f},{r['vs_apply|all'][3]:+.3f}] "
              f"vsU {r['vs_unpatched|all'][0]:+.3f} #p {r['n_patches']:.1f} step {r['has_step']:.1f} "
              f"C {r['C_verify']:.1f} run {r['run_cost']:+.3f}+-{r['run_cost_se']:.3f}")
    print(json.dumps({k: v for k, v in out["cost"].items() if k != "traces"}, indent=0))
    print(json.dumps(out["variance"]))
    print("transfer corr", out["transfer"]["corr_replay"], out["transfer"]["corr_judge"])
    for p in points:
        print(p["method"], p["seed"], p["has_step"], round(p["pred_replay"], 3), round(p["pred_judge"], 3),
              round(p["obs"], 3), round(p["obs_nostepfault"], 3))
    print(json.dumps(out["transfer"].get("loo"), indent=0))
    for m, r in out["payback"].items():
        print(m, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                  if k in ("dp", "hi", "C", "dc_run", "dn") or (k.endswith("k0") and "v1|" in k or "v10|" in k)})


if __name__ == "__main__":
    main()
