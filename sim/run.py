"""Run budgeted edge-identification episodes on the synthetic testbed."""
from __future__ import annotations

import numpy as np

from carve.model import FULL, GraphPosterior, ObsParams
from carve.policies import (CARVE, LLMOnly, ReplayEach, SequentialEach, ThompsonSampling,
                            UncertaintyMF, UncertaintySampling)
from sim.testbed import World, WorldConfig, edge_prior


def make_posterior(world: World, counts, rng, uniform_prior=False, K=None,
                   lam=None, audit_n=20, overrides=None):
    cfg = world.cfg
    # r is estimated from a small human audit of LLM attributions
    r_hat = rng.binomial(audit_n, cfg.r) / audit_n
    r_hat = float(np.clip(r_hat + cfg.extra.get("r_bias", 0.0), 0.0, 1.0))
    prior = edge_prior(counts, r_hat)
    if uniform_prior:
        prior = np.full_like(prior, 1.5 / cfg.J)
    params = ObsParams(b=cfg.b, lam=cfg.lam if lam is None else lam,
                       sens=cfg.sens, fpr=cfg.fpr)
    for key, val in (overrides or {}).items():
        setattr(params, key, val)
    return GraphPosterior(prior, cfg.K if K is None else K, params)


def make_policy(name, counts, costs, c_fp):
    if name == "CARVE":
        return CARVE(costs=costs, c_fp=c_fp)
    if name == "CARVE-myopic":
        return CARVE(costs=costs, c_fp=c_fp, horizons=(1,))
    if name == "CARVE-full-only":
        return CARVE(costs=costs, c_fp=c_fp, fidelities=(FULL,))
    if name in ("CARVE-uniform-prior", "CARVE-one-patch", "CARVE-no-label-noise",
                "CARVE-robust-check"):
        return CARVE(costs=costs, c_fp=c_fp)
    if name == "Uncertainty":
        return UncertaintySampling(c_fp=c_fp)
    if name == "LLM-only":
        return LLMOnly(counts)
    if name == "Replay-each":
        return ReplayEach(counts)
    if name == "Replay-each-3patch":
        return ReplayEach(counts, reps=2, need=2, patches=3)
    if name == "Uncertainty-MF":
        return UncertaintyMF(c_fp=c_fp)
    if name == "Sequential-each":
        return SequentialEach(counts, c_fp=c_fp)
    if name == "Thompson":
        return ThompsonSampling(c_fp=c_fp)
    raise ValueError(name)


def episode(cfg: WorldConfig, method: str, seed: int, checkpoints,
            costs=(1.0, 0.1), c_fp=0.02, model_overrides=None):
    """Returns dict checkpoint -> (normalised net gain, edge precision, recall, spent)."""
    rng = np.random.default_rng(seed)
    world = World(cfg, rng)
    counts = world.attributions()
    run_rng = np.random.default_rng(seed + 10_000)
    world.rng = run_rng
    post = make_posterior(
        world, counts, run_rng,
        uniform_prior=method == "CARVE-uniform-prior",
        K=1 if method == "CARVE-one-patch" else None,
        lam=1.0 if method == "CARVE-no-label-noise" else None,
        overrides=dict(model_overrides or {}, **({"fooled_fpr": 0.6}
                       if method == "CARVE-robust-check" else {})))
    pol = make_policy(method, counts, costs, c_fp)
    oracle_gain = world.oracle(c_fp)[0]

    spent, out, cps = 0.0, {}, sorted(checkpoints)
    ci = 0

    def snapshot():
        acc, patch = pol.decide(post, world.f)
        tp = (acc & world.E).sum()
        prec = tp / max(acc.sum(), 1)
        rec = tp / world.E.sum()
        return world.net_gain(acc, patch, c_fp) / oracle_gain, prec, rec, spent

    while ci < len(cps):
        act = pol.select(post, world.f)
        if act is None:
            break
        i, j, k, fid = act
        if spent + costs[fid] > cps[ci] + 1e-9:
            # record every checkpoint this action would cross before paying for it
            while ci < len(cps) and spent + costs[fid] > cps[ci] + 1e-9:
                out[cps[ci]] = snapshot()
                ci += 1
            if ci >= len(cps):
                break
        o = world.intervene(i, j, k, fid)
        post.update(i, j, k, fid, o)
        if hasattr(pol, "record"):
            pol.record(i, j, k, fid, o)
        spent += costs[fid]
    final = snapshot()
    for cp in cps[ci:]:
        out[cp] = final
    return out
