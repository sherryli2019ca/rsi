"""Multi-round RSI loop on the synthetic testbed.

Each round: collect LLM attributions for the current failures (more traces for
more frequent categories), spend a per-round intervention budget, accept edges,
apply the chosen patches, and recompute the failure distribution. Accepted
patches on a wrong component cost a regression of c_fp times the current
failure mass (the intervention paradox).
"""
from __future__ import annotations

import numpy as np

from sim.run import make_policy, make_posterior
from sim.testbed import World, WorldConfig, edge_prior


def rsi_run(cfg: WorldConfig, method: str, seed: int, rounds=6, budget=40.0,
            costs=(1.0, 0.1), c_fp=0.02, attr_per_round=240, base_success=0.4):
    rng = np.random.default_rng(seed)
    world = World(cfg, rng)
    run_rng = np.random.default_rng(seed + 10_000)
    m = (1 - base_success) * world.f            # residual failure mass per category
    reg = 0.0
    applied = np.zeros((cfg.I, cfg.J), dtype=bool)
    counts = np.zeros((cfg.I, cfg.J), dtype=int)
    post, pol = None, None
    r_hat = run_rng.binomial(20, cfg.r) / 20
    history = [(0.0, 1 - m.sum() - reg)]
    spent = 0.0
    for _ in range(rounds):
        f = m / m.sum()
        world.rng = rng
        counts = counts + world.attributions(run_rng.multinomial(attr_per_round, f))
        world.rng = run_rng
        if post is None:
            post = make_posterior(world, counts, run_rng,
                                  uniform_prior=method == "CARVE-uniform-prior",
                                  K=1 if method == "CARVE-one-patch" else None)
        else:
            pe = np.clip(edge_prior(counts, r_hat), 1e-4, 1 - 1e-4)
            if method == "CARVE-uniform-prior":
                pe = np.full_like(pe, 1.5 / cfg.J)
            post.prior_logit = np.log(pe) - np.log1p(-pe)
        pol = make_policy(method, counts, costs, c_fp)
        pol.blocked = applied
        round_spent = 0.0
        while True:
            act = pol.select(post, f)
            if act is None:
                break
            i, j, k, fid = act
            if round_spent + costs[fid] > budget + 1e-9:
                break
            o = world.intervene(i, j, k, fid, f)
            post.update(i, j, k, fid, o)
            if hasattr(pol, "record"):
                pol.record(i, j, k, fid, o)
            round_spent += costs[fid]
        spent += round_spent
        acc, patch = pol.decide(post, f)
        new = acc & ~applied
        n_fp = (new & ~world.E).sum()
        reg += c_fp * m.sum() * n_fp
        for i, j in zip(*np.nonzero(new & world.E)):
            m[i] *= 1 - world.Q[i, j, patch[i, j]]
        applied |= new
        history.append((spent, 1 - m.sum() - reg))
    return history
