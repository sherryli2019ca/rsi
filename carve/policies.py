"""Experiment-selection policies and decision rules.

Every policy sees the same interface:
  select(post, f)  -> (i, j, k, fidelity) or None to stop
  decide(post, f)  -> (accepted bool array (I, J), chosen patch int array (I, J))
"""
from __future__ import annotations

import numpy as np

from .model import FULL, SINGLE, GraphPosterior


def accept_value(p_edge, q_best, f, c_fp):
    """Expected net gain of accepting a cell: repaired failure mass minus regression."""
    return p_edge * f[:, None] * q_best - (1 - p_edge) * c_fp


def bayes_decide(post: GraphPosterior, f, c_fp, resolved=None):
    q = post.q_mean()
    acc = accept_value(post.p_edge(), q.max(-1), f, c_fp) > 0
    if resolved is not None:
        acc &= ~resolved
    return acc, q.argmax(-1)


class CARVE:
    """Cost-aware value-of-information selection (knowledge gradient with
    n-step look-ahead, over cells x patches x fidelities)."""

    name = "CARVE"

    def __init__(self, costs=(1.0, 0.1), c_fp=0.02, horizons=(1, 2, 4, 8),
                 fidelities=(FULL, SINGLE), min_score=1e-6):
        self.costs, self.c_fp, self.horizons = costs, c_fp, horizons
        self.fidelities, self.min_score = fidelities, min_score

    def scores(self, post: GraphPosterior, f, cell=None):
        """Score every (i, j, k, fidelity), or only those of cell=(i, j).

        Cells are independent given the data, so after an observation only the
        observed cell's scores change; select() caches the rest.
        """
        if cell is None:
            q, pe, fw = post.q_mean(), post.p_edge(), f[:, None]
        else:
            i, j = cell
            q = post.q_mean(cell)
            pe = post.p_edge(cell)
            fw = f[i:i + 1, None]
        v_now = np.maximum(accept_value(pe, q.max(-1), fw[:, 0], self.c_fp), 0)[..., None]
        # best quality among the *other* patches, per k
        K = q.shape[-1]
        if K > 1:
            q_other = np.stack([np.delete(q, k, axis=-1).max(-1) for k in range(K)], -1)
        else:
            q_other = np.zeros_like(q)
        best = np.full(q.shape + (len(self.fidelities),), -np.inf)
        for a, fid in enumerate(self.fidelities):
            for n in self.horizons:
                prob, logit, qk = post.lookahead(fid, n, cell)
                pe_new = 1 / (1 + np.exp(-logit))
                q_best_new = np.maximum(qk, q_other[None])
                v_new = np.maximum(pe_new * fw[None, :, :, None] * q_best_new
                                   - (1 - pe_new) * self.c_fp, 0)
                kg = (prob * v_new).sum(0) - v_now
                best[..., a] = np.maximum(best[..., a], kg / (n * self.costs[fid]))
        return best

    def select(self, post, f):
        cache = getattr(self, "_cache", None)
        if cache is None or cache[0] is not post or not np.array_equal(cache[1], f):
            s = self.scores(post, f)
        else:
            s = cache[2]
            i, j = cache[3]
            s[i, j] = self.scores(post, f, (i, j))[0, 0]
        blocked = getattr(self, "blocked", None)
        masked = s if blocked is None else np.where(blocked[..., None, None], -np.inf, s)
        idx = np.unravel_index(np.argmax(masked), s.shape)
        if masked[idx] < self.min_score:
            self._cache = None
            return None
        i, j, k, a = idx
        self._cache = (post, f.copy(), s, (int(i), int(j)))
        return int(i), int(j), int(k), self.fidelities[a]

    def decide(self, post, f):
        return bayes_decide(post, f, self.c_fp)


class UncertaintySampling:
    """Active baseline: replay the cell with the largest frequency-weighted
    posterior uncertainty, using the currently best patch, full replay only."""

    name = "Uncertainty"

    def __init__(self, c_fp=0.02, min_risk=1e-4):
        self.c_fp, self.min_risk = c_fp, min_risk

    def select(self, post, f):
        pe = post.p_edge()
        risk = f[:, None] * np.minimum(pe, 1 - pe)
        if getattr(self, "blocked", None) is not None:
            risk = np.where(self.blocked, 0.0, risk)
        i, j = np.unravel_index(np.argmax(risk), risk.shape)
        if risk[i, j] < self.min_risk:
            return None
        k = int(post.best_patch()[i, j])
        # explore an untried patch while the current one has few observations
        n = post.n_obs[i, j, :, FULL]
        if n[k] >= 3 and (n == 0).any():
            k = int(np.argmax(n == 0))
        return int(i), int(j), k, FULL

    def decide(self, post, f):
        return bayes_decide(post, f, self.c_fp)


class UncertaintyMF(UncertaintySampling):
    """Uncertainty sampling with access to single-step checks: each (cell, patch)
    first gets `n_check` cheap checks, after which full replays confirm."""

    name = "Uncertainty-MF"

    def __init__(self, c_fp=0.02, n_check=3, min_risk=1e-4):
        super().__init__(c_fp, min_risk)
        self.n_check = n_check

    def select(self, post, f):
        act = super().select(post, f)
        if act is None:
            return None
        i, j, k, _ = act
        fid = SINGLE if post.n_obs[i, j, k, SINGLE] < self.n_check else FULL
        return i, j, k, fid


class SequentialEach:
    """Do-then-verify with adaptive stopping: hypotheses in order of attribution
    count, each replayed (cycling through patches) until CARVE's own posterior
    leaves [lo, hi]; full replays only."""

    name = "Sequential-each"

    def __init__(self, counts, c_fp=0.02, lo=0.05, hi=0.95, max_per=12):
        self.c_fp, self.lo, self.hi, self.max_per = c_fp, lo, hi, max_per
        order = np.argsort(-counts, axis=None, kind="stable")
        self.queue = [np.unravel_index(o, counts.shape) for o in order if counts.flat[o] > 0]

    def select(self, post, f):
        pe = post.p_edge()
        while self.queue:
            i, j = self.queue[0]
            n = post.n_obs[i, j, :, FULL]
            blocked = getattr(self, "blocked", None)
            if (blocked is None or not blocked[i, j]) and self.lo < pe[i, j] < self.hi \
                    and n.sum() < self.max_per:
                return int(i), int(j), int(np.argmin(n)), FULL
            self.queue.pop(0)
        return None

    def decide(self, post, f):
        return bayes_decide(post, f, self.c_fp)


class ThompsonSampling:
    """Bandit baseline: sample (e, q) from the posterior and replay the cell and
    patch with the largest sampled frequency-weighted repair, among cells whose
    edge is not yet resolved; full replay only."""

    name = "Thompson"

    def __init__(self, c_fp=0.02, resolve=0.02, seed=0):
        self.c_fp, self.resolve = c_fp, resolve
        self.rng = np.random.default_rng(seed)

    def select(self, post, f):
        pe = post.p_edge()
        open_ = (pe > self.resolve) & (pe < 1 - self.resolve)
        if getattr(self, "blocked", None) is not None:
            open_ &= ~self.blocked
        if not open_.any():
            return None
        e = self.rng.random(pe.shape) < pe
        cdf = np.cumsum(post.W, -1)
        u = self.rng.random(cdf.shape[:-1] + (1,))
        q = post.grid[np.minimum((cdf < u).sum(-1), len(post.grid) - 1)]
        val = np.where(open_[..., None], f[:, None, None] * e[..., None] * q, -1.0)
        # ties (all sampled e=0) are broken by uncertainty
        val += 1e-6 * np.minimum(pe, 1 - pe)[..., None]
        i, j, k = np.unravel_index(np.argmax(val), val.shape)
        return int(i), int(j), int(k), FULL

    def decide(self, post, f):
        return bayes_decide(post, f, self.c_fp)


class LLMOnly:
    """No interventions: trust the LLM attributions (AdaMAST-style). Accept the
    top attributed component of each category plus any component holding at
    least `share` of its attributions; always use the first patch."""

    name = "LLM-only"

    def __init__(self, counts, share=0.4):
        self.counts, self.share = counts, share

    def select(self, post, f):
        return None

    def decide(self, post, f):
        c = self.counts
        tot = c.sum(1, keepdims=True)
        frac = np.divide(c, tot, out=np.zeros_like(c, dtype=float), where=tot > 0)
        acc = frac >= self.share
        top = c.argmax(1)
        acc[np.arange(len(c)), top] |= tot[:, 0] > 0
        return acc, np.zeros(c.shape, dtype=int)


class ReplayEach:
    """Verify LLM hypotheses one by one with a fixed number of full replays of
    the first patch (DoVer / REFLECT-style do-then-verify). Hypotheses are
    tested in order of attribution count; a hypothesis is accepted when at least
    `need` of `reps` replays succeed. Untested hypotheses fall back to LLMOnly."""

    name = "Replay-each"

    def __init__(self, counts, reps=3, need=2, patches=1):
        self.counts, self.reps, self.need, self.patches = counts, reps, need, patches
        order = np.argsort(-counts, axis=None, kind="stable")
        self.queue = [np.unravel_index(o, counts.shape) for o in order
                      if counts.flat[o] > 0]
        self.succ = np.zeros(counts.shape + (patches,), dtype=int)
        self.fallback = LLMOnly(counts)

    def select(self, post, f):
        while self.queue:
            i, j = self.queue[0]
            blocked = getattr(self, "blocked", None)
            if blocked is None or not blocked[i, j]:
                for k in range(self.patches):
                    if post.n_obs[i, j, k, FULL] < self.reps:
                        return int(i), int(j), k, FULL
            self.queue.pop(0)
        return None

    def decide(self, post, f):
        acc, patch = self.fallback.decide(post, f)
        n = post.n_obs[:, :, :self.patches, FULL]
        tested = (n >= self.reps).all(-1)
        best = self.succ.max(-1)
        acc[tested] = best[tested] >= self.need
        patch = np.where(tested, self.succ.argmax(-1), patch)
        return acc, patch

    def record(self, i, j, k, fidelity, outcome):
        self.succ[i, j, k] += outcome
