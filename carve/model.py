"""Bayesian model over the category -> component causal bipartite graph.

For every (category i, component j) cell we keep a posterior over
  e_ij in {0, 1}   -- whether patching component j can repair failures of category i
  q_ijk in [0, 1]  -- repair probability of candidate patch k when e_ij = 1
Separating e from q is what lets a bad patch be told apart from a wrong component.

Observation model for one intervention on a failure trace labelled with category i,
using patch k on component j:
  success probability   p1(q) = lam * q + (1 - lam) * b    if e_ij = 1
                        p0    = b                          if e_ij = 0
  where b is the spurious recovery rate (stochastic re-roll) and lam the
  accuracy of the LLM category labels used to sample traces.
  Full replay  observes y ~ Bernoulli(p).
  Single-step  observes z with P(z=1 | y=1) = sens, P(z=1 | y=0) = fpr,
               i.e. P(z=1) = sens * p + fpr * (1 - p).

The q-posterior of each (i, j, k) is kept on a fixed grid, so every update and
every one-step look-ahead is a vectorised dot product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FULL, SINGLE = 0, 1


@dataclass
class ObsParams:
    b: float = 0.05          # spurious recovery rate
    lam: float = 0.8         # category-label accuracy
    sens: float = 0.85       # single-step sensitivity
    fpr: float = 0.15        # single-step false-positive rate
    q_a: float = 2.0         # Beta prior on patch quality
    q_b: float = 2.0
    fooled_fpr: float | None = None   # single-step FPR for bad patches (q < bad_q), if modelled
    bad_q: float = 0.3


class GraphPosterior:
    def __init__(self, prior_edge: np.ndarray, n_patches: int, params: ObsParams,
                 grid_size: int = 101):
        self.I, self.J = prior_edge.shape
        self.K = n_patches
        self.p = params
        self.grid = np.linspace(0.0005, 0.9995, grid_size)
        prior_q = self.grid ** (params.q_a - 1) * (1 - self.grid) ** (params.q_b - 1)
        prior_q /= prior_q.sum()
        # normalised q-posterior per (i, j, k) under e=1, and its log evidence
        self.W = np.broadcast_to(prior_q, (self.I, self.J, self.K, grid_size)).copy()
        self.log_ev1 = np.zeros((self.I, self.J, self.K))   # log P(D_k | e=1)
        self.log_ev0 = np.zeros((self.I, self.J, self.K))   # log P(D_k | e=0)
        pe = np.clip(prior_edge, 1e-4, 1 - 1e-4)
        self.prior_logit = np.log(pe) - np.log1p(-pe)
        self.n_obs = np.zeros((self.I, self.J, self.K, 2), dtype=int)

    # ---- likelihood vectors -------------------------------------------------
    def _p1(self) -> np.ndarray:
        return self.p.lam * self.grid + (1 - self.p.lam) * self.p.b

    def lik(self, fidelity: int, outcome: int):
        """Return (likelihood over q-grid under e=1, likelihood under e=0)."""
        p1, p0 = self._p1(), self.p.b
        if fidelity == SINGLE:
            fpr1, fpr0 = self.p.fpr, self.p.fpr
            if self.p.fooled_fpr is not None:
                # judges are fooled more often by bad patches; under e=0 the patch's
                # quality is irrelevant to y, so average over the prior bad-patch rate
                bad = self.grid < self.p.bad_q
                fpr1 = np.where(bad, self.p.fooled_fpr, self.p.fpr)
                w = self.grid ** (self.p.q_a - 1) * (1 - self.grid) ** (self.p.q_b - 1)
                pb = w[bad].sum() / w.sum()
                fpr0 = pb * self.p.fooled_fpr + (1 - pb) * self.p.fpr
            p1 = self.p.sens * p1 + fpr1 * (1 - p1)
            p0 = self.p.sens * p0 + fpr0 * (1 - p0)
        if outcome == 1:
            return p1, p0
        return 1 - p1, 1 - p0

    # ---- posterior quantities ------------------------------------------------
    def _sl(self, cell):
        if cell is None:
            return np.s_[:, :]
        i, j = cell
        return np.s_[i:i + 1, j:j + 1]

    def logit_edge(self, cell=None) -> np.ndarray:
        sl = self._sl(cell)
        return self.prior_logit[sl] + (self.log_ev1[sl] - self.log_ev0[sl]).sum(-1)

    def p_edge(self, cell=None) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit_edge(cell)))

    def q_mean(self, cell=None) -> np.ndarray:
        """Posterior mean patch quality given e=1, shape (I, J, K) or (1, 1, K)."""
        return (self.W[self._sl(cell)] * self.grid).sum(-1)

    def best_patch(self) -> np.ndarray:
        return self.q_mean().argmax(-1)

    # ---- updates -------------------------------------------------------------
    def update(self, i: int, j: int, k: int, fidelity: int, outcome: int) -> None:
        l1, l0 = self.lik(fidelity, outcome)
        w = self.W[i, j, k] * l1
        z = w.sum()
        self.W[i, j, k] = w / z
        self.log_ev1[i, j, k] += np.log(z)
        self.log_ev0[i, j, k] += np.log(l0)
        self.n_obs[i, j, k, fidelity] += 1

    def lookahead(self, fidelity: int, n: int, cell=None):
        """Look ahead n repeated observations of one (patch, fidelity) for every
        (i, j, k) at once, or only for cell=(i, j).

        Returns, for each success count s = 0..n (leading axis):
          prob[s]   P(s successes | D)
          logit[s]  posterior edge logit afterwards
          qmean[s]  posterior mean of q_k afterwards
        each of shape (n+1, I, J, K), or (n+1, 1, 1, K) for a single cell.
        The binomial coefficient cancels in every ratio, so it is dropped from
        the likelihoods and re-applied only to prob.
        """
        from scipy.special import comb

        W, logit = self.W[self._sl(cell)], self.logit_edge(cell)
        l1_succ, l0_succ = self.lik(fidelity, 1)
        s = np.arange(n + 1)
        # (n+1, G) likelihood over the q-grid, and (n+1,) under e=0
        g1 = l1_succ[None, :] ** s[:, None] * (1 - l1_succ[None, :]) ** (n - s[:, None])
        g0 = l0_succ ** s * (1 - l0_succ) ** (n - s)
        pred1 = np.einsum('ijkg,sg->sijk', W, g1)            # P(s | D, e=1, k) / C
        qnum = np.einsum('ijkg,sg->sijk', W * self.grid, g1)
        logit = logit[None, ..., None]
        pe = 1 / (1 + np.exp(-logit))
        c = comb(n, s)[:, None, None, None]
        prob = c * (pe * pred1 + (1 - pe) * g0[:, None, None, None])
        new_logit = logit + np.log(pred1) - np.log(g0)[:, None, None, None]
        return prob, new_logit, qnum / pred1


class CategoryBPosterior(GraphPosterior):
    """GraphPosterior with a spurious recovery rate b_i per category.

    b_i gets a Beta prior centred on the global rate and is updated from control
    replays (the failure replayed with no patch). Because every cell likelihood
    depends on b_i, the cells of category i are recomputed from their sufficient
    statistics (success counts per patch and fidelity) whenever b_i changes.
    """

    def __init__(self, prior_edge, n_patches, params: ObsParams, grid_size=101,
                 b_strength=4.0):
        super().__init__(prior_edge, n_patches, params, grid_size)
        self.b_a = np.full(self.I, params.b * b_strength)
        self.b_b = np.full(self.I, (1 - params.b) * b_strength)
        self.n_ctrl = np.zeros(self.I, dtype=int)
        self.succ = np.zeros((self.I, self.J, self.K, 2), dtype=int)
        prior_q = self.grid ** (params.q_a - 1) * (1 - self.grid) ** (params.q_b - 1)
        self._prior_q = prior_q / prior_q.sum()

    def b_of(self, i):
        return self.b_a[i] / (self.b_a[i] + self.b_b[i])

    def _lik_b(self, fidelity, outcome, b):
        saved = self.p.b
        self.p.b = b
        try:
            return GraphPosterior.lik(self, fidelity, outcome)
        finally:
            self.p.b = saved

    def update(self, i, j, k, fidelity, outcome):
        l1, l0 = self._lik_b(fidelity, outcome, self.b_of(i))
        w = self.W[i, j, k] * l1
        z = w.sum()
        self.W[i, j, k] = w / z
        self.log_ev1[i, j, k] += np.log(z)
        self.log_ev0[i, j, k] += np.log(l0)
        self.n_obs[i, j, k, fidelity] += 1
        self.succ[i, j, k, fidelity] += outcome

    def control(self, i, outcome):
        """Record a control replay of category i and refresh its cells."""
        self.b_a[i] += outcome
        self.b_b[i] += 1 - outcome
        self.n_ctrl[i] += 1
        b = self.b_of(i)
        for j in range(self.J):
            for k in range(self.K):
                W = self._prior_q.copy()
                e1 = e0 = 0.0
                for fid in (FULL, SINGLE):
                    n, s = self.n_obs[i, j, k, fid], self.succ[i, j, k, fid]
                    if n == 0:
                        continue
                    l1s, l0s = self._lik_b(fid, 1, b)
                    l1f, l0f = self._lik_b(fid, 0, b)
                    w = W * l1s ** s * l1f ** (n - s)
                    z = w.sum()
                    W, e1 = w / z, e1 + np.log(z)
                    e0 += s * np.log(l0s) + (n - s) * np.log(l0f)
                self.W[i, j, k], self.log_ev1[i, j, k], self.log_ev0[i, j, k] = W, e1, e0

    def lookahead(self, fidelity, n, cell=None):
        if cell is not None:
            saved = self.p.b
            self.p.b = self.b_of(cell[0])
            try:
                return GraphPosterior.lookahead(self, fidelity, n, cell)
            finally:
                self.p.b = saved
        # one full pass per category (b_i differs), keeping that category's row
        res = None
        saved = self.p.b
        try:
            for i in range(self.I):
                self.p.b = self.b_of(i)
                out = GraphPosterior.lookahead(self, fidelity, n, None)
                if res is None:
                    res = [o.copy() for o in out]
                else:
                    for r, o in zip(res, out):
                        r[:, i] = o[:, i]
        finally:
            self.p.b = saved
        return tuple(res)
