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
            p1 = self.p.sens * p1 + self.p.fpr * (1 - p1)
            p0 = self.p.sens * p0 + self.p.fpr * (1 - p0)
        if outcome == 1:
            return p1, p0
        return 1 - p1, 1 - p0

    # ---- posterior quantities ------------------------------------------------
    def logit_edge(self) -> np.ndarray:
        return self.prior_logit + (self.log_ev1 - self.log_ev0).sum(-1)

    def p_edge(self) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit_edge()))

    def q_mean(self) -> np.ndarray:
        """Posterior mean patch quality given e=1, shape (I, J, K)."""
        return (self.W * self.grid).sum(-1)

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

        if cell is None:
            W, logit = self.W, self.logit_edge()
        else:
            i, j = cell
            W = self.W[i:i + 1, j:j + 1]
            logit = self.prior_logit[i:i + 1, j:j + 1] + \
                (self.log_ev1[i:i + 1, j:j + 1] - self.log_ev0[i:i + 1, j:j + 1]).sum(-1)
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
