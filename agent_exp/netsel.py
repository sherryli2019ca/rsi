"""Patch selection when every piece of evidence costs about an episode.

Each candidate patch k of cell (i, j) has a net effect on success

    Delta_k = f_i (p_k - b_i) / (1 - b_i)  -  w (r_k - r0) / (1 - r0)

where p_k is its targeted-replay success on failures of category i, b_i the
category's spurious recovery (null replays, no patch), r_k its failure rate on
previously successful tasks (regression runs), r0 the natural re-run failure
rate and w = successes / failures, which puts regressions in failure-mass units.
All three rates get Beta posteriors. The policy picks, among a targeted replay,
a null replay and a regression run, the action with the largest knowledge
gradient per unit cost, and finally applies, for every cell, the patch with the
largest positive posterior mean net effect.

The attribution counts set the prior mean of p_k through CARVE's edge prior:
b + pi_ij (q_bar - b), so with no evidence the policy applies the patches whose
prior net effect is positive.
"""
from __future__ import annotations

import numpy as np

HORIZONS = (1, 2, 4, 8)


def _bb_pmf(n, a, b):
    """Beta-binomial pmf over 0..n."""
    from scipy.special import betaln, gammaln
    s = np.arange(n + 1)
    lc = gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1)
    return np.exp(lc + betaln(s + a, n - s + b) - betaln(a, b))


class NetPosterior:
    def __init__(self, counts, inb, f, b0, r0, w, K=3, prior_n=2.0, q_bar=0.55, r_hat=0.7,
                 use_prior=True, c_fp=0.02):
        I, J = counts.shape
        self.f, self.r0, self.w, self.K, self.c_fp = f, r0, w, K, c_fp
        self.cells = [(i, j) for i, j in zip(*np.nonzero(inb))]
        from sim.testbed import edge_prior
        pi = edge_prior(counts, r_hat)
        self.p = {}
        for i, j in self.cells:
            # prior mean repair rate: spurious rate plus, if the edge is real
            # (prior probability pi from the attributions), a typical patch effect
            m = b0 + (pi[i, j] * (q_bar - b0) if use_prior else 0.0)
            m = float(np.clip(m, 0.02, 0.95))
            for k in range(K):
                self.p[i, j, k] = [m * prior_n, (1 - m) * prior_n]
        self.b = {i: [b0 * 4, (1 - b0) * 4] for i in {c[0] for c in self.cells}}
        self.r = {(i, j, k): [r0 * 4, (1 - r0) * 4] for i, j in self.cells for k in range(K)}

    @staticmethod
    def _mean(ab):
        return ab[0] / (ab[0] + ab[1])

    def delta(self, i, j, k, p=None, b=None, r=None):
        p = self._mean(self.p[i, j, k]) if p is None else p
        b = self._mean(self.b[i]) if b is None else b
        r = self._mean(self.r[i, j, k]) if r is None else r
        rep = self.f[i] * (p - b) / max(1e-6, 1 - b)
        # a regression rate below the natural one is noise, not a benefit
        reg = self.w * max(0.0, r - self.r0) / max(1e-6, 1 - self.r0)
        # c_fp: a fixed cost for any change, as in the objective (Eq. gain)
        return rep - reg - self.c_fp

    def cell_value(self, i, j, over=None):
        over = over or {}
        return max(0.0, max(self.delta(i, j, k, **over.get(k, {})) for k in range(self.K)))

    def decide(self, shape):
        acc = np.zeros(shape, bool)
        patch = np.zeros(shape, int)
        for i, j in self.cells:
            d = [self.delta(i, j, k) for k in range(self.K)]
            k = int(np.argmax(d))
            if d[k] > 0:
                acc[i, j], patch[i, j] = True, k
        return acc, patch


def _kg(post, i, j, k, kind, n):
    """Expected gain in decision value from n more observations of one rate."""
    if kind == "null":
        a, bb = post.b[i]
        cells = [(ii, jj) for ii, jj in post.cells if ii == i]
    else:
        a, bb = (post.p if kind == "rep" else post.r)[i, j, k]
        cells = [(i, j)]
    pmf = _bb_pmf(n, a, bb)
    now = sum(post.cell_value(ii, jj) for ii, jj in cells)
    exp = 0.0
    for s in range(n + 1):
        m = (a + s) / (a + bb + n)
        v = 0.0
        for ii, jj in cells:
            if kind == "null":
                v += max(0.0, max(post.delta(ii, jj, kk, b=m) for kk in range(post.K)))
            else:
                key = "p" if kind == "rep" else "r"
                v += post.cell_value(ii, jj, {k: {key: m}})
        exp += pmf[s] * v
    return exp - now


def run_netsel(bank, counts, f, inb, costs, b0, seed, budget, r0, w, use_reg=True,
               use_null=True, use_prior=True, n_ctrl=0):
    from agent_exp.bank import FULL, BankWorld
    rng = np.random.default_rng(seed)
    world = BankWorld(bank, rng)
    post = NetPosterior(counts, inb, f, b0, r0, w, use_prior=use_prior)
    tn = np.mean([v["tok"] for v in bank["null"].values()])
    tf = np.mean([t for c in bank["cells"].values() for t in c["tok_full"]])
    cost = {"rep": 1.0, "null": float(tn / tf), "reg": float(costs[2])}
    spent = 0.0
    n_null = {}
    while True:
        best, arg = 0.0, None
        for i, j in post.cells:
            acts = [("rep", k) for k in range(post.K)]
            if use_reg:
                acts += [("reg", k) for k in range(post.K)]
            for kind, k in acts:
                for n in HORIZONS:
                    if spent + cost[kind] > budget:
                        break
                    s = _kg(post, i, j, k, kind, n) / (n * cost[kind])
                    if s > best:
                        best, arg = s, (kind, i, j, k)
        if use_null:
            for i in post.b:
                for n in HORIZONS:
                    if spent + cost["null"] > budget:
                        break
                    s = _kg(post, i, None, None, "null", n) / (n * cost["null"])
                    if s > best:
                        best, arg = s, ("null", i, None, None)
        if arg is None or best <= 1e-9:
            break
        kind, i, j, k = arg
        if kind == "rep" and n_null.get(i, 0) < n_ctrl:
            # measure the category's spurious recovery before trusting replays
            kind = "null"
        if kind == "rep":
            o = world.intervene(i, j, k, FULL)
            post.p[i, j, k][1 - o] += 1
        elif kind == "reg":
            fail = 1 - world.regress(i, j, k)
            post.r[i, j, k][1 - fail] += 1
        else:
            o = world.control(i)
            post.b[i][1 - o] += 1
            n_null[i] = n_null.get(i, 0) + 1
        spent += cost[kind]
    acc, patch = post.decide(counts.shape)
    return acc, patch, spent
