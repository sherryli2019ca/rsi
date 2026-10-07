"""Patch selection when every piece of evidence costs about an episode.

Each candidate patch k of cell (i, j) has a net effect on success

    Delta_k = e_ij f_i q_k  -  w max(0, r_k - r0) / (1 - r0)  -  c_fp

where e_ij says whether component j causes category i at all, q_k is the
patch's repair rate on failures of category i (a targeted replay succeeds with
probability b_i + (1 - b_i) q_k, b_i the category's spurious recovery measured
by null replays), r_k its failure rate on
previously successful tasks (regression runs), r0 the natural re-run failure
rate and w = successes / failures, which puts regressions in failure-mass units.
The policy picks, among a targeted replay,
a null replay and a regression run, the action with the largest knowledge
gradient per unit cost, and finally applies, for every cell, the patch with the
largest positive posterior mean net effect.

The attribution counts set the prior of e_ij through CARVE's edge prior.
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


GRID = np.linspace(0.005, 0.995, 100)


def _binom_lik(s, n, p):
    return p ** s * (1 - p) ** (n - s)


class NetPosterior:
    """Per cell: is component j a cause of category i at all (edge, spike-and-slab
    as in CARVE, prior pi_ij from the attributions)? Per patch: repair rate q_k if
    it is (grid posterior), and regression rate r_k (Beta). Per category: spurious
    recovery b_i (Beta, plug-in mean in the likelihood). A replay of patch k on a
    failure of category i succeeds with probability b_i + (1 - b_i) q_k if the edge
    exists and b_i otherwise."""

    def __init__(self, counts, inb, f, b0, r0, w, K=3, r_hat=0.7, use_prior=True,
                 c_fp=0.02, q_prior=(2.0, 2.0), b_strength=4.0, r_strength=4.0):
        from sim.testbed import edge_prior
        self.f, self.r0, self.w, self.K, self.c_fp = f, r0, w, K, c_fp
        self.cells = [(i, j) for i, j in zip(*np.nonzero(inb))]
        I, J = counts.shape
        pi = edge_prior(counts, r_hat) if use_prior else np.full((I, J), 1.5 / J)
        self.logit_pi = {c: float(np.log(pi[c] / (1 - pi[c]))) for c in self.cells}
        a, b = q_prior
        self.q_prior = GRID ** (a - 1) * (1 - GRID) ** (b - 1)
        self.q_prior /= self.q_prior.sum()
        self.sn = {(i, j, k): [0, 0] for i, j in self.cells for k in range(K)}
        self.b = {i: [b0 * b_strength, (1 - b0) * b_strength] for i in {c[0] for c in self.cells}}
        self.r = {(i, j, k): [r0 * r_strength, (1 - r0) * r_strength]
                  for i, j in self.cells for k in range(K)}

    def bmean(self, i):
        a, b = self.b[i]
        return a / (a + b)

    def cell_state(self, i, j, sn=None, b=None, r=None):
        """(P(edge), [E q_k | edge], [regression term_k]) under optional overrides."""
        sn = sn or {}
        b = self.bmean(i) if b is None else b
        lo = self.logit_pi[i, j]
        qs = []
        for k in range(self.K):
            s, n = sn.get(k, self.sn[i, j, k])
            l1 = self.q_prior * _binom_lik(s, n, b + (1 - b) * GRID)
            z1 = l1.sum()
            lo += np.log(max(z1, 1e-300)) - np.log(max(_binom_lik(s, n, b), 1e-300))
            qs.append(float((l1 * GRID).sum() / max(z1, 1e-300)))
        pe = 1 / (1 + np.exp(-lo))
        regs = []
        for k in range(self.K):
            ra, rb = (r or {}).get(k, self.r[i, j, k])
            rm = ra / (ra + rb)
            regs.append(self.w * max(0.0, rm - self.r0) / max(1e-6, 1 - self.r0))
        return pe, qs, regs

    def deltas(self, i, j, **over):
        pe, qs, regs = self.cell_state(i, j, **over)
        return [pe * self.f[i] * qs[k] - regs[k] - self.c_fp for k in range(self.K)]

    def cell_value(self, i, j, **over):
        return max(0.0, max(self.deltas(i, j, **over)))

    def decide(self, shape):
        acc = np.zeros(shape, bool)
        patch = np.zeros(shape, int)
        for i, j in self.cells:
            d = self.deltas(i, j)
            k = int(np.argmax(d))
            if d[k] > 0:
                acc[i, j], patch[i, j] = True, k
        return acc, patch


def _kg(post, i, j, k, kind, n):
    """Expected gain in decision value from n more observations."""
    from scipy.stats import binom
    if kind == "null":
        a, bb = post.b[i]
        cells = [(ii, jj) for ii, jj in post.cells if ii == i]
        now = sum(post.cell_value(ii, jj) for ii, jj in cells)
        pmf = _bb_pmf(n, a, bb)
        exp = sum(pmf[s] * sum(post.cell_value(ii, jj, b=(a + s) / (a + bb + n))
                               for ii, jj in cells) for s in range(n + 1))
        return exp - now
    now = post.cell_value(i, j)
    if kind == "reg":
        a, bb = post.r[i, j, k]
        pmf = _bb_pmf(n, a, bb)
        exp = sum(pmf[s] * post.cell_value(i, j, r={k: [a + s, bb + n - s]})
                  for s in range(n + 1))
        return exp - now
    # targeted replay: predictive mixes the edge and no-edge hypotheses
    b = post.bmean(i)
    pe, _, _ = post.cell_state(i, j)
    s0, n0 = post.sn[i, j, k]
    w1 = post.q_prior * _binom_lik(s0, n0, b + (1 - b) * GRID)
    w1 /= w1.sum()
    ss = np.arange(n + 1)
    p1 = (w1[None, :] * binom.pmf(ss[:, None], n, b + (1 - b) * GRID[None, :])).sum(1)
    pmf = pe * p1 + (1 - pe) * binom.pmf(ss, n, b)
    exp = sum(pmf[s] * post.cell_value(i, j, sn={k: (s0 + s, n0 + n)}) for s in ss)
    return exp - now


def run_netsel(bank, counts, f, inb, costs, b0, seed, budget, r0, w, use_reg=True,
               use_null=True, use_prior=True, n_ctrl=0, c_fp=0.02):
    from agent_exp.bank import FULL, BankWorld
    rng = np.random.default_rng(seed)
    world = BankWorld(bank, rng)
    post = NetPosterior(counts, inb, f, b0, r0, w, use_prior=use_prior, c_fp=c_fp)
    tn = np.mean([v["tok"] for v in bank["null"].values()])
    tf = np.mean([t for c in bank["cells"].values() for t in c["tok_full"]])
    cost = {"rep": 1.0, "null": float(tn / tf), "reg": float(costs[2])}
    spent = 0.0
    n_null = {}

    def score(kind, i, j, k):
        return max(_kg(post, i, j, k, kind, n) / (n * cost[kind]) for n in HORIZONS)

    # scores only change for the category whose evidence was updated
    scores = {}

    def rescore(cat):
        for i, j in post.cells:
            if cat is not None and i != cat:
                continue
            for k in range(post.K):
                scores["rep", i, j, k] = score("rep", i, j, k)
                if use_reg:
                    scores["reg", i, j, k] = score("reg", i, j, k)
        if use_null:
            for i in post.b:
                if cat is None or i == cat:
                    scores["null", i, None, None] = score("null", i, None, None)

    rescore(None)
    while True:
        ok = [(v, a) for a, v in scores.items() if spent + cost[a[0]] <= budget]
        best, arg = max(ok, key=lambda x: x[0]) if ok else (0.0, None)
        if arg is None or best <= 1e-9:
            break
        kind, i, j, k = arg
        if kind == "rep" and n_null.get(i, 0) < n_ctrl:
            # measure the category's spurious recovery before trusting replays
            kind = "null"
        if kind == "rep":
            o = world.intervene(i, j, k, FULL)
            post.sn[i, j, k][0] += o
            post.sn[i, j, k][1] += 1
        elif kind == "reg":
            fail = 1 - world.regress(i, j, k)
            post.r[i, j, k][1 - fail] += 1
        else:
            o = world.control(i)
            post.b[i][1 - o] += 1
            n_null[i] = n_null.get(i, 0) + 1
        spent += cost[kind]
        rescore(i)
    acc, patch = post.decide(counts.shape)
    return acc, patch, spent
