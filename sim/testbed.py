"""Synthetic testbed with a known category -> component causal graph.

The generator is deliberately richer than the CARVE model so that the model is
misspecified in realistic ways:
  * mislabeled traces come from other categories and can be repaired by the
    patch if *their* true edge points at the same component (the model assumes
    a flat spurious rate b);
  * LLM attributions make systematic, not uniform, mistakes: each category has a
    "decoy" component (a symptom-level locus) that wrong attributions prefer;
  * patch quality is bimodal (some candidate patches are simply bad).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorldConfig:
    I: int = 12                 # failure categories
    J: int = 20                 # modifiable components
    K: int = 3                  # candidate patches per (category, component)
    edges_per_cat: tuple = (1, 2)
    zipf: float = 1.0           # skew of category frequencies
    bad_patch_rate: float = 0.3
    b: float = 0.05             # spurious recovery rate
    lam: float = 0.8            # label accuracy of traces sampled per category
    sens: float = 0.85          # single-step check sensitivity
    fpr: float = 0.15           # single-step check false-positive rate
    r: float = 0.5              # LLM attribution accuracy
    decoy_rate: float = 0.6     # P(wrong attribution goes to the decoy)
    n_attr: int = 20            # attributed traces per category
    joint_rate: float = 0.0     # share of categories needing two components changed together
    joint_solo: float = 0.2     # repair factor of a single patch in a joint category
    extra: dict = field(default_factory=dict)


class World:
    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        I, J, K = cfg.I, cfg.J, cfg.K
        f = 1.0 / np.arange(1, I + 1) ** cfg.zipf
        self.f = f / f.sum()
        self.E = np.zeros((I, J), dtype=bool)
        self.decoy = np.zeros(I, dtype=int)
        for i in range(I):
            d = rng.integers(cfg.edges_per_cat[0], cfg.edges_per_cat[1] + 1)
            js = rng.choice(J, size=d + 1, replace=False)
            self.E[i, js[:d]] = True
            self.decoy[i] = js[d]
        # joint categories have exactly two true components that must both change
        self.joint = np.zeros(I, dtype=bool)
        if cfg.joint_rate > 0:
            for i in range(I):
                if rng.random() < cfg.joint_rate:
                    self.joint[i] = True
                    if self.E[i].sum() < 2:
                        cand = np.flatnonzero(~self.E[i])
                        cand = cand[cand != self.decoy[i]]
                        self.E[i, rng.choice(cand)] = True
        bad = rng.random((I, J, K)) < cfg.bad_patch_rate
        good_q = rng.beta(6, 3, (I, J, K))
        bad_q = rng.beta(1, 9, (I, J, K))
        self.Q = np.where(bad, bad_q, good_q)

    # ---- LLM attributions --------------------------------------------------
    def attributions(self, n_per_cat=None) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        n_per_cat = np.full(cfg.I, cfg.n_attr) if n_per_cat is None else n_per_cat
        counts = np.zeros((cfg.I, cfg.J), dtype=int)
        for i in range(cfg.I):
            true_js = np.flatnonzero(self.E[i])
            for _ in range(int(n_per_cat[i])):
                u = rng.random()
                if u < cfg.r:
                    j = rng.choice(true_js)
                elif rng.random() < cfg.decoy_rate:
                    j = self.decoy[i]
                else:
                    j = rng.integers(cfg.J)
                counts[i, j] += 1
        return counts

    # ---- interventions -----------------------------------------------------
    def intervene(self, i, j, k, fidelity, f=None) -> int:
        cfg, rng = self.cfg, self.rng
        f = self.f if f is None else f
        src = i
        if rng.random() > cfg.lam:          # mislabeled trace from another category
            others = np.delete(np.arange(cfg.I), i)
            w = f[others] / f[others].sum()
            src = rng.choice(others, p=w)
        p = self.Q[src, j, k] if self.E[src, j] else 0.0
        if self.joint[src]:
            p *= cfg.joint_solo
        y = rng.random() < p + (1 - p) * cfg.b
        if fidelity == 0:
            return int(y)
        return int(rng.random() < (cfg.sens if y else cfg.fpr))

    # ---- evaluation --------------------------------------------------------
    def repaired_mass(self, acc, patch, f=None) -> np.ndarray:
        """Fraction of each category's failures repaired by the accepted patches."""
        f = self.f if f is None else f
        keep = np.ones(self.cfg.I)
        for i in range(self.cfg.I):
            hit = np.flatnonzero(acc[i] & self.E[i])
            qs = [self.Q[i, j, patch[i, j]] for j in hit]
            if self.joint[i]:
                if len(hit) == self.E[i].sum():          # all required changes made
                    keep[i] = 1 - np.prod(qs)
                else:
                    keep[i] = np.prod([1 - self.cfg.joint_solo * q for q in qs])
            else:
                keep[i] = np.prod([1 - q for q in qs])
        return 1 - keep

    def net_gain(self, acc, patch, c_fp, f=None) -> float:
        f = self.f if f is None else f
        fp = (acc & ~self.E).sum()
        return float((f * self.repaired_mass(acc, patch, f)).sum() - c_fp * fp)

    def oracle(self, c_fp, f=None):
        """Best achievable decision: accept true edges with their best patch
        (an edge whose best patch is useless is still accepted: no regression)."""
        patch = self.Q.argmax(-1)
        return self.net_gain(self.E.copy(), patch, c_fp, f), self.E.copy(), patch


def edge_prior(counts, r_hat, edges_per_cat=1.5):
    """Turn LLM attribution counts into prior edge probabilities.

    pi_ij = (1 - r_hat) * base + r_hat * min(1, d * n_ij / n_i), with base = d / J
    the edge density and d the expected number of true components per category.
    """
    I, J = counts.shape
    base = edges_per_cat / J
    tot = counts.sum(1, keepdims=True)
    share = np.divide(counts, tot, out=np.zeros_like(counts, dtype=float), where=tot > 0)
    return (1 - r_hat) * base + r_hat * np.minimum(1.0, edges_per_cat * share)
