import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from carve.model import FULL, SINGLE, GraphPosterior, ObsParams


def make():
    rng = np.random.default_rng(0)
    return GraphPosterior(rng.uniform(0.05, 0.6, (3, 4)), 2, ObsParams())


def test_lookahead_probabilities_sum_to_one():
    post = make()
    post.update(0, 1, 0, FULL, 1)
    for fid in (FULL, SINGLE):
        for n in (1, 3):
            prob, _, _ = post.lookahead(fid, n)
            assert np.allclose(prob.sum(0), 1.0)


def test_lookahead_matches_sequential_updates():
    post = make()
    post.update(1, 2, 1, SINGLE, 0)
    prob, logit, qmean = post.lookahead(FULL, 2)
    # observe two successes on (1, 2, patch 0) and compare with s=2 branch
    post.update(1, 2, 0, FULL, 1)
    post.update(1, 2, 0, FULL, 1)
    assert np.isclose(post.logit_edge()[1, 2], logit[2, 1, 2, 0])
    assert np.isclose(post.q_mean()[1, 2, 0], qmean[2, 1, 2, 0])


def test_cell_lookahead_equals_full():
    post = make()
    post.update(2, 3, 0, FULL, 1)
    a = post.lookahead(SINGLE, 4)
    b = post.lookahead(SINGLE, 4, cell=(2, 3))
    for x, y in zip(a, b):
        assert np.allclose(x[:, 2, 3], y[:, 0, 0])


def test_evidence_moves_edge_posterior_in_right_direction():
    post = make()
    p0 = post.p_edge()[0, 0]
    for _ in range(5):
        post.update(0, 0, 0, FULL, 1)
    assert post.p_edge()[0, 0] > p0
    post2 = make()
    for _ in range(5):
        post2.update(0, 0, 0, FULL, 0)
    assert post2.p_edge()[0, 0] < p0
