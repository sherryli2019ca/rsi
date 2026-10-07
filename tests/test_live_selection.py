"""verify/live.py applies the same rule and estimators as verify/analyze.py."""
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rrsi.config import RRSIConfig            # noqa: E402
from verify import analyze, live             # noqa: E402

CFG = dict(beta0=0.1, beta1=37.0, w_s=0.0, w_c=15.0, w_n=0.5)


def test_rule_matches_offline_decide():
    cfg = RRSIConfig(**CFG)
    rng = random.Random(0)
    for _ in range(5000):
        dS = rng.uniform(-0.1, 0.1)
        se = rng.choice([0.0, rng.uniform(0, 0.05)])
        dC = rng.uniform(-0.5, 0.5)
        nu = rng.choice([0, 1])
        ok, _, _ = live.admissible((dS, se, dC), nu, cfg, 2.0)
        assert ok == (analyze.decide({"A": (dS, se, dC, nu)}, CFG, 2.0) == "A")


def rec(reward, agent=100, user=10):
    return {"reward": reward, "tokens": {"agent_total": agent, "user": user}}


def test_estimators():
    f = 0.25
    ev = {"replays": [rec(1), rec(0), rec(1), None], "pairs": [("t1", 0, 1, 1.0, 100), ("t2", 1, 1, 0.0, 100)],
          "fresh": [rec(0, 150), rec(1, 150)]}
    null = [rec(0), rec(1), rec(0), rec(0)]
    dS, se, dC = live.estimate("replay", ev, null, f)
    assert dS == pytest.approx(f * 2 / 3)
    dS, se, dC = live.estimate("replaynull", ev, null, f)
    assert dS == pytest.approx(f * (2 / 3 - 1 / 4))
    dS, se, dC = live.estimate("net", ev, null, f)
    assert dS == pytest.approx(f * (2 / 3 - 1 / 4) + (1 - f) * ((0 - 1.0) + (1 - 0.0)) / 2)
    assert dC == pytest.approx(0.5)
    dS, se, dC = live.estimate("sample", ev, null, f)
    assert dS == pytest.approx(((0 - 1) + (1 - 1)) / 2)


def test_parse_mode():
    assert live.parse_mode("net@40") == ("net", 40)
    assert live.parse_mode("judge") == ("judge", 0)
    for bad in ("full@3", "net", "net@2", "judge@4", "foo@1"):
        with pytest.raises(SystemExit):
            live.parse_mode(bad)
