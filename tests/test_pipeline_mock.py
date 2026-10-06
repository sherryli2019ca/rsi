import os
import sys
import tempfile
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from agent_exp import pipeline as P
from agent_exp.env import ShopDB, make_tasks
from mock_llm import MockLLM


def test_pipeline_runs_end_to_end():
    with tempfile.TemporaryDirectory() as out:
        args = NS(out=out, seed=0, method="CARVE", budget=6, edge_share=0.3, audit=10,
                  n_test=10)
        llm, db = MockLLM(), ShopDB(0)
        tasks = make_tasks(db, 30, seed=1)
        data = P.stage_collect(llm, args, db, tasks)
        attrs = P.stage_attribute(llm, args, data)
        tax = P.stage_taxonomy(llm, args, attrs)
        for m in ["CARVE", "Uncertainty", "LLM-only", "Replay-each"]:
            args.method = m
            res = P.stage_verify(llm, args, db, P._load(os.path.join(out, "traces.json")),
                                 attrs, tax)
            assert 0 <= res["precision"] <= 1
            ev = P.stage_evaluate(llm, args, db, res)
            assert 0 <= ev["patched_success"] <= 1


def test_ground_truth_answers():
    db = ShopDB(0)
    for t in make_tasks(db, 50, seed=1):
        assert t.answer.replace(".", "").isdigit()
