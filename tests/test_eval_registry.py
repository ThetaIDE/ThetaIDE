import pytest
from src.app.core.eval_registry import (
    EVALUATOR_REGISTRY,
    get_evaluator_class,
    list_registered_evaluators,
    register_evaluator,
)
from src.app.core.interfaces import BaseEvalProtocol


def test_register_and_get_evaluator():
    @register_evaluator("dummy_custom_eval")
    class DummyCustomEvaluator(BaseEvalProtocol):
        def evaluate(self, agent, data_source, cfg):
            return {"val/custom_metric": 0.95}

    assert EVALUATOR_REGISTRY["dummy_custom_eval"] is DummyCustomEvaluator
    assert get_evaluator_class("dummy_custom_eval") is None
    cls = DummyCustomEvaluator
    instance = cls()
    res = instance.evaluate(None, None, None)
    assert res == {"val/custom_metric": 0.95}


def test_builtin_evaluators_registered():
    evals = list_registered_evaluators()
    assert "episodicrewardevalprotocol" in [e.lower() for e in evals]
    assert "offlinerlevalprotocol" in [e.lower() for e in evals]
    assert "classificationevalprotocol" in [e.lower() for e in evals]
