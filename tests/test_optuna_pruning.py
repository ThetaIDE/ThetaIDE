from types import SimpleNamespace

import pytest

pytest.importorskip("optuna")
pytest.importorskip("lightning")

from src.app.tuners.optuna.pruning import OptunaPruningCallback, make_pruner


def test_pruning_requires_an_objective_visible_during_validation():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=1, callback_metrics={}, current_epoch=0)
    callback.on_validation_end(trainer, None)
    with pytest.raises(ValueError, match="val/loss.*validation"):
        callback.on_fit_end(trainer, None)


def test_pruning_rejects_nonfinite_intermediate_metrics():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=1,
                              callback_metrics={"val/loss": float("nan")}, current_epoch=0)
    with pytest.raises(ValueError, match="finite"):
        callback.on_validation_end(trainer, None)


def test_pruning_rejects_multi_process_training():
    callback = OptunaPruningCallback({"objective_metric": "val/loss"})
    trainer = SimpleNamespace(sanity_checking=False, world_size=2)
    with pytest.raises(ValueError, match="single Lightning process"):
        callback.on_validation_end(trainer, None)


@pytest.mark.parametrize("startup,warmup", [(-1, 0), (0, -1), (True, 0)])
def test_invalid_pruner_counts_are_rejected(startup, warmup):
    with pytest.raises(ValueError, match="nonnegative"):
        make_pruner("median", startup, warmup)


def test_interrupted_training_exits_with_nonzero_status():
    import signal
    from src.app.core.lightning_builder import _exit_on_signal

    with pytest.raises(SystemExit) as exc:
        _exit_on_signal(signal.SIGTERM, None)
    assert exc.value.code == 128 + signal.SIGTERM
