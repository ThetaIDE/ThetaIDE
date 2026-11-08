import pytest
from omegaconf import OmegaConf

from src.app.core.datamodule_registry import (
    DATAMODULE_REGISTRY,
    get_datamodule_class,
    list_registered_datamodules,
    register_datamodule,
)
from src.app.core.paradigm_loader import load_paradigm_definition


def test_register_and_get_datamodule():
    @register_datamodule("dummy_custom_dm")
    class DummyCustomDataModule:
        def __init__(self, cfg=None, **kwargs):
            self.cfg = cfg
            self.kwargs = kwargs

    assert DATAMODULE_REGISTRY["dummy_custom_dm"] is DummyCustomDataModule
    assert get_datamodule_class("dummy_custom_dm") is None
    cls = DummyCustomDataModule

    instance = cls(None, grayscale=True, frame_size=[84, 84])
    assert instance.kwargs == {"grayscale": True, "frame_size": [84, 84]}


def test_builtin_datamodules_registered():
    dms = list_registered_datamodules()
    assert "simulatordatamodule" in [d.lower() for d in dms]
    assert "rlreplaybufferdatamodule" in [d.lower() for d in dms]
    assert "superviseddatamodule" in [d.lower() for d in dms]


def test_paradigm_default_data_module_resolved():
    p_online = load_paradigm_definition("online_rl")
    assert p_online.data_module_cls is not None
    assert p_online.data_module_cls.__name__ == "SimulatorDataModule"

    p_offline = load_paradigm_definition("offline_rl")
    assert p_offline.data_module_cls is not None
    assert p_offline.data_module_cls.__name__ in ("RLReplayBufferDataModule", "ReplayBufferDataModule")

    p_sup = load_paradigm_definition("supervised")
    assert p_sup.data_module_cls is not None
    assert p_sup.data_module_cls.__name__ == "SupervisedDataModule"
