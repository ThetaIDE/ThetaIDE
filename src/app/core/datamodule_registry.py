"""DataModule Registry — Single source of truth for pluggable DataModules.

Data modules manage data ingestion, preprocessing, augmentation, and dataloader emission.
They self-register using the @register_datamodule decorator:
    @register_datamodule("my_data_module")
    class MyDataModule(L.LightningDataModule):
        ...

The pipeline resolves data modules via get_datamodule_class():
    DataModuleClass = get_datamodule_class("my_data_module")
"""

import importlib
import logging
import os
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type

DATAMODULE_REGISTRY: Dict[str, Type[Any]] = {}
_DISCOVERY_COMPLETE = False
log = logging.getLogger(__name__)


def register_datamodule(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a DataModule under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = name.strip()
            DATAMODULE_REGISTRY[clean_name] = cls
            DATAMODULE_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component

                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_datamodule_class(dm_name):
    if not dm_name: return None
    from src.app.core.provider_loader import resolve_provider
    selected = resolve_provider("datamodule", str(dm_name))
    if selected is not None: return selected
    _register_builtins()
    value = DATAMODULE_REGISTRY.get(str(dm_name)) or DATAMODULE_REGISTRY.get(str(dm_name).lower())
    return value if value is not None and value.__module__.startswith('src.app.') else None


def auto_discover_datamodules() -> None:
    """Auto-discover data modules in src/usr/data/, src/app/data/, and plugin entrypoints."""
    from src.app.core.provider_loader import discover_kind
    discover_kind("datamodule")
    _register_builtins()


def _register_builtins() -> None:
    """Ensure core built-in data modules are registered."""
    try:
        from src.app.data.rl_data_module import RLDataModule

        register_datamodule("RLDataModule")(RLDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.online_rl import SimulatorDataModule

        register_datamodule("SimulatorDataModule")(SimulatorDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.offline_rl import RLReplayBufferDataModule

        register_datamodule("RLReplayBufferDataModule")(RLReplayBufferDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.supervised import SupervisedDataModule

        register_datamodule("SupervisedDataModule", "CrossValidationDataModule")(SupervisedDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.tabular import SupervisedNpzDataModule, TabularNpzDataModule

        register_datamodule("TabularNpzDataModule")(TabularNpzDataModule)
        register_datamodule("SupervisedNpzDataModule")(SupervisedNpzDataModule)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.unsupervised import UnsupervisedDataModule

        register_datamodule("UnsupervisedDataModule")(UnsupervisedDataModule)
    except Exception:
        pass


def list_registered_datamodules() -> List[str]:
    """Return sorted list of registered data module names."""
    from src.app.core.provider_catalog import get_catalog
    names = {key for p in get_catalog().packages for c in p.capabilities if c.kind == "datamodule" for key in (c.id, *c.aliases)}
    return sorted({key for key,value in DATAMODULE_REGISTRY.items() if value.__module__.startswith("src.app.")} | names | {"RLDataModule", "SimulatorDataModule", "RLReplayBufferDataModule", "SupervisedDataModule", "CrossValidationDataModule", "TabularNpzDataModule", "SupervisedNpzDataModule", "UnsupervisedDataModule"})
