"""Data module providers and offline datasets.

    ``BaseDataModule``       base interface of data module capabilities.
    ``register_datamodule``  decorator registering a data module class under one or more names.
    ``DatasetReader``        reader for Theta offline datasets (accessing it imports PyTorch).
"""
from ._exports import lazy_exports

_EXPORTS = {
    "BaseDataModule": "src.app.core.interfaces:BaseDataModule",
    "register_datamodule": "src.app.core.datamodule_registry:register_datamodule",
    "DatasetReader": "src.app.dataset_utils:DatasetReader",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
