"""Data module providers and offline datasets.

    ``BaseDataModule``       base interface of data module capabilities.
    ``register_datamodule``  decorator registering a data module class under one or more names.
    ``DatasetReader``        reader for Theta offline datasets (accessing it imports PyTorch).
    ``loader_selection(cfg, "data_module", method=None, paths=)``
                             the normalized data module selection and where it came from.
    ``resolve_dataset_for_method(method_name, method_cfg, cfg, paths=)``
                             the dataset file a method reads, anchored to the workspace.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "BaseDataModule": "src.app.core.interfaces:BaseDataModule",
    "register_datamodule": "src.app.core.datamodule_registry:register_datamodule",
    "DatasetReader": "src.app.dataset_utils:DatasetReader",
    "loader_selection": "src.app.core.data_loaders:loader_selection",
    "resolve_dataset_for_method": "src.app.pipeline.datasets:resolve_dataset_for_method",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
