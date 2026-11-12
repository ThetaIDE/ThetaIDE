"""Metadata for authoring selectors; never imports provider implementations."""
import yaml

from src.app.core.provider_catalog import DuplicateProvider, get_catalog
from src.app.core.provider_discovery import discovery_cached


# These framework providers predate package manifests. They intentionally have
# no compatibility restriction, matching their previous selector behavior.
BUILTIN_SELECTORS = {
    "datamodule": (
        "RLDataModule", "SimulatorDataModule", "RLReplayBufferDataModule",
        "SupervisedDataModule", "CrossValidationDataModule", "TabularNpzDataModule",
        "SupervisedNpzDataModule", "UnsupervisedDataModule",
    ),
    "evaluator": (
        "EpisodicRewardEvalProtocol", "OfflineRLEvalProtocol",
        "ClassificationEvalProtocol", "ReconstructionEvalProtocol",
    ),
}


@discovery_cached
def profile_values(path):
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return values if isinstance(values, dict) else {}


@discovery_cached
def capability_metadata(kind, name, paths):
    selected = get_catalog(paths).resolve(kind, name)
    if selected is None:
        return {}
    package, capability = selected
    metadata = {"id": capability.id, **capability.metadata}
    metadata["paradigms"] = (metadata.get("compatible_paradigms") or metadata.get("paradigms")
                             or package.manifest.get("compatible_paradigms") or package.manifest.get("paradigms"))
    return metadata


@discovery_cached
def selector_metadata(kind, paths):
    catalog = get_catalog(paths)
    names = {name for package in catalog.packages for cap in package.capabilities
             if cap.kind == kind for name in (cap.id, *cap.aliases)}
    result = {name: {} for name in BUILTIN_SELECTORS.get(kind, ())}
    for name in sorted(names):
        try:
            result[name] = capability_metadata(kind, name, paths)
        except DuplicateProvider:
            # An unresolved duplicate cannot be offered as a valid new choice.
            # Existing explicit selections stay visible in the form.
            result.pop(name, None)
    return result
