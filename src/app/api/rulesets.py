"""Symbolic rulesets owned by model and environment packages.

    ``resolve_ruleset``       the ruleset an environment-scoped name selects, with its files.
    ``available_rulesets``    every ruleset visible in a workspace, optionally for one environment.
    ``rules_folder_targets``  editor-action handler opening the selected ruleset's folder.
"""
from ._exports import lazy_exports

_EXPORTS = {name: f"src.app.services.rulesets:{name}"
            for name in ("resolve_ruleset", "available_rulesets", "rules_folder_targets")}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
