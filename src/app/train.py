"""Hydra configuration boundary for the Theta method worker."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra
from src.app.workspace import WorkspacePaths


@hydra.main(version_base=None, config_path=str(WorkspacePaths.for_process().primary_config), config_name="config")
def main(cfg):
    from src.app.pipeline.provider_worker import execute_method
    result = execute_method(cfg)
    if result.status == "pruned":
        return {"theta_trial_status": "pruned"}
    return result.objective


def launch():
    paths = WorkspacePaths.for_process()
    search = [f"file://{p}" for p in paths.config_roots() if p.is_dir() and p != paths.primary_config]
    if not any(arg.lstrip('+').startswith('hydra.searchpath=') for arg in sys.argv[1:]):
        sys.argv.append('hydra.searchpath=[' + ','.join(search) + ']')
    return main()


if __name__ == "__main__":
    launch()
