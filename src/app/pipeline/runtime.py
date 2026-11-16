"""Centralized runtime environment resolution.

Single source of truth for:
  - PROJECT_ROOT
  - Python executable path
  - PYTHONPATH for subprocesses
"""

import os
import sys
from pathlib import Path

# Resolved once at import time — always points to the repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_python_executable(site_cfg=None, *, paths=None) -> str:
    """Resolve the Python executable, honoring site config.

    Resolution order:
      1. site_cfg.venv_dir / bin / python3  (if it exists)
      2. venv/bin/python3  (default venv)
      3. sys.executable    (fallback)
    """
    if os.environ.get('THETA_PYTHON_EXECUTABLE'):
        return os.environ['THETA_PYTHON_EXECUTABLE']
    if site_cfg and site_cfg.get('python_executable') not in (None, 'auto'):
        return str(site_cfg['python_executable'])
    venv_dir = "venv"
    if site_cfg and hasattr(site_cfg, "venv_dir"):
        venv_dir = str(getattr(site_cfg, "venv_dir", "venv"))

    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths.for_process()
    venv_python = paths.root / venv_dir / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def get_pythonpath_entries(site_cfg=None, *, paths=None) -> list:
    """Build PYTHONPATH entries list.

    Always includes PROJECT_ROOT and src directories (app, usr, models, environments, eval).
    Adds site_cfg.extra_pythonpath entries (relative to PROJECT_ROOT).
    """
    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths.for_process()
    entries = [str(paths.application)]
    if site_cfg:
        for extra in getattr(site_cfg, "extra_pythonpath", []) or []:
            entries.append(str(paths.root / extra))
    return entries


def get_subprocess_env(site_cfg=None, *, paths=None) -> dict:
    """Build a subprocess environment dict with correct PYTHONPATH and extras."""
    env = os.environ.copy()
    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths.for_process()
    env.update(paths.process_environment())
    pp_entries = get_pythonpath_entries(site_cfg, paths=paths)
    env["PYTHONPATH"] = ":".join(pp_entries) + ":" + env.get("PYTHONPATH", "")
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    if site_cfg:
        for k, v in (getattr(site_cfg, "extra_env_vars", {}) or {}).items():
            env[str(k)] = str(v)
    if os.environ.get('THETA_MANAGED_THREADS'):
        from src.app.services.resource_budget import managed_environment
        env.update(managed_environment(int(os.environ['THETA_MANAGED_THREADS'])))
    if os.environ.get('THETA_RESOURCE_POOL'):
        env.update({key: value for key, value in os.environ.items()
                    if key.startswith(('THETA_RESOURCE_', 'THETA_MANAGED_')) or key == 'THETAIDE_SETTINGS_FILE'})
    return env
