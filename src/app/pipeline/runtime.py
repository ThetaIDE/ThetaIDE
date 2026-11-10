"""Centralized runtime environment resolution.

Single source of truth for:
  - PROJECT_ROOT
  - Python executable path
  - PYTHONPATH for subprocesses
  - Shell-embeddable environment setup block for Slurm scripts
"""

import os
import sys
from pathlib import Path

# Resolved once at import time — always points to the repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def get_python_executable(site_cfg=None) -> str:
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

    venv_python = PROJECT_ROOT / venv_dir / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def get_pythonpath_entries(site_cfg=None) -> list:
    """Build PYTHONPATH entries list.

    Always includes PROJECT_ROOT and src directories (app, usr, models, environments, eval).
    Adds site_cfg.extra_pythonpath entries (relative to PROJECT_ROOT).
    """
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths.for_process()
    entries = [str(paths.application)]
    if site_cfg:
        for extra in getattr(site_cfg, "extra_pythonpath", []) or []:
            entries.append(str(PROJECT_ROOT / extra))
    return entries


def get_subprocess_env(site_cfg=None) -> dict:
    """Build a subprocess environment dict with correct PYTHONPATH and extras."""
    env = os.environ.copy()
    from src.app.workspace import WorkspacePaths
    env.update(WorkspacePaths.for_process().process_environment())
    pp_entries = get_pythonpath_entries(site_cfg)
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


def get_shell_env_block(site_cfg=None) -> str:
    """Generate shell export lines for Slurm scripts.

    Includes PROJECT_ROOT, module loads, PYTHONPATH, and extra env vars.
    Uses dynamic SLURM_SUBMIT_DIR / $(pwd) resolution so scripts remain
    depersonalized and portable across users and cluster environments.
    """
    explicit_root = getattr(site_cfg, "project_root", None) if site_cfg else None
    if explicit_root:
        lines = [f'export PROJECT_ROOT="${{PROJECT_ROOT:-{explicit_root}}}"']
    else:
        lines = ['export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"']

    # Module loads (safe initialization for non-interactive Slurm subshells)
    mod_list = getattr(site_cfg, "module_loads", []) or []
    if mod_list:
        lines.append("if ! command -v module &> /dev/null; then")
        lines.append("    [ -f /usr/share/modules/init/bash ] && source /usr/share/modules/init/bash")
        lines.append("    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh")
        lines.append("fi")
        for mod in mod_list:
            lines.append(f"command -v module &> /dev/null && module load {mod} || true")

    # Remote workspaces resolve installed/user providers from the explicitly selected tree.
    lines.extend(['export THETA_WORKSPACE="${THETA_WORKSPACE:-$PROJECT_ROOT}"',
                  'export THETA_DATA_DIR="${THETA_DATA_DIR:-$THETA_WORKSPACE/.thetaide}"'])
    # PYTHONPATH
    pp_parts = ["$PROJECT_ROOT", "$PROJECT_ROOT/src"]
    for extra in getattr(site_cfg, "extra_pythonpath", []) or []:
        pp_parts.append(f"$PROJECT_ROOT/{extra}")
    pp_parts.append("$PYTHONPATH")
    lines.append(f"export PYTHONPATH={':'.join(pp_parts)}")

    snapshot = os.environ.get("THETA_PROVIDER_SNAPSHOT")
    if snapshot:
        from src.app.workspace import WorkspacePaths
        try:
            relative = Path(snapshot).relative_to(WorkspacePaths.for_process().root)
            lines.append(f'export THETA_PROVIDER_SNAPSHOT="$THETA_WORKSPACE/{relative.as_posix()}"')
        except ValueError:
            import shlex
            lines.append("export THETA_PROVIDER_SNAPSHOT=" + shlex.quote(snapshot))
    # Extra env vars
    for k, v in (getattr(site_cfg, "extra_env_vars", {}) or {}).items():
        lines.append(f"export {k}={v}")

    return "\n".join(lines) + "\n"


def get_shell_python_cmd(site_cfg=None) -> str:
    """Return the shell-embeddable python command for Slurm scripts."""
    venv_dir = "venv"
    if site_cfg and hasattr(site_cfg, "venv_dir"):
        venv_dir = str(getattr(site_cfg, "venv_dir", "venv"))
    return f"$PROJECT_ROOT/{venv_dir}/bin/python3"
