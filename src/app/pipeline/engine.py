"""Engine and execution entrypoint resolution for the experiment pipeline.

Dispatches training to an explicitly configured runner, defaulting to the
generic PyTorch training script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

DEFAULT_ENGINE_SCRIPT = "src/app/train.py"

ENGINE_SCRIPT_MAP = {
    "pytorch": "src/app/train.py",
    "lightning": "src/app/train.py",
}


def runner_supports_episode_outcomes(script: str) -> bool:
    """Query the declared runner contract without importing optional providers."""
    if script == DEFAULT_ENGINE_SCRIPT:
        return True
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import resolve_resource
    try:
        source = resolve_resource(script).resolve()
    except (ValueError, FileNotFoundError):
        return False
    return any(cap.metadata.get('episode_outcomes') is True
               and package.resource(cap.entry_point.split(':', 1)[0]).resolve() == source
               for package in get_catalog().packages if package.origin != 'python_distribution'
               for cap in package.capabilities if cap.kind == 'algorithm')


def resolve_engine(
    method_cfg: Dict[str, Any] | None = None,
    cfg: Any | None = None,
) -> Tuple[str, str | None]:
    """Resolve the target script entrypoint and optional custom Python interpreter.

    Resolution order for script:
      1. method_cfg['entrypoint'] or method_cfg['runner'] or method_cfg['script']
      2. cfg.agent.entrypoint or cfg.agent.runner (if present)
      3. The selected agent profile's ``entrypoint`` or ``runner`` field
      4. method_cfg['engine'] or cfg.agent.engine -> a built-in engine or script path
      5. Fallback: "src/app/train.py"

    Resolution order for python interpreter:
      1. method_cfg['python'] or method_cfg['venv_python']
      2. method_cfg['venv'] -> resolves to <venv>/bin/python3
      3. Fallback: None (caller uses default project venv/python)
    """
    script = None
    python_bin = None

    m_cfg = method_cfg or {}

    # Check direct script declaration
    for key in ("entrypoint", "runner", "script"):
        val = m_cfg.get(key)
        if val:
            script = str(val)
            break

    # Check cfg.agent if available
    if not script and cfg is not None and hasattr(cfg, "agent") and cfg.agent is not None:
        agent_cfg = cfg.agent
        for key in ("entrypoint", "runner", "script"):
            val = getattr(agent_cfg, key, None) if not hasattr(agent_cfg, "get") else agent_cfg.get(key)
            if val:
                script = str(val)
                break

    # Resolve the generic built-in engine or use an explicit plugin script path.
    if not script:
        engine_name = m_cfg.get("engine")
        if not engine_name:
            agent_val = m_cfg.get("agent")
            if isinstance(agent_val, dict):
                agent_str = str(agent_val.get("name") or agent_val.get("algorithm") or "").lower()
            else:
                agent_str = str(agent_val or "").lower()

            if agent_str in ENGINE_SCRIPT_MAP:
                engine_name = agent_str

            if not engine_name and agent_str:
                from src.app.workspace import WorkspacePaths
                agent_yaml = WorkspacePaths.for_process().resolve_config(f"agent/{agent_str}.yaml")
                if agent_yaml.exists():
                    try:
                        import yaml

                        adata = yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
                        if isinstance(adata, dict):
                            script = adata.get("entrypoint") or adata.get("runner") or adata.get("script")
                            if script:
                                script = str(script)
                            elif "engine" in adata:
                                engine_name = adata["engine"]
                    except Exception:
                        pass

        if not engine_name and cfg is not None and hasattr(cfg, "agent") and cfg.agent is not None:
            engine_name = (
                getattr(cfg.agent, "engine", None) if not hasattr(cfg.agent, "get") else cfg.agent.get("engine")
            )

        if not script and engine_name:
            clean_engine = str(engine_name).lower().strip()
            script = ENGINE_SCRIPT_MAP.get(clean_engine, clean_engine)

    # Fallback to standard PyTorch Lightning engine
    if not script:
        script = DEFAULT_ENGINE_SCRIPT

    # Check python interpreter overrides
    if "python" in m_cfg:
        python_bin = str(m_cfg["python"])
    elif "venv_python" in m_cfg:
        python_bin = str(m_cfg["venv_python"])
    elif "venv" in m_cfg:
        venv_dir = Path(m_cfg["venv"])
        candidate = venv_dir / "bin" / "python3"
        if candidate.exists():
            python_bin = str(candidate)

    return script, python_bin
