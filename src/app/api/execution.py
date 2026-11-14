"""Execution engines, worker processes, interpreters and resource admission.

Engines and workers
    ``resolve_engine``          the selected runner script and interpreter for a method.
    ``provider_worker_command`` shell command running an engine through the host worker.
    ``task_worker_main``        entry point of the host's task worker process.
    ``task_worker_command``     argument vector starting that worker in a new process.
    ``saved_run_id``            the run identifier recorded in a saved run directory.

Interpreters and environments
    ``get_python_executable``, ``get_shell_python_cmd``, ``get_shell_env_block``

Processes
    ``ProcessOutput``, ``kill_process_tree``, ``cancellation_requested``

Resource admission (the user's resource policy stays host-enforced)
    ``ResourcePool``, ``host_capacity``, ``workload_estimate``, ``uses_accelerator``
"""
import shlex

from ._exports import lazy_exports

PROVIDER_WORKER_MODULE = "src.app.pipeline.provider_worker"
TASK_WORKER_MODULE = "src.app.pipeline.task_worker"

_EXPORTS = {
    "resolve_engine": "src.app.pipeline.engine:resolve_engine",
    "task_worker_main": "src.app.pipeline.task_worker:main",
    "saved_run_id": "src.app.pipeline.result_paths:saved_run_id",
    "get_python_executable": "src.app.pipeline.runtime:get_python_executable",
    "get_shell_python_cmd": "src.app.pipeline.runtime:get_shell_python_cmd",
    "get_shell_env_block": "src.app.pipeline.runtime:get_shell_env_block",
    "ProcessOutput": "src.app.services.process_output:ProcessOutput",
    "kill_process_tree": "src.app.services.processes:kill_process_tree",
    "cancellation_requested": "src.app.services.cancellation:requested",
    "ResourcePool": "src.app.services.resource_budget:ResourcePool",
    "host_capacity": "src.app.services.resource_budget:host_capacity",
    "workload_estimate": "src.app.services.resource_budget:workload_estimate",
    "uses_accelerator": "src.app.services.resource_budget:uses_accelerator",
}
__all__ = sorted([*_EXPORTS, "PROVIDER_WORKER_MODULE", "TASK_WORKER_MODULE", "provider_worker_command",
                  "task_worker_command"])
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)


def provider_worker_command(python_cmd: str, engine_script: str, arguments) -> str:
    """Shell command that runs ``engine_script`` with ``arguments`` through the host worker.

    ``python_cmd`` is already shell-formatted (see ``get_shell_python_cmd``).
    """
    quoted = " ".join(shlex.quote(str(argument)) for argument in arguments)
    return f"{python_cmd} -m {PROVIDER_WORKER_MODULE} {shlex.quote(str(engine_script))} {quoted}"


def task_worker_command(python: str, *arguments) -> list[str]:
    """Argument vector running the host task worker unbuffered with ``python``."""
    return [str(python), "-u", "-m", TASK_WORKER_MODULE, *(str(argument) for argument in arguments)]
