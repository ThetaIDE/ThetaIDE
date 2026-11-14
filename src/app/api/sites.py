"""Remote sites for scheduler components.

See docs/reference/SCHEDULER_CONTRACT.md. Scheduler components own submission,
polling and cancellation; the host owns transport, snapshots, receipts,
publication and the user's site policy.

Profiles and connections
    ``profiles``, ``site_name``, ``load_site``, ``defaults_path``, ``connection``, ``push``
Transport
    ``ssh_arguments``, ``rsync_shell``
Workspace-contained files and locks
    ``safe_path``, ``digest``, ``read_json``, ``write_json``, ``record_lock``, ``transfer_lock``
Snapshots, receipts and continuation
    ``freeze``, ``extract``, ``receipt_path``, ``close_unfinished_runs``, ``TERMINAL``,
    ``pin_continuation``, ``restore_continuation``
Policy and publication
    ``resolve_policy``, ``stage_inputs``, ``own_work_root``, ``assert_local_scratch``, ``Publisher``
Host site command line (run on login and compute nodes)
    ``SITE_CLI_MODULE``, ``site_cli_command``
"""
from ._exports import lazy_exports

_EXPORTS = {
    "profiles": "src.app.sites:profiles",
    "site_name": "src.app.sites:site_name",
    "load_site": "src.app.sites.catalog:load",
    "defaults_path": "src.app.sites.catalog:defaults_path",
    "connection": "src.app.sites.transfer:connection",
    "push": "src.app.sites.transfer:push",
    "ssh_arguments": "src.app.sites.transport:ssh_arguments",
    "rsync_shell": "src.app.sites.transport:rsync_shell",
    **{name: f"src.app.sites.storage:{name}" for name in (
        "safe_path", "digest", "read_json", "write_json", "record_lock", "transfer_lock")},
    "freeze": "src.app.sites.snapshot:freeze",
    "extract": "src.app.sites.snapshot:extract",
    "receipt_path": "src.app.sites.execution:receipt_path",
    "close_unfinished_runs": "src.app.sites.execution:close_unfinished_runs",
    "TERMINAL": "src.app.sites.execution:TERMINAL",
    "pin_continuation": "src.app.sites.continuation:pin",
    "restore_continuation": "src.app.sites.continuation:restore",
    "resolve_policy": "src.app.sites.policy:resolve",
    "stage_inputs": "src.app.sites.policy:stage_inputs",
    "own_work_root": "src.app.sites.policy:own_work_root",
    "assert_local_scratch": "src.app.sites.policy:assert_local_scratch",
    "Publisher": "src.app.sites.publication:Publisher",
}
__all__ = sorted([*_EXPORTS, "SITE_CLI_MODULE", "site_cli_command"])
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)

SITE_CLI_MODULE = "src.app.sites"


def site_cli_command(python: str, *arguments) -> list[str]:
    """Argument vector running a host site operation (``_run``, ``_status``, ...) with ``python``."""
    return [str(python), "-m", SITE_CLI_MODULE, *(str(argument) for argument in arguments)]
