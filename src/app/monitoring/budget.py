"""Measured allocation-wide diagnostic storage admission, shared by trial processes."""
import json
import os
from pathlib import Path

from src.app.sites.storage import record_lock, read_json, write_json


class BudgetExceeded(RuntimeError):
    pass


def limits():
    return json.loads(os.environ.get('THETA_DIAGNOSTIC_BUDGET', '{}'))


def estimate(arrays):
    maximum = limits().get('snapshot_bytes')
    if maximum and sum(getattr(value, 'nbytes', 0) for value in arrays.values()) > maximum:
        raise BudgetExceeded('Diagnostic uncompressed arrays exceed snapshot_bytes; configure explicit probe/feature sampling')


def admit(files, *, position=None, final=False):
    """files: (temporary, final path). Account replacements and admit atomically."""
    policy = limits()
    if not policy: return
    root = Path(os.environ['THETA_OUTPUT_ROOT'])
    ledger_path = root / '.thetaide/diagnostic-budget.json'
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    updates = {str(Path(target).relative_to(root)): Path(temporary).stat().st_size for temporary, target in files}
    with record_lock(ledger_path):
        ledger = read_json(ledger_path, {'files': {}, 'coverage': []})
        proposed = {**ledger['files'], **updates}
        # Keep a quarter of the execution budget for explicitly final observations.
        byte_limit = policy['diagnostic_bytes'] * (1 if final else .75)
        file_limit = policy['diagnostic_files'] * (1 if final else .75)
        reason = None
        runs = max(1, policy.get('planned_runs', 1))
        prefixes = {name.partition('/monitors/')[0] for name in updates if '/monitors/' in name}
        for prefix in prefixes:
            run_files = {name: size for name, size in proposed.items() if name.startswith(prefix + '/monitors/')}
            if sum(run_files.values()) > byte_limit / runs: reason = 'run_diagnostic_bytes'
            elif len(run_files) > file_limit / runs: reason = 'run_diagnostic_files'
        if sum(updates.values()) > policy['snapshot_bytes']: reason = 'snapshot_bytes'
        elif sum(proposed.values()) > byte_limit: reason = 'diagnostic_bytes'
        elif len(proposed) > file_limit: reason = 'diagnostic_files'
        ledger['planned_runs'] = runs
        ledger['coverage'].append({'position': position, 'final': final, 'available': reason is None, 'reason': reason})
        # Coverage itself is bounded; totals retain skipped observations after compaction.
        ledger['skipped'] = ledger.get('skipped', 0) + bool(reason)
        ledger['coverage'] = ledger['coverage'][-256:]
        if not reason: ledger['files'] = proposed
        write_json(ledger_path, ledger)
        if reason: raise BudgetExceeded('Execution diagnostic budget: ' + reason)


def bounded_json(value, **kwargs):
    """Bound metadata before committing it or embedding it in an NPZ payload."""
    maximum = limits().get('snapshot_bytes')
    chunks, size = [], 0
    for chunk in json.JSONEncoder(allow_nan=False, **kwargs).iterencode(value):
        size += len(chunk.encode())
        if maximum and size > maximum:
            raise BudgetExceeded('Diagnostic metadata exceeds snapshot_bytes')
        chunks.append(chunk)
    return ''.join(chunks)
