"""Completion evidence shared by remote Monitor and Results views."""


def results_complete(record: dict) -> bool:
    """Only a committed final artifact manifest marks a remote result complete.

    Managed publications seal after every producer has closed and stage-out drains.
    Legacy executions use the finalized artifact manifest receipt instead.
    Scheduler status describes the allocation, not the downloaded result snapshot.
    """
    publication = record.get('publication_manifest') or {}
    if record.get('execution_policy') or publication or record.get('publication_state'):
        return publication.get('sealed') is True or record.get('publication_state') == 'sealed'
    return record.get('artifacts_finalized') is True


def execution_terminal(record: dict) -> bool:
    """Scheduler termination is independent of publication completeness."""
    from .execution import TERMINAL
    if record.get('state') == 'submission_failed':
        return True
    state = record.get('scheduler_state')
    # A worker can finish before its allocation; prefer scheduler evidence.
    return state in TERMINAL or (not state and record.get('state') in TERMINAL)


def results_available(record: dict) -> bool:
    """Results may be inspected after termination even when stage-out failed.

    This grants read access to the downloaded snapshot, never cleanup eligibility
    or permission to synthesize a sealed publication.
    """
    return results_complete(record) or execution_terminal(record)
