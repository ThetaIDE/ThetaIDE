"""API-1 Slurm allocation-policy adapter; no scheduler implementation is bundled."""
from .scheduler_api import scheduler_object


def __getattr__(name):
    if name not in {'normalize', 'submission_environment', 'CPU_PARTITIONS', 'EXTRA_FLAGS'}:
        raise AttributeError(name)
    return scheduler_object('slurm', 'resources.py:' + name)
