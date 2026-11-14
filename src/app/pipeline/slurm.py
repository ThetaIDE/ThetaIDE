"""API-1 Slurm header compatibility entry points; requires an installed scheduler."""
from src.app.sites.scheduler_api import scheduler_object


def __getattr__(name):
    if name not in {'get_gres_header', 'generate_sbatch_header', 'generate_sbatch_script', 'submit_sbatch'}:
        raise AttributeError(name)
    return scheduler_object('slurm', 'headers.py:' + name)
