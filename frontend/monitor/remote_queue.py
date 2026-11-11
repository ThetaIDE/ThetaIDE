"""Read-only queue cards for allocations discovered by manual site sync."""
from ..model import remote_run_is_active


def queue_id(run):
    remote = run.get('remote') or {}
    identity = remote.get('execution_id') or run.get('id') or remote.get('job_id')
    return f'remote:{remote.get("site", "")}:{identity}' if remote and identity else None


def find_run(runs, job_id):
    return next((run for run in runs if queue_id(run) == job_id), None)


def queue_jobs(runs):
    jobs = {}
    for run in runs:
        if not remote_run_is_active(run): continue
        key = queue_id(run)
        if key is None: continue
        remote, backend = run['remote'], run.get('backend') or {}
        config = remote.get('config') or {}
        methods = list(backend.get('agents') or (config.get('methods') or {}).keys())
        note = f'{remote.get("site")} · Slurm job {remote.get("job_id") or "unknown"}'
        note += f' · last synced {remote.get("last_sync") or "unknown"}'
        if run.get('description'): note += '\n' + run['description']
        job = jobs.setdefault(key, {
            'job_id': key, 'remote': remote, 'status': run['status'],
            'experiment_id': remote.get('experiment') or backend.get('experiment_id') or run.get('id'),
            'description': note, 'experiment_description': note,
            'agents': [], 'total_timesteps': backend.get('total_timesteps') or config.get('total_timesteps')})
        job['agents'] = list(dict.fromkeys([*job['agents'], *methods]))
        job['queue_root'] = bool(job['agents'])
    return list(jobs.values())
