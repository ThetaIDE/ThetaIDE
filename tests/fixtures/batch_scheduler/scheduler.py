"""Test scheduler: receipts and job states live in files; nothing is executed or contacted.

A test sets a job's scheduler state by writing it to ``batch-state`` beside the receipt.
"""
import uuid
from src.app.api import sites as host
from src.app.api.sites import read_json, record_lock, write_json


class Scheduler:
    api_version = 2

    @staticmethod
    def validate_allocation(cfg, *, paths=None):
        resources = dict((cfg.get('site') or {}).get('resources') or {})
        resources.setdefault('nodes', int(((cfg.get('trainer') or {}).get('num_nodes')) or 1))
        return resources

    @staticmethod
    def submit(root, experiment, overrides, cfg):
        source = host.current_bundle(root)
        execution_id = uuid.uuid4().hex
        path = host.receipt_path(root, execution_id)
        path.parent.mkdir(parents=True)
        record = {'schema_version': 1, 'scheduler_component': 'batch', 'tuner_execution': None,
                  'execution_id': execution_id, 'project_id': source['project_id'], 'bundle_sha256': source['bundle_sha256'], 'experiment': experiment,
                  'overrides': list(overrides), 'site': (cfg.get('site') or {}).get('name'), 'group': cfg.get('group'),
                  'result_experiment_id': f"{cfg.get('experiment_id')}__{execution_id}", 'job_id': '1',
                  'state': 'submitted', 'scheduler_state': 'pending', 'artifacts_finalized': False}
        write_json(path, record)
        return record

    @staticmethod
    def run(root, execution_id):
        raise ValueError('The batch fixture runs no allocation workers')

    @staticmethod
    def refresh_status(root, record):
        path = host.receipt_path(root, record['execution_id'])
        state = path.parent / 'batch-state'
        if state.is_file():
            record['scheduler_state'] = state.read_text().strip()
            if record['scheduler_state'] in host.TERMINAL:
                record['state'] = record['scheduler_state']
        with record_lock(path):
            write_json(path, record)
        return record

    @classmethod
    def refresh_many(cls, root, records, *, minimum_seconds=10):
        return [cls.refresh_status(root, record) for record in records]

    @classmethod
    def cancel(cls, root, record):
        (host.receipt_path(root, record['execution_id']).parent / 'batch-state').write_text('cancelled')
        return cls.refresh_status(root, record)

    @staticmethod
    def remote_requirements(site_config):
        return [('batch-submit', None)]

    @staticmethod
    def submission_command(config, experiment, site):
        return f'batch-submit {experiment} --site {site}'

    @staticmethod
    def launch_in_terminal(root, site, experiment):
        return 0
