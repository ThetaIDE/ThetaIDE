"""Independent JSON tuner: no Optuna, SQLAlchemy, or private scheduler imports."""
import json

from src.app.pipeline.tuner_execution import TunerArtifact, TunerExecutionPlan, TunerLaunchBindings


class GridTuner:
    api_version = 1
    remote_api_version = 1

    def settings(self): return []
    def validate_config(self, cfg): pass
    def validate_dimension(self, target, dimension): pass
    def compile(self, overrides, **context): return overrides
    def job_entrypoint(self, cfg, method_cfg, script, python): return script, python
    def study_name(self, cfg, method): return 'grid'
    def prepare(self, cfg, storage, study_name): pass
    def finalize(self, group, experiment, method, storage, study_name): pass
    def storage(self, cfg): return None

    def execution_plan(self, cfg, *, execution_id):
        options = ((cfg.get('tuning') or {}).get('options') or {}).get('fixture-grid') or {}
        return TunerExecutionPlan('fixture-grid', execution_id,
            artifacts=(TunerArtifact('search', f'results/grid/{execution_id}/search.json', True),),
            concurrency=options.get('workers', 1), trials=options.get('attempts', 2), storage_kind='managed')

    def launch_bindings(self, cfg, *, plan, runtime, continuation=None):
        return TunerLaunchBindings(environment={'GRID_STATE_PATH': str(runtime / plan.artifact('search').path)})

    def continuation_config(self, tuning):
        tuning = json.loads(json.dumps(tuning))
        options = (tuning.get('options') or {}).get('fixture-grid') or {}
        for key in ('workers', 'attempts'):
            options.pop(key, None)
        return tuning

    def snapshot_artifact(self, source, target, *, artifact):
        data = json.loads(source.read_text())
        target.write_text(json.dumps(data))

    def restore_continuation(self, cfg, *, plan, runtime, lineage):
        target = runtime / plan.artifact(lineage['artifact_id']).path
        data = json.loads(target.read_text())
        for trial in data['trials']:
            if trial['state'] == 'running': trial['state'] = 'failed'
        data['parent'] = lineage['parent_execution_id']
        target.write_text(json.dumps(data))

    def read_artifact_snapshot(self, artifact, path):
        data = json.loads(path.read_text())
        return [('grid-method', {'study': 'grid', 'trials': data['trials']})]
