"""Test tuner: the host's tuner contract with sweep expressions a grid sweeper can run.

Settings mirror a typical tuner's shape (a sampler with dependent options, a resumable
study, an optional pruner) so the host's generic tuning forms have something to render.
"""
import json
import math

from src.app.api.tuning import TunerArtifact, TunerExecutionPlan, TunerLaunchBindings
from src.app.api.workspace import WorkspacePaths

ID = 'search'


def sweep_expression(target, dimension):
    if not hasattr(dimension, 'get'):
        raise ValueError(f'Sweep dimension {target!r} must be a mapping')
    kind = dimension.get('type')
    if kind == 'choice':
        values = dimension.get('values')
        if not isinstance(values, (list, tuple)) or not values or any(v is None or v == '' for v in values):
            raise ValueError(f'Sweep dimension {target!r} requires non-empty choices')
        if any(not isinstance(v, (str, bool, int, float)) or (isinstance(v, float) and not math.isfinite(v)) for v in values):
            raise ValueError(f'Sweep dimension {target!r} choices must be finite scalar values')
        return 'choice(' + ','.join(json.dumps(v) for v in values) + ')'
    if kind not in ('int', 'float', 'log_float'):
        raise ValueError(f'Sweep dimension {target!r} has unsupported type {kind!r}')
    low, high, step = dimension.get('low'), dimension.get('high'), dimension.get('step')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (low, high)) or low > high:
        raise ValueError(f'Sweep dimension {target!r} requires ordered finite bounds')
    if kind == 'log_float':
        return f'tag(log, interval({low}, {high}))'
    if kind == 'int':
        return f'range({low}, {high}, {step or 1})'
    return f'range({float(low)}, {float(high)}, {float(step)})' if step is not None else f'interval({low}, {high})'


class SearchTuner:
    api_version = 1
    remote_api_version = 1

    def settings(self):
        fields = [
            {'key': 'n_trials', 'label': 'Trials', 'type': 'int', 'default': 40, 'min': 1, 'max': 100000},
            {'key': 'n_jobs', 'label': 'Maximum concurrent trials', 'type': 'int', 'default': 1, 'min': 1, 'max': 100000},
            {'key': 'sampler', 'label': 'Sampler', 'type': 'choice', 'default': 'adaptive', 'choices': ['adaptive', 'random']},
            {'key': 'seed', 'label': 'Sampler seed', 'type': 'int', 'default': 42, 'min': 0, 'max': 2147483647},
            {'key': 'candidates', 'label': 'Adaptive candidates', 'type': 'int', 'default': 24, 'min': 1, 'max': 100000,
             'when': {'sampler': 'adaptive'}},
            {'key': 'study_mode', 'label': 'Study', 'type': 'choice', 'default': 'new', 'choices': ['new', 'resume']},
            {'key': 'study_name', 'label': 'Study name', 'type': 'text', 'default': ''},
            {'key': 'pruner', 'label': 'Early stopping', 'type': 'choice', 'default': 'none', 'choices': ['none', 'median']},
            {'key': 'pruner_startup_trials', 'label': 'Completed trials before pruning', 'type': 'int', 'default': 5,
             'min': 0, 'max': 100000, 'when': {'pruner': 'median'}},
            {'key': 'timeout_seconds', 'label': 'Time budget (seconds)', 'type': 'int', 'default': 0, 'min': 0, 'max': 2147483647},
        ]
        for field in fields:
            field['optional'] = field['key'] not in ('n_trials', 'sampler')
            if field['key'] == 'study_name':
                field['required_when'] = {'study_mode': 'resume'}
            if field['key'] in ('seed', 'candidates'):
                field['parent'] = 'sampler'
            elif field['key'].startswith('pruner_'):
                field['parent'] = 'pruner'
        return fields

    def _options(self, cfg):
        tuning = (cfg.get('tuning') if hasattr(cfg, 'get') else None) or {}
        options = ((tuning.get('options') or {}).get(ID)) or {}
        return {field['key']: options.get(field['key'], field['default']) for field in self.settings()}

    def validate_config(self, cfg):
        options = self._options(cfg)
        for key in ('n_trials', 'n_jobs'):
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'Sweep {key} must be a positive integer.')
        if options['study_mode'] == 'resume' and not str(options['study_name'] or '').strip():
            raise ValueError('Resuming a study requires a study name.')

    def pruning_enabled(self, cfg):
        runtime = cfg.get('theta_tuning') or {}
        return runtime.get('pruner', self._options(cfg)['pruner']) != 'none'

    def validate_dimension(self, target, dimension):
        if not str(target).startswith(('agent.', 'model.', 'config.')):
            raise ValueError(f'Sweep dimension {target!r} requires an explicit agent, model, or config path')
        sweep_expression(target, dimension)

    def preview_override(self, override):
        if not override.is_sweep_override():
            return None
        return next(iter(override.sweep_iterator())) if override.is_discrete_sweep() else override.value().start

    def compile(self, overrides, **context):
        cfg, method_cfg, study_name = context['cfg'], context['method_cfg'], context['study_name']
        options = self._options(cfg)
        overrides.append('hydra/sweeper=theta_search')
        overrides.append(f'++hydra.sweeper.study_name={json.dumps(study_name)}')
        overrides.append(f"++hydra.sweeper.n_trials={options['n_trials']}")
        for key, value in (method_cfg.get('tune') or {}).items():
            full_key = key[len('config.'):] if key.startswith('config.') else key
            overrides = [arg for arg in overrides if arg.split('=', 1)[0].lstrip('+') != full_key]
            overrides.append(f'{full_key}={sweep_expression(key, value)}')
        if '--multirun' not in overrides and '-m' not in overrides:
            overrides.insert(0, '--multirun')
        return overrides

    def job_entrypoint(self, cfg, method_cfg, script, python):
        return script, python

    def study_name(self, cfg, method_name):
        explicit = str(self._options(cfg)['study_name'] or '').strip()
        return explicit.format(method=method_name) if explicit else f"{cfg.get('experiment_id')}_{method_name}"

    def storage(self, cfg):
        return None

    def prepare(self, cfg, storage, study_name):
        pass

    def finalize(self, group, experiment, method, storage, study_name):
        pass

    def execution_plan(self, cfg, *, execution_id):
        options = self._options(cfg)
        return TunerExecutionPlan(ID, execution_id,
            artifacts=(TunerArtifact('study', f'search/{execution_id}/study.sqlite', True),),
            concurrency=options['n_jobs'], trials=options['n_trials'], storage_kind='managed')

    def launch_bindings(self, cfg, *, plan, runtime, continuation=None):
        return TunerLaunchBindings(environment={'SEARCH_STUDY_PATH': str(WorkspacePaths(runtime).results / plan.artifact('study').path)})

    def continuation_config(self, tuning):
        tuning = json.loads(json.dumps(tuning))
        options = (tuning.get('options') or {}).get(ID) or {}
        for values in (tuning, options):
            for key in ('n_trials', 'n_jobs', 'study_name', 'study_mode', 'timeout_seconds'):
                values.pop(key, None)
        return tuning

    def snapshot_artifact(self, source, target, *, artifact):
        from src.app.api.tuning import sqlite_snapshot
        sqlite_snapshot(source, target)

    def restore_continuation(self, cfg, *, plan, runtime, lineage):
        pass

    def read_artifact_snapshot(self, artifact, path):
        return []
