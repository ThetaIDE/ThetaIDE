"""Portable tuner execution v1: core transports bytes; providers own state formats."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
import re

from src.app.pipeline.exceptions import ConfigurationError

SCHEMA = 'theta.tuner-execution/v1'


@dataclass(frozen=True)
class TunerArtifact:
    id: str
    path: str
    continuation: bool = False

    def __post_init__(self):
        path = PurePosixPath(self.path)
        if (not self.id or not isinstance(self.id, str) or not isinstance(self.path, str)
                or path.is_absolute() or '..' in path.parts or '\\' in self.path or ':' in self.path
                or len(path.parts) < 4 or path.parts[0] != 'results' or str(path) != self.path):
            raise ValueError('Tuner artifacts must name an ID and a relative results/<namespace>/<execution>/file path')
        if type(self.continuation) is not bool:
            raise ValueError('Tuner artifact continuation must be a boolean')


@dataclass(frozen=True)
class TunerExecutionPlan:
    component: str
    execution_id: str
    artifacts: tuple[TunerArtifact, ...] = ()
    concurrency: int = 1
    trials: int = 1
    storage_kind: str = 'none'
    distributions: tuple[str, ...] = ()
    continuation_notes: str = 'Provider-defined continuation; training checkpoints are not resumed'
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.component, str) or not self.component:
            raise ValueError('Unsupported tuner execution plan')
        if not re.fullmatch('[a-f0-9]{32}', self.execution_id):
            raise ValueError('Invalid tuner execution ID')
        for name in ('concurrency', 'trials'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Tuner execution {name} must be a positive integer')
        if self.storage_kind not in ('managed', 'external', 'none'):
            raise ValueError('Unsupported tuner storage kind')
        if not isinstance(self.continuation_notes, str) or not self.continuation_notes:
            raise ValueError('Tuner continuation notes must be a nonempty string')
        ids, paths = set(), set()
        for artifact in self.artifacts:
            if not isinstance(artifact, TunerArtifact):
                raise ValueError('Invalid tuner artifact declaration')
            if self.execution_id not in PurePosixPath(artifact.path).parts:
                raise ValueError('Tuner artifacts must be scoped to their execution ID')
            if artifact.id in ids or artifact.path in paths:
                raise ValueError('Duplicate tuner artifact ID/path')
            ids.add(artifact.id); paths.add(artifact.path)
        if sum(item.continuation for item in self.artifacts) > 1:
            raise ValueError('Portable tuner v1 supports one continuation state artifact')
        if isinstance(self.distributions, str) or any(not isinstance(x, str) or not x for x in self.distributions):
            raise ValueError('Tuner distributions must be names')

    def record(self):
        return asdict(self)

    @classmethod
    def from_record(cls, value):
        if not isinstance(value, dict) or value.get('schema') != SCHEMA:
            raise ValueError('Unsupported serialized tuner execution plan')
        for name in ('artifacts', 'distributions'):
            if not isinstance(value.get(name, []), (list, tuple)):
                raise ValueError(f'Tuner execution {name} must be a list')
        values = dict(value)
        values['artifacts'] = tuple(TunerArtifact(**item) for item in values.get('artifacts', []))
        values['distributions'] = tuple(values.get('distributions', []))
        return cls(**values)

    def artifact(self, artifact_id):
        for item in self.artifacts:
            if item.id == artifact_id:
                return item
        raise ValueError(f'Unknown tuner artifact: {artifact_id!r}')


@dataclass(frozen=True)
class TunerLaunchBindings:
    overrides: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.overrides, str) or any(not isinstance(x, str) or '=' not in x for x in self.overrides):
            raise ValueError('Tuner launch overrides must be assignments')
        if any(not arg.lstrip('+').split('=', 1)[0].startswith(('tuning.', 'theta_tuning.', 'hydra.sweeper.'))
               for arg in self.overrides):
            raise ValueError('Tuner launch bindings may only assign tuning runtime settings')
        protected = {'PYTHONPATH', 'THETA_WORKSPACE', 'THETA_DATA_DIR', 'THETA_OUTPUT_ROOT',
                     'THETA_EXECUTION_ID', 'THETA_MANAGED_THREADS', 'THETA_MANAGED_SERIAL',
                     'THETA_RESOURCE_POOL', 'THETA_RESOURCE_QUEUE_PARENT', 'THETAIDE_SETTINGS_FILE',
                     'THETA_PYTHON_EXECUTABLE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'}
        if any(not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) or key in protected
               or not isinstance(value, str) for key, value in self.environment.items()):
            raise ValueError('Tuner launch environment conflicts with host resource/workspace policy')


def remote_tuner(plan, *, paths=None, cfg=None):
    from .tuners import get_tuner
    tuner = get_tuner(cfg, name=plan.component, paths=paths)
    if getattr(tuner, 'remote_api_version', None) != 1:
        raise ConfigurationError(f'Tuning component {plan.component!r} does not support portable remote execution v1; update the component or run locally.')
    for method in ('execution_plan', 'launch_bindings', 'snapshot_artifact', 'restore_continuation', 'continuation_config'):
        if not callable(getattr(tuner, method, None)):
            raise ConfigurationError(f'Tuning component {plan.component!r} is missing remote hook {method}().')
    return tuner


def prepare_plan(cfg, execution_id, *, paths=None):
    from .tuners import tuner_name
    if cfg.get('tuning') is None:
        return None
    name = tuner_name(cfg)
    from src.app.core.provider_catalog import get_catalog
    selected = get_catalog(paths).resolve('tuner', name)
    if selected is not None:
        name = selected[1].id
    tuner = remote_tuner(TunerExecutionPlan(name, execution_id), paths=paths, cfg=cfg)
    plan = tuner.execution_plan(cfg, execution_id=execution_id)
    if not isinstance(plan, TunerExecutionPlan) or plan.component != name or plan.execution_id != execution_id:
        raise ConfigurationError('Tuner returned an inconsistent execution plan')
    return plan


def record_plan(record, *, paths=None):
    """New receipts explicitly carry a plan or null. Older receipts use a bounded reader."""
    if 'tuner_execution' not in record:
        from .tuner_execution_compat import legacy_plan
        return legacy_plan(record, paths=paths)
    value = record['tuner_execution']
    plan = TunerExecutionPlan.from_record(value) if value is not None else None
    if plan and plan.execution_id != record['execution_id']:
        raise ValueError('Receipt tuner execution identity mismatch')
    return plan


def record_config(record):
    """Read execution config; only pre-portable receipts inherit old selection."""
    if 'tuner_execution' not in record:
        from .tuner_execution_compat import legacy_config
        return legacy_config(record)
    from copy import deepcopy
    return deepcopy(record.get('config') or {})


def record_overrides(record):
    """Make a historical selection explicit for replay, rejecting conflicts."""
    import json
    overrides = list(record.get('overrides') or [])
    config = record_config(record)
    selected = (config.get('tuning') or {}).get('component')
    original = ((record.get('config') or {}).get('tuning') or {}).get('component')
    if not selected or selected == original:
        return overrides
    retained = []
    for arg in overrides:
        if arg.lstrip('+').split('=', 1)[0] != 'tuning.component':
            retained.append(arg)
            continue
        from hydra.core.override_parser.overrides_parser import OverridesParser
        try:
            value = OverridesParser.create().parse_override(arg).value()
        except Exception as exc:
            raise ConfigurationError('Invalid tuner selection override in historical receipt') from exc
        if not isinstance(value, str) or value.casefold() != selected.casefold():
            raise ConfigurationError('Historical receipt has conflicting tuner selections; resolve its config/override identity before replay')
    return retained + ['++tuning.component=' + json.dumps(selected)]


def launch_bindings(cfg, plan, runtime, *, continuation=None, paths=None):
    if plan is None:
        return TunerLaunchBindings()
    from .tuner_execution_compat import fixed_bindings
    historical = fixed_bindings(cfg, plan, Path(runtime))
    if historical is not None:
        return historical
    result = remote_tuner(plan, paths=paths).launch_bindings(
        cfg, plan=plan, runtime=Path(runtime), continuation=continuation)
    if not isinstance(result, TunerLaunchBindings):
        raise ConfigurationError('Tuner returned invalid launch bindings')
    return result


def apply_overrides(overrides, bindings):
    keys = {arg.lstrip('+').split('=', 1)[0] for arg in bindings.overrides}
    return [arg for arg in overrides if arg.lstrip('+').split('=', 1)[0] not in keys] + list(bindings.overrides)


def read_remote_snapshots(root, record, *, paths=None):
    """Read closed, downloaded artifacts through their selected tuner; never live state."""
    from src.app.workspace import WorkspacePaths
    from src.app.sites.storage import safe_path
    paths = paths or WorkspacePaths(root)
    plan = record_plan(record, paths=paths)
    if plan is None:
        return []
    reader = getattr(remote_tuner(plan, paths=paths, cfg=record_config(record)), 'read_artifact_snapshot', None)
    if not callable(reader):
        return []
    found = []
    for artifact in plan.artifacts:
        path = safe_path(root, artifact.path)
        if path.is_file():
            from .tuning_monitor import valid_study_summary
            for method, summary in reader(artifact, path):
                if valid_study_summary(summary):
                    # Provider identity comes from the selected execution plan,
                    # including historical exports that omitted it.
                    found.append((method, {**summary, 'component': plan.component}))
    return found


def sqlite_snapshot(source, target, *, seconds=20):
    """Optional generic same-node backup helper; imports no tuner implementation."""
    from src.app.sites.publication import sqlite_export
    return sqlite_export(source, target, seconds=seconds)


def commit_artifact(path, position=None):
    """Mark an atomically saved/closed artifact for incremental publication."""
    from src.app.sites.publication import committed
    return committed(path, position)


def request_artifact_snapshot(position=None):
    """Request a coalesced state snapshot within the host's configured cadence."""
    from src.app.sites.publication import request_database_export
    return request_database_export(position)


def check_artifact_admission():
    """Reject new trials when the host publication/storage budget is exhausted."""
    from src.app.sites.publication import admission
    return admission()
