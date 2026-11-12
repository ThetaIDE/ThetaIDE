"""Theta bindings for the independent TICC + EM-EDM numerical package."""
from pathlib import Path
from types import SimpleNamespace

from .themes_al.core import EMEDM
from .themes_al.data import load_data
from .themes_al.demo import run


def _resolve(kind, name, paths):
    from src.app.core.provider_catalog import get_catalog
    selected = get_catalog(paths).resolve(kind, str(name))
    if selected is None:
        raise ValueError(f'Unresolved apprenticeship {kind}: {name}')
    if 'apprenticeship' not in selected[1].metadata.get('paradigms', []):
        raise ValueError(f'{name} does not support apprenticeship')
    return selected


def _selections(cfg, paths):
    from src.app.core.data_loaders import loader_selection
    from src.app.core.execution import algorithm_selection
    profile, algorithm = algorithm_selection(cfg, paths=paths)
    model = cfg.get('model') or {}
    name = model if isinstance(model, str) else model.get('architecture') or model.get('name')
    selected_model = _resolve('model', name, paths)
    if 'themes.models/v1' not in selected_model[1].metadata.get('output_contracts', []):
        raise ValueError('Select a composite model with explicit stage and policy providers; stage-only models cannot supply policies')
    if selected_model[1].metadata.get('backend') != 'numerical':
        raise ValueError('Apprenticeship requires a numerical stage model')
    selection, _ = loader_selection(cfg, 'data_module', paths=paths)
    if selection is None:
        raise ValueError('Apprenticeship requires a trajectory data provider')
    data = _resolve('datamodule', selection['name'], paths)
    if 'themes.trajectories/v1' not in data[1].metadata.get('output_contracts', []):
        raise ValueError('Data provider must supply themes.trajectories/v1')
    evaluator = _resolve('evaluator', cfg.get('evaluator'), paths)
    return profile, algorithm, selected_model, selection, data, evaluator


class TrajectorySource:
    """Static demonstration source; preprocessing is fitted by the numerical schedule."""
    def __init__(self, **params):
        if params:
            raise ValueError(f'Unsupported trajectory parameters: {sorted(params)}')

    def prepare(self, data, seed):
        from .themes_al.data import prepare
        return prepare(data, seed)

    def load(self, path, patients, seed):
        return load_data(path, patients, seed)


class ApprenticeshipRunner:
    execution_contract = 1

    @staticmethod
    def validate_config(cfg, *, paths):
        from src.app.pipeline.datasets import resolve_dataset_for_method
        from src.app.pipeline.tuning_config import tuning_enabled
        if any(cfg.get(k) for k in ('resume', 'recover', 'resume_from')):
            raise ValueError('Apprenticeship supports inference reload; exact training resume is unsupported')
        if tuning_enabled(cfg):
            raise ValueError('Numerical bundle tuning/promotion is not supported; run a direct method or repeats')
        from src.app.core.execution import algorithm_selection
        profile, _ = algorithm_selection(cfg, paths=paths)
        implementation = profile.get('implementation', 'original')
        allowed = {'seed', 'name', 'algorithm', 'backend', 'implementation', 'patients', 'groups',
            'min_group_segments', 'em_iterations', 'updates', 'sgld_steps', 'density_weight',
            'monitor_probe_count', 'monitor_max_snapshots'}
        if implementation == 'paper_source_port':
            allowed = {'seed','name','algorithm','backend','implementation','patients','outer_iterations','reward_feedback','em','edm','irl','monitor_probe_count','monitor_max_snapshots'}
        unknown = set(profile) - allowed
        if unknown:
            raise ValueError(f'Unsupported algorithm parameters (model choices belong in model): {sorted(unknown)}')
        if 'seed' in profile and profile['seed'] != cfg.get('seed'):
            raise ValueError('agent.seed conflicts with the run seed; no seed override is permitted')
        implementation = profile.get('implementation', 'original')
        if implementation == 'original':
            from .source_readiness import require_original_source
            require_original_source()
        if implementation not in ('demo_reconstruction','paper_source_port'):
            raise ValueError('Unknown implementation; original or explicit historical demo_reconstruction required')
        profile, _, model, selection, data, _ = _selections(cfg, paths)
        selected_models = _model_bundle(cfg, model, paths)
        if implementation == 'paper_source_port':
            _validate_source_settings(profile, selected_models)
            if 'themes.temporal/v1' not in data[1].metadata.get('output_contracts', []):
                raise ValueError('Full THEMES requires a temporal data loader')
        if implementation == 'demo_reconstruction' and selected_models.reward_spec is not None:
            raise ValueError('Historical demo does not train a reward model; use the full THEMES method')
        import math
        for key in (() if implementation == 'paper_source_port' else ('groups', 'min_group_segments', 'em_iterations', 'updates', 'sgld_steps')):
            value = profile.get(key, 1)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        density = float(profile.get('density_weight', 0.1))
        if not math.isfinite(density) or density < 0:
            raise ValueError('density_weight must be finite and nonnegative')
        dataset = resolve_dataset_for_method('apprenticeship', {}, cfg, paths=paths)
        if not dataset.is_file():
            raise ValueError('Apprenticeship requires a demonstration file')
        from src.app.core.provider_loader import load_capability
        source = load_capability(*data, paths=paths)(**selection['params'])
        validator = getattr(source, 'validate_file', None)
        if callable(validator):
            validator(dataset)

    def run(self, cfg, context):
        self.validate_config(cfg, paths=context.workspace)
        from src.app.core.execution import execute_backend
        selected_model = _selections(cfg, context.workspace)[2]
        context.apprenticeship_models = _model_bundle(cfg, selected_model, context.workspace)
        return execute_backend(cfg, context)


def execute(cfg, context):
    import time
    from omegaconf import OmegaConf
    from src.app import telemetry
    from src.app.core.execution import RunResult
    from src.app.core.provider_loader import load_capability
    from src.app.pipeline.datasets import resolve_dataset_for_method
    from src.app.pipeline.trial_artifacts import trial_log_version
    from src.app.services.cancellation import TrainingCancelled

    profile, algorithm, stage, data_selection, data, evaluation = _selections(cfg, context.workspace)
    model = cfg.model.stage.get('params', cfg.model.stage)
    source = load_capability(*data, paths=context.workspace)(**data_selection['params'])
    models = context.apprenticeship_models
    evaluator = load_capability(*evaluation, paths=context.workspace)
    output = context.artifacts / 'bundle'
    args = SimpleNamespace(dataset=resolve_dataset_for_method('apprenticeship', {}, cfg, paths=context.workspace), output=output,
        seed=context.seed, patients=int(profile.get('patients', 0)),
        stages=int(model.get('stages', 2)), sparsity=float(model.get('sparsity', 0.01)),
        switch_penalty=float(model.get('switch_penalty', 4)), ticc_iterations=int(model.get('ticc_iterations', 2)),
        groups=int(profile.get('groups', 2)), min_group_segments=int(profile.get('min_group_segments', 1)),
        em_iterations=int(profile.get('em_iterations', 2)), updates=int(profile.get('updates', 3)),
        sgld_steps=int(profile.get('sgld_steps', 1)), density_weight=float(profile.get('density_weight', 0.1)))
    sinks, interval = telemetry.settings_from_config(cfg)
    log = telemetry.init(str(cfg.group), str(cfg.experiment_id), str(cfg.agent.name), sinks=sinks,
        config=OmegaConf.to_container(cfg, resolve=True), system_interval=interval,
        version=cfg.get('variability_version') or trial_log_version(cfg))
    start = time.time()
    phases = []
    status = 'failed'
    def phase(name):
        context.check_cancelled()
        phases.append(name)
        context.metadata['phase'] = name
        print(f'[Apprenticeship] {name}', flush=True)
        log.log({'progress/phase': len(phases)})
    try:
        from .training_monitor import TrainingMonitor
        monitor = TrainingMonitor(log, cfg, context)
        if profile.get('implementation') == 'paper_source_port':
            from .themes_al.source_run import run_source
            settings={key:OmegaConf.to_container(profile[key],resolve=True) if OmegaConf.is_config(profile[key]) else profile[key] for key in ('outer_iterations','em','edm','irl')}
            settings['reward_feedback']=profile.get('reward_feedback','row_action_probability')
            manifest=run_source(args,settings=settings,source=source,models=models,evaluator=evaluator,observer=monitor,phase_callback=phase)
        else:
            manifest = run(args, data_loader=source.load, prepare_fn=source.prepare, model_bundle=models,
                       learner_factory=EMEDM, evaluator=evaluator, phase_callback=phase, observer=monitor)
        context.check_cancelled()
        metrics = {'eval/mixture_nll': manifest['metrics']['heldout_mixture_nll_per_scored_step'],
                   'eval/constant_nll': manifest['metrics']['heldout_constant_action_nll_per_scored_step'],
                   'data/segments': float(manifest['segments']),
                   'model/policies': float(manifest['fitted_policy_models']),
                   'model/stages': float(manifest['physiological_stages'])}
        monitor.finish()
        log.log(metrics)
        for name, kind in [('manifest.json', 'metadata'), ('policies.pt', 'checkpoint'),
                           ('preprocessing_and_stages.npz', 'checkpoint'), ('overview.png', 'plot'),
                           ('segments.csv', 'table'), ('grouped_segments.npz', 'data')]:
            log.artifact(output / name, kind=kind)
        context.metadata.update(backend='numerical', model=stage[1].id,
            data_provider=data[1].id, evaluator=evaluation[1].id,
            scientific_milestone=manifest['method'])
        status = 'completed'
        meanings = {'eval/mixture_nll': 'Held-out segment mixture marginal action NLL per scoreable row',
                    'eval/constant_nll': 'Held-out Bernoulli NLL using training-fitted constant action rate'}
        return RunResult(metrics=metrics,
            metric_definitions={key: {'meaning': meanings.get(key, key),
                                     'direction': 'minimize' if key.startswith('eval/') else 'none'} for key in metrics},
            artifacts=[{'path': str(output), 'kind': 'checkpoint', 'format': f"themes.inference/v{manifest['artifact_version']}", 'resume': False},
                       {'path': str(output / 'overview.png'), 'kind': 'plot', 'format': 'png'}],
            progress={'unit': 'phases', 'completed': len(phases), 'phases': phases},
            objective=metrics['eval/mixture_nll'])
    except (TrainingCancelled, KeyboardInterrupt):
        status = 'cancelled'
        raise
    finally:
        log.save_runtime_metadata(config=OmegaConf.to_container(cfg, resolve=True), training_time_seconds=time.time()-start)
        log.finish(status=status)


def _model_bundle(cfg, selected, paths):
    from omegaconf import OmegaConf
    from src.app.core.provider_loader import load_capability
    params = OmegaConf.to_container(cfg.model, resolve=True)
    if set(params) - {'name', 'architecture', 'stage', 'feedback_stage', 'policy', 'reward'}:
        raise ValueError('Model bundle accepts only stage and policy configuration; no ignored parameters')
    selectors = [params.pop(k) for k in ('name', 'architecture') if k in params]
    if len(set(selectors)) != 1:
        raise ValueError('Conflicting model selectors')
    def resolver(name):
        return load_capability(*_resolve('model', name, paths), paths=paths)
    return load_capability(*selected, paths=paths)(**params, resolver=resolver)


def _validate_source_settings(profile, models):
    import inspect, torch
    from .themes_al.paper_methods import SourceEMEDM, maximum_likelihood_irl, validate_edm_parameters
    if type(profile.get('outer_iterations')) is not int or profile['outer_iterations'] < 2:
        raise ValueError('Full reward feedback requires outer_iterations >= 2')
    for key in ('em','edm','irl'):
        if not hasattr(profile.get(key),'items'):
            raise ValueError(f'Full THEMES requires explicit {key} settings')
    if profile.get('reward_feedback','row_action_probability') not in {'segment_posterior','row_action_probability'}:
        raise ValueError('Unknown reward_feedback convention')
    em=dict(profile['em']);edm=dict(profile['edm']);irl=dict(profile['irl'])
    if 'seed' in em or 'seed' in edm or 'policy_factory' in em or 'trainer_params' in em:
        raise ValueError('Run seed and selected model bindings cannot be overridden by algorithm settings')
    validate_edm_parameters(**edm)
    if type(profile.get('patients')) is not int or profile['patients']<0:
        raise ValueError('patients must be a nonnegative integer')
    SourceEMEDM(policy_factory=models.create_policy,trainer_params=edm,seed=0,**em)
    if 'initial_rewards' in irl:
        raise ValueError('Initial rewards belong to the selected reward model, not algorithm overrides')
    inspect.signature(maximum_likelihood_irl).bind(None,None,**irl)
    for key in ('discount','beta','reward_bound','tolerance','value_iterations','optimizer_iterations'):
        if key not in irl:raise ValueError(f'Missing IRL setting: {key}')
    import math
    if any(not isinstance(irl[k],(int,float)) or not math.isfinite(irl[k]) for k in ('discount','beta','reward_bound','tolerance')):
        raise ValueError('IRL parameters must be finite numbers')
    if not 0 <= irl['discount'] < 1 or irl['beta']<=0 or irl['reward_bound']<=0 or irl['tolerance']<=0:
        raise ValueError('Invalid IRL rates or reward bounds')
    for key in ('value_iterations','optimizer_iterations'):
        if type(irl[key]) is not int or irl[key]<1:raise ValueError(f'{key} must be a positive integer')
    with torch.random.fork_rng(devices=[]):
        models.create_policy(1)
    models.create_reward(1,1)
    stage=models.create_stage(0)
    getattr(models,'create_feedback_stage',models.create_stage)(0)
    signature=inspect.signature(stage.fit)
    if 'timestamps' not in signature.parameters or 'rewards' not in signature.parameters:
        raise ValueError('Selected stage model must support time and reward regulation')
