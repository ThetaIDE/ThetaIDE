"""Provider-owned scalar and rich monitors for the numerical schedule."""
from types import SimpleNamespace
import time
import uuid
import numpy as np
from .monitor_catalog import visual_outputs


class TrainingMonitor:
    def __init__(self, log, cfg, context):
        self.log, self.cfg, self.context = log, cfg, context
        from src.app.core.provider_catalog import get_catalog, provider_hash
        owner = get_catalog(context.workspace).resolve('algorithm', str(cfg.agent.algorithm))
        self.provider = {'package': owner[0].id, 'version': owner[0].version,
                         'capability': owner[1].id, 'sha256': provider_hash(owner[0])}
        self.step, self.updates, self.snapshots = 0, 0, 0
        self.lineage = uuid.uuid4().hex
        self.probes = None
        self.train_rows = 1
        self.last_data = None
        self.last_snapshot_updates = None
        self.snapshot_failed = False
        self.max_snapshots = int(cfg.agent.get('monitor_max_snapshots', 8))
        self.probe_count = int(cfg.agent.get('monitor_probe_count', 64))
        self.declarations = {row['name']: row for row in visual_outputs()}

    def wanted(self, name):
        return self.log.wants_monitor(name) and self.log.metrics.keeps(name)

    def __call__(self, event, data):
        self.context.check_cancelled()
        if event == 'prepared':
            p = data['prepared']
            valid = p['row_train'] & p['valid']
            self.train_rows = int(valid.sum())
            if any(self.wanted(name) for name in ('themes/policy_response', 'themes/feature_ablation')):
                from src.app.monitoring.api import ProbeSet
                ids = np.flatnonzero(valid)
                selected = np.sort(np.random.default_rng(self.context.seed).choice(ids, min(self.probe_count, len(ids)), replace=False))
                self.probes = ProbeSet(p['policy_states'][selected].copy(), selected,
                                       {'split': 'train', 'seed': self.context.seed, 'preprocessing': p.get('preprocessing', {'lag_rows':1})})
                path = self.log.version_dir / 'monitors/themes/probes.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                self.probes.save(path)
                self.log.artifact(path, kind='monitor_probes', instance='themes', sha256=self.probes.identity)
                self.median = np.median(p['policy_states'][valid], axis=0)
                names = data['data'].get('policy_base_feature_names')
                if names is None:
                    from .themes_al.data import PHYSIOLOGY
                    names = PHYSIOLOGY
                self.features = list(names) + ['max_' + n for n in names] + ['min_' + n for n in names]
            return
        self.step += 1
        if event == 'ticc':
            solvers = data['precision_solver']
            metrics = {'ticc/iteration': data['iteration']+1, 'ticc/changed_fraction': data['changed_fraction'],
                'ticc/active_stages': int(np.count_nonzero(data['counts'])),
                'ticc/solver_primal_residual': max(s['primal_residual'] for s in solvers),
                'ticc/solver_dual_residual': max(s['dual_residual'] for s in solvers),
                'ticc/solver_converged_fraction': np.mean([s['converged'] for s in solvers])}
            metrics.update({f'ticc/stage_{i}/support': count for i, count in enumerate(data['counts'])})
        elif event == 'edm':
            self.updates = data['optimizer_updates']
            metrics = {f"edm/policy_{data['policy']}/{key}": value for key, value in data.items() if key not in ('policy', 'optimizer_updates')}
        elif event == 'outer':
            metrics = {f'themes/{key}': value for key,value in data.items() if isinstance(value,(int,float))}
        elif event == 'em':
            metrics = {'em/round': data['iteration']+1, 'em/changed_fraction': data['changed_fraction'],
                       'em/nll': -data['conditional_log_likelihood']/self.train_rows,
                       'em/active_policies': len(data['component_ids'])}
            for policy, count, prior in zip(data['component_ids'], data['assigned_counts'], data['priors']):
                metrics[f'em/policy_{policy}/support'] = count
                metrics[f'em/policy_{policy}/prior'] = prior
            self.last_data = data
            self._publish_snapshot(data)
        else:
            return
        # These are already computed training quantities; optional probe work is
        # separately gated before allocation/forward passes below.
        self.log.log(metrics)

    def finish(self):
        if self.last_data and self.last_snapshot_updates != self.last_data['optimizer_updates']:
            self._publish_snapshot(self.last_data, final=True)

    def _publish_snapshot(self, data, final=False):
        if self.snapshot_failed:
            return
        from src.app.services.cancellation import TrainingCancelled
        try:
            self._snapshot(data, final=final)
        except (TrainingCancelled, KeyboardInterrupt):
            raise
        except Exception as exc:
            self.snapshot_failed = True
            self.log.event('monitor_failed', f'THEMES diagnostics skipped: {exc}', level='warning', instance='themes')

    def _snapshot(self, data, final=False):
        names = [name for name in self.declarations if self.wanted(name)]
        if not names or self.snapshots >= self.max_snapshots - (0 if final else 1):
            return
        import torch
        from src.app.monitoring.api import Context, Result, ProbeSet, preserve_rng, state_digest, write_snapshot
        from src.app.telemetry.visuals import validate_visual_data
        learner = data['learner']
        started = time.perf_counter()
        arrays, payloads = {}, {}
        with preserve_rng(), torch.no_grad():
            if self.probes is not None:
                inputs = torch.as_tensor(self.probes.observations)
                responses = np.column_stack([torch.softmax(getattr(learner, "beta", 1.) * model(inputs), dim=1)[:, 1].numpy() for model in learner.models])
                if 'themes/policy_response' in names:
                    arrays['responses'] = responses
                    payloads['themes/policy_response'] = {'schema': 'themes.diagnostics/v1',
                        'layers': [{'id': 'policies', 'array': 'responses', 'row_axis': 'training probes',
                                    'column_axis': 'policies', 'feature_ids': learner.component_ids,
                                    'value_range': [0, 1], 'unit': 'probability', 'value_semantics': 'P(binary action = 1)'}],
                        'component_ids': learner.component_ids}
                if 'themes/feature_ablation' in names:
                    sensitivity = np.empty((inputs.shape[1], len(learner.models)))
                    for i in range(inputs.shape[1]):
                        replaced = inputs.clone(); replaced[:, i] = float(self.median[i])
                        probability = np.column_stack([torch.softmax(getattr(learner, "beta", 1.) * model(replaced), dim=1)[:, 1].numpy() for model in learner.models])
                        sensitivity[i] = (probability-responses).mean(axis=0)
                    arrays['feature_replacement'] = sensitivity
                    payloads['themes/feature_ablation'] = {'schema': 'themes.diagnostics/v1',
                        'layers': [{'id': 'features', 'array': 'feature_replacement', 'row_axis': 'features',
                                    'column_axis': 'policies', 'feature_ids': learner.component_ids, 'row_ids': self.features,
                                    'unit': 'probability change', 'value_semantics': 'Training-median replacement ΔP'}],
                        'feature_names': self.features, 'component_ids': learner.component_ids,
                        'interpretation': 'Training-median feature replacement, not a causal intervention'}
            if 'themes/training_membership' in names:
                arrays['membership'] = np.asarray(data['probabilities']).max(axis=1)
                payloads['themes/training_membership'] = {'schema': 'themes.diagnostics/v1', 'array': 'membership'}
            result = Result(arrays=arrays, details={'round': data['iteration']+1, 'split': 'train',
                'component_ids': learner.component_ids, 'axis': 'EM round'}, visuals={})
            for name, payload in payloads.items():
                validate_visual_data(self.declarations[name], payload, arrays)
                result.visuals[name] = {**self.declarations[name], 'data': payload}
            if not result.visuals: return
            probes = self.probes or ProbeSet(np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=int), {'split': 'train'})
            context = Context('themes', {'target': 'policy_mixture'}, SimpleNamespace(score_kind='action_probability'),
                probes, 0, data['optimizer_updates'], self.snapshots, lineage=self.lineage)
            model_hash = state_digest(torch.nn.ModuleList(learner.models))
            path = write_snapshot(self.log.version_dir/'monitors/themes', context, result,
                model_hash=model_hash, provider=self.provider,
                elapsed=time.perf_counter()-started, final=final)
        self.snapshots += 1
        self.last_snapshot_updates = data['optimizer_updates']
        self.log.artifact(path, kind='monitor_snapshot', instance='themes', optimizer_updates=data['optimizer_updates'],
            visuals=[{k:v for k,v in visual.items() if k != 'data'} for visual in result.visuals.values()])
