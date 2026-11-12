"""Composable PlotMan panels from saved THEMES bundles. Never refit a model."""
from pathlib import Path, PurePosixPath
import json
import numpy as np
import pandas as pd
from plot.base import BasePlotter, PanelSpec
from src.app.workspace import WorkspacePaths


class ThemesPlotter(BasePlotter):
    name = 'themes-dev/themes'
    labels = {
        'likelihood': 'Held-out action likelihood',
        'support': 'Policy support by split',
        'action_rates': 'Observed action rates',
        'stage_policy': 'Stage × policy support',
        'segment_lengths': 'Segment lengths',
        'membership': 'Retrospective membership confidence',
        'stage_timeline': 'Stage timelines',
        'policy_timeline': 'Policy timelines',
        'stage_convergence': 'TICC assignment convergence',
        'em_convergence': 'EM training likelihood',
        'policy_training': 'Per-policy cross-entropy',
        'outer_convergence': 'THEMES outer-loop likelihood',
        'reward_matrix': 'Learned state/policy rewards',
        'reward_feedback': 'Reward feedback across rounds',
    }

    def __init__(self):
        super().__init__(self.name)

    def _bundles(self, exp_id):
        parts = PurePosixPath(str(exp_id))
        if parts.is_absolute() or '..' in parts.parts or not parts.parts:
            raise ValueError('Experiment must be a relative result selector')
        root = WorkspacePaths.for_process().results / 'checkpoints'
        roots = [root.joinpath(*parts.parts)] if len(parts.parts) > 1 else sorted(root.glob('*/' + parts.name))
        for base in roots:
            for path in sorted(base.glob('**/bundle/manifest.json')):
                if not path.resolve().is_relative_to(root.resolve()):
                    continue
                status = path.parent.parent / 'execution.json'
                if not status.is_file() or json.loads(status.read_text()).get('status') != 'completed':
                    continue
                meta = json.loads(path.read_text())
                if meta.get('artifact_version') not in (1,2) or meta.get('status') != 'complete':
                    continue
                if not (path.parent / 'segments.csv').is_file():
                    continue
                yield path.parent, meta

    def panels(self, exp_id, cli_overrides=None):
        config = {**self.default_cfg, **(cli_overrides or {})}
        result = []
        for bundle, meta in self._bundles(exp_id):
            method = bundle.parent.parent.name
            run = bundle.parent.name
            identity = f'{method}/{run}'
            for kind, label in self.labels.items():
                if kind in ('outer_convergence','reward_matrix','reward_feedback') and not meta.get('outer_history'):
                    continue
                if config.get('panels') and kind not in config['panels']:
                    continue
                result.append(PanelSpec(f'{self.name}/{identity}/{kind}', f'{label} · {method} · {run[:8]}',
                    lambda ax, k=kind, b=bundle, m=meta, c=config: self.draw_panel(ax, b, m, k, c)))
        return result

    def draw_panel(self, ax, bundle, meta, kind, config=None):
        config = config or {}
        frame = pd.read_csv(bundle / 'segments.csv', dtype={'visit_id': str})
        if 'policy_component_id' in frame:
            frame['policy_group'] = frame['policy_component_id']
        split = config.get('split', 'all')
        if split not in ('all', 'train', 'test'):
            raise ValueError('split must be all, train or test')
        part = frame if split == 'all' else frame[frame.split == split]
        if part.empty:
            return False
        groups = sorted(frame.policy_group.unique())
        palette = ['#247ba0', '#e57b25', '#6f4b99', '#2e8b57', '#c94c66']
        if kind == 'outer_convergence':
            history=meta['outer_history'];x=[h['outer_iteration'] for h in history]
            ax.plot(x,[h['train_nll'] for h in history],marker='o',label='Train')
            ax.plot(x,[h['heldout_nll'] for h in history],marker='o',label='Held out')
            ax.set(xlabel='Outer round',ylabel='Retrospective action NLL / scored row');ax.legend()
        elif kind == 'reward_feedback':
            history=meta['outer_history'];x=[h['outer_iteration'] for h in history]
            ax.plot(x,[h['reward_range'] for h in history],marker='o',label='Reward range')
            ax.plot(x,[h['feedback_std'] for h in history],marker='o',label='Feedback standard deviation')
            ax.set(xlabel='Outer round',ylabel='Normalized reward');ax.legend()
        elif kind == 'reward_matrix':
            with np.load(bundle/'round_structures.npz',allow_pickle=False) as arrays:
                rewards=arrays[f"r{len(meta['outer_history'])-1}_rewards"]
            ax.imshow(rewards,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
            ax.set(xlabel='Saved policy component',ylabel='Stage',xticks=range(rewards.shape[1]),
                xticklabels=meta['reward_component_ids'],yticks=range(len(rewards)))
        elif kind == 'likelihood':
            metrics = meta['metrics']
            values = [metrics['heldout_constant_action_nll_per_scored_step'], metrics['heldout_mixture_nll_per_scored_step']]
            bars = ax.bar(['Constant rate', 'THEMES' if meta.get('full_THEMES_reward_feedback') else 'TICC + EM-EDM'], values, color=['#a9b3bf', '#247ba0'])
            ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4)
            ax.set(ylabel='Held-out NLL / scored row (lower is better)', ylim=(0, max(values) * 1.18))
        elif kind == 'support':
            bottom = np.zeros(len(groups))
            for name, color in [('train', '#247ba0'), ('test', '#e5a150')]:
                counts = part[part.split == name].groupby('policy_group').size().reindex(groups, fill_value=0).to_numpy()
                ax.bar([str(g) for g in groups], counts, bottom=bottom, label=name, color=color)
                bottom += counts
            ax.set(xlabel='Policy group', ylabel='Segments'); ax.legend()
        elif kind == 'action_rates':
            rate = 'action_rate' if 'action_rate' in part else 'antiinfective_rate'
            values = [np.average(p[rate], weights=p.length) * 100 for _, p in part.groupby('policy_group')]
            bars = ax.bar([str(g) for g, _ in part.groupby('policy_group')], values, color=[palette[int(g)%len(palette)] for g, _ in part.groupby('policy_group')])
            ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=4)
            ax.set(xlabel='Policy group', ylabel=f"Observed {meta['action_feature']} = 1 (%)", ylim=(0, 110))
        elif kind == 'stage_policy':
            table = pd.crosstab(part.physiological_stage, part.policy_group)
            ax.imshow(table, aspect='auto', cmap='Blues')
            for i in range(len(table)):
                for j in range(len(table.columns)):
                    ax.text(j, i, str(table.iloc[i, j]), ha='center', va='center',
                            color='white' if table.iloc[i, j] > table.to_numpy().max() * .6 else '#17232d')
            ax.set(xticks=range(len(table.columns)), xticklabels=table.columns, yticks=range(len(table)),
                   yticklabels=table.index, xlabel='Policy group', ylabel='Stage; cells count segments')
        elif kind == 'segment_lengths':
            for group, p in part.groupby('policy_group'):
                ax.hist(p.length, bins=min(25, max(2, len(p))), histtype='step', label=f'Group {group}', linewidth=1.7, color=palette[int(group)%len(palette)])
            ax.set(xlabel='Recorded rows (not elapsed time)', ylabel='Segments'); ax.legend()
        elif kind == 'membership':
            scored = part[part.policy_scored_steps > 0]
            if scored.empty: return False
            for group, p in scored.groupby('policy_group'):
                ax.hist(p.membership_probability, bins=np.linspace(0, 1, 21), histtype='step', label=f'Group {group}', linewidth=1.7, color=palette[int(group)%len(palette)])
            ax.set(xlabel='Maximum retrospective membership probability', ylabel='Scored segments'); ax.legend()
        elif kind in ('stage_timeline', 'policy_timeline'):
            visits = list(part.visit_id.unique())[:int(config.get('max_visits', 6))]
            field = 'physiological_stage' if kind == 'stage_timeline' else 'policy_group'
            for i, visit in enumerate(visits):
                for row in part[part.visit_id == visit].itertuples():
                    start = row.visit_start_row if hasattr(row, 'visit_start_row') else row.source_start_slot
                    ax.broken_barh([(start, row.length)], (i-.35, .7), facecolors=palette[getattr(row, field) % len(palette)])
            ax.set(yticks=range(len(visits)), yticklabels=[f'Visit {i+1} ({part[part.visit_id == v].split.iloc[0]})' for i, v in enumerate(visits)],
                   xlabel='Within-visit row index' if 'visit_start_row' in part else 'Source slot', ylim=(len(visits)-.4, -1.7))
            from matplotlib.patches import Patch
            ids = sorted(part[field].unique())
            ax.legend(handles=[Patch(color=palette[int(identity)%len(palette)], label=str(identity)) for identity in ids],
                      title='Stage' if kind == 'stage_timeline' else 'Policy', loc='upper left',
                      ncol=min(5, len(ids)), fontsize=8, frameon=False)
        elif kind == 'stage_convergence':
            trace = meta.get('ticc_history', [])
            if not trace: return False
            ax.plot([t['iteration']+1 for t in trace], [t['changed_fraction'] for t in trace], 'o-', color=palette[0])
            ax.set(xlabel='TICC iteration', ylabel='Fraction of training assignments changed', ylim=(0, None))
        elif kind == 'em_convergence':
            trace = meta.get('em_edm_history', [])
            if not trace: return False
            rows = frame.loc[frame.split == 'train', 'policy_scored_steps'].sum()
            ax.plot([t['iteration']+1 for t in trace], [-t['conditional_log_likelihood']/rows for t in trace], 'o-', color=palette[0])
            ax.set(xlabel='EM round', ylabel='Training mixture NLL / scored row')
        elif kind == 'policy_training':
            trace = meta.get('em_edm_history', [])
            series = {}
            for t in trace:
                ids = t.get('component_ids', list(range(len(t['training_losses']))))
                for identity, losses in zip(ids, t['training_losses']):
                    if losses: series.setdefault(identity, []).append((t['iteration']+1, losses[-1]['cross_entropy']))
            if not series: return False
            for identity, points in series.items():
                x, y = zip(*points); ax.plot(x, y, 'o-', label=f'Policy {identity}' + ('' if all('component_ids' in t for t in trace) else ' (legacy slot)'))
            ax.set(xlabel='EM round', ylabel='Last sampled training cross-entropy'); ax.legend()
        else:
            raise ValueError(f'Unknown THEMES panel: {kind}')
        suffix = '' if kind in ('likelihood', 'stage_convergence', 'em_convergence', 'policy_training') else f' ({split} rows)'
        ax.set_title(self.labels[kind] + suffix)
        return True

    def run(self, exp_id, cli_overrides=None):
        import matplotlib.pyplot as plt
        config = {**self.default_cfg, **(cli_overrides or {})}
        paths = WorkspacePaths.for_process()
        selector = PurePosixPath(str(exp_id))
        if selector.is_absolute() or '..' in selector.parts: raise ValueError('Invalid experiment selector')
        directory = paths.results / 'plots' / str(selector) / 'themes'
        directory.mkdir(parents=True, exist_ok=True)
        outputs = []
        for panel in self.panels(exp_id, config):
            fig, ax = plt.subplots(figsize=tuple(config.get('figsize', [8, 4.5])))
            try:
                if panel.draw(ax):
                    fig.tight_layout()
                    name = panel.key.removeprefix(self.name + '/').replace('/', '__')
                    path = directory / (name + '.png')
                    fig.savefig(path, dpi=int(config.get('dpi', 160)))
                    outputs.append(path)
            finally:
                plt.close(fig)
        return outputs


class ThemesFullPlotter(ThemesPlotter):
    """Full-loop PlotMan entry, with its own defaults and no legacy result mixing."""
    name='themes-dev/themes_full'

    def _bundles(self,exp_id):
        for bundle,meta in super()._bundles(exp_id):
            if meta.get('implementation_mode')=='paper_source_port':
                yield bundle,meta
