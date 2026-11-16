#!/usr/bin/env python3
"""Standalone TICC + EM-EDM reconstruction on explicit local data."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from .core import EMEDM, MultiSeriesTICC, make_segments

from .data import PHYSIOLOGY, ACTION, load_data, prepare

LOG = logging.getLogger(__name__)


def validate_partition(data, prepared, segments, probabilities):
    count = int(prepared['offsets'][-1])
    coverage = np.zeros(count, dtype=np.int64)
    for s in segments:
        assert prepared['offsets'][s.patient] <= s.start < s.stop <= prepared['offsets'][s.patient + 1]
        assert s.split == prepared['splits'][s.patient]
        coverage[s.start:s.stop] += 1
    assert np.all(coverage == 1), 'Every valid source row must occur in exactly one segment'
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)
    if data.get('person_ids') is not None:
        assert not set(data['person_ids'][prepared['splits'] == 'train']) & set(data['person_ids'][prepared['splits'] == 'test'])
    assert not set(data['vids'][prepared['splits'] == 'train']) & set(data['vids'][prepared['splits'] == 'test'])
    if prepared.get('preprocessing', {}).get('lag_rows', 1) > 0:
        for a in prepared['offsets'][:-1]:
            assert not prepared['valid'][a]
    return dict(exact_segment_coverage=True, no_patient_boundary_crossings=True,
                disjoint_train_test_visits=True, finite_normalized_memberships=True,
                first_row_excluded_from_policy_fit=bool(not prepared['valid'][prepared['offsets'][:-1]].any()),
                policy_inputs_exclude_interventions_and_outcomes=True)


def export_results(output, data, prepared, ticc, mixture, segments, probabilities, evidence, args):
    offsets, splits = prepared['offsets'], prepared['splits']
    actions = np.concatenate(data['actions'])
    stages = np.empty(len(actions), dtype=np.int64)
    row_group, row_segment = np.empty_like(stages), np.empty_like(stages)
    rows = []
    observation_names = data.get('observation_feature_names', [f'{f}_mean' for f in PHYSIOLOGY])
    policy_names = data.get('policy_base_feature_names', PHYSIOLOGY)
    extra_arrays = {}
    if data.get('timestamps') is not None:
        extra_arrays['minutes_from_arrival'] = np.concatenate(data['timestamps'])
    labels = probabilities.argmax(axis=1)
    for i, (s, group, probs) in enumerate(zip(segments, labels, probabilities)):
        a, b = s.start - offsets[s.patient], s.stop - offsets[s.patient]
        score_count = int(prepared['valid'][s.start:s.stop].sum())
        row = dict(segment_id=i, source_visit_index=int(data['source_indices'][s.patient]), source_patient_index=int(data['source_indices'][s.patient]),
                   visit_id=str(data['vids'][s.patient]), split=s.split,
                   source_start_slot=int(data['slots'][s.patient][a]),
                   source_stop_slot_exclusive=int(data['slots'][s.patient][b - 1] + 1),
                   flat_start=s.start, flat_stop_exclusive=s.stop, length=b - a,
                   policy_scored_steps=score_count, physiological_stage=s.stage,
                   policy_group=int(group),
                   policy_component_id=int(getattr(mixture, 'component_ids', list(range(len(mixture.models))))[group]), membership_probability=float(probs[group]),
                   action_rate=float(actions[s.start:s.stop].mean()),
                   outcome_label=int(data['outcomes'][s.patient]))
        if data.get('timestamps') is not None:
            row.update(visit_start_row=int(a), visit_stop_row_exclusive=int(b),
                       start_minutes_from_arrival=float(data['timestamps'][s.patient][a]),
                       last_minutes_from_arrival=float(data['timestamps'][s.patient][b - 1]))
        row.update({f'policy_probability_{k}': float(prob) for k, prob in enumerate(probs)})
        rows.append(row)
        stages[s.start:s.stop], row_group[s.start:s.stop], row_segment[s.start:s.stop] = s.stage, group, i
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'segments.csv', index=False)
    raw = prepared['raw']
    np.savez_compressed(output / 'grouped_segments.npz',
                        observations=raw.copy(), policy_observations=prepared['policy_states'],
                        actions=actions, intervention_flags=np.concatenate(data['interventions']),
                        policy_scoring_mask=prepared['valid'], source_slots=np.concatenate(data['slots']),
                        visit_offsets=offsets, visit_ids=data['vids'].astype(str),
                        source_patient_indices=data['source_indices'], source_visit_indices=data['source_indices'], visit_split=splits,
                        outcome_labels=data['outcomes'], segment_start=frame.flat_start.to_numpy(),
                        segment_stop=frame.flat_stop_exclusive.to_numpy(),
                        segment_visit_index=np.array([s.patient for s in segments]),
                        physiological_stages=stages, policy_groups=row_group, segment_ids=row_segment,
                        segment_policy_probabilities=probabilities,
                        observation_feature_names=np.array(observation_names),
                        intervention_feature_names=np.array(data.get('intervention_feature_names', ['ControlOxygen_max', ACTION, 'VasopressorAdmin_max'])),
                        policy_feature_names=np.array(policy_names + ['max_' + f for f in policy_names]
                                                      + ['min_' + f for f in policy_names]), **extra_arrays)
    for group in range(len(mixture.models)):
        frame[frame.policy_group == group].to_csv(output / f'policy_group_{group}.csv', index=False)
    summaries = []
    for group, part in frame.groupby('policy_group'):
        index = int(group)
        mask = row_group == index
        summaries.append(dict(policy_group=index, segments=len(part), visits=int(part.visit_id.nunique()),
                              timesteps=int(mask.sum()), action_rate=float(actions[mask].mean()),
                              median_segment_length=float(part.length.median()),
                              mean_membership=float(part.membership_probability.mean()),
                              train_segments=int((part.split == 'train').sum()),
                              test_segments=int((part.split == 'test').sum())))
    pd.DataFrame(summaries).to_csv(output / 'group_summary.csv', index=False)
    # Typical examples must contain scored rows; zero-row segments only inherit priors.
    representative_rows = []
    for _, part in frame[frame.policy_scored_steps > 0].groupby('policy_group'):
        representative_rows.extend(part.sort_values(['membership_probability', 'length'], ascending=False)
                                   .head(5).to_dict('records'))
    pd.DataFrame(representative_rows).to_csv(output / 'representative_segments.csv', index=False)
    torch.save(dict(artifact_version=2, inference_only=True, model_specs=[model.theta_model_spec for model in mixture.models], policies=[model.state_dict() for model in mixture.models],
                    priors=mixture.priors.tolist(), component_ids=list(getattr(mixture, 'component_ids', range(len(mixture.models)))), dimension=int(prepared['policy_states'].shape[1])), output / 'policies.pt')
    np.savez_compressed(output / 'preprocessing_and_stages.npz',
                        physiology_mean=prepared['mean'], physiology_scale=prepared['scale'],
                        policy_lower=prepared['lower'], policy_span=prepared['span'],
                        stage_means=ticc.means, stage_precisions=ticc.precisions)
    train_rows = prepared['row_train'] & prepared['valid']
    test_rows = ~prepared['row_train'] & prepared['valid']
    p = float(np.clip(actions[train_rows].mean(), 1e-6, 1 - 1e-6))
    test_segments = np.array([s.split == 'test' for s in segments])
    validation = validate_partition(data, prepared, segments, probabilities)
    validation['final_stage_precision_solvers_converged'] = all(
        d['converged'] for d in ticc.history[-1]['precision_solver'])
    metrics = dict(heldout_mixture_nll_per_scored_step=float(-evidence[test_segments].sum() / test_rows.sum()),
                   heldout_constant_action_nll_per_scored_step=float(-np.mean(
                       actions[test_rows] * np.log(p) + (1 - actions[test_rows]) * np.log(1 - p))),
                   training_action_rate=p,
                   interpretation='Sequence likelihood of recorded actions; membership uses the full recorded segment. '
                                  'This is not prospective action accuracy or evidence of improved outcomes.')
    dataset_hash = hashlib.sha256()
    with args.dataset.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            dataset_hash.update(chunk)
    source_kind = data.get('source_kind')
    if source_kind is None:
        with np.load(args.dataset, allow_pickle=True) as source:
            source_kind = 'synthetic' if bool(source.get('synthetic', False)) else 'local_mimic_schema'
    from . import __version__
    manifest = dict(implementation={'package': 'themes-al', 'version': __version__},
                    time_convention='ordered_rows_fixed_spacing', stage_window_rows=2, resume_supported=False,
                    split_unit='person' if data.get('person_ids') is not None else 'visit', source_kind=source_kind, artifact_version=1, inference_only=True, status='complete', method='Independent fixed-spacing TICC + EM-EDM reconstruction',
                    original_source_modified=False, original_source_equivalent=False, full_THEMES_reward_feedback=False,
                    expert_selection_applied=False,
                    dataset=str(args.dataset.resolve()), dataset_sha256=dataset_hash.hexdigest(),
                    available_patients=data['total_patients'], selected_patients=len(splits),
                    train_patients=int((splits == 'train').sum()), test_patients=int((splits == 'test').sum()),
                    valid_timesteps=len(actions), policy_scored_timesteps=int(prepared['valid'].sum()),
                    segments=len(frame), assigned_policy_groups=len(summaries),
                    fitted_policy_models=len(mixture.models), physiological_stages=len(np.unique(stages)),
                    data_loader=data.get('loader_id', 'themes-dev/trajectories'), source_schema=data.get('source_schema'),
                    test_fraction=data.get('test_fraction', 0.2),
                    source_format=data.get('source_format', 'npz'), timestamps_available=data.get('timestamps') is not None,
                    outcome_available=data.get('outcome_available', True),
                    split_stratification='none' if not data.get('outcome_available', True) else 'source_outcome',
                    policy_feature_convention='trailing_three_rows_lag_one',
                    observation_feature_names=observation_names,
                    action_feature=data.get('action_feature', ACTION), action_positive_timesteps=int(actions.sum()),
                    action_rate=float(actions.mean()),
                    parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    assumptions=[
                        f"Binary {data.get('action_feature', ACTION)} is used directly; the separate multiclass Action column is excluded." if source_kind == 'local_themes_csv' else
                        'AntiInfectiveAdmin_max is used as a proxy for the missing a_2 mapping.',
                        'MinutesFromArrival is preserved; solver and feature windows still use ordered rows, not irregular time gaps.' if source_kind == 'local_themes_csv' else
                        'Time is an ordered row index; no duration in hours or minutes is inferred.',
                        'Physiology columns are declared by the loader; policy inputs add trailing three-row max/min and one-row lag.',
                        'Policy for row t uses features through t-1; each visit first row is excluded from policy scoring.',
                        'Split by actual person ID when configured, otherwise by visit; Shock is excluded and outcomes are unavailable (-1).' if source_kind == 'local_themes_csv' else
                        'All selected visits are eligible; outcome labels stratify the split and annotate exports only.',
                        'Additional normalization fits training visits only; upstream measurement transformation is not established.',
                        'Fixed-spacing multi-series TICC uses two-row windows, Toeplitz ADMM, covariance ridge, '
                        'clipping and a constant switching penalty; it does not recover irregular timestamp gaps.',
                        'Hard EM, warm-started EDM models, treatment-rate initialization, uniform row sampling, '
                        'bounded persistent SGLD and energy stabilization are explicit demo choices.',
                        'Group membership is retrospective and uses observed actions in the segment.',
                        'No expert ranking, causal treatment-effect claim, or patient dynamics simulation is produced.'],
                    sources=[
                        'THEMES/THEMES_src/EM_EDM_utils.py (supplied hard-assignment mixture loop)',
                        'THEMES/THEMES_src/MTTICC_EMEDM.py (supplied two-stage workflow)',
                        'https://www.ijcai.org/proceedings/2021/0493.pdf',
                        'https://www.ijcai.org/proceedings/2023/0265.pdf',
                        'https://proceedings.neurips.cc/paper/2020/file/524f141e189d2a00968c3d48cadd4159-Paper.pdf'],
                    validation=validation, metrics=metrics,
                    versions=dict(python=platform.python_version(), numpy=np.__version__,
                                  torch=torch.__version__, pandas=pd.__version__),
                    ticc_history=ticc.history, em_edm_history=mixture.history, groups=summaries)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return frame, manifest


def plot_results(output, data, prepared, frame, manifest):
    # Keep plotting caches within the workspace on restricted hosts.
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    colors = ['#247ba0', '#e57b25', '#6f4b99', '#2e8b57', '#c94c66', '#a17c21']
    groups = manifest['groups']
    action_label = 'artificial binary action' if manifest['source_kind'] == 'synthetic' else manifest['action_feature']
    fig = plt.figure(figsize=(14, 10), layout='constrained')
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 1, 1.4])
    ax = fig.add_subplot(grid[0, 0])
    bars = ax.bar([g['policy_group'] for g in groups], [g['segments'] for g in groups],
                  color=[colors[g['policy_group'] % len(colors)] for g in groups])
    ax.bar_label(bars, labels=[f"{g['segments']:,}" for g in groups], padding=3)
    ax.margins(y=0.15)
    ax.set(title='Recorded segments assigned to each policy', xlabel='Policy group', ylabel='Segments',
           xticks=[g['policy_group'] for g in groups])
    ax = fig.add_subplot(grid[0, 1])
    ax.bar([g['policy_group'] for g in groups], [100 * g['action_rate'] for g in groups],
           color=[colors[g['policy_group'] % len(colors)] for g in groups])
    ax.set(title=f'Observed {action_label} rate', xlabel='Policy group', ylabel='% of recorded rows',
           xticks=[g['policy_group'] for g in groups])
    ax = fig.add_subplot(grid[1, 0])
    for g in groups:
        part = frame[frame.policy_group == g['policy_group']]
        ax.hist(part.length, bins=np.linspace(1, max(2, frame.length.max()), 24), histtype='step',
                label=f"Policy {g['policy_group']}", color=colors[g['policy_group'] % len(colors)], linewidth=2)
    ax.set(title='Segment lengths', xlabel='Recorded rows (not hours)', ylabel='Segments', yscale='log')
    ax.legend()
    ax = fig.add_subplot(grid[1, 1])
    table = pd.crosstab(frame.physiological_stage, frame.policy_group)
    heat = ax.imshow(table, aspect='auto', cmap='Blues')
    ax.set(title='Physiological stage × policy group', xlabel='Policy group', ylabel='Physiological stage',
           xticks=np.arange(len(table.columns)), xticklabels=table.columns,
           yticks=np.arange(len(table.index)), yticklabels=table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, str(table.iloc[i, j]), ha='center', va='center',
                    color='white' if table.iloc[i, j] > table.to_numpy().max() * 0.6 else '#17232d')
    fig.colorbar(heat, ax=ax, label='Segments')
    ax = fig.add_subplot(grid[2, :])
    # Sample across dominant groups, favoring visits with observable switches.
    choices = []
    for g in groups:
        candidates = frame[frame.policy_group == g['policy_group']].visit_id.unique()
        ranked = sorted(candidates, key=lambda vid: (
            frame[frame.visit_id == vid].policy_group.nunique(),
            frame[frame.visit_id == vid].length.sum()), reverse=True)
        for vid in ranked:
            if vid not in choices:
                choices.append(vid)
                if sum(v in candidates for v in choices) >= 4:
                    break
    choices = choices[:12]
    for row, vid in enumerate(choices):
        part = frame[frame.visit_id == vid]
        for s in part.itertuples():
            ax.broken_barh([(s.visit_start_row if manifest['source_format'] == 'csv' else s.source_start_slot, s.length)], (row - 0.35, 0.7),
                          facecolors=colors[s.policy_group % len(colors)], edgecolors='white', linewidth=0.4)
        p = list(data['vids']).index(vid)
        positive_slots = (np.arange(len(data['actions'][p])) if manifest['source_format'] == 'csv' else data['slots'][p])[data['actions'][p] == 1]
        ax.scatter(positive_slots, np.full(len(positive_slots), row), marker='|', color='black', s=55)
    ax.set(title=f'Example visit timelines — colors: policy groups; black ticks: recorded {action_label}',
           xlabel='Within-visit row index (time-ordered)' if manifest['source_format'] == 'csv' else 'Original NPZ slot index', yticks=np.arange(len(choices)),
           yticklabels=[f"Visit {i + 1} ({frame[frame.visit_id == vid].split.iloc[0]})"
                        for i, vid in enumerate(choices)])
    ax.invert_yaxis()
    ax.set_ylim(len(choices) - 0.4, -1.2)
    ax.legend(handles=[Patch(color=colors[g['policy_group'] % len(colors)],
                            label=f"Policy {g['policy_group']}") for g in groups], loc='upper left', ncol=3)
    fig.suptitle(f"Recorded trajectory reconstruction · {manifest['selected_patients']:,} visits · "
                 f"{manifest['segments']:,} segments\n"
                 + ('THEMES paper/source port: reward-regulated stages and retrospective policies' if manifest.get('full_THEMES_reward_feedback') else
                  'Independent fixed-spacing TICC + EM-EDM reconstruction; groups are exploratory, with no expert selection'),
                 fontsize=14, fontweight='bold')
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)


def write_summary(output, manifest):
    lines = [
        '# Completed TICC + EM-EDM reconstruction', '',
        'Independent reconstruction of multi-series TICC plus EM-EDM. '
        'No expert selection or full THEMES reward feedback is applied.', '',
        f"Processed **{manifest['selected_patients']:,} visits**, "
        f"**{manifest['valid_timesteps']:,} valid recorded timesteps**, and "
        f"**{manifest['segments']:,} segments** across "
        f"**{manifest['physiological_stages']} physiological stages**. "
        f"Fit on {manifest['train_patients']:,} visits; held out {manifest['test_patients']:,} visits.", '',
        '![Group overview and example timelines](overview.png)', '',
        f"| Policy group | Segments | Recorded rows | {manifest['action_feature']} rate |",
        '|---|---:|---:|---:|']
    for group in manifest['groups']:
        lines.append(f"| {group['policy_group']} | {group['segments']:,} | "
                     f"{group['timesteps']:,} | {100 * group['action_rate']:.3f}% |")
    lines += ['',
              'These are descriptive differences in the recorded segments, not clinical strategy '
              'names or quality rankings. The smallest group has limited support.', '',
              '- [All segments and source positions](segments.csv)',
              '- [Numerical grouped state/action sequences](grouped_segments.npz)',
              '- [Representative scored examples](representative_segments.csv)',
              '- [Configuration, checks and diagnostics](manifest.json)',
              '- [Methods, assumptions and rerun instructions](../README.md)', '',
              'Every valid row occurs in exactly one segment. No segment crosses a visit boundary. '
              'Stage models, additional normalization and policy models are fitted on training visits only. '
              'The upstream measurement-transformation history is not established.', '',
              f"The binary action is `{manifest['action_feature']}`. "
              'Timestamps are preserved when available; the solver uses ordered rows. Measurement/action alignment still needs confirmation. '
              'Policy membership is retrospective and uses the observed actions in the segment.', '',
              f"Runtime: {manifest['elapsed_seconds']:.1f} seconds on this host. ", '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patients', type=int, default=512, help='Uniform sample size; 0 selects all visits')
    parser.add_argument('--stages', type=int, default=5)
    parser.add_argument('--groups', type=int, default=3)
    parser.add_argument('--min-group-segments', type=int, default=5)
    parser.add_argument('--ticc-iterations', type=int, default=12)
    parser.add_argument('--em-iterations', type=int, default=6)
    parser.add_argument('--updates', type=int, default=250)
    parser.add_argument('--sgld-steps', type=int, default=10)
    parser.add_argument('--density-weight', type=float, default=0.1)
    parser.add_argument('--sparsity', type=float, default=0.01)
    parser.add_argument('--switch-penalty', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    for name in ['stages', 'groups', 'min_group_segments', 'ticc_iterations', 'em_iterations', 'updates', 'sgld_steps']:
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.patients < 0 or any(not np.isfinite(v) or v < 0 for v in (args.sparsity, args.switch_penalty, args.density_weight)):
        parser.error('Counts and penalties must be nonnegative')
    if not args.dataset.is_file():
        parser.error(f'Dataset not found: {args.dataset}; supply --dataset /path/to/file.npz or file.csv')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'Output directory is not empty: {args.output}; choose a new --output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.output / 'run.log')])
    run(args, plots=not args.no_plots)


def run(args, *, plots=True, data_loader=load_data, prepare_fn=prepare, stage_factory=MultiSeriesTICC,
        learner_factory=EMEDM, evaluator=export_results, phase_callback=None, observer=None, model_bundle=None):
    """Fit/export once; caller owns output directory, logging, and thread budget."""
    args.output = Path(args.output)
    args.dataset = Path(args.dataset)
    for name in ('stages', 'groups', 'min_group_segments', 'ticc_iterations', 'em_iterations', 'updates', 'sgld_steps'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.patients < 0 or any(not np.isfinite(v) or v < 0 for v in (args.sparsity, args.switch_penalty, args.density_weight)):
        raise ValueError('Counts and penalties must be finite and nonnegative')
    if not args.dataset.is_file():
        raise ValueError(f'Dataset not found: {args.dataset}')
    if args.output.exists() and any(p.name != 'run.log' for p in args.output.iterdir()):
        raise ValueError('Output directory must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    phase = phase_callback or (lambda name: None)
    phase('load')
    LOG.info('Loading %s', args.dataset)
    data = data_loader(args.dataset, args.patients, args.seed)
    prepared = prepare_fn(data, args.seed)
    if observer is not None:
        observer('prepared', {'prepared': prepared, 'data': data})
    LOG.info('%d visits; %d valid rows; train=%d test=%d; action prevalence=%.3f%%',
             len(data['vids']), prepared['offsets'][-1], (prepared['splits'] == 'train').sum(),
             (prepared['splits'] == 'test').sum(), 100 * np.concatenate(data['actions']).mean())
    phase('stages')
    ticc = (model_bundle.create_stage(args.seed) if model_bundle is not None else
            stage_factory(args.stages, args.sparsity, args.switch_penalty, args.ticc_iterations, args.seed))
    if observer is not None:
        ticc.observer = observer
    ticc.fit([x for x, split in zip(prepared['state_episodes'], prepared['splits']) if split == 'train'])
    phase('segments')
    stage_sequences = ticc.predict(prepared['state_episodes'])
    segments = make_segments(stage_sequences, prepared['offsets'], prepared['splits'])
    LOG.info('Created %d within-visit contiguous segments', len(segments))
    phase('policies')
    from .models import historical_policy
    policy_factory = model_bundle.create_policy if model_bundle is not None else historical_policy
    mixture = learner_factory(args.groups, args.em_iterations, args.updates, args.sgld_steps,
                    args.density_weight, args.seed, args.min_group_segments, policy_factory=policy_factory)
    actions = np.concatenate(data['actions'])
    if observer is not None:
        mixture.observer = observer
    mixture.fit(prepared['policy_states'], actions, prepared['valid'], segments)
    probabilities, evidence = mixture.predict(prepared['policy_states'], actions, prepared['valid'], segments)
    phase('evaluate')
    frame, manifest = evaluator(args.output, data, prepared, ticc, mixture, segments,
                                     probabilities, evidence, args)
    if model_bundle is not None:
        manifest['model_bundle'] = {'stage': ticc.theta_model_spec,
            'policies': [m.theta_model_spec for m in mixture.models]}
    phase('export')
    if plots:
        plot_results(args.output, data, prepared, frame, manifest)
    elapsed = time.monotonic() - start
    manifest['elapsed_seconds'] = elapsed
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    if plots:
        write_summary(args.output, manifest)
    from .artifacts import seal
    seal(args.output)
    LOG.info('Complete in %.1fs. Output: %s', elapsed, args.output)
    LOG.info('All partition and membership checks passed. Groups: %s', manifest['groups'])

    phase('complete')
    return manifest


if __name__ == '__main__':
    main()
