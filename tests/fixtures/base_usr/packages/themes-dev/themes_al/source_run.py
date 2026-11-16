"""Complete offline THEMES paper/source schedule, with supplied model components."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,time
import numpy as np
import torch
from .core import make_segments
from .paper_methods import SourceEMEDM,discrete_dynamics,maximum_likelihood_irl


def row_rewards(learner,policy_states,actions,stages,rewards):
    """Paper equation 4: action-conditioned policy probability × high-level reward.

    Uses explicitly aligned component axes; no posterior pruning or truncation.
    """
    stage=np.concatenate(stages)
    if rewards.shape[1]!=len(learner.models) or (stage<0).any() or (stage>=len(rewards)).any():
        raise ValueError('Reward axes are not aligned with the saved stage/policy components')
    x=torch.as_tensor(policy_states,dtype=torch.float32)
    likelihoods=[]
    with torch.no_grad():
        for model in learner.models:
            logits=torch.cat([model(batch) for batch in x.split(4096)])*learner.beta
            probabilities=torch.softmax(logits,dim=1).numpy()
            likelihoods.append(probabilities[np.arange(len(actions)),actions])
    values=np.sum(np.column_stack(likelihoods)*rewards[stage],axis=1)/len(learner.models)
    offsets=np.r_[0,np.cumsum([len(x) for x in stages])]
    return [values[a:b] for a,b in zip(offsets[:-1],offsets[1:])]


def segment_rewards(segments,probabilities,rewards,offsets):
    """Source GenerateRewardsWeights: one posterior-weighted reward per segment.

    Explicit, unchanged policy axes replace source helper column truncation.
    Segments may arrive in any order; coverage and visit boundaries are checked.
    """
    probabilities=np.asarray(probabilities);rewards=np.asarray(rewards);offsets=np.asarray(offsets)
    if rewards.ndim!=2 or not rewards.shape[1] or not np.isfinite(rewards).all():
        raise ValueError('Reward table must have finite stage and policy axes')
    if probabilities.shape!=(len(segments),rewards.shape[1]) or not np.isfinite(probabilities).all() or (probabilities<0).any() or not np.allclose(probabilities.sum(axis=1),1):
        raise ValueError('Segment posterior and reward policy axes must align and normalize')
    flat=np.empty(int(offsets[-1]));coverage=np.zeros(len(flat),dtype=int)
    for segment,posterior in zip(segments,probabilities):
        if not (0<=segment.stage<len(rewards) and offsets[segment.patient]<=segment.start<segment.stop<=offsets[segment.patient+1]):
            raise ValueError('Segment stage or visit boundary is invalid')
        flat[segment.start:segment.stop]=rewards[segment.stage]@posterior/rewards.shape[1]
        coverage[segment.start:segment.stop]+=1
    if not np.all(coverage==1):raise ValueError('Segments must cover each row exactly once')
    return [flat[a:b] for a,b in zip(offsets[:-1],offsets[1:])]


def run_source(args,*,settings,source,models,evaluator,observer,phase_callback):
    from .demo import plot_results
    from .source_artifacts import save_rounds,seal_source
    output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();phase=phase_callback
    phase('load');data=source.load(args.dataset,args.patients,args.seed);prepared=source.prepare(data,args.seed)
    if 'preprocessing' not in prepared or data.get('timestamps') is None:
        raise ValueError('Full THEMES requires a temporal loader with explicit frozen preprocessing')
    if observer:observer('prepared',{'prepared':prepared,'data':data})
    actions=np.concatenate(data['actions']);train_visits=prepared['splits']=='train'
    episodes=prepared['state_episodes'];times=data['timestamps']
    train_episodes=[x for x,t in zip(episodes,train_visits) if t]
    train_times=[x for x,t in zip(times,train_visits) if t]
    feedback=None;rounds=[];outer_history=[];previous_stages=None;global_updates=0
    feedback_mode=settings.get('reward_feedback','row_action_probability')
    if feedback_mode not in {'segment_posterior','row_action_probability'}:
        raise ValueError('Unknown reward_feedback convention')
    for outer in range(settings['outer_iterations']):
        phase(f'outer_{outer+1}/stages')
        stage_factory=models.create_stage if outer==0 else getattr(models,'create_feedback_stage',models.create_stage)
        stage=stage_factory(args.seed+outer)
        if observer:stage.observer=observer
        train_feedback=None if feedback is None else [x for x,t in zip(feedback,train_visits) if t]
        stage.fit(train_episodes,timestamps=train_times,
            rewards=train_feedback)
        # Quantify baseline versus reward-aware smoothing on the training visits.
        def penalty_values(reward_arrays):
            return np.concatenate([stage.penalties(t,None if reward_arrays is None else reward_arrays[i])[1:]
                for i,t in enumerate(train_times)])
        baseline_penalties=penalty_values(None)
        applied_penalties=penalty_values(train_feedback)
        zero_penalties=penalty_values([np.zeros(len(t)) for t in train_times])
        def median(values):return float(np.median(values)) if len(values) else 0.
        penalty_diagnostics=dict(time_only_median=median(baseline_penalties),
            applied_median=median(applied_penalties),zero_reward_median=median(zero_penalties))
        stage_labels=stage.predict(episodes,timestamps=times,rewards=feedback)
        segments=make_segments(stage_labels,prepared['offsets'],prepared['splits'])
        phase(f'outer_{outer+1}/policies')
        em=dict(settings['em']);em['seed']=args.seed
        learner=SourceEMEDM(policy_factory=models.create_policy,trainer_params=settings['edm'],**em)
        def observe(event,payload):
            if observer:
                if 'optimizer_updates' in payload:payload={**payload,'optimizer_updates':payload['optimizer_updates']+global_updates}
                observer(event,payload)
        learner.observer=observe
        learner.fit(prepared['policy_states'],actions,prepared['valid'],segments)
        probabilities,evidence=learner.predict(prepared['policy_states'],actions,prepared['valid'],segments)
        phase(f'outer_{outer+1}/rewards')
        paths=[]
        for patient,is_train in enumerate(train_visits):
            if is_train:
                paths.append([(s.stage,int(probabilities[i].argmax())) for i,s in enumerate(segments) if s.patient==patient])
        reward_model=models.create_reward(stage.clusters,len(learner.models))
        dynamics,p0=discrete_dynamics(paths,stage.clusters,len(learner.models))
        # Degenerate one-segment paths contain no state transition. Do not invent an IRL estimate.
        if len(learner.models)>1 and any(len(p)>1 for p in paths):
            irl=maximum_likelihood_irl(dynamics,paths,initial_rewards=reward_model.values,**settings['irl'])
            reward_model.values[:]=irl['normalized_rewards']
            irl_status='converged' if irl['converged'] else 'optimizer_budget_or_failure'
        else:
            irl=dict(initial_nll=0.,final_nll=0.,converged=False,optimizer_message='Insufficient identifiable training stage/policy transitions',history=[])
            irl_status='unidentifiable_single_policy' if len(learner.models)==1 else 'unidentifiable_no_transitions'
        next_feedback=(segment_rewards(segments,probabilities,reward_model.values,prepared['offsets'])
            if feedback_mode=='segment_posterior' else
            row_rewards(learner,prepared['policy_states'],actions,stage_labels,reward_model.values))
        feedback=next_feedback
        train_segments=np.array([s.split=='train' for s in segments]);test_segments=~train_segments
        train_rows=prepared['row_train']&prepared['valid'];test_rows=~prepared['row_train']&prepared['valid']
        flat=np.concatenate(stage_labels)
        from sklearn.metrics.cluster import contingency_matrix
        from scipy.optimize import linear_sum_assignment
        if previous_stages is None:
            changed=1.
        else:
            table=contingency_matrix(previous_stages[prepared['row_train']],flat[prepared['row_train']])
            a,b=linear_sum_assignment(-table)
            changed=1.-float(table[a,b].sum()/table.sum())
        record=dict(outer_iteration=outer+1,stage_partition_disagreement=changed,segments=len(segments),
            active_policies=len(learner.models),component_ids=list(learner.component_ids),
            em_stop_reason=learner.stop_reason,em_accepted_iteration=learner.accepted_iteration,
            em_rejected_iterations=sum(not h['accepted'] for h in learner.history),
            switching_penalties=penalty_diagnostics,
            train_nll=float(-evidence[train_segments].sum()/train_rows.sum()),
            heldout_nll=float(-evidence[test_segments].sum()/test_rows.sum()),
            irl_initial_nll=irl['initial_nll'],irl_final_nll=irl['final_nll'],irl_status=irl_status,
            irl_message=irl['optimizer_message'],reward_range=float(np.ptp(reward_model.values)),
            feedback_std=float(np.std(np.concatenate(next_feedback))),optimizer_updates=global_updates+learner.optimizer_updates)
        outer_history.append(record)
        rounds.append(dict(stage=stage,learner=learner,reward_model=reward_model,dynamics=dynamics,p0=p0,irl=irl,feedback_mode=feedback_mode))
        if observer:observer('outer',record)
        global_updates+=learner.optimizer_updates
        # Fixed outer budget is explicit; no held-out-driven selection or early stopping.
        previous_stages=flat.copy()
    phase('evaluate')
    frame,manifest=evaluator(output,data,prepared,stage,learner,segments,probabilities,evidence,args)
    manifest.update(method='THEMES paper/source port',implementation_mode='paper_source_port',
        original_source_equivalent=False,full_THEMES_reward_feedback=True,time_convention='minutes',
        stage_window_rows=stage.window_rows,
        preprocessing=prepared['preprocessing'],policy_feature_convention='trailing_minutes_explicit_lag',
        source_reference='reference/original/THEMES_src/THEMES_learn.py',reward_feedback=feedback_mode,
        irl_objective_convention='sum_path_nll_divided_by_visit_count_squared',
        irl_gradient_method='greedy_policy_fixed_point',
        unresolved_source_dependencies=['REGTICCSplitTrajs Gaussian/threshold semantics','THEMES modified balanced replay buffer','original TICC preprocessing/stabilization'],
        unimplemented_reference_parameters={'smooth_reg_mu':[1,1],'smooth_reg_std':[.2,1],'rewards_thres':.5,'SAMPLE_BUFFER':'balanced'},
        model_bundle={'stage':stage.theta_model_spec,'initial_stage':rounds[0]['stage'].theta_model_spec,'policies':[m.theta_model_spec for m in learner.models],
                      'reward':reward_model.theta_model_spec},
        outer_history=outer_history,effective_algorithm=copy.deepcopy(settings),
        reward_component_ids=list(learner.component_ids),em_stop_reason=learner.stop_reason,
        artifact_version=2,elapsed_seconds=time.monotonic()-started,
        parameters={'seed':args.seed,'patients':args.patients},
        assumptions=['Paper/source implementation with documented differences; not exact missing-code parity.',
            'Train-only normalization; visit split unless a real person identifier is configured.',
            'Official EDM density-loss square restored; exact THEMES-modified EDM dependency is unavailable.',
            'EM rejects likelihood-degrading candidates and restores the previous accepted policies; deliberate source correction.',
            'Expert cohort is supplied, not ranked here. No causal or clinical benefit claim.',
            'Gaussian-CDF regulator interpretation; independent dimensions and explicit parameterization.',
            f'Reward feedback convention: {feedback_mode}.',
            'Membership is retrospective and uses recorded actions.'],
        sources=['https://arxiv.org/abs/2305.09070','reference/original/SOURCE_MANIFEST.json'])
    phase('export')
    save_rounds(output,rounds)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    plot_results(output,data,prepared,frame,manifest)
    lines=['# Completed THEMES paper/source port','',f"Visits: {len(times)}; train: {int(train_visits.sum())}; held out: {int((~train_visits).sum())}.",'',
        'Every saved outer round includes segmentation, EM-EDM policies, discrete reward fitting and feedback.','',
        '| Round | Held-out action NLL | IRL status | Reward range |','|---|---:|---|---:|']
    for record in outer_history:
        lines.append(f"| {record['outer_iteration']} | {record['heldout_nll']:.5f} | {record['irl_status']} | {record['reward_range']:.5f} |")
    lines+=['','Likelihood describes recorded actions with retrospective membership. It does not establish clinical benefit.',
        'See manifest.json for effective settings, model specifications, preprocessing and documented differences.']
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    seal_source(output)
    phase('complete')
    return manifest
