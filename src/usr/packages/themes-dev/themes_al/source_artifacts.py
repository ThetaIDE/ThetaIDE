"""Frozen, complete outer-loop inference. Replay saved rounds; never refit."""
from pathlib import Path
import json
import numpy as np
import torch
from .artifacts import sha256,FILES
from .models import instantiate_policy,instantiate_stage
from .paper_methods import SourceEMEDM
from .core import make_segments
from .source_run import row_rewards,segment_rewards

SOURCE_FILES=(*FILES,'round_models.pt','round_structures.npz')


def save_rounds(output,rounds):
    neural=[];structural={}
    for i,r in enumerate(rounds):
        s,m=r['stage'],r['learner']
        neural.append(dict(stage_spec=s.theta_model_spec,stage_seed=s.seed,
            policy_specs=[p.theta_model_spec for p in m.models],weights=[p.state_dict() for p in m.models],
            priors=m.priors.tolist(),component_ids=list(m.component_ids),beta=m.beta,
            dimension=m.dimension,reward_spec=r['reward_model'].theta_model_spec,
            feedback_mode=r.get('feedback_mode','row_action_probability')))
        for name,value in dict(means=s.means,precisions=s.precisions,rewards=r['reward_model'].values,
                               dynamics=r['dynamics'],p0=r['p0']).items():
            structural[f'r{i}_{name}']=value
    torch.save({'version':2,'rounds':neural},Path(output)/'round_models.pt')
    np.savez_compressed(Path(output)/'round_structures.npz',**structural)


def seal_source(output):
    output=Path(output);p=output/'manifest.json';meta=json.loads(p.read_text())
    meta['artifact_sha256']={n:sha256(output/n) for n in SOURCE_FILES}
    p.write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')


class SourceBundle:
    def __init__(self,output,*,model_resolver=None):
        self.output=Path(output);self.manifest=json.loads((self.output/'manifest.json').read_text())
        if self.manifest.get('artifact_version')!=2 or self.manifest.get('implementation_mode')!='paper_source_port':
            raise ValueError('Expected complete THEMES paper/source artifact v2')
        for n in SOURCE_FILES:
            if sha256(self.output/n)!=self.manifest['artifact_sha256'].get(n):raise ValueError(f'Artifact checksum mismatch: {n}')
        with np.load(self.output/'preprocessing_and_stages.npz',allow_pickle=False) as f:
            self.fitted={k:f[k] for k in f.files}
        checkpoint=torch.load(self.output/'round_models.pt',map_location='cpu',weights_only=True)
        if checkpoint.get('version')!=2 or not checkpoint['rounds']:raise ValueError('Missing saved outer-loop rounds')
        self.rounds=[]
        with np.load(self.output/'round_structures.npz',allow_pickle=False) as arrays:
            for i,saved in enumerate(checkpoint['rounds']):
                stage=instantiate_stage(saved['stage_spec'],saved['stage_seed'],model_resolver)
                stage.means=arrays[f'r{i}_means'];stage.precisions=arrays[f'r{i}_precisions']
                learner=object.__new__(SourceEMEDM)
                learner.beta=saved['beta'];learner.component_ids=saved['component_ids'];learner.priors=np.array(saved['priors'])
                if len(saved['policy_specs'])!=len(saved['weights']) or len(learner.priors)!=len(saved['weights']):raise ValueError('Misaligned saved policy axes')
                learner.models=[]
                for spec,weights in zip(saved['policy_specs'],saved['weights']):
                    model=instantiate_policy(spec,saved['dimension'],model_resolver);model.load_state_dict(weights);model.eval();learner.models.append(model)
                reward_spec=dict(saved['reward_spec']);name=reward_spec.pop('architecture')
                if model_resolver is None:
                    if name!='themes-dev/tabular_reward':raise ValueError(f'Reward model {name} requires its provider to reload')
                    from .rmt_ticc import TabularReward
                    factory=TabularReward
                else:factory=model_resolver(name)
                rewards=arrays[f'r{i}_rewards']
                reward=factory(states=len(rewards),actions=len(learner.models),**reward_spec)
                reward.values[:]=rewards
                if not np.isfinite(learner.priors).all() or (learner.priors<0).any() or not np.isclose(learner.priors.sum(),1):raise ValueError('Invalid saved priors')
                self.rounds.append(dict(stage=stage,learner=learner,rewards=reward.values.copy(),
                    feedback_mode=saved.get('feedback_mode','row_action_probability')))
        self.stage=self.rounds[-1]['stage'];self.mixture=self.rounds[-1]['learner']

    def predict(self,episodes,actions,*,timestamps):
        from .temporal import transform
        if len(episodes)!=len(actions) or any(np.asarray(a).shape!=(len(x),) or not np.isin(a,[0,1]).all() for x,a in zip(episodes,actions)):
            raise ValueError('Aligned binary recorded actions required per visit')
        stages,policy,valid,offsets=transform(episodes,timestamps,self.fitted,self.manifest['preprocessing'])
        feedback=None;flat_actions=np.concatenate(actions)
        for r in self.rounds:
            labels=r['stage'].predict(stages,timestamps=timestamps,rewards=feedback)
            if r['feedback_mode']=='segment_posterior':
                segments=make_segments(labels,offsets,['score']*len(episodes))
                posterior,_=r['learner'].predict(policy,flat_actions,valid,segments)
                feedback=segment_rewards(segments,posterior,r['rewards'],offsets)
            elif r['feedback_mode']=='row_action_probability':
                feedback=row_rewards(r['learner'],policy,flat_actions,labels,r['rewards'])
            else:raise ValueError('Unknown saved reward-feedback convention')
        segments=make_segments(labels,offsets,['score']*len(episodes))
        probabilities,evidence=self.mixture.predict(policy,flat_actions,valid,segments)
        return dict(stages=labels,segments=segments,probabilities=probabilities,evidence=evidence,
            policy_states=policy,valid=valid,offsets=offsets,feedback=feedback)


def verify_source(output,*,source=True):
    bundle=SourceBundle(output);output=Path(output)
    with np.load(output/'grouped_segments.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    offsets=data['visit_offsets']
    episodes=[data['observations'][a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    actions=[data['actions'][a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    times=[data['minutes_from_arrival'][a:b] for a,b in zip(offsets[:-1],offsets[1:])]
    if source:
        from .csv_data import load_csv
        path=Path(bundle.manifest['dataset'])
        if sha256(path)!=bundle.manifest['dataset_sha256']:raise ValueError('Source dataset checksum mismatch')
        original=load_csv(path,bundle.manifest['parameters']['patients'],bundle.manifest['parameters']['seed'],**bundle.manifest['source_schema'])
        for x,y in [(original['vids'],data['visit_ids']),(np.concatenate(original['actions']),data['actions']),
                    (np.concatenate(original['timestamps']),data['minutes_from_arrival']),
                    (np.concatenate(original['slots']),data['source_slots'])]:np.testing.assert_array_equal(x,y)
        np.testing.assert_allclose(np.concatenate(original['observations']),data['observations'],rtol=0,atol=0)
        episodes=original['observations']
    result=bundle.predict(episodes,actions,timestamps=times)
    np.testing.assert_array_equal(np.concatenate(result['stages']),data['physiological_stages'])
    np.testing.assert_allclose(result['policy_states'],data['policy_observations'],atol=1e-7)
    np.testing.assert_allclose(result['probabilities'],data['segment_policy_probabilities'],atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal([s.start for s in result['segments']],data['segment_start'])
    np.testing.assert_array_equal([s.stop for s in result['segments']],data['segment_stop'])
    test=np.array([data['visit_split'][s.patient]=='test' for s in result['segments']]);test_rows=np.repeat(data['visit_split']=='test',np.diff(offsets))&result['valid']
    nll=float(-result['evidence'][test].sum()/test_rows.sum())
    np.testing.assert_allclose(nll,bundle.manifest['metrics']['heldout_mixture_nll_per_scored_step'],atol=1e-10,rtol=1e-10)
    record=dict(status='passed',artifact_version=2,source_lineage_checked=source,outer_rounds=len(bundle.rounds),
                visits=len(episodes),segments=len(result['segments']),heldout_mixture_nll_per_scored_step=nll)
    (output/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
