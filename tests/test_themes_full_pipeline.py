"""Full source/paper loop, time/reward segmentation and frozen round replay."""
import itertools,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score
from test_themes_integration import workspace,launch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/usr/packages/themes-dev'))
from themes_al.temporal import aggregate
from themes_al.rmt_ticc import variable_viterbi,RewardTICC
from themes_al.source_artifacts import SourceBundle,verify_source
from themes_al.source_run import row_rewards


def stage_model():
    return RewardTICC(2,.001,4,4,2,.05,60.,[0.,1.],[.2,1.],.001,42)


def test_variable_viterbi_matches_exhaustive_paths():
    rng=np.random.default_rng(4)
    for n in range(1,6):
        cost=rng.normal(size=(n,3));penalty=rng.uniform(0,3,n)
        path=variable_viterbi(cost,penalty)
        def objective(p):return sum(cost[t,k] for t,k in enumerate(p))+sum(penalty[t]*(p[t]!=p[t-1]) for t in range(1,n))
        assert objective(path)==pytest.approx(min(objective(p) for p in itertools.product(range(3),repeat=n)))


def test_real_minutes_and_lag_are_loader_owned():
    x=np.array([[1.],[10.],[3.],[7.]])
    times=np.array([0.,20.,100.,110.])
    values,mask=aggregate([x],[times],60.,0)
    np.testing.assert_array_equal(values,[[1,1,1],[10,10,1],[3,3,3],[7,7,3]])
    lagged,valid=aggregate([x],[times],60.,1)
    np.testing.assert_array_equal(lagged[1:],values[:-1]);assert not valid[0]


def test_regulator_relaxes_for_larger_time_and_reward_changes():
    m=stage_model()
    small=m.penalties([0,1],[0,0])[1]
    long=m.penalties([0,600],[0,0])[1]
    change=m.penalties([0,1],[0,2])[1]
    assert small>long and small>change
    assert m.penalties([0,600])[1]<m.penalties([0,1])[1]


def test_stage_recovery_and_patient_boundary_isolation():
    rng=np.random.default_rng(4)
    episodes=[np.r_[rng.normal(-3,.1,(12,2)),rng.normal(3,.1,(12,2))] for _ in range(8)]
    times=[np.arange(24)*30. for _ in episodes]
    m=stage_model().fit(episodes,timestamps=times)
    labels=m.predict(episodes,timestamps=times)
    truth=np.tile(np.r_[np.zeros(12),np.ones(12)],8)
    assert adjusted_rand_score(truth,np.concatenate(labels))>.8
    alone=m.predict([episodes[0]],timestamps=[times[0]])[0]
    np.testing.assert_array_equal(alone,labels[0])


def test_reward_feedback_aligns_action_probabilities_and_model_axes():
    from types import SimpleNamespace
    class Fixed(torch.nn.Module):
        def __init__(self,logits):super().__init__();self.logits=torch.tensor(logits)
        def forward(self,x):return self.logits.expand(len(x),-1)
    mixture=SimpleNamespace(models=[Fixed([0.,0.]),Fixed([0.,np.log(3.)])],beta=1.)
    values=row_rewards(mixture,np.zeros((3,2)),np.array([0,1,1]),[np.array([0,0,1])],np.array([[2.,4.],[6.,8.]]))
    np.testing.assert_allclose(values[0],[(.5*2+.25*4)/2,(.5*2+.75*4)/2,(.5*6+.75*8)/2])
    with pytest.raises(ValueError,match='axes'):row_rewards(mixture,np.zeros((3,2)),np.array([0,1,1]),[np.array([0,0,1])],np.zeros((2,1)))


def test_complete_worker_round_replay_monitors_and_plotman(workspace):
    launch(workspace,experiment='themes_paper_synthetic')
    record_path=next(workspace.results.rglob('execution.json'));record=json.loads(record_path.read_text())
    bundle=record_path.parent/'bundle';meta=json.loads((bundle/'manifest.json').read_text())
    assert record['status']=='completed' and meta['full_THEMES_reward_feedback']
    assert meta['artifact_version']==2 and len(meta['outer_history'])==2
    assert meta['effective_algorithm']['outer_iterations']==2
    assert meta['preprocessing']['window_minutes']==60
    assert meta['preprocessing']['lag_rows']==0
    assert meta['preprocessing']['scaling']=='train_minmax_zero_one'
    assert meta['reward_feedback']=='segment_posterior'
    assert meta['stage_window_rows']==3
    assert all('em_stop_reason' in h and 'switching_penalties' in h for h in meta['outer_history'])
    assert (bundle/'round_models.pt').exists()
    assert verify_source(bundle)['outer_rounds']==2
    saved=SourceBundle(bundle)
    assert len(saved.rounds)==2 and saved.rounds[0]['stage'] is not saved.rounds[1]['stage']
    assert [r['stage'].window_rows for r in saved.rounds]==[2,3]
    assert all(r['feedback_mode']=='segment_posterior' for r in saved.rounds)
    subprocess.run([sys.executable,'-m','themes_al','verify','--output',str(bundle)],
        cwd=workspace.root,env={**os.environ,'PYTHONPATH':str(workspace.root/'src/usr/packages/themes-dev')},check=True,capture_output=True,timeout=45)
    import pandas as pd
    logs=pd.read_csv(next(workspace.results.rglob('metrics.csv')))
    assert {'themes/outer_iteration','themes/irl_final_nll','themes/feedback_std'}<=set(logs)
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_capability
    plotter=load_capability(*get_catalog(workspace).resolve('plotter','themes-dev/themes_full'),paths=workspace)()
    panels=plotter.panels('AL/codex/themes_paper_synthetic')
    assert any(p.key.endswith('/reward_matrix') for p in panels)
    from matplotlib.figure import Figure
    before={p:p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    for panel in panels:
        fig=Figure();ax=fig.add_subplot(111)
        assert panel.draw(ax),panel.key
    assert before=={p:p.read_bytes() for p in bundle.iterdir() if p.is_file()}


def test_bad_nested_training_parameter_fails_before_run(workspace):
    from src.app.pipeline.compose import compose_experiment,validate_composed
    with pytest.raises(Exception,match='unsupported|Unexpected|unexpected|Additional|additional'):
        validate_composed(compose_experiment('AL/codex/themes_paper_synthetic',
            ['++methods.themes.agent={name:themes-dev/themes,edm:{made_up_parameter:3}}'],workspace=workspace))
    assert not list(workspace.results.rglob('execution.json'))


def test_reward_feedback_changes_the_actual_viterbi_partition():
    model=stage_model();times=np.array([0.,1.,2.,3.])
    cost=np.array([[0.,5.],[0.,5.],[5.,0.],[5.,0.]])
    constant=variable_viterbi(cost,model.penalties(times,np.zeros(4)))
    change=variable_viterbi(cost,model.penalties(times,np.array([0.,0.,2.,2.])))
    assert not np.diff(constant).any()
    assert np.diff(change).any()


def test_temporal_scalers_ignore_heldout_extremes(tmp_path):
    import copy
    from test_themes_csv_loader import csv_fixture
    from themes_al.csv_data import load_csv
    from themes_al.temporal import fit_temporal
    path=tmp_path/'visits.csv';csv_fixture(path)
    data=load_csv(path,0,42)
    original=fit_temporal(data,42,test_fraction=.2,window_minutes=60.,lag_rows=1)
    changed=copy.deepcopy(data)
    for i,split in enumerate(original['splits']):
        if split=='test':changed['observations'][i][:]=1e8
    updated=fit_temporal(changed,42,test_fraction=.2,window_minutes=60.,lag_rows=1)
    for key in ('lower','span','mean','scale'):np.testing.assert_array_equal(original[key],updated[key])
    np.testing.assert_array_equal(original['policy_states'][original['row_train']],updated['policy_states'][updated['row_train']])


def test_invalid_edm_rate_fails_before_worker_launch(workspace):
    from src.app.pipeline.compose import compose_experiment,validate_composed
    with pytest.raises(Exception,match='positive|nonnegative|minimum|greater'):
        validate_composed(compose_experiment('AL/codex/themes_paper_synthetic',
            ['++methods.themes.agent={name:themes-dev/themes,edm:{learning_rate:-0.1}}'],workspace=workspace))
    assert not list(workspace.results.rglob('execution.json'))


def test_current_row_alignment_and_old_artifact_scaling_are_preserved():
    from themes_al.temporal import fit_temporal,transform,scale_policy
    from themes_al.data import prepare
    x=[np.array([[1.],[9.],[3.]])+i for i in range(10)]
    data={'observations':x,'timestamps':[np.array([0.,42.,102.]) for _ in x],
          'person_ids':None,'outcome_available':False,'outcomes':np.full(10,-1)}
    p=fit_temporal(data,42,test_fraction=.2,window_minutes=60.,lag_rows=0)
    assert p['valid'].all()
    raw,_=aggregate(x,data['timestamps'],60.,0)
    np.testing.assert_allclose(p['policy_states'],scale_policy(raw,p['lower'],p['span'],'train_minmax_zero_one',clip=False))
    fitted={'policy_lower':p['lower'],'policy_span':p['span'],
            'physiology_mean':p['mean'],'physiology_scale':p['scale']}
    _,policy,valid,_=transform(x,data['timestamps'],fitted,p['preprocessing'])
    np.testing.assert_array_equal(policy,p['policy_states'])
    old=fit_temporal(data,42,test_fraction=.2,window_minutes=60.,lag_rows=1,policy_scaling='train_minmax_minus_one_one',clip_policy_inputs=True)
    _,policy,valid,_=transform(x,data['timestamps'],
        {**fitted,'policy_lower':old['lower'],'policy_span':old['span']},old['preprocessing'])
    np.testing.assert_array_equal(policy,old['policy_states'])
    np.testing.assert_array_equal(valid,old['valid'])
