"""Loader-owned temporal aggregation and frozen transformations."""
import numpy as np
from .data import prepare


def aggregate(episodes, timestamps, window_minutes, lag_rows):
    if len(episodes)!=len(timestamps):
        raise ValueError('One timestamp array is required per visit')
    result, masks=[] , []
    for x,t in zip(episodes,timestamps):
        x,t=np.asarray(x),np.asarray(t)
        if t.shape!=(len(x),) or not np.isfinite(t).all() or (np.diff(t)<=0).any():
            raise ValueError('Timestamps must be finite and strictly increasing within each visit')
        left=np.searchsorted(t,t-window_minutes,side='left')
        low=np.array([x[a:b+1].min(axis=0) for b,a in enumerate(left)])
        high=np.array([x[a:b+1].max(axis=0) for b,a in enumerate(left)])
        raw=np.concatenate([x,high,low],axis=1)
        indices=np.maximum(np.arange(len(x))-lag_rows,0)
        result.append(raw[indices]); masks.append(np.arange(len(x))>=lag_rows)
    return np.concatenate(result),np.concatenate(masks)


def fit_temporal(data,seed,*,test_fraction,window_minutes,lag_rows,policy_scaling="train_minmax_zero_one",clip_policy_inputs=False):
    # Reuse only the established visit/person split. All temporal feature fitting is here.
    split=prepare(data,seed,test_fraction=test_fraction)
    policy_raw,valid=aggregate(data['observations'],data['timestamps'],window_minutes,lag_rows)
    train=split['row_train'] & valid
    if not train.any(): raise ValueError('No scoreable training history')
    lower=policy_raw[train].min(axis=0); span=np.ptp(policy_raw[train],axis=0);span[span<1e-8]=1
    spec=dict(window_minutes=window_minutes,lag_rows=lag_rows,scaling=policy_scaling,
              stage_scaling='train_zscore_clip5',clip_policy_inputs=clip_policy_inputs,split='person_if_available_else_visit')
    split.update(valid=valid,lower=lower,span=span,
        policy_states=scale_policy(policy_raw,lower,span,policy_scaling,clip=clip_policy_inputs),preprocessing=spec)
    return split


def transform(episodes,timestamps,fitted,spec):
    raw,valid=aggregate(episodes,timestamps,spec['window_minutes'],spec['lag_rows'])
    policy=scale_policy(raw,fitted['policy_lower'],fitted['policy_span'],spec['scaling'],clip=spec.get('clip_policy_inputs',True))
    stages=[np.clip((x-fitted['physiology_mean'])/fitted['physiology_scale'],-5,5) for x in episodes]
    offsets=np.r_[0,np.cumsum([len(x) for x in episodes])]
    return stages,policy,valid,offsets


def scale_policy(raw,lower,span,scaling,*,clip=True):
    values=(raw-lower)/span
    if clip:values=np.clip(values,0,1)
    if scaling == 'train_minmax_minus_one_one':
        values=2*values-1
    elif scaling != 'train_minmax_zero_one':
        raise ValueError(f'Unsupported policy scaling: {scaling}')
    return values.astype('float32')
