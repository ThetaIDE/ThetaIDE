"""Explicit local MIMIC-schema adapter and frozen preprocessing. No Theta imports."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

PHYSIOLOGY = ['SystolicBP', 'MAP', 'RespiratoryRate', 'PulseOx', 'HeartRate',
              'Temperature', 'WBC', 'BiliRubin', 'BUN', 'Lactate', 'Creatinine',
              'Platelet', 'Bands', 'FIO2']
ACTION = 'AntiInfectiveAdmin_max'


def load_data(path: Path, patients: int, seed: int):
    """Unpad trusted local data and preserve both visit and original slot IDs."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        from .csv_data import load_csv
        return load_csv(path, patients, seed)
    with np.load(path, allow_pickle=True) as source:
        names = [str(n) for n in source['feature_names_x']]
        feature_indices = [names.index(f'{f}_mean') for f in PHYSIOLOGY]
        action_index = names.index(ACTION)
        intervention_indices = [names.index(f) for f in
                                ['ControlOxygen_max', ACTION, 'VasopressorAdmin_max']]
        x, mask = source['X'], source['mask'].squeeze(-1) != -1
        y, vids = source['y'].ravel(), source['vids'].astype(str)
        person_ids = source['patient_ids'].astype(str).ravel() if 'patient_ids' in source else None
        if person_ids is not None and (len(person_ids) != len(x) or np.isin(person_ids, ['', 'nan', 'None']).any()):
            raise ValueError('patient_ids must identify one actual person per source visit')
        if len(np.unique(vids)) != len(vids):
            raise ValueError('Duplicate visit IDs: patient split must be revised before proceeding')
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Expected binary outcome labels')
        indices = np.arange(len(x))
        if 0 < patients < len(indices):
            # Uniform selection; outcomes do not determine membership or expertise.
            indices = np.sort(np.random.default_rng(seed).choice(indices, patients, replace=False))
        observations, actions, interventions, slots = [], [], [], []
        for index in indices:
            steps = np.flatnonzero(mask[index])
            if not len(steps) or not np.all(np.diff(steps) == 1):
                raise ValueError(f'Empty or discontinuous mask for source row {index}')
            observations.append(x[index][steps][:, feature_indices].astype(np.float64))
            actions.append(x[index, steps, action_index].astype(np.int64))
            interventions.append(x[index][steps][:, intervention_indices].astype(np.int64))
            slots.append(steps)
        if not all(np.isfinite(o).all() for o in observations):
            raise ValueError('Non-finite physiological observations')
        if not all(np.isin(a, [0, 1]).all() for a in actions):
            raise ValueError('Selected action is not binary')
        # Validate the original values before integer conversion as well.
        for index in indices:
            if not np.isin(x[index, mask[index], action_index], [0, 1]).all():
                raise ValueError('Selected action has fractional values')
        return dict(observations=observations, actions=actions, slots=slots,
                    interventions=interventions, vids=vids[indices], outcomes=y[indices],
                    source_indices=indices, total_patients=len(x), source_features=names,
                    person_ids=person_ids[indices] if person_ids is not None else None)


def prepare(data, seed, test_fraction=0.2):
    episodes = data['observations']
    if data.get('person_ids') is not None:
        train_ids, test_ids = next(GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
                                  .split(np.arange(len(episodes)), groups=data['person_ids']))
    else:
        train_ids, test_ids = train_test_split(np.arange(len(episodes)), test_size=test_fraction,
                                             random_state=seed, stratify=data['outcomes'] if data.get('outcome_available', True) else None)
    splits = np.full(len(episodes), 'train', dtype='<U5')
    splits[test_ids] = 'test'
    offsets = np.r_[0, np.cumsum([len(x) for x in episodes])]
    row_train = np.repeat(splits == 'train', np.diff(offsets))
    raw = np.concatenate(episodes)
    # X's mean features are already transformed in the supplied NPZ. We refit
    # only this additional scaler on training visits, without claiming recovery
    # of the upstream dataset's normalization or its original train/test split.
    mean = raw[row_train].mean(axis=0)
    scale = raw[row_train].std(axis=0)
    scale[scale < 1e-8] = 1
    physiology = np.clip((raw - mean) / scale, -5, 5)
    state_episodes = [physiology[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    policy_episodes, valid = [], []
    for x in episodes:
        # Current measurement plus trailing min/max over up to three rows.
        low = np.array([x[max(0, t - 2):t + 1].min(axis=0) for t in range(len(x))])
        high = np.array([x[max(0, t - 2):t + 1].max(axis=0) for t in range(len(x))])
        agg = np.concatenate([x, high, low], axis=1)
        # Conservative alignment: explain action t using measurements through
        # t-1. There is no prehistory for the first row, so exclude it from fit.
        policy_episodes.append(np.vstack([agg[:1], agg[:-1]]))
        valid.append(np.arange(len(x)) > 0)
    policy_raw, valid = np.concatenate(policy_episodes), np.concatenate(valid)
    fit_rows = row_train & valid
    if not fit_rows.any():
        raise ValueError('No training rows have scoreable prehistory')
    lower, upper = policy_raw[fit_rows].min(axis=0), policy_raw[fit_rows].max(axis=0)
    span = upper - lower
    span[span < 1e-8] = 1
    policy_states = np.clip(2 * (policy_raw - lower) / span - 1, -1, 1).astype(np.float32)
    return dict(offsets=offsets, splits=splits, row_train=row_train, raw=raw,
                state_episodes=state_episodes, policy_states=policy_states,
                valid=valid, mean=mean, scale=scale, lower=lower, span=span)


