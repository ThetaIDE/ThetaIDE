"""Standalone CSV loader used by the selectable Theta data provider."""
from pathlib import Path
import numpy as np
from .data import PHYSIOLOGY


def validate_csv_schema(path, *, visit_column='VisitIdentifier', time_column='MinutesFromArrival',
                        action_column='a_2', feature_columns=None, person_column=None):
    import pandas as pd
    features = list(PHYSIOLOGY if feature_columns is None else feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError('feature_columns must be nonempty and unique')
    forbidden = {'Action', 'Shock', 'a_0', 'a_1', 'a_2', 'a_3', visit_column, time_column, action_column, person_column}
    if set(features) & forbidden:
        raise ValueError('Policy features cannot include actions, outcomes, identifiers or timestamps')
    names = list(pd.read_csv(path, nrows=0).columns)
    required = [visit_column, time_column, action_column, *features] + ([person_column] if person_column else [])
    missing = set(required) - set(names)
    if missing:
        raise ValueError(f'THEMES CSV is missing columns: {sorted(missing)}')


def load_csv(path: Path, patients: int, seed: int, *, visit_column='VisitIdentifier',
             time_column='MinutesFromArrival', action_column='a_2', feature_columns=None, person_column=None):
    """Load downloaded THEMES CSVs without converting or modifying the source.

    Source rows are zero-based CSV data rows (the header is excluded). Visits
    are ordered by first appearance; rows within visits are stably time-sorted.
    The supplied binary a_2 is the original THEMES script's policy target.
    Shock, the multi-class Action, and other intervention flags are not inputs.
    """
    import pandas as pd
    feature_columns = list(PHYSIOLOGY if feature_columns is None else feature_columns)
    schema = dict(visit_column=visit_column, time_column=time_column, action_column=action_column,
                  feature_columns=feature_columns, person_column=person_column)
    validate_csv_schema(path, **schema)
    frame = pd.read_csv(path, dtype={visit_column: str, **({person_column: str} if person_column else {})})
    if frame.empty or frame[visit_column].isna().any() or (frame[visit_column].str.strip() == '').any():
        raise ValueError('THEMES CSV requires nonempty visit identifiers')
    numeric = frame[[time_column, *feature_columns, action_column]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('Non-finite physiology, timestamp or a_2 in THEMES CSV')
    if not np.isin(frame[action_column], [0, 1]).all():
        raise ValueError('THEMES CSV a_2 must be binary, without fractional values')
    if frame.duplicated([visit_column, time_column]).any():
        raise ValueError('Duplicate timestamps within a visit require an explicit aggregation rule')
    if person_column and (frame[person_column].isna().any() or (frame[person_column].str.strip() == '').any()):
        raise ValueError('Missing actual person identifier')
    if person_column and (frame.groupby(visit_column)[person_column].nunique() != 1).any():
        raise ValueError('One visit maps to multiple person identifiers')
    ids = frame[visit_column].drop_duplicates().to_numpy(dtype=str)
    indices = np.arange(len(ids))
    if 0 < patients < len(ids):
        indices = np.sort(np.random.default_rng(seed).choice(indices, patients, replace=False))
    flags = [name for name in ('a_0', 'a_1', 'a_2', 'a_3') if name in frame]
    if not np.isin(frame[flags].to_numpy(), [0, 1]).all():
        raise ValueError('THEMES CSV intervention flags must be binary')
    groups = frame.groupby(visit_column, sort=False)
    observations, actions, interventions, slots, timestamps = [], [], [], [], []
    for index in indices:
        visit = groups.get_group(ids[index]).sort_values(time_column, kind='stable')
        observations.append(visit[feature_columns].to_numpy(dtype=np.float64))
        actions.append(visit[action_column].to_numpy(dtype=np.int64))
        interventions.append(visit[flags].to_numpy(dtype=np.int64))
        slots.append(visit.index.to_numpy(dtype=np.int64))
        timestamps.append(visit[time_column].to_numpy(dtype=np.float64))
    return dict(observations=observations, actions=actions, interventions=interventions,
                slots=slots, timestamps=timestamps, vids=ids[indices], source_indices=indices,
                # Row-level Shock is not a visit outcome and can be missing.
                outcomes=np.full(len(indices), -1, dtype=np.int64),
                person_ids=np.array([groups.get_group(ids[i])[person_column].iloc[0] for i in indices]) if person_column else None,
                total_patients=len(ids), source_features=list(frame.columns),
                source_kind='local_themes_csv', source_format='csv', action_feature=action_column, source_schema=schema,
                intervention_feature_names=flags, observation_feature_names=feature_columns,
                policy_base_feature_names=feature_columns, outcome_available=False)
