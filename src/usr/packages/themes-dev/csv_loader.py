"""Selectable CSV trajectory loader. All dataset-specific work is owned here."""
from .themes_al.csv_data import load_csv, validate_csv_schema
from .themes_al.data import prepare, PHYSIOLOGY


class ThemesCsvLoader:
    def __init__(self, visit_column='VisitIdentifier', time_column='MinutesFromArrival',
                 action_column='a_2', feature_columns=None, person_column=None, test_fraction=0.2):
        if not 0 < float(test_fraction) < 1:
            raise ValueError('test_fraction must be between zero and one')
        self.schema = dict(visit_column=visit_column, time_column=time_column, action_column=action_column,
                           feature_columns=list(PHYSIOLOGY if feature_columns is None else feature_columns),
                           person_column=person_column)
        self.test_fraction = float(test_fraction)

    def validate_file(self, path):
        if path.suffix.lower() != '.csv':
            raise ValueError('themes-dev/csv_trajectories requires a CSV file')
        validate_csv_schema(path, **self.schema)

    def load(self, path, patients, seed):
        self.validate_file(path)
        data = load_csv(path, patients, seed, **self.schema)
        data['loader_id'] = 'themes-dev/csv_trajectories'
        data['test_fraction'] = self.test_fraction
        return data

    def prepare(self, data, seed):
        return prepare(data, seed, test_fraction=self.test_fraction)
