"""Custom source-data loader for the complete THEMES paper/source port."""
import math
from .csv_loader import ThemesCsvLoader
from .themes_al.temporal import fit_temporal


class ThemesTimeCsvLoader(ThemesCsvLoader):
    def __init__(self,visit_column='VisitIdentifier',time_column='MinutesFromArrival',action_column='a_2',
                 feature_columns=None,person_column=None,test_fraction=0.2,window_minutes=60.,lag_rows=0,policy_scaling="train_minmax_zero_one",clip_policy_inputs=False):
        super().__init__(visit_column,time_column,action_column,feature_columns,person_column,test_fraction)
        if not math.isfinite(window_minutes) or window_minutes<=0 or type(lag_rows) is not int or lag_rows<0:
            raise ValueError('window_minutes must be positive and lag_rows a nonnegative integer')
        if policy_scaling not in {"train_minmax_zero_one","train_minmax_minus_one_one"}:
            raise ValueError("Unsupported policy_scaling")
        self.window_minutes,self.lag_rows=window_minutes,lag_rows
        if type(clip_policy_inputs) is not bool:raise ValueError("clip_policy_inputs must be boolean")
        self.policy_scaling=policy_scaling
        self.clip_policy_inputs=clip_policy_inputs

    def load(self,path,patients,seed):
        data=super().load(path,patients,seed)
        data['loader_id']='themes-dev/time_csv_trajectories'
        return data

    def prepare(self,data,seed):
        return fit_temporal(data,seed,test_fraction=self.test_fraction,
            window_minutes=self.window_minutes,lag_rows=self.lag_rows,policy_scaling=self.policy_scaling,clip_policy_inputs=self.clip_policy_inputs)
