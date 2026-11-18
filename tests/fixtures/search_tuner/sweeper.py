"""Grid sweeper: Hydra's basic sweeper, plus the trial identity the host expects of a tuner."""
from hydra._internal.core_plugins.basic_sweeper import BasicSweeper


class SearchSweeper(BasicSweeper):
    def __init__(self, study_name, n_trials):
        super().__init__(max_batch_size=None)
        self.study_name, self.n_trials, self.trial = study_name, n_trials, 0

    def get_job_batch(self):
        batch = []
        for overrides in super().get_job_batch()[:self.n_trials - self.trial]:
            batch.append([*overrides, '++theta_tuning.component=search',
                          f'++theta_tuning.study_name={self.study_name}', f'++theta_tuning.trial_number={self.trial}'])
            self.trial += 1
        return batch
