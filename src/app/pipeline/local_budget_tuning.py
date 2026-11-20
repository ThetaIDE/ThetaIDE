"""Shared local budget execution and study records for native tuner packages."""
import json
import uuid

from hydra.core.plugins import Plugins
from omegaconf import OmegaConf

from src.app.core.execution import runner_tuning_profile
from src.app.pipeline.execution_data import trial_worker_results
from src.app.pipeline.trial_artifacts import trial_checkpoint_dir
from src.app.workspace import WorkspacePaths


class LocalBudgetTuner:
    api_version = 1
    requires_repeated_objective = True

    def options(self, cfg):
        values = ((cfg.get("tuning") or {}).get("options") or {}).get(self.component) or {}
        return {field["key"]: values.get(field["key"], field["default"]) for field in self.settings()}

    def validate_config(self, cfg):
        options = self.options(cfg)
        for field in self.settings():
            value = options[field["key"]]
            if type(value) is not int or value < field["min"]:
                raise ValueError(f"Invalid {self.component} setting {field['key']}")

    def validate_runner(self, cfg, *, paths):
        self.execution_profile(cfg, dimensions=cfg.theta_hpo_space, paths=paths)

    def execution_profile(self, cfg, *, dimensions, paths=None):
        profile = runner_tuning_profile(cfg, paths=paths)
        protected = set(dimensions).intersection(profile.get("protected_paths", ()))
        if protected:
            raise ValueError(f"Search dimensions change the runner's budget or execution mode: {sorted(protected)}")
        if profile["unit"] != "transitions" or cfg["tuning"]["metric"] not in profile["metrics"]:
            raise ValueError(f"{self.component} requires transition budgets and a supported objective metric")
        options = self.options(cfg)
        budget = options[self.budget_option]
        if budget % profile["quantum"]:
            raise ValueError(f"Budget {budget} must be a multiple of the runner's quantum {profile['quantum']}")
        return profile

    def compile(self, overrides, **context):
        return [
            "--multirun", *overrides, f"hydra/sweeper={self.component}",
            f"hydra.sweeper.study_name={json.dumps(context['study_name'])}",
            "++theta_hpo_space=" + hydra_value(OmegaConf.to_container(
                OmegaConf.create(context["method_cfg"]["tune"]), resolve=True)),
        ]

    def job_entrypoint(self, cfg, method_cfg, script, python):
        return script, python

    def study_name(self, cfg, method_name):
        return f"{cfg.get('experiment_id')}_{method_name}_{uuid.uuid4().hex[:12]}"

    def storage(self, cfg):
        return None

    def prepare(self, cfg, storage, study_name):
        pass

    def finalize(self, group, experiment, method, storage, study_name):
        pass


def hydra_value(value):
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{hydra_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(map(hydra_value, value)) + "]"
    return json.dumps(value, allow_nan=False)


def progress(record, unit):
    measured = (record.get("result") or {}).get("progress") or (record.get("metadata") or {}).get("progress") or {}
    count = measured.get("completed")
    if measured.get("unit") == unit and type(count) is int and count >= 0:
        return count
    return None


class LocalBudgetSweeper:
    def __init__(self, study_name):
        self.study_name = study_name

    def setup(self, *, hydra_context, task_function, config):
        self.config = config
        self.loader = hydra_context.config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            hydra_context=hydra_context, task_function=task_function, config=config)

    @staticmethod
    def worker_results(cfg, trial):
        return trial_worker_results(WorkspacePaths.for_process(), cfg.group, cfg.experiment_id,
                                    str(cfg.agent.name).replace("/", "_"), trial["log_version"])

    def study_archive(self, cfg):
        root = WorkspacePaths.for_process().results / "checkpoints" / str(cfg.group) / str(cfg.experiment_id)
        method = str(cfg.agent.name).replace("/", "_")
        archive = trial_checkpoint_dir(root / method, self.study_name, 0).parent
        archive.mkdir(parents=True, exist_ok=True)
        return archive

    @staticmethod
    def publish(archive, summary):
        temporary = archive / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2, allow_nan=False))
        temporary.replace(archive / "tuning_summary.json")
