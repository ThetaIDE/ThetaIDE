"""A tiny deterministic tuner demonstrating the external component contract."""

from pathlib import Path
import json


class ExampleTuner:
    api_version = 1

    def settings(self):
        return [{"key": "count", "label": "Candidates", "type": "int", "default": 2, "min": 1, "max": 10}]

    def validate_config(self, cfg):
        count = cfg.tuning.get("options", {}).get("example", {}).get("count", 2)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10:
            raise ValueError("Candidate count must be between 1 and 10")

    def validate_dimension(self, target, dimension):
        if dimension.get("type") != "choice" or not dimension.get("values"):
            raise ValueError("Example tuner expects a nonempty choice dimension")

    def compile(self, overrides, **context):
        cfg, method_cfg = context["cfg"], context["method_cfg"]
        method = next(arg.split("=", 1)[1] for arg in overrides if arg.startswith("++agent.name="))
        options = cfg.tuning.get("options", {}).get("example", {})
        dimensions = method_cfg.get("tune") or method_cfg.get("search_space")
        payload = {"group": str(cfg.group), "experiment": str(cfg.experiment_id), "method": method,
                   "count": int(options.get("count", 2)), "dimensions": dimensions,
                   "metric": cfg.tuning.metric, "direction": cfg.tuning.direction,
                   "study": context["study_name"]}
        return [*overrides, "--example-tuner=" + json.dumps(payload, separators=(",", ":"))]

    def job_entrypoint(self, cfg, method_cfg, script, python):
        return str(Path(__file__).with_name("runner.py")), python

    def study_name(self, cfg, method):
        return f"example-{method}"

    def storage(self, cfg):
        return None

    def prepare(self, cfg, storage, study_name):
        pass

    def finalize(self, group, experiment, method, storage, study_name):
        root = Path("results/checkpoints") / group / experiment / method
        if not (root / "tuning_summary.json").is_file():
            raise RuntimeError("Example tuning job did not produce its summary")
