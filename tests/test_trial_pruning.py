"""Multi-seed and objective-component trials report partial objective scores and stop when pruned."""
import json
import shutil
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.app.objectives.api import (
    ObjectiveProgress, RunResult, TrialResults, load_objective, validate_objective,
)
from src.app.services.cancellation import TrainingCancelled

PROGRESS_OBJECTIVE = '''
from src.app.api.tuning import ObjectiveProgress


def evaluate(results, params):
    return sum(run.metrics.get("score", 0.) for run in results.runs) / len(results.runs)


def intermediate(results, params):
    {body}
'''

PROGRESS_BODY = '''progress = results.common_progress("eval/reward")
    if progress is None:
        return None
    step, _ = progress
    values = [run.curve("eval/reward")[step][1] for run in results.runs]
    return ObjectiveProgress(step, sum(values) / len(values))'''

MONITOR = '''
    def pruning_enabled(self, cfg):
        return True

    def trial_monitor(self, cfg):
        runtime = cfg.get("theta_tuning")
        if not runtime or runtime.get("report_path") is None:
            return None
        return ReportingMonitor(runtime["report_path"], int(runtime["prune_at"]))


class Pruned(Exception):
    theta_trial_status = "pruned"


class ReportingMonitor:
    def __init__(self, path, prune_at):
        self.path, self.prune_at = path, prune_at

    def report(self, value, step):
        with open(self.path, "a") as stream:
            stream.write(json.dumps([step, value]) + "\\n")
        if step >= self.prune_at:
            raise Pruned(f"pruned at {step}")
'''


def run(seed, points, status="completed"):
    return RunResult(seed, status, events=[{"training_step": x, "metric": "eval/reward", "episode_outcomes": [y]}
                                           for x, y in points])


def package(root, package_id, kind, files, capability):
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / name).write_text(text)
    (root / ".theta_component.json").write_text(json.dumps({
        "id": package_id, "name": package_id, "kind": kind, "version": "1.0.0", "storage_version": 3,
        "api_version": 1, "capabilities": [capability],
        "payload": {"implementations": [n for n in files if n.endswith(".py")],
                    "resources": [n for n in files if not n.endswith(".py")], "configs": {}}}))


def workspace(tmp_path, *, body=PROGRESS_BODY, intermediate=True):
    """A pruning tuner with objective and repeated-run support, and a progress-scoring objective."""
    tuner = tmp_path / "src/usr/packages/grid"
    shutil.copytree(Path(__file__).parent / "fixtures/portable_tuner", tuner)
    (tuner / "component.py").write_text((tuner / "component.py").read_text() + MONITOR)
    manifest = json.loads((tuner / ".theta_component.json").read_text())
    manifest["capabilities"][0]["tuning_features"] = {
        "objective_modes": ["metric", "component"], "pruning": True,
        "pruning_objectives": True, "pruning_repeated_runs": True}
    (tuner / ".theta_component.json").write_text(json.dumps(manifest))
    descriptor = {"api_version": 1, "source": "progress.py"}
    if intermediate:
        descriptor["intermediate_function"] = "intermediate"
    package(tmp_path / "src/usr/packages/progress", "progress", "objective",
            {"progress.py": PROGRESS_OBJECTIVE.format(body=body), "progress.json": json.dumps(descriptor)},
            {"id": "progress", "kind": "objective", "entry_point": "progress.json"})


def trial_config(tmp_path, *, prune_at=2):
    return OmegaConf.create({
        "group": "g", "experiment_id": "e", "seed": 1, "model": {"name": "m"},
        "objective": {"component": "progress"},
        "variability": {"training_seeds": [1, 2], "parallel_runs": 2},
        "tuning": {"component": "fixture-grid", "direction": "maximize"},
        "theta_tuning": {"component": "fixture-grid", "study_name": "s", "trial_number": 0,
                         "log_version": "version_sweep_test_trial_0",
                         "report_path": str(tmp_path / "reports.jsonl"), "prune_at": prune_at}})


def checkpoint_writer(cancel_dir, checkpoints=6):
    """A seed that logs one evaluation per interval and stops when its trial is cancelled."""
    from src.app.services.cancellation import requested

    def execute(child, directory):
        events = Path(directory) / "evaluation_events.jsonl"
        target = "trial:version_sweep_test_trial_0"
        for index in range(checkpoints):
            if requested([target], str(cancel_dir)):
                return
            with events.open("a") as stream:
                stream.write(json.dumps({"training_step": index * 10, "metric": "eval/reward",
                                         "episode_outcomes": [float(index + child.seed)]}) + "\n")
            time.sleep(0.15)
    return execute


@pytest.fixture
def trial_workspace(tmp_path, monkeypatch):
    workspace(tmp_path)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("THETA_CANCEL_DIR", str(tmp_path / "cancel"))
    monkeypatch.setattr("src.app.pipeline.trial_pruning.POLL_SECONDS", 0.02)
    return tmp_path


def manifest_statuses(root):
    [path] = (root / "src/usr/results/executions/g/e").glob("*/manifest.json")
    return [run["status"] for run in json.loads(path.read_text())["runs"]]


def test_common_progress_waits_for_every_unfailed_seed():
    slow, fast = run(1, [(0, 1), (10, 2)], "running"), run(2, [(0, 3), (10, 4), (20, 5)], "running")
    assert TrialResults([slow, fast], {}).common_progress("eval/reward") == (1, 10)
    assert TrialResults([slow, run(3, [], "queued")], {}).common_progress("eval/reward") is None
    assert TrialResults([slow, run(3, [], "failed")], {}).common_progress("eval/reward") == (1, 10)
    snapshot = TrialResults([slow, run(3, [], "failed")], {}).snapshot()
    assert [r.status for r in snapshot.runs] == ["completed", "failed"]
    assert slow.status == "running"


def test_bundled_objectives_score_partial_panels():
    curve = load_objective("learning_curve")
    params = curve.parameters({"curve": {"budget": 100}, "aggregation": {"method": "mean"}})
    results = TrialResults([run(1, [(0, 0), (10, 500), (20, 500)], "running"),
                            run(2, [(0, 0), (10, 0)], "running")], {})
    progress = curve.intermediate(results, params)
    assert isinstance(progress, ObjectiveProgress) and progress.step == 1
    assert progress.value == pytest.approx((0.5 + 0.0) / 2)
    assert progress.diagnostics["horizon"] == 10
    final = load_objective("final_metric")
    panel = TrialResults([RunResult(1, "running", events=[{"step": 0, "data": {"loss": 4.}}, {"step": 5, "data": {"loss": 2.}}]),
                          RunResult(2, "running", events=[{"step": 0, "data": {"loss": 6.}}])], {})
    assert final.intermediate(panel, {"metric": "loss"}) == ObjectiveProgress(0, 5.0, {"per_run": [4., 6.]})


def test_progress_scorer_ignores_an_incomplete_trailing_line(tmp_path):
    from src.app.objectives.evaluation import ProgressScorer
    workspace(tmp_path)
    component = load_objective("progress", tmp_path)
    logs = tmp_path / "logs/m"
    for version, values in (("a", [1, 3]), ("b", [2, 4])):
        (logs / version).mkdir(parents=True)
        lines = [json.dumps({"training_step": i * 10, "metric": "eval/reward", "episode_outcomes": [v]})
                 for i, v in enumerate(values)]
        (logs / version / "evaluation_events.jsonl").write_text("\n".join(lines) + '\n{"training_step": 2')
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runs": [
        {"run_id": v, "training_seed": i, "status": "running", "logs": "logs", "agent": "m", "version": v,
         "checkpoints": "c"} for i, v in enumerate(("a", "b"))]}))
    scorer = ProgressScorer(component, {}, {}, tmp_path)
    assert scorer.score(manifest) == ObjectiveProgress(1, 3.5)
    assert scorer.score(manifest) is None  # unchanged files are not rescored


def test_pruning_requires_an_objective_that_scores_progress(tmp_path):
    workspace(tmp_path, intermediate=False)
    from src.app.workspace import WorkspacePaths
    with pytest.raises(ValueError, match="does not score intermediate progress"):
        validate_objective(trial_config(tmp_path), paths=WorkspacePaths(tmp_path))


def test_pruned_trial_stops_every_seed(trial_workspace):
    from src.app.pipeline.repeated_task import run_objective
    with pytest.raises(TrainingCancelled):
        run_objective(trial_config(trial_workspace), execute=checkpoint_writer(trial_workspace / "cancel"))
    reports = [json.loads(line) for line in (trial_workspace / "reports.jsonl").read_text().splitlines()]
    steps = [step for step, _ in reports]
    assert steps == sorted(set(steps)) and steps[-1] == 2
    # Seeds 1 and 2 log index + seed; the mean at checkpoint 2 is 3.5.
    assert reports[-1][1] == pytest.approx(3.5)
    assert manifest_statuses(trial_workspace) == ["cancelled", "cancelled"]


def test_unpruned_trial_is_scored_normally(trial_workspace):
    from src.app.pipeline.repeated_task import run_objective
    value = run_objective(trial_config(trial_workspace, prune_at=99), execute=checkpoint_writer(trial_workspace / "cancel", 3))
    assert value == 0.0
    assert manifest_statuses(trial_workspace) == ["completed", "completed"]
    assert [json.loads(line)[0] for line in (trial_workspace / "reports.jsonl").read_text().splitlines()][-1] <= 2


def test_broken_intermediate_objective_fails_the_trial(tmp_path, monkeypatch):
    workspace(tmp_path, body='raise RuntimeError("bad progress")')
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("THETA_CANCEL_DIR", str(tmp_path / "cancel"))
    monkeypatch.setattr("src.app.pipeline.trial_pruning.POLL_SECONDS", 0.02)
    from src.app.pipeline.repeated_task import run_objective
    with pytest.raises(RuntimeError, match="bad progress"):
        run_objective(trial_config(tmp_path), execute=checkpoint_writer(tmp_path / "cancel"))
