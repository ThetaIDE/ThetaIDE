import json

from src.app.pipeline.trial_artifacts import trial_log_version


def test_trial_log_versions_are_stable_and_study_specific():
    def version(study, number):
        return trial_log_version({"theta_tuning": {"study_name": study, "trial_number": number}})

    assert version("study", 0) == version("study", 0)
    assert len({version("study", 0), version("study", 1), version("other", 0)}) == 3
    assert trial_log_version({}) is None


def test_sweep_queue_groups_active_trials_and_tracks_resume(tmp_path):
    from frontend.job_manager import JobManager

    root = tmp_path / "results/checkpoints/g/e/m"
    root.mkdir(parents=True)
    summary = {"component": "optuna", "study": "s", "stop_reason": "running", "requested_trials": 4,
               "initial_trials": 1, "trials": [
                   {"id": 0, "state": "complete", "log_version": "version_old"},
                   {"id": 1, "state": "complete", "log_version": "version_one"},
                   {"id": 2, "state": "running", "log_version": "version_two"},
                   {"id": 3, "state": "running", "log_version": "version_three"}]}
    (root / "tuning_summary.json").write_text(json.dumps(summary))
    manager = JobManager(store_path=tmp_path / "jobs/jobs.db", project_root=tmp_path,
                         logs_dir=tmp_path / "results/logs")
    job = {"job_id": "j", "group": "g", "experiment_id": "e", "status": "running", "agents": ["m"]}
    items = manager._execution_queue_items(job)
    assert len(items) == 4
    assert items[0]["sweep_parent"] and "1/4" in items[0]["description"] and "1 pending" in items[0]["description"]
    assert items[1]['sweep_backlog'] and items[1]['status'] == 'pending'
    assert [item["agents"][0] for item in items if item.get('sweep_trial')] == ["m · s · trial 2", "m · s · trial 3"]
    sources = manager._run_sources(job)
    assert {source[3] for source in sources} == {"version_old", "version_one", "version_two", "version_three"}


def test_sweep_view_shows_trials_and_opens_the_selected_study_trial():
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from frontend.monitor.sweep_view import SweepView

    app = QApplication.instance() or QApplication([])
    view = SweepView()
    study = {"study": "s", "direction": "maximize", "objective_metric": "reward", "best_trial_id": 2,
             "trials": [{"id": 2, "state": "complete", "value": 3.0, "parameters": {"lr": 0.01}}]}
    view.set_studies([("m", study)])
    selected = []
    view.trial_selected.connect(selected.append)
    view._open_trial(0, 0)
    assert selected == ["m · s · trial 2"]
    assert view.table.rowCount() == 1
    view.table.selectRow(0)
    assert view.open_button.isEnabled()
    assert view.parameters.rowCount() == 1
    import copy
    updated = copy.deepcopy(study)
    updated["stop_reason"] = "trial_limit"
    view.set_studies([("m", updated)])
    assert view.table.currentItem().data(Qt.ItemDataRole.UserRole) == 2
    assert view.open_button.isEnabled()
    assert view.chart_type.findText("Parameter: lr") >= 0
    view.close()
    assert app is not None


def test_sweep_view_shows_future_pending_work_without_inventing_parameters():
    from PyQt6.QtWidgets import QApplication
    from frontend.monitor.sweep_view import SweepView
    app = QApplication.instance() or QApplication([])
    view = SweepView()
    study = {'study': 'rolling', 'direction': 'minimize', 'stop_reason': 'running',
             'requested_trials': 10, 'initial_trials': 0,
             'future_trials': 8, 'pending_trials': 9,
             'trials': [{'id': 0, 'state': 'running', 'parameters': {'lr': .01}},
                        {'id': 1, 'state': 'pending', 'parameters': {'lr': .02}}]}
    view.set_studies([('m', study)])
    assert '9 pending' in view.summary.text()
    assert view.table.rowCount() == 3
    row = next(row for row in range(3) if 'future trials' in view.table.item(row, 0).text())
    assert not view.table.isRowHidden(row)
    opened = []
    view.trial_selected.connect(opened.append)
    view.table.selectRow(row)
    assert not view.open_button.isEnabled()
    view._open_trial(row, 0)
    assert not opened
    view.filter.setText('pending')
    assert not view.table.isRowHidden(row)


def test_tuning_summary_counts_pending_sampled_and_future_trials(tmp_path):
    import optuna
    from component_fixtures import optional_module
    _optuna_utils = optional_module('optuna', 'utils.py')
    write_study_summary = _optuna_utils.write_study_summary
    study = optuna.create_study()
    study.set_user_attr('theta_requested_trials', 10)
    study.set_user_attr('theta_initial_trials', 0)
    study.set_user_attr('theta_last_stop_reason', 'running')
    trial = study.ask()
    trial.set_user_attr('theta_execution_state', 'pending')
    write_study_summary(study, tmp_path)
    summary = json.loads((tmp_path / 'tuning_summary.json').read_text())
    assert summary['pending_trials'] == 10 and summary['future_trials'] == 9
    assert summary['trials'][0]['state'] == 'pending'
    trial.set_user_attr('theta_execution_state', 'running')
    write_study_summary(study, tmp_path)
    summary = json.loads((tmp_path / 'tuning_summary.json').read_text())
    assert summary['pending_trials'] == 9 and summary['trials'][0]['state'] == 'running'


def test_queue_keeps_pending_backlog_card_separate_from_study_card():
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel
    app = QApplication.instance() or QApplication([])
    panel = QueuePanel(lambda *args: None)
    base = {'job_id': 'j', 'experiment_id': 'e', 'sweep_key': 'm:s', 'agents': []}
    panel.set_data({'active': [dict(base, sweep_parent=True, status='running'),
                               dict(base, sweep_backlog=True, status='pending', description='8 future trials pending')]})
    assert len(panel.cards) == 2
    backlog = next(card for key, card in panel.cards.items() if key.endswith(':pending'))
    assert 'Pending trials' in backlog.experiment_id.text()
    assert backlog.cancel_button.isHidden()
    assert backlog.elapsed.text() == 'Waiting'


def test_trial_feeds_keep_parallel_versions_separate(tmp_path):
    from frontend.live_run import LiveRunCache

    agent = tmp_path / "m"
    for version, value in [("version_sweep_a_trial_0", 1), ("version_sweep_a_trial_1", 2)]:
        directory = agent / version
        directory.mkdir(parents=True)
        (directory / "metrics.csv").write_text(f"step,eval/reward\n1,{value}\n")
    cache = LiveRunCache()
    first = cache.load_version(agent, "version_sweep_a_trial_0")
    second = cache.load_version(agent, "version_sweep_a_trial_1")
    assert first.latest("eval/reward") == 1
    assert second.latest("eval/reward") == 2


def test_summary_preserves_recorded_log_version_without_guessing_for_old_trials(tmp_path):
    import optuna
    from component_fixtures import optional_module
    _optuna_utils = optional_module('optuna', 'utils.py')
    write_study_summary = _optuna_utils.write_study_summary

    study = optuna.create_study()
    old = study.ask()
    study.tell(old, 1.0)
    current = study.ask()
    current.set_user_attr("theta_log_version", "version_sweep_current")
    write_study_summary(study, tmp_path)
    trials = json.loads((tmp_path / "tuning_summary.json").read_text())["trials"]
    assert trials[0]["log_version"] is None
    assert trials[1]["log_version"] == "version_sweep_current"
