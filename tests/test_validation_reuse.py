"""A validation does its discovery once, and the validation process reloads providers only when they change."""
import io
import json
from pathlib import Path

from frontend.model import BASE_EXPERIMENT
from src.app.workspace import WorkspacePaths


def test_provider_fingerprint_follows_provider_sources_not_config_yaml(tmp_path):
    from src.app.services.config_service import _inputs_fingerprint, provider_fingerprint

    paths = WorkspacePaths(tmp_path, application=tmp_path / "application")
    source = paths.research / "models/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    profile = paths.config / "model/example.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("hidden: 8\n")
    providers, inputs = provider_fingerprint(paths), _inputs_fingerprint(paths)

    profile.write_text("hidden: 16\n")
    assert provider_fingerprint(paths) == providers  # a config edit never re-imports providers
    assert _inputs_fingerprint(paths) != inputs  # but a cached check of it is stale

    source.write_text("value = 2\n")
    assert provider_fingerprint(paths) != providers


def test_one_check_scans_the_workspace_for_components_once(fixture_workspace, monkeypatch):
    from src.app.services import config_service

    paths = WorkspacePaths(fixture_workspace)
    config_service.compose_report.uncached(BASE_EXPERIMENT, [], None, workspace=paths)  # import providers first
    scans = []
    rglob = Path.rglob

    def counting(self, pattern, *args, **kwargs):
        if pattern == ".theta_component.json":
            scans.append(self)
        return rglob(self, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", counting)
    report = config_service.compose_report.uncached(BASE_EXPERIMENT, [], None, workspace=paths)
    assert report["valid"], report["errors"]
    roots = [root for root in paths.component_roots if root.is_dir()]
    assert len(scans) <= len(roots), f"{len(scans)} manifest scans for {len(roots)} component roots"

    scans.clear()
    config_service.prepare_launch(BASE_EXPERIMENT, ["++experiment_id='reuse_probe'"], workspace=paths)
    assert len(scans) <= len(roots)


def test_worker_reloads_providers_only_when_their_inputs_change(tmp_path, monkeypatch):
    import src.app.core.provider_loader as provider_loader
    import src.app.services as services
    from frontend import validation_worker
    from src.app.services import config_service

    fingerprints = iter([1, 1, 1, 2, 2])
    events = []
    monkeypatch.setattr(services, "configure_workspace", lambda paths: None)
    monkeypatch.setattr(WorkspacePaths, "for_process", classmethod(lambda cls: WorkspacePaths(tmp_path)))
    monkeypatch.setattr(config_service, "provider_fingerprint", lambda paths: next(fingerprints))
    monkeypatch.setattr(provider_loader, "invalidate", lambda paths: events.append("reload"))
    monkeypatch.setattr(config_service, "clear_compose_cache", lambda: events.append("clear"))
    monkeypatch.setattr(config_service, "compose_report", lambda *args, workspace: events.append("compose") or {"valid": True})

    def launch(experiment, overrides, *, overwrite, active_jobs, workspace):
        events.append("launch")
        raise config_service.LaunchError("taken", conflict=True)

    monkeypatch.setattr(config_service, "prepare_launch", launch)
    requests = [("compose", ["g/e", [], None]), ("compose", ["g/e", [], "draft"]), ("launch", ["g/e", [], False, []]),
                ("compose", ["g/e", [], None]), ("compose", ["g/e", [], None])]
    replies = [tmp_path / f"{index}.json" for index in range(len(requests))]
    lines = "".join(json.dumps({"operation": op, "args": args, "reply": str(reply)}) + "\n"
                    for (op, args), reply in zip(requests, replies))
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))

    validation_worker.main()

    assert events == ["reload", "clear", "compose", "compose", "launch", "reload", "clear", "compose", "compose"]
    assert json.loads(replies[2].read_text()) == {"data": None, "error": "taken", "launch_conflict": True}
