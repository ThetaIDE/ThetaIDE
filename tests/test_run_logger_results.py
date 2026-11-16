"""Run loggers declare the results category and files they write; the catalog names no logger."""
import json

from src.app.services import catalog
from src.app.workspace import WorkspacePaths


def tracer(paths, declared=True):
    root = paths.user("package") / "tracer"
    root.mkdir(parents=True)
    (root / "provider.py").write_text("def create_logger(context):\n    return None\n")
    capability = {"id": "tracer", "kind": "run_logger", "entry_point": "provider.py:create_logger", "label": "Trace"}
    if declared:
        capability["results"] = {"category": "traces", "patterns": ["*.trace"]}
    (root / ".theta_component.json").write_text(json.dumps({
        "id": "tracer", "name": "Tracer", "kind": "package", "version": "1.0.0", "storage_version": 3,
        "api_version": 1, "capabilities": [capability],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {}}}))


def run(paths):
    results = paths.results
    (results / "logs/g/e/agent/version_0").mkdir(parents=True)
    (results / "logs/g/e/agent/version_0/metrics.csv").write_text("step,loss\n0,1\n")
    (results / "traces/g/e/agent/version_0").mkdir(parents=True)
    (results / "traces/g/e/agent/version_0/run.trace").write_text("t")
    (results / "traces/g/e/agent/version_0/notes.txt").write_text("n")


def test_declared_category_joins_the_run_and_names_its_files(tmp_path):
    paths = WorkspacePaths(tmp_path)
    tracer(paths)
    run(paths)
    assert catalog.result_categories(workspace=paths)["traces"] == ("*.trace",)
    assert "traces" in catalog.result_roots(workspace=paths)
    kinds = {a["path"]: a["kind"] for a in catalog.run_manifest("g", "e", workspace=paths)["artifacts"]}
    assert kinds["traces/g/e/agent/version_0/run.trace"] == "traces"
    assert kinds["traces/g/e/agent/version_0/notes.txt"] == "report"
    catalog.delete_run("g", "e", workspace=paths)
    assert not (paths.results / "traces/g/e").exists()


def test_an_undeclared_folder_is_not_part_of_a_run(tmp_path):
    paths = WorkspacePaths(tmp_path)
    tracer(paths, declared=False)
    run(paths)
    assert "traces" not in catalog.result_categories(workspace=paths)
    artifacts = catalog.run_manifest("g", "e", workspace=paths)["artifacts"]
    assert all(a["root"] != "traces" for a in artifacts)
