"""Discussion storage: the experiment tree, figure files and discussion text (no Qt)."""

import pytest
import yaml

from discussion_fixtures import build_discussion_workspace
from frontend.discussion import model
from frontend.discussion.model import FigureDoc, FigureError
from src.app.workspace import RESERVED_RESEARCH_DIRECTORIES, WorkspacePaths


@pytest.fixture
def paths(tmp_path, monkeypatch):
    root = build_discussion_workspace(tmp_path / "workspace")
    monkeypatch.setenv("THETA_WORKSPACE", str(root))
    return WorkspacePaths(root)


def flatten(nodes):
    return [(node.kind, node.rel) for node in nodes] + [
        item for node in nodes for item in flatten(node.children)]


def test_discussion_is_a_reserved_workspace_directory(paths):
    assert paths.discussion == paths.root / "src/usr/discussion"
    assert "discussion" in RESERVED_RESEARCH_DIRECTORIES
    with pytest.raises(ValueError):
        paths.resource("discussion")


def test_tree_mirrors_experiment_groups_and_skips_group_defaults(paths):
    tree = model.experiment_tree(paths)
    assert [(n.kind, n.name) for n in tree] == [("group", "figs"), ("group", "other")]
    figs = tree[0]
    assert [(n.kind, n.rel) for n in figs.children] == [("folder", "figs/sweeps"), ("experiment", "figs/online")]
    assert ("experiment", "figs/sweeps/offline") in flatten(tree)
    assert not any(rel.endswith("_base") for _, rel in flatten(tree))
    assert model.experiment_ids(tree) == ["figs/sweeps/offline", "figs/online", "other/solo"]


def test_figures_lists_saved_figure_files(paths):
    assert [p.name for p in model.figures(paths, "figs/online")] == ["fixture_curve.yaml"]
    assert model.figures(paths, "other/solo") == []  # no folder until the first figure
    with pytest.raises(FigureError):
        model.figures(paths, "../escape")


def test_load_reads_plotter_params_and_schema(paths):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    assert doc.plotter == "fixture_curve"
    assert doc.params == {"color": "#d62728"}
    assert doc.experiment == "figs/online"
    assert doc.defaults() == {"color": "#1f77b4", "metric": "train/reward"}
    assert doc.values() == {"color": "#d62728", "metric": "train/reward"}
    assert doc.live_params() == ["color"]


@pytest.mark.parametrize("text, message", [
    ("plotter: fixture_curve\nparams: {}\nspec: {}\n", "unknown key spec"),
    ("plotter: fixture_curve\n", "missing params"),
    ("params: {}\n", "missing plotter"),
    ("plotter: fixture_curve\nparams: {width: 3}\n", "Unknown parameter: width"),
    ("plotter: fixture_curve\nparams: {color: 3}\n", "color"),
    ("plotter: nowhere\nparams: {}\n", "No installed plot component"),
    ("plotter: fixture_table\nparams: {}\n", "declares no figure parameter_schema"),
    ("- a list\n", "must be a mapping"),
])
def test_load_rejects_invalid_figure_files(paths, text, message):
    path = paths.discussion / "figs/online/broken.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(FigureError, match=message):
        FigureDoc.load(path, paths=paths)


def test_set_and_unset_write_explicit_values_only(paths):
    path = model.create_figure(paths, "other/solo", "fixture_curve")
    doc = FigureDoc.load(path, paths=paths)
    assert yaml.safe_load(path.read_text()) == {"plotter": "fixture_curve", "params": {}}

    assert doc.set_param("metric", "eval/reward")
    doc.save()
    assert yaml.safe_load(path.read_text()) == {"plotter": "fixture_curve", "params": {"metric": "eval/reward"}}
    assert not doc.set_param("metric", "eval/reward")  # unchanged

    assert doc.unset_param("metric")
    doc.save()
    assert yaml.safe_load(path.read_text())["params"] == {}
    assert doc.values()["metric"] == "train/reward"
    assert not doc.unset_param("metric")


def test_invalid_edits_leave_the_document_unchanged(paths):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    with pytest.raises(FigureError, match="Unknown parameter"):
        doc.set_param("width", 2)
    with pytest.raises(FigureError):
        doc.set_param("color", 5)
    assert doc.params == {"color": "#d62728"}


def test_create_figure_makes_unique_names(paths):
    first = model.create_figure(paths, "figs/online", "fixture_curve")
    second = model.create_figure(paths, "figs/online", "fixture_curve")
    assert [first.name, second.name] == ["fixture_curve_2.yaml", "fixture_curve_3.yaml"]
    assert [p.stem for p in model.figures(paths, "figs/online")] == [
        "fixture_curve", "fixture_curve_2", "fixture_curve_3"]
    with pytest.raises(FigureError):
        model.create_figure(paths, "figs/online", "nowhere")


def test_discussion_text_reads_and_writes(paths):
    saved = paths.discussion / "figs/online/fixture_curve.yaml"
    assert model.read_discussion(saved) == "PPO overtakes A2C after step 3.\n"

    path = model.create_figure(paths, "other/solo", "fixture_curve")
    assert model.read_discussion(path) == ""
    assert not model.write_discussion(path, "")  # no file until the first text
    assert not path.with_suffix(".md").exists()
    assert model.write_discussion(path, "First note")
    assert path.with_suffix(".md").read_text() == "First note"
    assert not model.write_discussion(path, "First note")
    assert model.write_discussion(path, "")
    assert model.read_discussion(path) == ""
