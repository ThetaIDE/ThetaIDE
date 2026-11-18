"""Discussion storage: the experiment tree, figure files and discussion text (no Qt)."""

import pytest
import yaml

from discussion_fixtures import build_discussion_workspace, write_seeded_runs
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


def test_tree_includes_catalog_runs_and_experiments_without_current_recipes(paths):
    from src.app.services.catalog import list_runs
    entries = list_runs(workspace=paths)
    tree = model.experiment_tree(paths, entries)
    assert ("run", "figs/online/20261116-120000") in flatten(tree)
    assert ("run", "figs/online/20261116-130000") in flatten(tree)
    (paths.config / "experiment/figs/online.yaml").unlink()
    assert ("experiment", "figs/online") in flatten(model.experiment_tree(paths, entries))


def test_run_candidates_use_saved_configuration_and_need_no_current_recipe(paths):
    target = "figs/online/20261116-120000"
    (paths.config / "experiment/figs/online.yaml").write_text("paradigm: offline_rl\n")
    candidates = {c.selector: c for c in model.figure_candidates(paths, target)}
    assert not candidates["fixture_curve"].blocked
    (paths.results / "logs" / target / "config.yaml").unlink()
    with pytest.raises(FigureError, match="has no config.yaml"):
        model.figure_candidates(paths, target)


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
    ("plotter: fixture_table\nparams: {}\n", "does not provide figure/vega-lite-v1"),
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


# ── Renaming, duplicating and deleting ───────────────────────────────────────


def test_rename_moves_the_figure_and_its_text_and_rejects_taken_or_invalid_names(paths):
    figure = paths.discussion / "figs/online/fixture_curve.yaml"
    model.write_discussion(figure, "Notes")
    renamed = model.rename_figure(figure, "  Reward curve ")
    assert renamed == figure.with_name("Reward curve.yaml")
    assert not figure.exists() and not model.discussion_path(figure).exists()
    assert model.read_discussion(renamed) == "Notes"
    assert model.rename_figure(renamed, "Reward curve") == renamed  # unchanged

    other = model.create_figure(paths, "figs/online", "fixture_curve")
    with pytest.raises(FigureError, match="already exists"):
        model.rename_figure(other, "Reward curve")
    for bad in ("", " ", "../escape", "_hidden", "a/b", "dots.yaml"):
        with pytest.raises(FigureError, match="Use letters"):
            model.rename_figure(other, bad)
    assert other.exists()


def test_duplicate_copies_figure_and_text_to_a_free_name_and_delete_removes_both(paths):
    figure = paths.discussion / "figs/online/fixture_curve.yaml"
    model.write_discussion(figure, "Original notes")
    first = model.duplicate_figure(figure)
    second = model.duplicate_figure(figure)
    assert (first.name, second.name) == ("fixture_curve_copy.yaml", "fixture_curve_copy_2.yaml")
    assert first.read_text() == figure.read_text() and model.read_discussion(first) == "Original notes"

    model.delete_figure(first)
    assert not first.exists() and not model.discussion_path(first).exists()
    model.delete_figure(second)
    assert model.figures(paths, "figs/online") == [figure]


def test_create_figure_skips_names_taken_by_a_figure_or_an_orphaned_text(paths):
    folder = paths.discussion / "figs/online"
    (folder / "convergence.md").write_text("Orphaned text")
    assert model.create_figure(paths, "figs/online", "convergence").name == "convergence_2.yaml"


# ── What a target logged ─────────────────────────────────────────────────────


def test_target_runs_lists_logged_methods_and_metrics_without_counters(paths):
    target = write_seeded_runs(paths)
    runs = model.target_runs(paths, target)
    assert runs.methods == ("a2c", "ppo")
    assert set(runs.metrics) == {"train/reward", "eval/reward", "losses/policy_loss", "losses/value_loss"}
    assert "step" not in runs.metrics and "transitions" not in runs.metrics
    assert model.target_runs(paths, "other/solo") == model.TargetRuns((), ())


def test_target_stamp_changes_with_metrics_config_and_sweeps(paths):
    target = write_seeded_runs(paths)
    stamp = model.target_stamp(paths, target)
    assert stamp == model.target_stamp(paths, target)
    logs = paths.results / "logs" / target
    files = {path for path, _, _ in stamp}
    assert str(logs / "ppo/version_0/metrics.csv") in files and str(logs / "config.yaml") in files
    assert str(paths.results / "checkpoints" / target / "ppo/tuning_summary.json") in files
    with (logs / "ppo/version_0/metrics.csv").open("a") as stream:
        stream.write("41,328,41.0,,0.1,0.1\n")
    assert model.target_stamp(paths, target) != stamp
