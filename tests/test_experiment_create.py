"""Tests for creating experiments that are valid for their group's paradigm.

The previous template was fixed text: it always wrote intervals_count and
eval_episodes, which offline_rl and supervised forbid, and it never wrote a
_base.yaml for a new group, so the resulting config could not be loaded at all.
"""

import tempfile
import unittest
from pathlib import Path

import pytest
import yaml

from frontend.config_model import (
    ConfigTree,
    ExperimentGroup,
    Model,
    add_method,
    default_method_model,
    experiment_yaml,
    group_base_yaml,
    remove_method,
    unique_method_name,
)


# Groups such as thetaide and tabular_classification are test profiles.
pytestmark = pytest.mark.usefixtures("fixture_workspace")


def parse(text):
    return yaml.safe_load(text)


class TestGroupDiscovery(unittest.TestCase):
    def setUp(self):
        self.tree = ConfigTree.discover()

    def test_group_reads_paradigm_from_base(self):
        self.assertEqual(self.tree.group("thetaide").paradigm, "online_rl")
        self.assertEqual(self.tree.group("tabular_classification").paradigm, "supervised")

    def test_group_reads_env_override_from_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "experiment" / "nested_env"
            directory.mkdir(parents=True)
            (directory / "_base.yaml").write_text(
                "defaults:\n  - override /env: cartpole\n",
                encoding="utf-8",
            )
            self.assertEqual(ExperimentGroup.from_dir(directory).env, "cartpole")

    def test_group_with_base_is_flagged(self):
        self.assertTrue(self.tree.group("thetaide").has_base)

    def test_unknown_group_reports_no_base(self):
        group = self.tree.group("does_not_exist")
        self.assertFalse(group.has_base)
        self.assertIsNone(group.paradigm)

    def test_group_without_paradigm_in_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "experiment" / "no_paradigm"
            directory.mkdir(parents=True)
            (directory / "_base.yaml").write_text("env: cartpole\n", encoding="utf-8")
            group = ExperimentGroup.from_dir(directory)
            self.assertTrue(group.has_base)
            self.assertIsNone(group.paradigm)

    def test_groups_includes_directories_without_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            from src.app.workspace import WorkspacePaths
            root = WorkspacePaths(Path(tmp)).config
            (root / "experiment" / "empty_group").mkdir(parents=True)
            self.assertIn("empty_group", ConfigTree(root).groups)


class TestGroupBaseYaml(unittest.TestCase):
    def setUp(self):
        self.tree = ConfigTree.discover()

    def test_binds_env_and_paradigm(self):
        data = parse(group_base_yaml("cartpole", "online_rl"))
        self.assertEqual(data["paradigm"], "online_rl")
        self.assertIn({"override /env": "cartpole"}, data["defaults"])

    def test_offline_base_pins_fields_the_root_defaults_would_break(self):
        """config.yaml defaults eval_episodes to 100, which offline_rl rejects,
        so the base must override it as the existing offline bases do."""
        data = parse(group_base_yaml("mimic", "offline_rl", self.tree.paradigms["offline_rl"]))
        self.assertNotIn("intervals_count", data)
        self.assertEqual(data["eval_episodes"], 0)

    def test_online_base_leaves_rollout_fields_to_the_defaults(self):
        data = parse(group_base_yaml("cartpole", "online_rl", self.tree.paradigms["online_rl"]))
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_generated_base_matches_the_shape_of_an_existing_offline_base(self):
        """An offline base must pin eval_episodes to 0 and omit intervals_count."""
        generated = parse(group_base_yaml("offline_test", "offline_rl", self.tree.paradigms["offline_rl"]))
        self.assertEqual(generated["eval_episodes"], 0)
        self.assertNotIn("intervals_count", generated)
        self.assertEqual(generated["paradigm"], "offline_rl")

    def test_is_a_global_package(self):
        self.assertTrue(group_base_yaml("mimic", "offline_rl").startswith("# @package _global_"))

    def test_round_trips_through_experiment_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "newgroup"
            directory.mkdir()
            (directory / "_base.yaml").write_text(group_base_yaml("mimic", "offline_rl"))
            group = ExperimentGroup.from_dir(directory)
            self.assertEqual(group.paradigm, "offline_rl")
            self.assertEqual(group.env, "mimic")
            self.assertTrue(group.has_base)


class TestExperimentYaml(unittest.TestCase):
    def setUp(self):
        self.tree = ConfigTree.discover()
        self.online = self.tree.paradigms["online_rl"]
        self.offline = self.tree.paradigms["offline_rl"]

    def test_inherits_the_group_base(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online))
        self.assertEqual(data["defaults"], ["cartpole/_base"])

    def test_sets_experiment_id(self):
        self.assertEqual(parse(experiment_yaml("cartpole", "demo", self.online))["experiment_id"], "demo")

    def test_online_keeps_rollout_fields(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online))
        self.assertNotIn("intervals_count", data)
        self.assertIn("eval_episodes", data)

    def test_offline_omits_forbidden_fields(self):
        """The group base already pins these to their only legal values."""
        data = parse(experiment_yaml("mimic", "demo", self.offline))
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_supervised_omits_forbidden_fields(self):
        data = parse(experiment_yaml("mimic", "demo", self.tree.paradigms["supervised"]))
        self.assertNotIn("intervals_count", data)
        self.assertNotIn("eval_episodes", data)

    def test_unknown_paradigm_keeps_both_fields(self):
        """With no paradigm to consult, stay permissive and let the backend rule."""
        data = parse(experiment_yaml("cartpole", "demo", None))
        self.assertNotIn("intervals_count", data)
        self.assertIn("eval_episodes", data)

    def test_methods_block_is_written(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online, agent="ppo", model="dnn"))
        self.assertEqual(data["methods"]["ppo_dnn"], {"agent": "ppo", "model": "dnn"})

    def test_method_name_follows_agent_model_convention(self):
        data = parse(experiment_yaml("mimic", "demo", self.offline, agent="cql", model="blendrl"))
        self.assertIn("cql_blendrl", data["methods"])

    def test_model_is_not_guessed_when_no_component_is_selected(self):
        data = parse(experiment_yaml("cartpole", "demo", self.online, agent="ppo"))
        self.assertEqual(data["methods"]["ppo"], {"agent": "ppo"})

    def test_no_methods_block_without_an_agent(self):
        self.assertNotIn("methods", parse(experiment_yaml("cartpole", "demo", self.online)))

    def test_output_is_valid_yaml_for_every_paradigm(self):
        for name, paradigm in self.tree.paradigms.items():
            agents = [a.name for a in self.tree.agents_for(name)]
            text = experiment_yaml("grp", "demo", paradigm, agents[0] if agents else None)
            self.assertIsInstance(parse(text), dict, f"{name} produced invalid YAML")


class TestGeneratedExperimentsMatchParadigmRules(unittest.TestCase):
    """The generated file must never contain a field its paradigm forbids."""

    def test_every_group_generates_a_compliant_experiment(self):
        tree = ConfigTree.discover()
        checked = 0
        for name, group in tree.experiment_groups.items():
            if not group.paradigm:
                continue
            paradigm = tree.paradigms.get(group.paradigm)
            if paradigm is None:
                continue
            data = parse(experiment_yaml(name, "demo", paradigm))
            for field in ("intervals_count", "eval_episodes"):
                if not paradigm.field_enabled(field):
                    self.assertNotIn(field, data, f"{name} ({group.paradigm}) wrote forbidden {field}")
            checked += 1
        self.assertGreater(checked, 0, "expected at least one group to declare a paradigm")


class TestMethodEditing(unittest.TestCase):
    """Adding a second method is how an experiment becomes a comparison; 17 of
    the 46 experiments configure more than one."""

    def test_add_uses_the_agent_model_naming_convention(self):
        data = {}
        self.assertEqual(add_method(data, "ppo", "dnn"), "ppo_dnn")

    def test_add_writes_agent_and_model(self):
        data = {}
        name = add_method(data, "cql", "blendrl")
        self.assertEqual(data["methods"][name], {"agent": "cql", "model": "blendrl"})

    def test_add_creates_the_methods_block_when_absent(self):
        data = {"seed": 1}
        add_method(data, "ppo", "dnn")
        self.assertIn("methods", data)

    def test_add_keeps_existing_methods(self):
        data = {"methods": {"ppo_dnn": {"agent": "ppo", "model": "dnn"}}}
        add_method(data, "iql", "dnn")
        self.assertEqual(sorted(data["methods"]), ["iql_dnn", "ppo_dnn"])

    def test_a_repeated_pairing_gets_a_suffix(self):
        data = {}
        first = add_method(data, "ppo", "dnn")
        second = add_method(data, "ppo", "dnn")
        self.assertNotEqual(first, second)
        self.assertEqual(second, "ppo_dnn_2")

    def test_suffixes_keep_climbing(self):
        data = {}
        for _ in range(3):
            add_method(data, "ppo", "dnn")
        self.assertIn("ppo_dnn_3", data["methods"])

    def test_unique_name_respects_names_already_taken(self):
        self.assertEqual(unique_method_name({"ppo_dnn"}, "ppo", "dnn"), "ppo_dnn_2")

    def test_remove_drops_only_the_named_method(self):
        data = {"methods": {"a": {"agent": "ppo"}, "b": {"agent": "iql"}}}
        self.assertTrue(remove_method(data, "a"))
        self.assertEqual(list(data["methods"]), ["b"])

    def test_remove_reports_an_unknown_method(self):
        self.assertFalse(remove_method({"methods": {}}, "nope"))

    def test_params_is_shared_settings_not_a_removable_method(self):
        data = {"methods": {"params": {"lr": 1}, "a": {"agent": "ppo"}}}
        self.assertFalse(remove_method(data, "params"))
        self.assertIn("params", data["methods"])

    def test_removing_the_last_method_drops_the_block(self):
        """An empty methods block satisfies no paradigm; absent lets the base supply one."""
        data = {"methods": {"a": {"agent": "ppo"}}}
        remove_method(data, "a")
        self.assertNotIn("methods", data)

    def test_a_block_left_with_only_params_is_also_dropped(self):
        data = {"methods": {"params": {"lr": 1}, "a": {"agent": "ppo"}}}
        remove_method(data, "a")
        self.assertNotIn("methods", data)


class TestAgentlessParadigms(unittest.TestCase):
    """supervised declares `allowed_agents: []` and still requires methods to be
    non-empty, which it satisfies with model-only methods (ep_lstm, ep_transformer).
    Nothing in the panel may assume a method has an agent."""

    def setUp(self):
        self.tree = ConfigTree.discover()
        self.supervised = self.tree.paradigms["supervised"]

    def test_supervised_declares_no_agents(self):
        self.assertEqual(self.tree.agents_for("supervised"), [])

    def test_group_base_methods_can_be_model_only(self):
        base = self.tree.group("tabular_classification")
        self.assertTrue(base.base_methods, "tabular_classification/_base.yaml should declare methods")
        for name, spec in base.base_methods.items():
            self.assertNotIn("agent", spec, f"{name} unexpectedly names an agent")

    def test_a_model_alone_writes_a_methods_block(self):
        data = parse(experiment_yaml("early_prediction", "demo", self.supervised, model="lstm"))
        self.assertEqual(data["methods"], {"lstm": {"model": "lstm"}})

    def test_a_model_only_method_omits_the_agent_key(self):
        """A null agent would fail the registry lookup rather than be ignored."""
        data = parse(experiment_yaml("early_prediction", "demo", self.supervised, model="lstm"))
        self.assertNotIn("agent", data["methods"]["lstm"])

    def test_add_method_without_an_agent_is_named_for_the_model(self):
        data = {}
        self.assertEqual(add_method(data, None, "transformer"), "transformer")

    def test_add_method_without_an_agent_writes_the_model_only(self):
        data = {}
        name = add_method(data, None, "lstm")
        self.assertEqual(data["methods"][name], {"model": "lstm"})

    def test_repeated_model_only_methods_still_get_suffixes(self):
        data = {}
        add_method(data, None, "lstm")
        self.assertEqual(add_method(data, None, "lstm"), "lstm_2")

    def test_unique_name_handles_a_missing_agent(self):
        self.assertEqual(unique_method_name({"lstm"}, None, "lstm"), "lstm_2")

    def test_removing_a_model_only_method_works(self):
        data = {}
        name = add_method(data, None, "lstm")
        self.assertTrue(remove_method(data, name))

    def test_default_model_follows_the_group_base(self):
        group = ExperimentGroup(name="fixture", base_methods={"chosen": {"model": "fixture_model"}})
        self.assertEqual(default_method_model(group, self.tree.models), "fixture_model")

    def test_group_with_no_methods_has_no_model_default(self):
        empty_group = ExperimentGroup(name="empty", base_methods={})
        chosen = default_method_model(empty_group, self.tree.models)
        self.assertIsNone(chosen)

    def test_unknown_group_has_no_model_default(self):
        chosen = default_method_model(None, self.tree.models)
        self.assertIsNone(chosen)

    def test_explicit_installed_fallback_is_used(self):
        self.assertEqual(default_method_model(None, {"transformer"}, fallback="transformer"), "transformer")

    def test_a_generated_supervised_experiment_satisfies_non_empty_methods(self):
        """The constraint the paradigm declares is the one this must not break."""
        self.assertEqual(self.supervised.requires.get("methods"), "non_empty")
        data = parse(experiment_yaml("early_prediction", "demo", self.supervised, model="lstm"))
        self.assertTrue(data["methods"])


class TestEveryAgentlessParadigm(unittest.TestCase):
    """Whatever agent-less paradigms the config declares, the panel must generate
    valid experiments for all of them. Discovering them from the tree rather than
    naming them means a paradigm added later is covered without editing this."""

    def setUp(self):
        self.tree = ConfigTree.discover()
        self.agentless = {
            name: paradigm
            for name, paradigm in self.tree.paradigms.items()
            if "allowed_agents" in paradigm.raw and not paradigm.allowed_agents
        }

    def test_there_is_more_than_one(self):
        """supervised and unsupervised both forgo agents; neither is a special case."""
        self.assertGreaterEqual(len(self.agentless), 2, f"found only {sorted(self.agentless)}")

    def test_each_generates_a_methods_block_from_a_model_alone(self):
        for name, paradigm in self.agentless.items():
            with self.subTest(paradigm=name):
                data = parse(experiment_yaml("grp", "demo", paradigm, None, "lstm"))
                self.assertEqual(data["methods"], {"lstm": {"model": "lstm"}})

    def test_none_of_them_writes_an_agent_key(self):
        for name, paradigm in self.agentless.items():
            with self.subTest(paradigm=name):
                data = parse(experiment_yaml("grp", "demo", paradigm, None, "lstm"))
                for spec in data["methods"].values():
                    self.assertNotIn("agent", spec)

    def test_each_omits_the_fields_it_forbids(self):
        for name, paradigm in self.agentless.items():
            with self.subTest(paradigm=name):
                data = parse(experiment_yaml("grp", "demo", paradigm, None, "lstm"))
                for field_name in ("eval_episodes",):
                    if not paradigm.field_enabled(field_name):
                        self.assertNotIn(field_name, data)

    def test_each_requires_non_empty_methods(self):
        """Which is why a model-only method had to become possible at all."""
        for name, paradigm in self.agentless.items():
            with self.subTest(paradigm=name):
                self.assertEqual(paradigm.requires.get("methods"), "non_empty")

    def test_a_generated_base_pins_what_the_root_defaults_would_break(self):
        for name, paradigm in self.agentless.items():
            with self.subTest(paradigm=name):
                base = parse(group_base_yaml("mimic", name, paradigm))
                self.assertNotIn("intervals_count", base)
                self.assertEqual(base["eval_episodes"], 0)

    def test_every_group_on_such_a_paradigm_declares_model_only_methods(self):
        checked = 0
        for group_name, group in self.tree.experiment_groups.items():
            if group.paradigm not in self.agentless or not group.base_methods:
                continue
            for method_name, spec in group.base_methods.items():
                if method_name == "params" or not isinstance(spec, dict):
                    continue
                with self.subTest(group=group_name, method=method_name):
                    self.assertNotIn("agent", spec)
                    self.assertIn("model", spec)
                checked += 1
        self.assertGreater(checked, 0, "expected at least one agent-less group with methods")


class TestUnsupervisedParadigmInThePanel(unittest.TestCase):
    """The panel reads src/usr/config/paradigms/, so a new paradigm needs no UI change —
    this is the test that would fail if that stopped being true."""

    def setUp(self):
        self.tree = ConfigTree.discover()

    def test_the_panel_offers_the_paradigm(self):
        self.assertIn("unsupervised", self.tree.paradigms)

    def test_it_permits_no_agents(self):
        self.assertEqual(self.tree.agents_for("unsupervised"), [])

    def test_it_permits_environments(self):
        envs = self.tree.environments_for("unsupervised")
        if not envs and not (self.tree.config_root / "env" / "breast_cancer.yaml").exists():
            self.skipTest("Unsupervised environments not installed")
        self.assertTrue(envs, "expected at least one permitted environment")

    def test_it_disables_the_rollout_fields_with_a_reason(self):
        paradigm = self.tree.paradigms["unsupervised"]
        for field_name in ("eval_episodes",):
            self.assertFalse(paradigm.field_enabled(field_name))
            self.assertIn("unsupervised", paradigm.disabled_reason(field_name))

    def test_a_new_method_defaults_to_the_architecture_the_group_uses(self):
        group = ExperimentGroup(name="fixture", base_methods={"encoder": {"model": "fixture_encoder"}})
        self.assertEqual(default_method_model(group, self.tree.models), "fixture_encoder")

    def test_the_autoencoder_models_are_visible_to_the_panel(self):
        if not self.tree.models:
            self.skipTest("No model components installed")
        self.assertTrue(all(isinstance(model, Model) for model in self.tree.models.values()))


class TestParadigmInheritance(unittest.TestCase):
    """Only 2 of 66 experiments declare `paradigm`; the rest inherit it from the
    group base, so anything reading it must resolve through the group."""

    def setUp(self):
        self.tree = ConfigTree.discover()

    def test_most_experiments_do_not_declare_a_paradigm(self):
        declared = sum(1 for e in self.tree.experiments.values() if "paradigm" in e.raw)
        self.assertLess(declared, len(self.tree.experiments) / 2)

    def test_group_supplies_the_paradigm_the_experiment_omits(self):
        for name, expected in (("tabular_classification", "supervised"), ("thetaide", "online_rl")):
            for experiment in self.tree.experiments_in(name):
                if "paradigm" not in experiment.raw:
                    self.assertEqual(self.tree.group(name).paradigm, expected)
                    break


if __name__ == "__main__":
    unittest.main()
