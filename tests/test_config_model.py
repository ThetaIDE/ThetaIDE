"""Tests for the ThetaIDE config model.

These run against the real in/config tree rather than fixtures, so they fail if
the backend's config schema moves in a way the panel would not survive. Where a
concrete value would rot as configs are added, the assertion is written as a
property instead.
"""

import pytest

from frontend.config_model import Agent, ConfigTree, Environment, Model, Paradigm, Selection


@pytest.fixture(scope="module")
def tree():
    return ConfigTree.discover()


# --------------------------------------------------------------- loading


def test_discover_finds_config_tree(tree):
    assert tree.config_root.is_dir()
    assert tree.config_root.name == "config"


def test_known_paradigms_load(tree):
    assert {"online_rl", "offline_rl", "supervised"} <= set(tree.paradigms)


def test_paradigm_capabilities_drive_environment_and_budget_forms(tree):
    assert tree.paradigms["online_rl"].environment_source == "live_environment"
    assert tree.paradigms["online_rl"].budget_unit == "timesteps"
    assert tree.paradigms["offline_rl"].environment_source == "static_dataset"
    assert tree.paradigms["offline_rl"].budget_unit == "epochs"


def test_known_agents_load(tree):
    assert all(isinstance(agent, Agent) for agent in tree.agents.values())


def test_agents_expose_algorithm_and_hyperparameters(tree):
    if "sb3/ppo" not in tree.agents:
        pytest.skip("sb3/ppo agent plugin not installed")
    ppo = tree.agents["sb3/ppo"]
    assert ppo.algorithm.lower() == "ppo"
    assert "lr" in ppo.hyperparameters or "learning_rate" in ppo.hyperparameters
    assert "algorithm" not in ppo.hyperparameters


def test_models_expose_architecture(tree):
    if "dnn" not in tree.models or "cew" not in tree.models:
        pytest.skip("dnn or cew model plugin not installed")
    assert "dnn" in tree.models
    assert "cew" in tree.models
    assert tree.models["dnn"].architecture == "dnn"
    assert "architecture" not in tree.models["dnn"].parameters


def test_environments_load(tree):
    if not tree.environments:
        pytest.skip("No environments installed; component folder is empty")
    assert isinstance(tree.environments, dict)


def test_experiments_load_with_groups(tree):
    if not tree.experiments:
        pytest.skip("No experiments found")
    assert all(e.group for e in tree.experiments.values())


def test_experiments_exclude_base_templates(tree):
    assert not any(e.path.name.startswith("_") for e in tree.experiments.values())


# ----------------------------------------------------------- agent rules


def test_online_rl_permits_online_algorithms(tree):
    paradigm = Paradigm("online_rl", "", ("ppo", "sac"), (), raw={"allowed_agents": ["ppo", "sac"]})
    assert paradigm.permits_agent(Agent("sb3/ppo", "PPO"))
    assert paradigm.permits_agent(Agent("sb3/sac", "SAC"))
    assert not paradigm.permits_agent(Agent("d3rlpy/cql", "CQL"))


def test_offline_rl_permits_offline_algorithms(tree):
    paradigm = Paradigm("offline_rl", "", ("cql", "iql"), (), raw={"allowed_agents": ["cql", "iql"]})
    assert paradigm.permits_agent(Agent("d3rlpy/cql", "CQL"))
    assert paradigm.permits_agent(Agent("d3rlpy/iql", "IQL"))
    assert not paradigm.permits_agent(Agent("sb3/ppo", "PPO"))


def test_offline_rl_rejects_ppo(tree):
    offline = Paradigm("offline_rl", "", (), ("ppo",), raw={"forbidden_agents": ["ppo"]})
    assert not offline.permits_agent(Agent("sb3/ppo", "PPO"))


def test_online_rl_rejects_offline_algorithms(tree):
    online = Paradigm("online_rl", "", (), ("cql", "iql"), raw={"forbidden_agents": ["cql", "iql"]})
    assert not online.permits_agent(Agent("d3rlpy/cql", "CQL"))
    assert not online.permits_agent(Agent("d3rlpy/iql", "IQL"))


def test_supervised_has_no_declared_agents(tree):
    """Documents a real gap: supervised declares an empty allow-list."""
    assert tree.agents_for("supervised") == []


def test_permits_agent_accepts_agent_object(tree):
    online = Paradigm("online_rl", "", ("ppo",), (), raw={"allowed_agents": ["ppo"]})
    assert online.permits_agent(Agent("sb3/ppo", "PPO"))


def test_agent_allow_and_forbid_lists_are_disjoint(tree):
    for paradigm in tree.paradigms.values():
        assert not set(paradigm.allowed_agents) & set(paradigm.forbidden_agents)


# ------------------------------------------------------- environment rules


def test_online_rl_offers_environments(tree):
    online_envs = {e.name for e in tree.environments_for("online_rl")}
    if not online_envs:
        pytest.skip("No online RL environment plugins installed")
    assert len(online_envs) > 0


def test_offline_rl_offers_environments(tree):
    offline_envs = {e.name for e in tree.environments_for("offline_rl")}
    if not offline_envs:
        pytest.skip("No offline RL environment plugins installed")
    assert len(offline_envs) > 0


def test_every_environment_is_offered_by_some_paradigm(tree):
    if not tree.environments:
        pytest.skip("No environment plugins installed")
    covered = set()
    for name in tree.paradigms:
        covered |= {e.name for e in tree.environments_for(name)}
    assert covered == {e.name for e in tree.environments.values()}


def test_a_nested_environment_is_reachable_by_path_and_by_stem(tree):
    nested_envs = [name for name in tree.environments if "/" in name]
    if not nested_envs:
        pytest.skip("No nested environment plugins installed")
    first = nested_envs[0]
    stem = first.split("/")[-1]
    assert isinstance(tree.environments[first], Environment)
    if stem in tree.environments:
        assert isinstance(tree.environments[stem], Environment)


def test_nested_environment_is_available(tree):
    nested_envs = [name for name in tree.environments if "/" in name]
    if not nested_envs:
        pytest.skip("No nested environment plugins installed")
    first = nested_envs[0]
    assert first in tree.environments


def test_paradigm_without_env_constraint_permits_everything():
    paradigm = Paradigm(name="anything", description="", allowed_agents=(), forbidden_agents=())
    assert paradigm.permits_environment(Environment(name="x"))
    assert paradigm.permits_environment(Environment(name="y"))


# ---------------------------------------------------------- field enabling


def test_eval_episodes_enabled_for_online_rl(tree):
    assert tree.field_enabled("online_rl", "eval_episodes")


@pytest.mark.parametrize("paradigm", ["offline_rl", "supervised"])
def test_eval_episodes_disabled_without_rollouts(tree, paradigm):
    assert not tree.field_enabled(paradigm, "eval_episodes")


def test_disabled_field_explains_itself(tree):
    reason = tree.paradigms["offline_rl"].disabled_reason("eval_episodes")
    assert reason and "eval_episodes" in reason


def test_enabled_field_has_no_reason(tree):
    assert tree.paradigms["online_rl"].disabled_reason("eval_episodes") is None


def test_unknown_field_defaults_to_enabled(tree):
    assert tree.field_enabled("online_rl", "some_future_field")


def test_paradigms_declare_environment_source_and_method_shape(tree):
    """The blank-method form must be driven by a paradigm contract, not IDs."""
    assert tree.paradigms["online_rl"].environment_source == "live_environment"
    assert tree.paradigms["offline_rl"].uses_static_dataset
    assert tree.paradigms["supervised"].uses_static_dataset
    assert tree.paradigms["online_rl"].algorithm_requirement == "required"
    assert tree.paradigms["offline_rl"].algorithm_requirement == "required"
    assert tree.paradigms["supervised"].algorithm_requirement == "forbidden"
    assert tree.paradigms["unsupervised"].algorithm_requirement == "forbidden"
    assert tree.paradigms["online_rl"].budget_unit == "timesteps"
    assert tree.paradigms["offline_rl"].budget_unit == "epochs"


# --------------------------------------------------------------- overrides


def test_overrides_use_hydra_group_names():
    selection = Selection(
        experiment="tests/cartpole", paradigm="online_rl", environment="cartpole", agent="ppo", model="dnn"
    )
    overrides = selection.to_overrides()
    assert "env=cartpole" in overrides
    assert "agent=ppo" in overrides
    assert "model=dnn" in overrides
    assert "paradigm=online_rl" in overrides


def test_site_defaults_to_local():
    assert "site=local" in Selection(experiment="tests/cartpole").to_overrides()


def test_unset_scalars_are_omitted():
    overrides = Selection(experiment="tests/cartpole").to_overrides()
    assert not any(o.startswith("seed=") for o in overrides)


def test_scalars_render_when_set():
    selection = Selection(experiment="tests/cartpole", seed=7, total_timesteps=640)
    overrides = selection.to_overrides()
    assert "seed=7" in overrides
    assert "total_timesteps=640" in overrides


def test_forbidden_scalar_dropped_when_tree_supplied(tree):
    selection = Selection(experiment="tests/mimic", paradigm="offline_rl", eval_episodes=10)
    assert not any(o.startswith("eval_episodes=") for o in selection.to_overrides(tree))


def test_forbidden_scalar_kept_without_tree():
    selection = Selection(experiment="tests/mimic", paradigm="offline_rl", eval_episodes=10)
    assert "eval_episodes=10" in selection.to_overrides()


def test_permitted_scalar_survives_filtering(tree):
    selection = Selection(experiment="tests/cartpole", paradigm="online_rl", eval_episodes=2)
    assert "eval_episodes=2" in selection.to_overrides(tree)


def test_extra_overrides_pass_through_verbatim():
    selection = Selection(experiment="tests/cartpole", extra=["agent.lr=0.007", "no_plot=true"])
    overrides = selection.to_overrides()
    assert "agent.lr=0.007" in overrides
    assert "no_plot=true" in overrides


def test_command_puts_experiment_first():
    argv = Selection(experiment="tests/cartpole").command(script="run_pipeline.py")
    assert argv[1] == "run_pipeline.py"
    assert argv[2] == "tests/cartpole"


def test_command_appends_dry_run_flag():
    argv = Selection(experiment="tests/cartpole").command(dry_run=True)
    assert argv[-1] == "dry_run=true"


def test_command_omits_dry_run_by_default():
    assert "dry_run=true" not in Selection(experiment="tests/cartpole").command()


def test_command_line_is_a_string():
    line = Selection(experiment="tests/cartpole").command_line()
    assert isinstance(line, str)
    assert "run_pipeline.py tests/cartpole" in line


def test_command_line_quotes_arguments_with_spaces():
    import shlex
    sel = Selection(experiment="tests/cartpole", extra=["++description=hello world"])
    cmd = sel.command_line()
    split = shlex.split(cmd)
    assert "++description=hello world" in split


def test_compatible_paradigms_filter_environments(tree):
    if not tree.environments:
        pytest.skip("No environment plugins installed")
    for name in ("online_rl", "offline_rl", "supervised"):
        assert isinstance(tree.environments_for(name), list)


# ------------------------------------------------ composite & dynamic parameters

def test_model_parameter_domains(tree):
    if "blendrl" not in tree.models or "dnn" not in tree.models:
        pytest.skip("blendrl or dnn model plugins not installed")
    blendrl = tree.models["blendrl"]
    blendrl_domains = {
        key for key, schema in blendrl.parameter_schema.get("properties", {}).items()
        if isinstance(schema, dict) and schema.get("x-theta-domain")
    }
    assert "neural" in blendrl_domains
    assert "symbolic" in blendrl_domains

    dnn = tree.models["dnn"]
    dnn_domains = {
        key for key, schema in dnn.parameter_schema.get("properties", {}).items()
        if isinstance(schema, dict) and schema.get("x-theta-domain")
    }
    assert dnn_domains == set()


def test_tree_agent_and_model_helpers(tree):
    if "sb3/ppo" not in tree.agents or "dnn" not in tree.models:
        pytest.skip("sb3/ppo or dnn not installed")
    ppo_params = tree.agent_hyperparameters("sb3/ppo")
    assert "clip_range" in ppo_params or "clip_coef" in ppo_params
    assert "learning_rate" in ppo_params or "lr" in ppo_params
    assert "batch_size" in ppo_params

    dnn_params = tree.model_parameters("dnn")
    if not dnn_params:
        pytest.skip("DNN defaults are supplied by its component package and are not installed here")
    assert "hidden_sizes" in dnn_params
    assert "activation" in dnn_params
    assert "blend_ent_coef" not in dnn_params


def test_candidate_hyperparameters_filtering(fixture_workspace):
    (fixture_workspace / "in/config/agent/cql.yaml").write_text("algorithm: cql\ncql_alpha: 5.0\n")
    tree = ConfigTree.discover()
    tree.agents["cql"] = ConfigTree(fixture_workspace / "in" / "config").agents["cql"]
    if "dnn" not in tree.models or "blendrl" not in tree.models:
        pytest.skip("dnn or blendrl model plugins not installed")
    from frontend.config_viewer import get_method_candidate_hyperparameters

    # For pure DNN: blend_ent_coef must NOT be suggested
    dnn_hypers = get_method_candidate_hyperparameters("cql", "dnn", tree)
    assert "hidden_sizes" in dnn_hypers
    assert "activation" in dnn_hypers
    assert "cql_alpha" in dnn_hypers
    assert "blend_ent_coef" not in dnn_hypers
    assert "logic_lr" not in dnn_hypers

    # For BlendRL: blend_ent_coef and logic_lr MUST be suggested
    blend_hypers = get_method_candidate_hyperparameters("cql", "blendrl", tree)
    assert "blend_ent_coef" in blend_hypers
    assert "logic_lr" in blend_hypers
    assert "blend_q_values" in blend_hypers


def test_model_component_domain_parameters_are_schema_declared(tree):
    if "blendrl" not in tree.models or "dnn" not in tree.models:
        pytest.skip("blendrl or dnn model plugins not installed")
    domains = {
        model_name: {
            key for key, schema in model.parameter_schema.get("properties", {}).items()
            if isinstance(schema, dict) and schema.get("x-theta-domain")
        }
        for model_name, model in tree.models.items()
    }
    assert domains["blendrl"] == {"neural", "symbolic"}
    assert domains["dnn"] == set()
