import pytest

from src.usr.algorithms.agent_registry import (
    AGENT_REGISTRY,
    ALGORITHM_REGISTRY,
    auto_discover,
    get_agent_class,
    get_algorithm_class,
    list_registered_agents,
    list_registered_algorithms,
    register_agent,
    register_algorithm,
)


def declare_agent(root,name,class_name):
    import json
    package=root/'src/usr/algorithms'/name;package.mkdir(parents=True)
    (package/'agent.py').write_text(f'class {class_name}: pass\n')
    (package/'.theta_component.json').write_text(json.dumps({'id':name,'name':name,'kind':'algorithm','version':'1.0.0','storage_version':2,'api_version':1,
        'capabilities':[{'id':name,'kind':'algorithm','entry_point':'agent.py:'+class_name}],
        'payload':{'implementations':['agent.py'],'resources':[],'configs':{}}}))


@pytest.fixture(autouse=True)
def clean_registry():
    auto_discover()
    original = AGENT_REGISTRY.copy()
    AGENT_REGISTRY.clear()
    yield
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY.update(original)


def test_register_agent_single():
    @register_agent("my_algo")
    class DummyAgent:
        pass

    assert "my_algo" in AGENT_REGISTRY
    assert AGENT_REGISTRY["my_algo"] is DummyAgent


def test_register_agent_multiple():
    @register_agent("algo1", "algo2")
    class DummyAgent:
        pass

    assert "algo1" in AGENT_REGISTRY
    assert "algo2" in AGENT_REGISTRY


def test_get_agent_class_exact_match(tmp_path, monkeypatch):
    @register_agent("exact")
    class DummyAgent:
        pass

    monkeypatch.setenv("THETA_WORKSPACE",str(tmp_path))
    with pytest.raises(ValueError):get_agent_class("exact")
    declare_agent(tmp_path,"exact","DummyAgent")
    assert get_agent_class("exact").__name__=="DummyAgent"


def test_get_agent_class_longest_prefix(tmp_path, monkeypatch):
    @register_agent("base")
    class BaseAgent:
        pass

    @register_agent("base_advanced")
    class AdvancedAgent:
        pass

    monkeypatch.setenv("THETA_WORKSPACE",str(tmp_path))
    declare_agent(tmp_path,"base","BaseAgent");declare_agent(tmp_path,"base_advanced","AdvancedAgent")
    cls1 = get_agent_class("base_advanced_suffix")
    assert cls1.__name__=="AdvancedAgent"

    cls2 = get_agent_class("base_other")
    assert cls2.__name__=="BaseAgent"


def test_get_agent_unknown_raises():
    with pytest.raises(ValueError, match="Unknown algorithm provider"):
        get_agent_class("unknown_algo")


def test_list_registered_agents():
    @register_agent("z_algo", "a_algo")
    class DummyAgent:
        pass

    agents = list_registered_agents()
    assert isinstance(agents, list)
    assert "a_algo" not in agents and "z_algo" not in agents
    assert "a_algo" in AGENT_REGISTRY and "z_algo" in AGENT_REGISTRY


def test_algorithm_aliases():
    @register_algorithm("custom_algo")
    class CustomAlgo:
        pass

    assert "custom_algo" in ALGORITHM_REGISTRY
    with pytest.raises(ValueError):get_algorithm_class("custom_algo")
    algos = list_registered_algorithms()
    assert "custom_algo" not in algos

