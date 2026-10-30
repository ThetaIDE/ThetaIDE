"""Tests for pipeline configuration utilities (src/pipeline/config.py)."""

import yaml

from src.app.pipeline.config import normalize_agent_name


class TestNormalizeAgentName:
    """Tests for agent name normalization (slash to underscore, etc)."""

    def test_slash_to_underscore(self):
        result = normalize_agent_name("ppo/cp_tuned")
        assert "/" not in result
        assert "_" in result

    def test_already_normalized(self):
        result = normalize_agent_name("ppo_cp_tuned")
        assert result == "ppo_cp_tuned"

    def test_empty_string(self):
        result = normalize_agent_name("")
        assert result == ""


class TestParseMethodsDict:
    """Tests for parse_methods_dict and shared parameter inheritance."""

    def test_basic_methods_without_params(self):
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_dnn": {"agent": "cql", "model": "dnn", "lr": 1e-3},
                "cql_resnet": {"agent": "cql", "model": "dueling_resnet"},
            }
        }
        res = parse_methods_dict(cfg)
        assert len(res) == 2
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["lr"] == 1e-3
        assert res["cql_resnet"]["model"] == "dueling_resnet"

    def test_shared_params_under_methods(self):
        """methods.params should be merged into all methods and excluded from returned keys."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": "cql",
                    "epochs": 25,
                    "lr": 3e-4,
                },
                "cql_dnn": {"model": "dnn"},
                "cql_resnet": {"model": "dueling_resnet", "lr": 1e-4},
            }
        }
        res = parse_methods_dict(cfg)
        assert "params" not in res
        assert len(res) == 2
        # cql_dnn inherited agent and lr
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["model"] == "dnn"
        assert res["cql_dnn"]["epochs"] == 25
        assert res["cql_dnn"]["lr"] == 3e-4
        # cql_resnet overridden lr
        assert res["cql_resnet"]["agent"] == "cql"
        assert res["cql_resnet"]["model"] == "dueling_resnet"
        assert res["cql_resnet"]["epochs"] == 25
        assert res["cql_resnet"]["lr"] == 1e-4

    def test_top_level_params(self):
        """Top-level params should be merged into all methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "params": {"epochs": 50, "gamma": 0.95},
            "methods": {
                "ppo_dnn": {"agent": "ppo", "model": "dnn"},
            }
        }
        res = parse_methods_dict(cfg)
        assert res["ppo_dnn"]["epochs"] == 50
        assert res["ppo_dnn"]["gamma"] == 0.95

    def test_nested_deep_merge(self):
        """Nested dictionaries should be deep-merged rather than completely replaced."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": "cql",
                    "blender": {"mode": "neural", "blend_function": "softmax"},
                },
                "cql_blendrl": {
                    "model": "blendrl",
                    "blender": {"blend_function": "gumbel"},
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert res["cql_blendrl"]["blender"]["mode"] == "neural"
        assert res["cql_blendrl"]["blender"]["blend_function"] == "gumbel"

    def test_reserved_keys_excluded(self):
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {"lr": 1e-3},
                "_params_": {"gamma": 0.99},
                "defaults": {"epochs": 10},
                "common": {"batch_size": 32},
                "real_method": {"agent": "ppo", "model": "dnn"},
            }
        }
        res = parse_methods_dict(cfg)
        assert list(res.keys()) == ["real_method"]
        assert res["real_method"]["lr"] == 1e-3
        assert res["real_method"]["gamma"] == 0.99
        assert res["real_method"]["epochs"] == 10
        assert res["real_method"]["batch_size"] == 32

    def test_hierarchical_agent_and_model_overrides(self):
        """params.agent.<algo> and params.model.<arch> should apply selectively to matching methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "epochs": 25,
                    "eval_every_n_epochs": 2,
                    "gamma": 0.95,
                    "agent": {
                        "cql": {
                            "cql_alpha": 0.1,
                            "batch_size": 1024,
                            "lr": 3e-4,
                        },
                        "ppo": {
                            "clip_coef": 0.2,
                            "batch_size": 64,
                        },
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        },
                        "blendrl": {
                            "blend_q_values": True,
                            "rules": "rigid",
                        },
                    },
                },
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                },
                "cql_dueling_resnet": {
                    "agent": "cql",
                    "model": "dueling_resnet",
                },
                "ppo_dnn": {
                    "agent": "ppo",
                    "model": "dnn",
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert len(res) == 3

        # cql_dnn gets universal globals + agent.cql + model.dnn
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["model"] == "dnn"
        assert res["cql_dnn"]["epochs"] == 25
        assert res["cql_dnn"]["gamma"] == 0.95
        assert res["cql_dnn"]["cql_alpha"] == 0.1
        assert res["cql_dnn"]["batch_size"] == 1024
        assert res["cql_dnn"]["lr"] == 3e-4
        assert res["cql_dnn"]["hidden_sizes"] == [512, 512]
        assert "clip_coef" not in res["cql_dnn"]
        assert "rules" not in res["cql_dnn"]

        # cql_dueling_resnet gets agent.cql but NOT model.dnn (relies on dueling_resnet defaults)
        assert res["cql_dueling_resnet"]["agent"] == "cql"
        assert res["cql_dueling_resnet"]["model"] == "dueling_resnet"
        assert res["cql_dueling_resnet"]["cql_alpha"] == 0.1
        assert res["cql_dueling_resnet"]["batch_size"] == 1024
        assert "hidden_sizes" not in res["cql_dueling_resnet"]

        # ppo_dnn gets agent.ppo + model.dnn, but NOT agent.cql
        assert res["ppo_dnn"]["agent"] == "ppo"
        assert res["ppo_dnn"]["model"] == "dnn"
        assert res["ppo_dnn"]["clip_coef"] == 0.2
        assert res["ppo_dnn"]["batch_size"] == 64
        assert res["ppo_dnn"]["hidden_sizes"] == [512, 512]
        assert "cql_alpha" not in res["ppo_dnn"]

    def test_method_override_precedence_over_universal(self):
        """Method-level declarations must override both universal globals and agent/model blocks."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "lr": 3e-4,
                            "batch_size": 1024,
                        }
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        }
                    },
                },
                "cql_custom": {
                    "agent": "cql",
                    "model": "dnn",
                    "batch_size": 256,
                    "hidden_sizes": [1024, 512, 256],
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert res["cql_custom"]["lr"] == 3e-4
        assert res["cql_custom"]["batch_size"] == 256
        assert res["cql_custom"]["hidden_sizes"] == [1024, 512, 256]

    def test_nested_model_component_params_pass_through_unchanged(self):
        """Nested component params stay opaque for the model component to resolve."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "model": {
                        "dense_encoder": {"hidden_sizes": [256, 128], "dropout": 0.1},
                        "rule_engine": {"rules": "fixture_rules", "threshold": 0.7},
                    }
                },
                "custom_method": {
                    "agent": "test_agent",
                    "model": {
                        "custom_composite": {
                            "encoder": "dense_encoder",
                            "reasoner": {"rule_engine": {"threshold": 0.9}},
                        }
                    },
                },
            }
        }
        resolved = parse_methods_dict(cfg)["custom_method"]
        assert resolved["model"]["name"] == "custom_composite"
        assert resolved["model_params"]["encoder"] == "dense_encoder"
        assert resolved["model_params"]["reasoner"] == {"rule_engine": {"threshold": 0.9}}

    def test_nested_model_params_do_not_infer_child_components(self):
        """Nested dictionaries are preserved rather than converted to child slots."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "model": {
                        "custom_composite": {"blend_mode_index": 1},
                        "encoder": {"widths": [128, 64]},
                    }
                },
                "first": {
                    "agent": "runner_a",
                    "model": {"custom_composite": {"encoder": {"encoder": {}}}},
                },
                "second": {
                    "agent": "runner_b",
                    "model": {"custom_composite": {"encoder": {"encoder": {"widths": [32]}}}},
                },
            }
        }
        resolved = parse_methods_dict(cfg)
        assert resolved["first"]["model_params"]["blend_mode_index"] == 1
        assert resolved["first"]["model_params"]["encoder"] == {"encoder": {}}
        assert resolved["second"]["model_params"]["encoder"] == {"encoder": {"widths": [32]}}
        assert "modules" not in resolved["second"]["model_params"]

    def test_group_method_resolution(self, monkeypatch):
        """Methods defined in in/config/experiment/<group>/methods/ should resolve automatically."""
        from src.app.pipeline.config import parse_methods_dict

        mock_templates = {
            "cql_blendrl_cew_dueling_resnet": {
                "agent": "cql",
                "model": {
                    "blendrl": {
                        "neural": "dueling_resnet",
                        "symbolic": {"cew": {"ecm_dthr": 0.03}},
                    }
                },
                "style": {"label": "BlendRL CEW+ResNet"},
            },
            "cql_dnn": {
                "agent": "cql",
                "model": "dnn",
                "style": {"label": "DNN"},
            },
        }
        monkeypatch.setattr(
            "src.app.pipeline.config.find_group_method_config",
            lambda name, group=None: mock_templates.get(name, {}),
        )

        cfg = {
            "group": "test_group",
            "methods": {
                "cql_blendrl_cew_dueling_resnet": None,
                "cql_dnn": {},
            },
        }
        res = parse_methods_dict(cfg)
        assert "cql_blendrl_cew_dueling_resnet" in res
        m_cew = res["cql_blendrl_cew_dueling_resnet"]
        assert m_cew["agent"] == "cql"
        assert (m_cew["model"] == "blendrl" or m_cew["model"]["name"] == "blendrl")
        assert m_cew["model_params"]["neural"] == "dueling_resnet"
        assert m_cew["model_params"]["symbolic"] == {"cew": {"ecm_dthr": 0.03}}
        assert "style" in m_cew
        assert m_cew["style"]["label"] == "BlendRL CEW+ResNet"

        assert "cql_dnn" in res
        m_dnn = res["cql_dnn"]
        assert m_dnn["agent"] == "cql"
        assert m_dnn["model"] == "dnn"
        assert m_dnn["style"]["label"] == "DNN"

    def test_group_method_override_precedence(self, monkeypatch):
        """Experiment-level overrides on top of group methods should take precedence."""
        from src.app.pipeline.config import parse_methods_dict

        mock_templates = {
            "cql_blendrl_cew_dueling_resnet": {
                "agent": "cql",
                "model": {
                    "blendrl": {
                        "neural": "dueling_resnet",
                        "symbolic": {"cew": {"ecm_dthr": 0.03}},
                    }
                },
                "style": {"label": "BlendRL CEW+ResNet"},
            },
        }
        monkeypatch.setattr(
            "src.app.pipeline.config.find_group_method_config",
            lambda name, group=None: mock_templates.get(name, {}),
        )

        cfg = {
            "group": "test_group",
            "methods": {
                "cql_blendrl_cew_dueling_resnet": {
                    "model": {
                        "blendrl": {
                            "symbolic": {
                                "cew": {
                                    "ecm_dthr": 0.09
                                }
                            }
                        }
                    }
                }
            },
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl_cew_dueling_resnet"]
        assert m["model_params"]["symbolic"]["cew"]["ecm_dthr"] == 0.09
        assert m["model_params"]["neural"] == "dueling_resnet"
        assert m["style"]["label"] == "BlendRL CEW+ResNet"

    def test_group_method_base_decoupling(self, monkeypatch):
        """Methods can declare a custom key with base: pointing to a group method template."""
        from src.app.pipeline.config import parse_methods_dict

        mock_templates = {
            "cql_dnn": {
                "agent": "cql",
                "model": "dnn",
                "style": {"label": "DNN"},
            },
        }
        monkeypatch.setattr(
            "src.app.pipeline.config.find_group_method_config",
            lambda name, group=None: mock_templates.get(name, {}),
        )

        cfg = {
            "group": "test_group",
            "methods": {
                "custom_baseline": {
                    "base": "cql_dnn",
                    "lr": 5e-4,
                }
            },
        }
        res = parse_methods_dict(cfg)
        assert "custom_baseline" in res
        m = res["custom_baseline"]
        assert m["agent"] == "cql"
        assert m["model"] == "dnn"
        assert m["lr"] == 5e-4
        assert m["style"]["label"] == "DNN"

    def test_method_level_tune_search_space(self):
        """Methods can declare their own tune: block specifying search space intervals."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                    "tune": {
                        "lr": "interval(1e-4, 1e-2)",
                        "cql_alpha": "choice(0.1, 1.0)",
                    },
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_dnn"]
        assert "tune" in m
        assert m["tune"]["lr"] == "interval(1e-4, 1e-2)"
        assert m["tune"]["cql_alpha"] == "choice(0.1, 1.0)"
        # Ensure tune was not copied into agent_params as a literal parameter
        assert "tune" not in m.get("agent_params", {})

    def test_shared_params_tune_inheritance(self):
        """methods.params.tune provides universal search spaces inherited by all methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "tune": {
                        "lr": "interval(1e-4, 1e-2)",
                    }
                },
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                    "tune": {
                        "cql_alpha": "choice(0.1, 1.0)",
                    },
                },
                "ppo_dnn": {
                    "agent": "ppo",
                    "model": "dnn",
                },
            }
        }
        res = parse_methods_dict(cfg)
        # cql_dnn merges shared + method tune
        assert res["cql_dnn"]["tune"]["lr"] == "interval(1e-4, 1e-2)"
        assert res["cql_dnn"]["tune"]["cql_alpha"] == "choice(0.1, 1.0)"
        # ppo_dnn inherits shared tune
        assert res["ppo_dnn"]["tune"]["lr"] == "interval(1e-4, 1e-2)"

    def test_from_study_loads_best_params(self, tmp_path, monkeypatch):
        """from_study loads winning best_params.yaml into agent and model parameters."""
        import yaml

        from src.app.pipeline.config import parse_methods_dict

        study_dir = tmp_path / "results" / "checkpoints" / "mimic" / "tune_cql" / "cql_dnn"
        study_dir.mkdir(parents=True)
        best_params = {
            "agent.lr": 0.0005,
            "agent.cql_alpha": 2.5,
            "hidden_sizes": [512, 512],
        }
        with open(study_dir / "best_params.yaml", "w") as f:
            yaml.dump(best_params, f)

        monkeypatch.chdir(tmp_path)

        cfg = {
            "group": "mimic",
            "methods": {
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                    "from_study": "mimic/tune_cql",
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_dnn"]
        assert m["agent_params"]["lr"] == 0.0005
        assert m["agent_params"]["cql_alpha"] == 2.5
        assert m["model_params"]["hidden_sizes"] == [512, 512]





def test_shared_component_values_and_sweeps_follow_inheritance():
    from src.app.pipeline.config import parse_methods_dict

    sweep = {"type": "float", "low": 0.0001, "high": 0.001}
    cfg = {"methods": {
        "params": {"agent": {"name": "sb3/ppo", "learning_rate": 0.0003},
                   "tune": {"agent.learning_rate": sweep}},
        "one": {"agent": "sb3/ppo", "model": "dnn"},
        "two": {"agent": {"name": "sb3/ppo", "learning_rate": 0.0007}, "model": "dnn"},
        "three": {"agent": "sb3/a2c", "model": "dnn"},
    }}
    result = parse_methods_dict(cfg)
    assert result["one"]["agent_params"]["learning_rate"] == 0.0003
    assert result["one"]["tune"]["agent.learning_rate"] == sweep
    assert result["two"]["agent_params"]["learning_rate"] == 0.0007
    assert "tune" not in result["two"]
    assert "learning_rate" not in result["three"].get("agent_params", {})
    assert "tune" not in result["three"]


def test_component_qualified_shared_sweep():
    from src.app.pipeline.config import parse_methods_dict

    sweep = {"type": "float", "low": 0.0001, "high": 0.001}
    cfg = {"methods": {"params": {"agent": {"sb3/ppo": {"learning_rate": 0.0003}},
                                 "tune": {"agent.sb3/ppo.learning_rate": sweep}},
                       "one": {"agent": "sb3/ppo", "model": "dnn"},
                       "two": {"agent": "sb3/a2c", "model": "dnn"}}}
    result = parse_methods_dict(cfg)
    assert result["one"]["tune"] == {"agent.learning_rate": sweep}
    assert "tune" not in result["two"]
