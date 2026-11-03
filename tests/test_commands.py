"""Unit tests for training command construction (src/app/pipeline/commands.py)."""

import pytest

from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.config import parse_methods_dict


class TestBuildMethodOverrides:
    """Tests for build_method_overrides generating clean Hydra override arguments."""

    def test_cql_dnn_hierarchical_overrides(self):
        """cql_dnn should get agent.cql overrides and model.dnn overrides."""
        cfg_dict = {
            "experiment_name": "mimic/test_cql",
            "paradigm": "offline_rl",
            "methods": {
                "params": {
                    "epochs": 25,
                    "agent": {
                        "cql": {
                            "cql_alpha": 0.1,
                            "lr": 3e-4,
                        }
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        }
                    },
                },
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_dnn",
            method_cfg=methods["cql_dnn"],
            cfg=cfg_dict,
        )

        assert "agent=cql" in overrides
        assert "model=dnn" in overrides
        assert "++agent.name=cql_dnn" in overrides
        assert "++agent.cql_alpha=0.1" in overrides
        assert "++agent.lr=0.0003" in overrides
        assert "++model.hidden_sizes=[512,512]" in overrides
        assert "++agent.epochs=25" in overrides

    def test_dueling_resnet_relies_on_model_defaults(self):
        """dueling_resnet should NOT receive model.dnn overrides, relying on its own defaults."""
        cfg_dict = {
            "experiment_name": "mimic/test_cql",
            "paradigm": "offline_rl",
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "lr": 3e-4,
                        }
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        }
                    },
                },
                "cql_dueling_resnet": {
                    "agent": "cql",
                    "model": "dueling_resnet",
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_dueling_resnet",
            method_cfg=methods["cql_dueling_resnet"],
            cfg=cfg_dict,
        )

        assert "agent=cql" in overrides
        assert "model=dueling_resnet" in overrides
        assert "++agent.lr=0.0003" in overrides
        # Should NOT receive hidden_sizes from dnn
        assert not any("hidden_sizes" in o for o in overrides)

    def test_supervised_paradigm_overrides(self):
        """Supervised learning should configure model without an RL agent."""
        cfg_dict = {
            "experiment_name": "tests/ep",
            "paradigm": "supervised",
            "methods": {
                "params": {
                    "epochs": 1,
                    "batch_size": 128,
                    "model": {
                        "lstm": {
                            "hidden_dim": 64,
                        }
                    },
                },
                "ep_lstm": {
                    "model": "lstm",
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="ep_lstm",
            method_cfg=methods["ep_lstm"],
            cfg=cfg_dict,
        )

        assert "paradigm=supervised" in overrides
        assert "model=lstm" in overrides
        assert "++model.name=ep_lstm" in overrides
        assert "++agent.name=ep_lstm" in overrides
        assert not any(o.startswith("agent=") for o in overrides)
        assert "++model.hidden_dim=64" in overrides
        assert "++batch_size=128" in overrides

    def test_method_level_override_precedence(self):
        """Direct method override should take precedence over universal parameters."""
        cfg_dict = {
            "experiment_name": "mimic/test_cql",
            "paradigm": "offline_rl",
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "batch_size": 1024,
                            "lr": 3e-4,
                        }
                    }
                },
                "cql_fast": {
                    "agent": "cql",
                    "model": "dnn",
                    "batch_size": 256,
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_fast",
            method_cfg=methods["cql_fast"],
            cfg=cfg_dict,
        )

        assert "++agent.batch_size=256" in overrides
        assert "++agent.batch_size=1024" not in overrides
        assert "++agent.lr=0.0003" in overrides

    def test_nested_component_parameters_are_forwarded_without_installed_model_defaults(self):
        """Nested parameters travel through the generic builder without plugin defaults."""
        cfg_dict = {
            "experiment_name": "mimic/final",
            "paradigm": "offline_rl",
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "lr": 3e-4,
                        }
                    },
                },
                "cql_composite_fixture": {
                    "agent": "cql",
                    "model": "composite_fixture",
                    "model_params": {
                        "encoder": {"architecture": "fixture_encoder", "hidden_sizes": [512, 256]},
                        "reasoner": {"type": "fixture_reasoner", "rules": "rigid"},
                    },
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_composite_fixture",
            method_cfg=methods["cql_composite_fixture"],
            cfg=cfg_dict,
        )

        assert "agent=cql" in overrides
        assert "model=composite_fixture" in overrides
        assert "++model.encoder.architecture=fixture_encoder" in overrides
        assert "++model.encoder.hidden_sizes=[512,256]" in overrides
        assert "++model.reasoner.type=fixture_reasoner" in overrides
        assert "++model.reasoner.rules=rigid" in overrides

    def test_nested_component_parameters_preserve_explicit_values(self):
        """The command builder forwards explicitly declared nested component values."""
        cfg_dict = {
            "experiment_name": "mimic/final",
            "paradigm": "offline_rl",
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "lr": 3e-4,
                        }
                    },
                },
                "cql_composite_fixture": {
                    "agent": "cql",
                    "model": "composite_fixture",
                    "model_params": {
                        "encoder": {"architecture": "fixture_encoder"},
                        "reasoner": {
                            "type": "fixture_reasoner",
                            "threshold": 0.03,
                        },
                    },
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_composite_fixture",
            method_cfg=methods["cql_composite_fixture"],
            cfg=cfg_dict,
        )

        assert "agent=cql" in overrides
        assert "model=composite_fixture" in overrides
        assert "++model.encoder.architecture=fixture_encoder" in overrides
        assert "++model.reasoner.type=fixture_reasoner" in overrides
        assert "++model.reasoner.threshold=0.03" in overrides

    def test_fully_hierarchical_model_overrides(self):
        """A nested model declaration is flattened without knowledge of its plugin type."""
        cfg_dict = {
            "experiment_name": "mimic/final",
            "paradigm": "offline_rl",
            "methods": {
                "cql_composite_hierarchical": {
                    "agent": "cql",
                    "model": {
                        "composite_fixture": {
                            "encoder": {"architecture": "fixture_encoder", "hidden_sizes": [256, 128]},
                            "reasoner": {"type": "fixture_reasoner", "threshold": 0.03},
                        }
                    }
                }
            }
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_composite_hierarchical",
            method_cfg=methods["cql_composite_hierarchical"],
            cfg=cfg_dict,
        )
        assert "agent=cql" in overrides
        assert "model=composite_fixture" in overrides
        assert "++model.encoder.architecture=fixture_encoder" in overrides
        assert "++model.encoder.hidden_sizes=[256,128]" in overrides
        assert "++model.reasoner.type=fixture_reasoner" in overrides
        assert "++model.reasoner.threshold=0.03" in overrides

    def test_untuned_method_produces_no_multirun_or_sweeper(self):
        """Untuned methods must NOT receive --multirun or hydra/sweeper overrides."""
        cfg_dict = {
            "experiment_name": "mimic/test_cql",
            "paradigm": "offline_rl",
            "methods": {
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                }
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_dnn",
            method_cfg=methods["cql_dnn"],
            cfg=cfg_dict,
            is_sweep=False,
            extra_args=["--multirun"],  # Even if outer CLI passed multirun, untuned method strips it
        )
        assert "--multirun" not in overrides
        assert "-m" not in overrides
        assert not any("hydra/sweeper" in o for o in overrides)
        assert not any("hydra.sweeper" in o for o in overrides)
        assert "agent=cql" in overrides
        assert "model=dnn" in overrides

    def test_tuned_method_produces_multirun_and_sweeper_params(self):
        """Tuned methods must receive --multirun, sweeper group, and parameter intervals."""
        cfg_dict = {
            "experiment_name": "mimic/test_cql",
            "paradigm": "offline_rl",
            "tuning": {
                "n_trials": 25,
                "direction": "minimize",
            },
            "methods": {
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                    "tune": {
                        "lr": "interval(1e-4, 1e-2)",
                        "cql_alpha": "choice(0.1, 1.0, 5.0)",
                    },
                }
            },
        }
        methods = parse_methods_dict(cfg_dict)
        overrides = build_method_overrides(
            method_name="cql_dnn",
            method_cfg=methods["cql_dnn"],
            cfg=cfg_dict,
            study_name="study_cql_v1",
            is_sweep=True,
        )
        assert "--multirun" in overrides
        assert overrides[0] == "--multirun"
        assert "hydra/sweeper=optuna_offline" in overrides
        assert '++hydra.sweeper.study_name="study_cql_v1"' in overrides
        assert "++hydra.sweeper.n_trials=25" in overrides
        assert "++hydra.sweeper.direction=minimize" in overrides
        assert "agent.lr=interval(1e-4, 1e-2)" in overrides
        assert "agent.cql_alpha=choice(0.1, 1.0, 5.0)" in overrides
        # Ensure tune dict is not passed as a hyperparameter
        assert not any("agent.tune" in o for o in overrides)

    def test_typed_sweep_dimensions(self):
        method = {"agent": "cql", "model": "dnn", "tune": {
            "agent.lr": {"type": "log_float", "low": 0.0001, "high": 0.01},
            "model.width": {"type": "int", "low": 2, "high": 6, "step": 2},
            "agent.enabled": {"type": "choice", "values": [True, False]},
        }}
        overrides = build_method_overrides("cql", method, cfg={"paradigm": "offline_rl"}, is_sweep=True)
        assert "agent.lr=tag(log, interval(0.0001, 0.01))" in overrides
        assert "model.width=range(2, 6, 2)" in overrides
        assert "agent.enabled=choice(true,false)" in overrides

    def test_stepped_float_sweep_compiles_without_integer_truncation(self):
        method = {"agent": "ppo", "model": "dnn", "tune": {
            "agent.lr": {"type": "float", "low": 0.1, "high": 0.5, "step": 0.2},
        }}
        overrides = build_method_overrides("m", method, cfg={"paradigm": "online_rl"})
        assert "agent.lr=range(0.1, 0.5, 0.2)" in overrides

    def test_setting_sweeper_option_still_selects_the_optuna_sweeper(self):
        method = {"agent": "cql", "model": "dnn", "tune": {"agent.lr": "interval(0.1, 0.2)"}}
        overrides = build_method_overrides(
            "cql", method, cfg={"paradigm": "offline_rl"}, is_sweep=True,
            extra_args=["hydra.sweeper.n_trials=5"],
        )
        assert "hydra/sweeper=optuna_offline" in overrides

    def test_sweep_does_not_override_sample_with_base_value(self):
        method = {"agent": "cql", "model": "dnn", "lr": 0.001,
                  "tune": {"agent.lr": {"type": "float", "low": 0.0001, "high": 0.01}}}
        overrides = build_method_overrides("cql", method, cfg={"paradigm": "offline_rl"}, is_sweep=True)
        assert "agent.lr=interval(0.0001, 0.01)" in overrides
        assert not any(arg.startswith("++agent.lr=") for arg in overrides)

    def test_mixed_experiment_single_and_sweep(self):
        """Experiment with one untuned and one tuned method dispatches single vs sweep cleanly."""
        cfg_dict = {
            "experiment_name": "cartpole/mixed",
            "paradigm": "online_rl",
            "methods": {
                "ppo_normal": {
                    "agent": "ppo",
                    "model": "dnn",
                },
                "ppo_tuned": {
                    "agent": "ppo",
                    "model": "dnn",
                    "tune": {
                        "lr": "interval(1e-4, 1e-2)",
                    },
                },
            },
        }
        methods = parse_methods_dict(cfg_dict)

        # Method 1: untuned -> single normal training
        m1_overrides = build_method_overrides(
            method_name="ppo_normal",
            method_cfg=methods["ppo_normal"],
            cfg=cfg_dict,
            is_sweep=False,
        )
        assert "--multirun" not in m1_overrides
        assert not any("hydra/sweeper" in o for o in m1_overrides)

        # Method 2: tuned -> Optuna sweep
        m2_overrides = build_method_overrides(
            method_name="ppo_tuned",
            method_cfg=methods["ppo_tuned"],
            cfg=cfg_dict,
            study_name="cartpole_ppo_tuned_v1",
            is_sweep=True,
        )
        assert "--multirun" in m2_overrides
        assert "hydra/sweeper=optuna_online" in m2_overrides
        assert "agent.lr=interval(1e-4, 1e-2)" in m2_overrides


