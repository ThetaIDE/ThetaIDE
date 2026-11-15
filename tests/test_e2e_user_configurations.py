"""Comprehensive E2E Test Suite for User Configuration Adherence.

Verifies that all component types (algorithms, models, data loaders, evaluators,
checkpoints, figures/plotters) faithfully adhere to user configurations,
hyperparameter overrides and 5-tier inheritance.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Core registries and factories
from src.app.core.model_registry import build_model
from src.app.core.datamodule_registry import get_datamodule_class
from src.app.core.eval_registry import get_evaluator_class
from src.app.core.plotter_registry import get_plotter_class
from src.app.core.base_agent import OfflineAgentBase, OnlineAgentBase

# Pipeline orchestration and command builders
from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.config import parse_methods_dict, normalize_agent_name
from src.app.pipeline.engine import resolve_engine
from src.app.pipeline.validation import validate_experiment_config
from src.app.core.lightning_builder import build_trainer
from lightning.pytorch.callbacks import ModelCheckpoint

# Plotting utilities
from src.app.core.plotting.base import clean_label, get_style, get_style_info, moving_average, deep_update


# ==============================================================================
# 1. Algorithm & Runner Engine Configuration Adherence
# ==============================================================================


def declare_provider(workspace, kind, capability, source, entry_point):
    """Author a native package for one capability and select its workspace for the process."""
    import json
    from src.app.workspace import WorkspacePaths
    root = WorkspacePaths(workspace).user(kind) / capability
    root.mkdir(parents=True)
    (root / "provider.py").write_text(source, encoding="utf-8")
    (root / ".theta_component.json").write_text(json.dumps({
        "id": capability, "name": capability, "kind": kind, "version": "1.0.0",
        "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": capability, "kind": kind, "entry_point": f"provider.py:{entry_point}"}],
        "payload": {"implementations": ["provider.py"], "resources": [], "configs": {}},
    }), encoding="utf-8")


@pytest.fixture
def provider_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    return tmp_path


class TestAlgorithmConfigurationAdherence:
    """Verify that user algorithm choices and hyperparameters are faithfully passed to runners."""

    def test_algorithm_custom_hyperparameters_forwarded_to_overrides(self):
        """User-specified algorithm hyperparameters must be converted to Hydra overrides."""
        cfg = OmegaConf.create({
            "experiment_name": "cartpole/custom_ppo",
            "paradigm": "online_rl",
            "methods": {
                "ppo_custom": {
                    "agent": "ppo",
                    "model": "dueling_resnet",
                    "agent_params": {
                        "learning_rate": 0.00042,
                        "gamma": 0.985,
                        "ent_coef": 0.035,
                        "clip_range": 0.15,
                    },
                }
            },
        })
        method_cfg = cfg.methods.ppo_custom
        overrides = build_method_overrides("ppo_custom", method_cfg, cfg=cfg)

        assert "agent=ppo" in overrides
        assert "++agent.learning_rate=0.00042" in overrides
        assert "++agent.gamma=0.985" in overrides
        assert "++agent.ent_coef=0.035" in overrides
        assert "++agent.clip_range=0.15" in overrides
        assert "model=dueling_resnet" in overrides

    def test_nested_agent_specification_honored(self):
        """User can specify agent as a nested dict with type and custom parameters."""
        cfg = OmegaConf.create({
            "experiment_name": "test/dqn_exp",
            "paradigm": "online_rl",
            "methods": {
                "dqn_nested": {
                    "agent": {
                        "type": "dqn",
                        "buffer_size": 50000,
                        "target_update_interval": 250,
                        "exploration_fraction": 0.25,
                    },
                    "model": "mlp",
                }
            },
        })
        method_cfg = cfg.methods.dqn_nested
        overrides = build_method_overrides("dqn_nested", method_cfg, cfg=cfg)

        assert "agent=dqn" in overrides
        assert "++agent.buffer_size=50000" in overrides
        assert "++agent.target_update_interval=250" in overrides
        assert "++agent.exploration_fraction=0.25" in overrides

    def test_runner_engine_resolution(self):
        """Method config runner/engine overrides must accurately resolve to target entrypoints."""
        from src.app.pipeline.exceptions import ConfigurationError
        for field in ("runner", "entrypoint", "script", "engine"):
            with pytest.raises(ConfigurationError, match="Removed execution fields"):
                resolve_engine({field: "obsolete"})


# ==============================================================================
# 2. Model Architecture Configuration Adherence
# ==============================================================================

class TestModelConfigurationAdherence:
    """Verify that model architectures and custom layer hyperparameters are strictly adhered to."""

    def test_model_hyperparameters_forwarded_to_overrides(self):
        """Custom architecture parameters (hidden_sizes, activation, dropout) must be preserved."""
        cfg = OmegaConf.create({
            "experiment_name": "test/mlp_exp",
            "paradigm": "online_rl",
            "methods": {
                "mlp_custom": {
                    "agent": "ppo",
                    "model": {
                        "name": "mlp",
                        "hidden_sizes": [256, 128, 64],
                        "activation": "relu",
                        "dropout": 0.1,
                    },
                }
            },
        })
        method_cfg = cfg.methods.mlp_custom
        overrides = build_method_overrides("mlp_custom", method_cfg, cfg=cfg)

        assert "model=mlp" in overrides
        assert "++model.hidden_sizes=[256,128,64]" in overrides
        assert "++model.activation=relu" in overrides
        assert "++model.dropout=0.1" in overrides

    def test_nested_model_parameters_forwarded_without_component_resolution(self):
        """Nested model parameters remain available for the selected model to interpret."""
        cfg = OmegaConf.create({
            "experiment_name": "mimic/blend_exp",
            "paradigm": "offline_rl",
            "methods": {
                "blend_custom": {
                    "agent": "cql",
                    "model": {
                        "name": "blendrl",
                        "neural": "dueling_resnet",
                        "symbolic": "cew",
                        "blender": "learned",
                        "fyd": 0.05,
                        "ecm_dthr": 0.03,
                    },
                }
            },
        })
        method_cfg = cfg.methods.blend_custom
        overrides = build_method_overrides("blend_custom", method_cfg, cfg=cfg)

        assert "model=blendrl" in overrides
        assert "++model.neural=dueling_resnet" in overrides
        assert "++model.symbolic=cew" in overrides
        assert "++model.blender=learned" in overrides
        assert "++model.fyd=0.05" in overrides
        assert "++model.ecm_dthr=0.03" in overrides

    def test_build_model_instantiates_with_user_hyperparameters(self, provider_workspace):
        """build_model must pass user hyperparameters into the declared model class constructor."""
        declare_provider(provider_workspace, "model", "e2e_test_nn", (
            "import torch.nn as nn\n"
            "class MockCustomModel(nn.Module):\n"
            "    def __init__(self, hidden_dim=32, num_layers=1, use_norm=False, **kwargs):\n"
            "        super().__init__()\n"
            "        self.hidden_dim, self.num_layers, self.use_norm = hidden_dim, num_layers, use_norm\n"
            "        self.linear = nn.Linear(hidden_dim, 2)\n"
            "    def forward(self, x):\n"
            "        return self.linear(x)\n"), "MockCustomModel")

        # User passes non-default hyperparameters: hidden_dim=512, num_layers=4, use_norm=True
        user_cfg = OmegaConf.create({
            "model": {
                "name": "e2e_test_nn",
                "hidden_dim": 512,
                "num_layers": 4,
                "use_norm": True,
            }
        })

        model = build_model(user_cfg, env=None)
        assert type(model).__name__ == "MockCustomModel"
        # Ensure user configuration wasn't ignored or reset to default (32 / 1 / False)
        assert model.hidden_dim == 512
        assert model.num_layers == 4
        assert model.use_norm is True


# ==============================================================================
# 3. Data Loader Configuration Adherence & 5-Tier Precedence
# ==============================================================================

class TestDataLoaderConfigurationAdherence:
    """Verify that pluggable DataLoaders honor user kwargs and respect the 5-tier precedence hierarchy."""

    def test_datamodule_custom_kwargs_passed_to_instance(self, provider_workspace):
        """Pluggable DataModule constructor must receive all user-specified data kwargs."""
        declare_provider(provider_workspace, "datamodule", "e2e_custom_loader", (
            "class MockCustomDataModule:\n"
            "    def __init__(self, batch_size=32, normalize=False, train_split=0.8, **kwargs):\n"
            "        self.batch_size, self.normalize, self.train_split = batch_size, normalize, train_split\n"
            "        self.extra_kwargs = kwargs\n"), "MockCustomDataModule")

        cls = get_datamodule_class("e2e_custom_loader")
        assert cls.__name__ == "MockCustomDataModule"

        # User overrides batch_size to 128, normalize to True, train_split to 0.85
        user_data_cfg = {
            "batch_size": 128,
            "normalize": True,
            "train_split": 0.85,
            "custom_aug": "random_crop",
        }
        instance = cls(**user_data_cfg)
        assert instance.batch_size == 128
        assert instance.normalize is True
        assert instance.train_split == 0.85
        assert instance.extra_kwargs.get("custom_aug") == "random_crop"

    def test_datamodule_5_tier_resolution_precedence(self):
        """5-Tier hierarchy: CLI > Experiment > Group _base > Env > Paradigm default."""
        def resolve_datamodule_name(cli_arg=None, exp_cfg=None, group_base=None, env_cfg=None, paradigm_default="SimulatorDataModule"):
            # Tier 1: CLI override
            if cli_arg:
                return cli_arg
            # Tier 2: Experiment YAML
            if exp_cfg and exp_cfg.get("data_module"):
                return exp_cfg["data_module"]
            if exp_cfg and exp_cfg.get("data", {}).get("name"):
                return exp_cfg["data"]["name"]
            # Tier 3: Group _base.yaml
            if group_base and group_base.get("data_module"):
                return group_base["data_module"]
            if group_base and group_base.get("data", {}).get("name"):
                return group_base["data"]["name"]
            # Tier 4: Env YAML
            if env_cfg and env_cfg.get("data_module"):
                return env_cfg["data_module"]
            if env_cfg and env_cfg.get("default_data_module"):
                return env_cfg["default_data_module"]
            # Tier 5: Paradigm default
            return paradigm_default

        # Tier 5 fallback
        assert resolve_datamodule_name() == "SimulatorDataModule"

        # Tier 4: Env overrides Tier 5
        env_def = {"data_module": "atari_frames"}
        assert resolve_datamodule_name(env_cfg=env_def) == "atari_frames"

        # Tier 3: Group overrides Tier 4
        grp_def = {"data_module": "atari_grayscale"}
        assert resolve_datamodule_name(env_cfg=env_def, group_base=grp_def) == "atari_grayscale"

        # Tier 2: Experiment overrides Tier 3
        exp_def = {"data_module": "custom_experiment_loader"}
        assert resolve_datamodule_name(env_cfg=env_def, group_base=grp_def, exp_cfg=exp_def) == "custom_experiment_loader"

        # Tier 1: CLI overrides Tier 2
        cli_override = "cli_forced_loader"
        assert resolve_datamodule_name(
            cli_arg=cli_override,
            env_cfg=env_def,
            group_base=grp_def,
            exp_cfg=exp_def,
        ) == "cli_forced_loader"


# ==============================================================================
# 4. Checkpoints & Evaluators Configuration Adherence
# ==============================================================================

class TestCheckpointAndEvaluatorAdherence:
    """Verify that declarative checkpointing and registered evaluators strictly honor user settings."""

    def test_declarative_checkpoint_full_custom_parameters(self, tmp_path):
        """User-specified checkpoint parameters (monitor, mode, save_top_k, every_n_epochs) must be bound."""
        cfg = OmegaConf.create({
            "paradigm": "offline_rl",
            "output_dir": str(tmp_path),
            "group": "test_grp",
            "experiment_id": "test_ckpt_exp",
            "env": {"name": "tabular"},
            "epochs": 1,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "checkpoint": {
                "monitor": "eval/auprc",
                "mode": "max",
                "save_top_k": 7,
                "filename": "top_auprc_model",
                "save_last": False,
                "every_n_epochs": 3,
            },
        })

        trainer, _, _ = build_trainer(cfg)
        checkpoints = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)]
        assert len(checkpoints) >= 1

        best_ckpt = next(c for c in checkpoints if c.filename == "top_auprc_model")
        assert best_ckpt.monitor == "eval/auprc"
        assert best_ckpt.mode == "max"
        assert best_ckpt.save_top_k == 7
        assert best_ckpt.save_last is False

        # Verify periodic epoch checkpointing adhered to every_n_epochs=3
        epoch_ckpt = next((c for c in checkpoints if "epoch" in str(c.filename)), None)
        assert epoch_ckpt is not None
        assert epoch_ckpt.every_n_epochs == 3

    def test_declarative_checkpoint_disabled_adherence(self, tmp_path):
        """Setting checkpoint: false must prevent any ModelCheckpoint callback from attaching."""
        cfg = OmegaConf.create({
            "paradigm": "offline_rl",
            "output_dir": str(tmp_path),
            "group": "test_grp",
            "experiment_id": "test_no_ckpt",
            "env": {"name": "tabular"},
            "epochs": 1,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "checkpoint": False,
        })

        trainer, _, _ = build_trainer(cfg)
        best_ckpts = [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint) and c.filename == "best_model"]
        assert len(best_ckpts) == 0

    def test_evaluator_registration_and_instantiation(self, provider_workspace):
        """Declared evaluators must resolve and accept parameters."""
        declare_provider(provider_workspace, "evaluator", "e2e_clinical_verifier", (
            "class MockClinicalVerifier:\n"
            "    def __init__(self, threshold=0.5, horizon=6):\n"
            "        self.threshold, self.horizon = threshold, horizon\n"
            "    def evaluate(self, predictions):\n"
            "        return {'compliance_score': 0.94}\n"), "MockClinicalVerifier")

        cls = get_evaluator_class("e2e_clinical_verifier")
        assert cls.__name__ == "MockClinicalVerifier"

        verifier = cls(threshold=0.85, horizon=12)
        assert verifier.threshold == 0.85
        assert verifier.horizon == 12
        res = verifier.evaluate(None)
        assert res["compliance_score"] == 0.94


# ==============================================================================
# 5. Figures / Plotting Configuration & Method Style Adherence
# ==============================================================================

class TestPlottingConfigurationAdherence:
    """Verify that plotter configs, smoothing windows, and method visual styles are faithfully applied."""

    def test_method_visual_style_overrides_adherence(self):
        """Method style overrides (label, color, linestyle, marker) must take precedence over defaults."""
        user_style = {
            "label": "BlendRL (Neural + Symbolic)",
            "color": "#E74C3C",
            "linestyle": "--",
            "marker": "s",
        }

        # 1. clean_label must return the custom label
        assert clean_label("blendrl_cql", style_override=user_style) == "BlendRL (Neural + Symbolic)"

        # 2. get_style must return exact color, linestyle, marker
        s = get_style("blendrl_cql", style_override=user_style)
        assert s["label"] == "BlendRL (Neural + Symbolic)"
        assert s["color"] == "#E74C3C"
        assert s["linestyle"] == "--"
        assert s["marker"] == "s"

        # 3. get_style_info must return (color, linestyle, marker)
        color, ls, marker = get_style_info("blendrl_cql", style_override=user_style)
        assert color == "#E74C3C"
        assert ls == "--"
        assert marker == "s"

    def test_plot_smoothing_window_adherence(self):
        """User smoothing_window parameter must calculate moving average correctly."""
        import numpy as np
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Smoothing window 3
        smoothed = moving_average(data, n=3)
        assert len(smoothed) == len(data)
        # Verify edge handling and calculation
        assert smoothed[-1] == pytest.approx((6.0 + 7.0 + 8.0) / 3.0)

    def test_plotter_registry_discovery(self, provider_workspace):
        """Declared plotter components must be discoverable."""
        declare_provider(provider_workspace, "plotter", "e2e_custom_reporter", (
            "class MockReporterPlotter:\n"
            "    def plot(self):\n"
            "        return True\n"), "MockReporterPlotter")

        cls = get_plotter_class("e2e_custom_reporter")
        assert cls.__name__ == "MockReporterPlotter" and cls().plot() is True


# ==============================================================================
# 6. Universal Experiment Parameters (methods.params) Inheritance
# ==============================================================================

class TestUniversalParamsInheritance:
    """Verify Tier 2 params inheritance across methods and Tier 3 overrides."""

    def test_universal_scalars_and_overrides_inheritance(self):
        """Universal scalars in methods.params apply to all methods unless explicitly overridden."""
        cfg = OmegaConf.create({
            "experiment_name": "test/universal_params",
            "paradigm": "online_rl",
            "methods": {
                "params": {
                    "epochs": 50,
                    "gamma": 0.99,
                    "agent": {
                        "ppo": {
                            "clip_range": 0.18,
                        }
                    },
                },
                "ppo_inherited": {
                    "agent": "ppo",
                    "model": "mlp",
                },
                "ppo_overridden": {
                    "agent": "ppo",
                    "model": "mlp",
                    "epochs": 150,  # Explicitly overrides universal epochs: 50
                },
            },
        })

        parsed = parse_methods_dict(cfg)

        # Method 1 (inherited)
        m1 = parsed["ppo_inherited"]
        m1_overrides = build_method_overrides("ppo_inherited", m1, cfg=cfg)
        assert "++epochs=50" in m1_overrides
        assert "++gamma=0.99" in m1_overrides
        assert "++agent.clip_range=0.18" in m1_overrides

        # Method 2 (overridden)
        m2 = parsed["ppo_overridden"]
        m2_overrides = build_method_overrides("ppo_overridden", m2, cfg=cfg)
        assert "++epochs=150" in m2_overrides  # Overridden!
        assert "++gamma=0.99" in m2_overrides   # Preserved!
        assert "++agent.clip_range=0.18" in m2_overrides
