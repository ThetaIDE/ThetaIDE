"""Component and Plugin Scaffolding Service for Theta-IDE.

Automates the generation of boilerplate code, runtime protocol stubs,
base configuration profiles, and .theta_component.json manifests for:
  - Models
  - RL Algorithms & Agents
  - Environments, Groundings & Simulators
  - DataModules & Trajectory Samplers
  - Evaluators, Safety & Verification Hooks
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import PROJECT_ROOT
from src.app.workspace import WorkspacePaths

log = logging.getLogger(__name__)


@dataclass
class ScaffoldOptions:
    """Options for generating a new component plugin scaffold."""

    name: str
    kind: str  # "model", "agent", "env", "datamodule", "evaluator"
    author: str = "Theta Researcher"
    version: str = "0.1.0"
    description: str = ""
    paradigms: list[str] = field(default_factory=list)  # Supported paradigms (e.g. ['online_rl'])
    output_dir: Path | None = None


@dataclass
class ScaffoldResult:
    """Result of component scaffolding operation."""

    success: bool
    component_id: str
    target_dir: Path
    created_files: list[Path] = field(default_factory=list)
    error: str | None = None


class PluginScaffolder:
    """Scaffolds new component plugins adhering to Theta-IDE protocols."""

    def __init__(self, workspace_root: Path | None = None):
        self.paths = WorkspacePaths(workspace_root or PROJECT_ROOT)
        self.workspace_root = self.paths.root

    def scaffold(self, options: ScaffoldOptions) -> ScaffoldResult:
        """Generate the full scaffolding files for the requested component kind."""
        clean_name = options.name.strip().lower().replace(" ", "_").replace("-", "_")
        if not clean_name:
            return ScaffoldResult(
                success=False,
                component_id="",
                target_dir=Path("."),
                error="Component name cannot be empty.",
            )

        valid_kinds = {"model", "agent", "algorithm", "env", "datamodule", "evaluator", "plotter", "plot"}
        if options.kind not in valid_kinds:
            return ScaffoldResult(
                success=False,
                component_id=clean_name,
                target_dir=Path("."),
                error=f"Invalid component kind '{options.kind}'. Must be one of {valid_kinds}",
            )

        try:
            if options.kind == "model":
                return self._scaffold_model(clean_name, options)
            elif options.kind in ("agent", "algorithm"):
                return self._scaffold_agent(clean_name, options)
            elif options.kind == "env":
                return self._scaffold_env(clean_name, options)
            elif options.kind == "datamodule":
                return self._scaffold_datamodule(clean_name, options)
            elif options.kind == "evaluator":
                return self._scaffold_evaluator(clean_name, options)
            elif options.kind in ("plotter", "plot"):
                return self._scaffold_plotter(clean_name, options)
            else:
                return ScaffoldResult(
                    success=False,
                    component_id=clean_name,
                    target_dir=Path("."),
                    error=f"Unsupported kind {options.kind}",
                )
        except Exception as exc:
            log.exception("Failed to scaffold component %s", clean_name)
            return ScaffoldResult(
                success=False,
                component_id=clean_name,
                target_dir=Path("."),
                error=str(exc),
            )

    def _scaffold_model(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "src" / "usr" / "models" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "Model"
        desc = opts.description or f"Custom {name} model architecture for Theta-IDE."

        # 1. Python model implementation
        code_file = target_dir / f"{name}_model.py"
        code_content = f'''"""Custom {class_name} neural architecture."""
from typing import Any
import torch
import torch.nn as nn

from src.app.core.model_registry import register_model


@register_model("{name}")
class {class_name}(nn.Module):
    """Custom neural network architecture."""

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        hidden_sizes: list[int] = None,
        device: str = "cpu",
        **kwargs: Any,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        hidden_sizes = hidden_sizes or [64, 64]

        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass through the network."""
        return self.net(obs.float().reshape(obs.shape[0], -1))
'''
        code_file.write_text(code_content, encoding="utf-8")
        created.append(code_file)

        # 2. __init__.py
        init_file = target_dir / "__init__.py"
        init_content = f'''"""Package initialization for {name}."""
from . import {name}_model  # noqa: F401
'''
        init_file.write_text(init_content, encoding="utf-8")
        created.append(init_file)

        # 3. Base config in in/config/model/<name>.yaml
        paradigms = opts.paradigms or ["online_rl", "offline_rl", "supervised"]
        paradigms_yaml = "\n".join(f"  - {p}" for p in paradigms)
        cfg_file = self.workspace_root / "in" / "config" / "model" / f"{name}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_yaml = f"""# Default configuration for {name}
type: {name}
paradigms:
{paradigms_yaml}
hidden_sizes: [64, 64]
"""
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        created.append(cfg_file)

        # 4. Manifest .theta_component.json & plugin.json
        meta = {
            "id": name,
            "name": class_name,
            "kind": "model",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"src/usr/models/{name}",
            "config_path": f"in/config/model/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta["parameter_schema"] = {
            "type": "object",
            "properties": {
                "hidden_sizes": {"type": "array", "items": {"type": "integer"}, "default": [64, 64]},
            },
        }
        meta_file = target_dir / ".theta_component.json"
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        # 5. pyproject.toml for standard entry points
        pyproject_file = target_dir / "pyproject.toml"
        pyproject_content = f"""[project]
name = "theta-model-{name}"
version = "{opts.version}"
description = "{desc}"
dependencies = ["torch>=2.0.0"]

[project.entry-points."theta.models"]
{name} = "src.usr.models.{name}.{name}_model:{class_name}"
"""
        pyproject_file.write_text(pyproject_content, encoding="utf-8")
        created.append(pyproject_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)

    def _scaffold_agent(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "src" / "usr" / "algorithms" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "Agent"
        desc = opts.description or f"Custom {name} algorithm component for Theta-IDE."

        paradigms = opts.paradigms or ["online_rl"]
        paradigms_yaml = "\n".join(f"  - {p}" for p in paradigms)

        # 1. Python agent implementation
        code_file = target_dir / f"{name}_agent.py"
        code_content = f'''"""Custom {class_name} algorithm component."""
from typing import Any, Dict, List
import lightning as L
import torch
import torch.optim as optim

from src.app.core.model_registry import build_model
from src.app.core.protocols import AuxiliaryLossProtocol, OptimizerGroupsProtocol
from src.usr.algorithms.agent_registry import register_agent, register_algorithm


@register_algorithm("{name}")
class {class_name}(L.LightningModule):
    """Pluggable algorithm component extending PyTorch Lightning."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        """Declare which learning paradigms this algorithm supports."""
        return {repr(paradigms)}

    def __init__(self, cfg: Any, env: Any = None, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.cfg = cfg
        self.env = env
        self.lr = float(getattr(cfg, "lr", 0.0003))

        # Model is decoupled and instantiated dynamically via build_model
        model_spec = getattr(cfg, "model", None)
        if not model_spec:
            raise ValueError("This algorithm requires a model component; set model in its agent profile.")
        self.model = build_model(model_spec, env=env)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        obs, action, reward, next_obs, done = batch[:5]
        q_pred = self.model(obs)
        loss = torch.nn.functional.mse_loss(q_pred, reward)

        # Collect optional auxiliary losses from the model
        if isinstance(self.model, AuxiliaryLossProtocol):
            aux_losses = self.model.compute_auxiliary_losses(batch, q_pred)
            for k, v in aux_losses.items():
                loss = loss + v
                self.log(f"train/{{k}}", v)

        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        # Query custom optimizer parameter groups if model provides them
        if isinstance(self.model, OptimizerGroupsProtocol):
            groups = self.model.configure_optimizer_groups(base_lr=self.lr)
            return optim.Adam(groups)
        return optim.Adam(self.parameters(), lr=self.lr)
'''
        code_file.write_text(code_content, encoding="utf-8")
        created.append(code_file)

        # 2. __init__.py
        init_file = target_dir / "__init__.py"
        init_file.write_text(f"from . import {name}_agent  # noqa: F401\n", encoding="utf-8")
        created.append(init_file)

        # 3. Base config in in/config/agent/<name>.yaml
        cfg_file = self.workspace_root / "in" / "config" / "agent" / f"{name}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_yaml = f"""# Default algorithm hyperparameters for {name}
algorithm: {name}
agent: {name}
paradigms:
{paradigms_yaml}
lr: 0.0003
batch_size: 64
gamma: 0.99
epochs: 10
"""
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        created.append(cfg_file)

        # 4. Manifest .theta_component.json & plugin.json
        meta = {
            "id": name,
            "name": class_name,
            "kind": "algorithm",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"src/usr/algorithms/{name}",
            "config_path": f"in/config/agent/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta_file = target_dir / ".theta_component.json"
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        # 5. pyproject.toml
        pyproject_file = target_dir / "pyproject.toml"
        pyproject_content = f"""[project]
name = "theta-agent-{name}"
version = "{opts.version}"
description = "{desc}"
dependencies = ["torch>=2.0.0", "lightning>=2.0.0"]

[project.entry-points."theta.agents"]
{name} = "src.usr.algorithms.{name}.{name}_agent:{class_name}"
"""
        pyproject_file.write_text(pyproject_content, encoding="utf-8")
        created.append(pyproject_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)

    def _scaffold_env(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "in" / "envs" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "EnvironmentPlugin"
        desc = opts.description or f"Custom {name} simulator & grounding environment for Theta-IDE."

        paradigms = opts.paradigms or ["online_rl"]
        paradigms_yaml = "\n".join(f"  - {p}" for p in paradigms)

        # 1. Environment implementation adhering to EnvironmentPluginProtocol
        env_file = target_dir / "env.py"
        env_content = f'''"""Custom environment definition for {name}."""
from typing import Any, Callable, List, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.app.core.protocols import EnvironmentPluginProtocol


class {name.capitalize()}GymEnv(gym.Env):
    """Underlying Gymnasium environment."""

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        return np.zeros((4,), dtype=np.float32), {{}}

    def step(self, action: int):
        obs = np.random.uniform(-0.1, 0.1, size=(4,)).astype(np.float32)
        reward = 1.0
        done = False
        return obs, reward, done, False, {{}}


class {class_name}(EnvironmentPluginProtocol):
    """Environment plugin contract implementation."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        return {repr(paradigms)}

    @classmethod
    def make_env(cls, env_id: str = "{name}", **kwargs: Any) -> gym.Env:
        return {name.capitalize()}GymEnv(**kwargs)

    @classmethod
    def get_default_wrappers(cls) -> List[Callable[[Any], Any]]:
        return []

    @classmethod
    def get_reward_shaping_hook(cls) -> Optional[Callable]:
        return None

    @classmethod
    def get_logic_valuation_hook(cls) -> Optional[Callable]:
        return None
'''
        env_file.write_text(env_content, encoding="utf-8")
        created.append(env_file)

        # 2. __init__.py
        init_file = target_dir / "__init__.py"
        init_file.write_text(
            f"from .env import {class_name}, {name.capitalize()}GymEnv  # noqa: F401\n", encoding="utf-8"
        )
        created.append(init_file)

        # 3. Base config in in/config/env/<name>.yaml
        cfg_file = self.workspace_root / "in" / "config" / "env" / f"{name}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_yaml = f"""# @package _global_
env:
  name: {name}
  paradigms:
{paradigms_yaml}
  monitor_metric: "eval/reward"
  preprocess_on_load: false
  num_envs: 1
  default_plots:
    - convergence
    - losses
"""
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        created.append(cfg_file)

        # 4. Optional starter domain rules in in/rules/<name>/rules.pl
        rules_dir = self.workspace_root / "in" / "rules" / name
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_file = rules_dir / "default.rules"
        rules_file.write_text(
            f"# Domain rules for {name}\n# Example: action_preferred(X) :- state_positive(X).\n", encoding="utf-8"
        )
        created.append(rules_file)

        # 5. Manifest
        meta_file = target_dir / ".theta_component.json"
        meta = {
            "id": name,
            "name": class_name,
            "kind": "env",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"in/envs/{name}",
            "config_path": f"in/config/env/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)

    def _scaffold_datamodule(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "src" / "usr" / "data" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "DataModule"
        desc = opts.description or f"Custom {name} DataModule for Theta-IDE."
        paradigms = opts.paradigms or ["offline_rl", "supervised"]

        code_file = target_dir / f"{name}_data_module.py"
        code_content = f'''"""Custom DataModule adhering to DataModulePluginProtocol."""
from typing import Any, List
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.app.core.paradigm_loader import register_component
from src.app.core.protocols import DataModulePluginProtocol, TrajectorySamplerProtocol


class {name.capitalize()}Dataset(Dataset):
    def __init__(self, size: int = 100):
        self.data = torch.randn(size, 4)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@register_component("{class_name}")
class {class_name}(L.LightningDataModule, DataModulePluginProtocol, TrajectorySamplerProtocol):
    """Custom data module providing training batches and trajectory window sampling."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        return {repr(paradigms)}

    def __init__(self, batch_size: int = 64, **kwargs: Any):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = None

    def setup(self, stage: str = None) -> None:
        self.dataset = {name.capitalize()}Dataset()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def sample_trajectory_window(
        self, batch_size: int, context_len: int
    ) -> dict[str, torch.Tensor]:
        return {{"obs": torch.randn(batch_size, context_len, 4)}}
'''
        code_file.write_text(code_content, encoding="utf-8")
        created.append(code_file)

        init_file = target_dir / "__init__.py"
        init_file.write_text(f"from . import {name}_data_module  # noqa: F401\n", encoding="utf-8")
        created.append(init_file)

        # 3. Base config in in/config/data/<name>.yaml
        cfg_file = self.workspace_root / "in" / "config" / "data" / f"{name}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_yaml = f"""# Default configuration for {name}
data_module: {name}
batch_size: 64
num_workers: 0
"""
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        created.append(cfg_file)

        meta_file = target_dir / ".theta_component.json"
        meta = {
            "id": name,
            "name": class_name,
            "kind": "datamodule",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"src/usr/data/{name}",
            "config_path": f"in/config/data/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)

    def _scaffold_evaluator(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "src" / "usr" / "eval" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "Evaluator"
        desc = opts.description or f"Custom {name} evaluation & verification hook for Theta-IDE."
        paradigms = opts.paradigms or ["online_rl", "offline_rl", "supervised"]

        code_file = target_dir / f"{name}_evaluator.py"
        code_content = f'''"""Custom evaluator adhering to EvaluationHookProtocol."""
from typing import Any, Dict, List
from src.app.core.protocols import EvaluationHookProtocol
from src.app.core.eval_registry import register_evaluator


@register_evaluator("{name}")
class {class_name}(EvaluationHookProtocol):
    """Evaluates agent safety, verification bounds, or saliency attributions."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        return {repr(paradigms)}

    def __init__(self, **kwargs: Any):
        self.step_metrics: list[float] = []

    def on_eval_start(self, agent: Any, env: Any, stage: str = "val") -> None:
        self.step_metrics.clear()

    def evaluate_step(self, agent: Any, batch_or_transition: Any) -> Dict[str, float]:
        metric_val = 1.0
        self.step_metrics.append(metric_val)
        return {{"eval/{name}": metric_val}}

    def on_eval_end(self) -> Dict[str, float]:
        mean_val = sum(self.step_metrics) / max(len(self.step_metrics), 1)
        return {{"eval/{name}_mean": mean_val}}
'''
        code_file.write_text(code_content, encoding="utf-8")
        created.append(code_file)

        init_file = target_dir / "__init__.py"
        init_file.write_text(f"from . import {name}_evaluator  # noqa: F401\n", encoding="utf-8")
        created.append(init_file)

        # 3. Base config in in/config/eval/<name>.yaml
        cfg_file = self.workspace_root / "in" / "config" / "eval" / f"{name}.yaml"
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_yaml = f"""# Default configuration for {name}
evaluator: {name}
enabled: true
frequency: 1
"""
        cfg_file.write_text(cfg_yaml, encoding="utf-8")
        created.append(cfg_file)

        meta_file = target_dir / ".theta_component.json"
        meta = {
            "id": name,
            "name": class_name,
            "kind": "evaluator",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"src/usr/eval/{name}",
            "config_path": f"in/config/eval/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)

    def _scaffold_plotter(self, name: str, opts: ScaffoldOptions) -> ScaffoldResult:
        target_dir = opts.output_dir or (self.workspace_root / "plot" / name)
        target_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []

        class_name = "".join(part.capitalize() for part in name.split("_")) + "Plotter"
        desc = opts.description or f"Custom {name} experiment plotter & visualization component for Theta-IDE."
        paradigms = opts.paradigms or ["online_rl", "offline_rl", "supervised"]

        code_file = target_dir / f"{name}_plotter.py"
        code_content = f'''"""Custom plotter component adhering to BasePlotter and registered via @register_plotter."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot.base import BasePlotter, clean_label, get_style_info, moving_average
from src.app.core.plotter_registry import register_plotter


@register_plotter("{name}")
class {class_name}(BasePlotter):
    """Custom visualization and performance diagnostic plotter."""

    @classmethod
    def get_supported_paradigms(cls) -> List[str]:
        return {repr(paradigms)}

    def __init__(self):
        super().__init__("{name}")

    def run(self, exp_id: str, cli_overrides: Optional[Dict[str, Any]] = None) -> None:
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)
'''
        code_file.write_text(code_content, encoding="utf-8")
        created.append(code_file)

        init_file = target_dir / "__init__.py"
        init_file.write_text(f"from . import {name}_plotter  # noqa: F401\n", encoding="utf-8")
        created.append(init_file)

        # Default configuration in in/config/plot/<name>.yaml
        config_dir = self.workspace_root / "in" / "config" / "plot"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"{name}.yaml"
        config_content = f"""# Default configuration profile for {name} plotter component
metrics:
  - "eval/reward"
  - "train/reward"
  - "train/length"
smoothing_window: 10
dpi: 300
format: "png"
output_dir: null
"""
        config_file.write_text(config_content, encoding="utf-8")
        created.append(config_file)

        meta_file = target_dir / ".theta_component.json"
        meta = {
            "id": name,
            "name": class_name,
            "kind": "plotter",
            "version": opts.version,
            "description": desc,
            "author": {"name": opts.author},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "target_path": f"plot/{name}",
            "config_path": f"in/config/plot/{name}.yaml",
            "compatibility": {"theta": ">=2.0.0"},
            "paradigms": paradigms,
        }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(meta_file)

        plugin_json_file = target_dir / "plugin.json"
        plugin_json_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        created.append(plugin_json_file)

        return ScaffoldResult(success=True, component_id=name, target_dir=target_dir, created_files=created)
