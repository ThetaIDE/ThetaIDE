import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(APP_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)
# Direct script execution needs only the framework import root. Providers use
# declared package namespaces and never add authored directories to sys.path.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import time
from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

try:
    safe_types = [
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
        omegaconf.base.Container,
    ]
    for node_name in ["AnyNode", "Node", "ValueNode", "UntypedNode"]:
        if hasattr(omegaconf.nodes, node_name):
            safe_types.append(getattr(omegaconf.nodes, node_name))
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_types)
except (AttributeError, TypeError) as e:
    logger.debug("PyTorch safe globals registration skipped: %s", e)
except Exception as e:
    logger.debug("Unexpected error registering safe globals: %s", e)

from src.app.core.lightning_builder import build_trainer, finalize_training
from src.app.core.agent_registry import auto_discover, get_agent_class


from src.app.workspace import WorkspacePaths


@hydra.main(version_base=None, config_path=str(WorkspacePaths.for_process().primary_config), config_name="config")
def main(cfg: DictConfig):
    if os.environ.get('THETA_MANAGED_THREADS'):
        torch.set_num_threads(int(os.environ['THETA_MANAGED_THREADS']))
    if cfg.get("experiment_id", "default_exp") == "default_exp":
        try:
            from hydra.core.hydra_config import HydraConfig

            if HydraConfig.initialized():
                for override in HydraConfig.get().overrides.task:
                    if override.startswith("+experiment=") or override.startswith("experiment="):
                        exp_stem = Path(override.split("=")[-1]).stem
                        cfg.experiment_id = exp_stem
                        break
        except Exception as e:
            logger.debug("Could not infer experiment_id from Hydra task overrides: %s", e)

    print(OmegaConf.to_yaml(cfg))

    paradigm_name = cfg.get("paradigm", "online_rl")
    from src.app.core.paradigm_loader import get_component, load_paradigm_definition

    paradigm_def = load_paradigm_definition(paradigm_name)

    # 1. Resolve Data Module from explicit config (CLI > experiment > group > env) or paradigm fallback
    from src.app.core.datamodule_registry import get_datamodule_class

    dm_cfg = cfg.get("data_module")
    if dm_cfg is None and hasattr(cfg, "env") and hasattr(cfg.env, "get"):
        dm_cfg = cfg.env.get("data_module")

    dm_name = None
    dm_params = {}
    if isinstance(dm_cfg, str):
        dm_name = dm_cfg
    elif hasattr(dm_cfg, "get"):
        dm_name = dm_cfg.get("name") or dm_cfg.get("class") or dm_cfg.get("type")
        dm_params = {k: v for k, v in dm_cfg.items() if k not in ("name", "class", "type", "params")}
        if "params" in dm_cfg and hasattr(dm_cfg["params"], "items"):
            dm_params.update(dict(dm_cfg["params"]))

    DataModuleCls = None
    if dm_name:
        DataModuleCls = get_datamodule_class(dm_name)
        if DataModuleCls is None:
            try:
                DataModuleCls = get_component(dm_name)
            except KeyError as exc:
                raise ValueError(f"Data module '{dm_name}' is not registered") from exc
    else:
        DataModuleCls = paradigm_def.data_module_cls
        if DataModuleCls is None:
            raise ValueError(
                f"Paradigm '{paradigm_name}' does not declare a data module; configure one with 'data_module'."
            )

    supported = getattr(DataModuleCls, "get_supported_paradigms", None)
    if dm_name and callable(supported) and paradigm_name not in supported():
        raise ValueError(f"Data module '{dm_name}' does not support paradigm '{paradigm_name}'")

    datamodule = DataModuleCls(cfg, **dm_params) if dm_params else DataModuleCls(cfg)

    # 2. Build Model / Agent based on paradigm
    # Driven by whether the paradigm declares agents, not by its name, so an
    # agent-less paradigm such as unsupervised builds a model the same way.
    from src.app.pipeline.validation import paradigm_uses_agents

    if not paradigm_uses_agents(paradigm_name):
        input_dim = getattr(datamodule, "input_dim", 64)

        model_cfg = cfg.get("model", {}) if hasattr(cfg, "get") else getattr(cfg, "model", {})
        method_name = str(model_cfg.get("name", ""))
        method_cfg = cfg.get("methods", {}).get(method_name, {}) if hasattr(cfg, "get") else {}
        arch_name = model_cfg.get("architecture") or model_cfg.get("type") or model_cfg.get("arch")
        if not arch_name and hasattr(method_cfg, "get"):
            arch_name = method_cfg.get("model")
        if not arch_name:
            raise ValueError("The selected model component must declare an architecture name.")
        arch_name = str(arch_name).lower()
        lr = float(model_cfg.get("lr", cfg.get("lr", 1e-3)))

        # Model components select their own training module; the IDE does not
        # infer one from architecture names or training paradigm labels.
        target_module_name = model_cfg.get("lightning_module") or model_cfg.get("module")
        if target_module_name:
            try:
                model_cls = get_component(target_module_name)
            except KeyError as exc:
                raise ValueError(
                    f"Model component '{arch_name}' declares unavailable Lightning module '{target_module_name}'."
                ) from exc
        else:
            try:
                model_cls = get_component(arch_name)
            except KeyError as exc:
                raise ValueError(
                    f"Model component '{arch_name}' must register a training module or declare 'lightning_module'."
                ) from exc

        kwargs = {}
        if hasattr(model_cfg, "items"):
            reserved = {
                "architecture",
                "name",
                "lightning_module",
                "module",
                "lr",
                "type",
                "epochs",
                "eval_every_n_epochs",
            }
            for k, v in model_cfg.items():
                if k not in reserved and v is not None:
                    kwargs[k] = v

        for k in (
            "hidden_dim",
            "num_layers",
            "dropout",
            "use_dual_pooling",
            "use_tcn_conv",
            "bidirectional",
            "d_model",
            "nhead",
            "dim_feedforward",
            "pos_type",
            "max_len",
            "use_cls_token",
            "use_focal_loss",
            "pos_weight",
            "weight_decay",
        ):
            if k not in kwargs and hasattr(cfg, "get") and cfg.get(k) is not None:
                kwargs[k] = cfg.get(k)

        print(
            f"{paradigm_name} paradigm: constructing {model_cls.__name__} "
            f"({arch_name.upper()}, input_dim={input_dim}, lr={lr})"
        )
        model = model_cls(architecture_name=arch_name, input_dim=input_dim, lr=lr, **kwargs)
    else:
        auto_discover()
        agent_cfg = cfg.agent
        base_algo_name = agent_cfg.get("algorithm", agent_cfg.get("name", None))
        print(f"Extracted algorithm name: {base_algo_name}")

        if not base_algo_name:
            raise ValueError("Could not extract algorithm name from config.")

        AgentClass = get_agent_class(base_algo_name)
        print(f"Resolved agent class: {AgentClass.__name__}")
        model = AgentClass(cfg)

    trainer, ckpt_dir, ckpt_path = build_trainer(cfg, model)

    from src.app.telemetry.lightning import finish_telemetry

    start_time = time.time()
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    except Exception as exc:
        if getattr(exc, "theta_trial_status", None) == "pruned":
            finish_telemetry(trainer, status="pruned")
            return {"theta_trial_status": "pruned"}
        raise
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n[Training Complete] Total execution time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)")

    try:
        metric = finalize_training(trainer, cfg, ckpt_dir, training_time, start_time, end_time)
    except Exception:
        finish_telemetry(trainer, status="failed")
        raise
    finish_telemetry(trainer)  # after finalize_training, whose last values (training time, GPU) are streamed too
    return metric


def launch():
    paths = WorkspacePaths.for_process()
    search = [f"file://{p}" for p in paths.config_roots() if p.is_dir() and p != paths.primary_config]
    if not any(arg.lstrip('+').startswith('hydra.searchpath=') for arg in sys.argv[1:]):
        sys.argv.append('hydra.searchpath=[' + ','.join(search) + ']')
    main()


if __name__ == "__main__":
    launch()
