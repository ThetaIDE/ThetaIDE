"""Execute a resolved training task in a fresh interpreter, independent of the tuner."""
import json
import os
import sys
import traceback
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.conf import HydraConf
from hydra.core.utils import configure_log, setup_globals
from omegaconf import OmegaConf, open_dict


def main():
    from src.app.services.worker_priority import research_activity
    with research_activity():
        return _main()


def _main():
    script, config_path, result_path = sys.argv[1:]
    setup_globals()
    config = OmegaConf.load(config_path)
    schema = OmegaConf.structured(HydraConf)
    OmegaConf.set_struct(schema, False)  # preserve additional plugin Hydra fields
    with open_dict(config):
        config.hydra = OmegaConf.merge(schema, config.hydra)
    output = Path(config.hydra.sweep.dir) / str(config.hydra.sweep.subdir)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with open_dict(config):
        config.hydra.runtime.output_dir = str(output)
    HydraConfig.instance().set_config(config)
    managed = {key: value for key, value in os.environ.items()
               if key.startswith(('THETA_RESOURCE_', 'THETA_MANAGED_')) or key == 'THETAIDE_SETTINGS_FILE'}
    os.environ.update({str(key): str(value) for key, value in config.hydra.job.env_set.items()})
    os.environ.update(managed)
    if os.environ.get('THETA_RESOURCE_POOL'):
        from src.app.services.resource_budget import configure_worker_threads
        configure_worker_threads()
    elif os.environ.get('THETA_MANAGED_THREADS'):
        from src.app.services.resource_budget import managed_environment
        os.environ.update(managed_environment(int(os.environ['THETA_MANAGED_THREADS'])))
    configure_log(config.hydra.job_logging, config.hydra.verbose)
    task_config = OmegaConf.masked_copy(config, [key for key in config if key != 'hydra'])
    metadata = output / '.hydra'
    metadata.mkdir(exist_ok=True)
    OmegaConf.save(task_config, metadata / 'config.yaml')
    OmegaConf.save(config.hydra, metadata / 'hydra.yaml')
    try:
        from src.app.pipeline.engine import DEFAULT_ENGINE_SCRIPT
        application = Path(__file__).resolve().parents[2]
        if script in (DEFAULT_ENGINE_SCRIPT, str(application / "app/train.py")):
            from src.app.train import main
        elif script in ("src/app/pipeline/repeated_task.py", str(application / "app/pipeline/repeated_task.py")):
            from src.app.pipeline.repeated_task import main
        else:
            raise ValueError("Trial workers execute only Theta-owned method/coordinator bootstraps")
        value = main.__wrapped__(task_config)
        result = {'value': value}
    except BaseException as exc:
        traceback.print_exc()
        result = {'error': f'{type(exc).__name__}: {exc}'}
    Path(result_path).write_text(json.dumps(result))
    return 1 if 'error' in result else 0


if __name__ == '__main__':
    sys.exit(main())
