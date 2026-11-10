"""Execute a resolved training task in a fresh interpreter, independent of the tuner."""
import json
import os
import runpy
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
        from src.app.core.provider_loader import resolve_resource
        from src.app.components.transactions import provider_lease
        from src.app.workspace import WorkspacePaths
        lease = provider_lease(WorkspacePaths.for_process())
        lease.__enter__()
        from src.app.pipeline.provider_worker import run as run_provider_script
        namespace = run_provider_script(str(resolve_resource(script)), run_name='_theta_trial')
        entrypoint = namespace['main']
        # The config is already composed in the coordinator. Calling Hydra's
        # wrapped task avoids starting a second sweep in this worker.
        entrypoint = getattr(entrypoint, '__wrapped__', entrypoint)
        if config.hydra.job.chdir:
            os.chdir(output)
        value = entrypoint(task_config)
        result = {'value': value}
    except BaseException as exc:
        traceback.print_exc()
        result = {'error': f'{type(exc).__name__}: {exc}'}
    finally:
        if "lease" in locals(): lease.__exit__(None, None, None)
    Path(result_path).write_text(json.dumps(result))
    return 1 if 'error' in result else 0


if __name__ == '__main__':
    sys.exit(main())
