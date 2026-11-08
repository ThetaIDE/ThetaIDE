from omegaconf import OmegaConf

from src.app.pipeline.slurm import generate_sbatch_header, generate_sbatch_script


def test_slurm_compute_node_defaults():
    site_cfg = OmegaConf.create({
        "name": "ncshare",
        "mail_user": "test@example.com",
        "mail_type": "END,FAIL",
        "resources": {
            "partition": "gpu",
            "cores": 16,
            "memory": "32G",
            "time": "01:00:00",
            "gpus": 1,
        },
        "compute_resources": {
            "partition": "common",
            "cores": 4,
            "memory": "8G",
            "time": "01:00:00",
            "gpus": 0,
        },
    })

    # Test 1: Targeting compute nodes (common partition) without explicit cores/memory in exp
    cfg = OmegaConf.create({
        "site": site_cfg,
        "partition": "common",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header = generate_sbatch_header("test_job", "logs", cfg, is_consolidated=False)
    assert "#SBATCH --partition=common" in header
    assert "#SBATCH --cpus-per-task=4" in header
    assert "#SBATCH --mem=8G" in header
    assert "--gres" not in header

    # Test 2: Targeting GPU partition retains 16 cores and 32G
    cfg_gpu = OmegaConf.create({
        "site": site_cfg,
        "partition": "gpu",
        "consolidate": False,
        "resources": {
            "cores": None,
            "memory": None,
        },
    })
    header_gpu = generate_sbatch_header("test_gpu_job", "logs", cfg_gpu, is_consolidated=False)
    assert "#SBATCH --partition=gpu" in header_gpu
    assert "#SBATCH --cpus-per-task=16" in header_gpu
    assert "#SBATCH --mem=32G" in header_gpu
    assert "#SBATCH --gres=gpu:1" in header_gpu


def test_slurm_ncshare_split_sites():
    """Verify live site=ncshare_common and site=ncshare_gpu Hydra profiles."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "in/config"), version_base=None)

    # 1. ncshare_common: CPU partition, 4 cores, 8G mem, consolidate=False, no GPU
    cfg_common = compose(config_name="config", overrides=["site=ncshare_common"])
    header_common = generate_sbatch_header("test_common", "logs", cfg_common)
    assert "#SBATCH --partition=common" in header_common
    assert "#SBATCH --cpus-per-task=4" in header_common
    assert "#SBATCH --mem=8G" in header_common
    assert "--gres" not in header_common
    assert cfg_common.site.consolidate is False

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "in/config"), version_base=None)

    # 2. ncshare_gpu: GPU partition, 16 cores, 32G mem, consolidate=True, 1 GPU
    cfg_gpu = compose(config_name="config", overrides=["site=ncshare_gpu"])
    header_gpu = generate_sbatch_header("test_gpu", "logs", cfg_gpu)
    assert "#SBATCH --partition=gpu" in header_gpu
    assert "#SBATCH --cpus-per-task=16" in header_gpu
    assert "#SBATCH --mem=32G" in header_gpu
    assert "#SBATCH --gres=gpu:1" in header_gpu
    assert cfg_gpu.site.consolidate is True

    GlobalHydra.instance().clear()


def test_slurm_script_is_depersonalized():
    """Verify that generated Slurm scripts contain no hardcoded home paths or user emails."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "in/config"), version_base=None)
    cfg = compose(config_name="config", overrides=["site=ncshare"])
    script = generate_sbatch_script(
        job_name="depersonalized_test",
        cmd_args=["+experiment=cartpole/test"],
        log_dir="results/logs/slurm",
        cfg=cfg,
    )
    assert 'export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"' in script
    assert "/Users/" not in script
    assert "/home/" not in script
    assert "--mail-user" not in script

    GlobalHydra.instance().clear()
