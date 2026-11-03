"""Entry point for finalizing a tuning component's Slurm job."""

import sys

from src.app.pipeline.tuners import get_tuner


def main():
    name, group, experiment, method, storage, study = sys.argv[1:]
    get_tuner(name=name).finalize(group, experiment, method, storage or None, study)


if __name__ == "__main__":
    main()
