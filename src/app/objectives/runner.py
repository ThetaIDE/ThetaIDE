"""Compatibility entrypoint. Seed execution belongs to the shared variance executor."""
from src.app.pipeline.repeated_task import main, run_objective
from src.app.objectives.evaluation import read_run

if __name__ == "__main__":
    main()
