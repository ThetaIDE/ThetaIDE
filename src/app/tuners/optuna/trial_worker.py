"""Compatibility entrypoint for the shared resolved-task worker."""
from src.app.pipeline.task_worker import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
