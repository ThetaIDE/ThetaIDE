"""Compatibility entrypoint for the shared resolved-task worker."""
from src.app.api.execution import task_worker_main as main

if __name__ == "__main__":
    import sys
    sys.exit(main())
