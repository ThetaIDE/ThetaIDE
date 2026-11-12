"""Execute an explicit preprocessing resource, outside method training dispatch."""
import sys
from src.app.pipeline.provider_worker import run

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: resource_worker RESOURCE [ARGS...]')
    run(sys.argv[1], sys.argv[2:])
