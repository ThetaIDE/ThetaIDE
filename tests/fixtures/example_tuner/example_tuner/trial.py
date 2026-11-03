"""A separate deterministic objective process for the example tuner."""

import sys
from pathlib import Path


value = float(sys.argv[1])
Path(sys.argv[2]).write_text(str(value), encoding="utf-8")
print((value - 0.1) ** 2)
