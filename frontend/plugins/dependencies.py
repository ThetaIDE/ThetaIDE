"""Compatibility alias for the shared headless component dependencies."""
import sys
from src.app.components import dependencies as _implementation

sys.modules[__name__] = _implementation
