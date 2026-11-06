"""Compatibility alias for the shared headless component installer."""
import sys
from src.app.components import installer as _implementation

sys.modules[__name__] = _implementation
