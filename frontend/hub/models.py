"""Compatibility alias for the shared headless component models."""
import sys
from src.app.components import models as _implementation

sys.modules[__name__] = _implementation
