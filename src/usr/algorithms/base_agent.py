"""Compatibility alias for framework-owned RL base classes."""
import sys
from src.app.core import base_agent as _canonical
sys.modules[__name__] = _canonical
