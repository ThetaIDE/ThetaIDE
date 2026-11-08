"""Compatibility alias for the framework-owned agent registry."""
import sys
from src.app.core import agent_registry as _canonical
sys.modules[__name__] = _canonical
