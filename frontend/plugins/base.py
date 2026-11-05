"""Base classes and interfaces for the ThetaIDE plugin system."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from .context import PluginContext


@dataclass
class PluginManifest:
    """Metadata specification for a plugin."""
    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = "ThetaIDE Team"
    default_enabled: bool = False
    icon: Optional[str] = None
    entry_point: str = "Plugin"
    plugin_dir: Optional[Path] = None
    is_core: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class Plugin(ABC):
    """Abstract base class that all ThetaIDE plugins must inherit from."""

    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest

    @abstractmethod
    def activate(self, context: PluginContext) -> None:
        """Invoked when the plugin is activated by the PluginManager.
        
        Use the provided context to contribute UI components, tabs,
        menu actions, or listen to IDE events.
        """
        pass

    @abstractmethod
    def deactivate(self) -> None:
        """Invoked when the plugin is deactivated or IDE is shutting down.
        
        Clean up all registered widgets, resources, and event listeners.
        """
        pass

    def can_deactivate(self) -> bool:
        """Allow a plugin with unsaved work to cancel disabling or app shutdown."""
        return True

    def get_settings_widget(self, context: PluginContext):
        """Return a custom QWidget for configuring this plugin's settings.
        
        Return None if this plugin has no configurable settings.
        """
        return None
