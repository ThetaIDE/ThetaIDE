"""Desktop SDK for app plugins and contributed settings/monitor widgets.

Importing an export loads Qt; headless code must not import this module.

Plugin lifecycle
    ``Plugin``, ``PluginManifest``, ``PluginContext`` (the host facade passed to
    ``Plugin.activate``; use its services instead of host window attributes)
Shared controls and appearance
    ``label``, ``ComboBox``, ``SpinBox``, ``DoubleSpinBox``, ``svg_icon``,
    ``theme_color``, ``current_theme``, ``style_tool_button``, ``refresh_tool_icons``
Shared tree toolbars (persist through ``PluginContext.tree_toolbar_store()``)
    ``TreeHeaderWidget``, ``DraggableToolButton``, ``TreeToolbarPersistence``
"""
from ._exports import lazy_exports

_EXPORTS = {
    "Plugin": "frontend.plugins.base:Plugin",
    "PluginManifest": "frontend.plugins.base:PluginManifest",
    "PluginContext": "frontend.plugins.context:PluginContext",
    "label": "frontend.widgets:label",
    "ComboBox": "frontend.widgets:ComboBox",
    "SpinBox": "frontend.widgets:SpinBox",
    "DoubleSpinBox": "frontend.widgets:DoubleSpinBox",
    "svg_icon": "frontend.sidetabs:svg_icon",
    "theme_color": "frontend.theme:theme_color",
    "current_theme": "frontend.theme:current_theme",
    "style_tool_button": "frontend.tree_tools:style_tool_button",
    "refresh_tool_icons": "frontend.tree_tools:refresh_tool_icons",
    "TreeHeaderWidget": "frontend.tree_tools:TreeHeaderWidget",
    "DraggableToolButton": "frontend.tree_tools:DraggableToolButton",
    "TreeToolbarPersistence": "frontend.tree_tools:TreeToolbarPersistence",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
