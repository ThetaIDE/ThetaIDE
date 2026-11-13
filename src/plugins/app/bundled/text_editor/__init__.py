"""Optional built-in text editor and default-editor provider."""

from __future__ import annotations

from frontend.plugins.base import Plugin


class TextEditorPlugin(Plugin):
    def __init__(self, manifest):
        super().__init__(manifest)
        self.context = None
        self.panel = None

    def activate(self, context):
        from .panel import TextEditorPanel

        self.context = context
        self.panel = TextEditorPanel(context, lazy=True)
        context.add_sidebar_tab("text_editor", self.panel, "Text Editor", "editor", "Editor")

        def open_file(path, line=None, column=None):
            context.select_sidebar_tab("text_editor")
            return self.panel.open_file(path, line=line, column=column)

        def open_folder(root, **kwargs):
            context.select_sidebar_tab("text_editor")
            return self.panel.open_folder(root, **kwargs)

        context.window.editor_service.register(self.manifest.id, "Theta Text Editor", open_file, open_folder, self.panel.can_close)
        # Explicit enabling selects the editor; restoring an enabled plugin at startup
        # preserves a user's subsequent choice of another default provider.
        if not context.window.plugin_manager.is_plugin_enabled(self.manifest.id):
            context.window.settings_manager.default_text_editor = self.manifest.id
        for command_id, title, callback in (
            ("text_editor.open", "View: Text Editor", lambda: context.select_sidebar_tab("text_editor")),
            ("text_editor.new", "Editor: New document", lambda: self._action("new")),
            ("text_editor.open_file", "Editor: Open file…", lambda: self._action("open")),
            ("text_editor.open_folder", "Editor: Open folder…", lambda: self._action("open_folder")),
            ("text_editor.save", "Editor: Save file", lambda: self._action("save")),
            ("text_editor.save_as", "Editor: Save file as…", lambda: self._action("save_as")),
            ("text_editor.find", "Editor: Find / Replace", lambda: self._action("find")),
        ):
            context.register_command(command_id, title, callback, icon="editor", keywords=("text", "code", "file"))

    def _action(self, action):
        self.context.select_sidebar_tab("text_editor")
        return self.panel.perform_action(action)

    def can_deactivate(self):
        return self.panel is None or self.panel.can_close()

    def deactivate(self):
        if self.context:
            self.context.window.editor_service.unregister(self.manifest.id)
        if self.panel:
            self.panel.cleanup()
            self.panel.deleteLater()
            self.panel = None
        if self.context:
            self.context.remove_sidebar_tab("text_editor")
            self.context = None
