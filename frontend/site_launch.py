"""Project upload and SSH handoff in the existing terminal pane."""
import sys

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QMessageBox, QPushButton, QWidget
from src.app.workspace import WorkspacePaths
from src.app.sites import transfer


def launch(window):
    from src.app.sites import site_name
    if not getattr(window, 'compose_valid', False):
        QMessageBox.warning(window, 'Site launch', 'Resolve experiment validation errors before launching.')
        return
    try:
        paths = window.workspace_paths
        site = site_name(window.composed_config)
        experiment = window.current_experiment
        viewer = getattr(window, 'config_viewer', None)
        if viewer and viewer.is_dirty and not viewer.save_to_disk(show_error=True):
            return
        from src.app.core.contributions import ContributionRegistry
        if not ContributionRegistry(paths).contributions('commands.site', config=window.composed_config):
            raise ValueError('The selected site provider is unavailable, disabled or needs configuration')
        config = transfer.connection(paths.root, site)
        command = [sys.executable, '-u', '-m', 'src.app.sites', 'launch', '--root', str(paths.root),
                   site, '--experiment', experiment]
        from src.app.sites.scheduler_api import get_scheduler
        provider = get_scheduler(cfg=window.composed_config, paths=paths)
        QApplication.clipboard().setText(provider.submission_command(config, experiment, site))
        terminal = window.terminal_panel.new_shell(command=command, cwd=str(paths.root))
        window.tabs.setCurrentWidget(window.terminal_panel)
        terminal.focus_terminal()
    except Exception as exc:
        QMessageBox.warning(window, 'Cluster terminal', str(exc))


def SiteSetup(parent=None, *, workspace=None):
    """API-1 constructor name for site setup: the host's `settings.site` contribution slot.

    Installed, enabled scheduler components mount their own setup widgets here; with none
    installed the slot is empty. No scheduler is selected by ID.
    """
    from .extension_settings import ExtensionSettings
    return ExtensionSettings('settings.site', parent, workspace=workspace)
