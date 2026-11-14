"""Lifecycle for the host-rendered, capability-driven tuning monitor slot."""
from src.app.core.contributions import ContributionRegistry, ContributionScope
from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_discovery import provider_discovery_snapshot


def monitoring_config(run, lives, folder=None):
    """Use the monitored launch's configuration, never the open editor's recipe."""
    remote = (run.get('remote') or {}).get('config')
    if hasattr(remote, 'get') and remote:
        return remote
    for live in lives.values():
        config = getattr(live, 'config', None)
        if hasattr(config, 'get') and 'tuning' in config:
            return config
    config = run.get('config') or {}
    if hasattr(config, 'get') and 'tuning' in config:
        return config
    if folder is not None:
        from pathlib import Path
        import yaml
        try:
            saved = yaml.safe_load((Path(folder) / 'config.yaml').read_text(encoding='utf-8'))
            if isinstance(saved, dict):
                return saved
        except (OSError, ValueError, yaml.YAMLError):
            pass
    return config if hasattr(config, 'get') else {}


class TuningMonitorSlot:
    def __init__(self, panel):
        self.panel = panel
        self.scope = ContributionScope(self._mount)
        self.mounted = set()
        self.workspace = None

    def _mount(self, contribution):
        panel = self.panel
        if contribution.entry_point:
            raise ValueError('monitor.tuning is a host-rendered slot; omit its entry_point')
        if panel.sweep_view is None:
            from .sweep_view import SweepView
            view = SweepView()
            view.trial_selected.connect(panel._open_sweep_trial)
            panel.sweep_view = view
            panel._views['Sweep'] = view
            blocked = panel.views.blockSignals(True)
            try:
                panel.views.addWidget(view)
            finally:
                panel.views.blockSignals(blocked)
        self.mounted.add(contribution.key)

        def cleanup():
            self.mounted.discard(contribution.key)
            for button in (panel.method_button, panel.trial_button):
                button.menu().close()
                button.menu().clear()
            panel.trial_button.hide()
            if self.mounted:
                return
            view = panel.sweep_view
            if view is None:
                return
            panel.sweep_view = None
            panel._views.pop('Sweep', None)
            panel._focused_sweep = None
            blocked = panel.views.blockSignals(True)
            try:
                if panel.views.currentWidget() is view:
                    panel.views.setCurrentWidget(panel.charts_view)
                    for button in panel.view_buttons.values():
                        button.setChecked(False)
                panel.views.removeWidget(view)
            finally:
                panel.views.blockSignals(blocked)
            view.trial_selected.disconnect(panel._open_sweep_trial)
            view.set_studies([])
            view.set_lives({})
            view.close()
            view.setParent(None)
            view.deleteLater()
        return cleanup

    def reconcile(self, paths, config, studies):
        """Return supported studies and reasons for unavailable saved providers."""
        identity = (paths.root, paths.application, paths.data_dir)
        if self.workspace != identity:
            self.close()
            self.workspace = identity
        tuning = config.get('tuning') or {}
        selected = tuning.get('component') if hasattr(tuning, 'get') else None
        components = {study.get('component') or selected for _, study in studies}
        if selected:
            components.add(selected)
        components.discard(None)
        desired = []
        supported = set()
        reasons = []
        if None in {study.get('component') or selected for _, study in studies}:
            reasons.append('Saved tuning results do not identify their provider; restore tuning.component in the recipe')
        with provider_discovery_snapshot():
            registry = ContributionRegistry(paths)
            catalog = get_catalog(paths)
            contributions = registry.contributions('monitor.tuning', config=config)
            for name in sorted(components):
                owner = catalog.resolve('tuner', name)
                matches = [item for item in contributions if item.capability.kind == 'tuner'
                           and item.status.state == 'ready'
                           and owner == (item.package, item.capability)]
                if matches:
                    desired.extend(item for item in matches if item.key not in {d.key for d in desired})
                    supported.add(name)
                else:
                    status = registry.status('tuner', name, config=config)
                    reasons.append(status.reason or f'Tuner {name!r} does not contribute a tuning monitor')
        self.scope.reconcile(desired)
        return ([(method, study) for method, study in studies if (study.get('component') or selected) in supported],
                '; '.join(reasons))

    def close(self):
        self.scope.close()
