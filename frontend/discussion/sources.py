"""Nested menus for selecting the metric logs behind a Discussion figure."""

from collections import defaultdict

from src.app.core.plotting.metric_sources import metric_files
from src.app.pipeline.result_paths import iter_run_dirs, result_version_order


def source_label(source):
    return " / ".join(source[key] for key in ("target", "method", "version") if key in source)


def populate_sources(menu, paths, selected, choose):
    root = paths.results / "logs"
    branches = {(): menu}
    try:
        directories = list(iter_run_dirs(root))
    except OSError:
        menu.addAction("Could not read saved results").setEnabled(False)
        return
    for directory in directories:
        parts = directory.relative_to(root).parts
        for depth, name in enumerate(parts, 1):
            prefix = parts[:depth]
            if prefix not in branches:
                branches[prefix] = branches[prefix[:-1]].addMenu(name.replace("&", "&&"))
        branch = branches[parts]
        target = "/".join(parts)
        # Leave the folder hierarchy in place and refresh only this result's actions.
        branch.aboutToShow.connect(
            lambda b=branch, t=target: _populate_result(b, paths, t, selected, choose)
        )
    if menu.isEmpty():
        menu.addAction("No saved results").setEnabled(False)


def _add_source(menu, label, source, selected, choose):
    action = menu.addAction(label.replace("&", "&&"))
    action.setProperty("figureSource", True)
    action.setEnabled(source not in selected)
    action.triggered.connect(lambda _: choose(source))


def _populate_result(menu, paths, target, selected, choose):
    for action in menu.actions():
        if action.property("figureSource"):
            menu.removeAction(action)
            submenu = action.menu()
            (submenu if submenu is not None else action).deleteLater()
    methods = defaultdict(list)
    try:
        entries = metric_files(paths, target)
    except OSError:
        entries = []
    if not entries:
        action = menu.addAction("No accessible metric logs")
        action.setProperty("figureSource", True)
        action.setEnabled(False)
        return
    for entry in entries:
        methods[entry.method].append(entry.version)
    legacy_seeds = {}
    if any(version.startswith("version_") for versions in methods.values() for version in versions):
        from src.app.services.catalog import run_version_details
        group, _, experiment = target.partition("/")
        legacy_seeds = run_version_details(group, experiment, workspace=paths)
    _add_source(menu, "All methods in this result", {"target": target}, selected, choose)
    for method, versions in methods.items():
        source = {"target": target, "method": method}
        if versions == ["unversioned"]:
            _add_source(menu, method, source, selected, choose)
            continue
        branch = menu.addMenu(method.replace("&", "&&"))
        branch.menuAction().setProperty("figureSource", True)
        _add_source(branch, "All runs of this method", source, selected, choose)
        trials = {}
        trial_names = {version.split("/", 1)[0] for version in versions if "/" in version}
        for version in sorted(versions, key=result_version_order):
            trial, sep, seed = version.partition("/")
            parent = branch
            if trial in trial_names:
                if trial not in trials:
                    trials[trial] = branch.addMenu(trial)
                parent = trials[trial]
            seed_number = legacy_seeds.get((method, version), {}).get("seed")
            if seed_number is not None:
                label = f"Seed {seed_number} ({version})"
            elif trial in trial_names:
                label = seed if sep else "Trial summary"
            else:
                label = "Unversioned log" if version == "unversioned" else version
            _add_source(parent, label, {**source, "version": version}, selected, choose)
