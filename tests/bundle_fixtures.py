"""Bundles of small fake projects whose own root is the application.

Site tests exercise transport, receipts and allocation workers with a fake
``run_pipeline.py``. These helpers bundle such a project without composing an
experiment: the selection is every package the project has, under the given
configuration. Real composition and selection are covered in ``test_bundles``.
"""
import tempfile


def fake_selection(root, experiment, cfg, overrides=()):
    from src.app.core.provider_catalog import get_catalog
    from src.app.sites.bundle import Selection
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(root, application=root)
    packages = [package for package in get_catalog(paths).packages
                if package.origin not in ('bundled', 'python_distribution')
                and package.root.resolve().is_relative_to(paths.root)]
    return Selection(paths, experiment, list(overrides), cfg, packages, [])


def fake_bundles(monkeypatch, cfg):
    """Push and shared-workspace submission bundle the fake project instead of composing it."""
    from src.app.sites import bundle
    monkeypatch.setattr(bundle, 'select', lambda root, experiment, overrides=(), *, paths=None:
                        fake_selection(root, experiment, cfg, overrides))
    monkeypatch.setattr(bundle, 'check', lambda directory, manifest: None)


def activate_fake_bundle(root, experiment, cfg):
    """What a site push leaves in a remote workspace: the activated bundle and its archive."""
    from src.app.sites import bundle, transfer
    with tempfile.TemporaryDirectory(prefix='theta-test-bundle-') as stage:
        bundle.build(fake_selection(root, experiment, cfg), stage)
        transfer.activate(root, stage)
    return bundle.current(root)
