"""The application's own packaging identity, from its source tree or its installed distribution."""
from __future__ import annotations

from pathlib import Path

from src.app.workspace import PROJECT_ROOT


def application_project(application=PROJECT_ROOT, extras=()):
    """Name, version and requirements (with the named extras') of the application: its source
    tree's ``pyproject.toml``, or the installed distribution's metadata when it is installed."""
    pyproject = Path(application) / 'pyproject.toml'
    if pyproject.is_file():
        import tomllib
        project = tomllib.loads(pyproject.read_text()).get('project', {})
        optional = project.get('optional-dependencies', {})
        requirements = list(project.get('dependencies', [])) + [value for extra in extras for value in optional[extra]]
        return {'name': project.get('name'), 'version': project.get('version')}, requirements
    from importlib import metadata
    from packaging.requirements import Requirement
    names = metadata.packages_distributions().get('run_pipeline') or []
    if len(names) != 1:
        raise ValueError(f'The application at {application} is neither a source tree nor one installed distribution')
    distribution = metadata.distribution(names[0])
    requirements = []
    for value in distribution.requires or []:
        marker = Requirement(value).marker
        # Base requirements keep their own markers; an extra's are listed when that extra is asked for.
        if marker is None or 'extra' not in str(marker) or any(marker.evaluate({'extra': extra}) for extra in extras):
            requirements.append(value)
    return {'name': distribution.metadata['Name'], 'version': distribution.version}, requirements
