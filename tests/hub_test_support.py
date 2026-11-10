"""Build explicit native archives for isolated Hub lifecycle tests."""

import hashlib
import json
from zipfile import ZipFile

from src.app.components.models import HubComponent, ReleaseInfo


def native_component(directory, package_id="example", *, kind="model", version="1.0.0",
                     configs=None, files=None, capabilities=None, dependencies=()):
    directory.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = {} if kind in ("suite", "experiment") else {"provider.py": "class Provider: pass\n"}
    if capabilities is None:
        capabilities = ([{"id": package_id, "kind": kind, "entry_point": "provider.py:Provider"}]
                        if "provider.py" in files else [])
    config_map = {dest: "configs/" + dest.removeprefix("in/config/") for dest in (configs or {})}
    meta = {
        "id": package_id, "name": package_id, "kind": kind, "version": version,
        "storage_version": 2, "api_version": 1, "capabilities": capabilities,
        "payload": {
            "implementations": [name for name in files if name.endswith(".py")],
            "resources": [name for name in files if not name.endswith(".py")],
            "configs": config_map,
        },
        "dependencies": {"components": list(dependencies), "pip": []},
    }
    archive = directory / f"{package_id}-{version}.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr(".theta_component.json", json.dumps(meta))
        for name, content in files.items():
            zf.writestr(name, content)
        for dest, content in (configs or {}).items():
            zf.writestr(config_map[dest], content)
    component = HubComponent.from_dict(meta)
    component.releases = {version: ReleaseInfo(version, archive.as_uri(), hashlib.sha256(archive.read_bytes()).hexdigest())}
    return component
