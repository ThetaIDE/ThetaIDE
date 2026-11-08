"""Theta Hub client for fetching community registries, searching, and managing installations."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

from .installer import HubInstaller
from .models import AuthorInfo, HubComponent, ReleaseInfo

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/ThetaIDE/ThetaHub/main/dist/index.json"

class HubClient(QObject):
    """Client for browsing, searching, and installing community hub items."""

    indexLoaded = pyqtSignal(list)
    fetchFailed = pyqtSignal(str)
    installProgress = pyqtSignal(str, int)  # (component_id, percentage)
    installFinished = pyqtSignal(str, bool, str)  # (component_id, success, message)
    uninstallFinished = pyqtSignal(str, bool, str)  # (component_id, success, message)
    componentChanged = pyqtSignal(str, str)  # (component_id, action: "install" | "uninstall")
    _indexFetched = pyqtSignal(object, str)
    _operationDone = pyqtSignal(str, str, bool, str, object)
    _installerChanged = pyqtSignal(str, str)

    def __init__(
        self,
        workspace_dir: Path,
        data_dir: Path,
        registry_url: str = DEFAULT_REGISTRY_URL,
        on_change_callback=None,
        settings_manager=None,
    ):
        super().__init__()
        self._operation_lock = threading.RLock()
        self._worker_state = threading.local()
        self._busy = set()
        self._fetching = False
        self._refresh_pending = False
        self._indexFetched.connect(self._accept_index, Qt.ConnectionType.QueuedConnection)
        self._operationDone.connect(self._finish_operation, Qt.ConnectionType.QueuedConnection)
        self._installerChanged.connect(self._handle_installer_change, Qt.ConnectionType.QueuedConnection)
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.registry_url = registry_url
        self.settings_manager = settings_manager

        self.cache_dir = self.data_dir / "cache" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "index.json"
        self.meta_file = self.cache_dir / "meta.json"

        self.installer = HubInstaller(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
            on_change_callback=self._record_installer_change,
            settings_manager=self.settings_manager,
            lookup_component_fn=self.get_component,
        )
        self.installer.protect_installed_components()
        if on_change_callback:
            self.componentChanged.connect(on_change_callback)

        from src.app.services.local_components import LocalComponents
        self.local_components = LocalComponents(self.installer.paths)
        for component_id, record in list(self.local_components.state()["records"].items()):
            if record["source"] != "local":
                self.local_components.discard_hub_snapshot(component_id)
        self.components: List[HubComponent] = self.local_components.entries()
        self.refresh_installed_status()

    def _record_installer_change(self, component_id, action):
        events = getattr(self._worker_state, "events", None)
        if events is not None:
            events.append((component_id, action))
        elif QThread.currentThread() == self.thread():
            self._handle_installer_change(component_id, action)
        else:
            self._safe_emit(self._installerChanged, component_id, action)

    @staticmethod
    def _safe_emit(signal, *args):
        try:
            signal.emit(*args)
        except RuntimeError:
            pass  # The owning window/client was destroyed while work finished.

    @pyqtSlot(str, str)
    def _handle_installer_change(self, component_id: str, action: str):
        try:
            self.refresh_local_components()
        except Exception as exc:
            self.fetchFailed.emit(str(exc))
            return
        self.componentChanged.emit(component_id, action)
        self.indexLoaded.emit(self.components)

    @pyqtSlot(object, str)
    def _accept_index(self, components, error):
        self._fetching = False
        if error:
            self.fetchFailed.emit(error)
        else:
            self.components = components
            try:
                self.refresh_local_components()
            except Exception as exc:
                self.fetchFailed.emit(str(exc))
            else:
                self.indexLoaded.emit(self.components)
        if self._refresh_pending:
            self._refresh_pending = False
            self.fetch_index_async(force=True)

    @pyqtSlot(str, str, bool, str, object)
    def _finish_operation(self, component_id, action, success, message, events):
        self._busy.discard(component_id)
        try:
            self.refresh_local_components()
        except Exception as exc:
            success, message = False, f"Could not refresh installed status: {exc}"
        for changed_id, changed_action in dict.fromkeys(events):
            self.componentChanged.emit(changed_id, changed_action)
        # Rebuild only once, after the complete dependency transaction finishes.
        self.indexLoaded.emit(self.components)
        signal = self.installFinished if action == "install" else self.uninstallFinished
        signal.emit(component_id, success, message)

    def refresh_installed_status(self) -> None:
        """Update each loaded component with its local installation state."""
        for comp in self.components:
            is_installed, ver = self.installer.check_installed(comp)
            comp.is_installed = is_installed
            comp.installed_version = ver

    def refresh_local_components(self):
        """Merge private and retained records without replacing remote listings."""
        records = self.local_components.state()["records"]
        by_id = {comp.id: comp for comp in self.components
                 if comp.extra.get("local_source") != "local" or comp.id in records}
        for local in self.local_components.entries():
            if local.id not in by_id:
                by_id[local.id] = local
            elif local.extra.get("local_source") == "local":
                current = by_id[local.id]
                current.extra.update(local.extra)
                current.name, current.version, current.target_path = local.name, local.version, local.target_path
        for component in by_id.values():
            if component.extra.get("local_source") != "local":
                component.extra.pop("retained_source", None)
        self.components = list(by_id.values())
        self.refresh_installed_status()

    def fetch_index_sync(self, force: bool = False, *, _publish=True) -> List[HubComponent]:
        """Fetch index synchronously using HTTP conditional caching (ETag)."""
        headers = {"User-Agent": "ThetaIDE-HubClient/1.0"}
        
        if force:
            headers["Cache-Control"] = "no-cache"
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.meta_file.exists():
                self.meta_file.unlink()

        etag = None

        if not force and self.meta_file.exists():
            try:
                meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
                etag = meta.get("etag")
                if etag:
                    headers["If-None-Match"] = etag
            except Exception:
                pass

        data = None

        if self.registry_url.startswith("file://"):
            local_file = Path(self.registry_url[7:])
            if local_file.exists():
                data = json.loads(local_file.read_text(encoding="utf-8"))
        else:
            candidates = [self.registry_url]
            for alt in [
                "https://thetaide.github.io/ThetaHub/index.json",
                "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/dist/index.json",
                "https://cameronegb.github.io/theta-hub/index.json",
                "https://cdn.jsdelivr.net/gh/CameronEgb/theta-hub@main/dist/index.json",
            ]:
                if alt not in candidates:
                    candidates.append(alt)

            for target_url in candidates:
                try:
                    if target_url.startswith("file://"):
                        local_p = Path(target_url[7:])
                        if local_p.exists():
                            raw_body = local_p.read_text(encoding="utf-8")
                            data = json.loads(raw_body)
                            self.cache_file.write_text(raw_body, encoding="utf-8")
                            break
                        continue

                    req = urllib.request.Request(target_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=8) as response:
                        raw_body = response.read().decode("utf-8")
                        data = json.loads(raw_body)

                        # Update local cache and etag
                        self.cache_file.write_text(raw_body, encoding="utf-8")
                        new_etag = response.headers.get("ETag")
                        if new_etag:
                            self.meta_file.write_text(
                                json.dumps({"etag": new_etag}, indent=2) + "\n",
                                encoding="utf-8",
                            )
                        break
                except urllib.error.HTTPError as http_err:
                    if http_err.code == 304 and self.cache_file.exists():
                        # Cache is fresh
                        data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                        break
                except Exception:
                    continue

        # Fallback to cached copy if network request failed
        if data is None and self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                data = None

        if data is None:
            data = {"components": []}

        raw_list = data if isinstance(data, list) else data.get("components", [])
        components = [HubComponent.from_dict(item) for item in raw_list]
        if _publish:
            self.components = components
            self.refresh_local_components()
            return self.components
        return components

    def fetch_index_async(self, force: bool = False) -> None:
        """Fetch registry asynchronously in a background thread."""
        if self._fetching:
            self._refresh_pending |= force
            return
        self._fetching = True
        def worker():
            try:
                comps = self.fetch_index_sync(force=force, _publish=False)
                self._safe_emit(self._indexFetched, comps, "")
            except Exception as exc:
                self._safe_emit(self._indexFetched, None, str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    @property
    def registry(self) -> Dict[str, HubComponent]:
        """Dictionary lookup of components by ID."""
        return {comp.id: comp for comp in self.components}

    @registry.setter
    def registry(self, mapping: Dict[str, HubComponent]) -> None:
        self.components = list(mapping.values())

    def get_component(self, component_id: str) -> Optional[HubComponent]:
        """Find a component by its unique ID, target path, or alias."""
        cid = component_id.lower()
        cid_norm = cid.replace("-", "_")

        # Resolve an exact identifier across the full registry before allowing
        # prefix matches. Otherwise `robotics` can incorrectly resolve to the
        # earlier `robotics-mujoco` entry and its experiment configs never install.
        for comp in self.components:
            comp_id = comp.id.lower()
            if comp_id == cid or comp_id.replace("-", "_") == cid_norm:
                return comp

        for comp in self.components:
            comp_id = comp.id.lower()
            if comp.target_path:
                tname = Path(comp.target_path).name.lower()
                if tname == cid or tname.replace("-", "_") == cid_norm:
                    return comp
            if comp.id.startswith(cid) or cid.startswith(comp.id):
                return comp

        # Packages may provide several component identifiers from one archive.
        # This also gives older dependency declarations a metadata-backed upgrade
        # path without teaching the client about package names.
        for comp in self.components:
            package_names = [comp.id, *comp.provides, *comp.aliases]
            if any(name.lower().replace("-", "_") == cid_norm for name in package_names):
                return comp

        return None

    def search(
        self,
        query: str = "",
        kind: Optional[str] = None,
        tag: Optional[str] = None,
        include_internal: bool = False,
        show_hidden: bool = False,
        local_only: bool = False,
    ) -> List[HubComponent]:
        """Filter components by search query, component kind, or tag."""
        q = query.strip().lower()
        results = []

        for comp in self.components:
            if local_only and comp.extra.get("local_source") != "local":
                continue
            if not show_hidden and self.local_components.hidden(comp.id):
                continue
            # Skip hidden and package-internal subcomponents from browser search.
            if not include_internal and (comp.is_grouped_component or comp.hidden):
                continue

            # Kind filter
            if kind and kind.lower() != "all":
                k = kind.lower()
                ck = comp.kind.lower()
                if k in ("suite", "suites", "package") and ck in ("suite", "package"):
                    pass
                elif k in ("method", "methods", "agent", "algorithm", "algorithms") and ck in ("method", "agent", "algorithm"):
                    pass
                elif k in ("data", "datamodule", "datamodules") and ck in ("data", "datamodule"):
                    pass
                elif k in ("eval", "evaluator", "evaluators") and ck in ("eval", "evaluator"):
                    pass
                elif k in ("plot", "plotter", "plotters") and ck in ("plot", "plotter"):
                    pass
                elif ck != k:
                    continue

            # Tag filter
            if tag and tag.lower() not in [t.lower() for t in comp.tags]:
                continue

            # Query filter (matches name, description, tags, author, or id)
            if q:
                match_id = q in comp.id.lower()
                match_name = q in comp.name.lower()
                match_desc = q in comp.description.lower()
                match_author = q in comp.author.name.lower()
                match_tags = any(q in t.lower() for t in comp.tags)
                if not (match_id or match_name or match_desc or match_author or match_tags):
                    continue

            results.append(comp)

        return results

    def install_async(self, component: HubComponent, version: Optional[str] = None) -> None:
        self._start_operation(component, "install", version)

    def uninstall_async(self, component: HubComponent) -> None:
        self._start_operation(component, "uninstall")

    def _start_operation(self, component, action, version=None):
        if component.id in self._busy:
            return
        self._busy.add(component.id)

        def progress_cb(downloaded, total):
            if total > 0:
                self._safe_emit(self.installProgress, component.id, min(95, int(downloaded / total * 95)))

        def worker():
            success, message = False, ""
            events = []
            with self._operation_lock:
                self._worker_state.events = events
                try:
                    if action == "install":
                        record = self.local_components.state()["records"].get(component.id)
                        if record and record['source'] == 'local' and not component.is_installed:
                            result = self.local_components.install(component, self.installer)
                        else:
                            result = self.installer.install(component=component, version=version, progress_callback=progress_cb)
                            if result is not False:
                                self.local_components.discard_hub_snapshot(component.id)
                    else:
                        result = self.installer.uninstall(component)
                    if result is False:
                        raise RuntimeError(f"The installer could not {action} this component.")
                    success = True
                    message = "Installation complete." if action == "install" else "Uninstalled."
                except Exception as exc:
                    message = str(exc)
                finally:
                    del self._worker_state.events
            self._safe_emit(self._operationDone, component.id, action, success, message, events)

        threading.Thread(target=worker, daemon=True).start()
