"""Recoverable, serialized filesystem lifecycle. Python dependency installs are separate."""
from __future__ import annotations

import functools
import json
import os
import shutil
import socket
import stat
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def remove(path):
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        for item in path.rglob("*"):
            if not item.is_symlink(): item.chmod(stat.S_IMODE(item.stat().st_mode) | stat.S_IWUSR)
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink(): path.unlink()


def alive(pid):
    try: os.kill(int(pid), 0); return True
    except ProcessLookupError: return False
    except PermissionError: return True


@contextmanager
def workspace_lock(paths, *, wait=False):
    # Always anchor the lock in the workspace even when state lives in a custom data directory.
    lock = paths.root / ".thetaide/lifecycle.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0); stream.write(b"0"); stream.flush(); stream.seek(0)
            try: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError("Another workspace lifecycle operation is active; retry after it finishes") from exc
        else:
            import fcntl
            try: fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
            except BlockingIOError as exc: raise RuntimeError("Another workspace lifecycle operation is active; retry after it finishes") from exc
        try:
            recover(paths)
            yield
        finally:
            if os.name == "nt":
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream, fcntl.LOCK_UN)


@functools.cache
def machine_id():
    """This computer's stable identity. The host name is not one: macOS renames the host when
    the network changes, which would make every lease written before look like another machine's."""
    try:
        if sys.platform == "darwin":
            import ctypes, ctypes.util
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            value = ctypes.create_string_buffer(16)
            wait = (ctypes.c_long * 2)(1, 0)  # struct timespec: give up after one second
            if libc.gethostuuid(value, wait) == 0:
                return str(uuid.UUID(bytes=value.raw))
        elif os.name == "nt":
            import winreg
            access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0, access) as key:
                return str(winreg.QueryValueEx(key, "MachineGuid")[0])
        else:
            for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                path = Path(candidate)
                if path.is_file() and path.read_text().strip():
                    return path.read_text().strip()
    except (OSError, AttributeError, ValueError):
        pass
    return "host:" + socket.gethostname()


def assert_idle(paths, package_id=None):
    for lease in (paths.root / ".thetaide/provider_leases").glob("*.json"):
        record = json.loads(lease.read_text())
        if record.get("machine") != machine_id() or alive(record["pid"]):
            if package_id is None or not record.get("packages") or package_id in record["packages"]:
                raise RuntimeError(f"Implementation is in use by {record.get('purpose', 'an active job')} (PID {record['pid']}). Stop it or wait, then retry.")
        else: lease.unlink(missing_ok=True)


@contextmanager
def provider_lease(paths, packages=(), purpose="active job"):
    if paths.launch_inputs_snapshot() is not None:
        # This process uses private copies, so live component lifecycle operations
        # cannot replace its inputs and need not wait for the job to finish.
        yield None
        return
    lease = paths.root / ".thetaide/provider_leases" / (uuid.uuid4().hex + ".json")
    with workspace_lock(paths, wait=True):
        write_json(lease, {"pid": os.getpid(), "machine": machine_id(), "packages": list(packages), "purpose": purpose})
    try: yield lease
    finally: lease.unlink(missing_ok=True)


class FileTransaction:
    def __init__(self, paths, operation, package_id):
        self.paths = paths
        self.directory = paths.root / ".thetaide/transactions" / uuid.uuid4().hex
        self.journal = self.directory / "journal.json"
        self.record = {"operation": operation, "package": package_id, "status": "preparing", "files": []}

    def protect(self, paths):
        self.directory.mkdir(parents=True, exist_ok=True)
        for target in dict.fromkeys(Path(p).resolve() for p in paths):
            if any(target.is_relative_to(Path(r["target"])) for r in self.record["files"]): continue
            backup = self.directory / str(len(self.record["files"]))
            exists = target.exists()
            if exists:
                if target.is_dir(): shutil.copytree(target, backup, symlinks=True)
                else: shutil.copy2(target, backup)
            self.record["files"].append({"target": str(target), "backup": str(backup), "existed": exists})
        self.record["status"] = "active"
        write_json(self.journal, self.record)

    def commit(self):
        self.record["status"] = "committed"
        write_json(self.journal, self.record)
        remove(self.directory)

    def rollback(self):
        restore(self.record)
        self.record["status"] = "rolled_back"
        write_json(self.journal, self.record)
        remove(self.directory)

    def __enter__(self): return self
    def __exit__(self, kind, value, tb):
        if kind: self.rollback()
        else: self.commit()


def restore(record):
    for item in reversed(record["files"]):
        target, backup = Path(item["target"]), Path(item["backup"])
        remove(target)
        if item["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if backup.is_dir(): shutil.copytree(backup, target, symlinks=True)
            else: shutil.copy2(backup, target)


def recover(paths):
    for journal in (paths.root / ".thetaide/transactions").glob("*/journal.json"):
        record = json.loads(journal.read_text())
        if record["status"] == "active": restore(record)
        remove(journal.parent)


@contextmanager
def implementation_edit(paths, target):
    """Serialize editor saves to authored implementation/resource files with job launch."""
    target = Path(target).resolve()
    roots = (paths.research, *paths.managed_roots, *paths.bundled_roots)
    if any(target.is_relative_to(root) for root in roots) and not target.is_relative_to(paths.results.resolve()):
        with workspace_lock(paths):
            assert_idle(paths)
            yield
    else:
        yield
