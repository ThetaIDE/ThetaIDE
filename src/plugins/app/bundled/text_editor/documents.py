"""Text buffers and conservative, atomic file I/O, independent of the renderer."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

MAX_FILE_BYTES = 10 * 1024 * 1024


class FileConflictError(OSError):
    """The file on disk no longer matches the version opened or last saved."""


def fingerprint(path: Path) -> str | None:
    try:
        # Stream the hash: an externally replaced file may be much larger than our limit.
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(128 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except FileNotFoundError:
        return None


@dataclass
class Document:
    text: str = ""
    path: Path | None = None
    name: str = "Untitled"
    id: str = field(default_factory=lambda: uuid4().hex)
    saved_text: str = ""
    encoding: str = "utf-8"
    newline: str = "\n"
    disk_fingerprint: str | None = None

    @property
    def dirty(self) -> bool:
        return self.text != self.saved_text

    @property
    def title(self) -> str:
        return self.path.name if self.path else self.name

    @classmethod
    def open(cls, path: Path) -> Document:
        path = path.expanduser().resolve()
        with path.open("rb") as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        if len(raw) > MAX_FILE_BYTES:
            raise ValueError("This file exceeds the editor's 10 MiB limit.")
        encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise ValueError("This file is not UTF-8. Convert it to UTF-8 before editing in Theta.") from exc
        if "\x00" in text:
            raise ValueError("This appears to be a binary file. Choose a UTF-8 text file.")
        newline = "\r\n" if "\r\n" in text else ("\r" if "\r" in text else "\n")
        # Mixed line endings cannot be round-tripped by CodeMirror without conversion.
        # Reject them explicitly rather than silently changing untouched lines on save.
        remainder = text.replace(newline, "")
        if "\r" in remainder or "\n" in remainder:
            raise ValueError("This file has mixed line endings. Normalize them before editing in Theta.")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return cls(
            text=text,
            saved_text=text,
            path=path,
            encoding=encoding,
            newline=newline,
            disk_fingerprint=hashlib.sha256(raw).hexdigest(),
        )

    def changed_on_disk(self) -> bool:
        return self.path is not None and fingerprint(self.path) != self.disk_fingerprint

    def save(self, path: Path | None = None, *, force: bool = False) -> None:
        target = path or self.path
        if target is None:
            raise ValueError("Choose a path before saving.")
        target = target.expanduser().resolve()
        from src.app.components.transactions import implementation_edit
        from src.app.workspace import WorkspacePaths
        with implementation_edit(WorkspacePaths.for_process(), target):
            current = fingerprint(target)
            expected = self.disk_fingerprint if target == self.path else None
            if not force and current != expected:
                raise FileConflictError("The file has changed on disk or already exists.")
            mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
            if mode is not None and not mode & 0o222:
                raise PermissionError(f"The file is read-only: {target}")
            raw = self.text.replace("\n", self.newline).encode(self.encoding)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
                    temporary = Path(stream.name)
                    if mode is not None:
                        os.chmod(temporary, mode)
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            self.path = target
            self.saved_text = self.text
            self.disk_fingerprint = hashlib.sha256(raw).hexdigest()
