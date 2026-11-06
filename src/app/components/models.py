"""Data models for Theta Hub components, releases, and registry indices."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class AuthorInfo:
    name: str
    github: str | None = None
    url: str | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | str) -> AuthorInfo:
        if isinstance(data, str):
            return cls(name=data)
        return cls(
            name=data.get("name", "Unknown"),
            github=data.get("github"),
            url=data.get("url"),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name}
        if self.github:
            d["github"] = self.github
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class ReleaseInfo:
    tag: str
    url: str
    sha256: str
    published_at: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReleaseInfo:
        return cls(
            tag=data.get("tag", ""),
            url=data.get("url", ""),
            sha256=data.get("sha256", ""),
            published_at=data.get("published_at"),
            size_bytes=data.get("size_bytes"),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "tag": self.tag,
            "url": self.url,
            "sha256": self.sha256,
        }
        if self.published_at:
            d["published_at"] = self.published_at
        if self.size_bytes is not None:
            d["size_bytes"] = self.size_bytes
        return d


@dataclass
class HubComponent:
    """Represents a component available on the Theta Hub (plugin, method, model, etc.)."""
    id: str
    name: str
    kind: str  # "plugin", "method", "model", "env", "experiment"
    version: str
    description: str = ""
    author: AuthorInfo = field(default_factory=lambda: AuthorInfo(name="Unknown"))
    repository: str | None = None
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    releases: Dict[str, ReleaseInfo] = field(default_factory=dict)
    target_path: str | None = None
    compatibility: Dict[str, str] = field(default_factory=dict)
    icon: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Local installation state (populated by installer)
    is_installed: bool = False
    installed_version: str | None = None

    @property
    def has_update(self) -> bool:
        if not self.is_installed or not self.installed_version:
            return False
        return self.installed_version != self.version

    @property
    def latest_release(self) -> ReleaseInfo | None:
        if self.version in self.releases:
            return self.releases[self.version]
        if self.releases:
            # Return release with latest key
            latest_k = sorted(self.releases.keys())[-1]
            return self.releases[latest_k]
        return None

    @property
    def suite(self) -> str | None:
        value = self.extra.get("suite")
        return value if isinstance(value, str) and value else None

    @property
    def provides(self) -> List[str]:
        """Identifiers supplied by this package in addition to its own ID."""
        values = self.extra.get("provides", [])
        return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []

    @property
    def aliases(self) -> List[str]:
        """Legacy or alternate identifiers accepted by the package."""
        values = self.extra.get("aliases", [])
        return [value for value in values if isinstance(value, str)] if isinstance(values, list) else []

    @property
    def hidden(self) -> bool:
        return bool(self.extra.get("hidden", False))

    @property
    def is_grouped_component(self) -> bool:
        """Whether this component is supplied by another Hub package."""
        return self.suite is not None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HubComponent:
        author_raw = data.get("author", "Unknown")
        author = AuthorInfo.from_dict(author_raw)

        releases_raw = data.get("releases", {})
        releases = {}
        for ver, rdata in releases_raw.items():
            releases[ver] = ReleaseInfo.from_dict(rdata)

        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            kind=data.get("kind", "plugin"),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=author,
            repository=data.get("repository"),
            license=data.get("license", "MIT"),
            tags=list(data.get("tags", [])),
            releases=releases,
            target_path=data.get("target_path"),
            compatibility=dict(data.get("compatibility", {})),
            icon=data.get("icon"),
            extra=data,
        )

    def to_dict(self) -> Dict[str, Any]:
        # Preserve Hub-specific package metadata (suite, provides, aliases, etc.)
        # through a fetch/cache/serialize cycle.
        data = dict(self.extra)
        data.update({
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "author": self.author.to_dict(),
            "repository": self.repository,
            "license": self.license,
            "tags": self.tags,
            "releases": {v: r.to_dict() for v, r in self.releases.items()},
            "target_path": self.target_path,
            "compatibility": self.compatibility,
            "icon": self.icon,
        })
        return data
