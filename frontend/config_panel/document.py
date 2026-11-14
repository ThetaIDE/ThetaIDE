"""Qt-independent typed edits. Reading an inherited value never mutates YAML.

The caller supplies domain normalization and persistence; this adapter deliberately
does not promise source-format/comment round trips.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

import yaml
from jsonschema import Draft202012Validator


class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()
Path = tuple[str, ...]


def lookup(data, path):
    for key in path:
        if not isinstance(data, Mapping) or key not in data:
            return MISSING
        data = data[key]
    return data


class ChangeKind(Enum):
    VALUE = "value"
    STRUCTURE = "structure"
    DEPENDENCY = "dependency"


@dataclass(frozen=True)
class EditResult:
    path: Path
    changed: bool = False
    kind: ChangeKind = ChangeKind.VALUE
    error: str | None = None


def validation_error(value, schema):
    """Validate drafts, while allowing already-loaded invalid values to be read."""
    if (schema or {}).get("type") == "integer" and type(value) is not int:
        return "Enter a whole number"
    errors = list(Draft202012Validator(schema or {}).iter_errors(value))
    return errors[0].message if errors else None


def same_value(left, right):
    """Python equates False and 0, including inside containers; YAML must not."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_value(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return left == right


class ConfigDocument:
    def __init__(
        self,
        explicit: dict,
        inherited: Mapping | None = None,
        *,
        read_only: bool = False,
        on_change: Callable[[EditResult], None] | None = None,
        selector_identity: Callable[[Path, Any], dict | None] | None = None,
    ):
        self.explicit = explicit
        self.inherited = inherited or {}
        self.read_only = read_only
        self.on_change = on_change
        self.selector_identity = selector_identity or (lambda path, value: None)

    def get(self, path, *, effective=True):
        value = lookup(self.explicit, path)
        if value is not MISSING or not effective:
            return value
        # An explicit scalar/null ancestor masks inherited children.
        for depth in range(1, len(path)):
            ancestor = lookup(self.explicit, path[:depth])
            if ancestor is not MISSING and not isinstance(ancestor, Mapping):
                return MISSING
        return lookup(self.inherited, path)

    def _notify(self, result):
        if result.changed and self.on_change:
            self.on_change(result)
        return result

    def set(self, path: Path, value: Any, *, schema=None, kind=None):
        if self.read_only:
            return EditResult(path, error="Document is read-only")
        if not path:
            return EditResult(path, error="A parameter path is required")
        error = validation_error(value, schema)
        if error:
            return EditResult(path, error=error)
        previous = self.get(path, effective=False)
        if same_value(previous, value):
            return EditResult(path)
        target = self.explicit
        for depth, key in enumerate(path[:-1], 1):
            if not isinstance(target.get(key), dict):
                current = self.get(path[:depth])
                identity = self.selector_identity(path[:depth], current)
                target[key] = deepcopy(identity) if identity is not None else {}
            target = target[key]
        target[path[-1]] = deepcopy(value)
        structural = previous is MISSING or isinstance(previous, (dict, list)) or isinstance(value, (dict, list))
        return self._notify(EditResult(path, True, kind or (ChangeKind.STRUCTURE if structural else ChangeKind.VALUE)))

    def add(self, path, value, **kwargs):
        if self.get(path, effective=False) is not MISSING:
            return EditResult(path, error="Parameter is already configured")
        return self.set(path, value, **kwargs)

    def remove(self, path: Path, *, required=False):
        if self.read_only or required:
            return EditResult(path, error="Parameter cannot be removed")
        if not path or self.get(path, effective=False) is MISSING:
            return EditResult(path)
        target = lookup(self.explicit, path[:-1])
        del target[path[-1]]
        return self._notify(EditResult(path, True, ChangeKind.STRUCTURE))

    def save(self, write, *, normalize=(), preamble=""):
        """Preserve the established cleanup order and leading directives."""
        if self.read_only:
            return False
        import yaml

        for operation in normalize:
            operation()
        write(preamble + yaml.safe_dump(self.explicit, sort_keys=False))
        return True


def _parse_yaml_value(text: str):
    """Convert text representation to typed YAML scalar, list, or dict."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() in ("null", "none", "~", ""):
        return None
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            return yaml.safe_load(stripped)
        except Exception:
            pass
    return stripped


def parse_list_field(text):
    """Comma-separated text back to a YAML list, keeping each item's type.

    `plots` holds names while `eval_protocol.horizons` holds numbers, so items
    go through the YAML scalar parser instead of all becoming strings.
    """
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(yaml.safe_load(part))
        except yaml.YAMLError:
            items.append(part)
    return items


def _leading_directives(text):
    """The comment block a config opens with, kept verbatim across a save.

    "# @package _global_" is a Hydra directive, not decoration: without it the
    file's keys are not applied at the global package level, so methods and the
    rest simply are not seen. Every one of the experiment configs starts with
    it, and yaml.safe_dump drops comments, so it has to be re-attached by hand.
    """
    kept = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        kept.append(line)
    return "".join(kept)
