"""Storage identity and capabilities, independent of row geometry."""

from dataclasses import dataclass, field
from typing import Any, Callable

from .document import MISSING, ConfigDocument, Path


@dataclass
class ParameterBinding:
    document: ConfigDocument
    path: Path
    schema: dict = field(default_factory=dict)
    required: bool = False
    schema_owner: str | None = None
    schema_path: Path = ()
    add_child: Callable | None = None
    select: Callable | None = None
    open: Callable | None = None

    @property
    def identity(self):
        return self.path

    @property
    def explicit(self):
        return self.document.get(self.path, effective=False)

    @property
    def value(self):
        return self.document.get(self.path)

    @property
    def inherited(self):
        return self.explicit is MISSING and self.value is not MISSING

    @property
    def can_remove(self):
        return not self.required and self.explicit is not MISSING and not self.document.read_only

    def set(self, value: Any):
        return self.document.set(self.path, value, schema=self.schema)

    def remove(self):
        return self.document.remove(self.path, required=self.required)


def component_identity(path, value):
    """Promote a scalar selector without losing the selected component."""
    if path and path[-1] in ("env", "site", "data_module", "symbolic_state_loader", "evaluator", "model", "agent"):
        if isinstance(value, str):
            return {"name": value}
    return None


def method_parameter_path(data, method_name, kind, path=()):
    """Resolve legacy flat, named and component-keyed method storage explicitly."""
    from ..config_model import COMPONENT_ID_KEYS, component_selection

    root = ("methods", method_name)
    spec = data.get("methods", {}).get(method_name, {}).get(kind)
    if isinstance(spec, dict):
        if any(isinstance(spec.get(key), str) for key in COMPONENT_ID_KEYS):
            root += (kind,)
        else:
            name, _ = component_selection(spec)
            if name:
                root += (kind, name)
    return root + tuple(path)
