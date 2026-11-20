"""Version 1 safe-boundary configuration protocol."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Mapping, Protocol


@dataclass(frozen=True)
class MutableParameter:
    minimum: float
    maximum: float | None = None
    exclusive_minimum: bool = False

    def validate(self, value):
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
            raise ValueError("Mutable parameters require finite numeric values")
        if value < self.minimum or (self.exclusive_minimum and value == self.minimum):
            raise ValueError(f"Value {value} is below the allowed range")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"Value {value} is above the allowed range")
        return float(value)


@dataclass(frozen=True)
class Boundary:
    step: int
    metrics: Mapping[str, float]
    parameters: Mapping[str, float]


class Controller(Protocol):
    api_version: int

    def update(self, boundary: Boundary) -> Mapping[str, float]: ...


def validate_changes(changes, parameters):
    if not isinstance(changes, Mapping):
        raise ValueError("Dynamic changes must be a parameter mapping")
    unknown = set(changes) - parameters.keys()
    if unknown:
        raise ValueError(f"Unsupported mutable parameters: {sorted(unknown)}")
    return {path: parameters[path].validate(value) for path, value in changes.items()}


class Replay:
    api_version = 1

    def __init__(self, entries, parameters, *, rollout, start, stop):
        self.changes = {}
        for entry in entries:
            step = entry["step"]
            if type(step) is not int or not start <= step < stop or step % rollout or step in self.changes:
                raise ValueError(f"Invalid or duplicate dynamic boundary {step}")
            self.changes[step] = validate_changes(entry["parameters"], parameters)

    def update(self, boundary):
        return self.changes.get(boundary.step, {})
