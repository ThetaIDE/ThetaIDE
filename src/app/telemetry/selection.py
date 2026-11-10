"""Which metrics a run records, set by the experiment:

    telemetry:
      metrics: [eval/reward, losses/]    # a name, or a section ending in "/" for all of its metrics

Unset (or empty): every metric is recorded, as before. Set: only the listed ones reach metrics.csv, the
live stream and TensorBoard, and work that only exists to produce unlisted metrics is skipped (see
`keeps_section`): no evaluation episodes without an eval/ metric, no system sampling without a system/
one. Counters the plots need as an x-axis are always recorded.

Plain Python: the GUI's experiment editor and validation import it, so it must not pull in Lightning.
"""

from __future__ import annotations

from typing import Any, Iterable

ALWAYS_KEPT = frozenset({"step", "epoch", "transitions", "training_time_seconds"})
# Sections the experiment editor offers even before a run has logged anything.
SECTIONS = ("eval", "train", "losses", "diagnostics", "val", "test", "time", "system")


class MetricSelection:
    """`patterns` None or empty: keep everything. Otherwise names, and sections ending in "/"."""

    def __init__(self, patterns: Iterable[str] | None = None):
        self.patterns = tuple(str(p) for p in patterns or ())

    @property
    def everything(self) -> bool:
        return not self.patterns

    def keeps(self, name: str) -> bool:
        if self.everything or name in ALWAYS_KEPT:
            return True
        return any(name.startswith(p) if p.endswith("/") else name == p for p in self.patterns)

    def keeps_section(self, section: str) -> bool:
        """Whether any metric of `section` (e.g. "eval") is kept, so the work producing it must run."""
        if self.everything:
            return True
        prefix = section.rstrip("/") + "/"
        return any(p.startswith(prefix) or (p.endswith("/") and prefix.startswith(p))
                   for p in self.patterns)

    def filter(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.everything:
            return metrics
        return {key: value for key, value in metrics.items() if self.keeps(str(key))}

    def __repr__(self) -> str:
        return f"MetricSelection({list(self.patterns) or 'everything'})"


def _section(cfg: Any):
    section = cfg.get("telemetry") if hasattr(cfg, "get") else None
    return section if hasattr(section, "get") else {}


def selection_from_config(cfg: Any) -> MetricSelection:
    patterns = _section(cfg).get("metrics")
    if isinstance(patterns, str):
        patterns = [patterns]
    return MetricSelection(patterns)


def selection_errors(cfg: Any, monitor_metric: str | None = None, is_sweep: bool = False) -> list[str]:
    """What is wrong with `telemetry.metrics`, if anything. A sweep needs the metric it optimises."""
    patterns = _section(cfg).get("metrics")
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)) or not all(isinstance(p, str) and p.strip() for p in patterns):
        return ["'telemetry.metrics' must be a list of metric names or sections (e.g. [eval/reward, losses/])."]
    selection = MetricSelection(patterns)
    hydra = cfg.get("hydra") if hasattr(cfg, "get") else None
    sweeping = is_sweep or (hasattr(hydra, "get") and hydra.get("sweeper")) or (hasattr(cfg, "get") and cfg.get("sweeper"))
    if sweeping and monitor_metric and not selection.keeps(monitor_metric):
        return [f"This experiment sweeps on '{monitor_metric}', so 'telemetry.metrics' must keep it."]
    return []
