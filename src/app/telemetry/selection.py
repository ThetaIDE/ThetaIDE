"""Which metrics a run records, set by the experiment:

    telemetry:
      metrics: [eval/reward, losses/]    # a name, or a section ending in "/" for all of its metrics

Unset (or empty): every metric is recorded, as before. Set: only the listed ones reach metrics.csv, the
live stream and run loggers, and work that only exists to produce unlisted metrics is skipped (see
`keeps_section`): no evaluation episodes without an eval/ metric, no system sampling without a system/
one. Counters the plots need as an x-axis are always recorded.

An eval frequency of 0 (`eval_freq_steps: 0`) turns evaluation off: the eval/ section is left out of
every selection, so no evaluation episodes run and no eval/ metric is recorded or streamed.

Plain Python: the GUI's experiment editor and validation import it, so it must not pull in Lightning.
"""

from __future__ import annotations

from typing import Any, Iterable

ALWAYS_KEPT = frozenset({"step", "epoch", "transitions", "training_time_seconds"})
# Sections the experiment editor offers even before a run has logged anything.
SECTIONS = ("eval", "train", "losses", "diagnostics", "val", "test", "time", "system")


class MetricSelection:
    """`patterns` None or empty: keep everything. Otherwise names, and sections ending in "/".

    `without` names sections that are never kept, whatever `patterns` says (eval when evaluation is off)."""

    def __init__(self, patterns: Iterable[str] | None = None, *, without: Iterable[str] = ()):
        self.patterns = tuple(str(p) for p in patterns or ())
        self.without = tuple(str(section).rstrip("/") + "/" for section in without)

    @property
    def everything(self) -> bool:
        return not self.patterns and not self.without

    def keeps(self, name: str) -> bool:
        if name in ALWAYS_KEPT:
            return True
        if any(name.startswith(section) for section in self.without):
            return False
        if not self.patterns:
            return True
        return any(name.startswith(p) if p.endswith("/") else name == p for p in self.patterns)

    def keeps_section(self, section: str) -> bool:
        """Whether any metric of `section` (e.g. "eval") is kept, so the work producing it must run."""
        prefix = section.rstrip("/") + "/"
        if prefix in self.without:
            return False
        if not self.patterns:
            return True
        return any(p.startswith(prefix) or (p.endswith("/") and prefix.startswith(p))
                   for p in self.patterns)

    def filter(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.everything:
            return metrics
        return {key: value for key, value in metrics.items() if self.keeps(str(key))}

    def __repr__(self) -> str:
        without = f", without={list(self.without)}" if self.without else ""
        return f"MetricSelection({list(self.patterns) or 'everything'}{without})"


def _section(cfg: Any):
    section = cfg.get("telemetry") if hasattr(cfg, "get") else None
    return section if hasattr(section, "get") else {}


def evaluation_off(cfg: Any) -> bool:
    """Whether the experiment's eval frequency is 0: no evaluation runs and no eval/ metric is recorded."""
    if not hasattr(cfg, "get"):
        return False
    frequency = cfg.get("eval_freq_steps")
    if frequency is None:
        frequency = cfg.get("eval_freq")
    return isinstance(frequency, (int, float)) and not isinstance(frequency, bool) and frequency == 0


def selection_from_config(cfg: Any) -> MetricSelection:
    patterns = _section(cfg).get("metrics")
    if isinstance(patterns, str):
        patterns = [patterns]
    return MetricSelection(patterns, without=("eval",) if evaluation_off(cfg) else ())


def selection_errors(cfg: Any, tuning_metric: str | None = None) -> list[str]:
    """What is wrong with `telemetry.metrics`, if anything. A sweep must keep `tuning_metric`."""
    if tuning_metric and str(tuning_metric).startswith("eval/") and evaluation_off(cfg):
        return [f"This experiment sweeps on '{tuning_metric}', but evaluation is off (eval frequency 0), "
                "so it is never recorded. Set an eval frequency above 0 or sweep on another metric."]
    patterns = _section(cfg).get("metrics")
    if patterns is None:
        return []
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, (list, tuple)) or not all(isinstance(p, str) and p.strip() for p in patterns):
        return ["'telemetry.metrics' must be a list of metric names or sections (e.g. [eval/reward, losses/])."]
    selection = MetricSelection(patterns)
    if tuning_metric and not selection.keeps(tuning_metric):
        return [f"This experiment sweeps on '{tuning_metric}', so 'telemetry.metrics' must keep it."]
    return []
