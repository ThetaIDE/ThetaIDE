"""The ``figure/vega-lite-v1`` contract: plot components that build Discussion figures.

A plotter capability opts in by listing ``FIGURE_CONTRACT`` in ``provides``, declaring a
``parameter_schema`` (properties marked ``"x-theta-live": true`` are live parameters) and
implementing ``figure_spec(self, exp_id, params) -> dict`` on its ``BasePlotter`` subclass.
The returned Vega-Lite spec declares one top-level ``params`` entry per live parameter and
no ``bind``. The host never edits a spec's structure; it only sets those signals.
See docs/reference/FIGURE_CONTRACT.md. Headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FIGURE_CONTRACT = "figure/vega-lite-v1"
LIVE = "x-theta-live"


@dataclass(frozen=True)
class FigureCandidate:
    """A plotter capability that provides the figure contract, with its findings for one experiment."""

    package: object
    capability: object
    findings: list = field(default_factory=list)

    @property
    def selector(self) -> str:
        return self.capability.id

    @property
    def blocked(self) -> bool:
        return any(f.level == "block" for f in self.findings)

    @property
    def warnings(self) -> list:
        return [f for f in self.findings if f.level == "warn"]


def figure_capabilities(*, paths) -> list:
    """Every installed (package, capability) that provides the figure contract, by capability ID."""
    from src.app.core.provider_catalog import get_catalog
    found = [(package, capability) for package in get_catalog(paths).packages for capability in package.capabilities
             if capability.kind == "plotter" and FIGURE_CONTRACT in (capability.metadata.get("provides") or [])]
    return sorted(found, key=lambda pair: (pair[1].id.lower(), pair[0].id))


def compatible_figure_plotters(cfg, *, paths) -> list[FigureCandidate]:
    """Figure plotters with their declared findings against a composed experiment.

    Reads ``requires`` declarations only (``figure_findings``); preflight hooks never run.
    Callers omit blocked candidates and mark candidates with warnings.
    """
    from src.app.core.provider_discovery import provider_discovery_snapshot
    from src.app.core.requirements import figure_findings
    with provider_discovery_snapshot():
        return [FigureCandidate(package, capability, figure_findings(cfg, (package, capability), paths=paths))
                for package, capability in figure_capabilities(paths=paths)]


def spec_error(spec, live: list[str]) -> str | None:
    """Why a returned spec breaks the contract, or None."""
    if not isinstance(spec, dict):
        return f"figure_spec must return a Vega-Lite spec mapping, not {type(spec).__name__}"
    declared = spec.get("params") or []
    if not isinstance(declared, list):
        return "The spec's top-level params must be a list"
    names = {p.get("name") for p in declared if isinstance(p, dict)}
    missing = [name for name in live if name not in names]
    if missing:
        return ("The spec declares no top-level param for live parameter"
                f"{'s' if len(missing) > 1 else ''} {', '.join(missing)}")
    bound = sorted({str(p.get("name")) for p in _all_params(spec)
                    if "bind" in p and p["bind"] not in VIEW_BINDINGS})
    if bound:
        return f"Spec params must not bind controls ({', '.join(bound)}); Theta's form owns them"
    return None


# Selections may bind to the view itself: zoom/pan (``"scales"``) and legend toggling (``"legend"``).
VIEW_BINDINGS = ("scales", "legend")


def _all_params(node):
    """Every param declared anywhere in a spec: top level, layers, concatenations and facets."""
    if isinstance(node, dict):
        for param in node.get("params") or []:
            if isinstance(param, dict):
                yield param
        for key, value in node.items():
            if key not in ("params", "data", "datasets") and isinstance(value, (dict, list)):
                yield from _all_params(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_params(item)


def build_figure_spec(package, capability, exp_id: str, params: dict, live: list[str], *, paths) -> dict:
    """Run a plotter capability's ``figure_spec`` and check the result against the contract.

    Raises ValueError for a contract violation; the component's own exceptions propagate.
    """
    from src.app.core.provider_loader import load_capability
    plotter_class = load_capability(package, capability, paths=paths)
    plotter = plotter_class(paths=paths)
    if not callable(getattr(plotter, "figure_spec", None)):
        raise ValueError(f"Plot component {capability.id!r} lists {FIGURE_CONTRACT} but has no figure_spec")
    spec = plotter.figure_spec(exp_id, params)
    error = spec_error(spec, live)
    if error:
        raise ValueError(error)
    return spec
