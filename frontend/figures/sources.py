"""Where panels come from: the plotters in plot/.

Two tiers, because not every plotter can draw into an Axes someone else
owns. Those that can (BasePlotter.panels()) become composable panels. Those
that build their own figures are offered through plot.capture instead, as
whole figures that can be restyled and exported but not merged into a grid.

Discovery is cached per (plotter, experiment): panels() reloads every
metrics.csv for the run, which is far too slow to repeat on each keystroke
in the styling controls.
"""

from __future__ import annotations

from typing import Any

from .model import PanelRef

# Marks a panel that is a whole captured figure rather than a drawn metric.
CAPTURED = "captured:"


class _CapturedPanel:
    """A captured figure, drawn into someone else's Axes as an image."""

    def __init__(self, figure):
        self.figure = figure

    def draw(self, ax) -> bool:
        import io

        from matplotlib.image import imread

        buffer = io.BytesIO()
        self.figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
        buffer.seek(0)
        ax.imshow(imread(buffer))
        ax.set_axis_off()
        return True


class PanelSource:
    """The catalogue of panels available for an experiment."""

    def __init__(self) -> None:
        self._registry: dict[str, Any] | None = None
        self._panels: dict[tuple[str, str], list] = {}
        self._figures: dict[tuple[str, str], list] = {}
        self._errors: dict[tuple[str, str], str] = {}

    # ── Plotters ─────────────────────────────────────────────────────────

    def registry(self) -> dict[str, Any]:
        if self._registry is None:
            from plot.manager import discover_plotters

            from src.app.core.plotter_registry import get_plotter_class
            self._registry = {}
            for name, cls in discover_plotters().items():
                try:
                    cls = cls or get_plotter_class(name)
                    if cls is not None:
                        self._registry[name] = cls
                except Exception as exc:
                    self._errors[(name, "")] = f"{type(exc).__name__}: {exc}"
        return self._registry

    def plotter_names(self) -> list[str]:
        """One name per distinct plotter, not one per registry key.

        The registry maps a plotter under both its own `name` and its module
        name, so iterating the keys offers every installed plotter twice. The
        plotter's own name wins, since that is what its config file and its
        panels are keyed on.
        """
        best: dict[int, str] = {}
        for key, cls in self.registry().items():
            own = str(getattr(cls, "name", "") or "")
            current = best.get(id(cls))
            if current is None or key == own or (current != own and len(key) < len(current)):
                best[id(cls)] = key
        return sorted(best.values())

    # ── Composable panels ────────────────────────────────────────────────

    def _panels_for(self, plotter_name: str, exp_id: str) -> list:
        cache_key = (plotter_name, exp_id)
        if cache_key in self._panels:
            return self._panels[cache_key]

        cls = self.registry().get(plotter_name)
        if cls is None:
            self._panels[cache_key] = []
            return []
        try:
            found = cls().panels(exp_id)
        except Exception as exc:
            # One broken plotter must not empty the whole catalogue.
            self._errors[cache_key] = f"{type(exc).__name__}: {exc}"
            found = []
        self._panels[cache_key] = found
        return found

    def catalog(self, exp_id: str) -> list[PanelRef]:
        """Every panel available for *exp_id*, from both tiers.

        Plotters that draw into a supplied Axes contribute one entry per
        metric. The rest contribute their whole figures, placed as images;
        without this they would be absent from the pane entirely, which is
        what happened to `reports`.
        """
        refs: list[PanelRef] = []
        for plotter_name in self.plotter_names():
            specs = self._panels_for(plotter_name, exp_id)
            if specs:
                for spec in specs:
                    refs.append(
                        PanelRef(
                            key=spec.key,
                            exp_id=exp_id,
                            plotter=plotter_name,
                            label=f"{plotter_name}: {spec.label}",
                        )
                    )
                continue

            # Listed without running it. Capturing means executing the
            # plotter, which is slow and not side-effect free: `reports`
            # writes .md and .csv files of its own that savefig interception
            # does not catch. That happens only if the panel is actually
            # placed on a figure.
            refs.append(
                PanelRef(
                    key=f"{CAPTURED}{plotter_name}/0",
                    exp_id=exp_id,
                    plotter=plotter_name,
                    label=f"{plotter_name}: whole figure",
                )
            )
        return refs

    def _captured(self, plotter_name: str, exp_id: str) -> list:
        """Figures a non-composable plotter makes, run once and cached.

        Running a plotter is expensive and can write files of its own, and
        the styling controls redraw on every keystroke, so this runs at most
        once per plotter and run.
        """
        key = (plotter_name, exp_id)
        if key not in self._figures:
            self._figures[key] = self.capture(plotter_name, exp_id)
        return self._figures[key]

    def resolve(self, ref: PanelRef):
        """Something with .draw(ax) for this ref, or None when it is gone."""
        if ref.key.startswith(CAPTURED):
            return self._image_panel(ref)
        for spec in self._panels_for(ref.plotter, ref.exp_id):
            if spec.key == ref.key:
                return spec
        return None

    def _image_panel(self, ref: PanelRef):
        """Place a captured figure into an Axes as an image.

        matplotlib refuses to move an Axes between figures, so a plotter that
        builds its own cannot be merged into a composed one as vector art.
        Rasterising it keeps it usable on the figure; the trade-off is that
        this panel does not scale like the drawn ones, so it is rendered at
        300 dpi to stay sharp in print.
        """
        try:
            index = int(ref.key.rsplit("/", 1)[1])
        except (IndexError, ValueError):
            return None
        figures = self._captured(ref.plotter, ref.exp_id)
        if index >= len(figures):
            return None
        return _CapturedPanel(figures[index])

    # ── Whole-figure fallback ────────────────────────────────────────────

    def capture(self, plotter_name: str, exp_id: str) -> list:
        """Figures from a plotter that cannot draw into a supplied Axes.

        Nothing is written to disk; see plot/capture.py.
        """
        from plot.capture import figures_from

        cls = self.registry().get(plotter_name)
        if cls is None:
            return []
        try:
            return figures_from(cls(), exp_id)
        except Exception as exc:
            self._errors[(plotter_name, exp_id)] = f"{type(exc).__name__}: {exc}"
            return []

    def composable(self, exp_id: str) -> list[str]:
        """Plotters offering panels for this experiment."""
        return [name for name in self.plotter_names() if self._panels_for(name, exp_id)]

    def errors(self) -> dict[tuple[str, str], str]:
        return dict(self._errors)

    def invalidate(self) -> None:
        """Drop the cache so a finished run shows its new metrics."""
        self._panels.clear()
        self._figures.clear()
        self._errors.clear()
