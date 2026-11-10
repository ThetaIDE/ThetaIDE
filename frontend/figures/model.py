"""The figure being composed, with no Qt in it.

A figure is a grid of slots; each filled slot names a panel some plotter can
draw, plus the overrides the researcher applied on top. Keeping that
separate from the widgets means the rules about what a figure *is* run
headless in CI, the same way config_model.py does for the Experiment panel.

Overrides are applied after the plotter has drawn, rather than pushed into
its config beforehand. A plotter decides its own colours from the methods it
finds, so the only point where a specific line can be recoloured by hand is
once it exists on the Axes.

This module never touches frontend/plots.py: run viewing lives there and
figure composition lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from matplotlib.figure import Figure

LOGS_ROOT = Path("results/logs")
MAX_SLOTS = 24


@dataclass(frozen=True)
class PanelRef:
    """A panel placed on the figure, and what was changed about it.

    `key` is the PanelSpec key the plotter offered. The overrides are all
    optional: None means "leave whatever the plotter chose".
    """

    key: str
    exp_id: str
    plotter: str
    label: str
    title: str | None = None
    # Keyed by the series' legend label, so methods in the same panel keep
    # different colours. One colour for the whole panel
    # would merge them, which is the opposite of what the panel is for.
    colors: dict[str, str] = field(default_factory=dict)
    line_width: float | None = None
    legend: bool = True
    grid: bool = True

    def titled(self, text: str | None) -> PanelRef:
        return replace(self, title=text or None)


@dataclass
class FigureSpec:
    """Grid, panels and page setup for one composed figure."""

    rows: int = 1
    cols: int = 2
    panels: list[PanelRef | None] = field(default_factory=list)
    width: float = 10.0
    height: float = 4.0
    dpi: int = 300
    suptitle: str = ""
    tight: bool = True

    def __post_init__(self) -> None:
        self._fit_slots()

    # ── Slots ────────────────────────────────────────────────────────────

    @property
    def slot_count(self) -> int:
        return max(1, self.rows) * max(1, self.cols)

    def _fit_slots(self) -> None:
        """Pad or trim the slot list so it matches rows x cols."""
        wanted = self.slot_count
        while len(self.panels) < wanted:
            self.panels.append(None)
        if len(self.panels) > wanted:
            # Keep what fits, in order, so shrinking the grid drops empties
            # before it drops real panels.
            kept = [p for p in self.panels if p is not None][:wanted]
            self.panels = kept + [None] * (wanted - len(kept))

    def set_grid(self, rows: int, cols: int) -> list[PanelRef]:
        """Resize the grid. Returns any panels that no longer fit."""
        rows, cols = max(1, int(rows)), max(1, int(cols))
        if rows * cols > MAX_SLOTS:
            raise ValueError(f"a figure is limited to {MAX_SLOTS} panels")
        filled = [p for p in self.panels if p is not None]
        dropped = filled[rows * cols :]
        self.rows, self.cols = rows, cols
        self.panels = filled[: rows * cols]
        self._fit_slots()
        return dropped

    def used(self) -> list[PanelRef]:
        return [p for p in self.panels if p is not None]

    def first_free(self) -> int | None:
        for index, panel in enumerate(self.panels):
            if panel is None:
                return index
        return None

    # ── Editing ──────────────────────────────────────────────────────────

    def add(self, ref: PanelRef) -> int:
        """Place *ref* in the first free slot, growing the grid if needed."""
        slot = self.first_free()
        if slot is None:
            if self.slot_count >= MAX_SLOTS:
                raise ValueError(f"a figure is limited to {MAX_SLOTS} panels")
            # Grow by a row rather than a column: a paper figure is usually
            # read top to bottom and columns get narrow fast.
            self.rows += 1
            self._fit_slots()
            slot = self.first_free()
        assert slot is not None
        self.panels[slot] = ref
        return slot

    def remove(self, slot: int) -> PanelRef | None:
        if not 0 <= slot < len(self.panels):
            return None
        gone, self.panels[slot] = self.panels[slot], None
        return gone

    def move(self, src: int, dst: int) -> None:
        """Swap two slots, which is how panels get reordered."""
        if src == dst:
            return
        for index in (src, dst):
            if not 0 <= index < len(self.panels):
                return
        self.panels[src], self.panels[dst] = self.panels[dst], self.panels[src]

    def update(self, slot: int, **changes: Any) -> None:
        """Replace one slot's panel with a restyled copy."""
        if 0 <= slot < len(self.panels) and self.panels[slot] is not None:
            self.panels[slot] = replace(self.panels[slot], **changes)

    def compact(self) -> None:
        """Shuffle panels to the front, leaving the empty slots at the end."""
        filled = self.used()
        self.panels = filled + [None] * (self.slot_count - len(filled))


# ── Rendering ─────────────────────────────────────────────────────────────


def _same_color(rgba, spec) -> bool:
    """Whether a drawn artist's colour is the one a line started with."""
    from matplotlib.colors import to_rgba

    try:
        a, b = to_rgba(rgba), to_rgba(spec)
    except (ValueError, TypeError):
        return False
    return all(abs(x - y) < 1e-3 for x, y in zip(a[:3], b[:3]))


def series_labels(ax) -> list[str]:
    """Legend labels of the drawn series, in order, skipping matplotlib's own."""
    return [line.get_label() for line in ax.lines if not str(line.get_label()).startswith("_")]


def apply_overrides(ax, ref: PanelRef) -> None:
    """Restyle an Axes after its plotter has drawn into it.

    Colours are per series. A shaded mean+-SEM band is recoloured with its
    line by matching the colour the plotter gave both, since the band carries
    no label of its own to match on.
    """
    # One rule for every panel, rather than whatever each plotter's config
    # happened to produce: convergence titles itself "FIG_DEMO: Reward" from
    # the experiment name while losses titles itself "losses: Actor Loss", and
    # a figure mixing both looks unfinished. A composed figure carries its
    # subject in the suptitle, so a short panel name is what is wanted here.
    ax.set_title(ref.title if ref.title is not None else ref.label)
    for line in ax.lines:
        chosen = ref.colors.get(str(line.get_label()))
        if not chosen:
            continue
        was = line.get_color()
        line.set_color(chosen)
        for collection in ax.collections:  # the mean+-SEM band
            try:
                current = collection.get_facecolor()
            except Exception:
                continue
            if len(current) and _same_color(current[0], was):
                collection.set_color(chosen)
    if ref.line_width is not None:
        for line in ax.lines:
            line.set_linewidth(ref.line_width)

    ax.grid(ref.grid, alpha=0.3)

    legend = ax.get_legend()
    if ref.legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Inside the axes: a composed figure has no room for the margin
            # legend a standalone plot can afford.
            ax.legend(loc="best", fontsize="small")
    elif legend is not None:
        legend.remove()


def render(
    spec: FigureSpec,
    resolve: Callable[[PanelRef], Any],
    figure: Figure | None = None,
    dpi: int | None = None,
) -> Figure:
    """Draw *spec* into a Figure.

    *resolve* maps a PanelRef to something with a .draw(ax) -> bool, which is
    what plot.base.PanelSpec provides. *dpi* is for export; leave it unset
    for a preview and let the figure keep the screen resolution it has. A panel that cannot be resolved or
    draws nothing leaves a labelled empty Axes rather than vanishing, so a
    missing source is visible instead of silently changing the layout.
    """
    fig = figure if figure is not None else Figure()
    fig.clear()
    fig.set_size_inches(spec.width, spec.height)
    # Only when the caller asks. The on-screen preview draws at screen
    # resolution; baking the export DPI into it would size every label for a
    # 3300px canvas and render them enormous in an 800px widget.
    if dpi is not None:
        fig.set_dpi(dpi)

    for index, ref in enumerate(spec.panels):
        if ref is None:
            continue
        ax = fig.add_subplot(spec.rows, spec.cols, index + 1)
        panel = None
        try:
            panel = resolve(ref)
        except Exception as exc:  # a plotter raising must not lose the figure
            _mark_empty(ax, f"{ref.label}\ncould not load: {type(exc).__name__}")
            continue

        if panel is None:
            _mark_empty(ax, f"{ref.label}\nsource unavailable")
            continue

        try:
            drew = panel.draw(ax)
        except Exception as exc:
            _mark_empty(ax, f"{ref.label}\nfailed to draw: {type(exc).__name__}")
            continue

        if not drew:
            _mark_empty(ax, f"{ref.label}\nno data")
            continue
        apply_overrides(ax, ref)

    if spec.suptitle:
        fig.suptitle(spec.suptitle)
    if spec.tight:
        try:
            fig.tight_layout()
        except Exception:
            # tight_layout gives up on some arrangements; an untidy figure
            # beats no figure.
            pass
    return fig


def _mark_empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])


# ── Discovery ─────────────────────────────────────────────────────────────


def available_experiments(logs_root: Path | None = None) -> list[tuple[str, str]]:
    """(group, exp_id) for every run directory that holds metrics.

    Read off disk rather than from configs: a figure is made from runs that
    actually happened, and a configured experiment that never ran has
    nothing to plot.
    """
    root = Path(logs_root) if logs_root is not None else LOGS_ROOT
    if not root.is_dir():
        return []

    found = []
    from src.app.pipeline.result_paths import iter_run_dirs
    for exp_dir in iter_run_dirs(root):
        if next(exp_dir.rglob("metrics.csv"), None) is not None:
            group, experiment = exp_dir.relative_to(root).as_posix().split("/", 1)
            found.append((group, experiment))
    return found
