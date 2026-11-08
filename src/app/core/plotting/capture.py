"""Reuse the figures a plotter builds for itself.

Most plotters here drive the pyplot state machine: `plt.figure()`, draw,
`plt.savefig()`, `plt.close()`. That hands back a PNG and throws the figure
away, so nothing downstream can restyle it.

matplotlib refuses to move an Axes between figures ("Can not put single
artist in more than one figure"), so a captured figure cannot be merged into
a composed one. What it can do is stay live: its title, colours and limits
are still editable and it still exports as vector. That is what lets the
Plots pane show every plotter, including the ones that never learned to draw
into an Axes supplied to them.

Prefer BasePlotter.panels() where a plotter offers it; this is the fallback.
"""

from __future__ import annotations

import contextlib
import io
from contextlib import contextmanager
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


@contextmanager
def capture_figures() -> Iterator[list[Figure]]:
    """Collect the figures made inside this block instead of writing them.

    `savefig` and `close` are intercepted for the duration, so a plotter runs
    unchanged while its figures survive. The caller owns what comes back and
    is responsible for closing it.
    """
    captured: list[Figure] = []
    real_savefig, real_close = plt.savefig, plt.close

    def remember() -> None:
        fig = plt.gcf()
        # A plotter saves then closes the same figure, so guard against
        # recording it twice.
        if fig is not None and all(fig is not seen for seen in captured):
            captured.append(fig)

    def fake_savefig(*_args, **_kwargs) -> None:
        remember()

    def fake_close(*_args, **_kwargs) -> None:
        remember()

    plt.savefig, plt.close = fake_savefig, fake_close
    try:
        yield captured
    finally:
        plt.savefig, plt.close = real_savefig, real_close


def figures_from(plotter, exp_id: str, cli_overrides: dict | None = None, quiet: bool = True) -> list[Figure]:
    """Run *plotter* and hand back the figures it would have saved.

    No *figure* is written to disk: savefig is intercepted. A plotter that
    writes other files itself, as `reports` does with its .md and .csv, still
    writes them, so this is not side-effect free. An empty list means the
    plotter produced no figure, which is normal for the ones that emit tables
    or reports rather than plots.

    Plotters announce each file they "saved"; with savefig intercepted those
    lines are untrue and would be alarming in the Plots pane, so stdout is
    swallowed unless the caller asks to see it.
    """
    sink = io.StringIO()
    with capture_figures() as figures:
        if quiet:
            with contextlib.redirect_stdout(sink):
                plotter.run(exp_id, cli_overrides=cli_overrides)
        else:
            plotter.run(exp_id, cli_overrides=cli_overrides)
    return figures
