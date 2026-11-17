"""A tiny figure/vega-lite-v1 provider over the fixture runs' metrics."""
from src.app.api.plotting import BasePlotter


class FixtureCurve(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("fixture_curve", paths=paths)

    def run(self, exp_id, cli_overrides=None):
        pass

    def figure_spec(self, exp_id, params):
        group = exp_id.split("/")[0]
        values = []
        for method, versions in sorted(self.load_metrics(group, exp_id).items()):
            for df in versions.values():
                rows = df.dropna(subset=[params["metric"]]) if params["metric"] in df.columns else df.iloc[0:0]
                values += [{"method": method, "step": float(r["step"]), "value": float(r[params["metric"]])}
                           for _, r in rows.iterrows()]
        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
            "params": [{"name": "color", "value": params["color"]}],
            "data": {"values": values},
            "mark": {"type": "line", "color": {"expr": "color"}},
            "encoding": {"x": {"field": "step", "type": "quantitative"},
                         "y": {"field": "value", "type": "quantitative", "title": params["metric"]},
                         "detail": {"field": "method"}},
        }


class FixtureTable(BasePlotter):
    """A plotter without the figure contract: never offered in Discussion."""

    def __init__(self, *, paths):
        super().__init__("fixture_table", paths=paths)

    def run(self, exp_id, cli_overrides=None):
        pass
