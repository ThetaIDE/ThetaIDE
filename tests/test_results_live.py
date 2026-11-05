"""Results are static completed snapshots, with bounded chart rendering at scale."""

import os
import sys
from urllib.parse import parse_qs, urlencode, urlparse

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.results.data import MetricTable, Source, thin  # noqa: E402
from frontend.results.loader import ResultsLoader  # noqa: E402
from frontend.results.panel import ResultsPanel  # noqa: E402

SOURCE = Source("g", "live_run", "ppo")


class LocalBackendCalls:
    """LocalBackend's results methods for a fake that answers the old API paths (GET /api/runs/...).

    The fakes keep recording requests as those paths, so tests can check what was asked for and with which
    version/since, the same as when the Results panel talked to the FastAPI server.
    """

    def list_runs(self, callback, *, completed_only=False):
        self.get("/api/runs", callback)

    def run_manifest(self, group, experiment_id, callback, *, completed_only=False):
        self.get(f"/api/runs/{group}/{experiment_id}/manifest", callback)

    def run_metrics(self, group, experiment_id, agent, callback, version=None, since=0, *, completed_only=False):
        params = {key: value for key, value in (("version", version), ("since", since)) if value}
        query = f"?{urlencode(params)}" if params else ""
        self.get(f"/api/runs/{group}/{experiment_id}/{agent}/metrics{query}", callback)

    def run_file(self, group, experiment_id, path, callback):
        self.get_bytes(f"/api/runs/{group}/{experiment_id}/files/{path}", callback)


class LiveBackend(LocalBackendCalls):
    """A run whose metrics.csv grows between requests, served like GET …/metrics?since=N."""

    def __init__(self):
        self.rows = [{"transitions": "1000", "eval/reward": "10"}]
        self.version = "version_0"
        self.finished = False
        self.requests = []

    def get(self, path, callback):
        self.requests.append(path)
        url = urlparse(path)
        if url.path == "/api/runs":
            agents = [{"name": "ppo"}]
            callback(
                {"runs": [{"group": "g", "experiment_id": "live_run", "agents": agents, "finished": self.finished}]},
                None,
            )
        elif url.path.endswith("/manifest"):
            callback(
                {"group": "g", "experiment_id": "live_run", "agents": {"ppo": [self.version]}, "artifacts": []}, None
            )
        elif url.path.endswith("/metrics"):
            since = int(parse_qs(url.query).get("since", ["0"])[0])
            callback(
                {
                    "metrics": [dict(r) for r in self.rows[since:]],
                    "columns": ["transitions", "eval/reward"],
                    "total_rows": len(self.rows),
                    "version": self.version,
                },
                None,
            )
        else:
            callback(None, f"HTTP 404: {path}")

    def get_bytes(self, path, callback):
        callback(None, "HTTP 404")


def metrics_requests(backend):
    return [p for p in backend.requests if "/metrics" in p]


# ── Loader ───────────────────────────────────────────────────────────────────


def test_results_loader_requests_completed_snapshots():
    class Backend(LiveBackend):
        def list_runs(self, callback, *, completed_only=False):
            assert completed_only
            super().list_runs(callback)

        def run_manifest(self, group, experiment_id, callback, *, completed_only=False):
            assert completed_only
            super().run_manifest(group, experiment_id, callback)

        def run_metrics(self, group, experiment_id, agent, callback, **kwargs):
            assert kwargs.pop("completed_only")
            super().run_metrics(group, experiment_id, agent, callback, **kwargs)

    backend = Backend()
    loader = ResultsLoader(backend)
    entries = []
    loader.runs_listed.connect(entries.extend)
    loader.list_runs()
    assert entries == []
    backend.finished = True
    loader.list_runs()
    assert len(entries) == 1
    loader.load_manifest("g", "live_run")
    loader.load_table(SOURCE)
    before = list(backend.requests)
    backend.rows.append({"transitions": "2000", "eval/reward": "30"})
    loader.load_table(SOURCE)
    assert backend.requests == before
    assert loader.tables[SOURCE].row_count == 1


def test_results_exclude_active_runs_and_only_refresh_on_request():
    from PyQt6.QtCore import QTimer
    backend = LiveBackend()
    panel = ResultsPanel(backend)
    panel.refresh()
    assert panel.explorer.tree.topLevelItemCount() == 0
    assert not panel.findChildren(QTimer)
    backend.finished = True
    panel.refresh()
    assert panel.explorer.tree.topLevelItemCount() == 1
    panel.explorer.select_run("g", "live_run")
    panel.show()
    before = list(backend.requests)
    tree_before = panel.explorer.tree.topLevelItem(0)
    backend.rows.append({"transitions": "2000", "eval/reward": "30"})
    app.processEvents()
    assert backend.requests == before
    assert panel.explorer.tree.topLevelItem(0) is tree_before
    assert panel.loader.tables[SOURCE].row_count == 1
    panel.refresh()
    assert panel.loader.tables[SOURCE].row_count == 2
    panel.close()
    panel.deleteLater()


def test_refresh_discards_old_requests_instead_of_repopulating_the_tree():
    class DeferredBackend(LiveBackend):
        def get(self, path, callback):
            self.requests.append((path, callback))

    backend = DeferredBackend()
    loader = ResultsLoader(backend)
    listings, manifests = [], []
    loader.runs_listed.connect(listings.append)
    loader.manifest_loaded.connect(manifests.append)
    loader.list_runs()
    loader.load_manifest("g", "live_run")
    loader.load_table(SOURCE)
    old = list(backend.requests)
    loader.clear()
    loader.list_runs()
    loader.load_manifest("g", "live_run")
    loader.load_table(SOURCE)
    assert len(backend.requests) == 6  # stale pending requests cannot block a fresh load
    old[0][1]({"runs": [{"finished": True}]}, None)
    old[1][1]({"group": "g", "experiment_id": "live_run", "agents": {}, "artifacts": []}, None)
    old[2][1]({"metrics": [{"transitions": "1"}], "version": "version_0"}, None)
    assert listings == manifests == []
    assert loader.manifests == loader.tables == {}
    backend.requests[3][1]({"runs": []}, None)
    assert listings == [[]]


# ── Scale ────────────────────────────────────────────────────────────────────


def test_thin_keeps_spikes_and_the_latest_point():
    points = [{"step": i, "m": 0.0} for i in range(10_000)]
    points[4321]["m"] = 99.0  # a spike
    points[777]["m"] = -5.0  # a dip
    drawn = thin(points, "m", limit=200)
    assert len(drawn) <= 202
    assert {p["m"] for p in drawn} >= {99.0, -5.0, 0.0}
    assert drawn[-1] is points[-1] and [p["step"] for p in drawn] == sorted(p["step"] for p in drawn)
    short = points[:50]
    assert thin(short, "m", limit=200) is short


def test_long_series_are_thinned_for_drawing_only(tmp_path):
    backend = LiveBackend()
    backend.finished = True
    backend.rows = [{"transitions": str(i), "eval/reward": str(i % 7)} for i in range(6000)]
    panel = ResultsPanel(backend)
    panel.refresh()
    panel.explorer.select_run("g", "live_run")
    drawn = panel.grid.cards["eval/reward"].chart.series[0][1]
    assert len(drawn) <= 2001
    assert panel.summary.rows[0][-1] == 6000  # Summary uses every point
    assert len(panel.curves_csv().splitlines()) == 6001  # so does the export (header + every point)


def test_metric_table_truncate_and_row():
    table = MetricTable.from_rows([{"x": "1", "a": "2"}, {"x": "2", "b": "5"}])
    assert table.row(1) == {"x": 2.0, "b": 5.0}
    table.truncate(1)
    assert table.row_count == 1 and "b" not in table.columns and table.row(0) == {"x": 1.0, "a": 2.0}


def test_backend_run_summaries_are_cached_until_files_change(tmp_path, monkeypatch):
    from src.app.services import catalog

    logs = tmp_path / "results" / "logs"
    monkeypatch.setattr(catalog, "LOGS_DIR", logs)
    version = logs / "g" / "exp" / "ppo" / "version_0"
    version.mkdir(parents=True)
    metrics = version / "metrics.csv"
    metrics.write_text("transitions,eval/reward\n1000,10\n", encoding="utf-8")
    reads = []
    real = catalog._read_agent_summary
    monkeypatch.setattr(catalog, "_read_agent_summary", lambda *a: reads.append(a[0]) or real(*a))
    first = catalog.list_runs()[0]["agents"][0]
    catalog.list_runs()
    assert len(reads) == 1  # unchanged files: served from the cache
    metrics.write_text("transitions,eval/reward\n1000,10\n2000,50\n", encoding="utf-8")  # training went on
    second = catalog.list_runs()[0]["agents"][0]
    assert len(reads) == 2 and (first["best_reward"], second["best_reward"]) == (10.0, 50.0)
