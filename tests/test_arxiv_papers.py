"""arXiv Papers plugin: component tags, query building, searching (with a fake arXiv) and the panel."""

import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from src.plugins.app.bundled.arxiv_papers import search as search_module  # noqa: E402
from src.plugins.app.bundled.arxiv_papers.catalog import build_catalog, config_selectors, tag_counts, tag_phrase  # noqa: E402
from src.plugins.app.bundled.arxiv_papers.panel import ArxivPanel  # noqa: E402
from src.plugins.app.bundled.arxiv_papers.search import ArxivSearcher, Paper, SearchError, build_query  # noqa: E402



def paper(n, title="Conservative Q-Learning for Offline Reinforcement Learning"):
    return Paper(arxiv_id=f"2006.0477{n}v1", title=f"{title} {n}", authors=["Aviral Kumar", "Aurick Zhou"],
                 summary="An abstract.", published="2020-06-08", updated="2020-08-19", primary_category="cs.LG",
                 categories=["cs.LG", "stat.ML"], abs_url=f"http://arxiv.org/abs/2006.0477{n}v1",
                 pdf_url=f"http://arxiv.org/pdf/2006.0477{n}v1")


class FakeArxiv:
    def __init__(self, papers=None, error=None):
        self.calls, self.papers, self.error = [], papers or [paper(1), paper(2)], error

    def __call__(self, query, max_results, sort):
        self.calls.append((query, max_results, sort))
        if self.error:
            raise self.error
        return list(self.papers)


@pytest.fixture(autouse=True)
def no_spacing(monkeypatch):
    monkeypatch.setattr(search_module, "MIN_INTERVAL", 0.0)
    search_module.RATE_GATE.reset()
    yield
    search_module.RATE_GATE.reset()


def wait_for(predicate, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def mock_config_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("config_root")
    (root / "agent").mkdir(parents=True)
    (root / "model").mkdir(parents=True)
    (root / "env").mkdir(parents=True)
    (root / "agent" / "cql.yaml").write_text("agent: cql\n", encoding="utf-8")
    (root / "agent" / "ppo.yaml").write_text("agent: ppo\n", encoding="utf-8")
    (root / "agent" / "iql.yaml").write_text("agent: iql\n", encoding="utf-8")
    (root / "model" / "nsfr.yaml").write_text("model: nsfr\n", encoding="utf-8")
    (root / "env" / "cartpole.yaml").write_text("env: cartpole\n", encoding="utf-8")
    (root / "env" / "pyrenees.yaml").write_text("env: pyrenees\n", encoding="utf-8")
    return root


# ── catalog and queries ─────────────────────────────────────────────────────

def test_local_components_get_built_in_tags_and_hub_components_their_index_tags(mock_config_root):
    hub = SimpleNamespace(components=[SimpleNamespace(id="blendrl", name="BlendRL", kind="model",
                                                      tags=["BlendRL", "neural-symbolic"], description="")],
                          cache_file=None)
    catalog = {c.key: c for c in build_catalog(config_selectors(mock_config_root), hub)}
    assert catalog["agent/cql"].tags == ["cql", "offline-rl", "q-learning"]
    assert catalog["hub/blendrl"].tags == ["blendrl", "neural-symbolic"] and catalog["hub/blendrl"].source == "hub"
    assert "env/cartpole" in catalog and "model/nsfr" in catalog
    counts = tag_counts(catalog.values())
    assert counts["offline-rl"] >= 2 and list(counts) == sorted(counts, key=lambda t: (-counts[t], t))


def test_user_tags_replace_the_defaults(mock_config_root):
    catalog = {c.key: c for c in build_catalog(config_selectors(mock_config_root), None, {"env/pyrenees": ["Offline RL", "health"]})}
    assert catalog["env/pyrenees"].tags == ["offline-rl", "health"] and catalog["env/pyrenees"].edited


def test_tags_expand_to_search_phrases():
    assert tag_phrase("cql") == "conservative q-learning"
    assert tag_phrase("graph-neural-network") == "graph neural network"
    assert build_query(["offline-rl", "cql"], "benchmark") == (
        'all:"offline reinforcement learning" AND all:"conservative q-learning" AND all:benchmark')
    assert build_query(["ppo", "iql"], "", match_all=False) == (
        '(all:"proximal policy optimization" OR all:"implicit q-learning")')
    assert build_query([], 'robust "offline" (rl)') == "all:robust AND all:offline AND all:rl"
    assert build_query([], "  ") == ""


# ── searcher ────────────────────────────────────────────────────────────────

def test_repeat_searches_come_from_the_cache_on_disk():
    fake = FakeArxiv()
    with tempfile.TemporaryDirectory() as temp:
        cache = Path(temp) / "arxiv_cache.json"
        papers, cached = ArxivSearcher(cache, fetch=fake).search('all:"cql"', 10, "newest")
        assert [p.arxiv_id for p in papers] == ["2006.04771v1", "2006.04772v1"] and not cached
        again, cached = ArxivSearcher(cache, fetch=fake).search('all:"cql"', 10, "newest")  # a new session
        assert cached and again == papers and len(fake.calls) == 1
        ArxivSearcher(cache, fetch=fake).search('all:"cql"', 25, "newest")  # different size: asks arXiv
        assert len(fake.calls) == 2


def test_requests_are_spaced_out_across_every_searcher(monkeypatch):
    monkeypatch.setattr(search_module, "MIN_INTERVAL", 0.3)
    start = time.monotonic()
    ArxivSearcher(fetch=FakeArxiv()).search("all:a")
    ArxivSearcher(fetch=FakeArxiv()).search("all:b")  # e.g. the plugin was turned off and on again
    assert time.monotonic() - start >= 0.25  # Windows' clock ticks every ~16 ms


def test_after_a_429_arxiv_is_left_alone_for_a_growing_cool_down():
    fake = FakeArxiv(error=search_module.RateLimited("rate limited"))
    searcher = ArxivSearcher(fetch=fake)
    with pytest.raises(search_module.RateLimited, match="about 2 minutes"):
        searcher.search("all:a")
    with pytest.raises(search_module.RateLimited, match="slow down"):
        searcher.search("all:b")
    assert len(fake.calls) == 1  # the second search never reached arXiv
    assert search_module.RATE_GATE.cooldown == 2 * search_module.COOLDOWN_START  # the next 429 waits longer
    search_module.RATE_GATE.blocked_until = 0.0  # the cool-down has passed
    fake.error = None
    assert searcher.search("all:c")[0]
    assert search_module.RATE_GATE.cooldown == search_module.COOLDOWN_START


def test_saved_results_still_open_during_a_cool_down():
    searcher = ArxivSearcher(fetch=FakeArxiv())
    searcher.search("all:a")
    search_module.RATE_GATE.blocked_until = time.monotonic() + 60
    papers, cached = searcher.search("all:a")
    assert cached and papers


def test_empty_query_is_refused_without_a_request():
    fake = FakeArxiv()
    with pytest.raises(SearchError):
        ArxivSearcher(fetch=fake).search("  ")
    assert fake.calls == []


def test_rate_limit_is_reported_plainly(monkeypatch):
    arxiv = pytest.importorskip("arxiv")

    made = []

    class Client:
        def __init__(self, **kwargs):
            made.append(kwargs)

        def results(self, _search):
            raise arxiv.HTTPError("https://export.arxiv.org/api/query", 0, 429)

    monkeypatch.setattr(arxiv, "Client", Client)
    with pytest.raises(SearchError, match="slow down"):
        ArxivSearcher().search('all:"cql"')
    assert made[0]["num_retries"] == 0  # never retried automatically


def test_citation_is_bibtex():
    bib = paper(1).citation()
    assert bib.startswith("@misc{kumar2020") and "eprint        = {2006.04771}" in bib
    assert "author        = {Aviral Kumar and Aurick Zhou}" in bib


# ── panel ───────────────────────────────────────────────────────────────────

@pytest.fixture
def panel(mock_config_root):
    fake = FakeArxiv()
    widget = ArxivPanel(searcher=ArxivSearcher(fetch=fake), hub_client=None, config_root=mock_config_root)
    widget.fake = fake
    widget.resize(1000, 700)
    widget.show()
    yield widget
    widget.cleanup()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def find_component(panel, key):
    tree = panel.component_tree
    for i in range(tree.topLevelItemCount()):
        group = tree.topLevelItem(i)
        for j in range(group.childCount()):
            if group.child(j).data(0, Qt.ItemDataRole.UserRole) == key:
                return group.child(j)
    raise AssertionError(key)


def test_clicking_a_component_searches_its_tags(panel):
    item = find_component(panel, "agent/cql")
    panel.component_tree.setCurrentItem(item)
    panel._on_component_clicked(item)
    assert set(panel.selected_tags()) == {"cql", "offline-rl", "q-learning"}
    assert wait_for(lambda: panel.results_list.count() == 2)
    query = panel.fake.calls[-1][0]
    assert 'all:"conservative q-learning"' in query and " AND " in query
    assert "2 papers found" in panel.status_label.text()
    assert panel.results_list.currentRow() == 0 and "Aviral Kumar" in panel.detail.toPlainText()
    assert panel.open_button.isEnabled() and panel.cite_button.isEnabled()


def test_checking_tags_updates_the_query_and_any_mode(panel):
    panel.set_selected_tags(["ppo", "iql"])
    panel.match_combo.setCurrentIndex(1)  # match any
    text = panel.query_label.text()
    assert text.startswith("Query: (") and " OR " in text
    assert 'all:"proximal policy optimization"' in text and 'all:"implicit q-learning"' in text
    panel.clear_tags()
    assert not panel.search_button.isEnabled()
    panel.search_input.setText("world models")
    assert panel.search_button.isEnabled()


def test_errors_show_in_the_status_line(panel):
    panel.fake.error = SearchError("Couldn't reach arXiv (ConnectionError). Check your connection.")
    panel.search_input.setText("anything")
    panel.search()
    assert wait_for(lambda: "Couldn't reach arXiv" in panel.status_label.text())
    assert panel.results_list.count() == 0


def test_only_the_latest_search_is_shown(panel):
    panel.search_input.setText("first")
    panel.search()
    stale = panel.serial
    panel.search_input.setText("second")
    panel.search()
    wait_for(lambda: panel.results_list.count() == 2)
    panel.results_list.clear()
    panel._on_results(stale, [paper(9)], False, "")
    assert panel.results_list.count() == 0  # the older search's results were dropped


def test_editing_tags_is_saved_and_empty_restores_defaults(panel):
    panel.set_component_tags("env/pyrenees", ["sepsis", "offline rl"])
    assert panel.component("env/pyrenees").tags == ["sepsis", "offline-rl"]
    assert any(panel.tag_list.item(i).data(Qt.ItemDataRole.UserRole) == "sepsis" for i in range(panel.tag_list.count()))
    panel.set_component_tags("env/pyrenees", [])
    assert panel.component("env/pyrenees").tags == []


def test_copy_citation(panel):
    panel.search_input.setText("cql")
    panel.search()
    assert wait_for(lambda: panel.results_list.count() == 2)
    panel.copy_citation()
    assert QApplication.clipboard().text().startswith("@misc{kumar2020")


def test_plugin_toggles_its_sidebar_tab():
    from PyQt6.QtWidgets import QWidget

    from frontend.app import Window

    class HeadlessTerminalPanel(QWidget):
        def __init__(self, *, parent=None, **_kwargs):
            super().__init__(parent)
            self.terminals = []

        def configure(self, **_kwargs):
            pass

        def set_font_size(self, _size):
            pass

        def apply_theme(self, _theme):
            pass

    class InertBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def config_schema(self, *_args, **_kwargs):
            pass

        def compose(self, *_args, **_kwargs):
            pass

        def list_runs(self, *_args, **_kwargs):
            pass

        def run_metrics(self, *_args, **_kwargs):
            pass

    with tempfile.TemporaryDirectory() as temp:
        # This test is about plugin tab registration, not shells or backend I/O.
        with (
            patch("frontend.app.TerminalPanel", HeadlessTerminalPanel),
            patch("frontend.app.LocalBackend", InertBackend),
            patch.object(Window, "init_default_experiment", lambda self: None),
        ):
            window = Window(Path(temp) / "runs")
        try:
            manager = window.plugin_manager
            assert "arxiv_papers" in manager.manifests
            assert manager.enable_plugin("arxiv_papers")
            assert "arxiv_papers" in window.tabs.tabs
            manager.disable_plugin("arxiv_papers")
            assert "arxiv_papers" not in manager.instances and "arxiv_papers" not in window.tabs.tabs
            assert manager.enable_plugin("arxiv_papers") and "arxiv_papers" in window.tabs.tabs
        finally:
            window._quitting = True
            window.close()
