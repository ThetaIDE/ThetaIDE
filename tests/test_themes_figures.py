"""Native PlotMan discovery and composable, read-only THEMES panels."""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import load_capability
from src.app.workspace import WorkspacePaths

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/usr/packages/themes-dev'


@pytest.fixture
def plots(tmp_path, monkeypatch):
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    monkeypatch.delenv('THETA_DATA_DIR', raising=False)
    provider = tmp_path / 'src/usr/packages/themes-dev'
    shutil.copytree(PACKAGE, provider, ignore=shutil.ignore_patterns('__pycache__', 'build', '*.egg-info'))
    selected = get_catalog(WorkspacePaths(tmp_path)).resolve('plotter', 'themes-dev/themes')
    cls = load_capability(*selected, paths=WorkspacePaths(tmp_path))
    bundle = tmp_path / 'results/checkpoints/AL/codex/example/method/run-a/bundle'
    bundle.mkdir(parents=True)
    meta = {'artifact_version':1,'status':'complete','action_feature':'a_2',
        'metrics':{'heldout_constant_action_nll_per_scored_step':.67,'heldout_mixture_nll_per_scored_step':.47},
        'ticc_history':[{'iteration':0,'changed_fraction':.2},{'iteration':1,'changed_fraction':.01}],
        'em_edm_history':[{'iteration':0,'conditional_log_likelihood':-4.,'component_ids':[0,2],
                          'training_losses':[[{'cross_entropy':.3}],[{'cross_entropy':.4}]]}]}
    (bundle/'manifest.json').write_text(json.dumps(meta))
    (bundle.parent/'execution.json').write_text(json.dumps({'status':'completed'}))
    pd.DataFrame([{'visit_id':f'visit-{i//2}', 'split':'train' if i<4 else 'test',
        'policy_group':i%2, 'physiological_stage':i%3, 'length':3,
        'policy_scored_steps':2, 'membership_probability':.8,
        'visit_start_row':3*(i%2),'source_start_slot':3*i, 'action_rate':.1 if i%2 else .8}
        for i in range(6)]).to_csv(bundle/'segments.csv',index=False)
    return cls(), bundle


def test_plotman_discovers_real_provider_panels(plots):
    from frontend.figures.sources import PanelSource
    source = PanelSource()
    references = [r for r in source.catalog('AL/codex/example') if r.plotter=='themes-dev/themes']
    assert len(references)==11
    assert all(source.resolve(r) is not None for r in references)


def test_all_panels_draw_into_supplied_axes_without_fitting_or_writing(plots):
    plotter, bundle = plots
    original = {p:p.read_bytes() for p in bundle.iterdir() if p.is_file()}
    panels = plotter.panels('AL/codex/example')
    assert len(panels)==11
    import matplotlib.pyplot as plt
    before=set(plt.get_fignums())
    for panel in panels:
        fig=Figure();ax=fig.add_subplot(111)
        assert panel.draw(ax),panel.key
        assert len(fig.axes)==1
        assert ax.get_title()
    assert before==set(plt.get_fignums())
    assert original=={p:p.read_bytes() for p in bundle.iterdir() if p.is_file()}


def test_cancelled_runs_are_not_listed(plots):
    plotter,bundle=plots
    (bundle.parent/'execution.json').write_text('{"status":"cancelled"}')
    assert plotter.panels('AL/codex/example')==[]


def test_multiple_runs_have_distinct_stable_panel_keys(plots):
    plotter,bundle=plots
    shutil.copytree(bundle.parent,bundle.parent.parent/'run-b')
    panels=plotter.panels('AL/codex/example')
    assert len(panels)==22 and len({p.key for p in panels})==22


def test_path_traversal_is_rejected(plots):
    with pytest.raises(ValueError):plots[0].panels('../private')


def test_missing_experiment_offers_no_panels(plots):
    assert plots[0].panels('AL/codex/absent')==[]


def test_batch_plotter_exports_panels_and_closes_figures(plots):
    import matplotlib.pyplot as plt
    plotter,bundle=plots
    before=set(plt.get_fignums())
    paths=plotter.run('AL/codex/example', {'panels':['likelihood','stage_policy','policy_timeline'], 'dpi':80})
    assert len(paths)==3 and all(p.is_file() and p.stat().st_size>1000 for p in paths)
    assert before==set(plt.get_fignums())
