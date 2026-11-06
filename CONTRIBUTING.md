# Contributing to ThetaIDE

Thanks for helping out. This guide covers setting up a development environment, the checks your change has to
pass, and where to start for the most common kinds of contribution.

## Set up

Follow [INSTALL.md](INSTALL.md), cloning your fork instead of the main repository, then install the git hooks:

```bash
git clone https://github.com/<your-username>/Theta-IDE.git
cd Theta-IDE
python3 -m venv venv                      # Python 3.10–3.13
source venv/bin/activate
pip install -e ".[dev,gui]"
pre-commit install
```

Hub components are optional for core development. Install a component only when working on an integration that
requires it.

## Workflow

1. **Branch from `main`** with a descriptive name, such as `feat/results-export` or `fix/queue-shortcut`.
2. **Add or update tests** in `tests/` for what you change.
3. **Run the checks below** before you push.
4. **Open a pull request against `main`.** CI runs lint, format, type and test checks on every pull request,
   and they must pass before merge.

## Checks

CI runs these. Run the same ones locally:

```bash
# Lint and format (CI checks src/, plot/ and run_pipeline.py)
ruff check src/ plot/ run_pipeline.py
ruff format --check src/ plot/ run_pipeline.py

# Type check
mypy src/app/pipeline/ src/usr/algorithms/agent_registry.py

# Tests, as CI runs them
QT_QPA_PLATFORM=offscreen pytest tests/ --timeout=120 --cov=src --cov=frontend
```

The pre-commit hooks run ruff (with `--fix`), ruff-format and mypy on the files in each commit. Much of the
codebase is not ruff-formatted yet, so a commit can reformat the whole of a file you only touched in one place.
If that happens, commit the formatting on its own (`style: ruff format <file>`) before your change, so reviewers
can tell the two apart.

Don't run `ruff format .` on the whole repository: it reformats a large part of `frontend/` and buries your
change in an unrelated diff.

### Running tests

```bash
pytest tests/test_launcher.py           # one file
pytest tests/ -k hotkeys                # tests whose name matches
pytest tests/ -m "not slow"             # skip tests that train for real
```

GUI tests need a display. On a headless machine, set `QT_QPA_PLATFORM=offscreen`. Optional integration tests
may require Hub components or datasets; use the test's reported skip/failure reason. Core development uses
isolated fixtures rather than a fixed set of private researcher files.

## Where things live

```
run_pipeline.py          # Command-line entry point
frontend/                # The desktop IDE (PyQt6)
  plugins/core/          # Built-in IDE plugins: documentation, TensorBoard, arXiv papers
  hub/                   # Theta Hub GUI client and compatibility imports
src/app/                 # Framework
  components/            # Headless installer, development operations, config I/O and CLI
  workspace.py           # Shared paths for the existing checkout layout
  pipeline/              # Config composition, validation and orchestration
  core/                  # Interfaces, registries (agents, plotters, ...), factories, callbacks
  services/              # Run catalog, metrics and job store used by the IDE
  data/                  # Offline replay data modules
src/usr/                 # Components: algorithms, methods, models, environments, evaluators
in/config/               # Hydra configs: experiment/, agent/, model/, env/, paradigms/, plot/
plot/                    # Plotting framework and component-supplied research plotters
tests/                   # pytest suite
```

Most runtime files under `src/usr/`, `in/config/agent/`, `in/config/model/` and `in/config/env/` are installed
from the Theta Hub and gitignored. Component authors can develop packages separately and publish them to the
Hub.

## Common contributions

### A new algorithm, model or environment

Components are packaged and installed through the Theta Hub. The
[Component Plugin Authoring Guide](docs/COMPONENT_PLUGIN_GUIDE.md) covers the protocols each kind implements
(section 2.2 for algorithms) and how to export one as a Hub package. In the IDE, **File → Scaffold New Component
Plugin…** (Ctrl+Shift+P) generates a starting point.

### A new plotter

Plotters are discovered automatically from modules in `plot/` and its subfolders, such as
`plot/standard_figures/`:

1. Subclass `BasePlotter` from `plot/base.py`, passing the plotter's name to `super().__init__("my_plot")`.
2. Implement `run(self, exp_id, cli_overrides=None)`. Call `self.get_effective_config(exp_id, cli_overrides)`
   to get the merged settings, the experiment's group and the output folder.
3. Put the default settings in `in/config/plot/<library>/my_plot.yaml`. Research plotters and their defaults
   are supplied by components; do not assume a particular losses/convergence package is installed.
4. Experiments request the plotter by name under their `plots:` section, which can also override its settings.

### An IDE feature or plugin

The IDE lives in `frontend/`. Self-contained features (a new sidebar tab, for example) work best as plugins:
see `frontend/plugins/core/` for working examples, and the **Authoring Guide: All Component & Plugin Types**
article in the built-in documentation (press **F1**).

If your change affects what users see or do, update the built-in documentation in
`frontend/plugins/core/documentation/content.py` in the same pull request.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/), with a scope where it helps:

- `feat(results): export the comparison table as CSV`
- `fix(hotkeys): stop two menu actions sharing Ctrl+Shift+R`
- `docs: rewrite the install guide`
- `test(queue): cover pausing an empty queue`
- `chore: bump PyQt6`

## Code style

- Formatter and linter: [Ruff](https://github.com/astral-sh/ruff), configured in `pyproject.toml`, with
  120-character lines.
- Type checker: [mypy](https://mypy-lang.org/), configured in `pyproject.toml`.
- Python 3.10+ syntax. Code must also run on 3.10, so avoid features newer than that.
