# ThetaIDE

[![CI](https://github.com/ThetaIDE/ThetaIDE/actions/workflows/ci.yml/badge.svg)](https://github.com/ThetaIDE/ThetaIDE/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-pytest-blue)](https://github.com/ThetaIDE/ThetaIDE/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Formatter: ruff-format](https://img.shields.io/badge/formatter-ruff--format-261230)](https://github.com/ThetaIDE/ThetaIDE/blob/main/.github/workflows/ci.yml)
[![Types: mypy](https://img.shields.io/badge/types-mypy-blue)](https://github.com/ThetaIDE/ThetaIDE/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit)](./.pre-commit-config.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10--3.13-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23537094.svg)](https://doi.org/10.5281/zenodo.23537094)


https://github.com/user-attachments/assets/3469954e-a74a-4427-808d-d8af654cc8cb

ThetaIDE is a desktop IDE and experiment pipeline for reinforcement learning and deep learning research. You
compose experiments from Hydra configs, launch training on your machine, watch metrics stream in live, and
compare runs side by side. The pipeline is built on PyTorch Lightning and Hydra.

Algorithms, models and environments are components. A fresh clone ships the framework without them; you
install the ones you need from the **Theta Hub**, a registry that includes Stable-Baselines3 and d3rlpy suites,
MLP, CNN, ResNet, LSTM and Transformer models, Gymnasium and Atari environments, and neuro-symbolic models
such as BlendRL, NSFR and NEUMANN. You can also write your own: see the
[Component Plugin Authoring Guide](docs/COMPONENT_PLUGIN_GUIDE.md).

Training objectives can also be edited as Python components, with optional GUI parameters and
scores across repeated training seeds. See [Editable objectives](CUSTOM_OBJECTIVES.md).

---

## Install

Python 3.10–3.13 is required. Python 3.14 is not supported yet: Hydra 1.3 fails on it at startup.

```bash
git clone https://github.com/ThetaIDE/ThetaIDE.git Theta-IDE
cd Theta-IDE
python3 -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
pip install -e ".[dev,gui]"
```

[INSTALL.md](INSTALL.md) covers Windows, optional extras such as Atari, and how to verify the install.

## Run your first experiment

1. Start the IDE from the repository root:

   ```bash
   # macOS (automatically creates and opens the app bundle)
   venv/bin/python -m frontend

   # Windows/Linux:
   python -m frontend
   ```

2. In the **Components** tab, open the Hub (the globe button at the top) and install **ThetaIDE Basics**. It
   brings Stable-Baselines3, Gymnasium classic control and the basic network models.
3. In the **Experiment** tab, open `thetaide/cartpole_ppo_reference` and press **F5** to launch it.
4. Watch it train in the **Monitor** tab, then compare runs in **Results**.

Press **F1** for the built-in documentation, or **Ctrl+Space** to jump to any panel, plugin or command.

## Text editor

Enable **Text Editor** in **Settings → Core Plugins** to add the **Editor** panel and select it as
Theta's default text editor. **Settings → Text Editor** lets you switch back to the system editor
or change line wrapping. This preference controls files opened inside Theta, including `settings.toml`.

The editor has document tabs, New/Open/Save/Save As, undo/redo, find/replace, folding, and syntax
highlighting for Python, YAML, JSON, TOML, and Markdown. File-menu shortcuts act on the Editor panel
while it is selected. Closing a modified tab, disabling the plugin, or quitting prompts to save;
external changes are flagged before overwriting. CodeMirror 6 is bundled locally and works offline.
The initial version accepts UTF-8 files (with or without a BOM) up to 10 MiB and preserves consistent
LF, CRLF, or CR line endings. It reports binary files, other encodings, and mixed line endings instead
of silently converting them.

## Command line

The IDE and the command line run the same pipeline. Experiments are named `<group>/<experiment>`, after their
YAML file under `in/config/experiment/`:

```bash
# Train an experiment (install its components from the Hub first)
python run_pipeline.py thetaide/cartpole_ppo_reference

# Hydra overrides go after the name
python run_pipeline.py thetaide/cartpole_ppo_reference total_timesteps=2000 seed=1

# Validate the config without training
python run_pipeline.py thetaide/cartpole_ppo_reference dry_run=true
```

Each command-line run writes its results under the recipe's `experiment_id`, replacing the previous run with
that ID. Pass `experiment_id=<new name>` to keep both. (In the IDE, enable Timestamp experiment to give reruns a timestamped ID and keep previous results.)

`python run_pipeline.py --help` lists the orchestration overrides (`plot_only`, `no_plot`,
`no_online`, `no_offline` and more).

Component development also works without the desktop app:

```bash
python run_pipeline.py component inspect blendrl
python run_pipeline.py component copy blendrl my_blendrl
python run_pipeline.py component inspect my_blendrl
```

The copy is editable in the existing research directories. Use its returned profile paths with
`component config show`, `config set`, and `config validate`. Hub originals remain protected.
See [Component development](COMPONENT_DEVELOPMENT.md) for configuration, validation, export, and
coding-agent workflows. GUI and terminal operations share headless services.

### Component overrides in experiments

Open a group default or individual experiment in the **Experiment** panel and use the **Data Loaders**, **Evaluators**, and **Figures and Tables** boxes to choose components. Add **Figures and Tables** from **Add Overrides** to enable plotting; without a selection, the pipeline skips plotting. Use the plus beside the title to add an installed plotter, then edit its parameters in its card or use the card’s plus to add parameters. Selecting a component fills its first level of parameters from the installed profile; you can edit those defaults or add custom parameters. Inherited selectors show the resolved component name. Individual recipes inherit group values until they add an override. The same fields can be written directly in an experiment YAML:

```yaml
data_module:
  name: tabular_data
  params:
    batch_size: 32
evaluator:
  name: my_evaluator
  params:
    threshold: 0.8
plot_overrides:
  losses:
    smoothing_window: 7
```

`data_module` and `evaluator` also accept a registered name as a string. `plots` selects figures as a list or mapping; `plot_overrides` adjusts a figure's parameters without changing that selection. Existing `plots.<figure>` options still work. Evaluators can implement Lightning's `Callback`, the `evaluate(model, data_source, cfg)` protocol, or the evaluation hook methods `on_eval_start`, `evaluate_step`, and `on_eval_end`. Evaluator parameters can include `enabled: false`; `frequency: 2` runs evaluation hooks every second validation epoch.

Plots and reports are generated automatically at the end of a run. To regenerate them, pass the run's ID,
which is its folder name under `results/logs/<group>/`:

```bash
python plot/manager.py thetaide/cartpole_ppo_baseline --wipe
```

## Project layout

```
├── run_pipeline.py            # Command-line entry point: compose, validate, train, plot
├── frontend/                  # The desktop IDE (PyQt6); run it with python -m frontend
│   ├── plugins/core/          # Built-in plugins: documentation, TensorBoard, arXiv papers
│   └── hub/                   # Theta Hub GUI client and compatibility imports
├── src/
│   ├── app/                   # Framework
│   │   ├── components/        # Headless installer, component development and CLI
│   │   ├── workspace.py       # Shared paths for the existing checkout layout
│   │   ├── pipeline/          # Config composition, validation and orchestration
│   │   ├── core/              # Interfaces, registries, factories, callbacks
│   │   ├── services/          # Shared run catalog, exports, metrics and job store
│   │   ├── data/              # Offline replay data modules
│   │   └── train.py           # PyTorch Lightning training driver
│   ├── plugins/               # components/ and app/, each with bundled/ and installed/
│   └── usr/                   # User-authored implementations, environments and rules
├── in/
│   ├── config/                # Hydra configs: experiment/, agent/, model/, env/, paradigms/, site/, plot/
│   └── datasets/              # Local datasets (not tracked)
├── plot/                      # Compatibility imports/CLI for framework plotting
├── .thetaide/                 # IDE state, ownership records, retained packages and cache
├── tests/                     # pytest suite
├── docs/                      # Guides and the README media
└── results/                   # Run output (not tracked): logs/, plots/, checkpoints/, tensorboard/
```

## Documentation

- **In the IDE:** press **F1** or Help→Documentation.
- [INSTALL.md](INSTALL.md): setup, extras and verifying the install.
- [CONTRIBUTING.md](CONTRIBUTING.md): development workflow, tests and code style.
- [Component Plugin Authoring Guide](docs/COMPONENT_PLUGIN_GUIDE.md): writing algorithms, models,
  environments and IDE plugins.
- [Native provider storage](docs/PLUGIN_STORAGE.md): ownership, version 2 archives, workers and rollback.
- [Component development](COMPONENT_DEVELOPMENT.md): GUI/terminal copying, editing, validation and storage.
- [Pipeline interface](docs/API.md): shared services and desktop execution.
- [Archived workflow notes](docs/WORKFLOW_GUIDE.md): historical BlendRL commands; use the CLI section above for current commands.

## License

MIT. See [LICENSE.md](LICENSE.md).

ThetaIDE depends on PyQt6, which is GPL-3.0-only, so a *distributed* build
carries obligations the MIT licence alone does not cover. Every dependency and
its licence is listed in [THIRD_PARTY_LIBRARIES.md](THIRD_PARTY_LIBRARIES.md).

## Contributing

Coding standards and how to extend the system are in
[CONTRIBUTING.md](CONTRIBUTING.md). Participation is governed by our
[Code of Conduct](CODE-OF-CONDUCT.md).

Training jobs run in an independent local runner. Closing or quitting ThetaIDE leaves
active jobs and a started queue running; reopening the same workspace reconnects the
monitor and replays metrics and logs written while the desktop was closed. Use Stop
or the queue controls to cancel jobs or pause new launches explicitly. This keeps
jobs running across desktop exits, not machine shutdowns. Runner diagnostics are in
the job data directory under `runner/runner.log`.
