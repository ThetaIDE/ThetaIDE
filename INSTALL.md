# Installing ThetaIDE

This guide sets up the training pipeline and the desktop IDE in one virtual environment, then checks that
both work. For a tour of the IDE once it is running, press **F1** inside it.

---

## 1. Requirements

- **Python 3.10, 3.11, 3.12 or 3.13.** Python 3.14 is not supported yet: Hydra 1.3 fails on it with
  `ValueError: badly formed help string` as soon as training starts. If your system Python is 3.14 (current
  Arch and Fedora), install an older one with your package manager, `pyenv` or `uv` (see below).
- **Git.**
- **Disk space for the environment:** about 7 GB with the default CUDA build of PyTorch on Linux, or about 2 GB
  with the [CPU-only build](#cpu-only-pytorch).
- **On a minimal or headless Linux system**, the Qt libraries PyQt6 needs at import time. On Ubuntu 24.04:

  ```bash
  sudo apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3 libnss3 libxcomposite1 \
      libxdamage1 libxrandr2 libxtst6 libxkbfile1 libgbm1 libasound2t64
  ```

  Desktop Linux installs usually have these already.

---

## 2. Install

### With venv and pip

```bash
git clone https://github.com/ThetaIDE/ThetaIDE.git Theta-IDE
cd Theta-IDE
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,gui]"
```

On Windows (PowerShell):

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev,gui]"
```

### With uv

[uv](https://docs.astral.sh/uv/) can also download a supported Python for you, which helps when your system
Python is 3.14:

```bash
git clone https://github.com/ThetaIDE/ThetaIDE.git Theta-IDE
cd Theta-IDE
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -e ".[dev,gui]"
```

### Extras

| Extra | Adds |
|---|---|
| `gui` | The desktop IDE: PyQt6, PyQt6-WebEngine, psutil, pywinpty (Windows), and `arxiv` and `truststore` for the arXiv panel |
| `dev` | pytest (with coverage and timeouts), ruff, mypy, pre-commit, and the Optuna test dependencies |
| `optuna` | Optuna hyperparameter tuning and the Hydra Optuna adapter |
| `video` | pygame, vidmaker and OpenCV for rendering and recording |
| `tensorboard` | TensorBoard for the optional built-in viewer plugin |
| `all` | All of the above |

The IDE composes and validates configurations in-process, then launches training in child processes.
The pipeline and the `gui` extra use the same environment. There is no separate frontend environment or
server. Atari-specific dependencies are provided by the relevant Hub components, not a project `atari` extra.

### CPU-only PyTorch

On Linux, pip installs the CUDA build of PyTorch by default. Without an NVIDIA GPU,
install the CPU build instead:

```bash
pip install -e ".[dev,gui]" --extra-index-url https://download.pytorch.org/whl/cpu
```

With uv, also pass `--index-strategy unsafe-best-match`.

---

## 3. Launch the IDE

From the repository root, with the environment active:

```bash
# macOS: automatically creates and opens the local app bundle
venv/bin/python -m frontend

# Windows/Linux:
python -m frontend
```

There is no server to start first. The status bar shows **Pipeline ready** once the training pipeline has
loaded. On macOS, the command automatically launches the local app bundle so the Dock and Mission Control
associate the window with ThetaIDE instead of the Python interpreter.

ThetaIDE stores local IDE state under `.thetaide/` by default: settings in `.thetaide/settings.toml`,
run records and the job queue under `.thetaide/runs/`, Hub cache in `.thetaide/cache/hub/`, and retained
component packages/context under `.thetaide/components/`. Active component source, YAMLs, datasets, and
experiment results also live elsewhere in the checkout; `.thetaide/` alone is not a complete research backup.
See [Component development](COMPONENT_DEVELOPMENT.md) for the storage map.

The existing `--data-dir` option selects application data paths; it does not migrate existing research files
or establish an independent research-workspace lifecycle. User-global preferences live in
`~/.config/thetaide/settings.toml`.

---

## 4. Install components

A fresh clone supplies the framework and built-in app support. Install research algorithms, models and
environments from the Theta Hub:

1. Open the **Components** tab and click the globe button at the top to open the Hub.
2. Install **ThetaIDE Basics**. It brings Stable-Baselines3, d3rlpy, Gymnasium classic control, Atari and the
   basic network models.

Managed components install into `src/plugins/components/installed`, app extensions into `src/plugins/app/installed`, and active profiles into `in/config`. `src/usr` is reserved for authored implementations. See [provider storage](docs/PLUGIN_STORAGE.md). The
README's [Run your first experiment](README.md#run-your-first-experiment) section continues from here.

Hub-installed files are read-only; make an independent model copy to change its implementation, or copy
its YAML profile to configure shared code. Both GUI and terminal workflows are described in
[Component development](COMPONENT_DEVELOPMENT.md). Uninstall and updates remain available through the Hub.

---

## 5. Verify the install

Core development tests use isolated fixtures; optional integration/training tests may require installed
components and datasets. Run:

```bash
pytest tests/ -m "not slow"
```

The GUI tests need a display. On a headless machine, run them with `QT_QPA_PLATFORM=offscreen`. Tests marked
`slow` may train for real; leave out `-m "not slow"` to include them. Skip conditions and dependencies vary by
test. Use the reported skip/failure reason when diagnosing a result rather than assuming a fixed failure count.

---

## 6. Troubleshooting

**The status bar shows "Pipeline unavailable".** The IDE is not running from the environment the pipeline
is installed in. Activate it, reinstall with `pip install -e ".[dev,gui]"`, and start the IDE again.

**Training fails with `ValueError: badly formed help string`.** You are on Python 3.14. Recreate the
environment with Python 3.10–3.13.

**A run fails because an agent, model or environment is missing.** Install it from the Hub (step 4). The
experiment's recipe names what it needs under `methods:`.

**F1 does not open the documentation.** Your desktop or window manager has claimed F1 (tiling window managers
and some laptops' media keys do). Rebind it in **Settings → Hotkeys**, or use **Help → Documentation**.

**`pip` warns that gymnasium has no extra `accept-rom-license`.** This is harmless: Gymnasium 1.x ships the
Atari ROMs without it.
