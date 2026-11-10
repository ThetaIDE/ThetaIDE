# Third-Party Libraries

Every dependency ThetaIDE declares, with the licence of the version actually
installed. Generated from `pyproject.toml` and the installed package metadata,
so the versions below are the ones this project was built and tested against.

To regenerate after changing dependencies, read the licences out of the
installed distributions rather than editing this table by hand.

## Licences that constrain how ThetaIDE may be distributed

ThetaIDE itself is MIT (see [LICENSE.md](../../LICENSE.md)), but three
dependencies are copyleft, and two of them are not optional:

| Package | Licence | Why it matters |
|---|---|---|
| PyQt6 | GPL-3.0-only | The GUI is built on it. Riverbank dual-licenses PyQt6: GPL-3.0, or a paid commercial licence. Distributing a binary that links it under GPL terms means the combined work is offered under the GPL. |
| PyQt6-WebEngine | GPL-3.0-only | Same terms; used by the embedded terminal and documentation views. |
| igraph | GPL | Used by the symbolic reasoning components. |
| tqdm | MPL-2.0 AND MIT | Weak copyleft, file-level. Unmodified use imposes no obligation on the rest of the project. |

This is a statement of the facts, not legal advice. The practical reading is
that running and developing ThetaIDE is unencumbered, while *distributing* a
packaged build needs either GPL-compatible terms or a commercial Qt licence.
Worth settling before any public release.

## Full dependency table

`Version in use` is what is installed in the development environment;
`Declared as` is the constraint in `pyproject.toml`.

### runtime

| Package | Version in use | License | Declared as |
|---|---|---|---|
| [captum](https://captum.ai) | 0.9.0 | BSD-3-Clause | `captum>=0.7` |
| [gymnasium](https://farama.org) | 1.3.0 | MIT License | `gymnasium[atari,accept-rom-license]>=0.28` |
| [hydra-core](https://github.com/facebookresearch/hydra) | 1.3.7 | MIT | `hydra-core>=1.3,<2` |
| [hydra-optuna-sweeper](https://github.com/facebookresearch/hydra/) | 1.1.2 | UNKNOWN | `hydra-optuna-sweeper<=1.1.2` |
| [igraph](https://igraph.org/python) | 1.0.0 | GNU General Public License (GPL) | `igraph>=0.10` |
| [kneed](https://github.com/arvkevi/kneed) | 0.8.6 | see project | `kneed>=0.8` |
| [lark](https://github.com/lark-parser/lark) | 1.3.1 | MIT | `lark>=1.1` |
| [lightning](https://github.com/Lightning-AI/lightning) | 2.6.6 | Apache-2.0 | `lightning>=2.0,<3` |
| [matplotlib](https://matplotlib.org) | 3.11.2 | Python Software Foundation License | `matplotlib>=3.7,<4` |
| [numpy](https://numpy.org) | 2.4.6 | see project | `numpy>=1.24,<3` |
| [omegaconf](https://github.com/omry/omegaconf) | 2.3.1 | BSD License | `omegaconf>=2.3,<3` |
| [optuna](https://optuna.org/) | 4.9.0 | MIT License | `optuna>=3.0,<5` |
| [pandas](https://pandas.pydata.org) | 2.3.3 | BSD License | `pandas>=2.0,<3` |
| [pillow](https://tidelift.com/subscription/pkg/pypi-pillow?utm_source=pypi-pillow&utm_medium=pypi) | 12.3.0 | MIT-CMU | `pillow>=9.0` |
| [pytorch-lightning](https://github.com/Lightning-AI/lightning) | 2.6.6 | Apache-2.0 | `pytorch-lightning>=2.0,<3` |
| [pyyaml](https://pyyaml.org/) | 6.0.3 | MIT | `pyyaml>=6.0` |
| [scikit-learn](https://scikit-learn.org) | 1.9.1 | BSD-3-Clause | `scikit-learn>=1.2` |
| [scipy](https://scipy.org/) | 1.17.1 | BSD License | `scipy>=1.10` |
| [seaborn](https://github.com/mwaskom/seaborn) | 0.13.2 | BSD License | `seaborn>=0.12` |
| [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | 2.9.0 | MIT | `stable-baselines3>=2.0` |
| [tabulate](https://github.com/astanin/python-tabulate) | 0.10.0 | MIT | `tabulate>=0.9` |
| [tensorboard](https://github.com/tensorflow/tensorboard) | 2.21.0 | Apache 2.0 | `tensorboard>=2.12` (`tensorboard` extra) |
| [torch](https://pytorch.org) | 2.14.0 | see project | `torch>=2.0,<3` |
| [torchaudio](https://github.com/pytorch/audio) | 2.11.0 | BSD License | `torchaudio>=2.0,<3` |
| [torchvision](https://github.com/pytorch/vision) | 0.29.0 | BSD | `torchvision>=0.15,<1` |
| [tqdm](https://tqdm.github.io) | 4.70.1 | MPL-2.0 AND MIT | `tqdm>=4.60` |

### dev

| Package | Version in use | License | Declared as |
|---|---|---|---|
| [mypy](https://www.mypy-lang.org/) | 2.3.1 | MIT | `mypy>=1.0` |
| [pre-commit](https://github.com/pre-commit/pre-commit) | 4.6.2 | MIT | `pre-commit>=3.0` |
| [pytest](https://docs.pytest.org/en/latest/) | 9.1.1 | MIT | `pytest>=7.0` |
| [pytest-cov](https://github.com/pytest-dev/pytest-cov) | 7.1.0 | MIT | `pytest-cov>=4.0` |
| [pytest-timeout](https://github.com/pytest-dev/pytest-timeout) | 2.4.0 | MIT | `pytest-timeout>=2.3` |
| [ruff](https://docs.astral.sh/ruff) | 0.16.9 | MIT | `ruff>=0.5` |
| [types-PyYAML](https://github.com/python/typeshed) | 6.0.12.20260906 | Apache-2.0 | `types-PyYAML>=6.0` |

### gui

| Package | Version in use | License | Declared as |
|---|---|---|---|
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | 6.11.0 | GPL-3.0-only | `PyQt6>=6.7,<7` |
| [PyQt6-WebEngine](https://www.riverbankcomputing.com/software/pyqtwebengine/) | 6.11.0 | GPL-3.0-only | `PyQt6-WebEngine>=6.7,<7` |
| [arxiv](https://github.com/lukasschwab/arxiv.py) | 4.0.1 | MIT | `arxiv>=2.1` |
| [psutil](https://github.com/giampaolo/psutil) | 7.2.2 | BSD-3-Clause | `psutil>=5.9` |
| pywinpty | — | not installed locally | `pywinpty>=2.0; sys_platform == 'win32'` |
| [truststore](https://github.com/sethmlarson/truststore) | 0.10.4 | MIT | `truststore>=0.9` |

### atari

| Package | Version in use | License | Declared as |
|---|---|---|---|
| hackatari | — | not installed locally | `hackatari @ git+https://github.com/k4ntz/HackAtari.git` |
| ocatari | — | not installed locally | `ocatari` |

### video

| Package | Version in use | License | Declared as |
|---|---|---|---|
| opencv-python | — | not installed locally | `opencv-python` |
| pygame | — | not installed locally | `pygame` |
| vidmaker | — | not installed locally | `vidmaker` |

### all

| Package | Version in use | License | Declared as |
|---|---|---|---|
| nesyrl | 0.1.0 | MIT | `nesyrl[dev,gui,atari,video]` |

## Notes on the unclear rows

- **hydra-optuna-sweeper** reports `UNKNOWN`: the package ships no licence
  metadata. The Hydra project it belongs to is MIT.
- **kneed**, **numpy**, **torch** report `see project`: their metadata points
  at a bundled licence file rather than naming an SPDX identifier. All three
  are permissive (BSD-3-Clause for numpy and torch, BSD-3-Clause for kneed).
