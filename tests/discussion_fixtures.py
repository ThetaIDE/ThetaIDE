"""The Discussion tests' workspace: the base workspace plus ``tests/fixtures/discussion``.

``tests/fixtures/discussion`` is laid out like ``src/usr``: an experiment tree (``figs`` with a
``sweeps`` subfolder, and ``other``), the ``fixture_figures`` plot component (``fixture_curve``
provides ``figure/vega-lite-v1``; ``fixture_table`` does not), run metrics for
``figs/online`` and one saved figure with its discussion text.
"""

import shutil
from pathlib import Path

DISCUSSION_FIXTURE = Path(__file__).parent / "fixtures" / "discussion"


def build_discussion_workspace(root):
    from conftest import base_workspace
    root = Path(root)
    shutil.copytree(base_workspace(), root, dirs_exist_ok=True)
    shutil.copytree(DISCUSSION_FIXTURE, root / "src/usr", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return root


SEEDED_RUN = "figs/online/20261117-090000"


def write_seeded_runs(paths, target=SEEDED_RUN):
    """A saved run with three seeds of ppo and a2c (rewards and losses) and a two-parameter sweep.

    Written into a test's own workspace copy; returns the target. Values are deterministic:
    ppo improves faster than a2c, and seeds differ by a fixed offset.
    """
    import json
    import math
    logs = paths.results / "logs" / target
    for method, rate in (("ppo", 1.0), ("a2c", 0.6)):
        for seed in range(3):
            folder = logs / method / f"version_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            lines = ["step,transitions,train/reward,eval/reward,losses/policy_loss,losses/value_loss"]
            for step in range(1, 41):
                reward = rate * step + seed - 1
                evaluation = f"{reward + 0.5:.3f}" if step % 4 == 0 else ""
                lines.append(f"{step},{step * 8},{reward:.3f},{evaluation},{math.exp(-step / 10) + 0.01 * seed:.5f},"
                             f"{2.0 / step + 0.02 * seed:.5f}")
            (folder / "metrics.csv").write_text("\n".join(lines) + "\n")
    (logs / "config.yaml").write_text(
        f"group: figs\nexperiment_id: {target.split('/', 1)[1]}\nparadigm: online_rl\n"
        "methods:\n  ppo:\n    agent: sb3-dev/ppo\n    model: basics/dnn\n"
        "  a2c:\n    agent: sb3-dev/a2c\n    model: basics/dnn\n"
        "tuning:\n  component: fixture_tuner\n  objective: eval/reward\n")
    trials = []
    for index, (rate, gamma) in enumerate([(1e-4, 0.9), (3e-4, 0.95), (1e-3, 0.99), (3e-3, 0.9), (1e-2, 0.95)]):
        trials.append({"id": index, "state": "complete", "value": 100 - abs(math.log10(rate) + 3) * 20 + gamma,
                       "parameters": {"learning_rate": rate, "gamma": gamma}, "intermediate_values": {}})
    trials.append({"id": 5, "state": "pruned", "value": None, "parameters": {"learning_rate": 5e-2, "gamma": 0.9},
                   "intermediate_values": {"0": 12.0, "1": 15.5}})
    summary = {"schema_version": 1, "study": "ppo_study", "component": "fixture_tuner", "direction": "maximize",
               "objective_metric": "eval/reward", "trials": trials}
    study = paths.results / "checkpoints" / target / "ppo" / "tuning_summary.json"
    study.parent.mkdir(parents=True, exist_ok=True)
    study.write_text(json.dumps(summary))
    return target
