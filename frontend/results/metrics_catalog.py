"""Display information for metric columns, with a fallback for any name an engine invents.

Nothing here decides *which* metrics exist: the Results panel shows every numeric column a
run logged. The catalog only makes known metrics read better (label, format, which direction
is good). Unknown metrics still get a readable label and a sensible format.
"""

from __future__ import annotations

from dataclasses import dataclass

# Columns usable as an x-axis, in order of preference.
AXIS_COLUMNS = ("transitions", "step", "epoch", "time/total", "training_time_seconds")

# Pure counters: only ever an x-axis, never plotted as a metric. (Time columns are both.)
COUNTER_COLUMNS = ("transitions", "step", "epoch")

# Order in which metric groups (the part before the first '/') are listed.
GROUP_ORDER = ("eval", "train", "losses", "diagnostics", "val", "test", "time")


@dataclass(frozen=True)
class MetricInfo:
    name: str
    label: str
    fmt: str = ".4g"
    higher_is_better: bool | None = None
    description: str = ""

    @property
    def group(self) -> str:
        return metric_group(self.name)


_KNOWN: dict[str, MetricInfo] = {}


def _add(name, label, fmt=".4g", higher_is_better=None, description=""):
    _KNOWN[name] = MetricInfo(name, label, fmt, higher_is_better, description)


# Evaluation and training reward
_add("eval/reward", "Episode reward", ".1f", True, "Mean episode reward over evaluation episodes")
_add("eval/reward_std", "Episode reward (std)", ".2f", None, "Spread of evaluation episode rewards")
_add("train/reward", "Training reward", ".1f", True, "Mean episode reward while collecting experience (SB3: most recent 100 episodes)")
_add("train/episode_reward", "Training episode reward", ".1f", True, "Reward for each completed training episode")
_add("train/episode_length", "Training episode length", ".1f", None, "Length of each completed training episode")
_add("train/length", "Training length (rolling mean)", ".1f", None, "Mean length over the most recent 100 training episodes")
# Common training metrics
_add("losses/total_loss", "Total loss", ".4f", False, "Combined training objective")
_add("losses/policy_loss", "Policy loss", ".4f", False, "Policy optimization objective")
_add("losses/value_loss", "Value loss", ".3f", False, "Value-function error (critic MSE)")
_add("losses/entropy", "Policy entropy", ".3f", None, "How random actions are")
_add("losses/approx_kl", "Approximate KL", ".5f", None, "Size of each policy update")
# Offline RL and supervised validation
_add("train/loss", "Training loss", ".4f", False)
_add("val/loss", "Validation loss", ".4f", False)
_add("val/loss_std", "Validation loss (std)", ".4f", None)
_add("val/precision", "Precision", ".3f", True)
_add("val/recall", "Recall", ".3f", True)
_add("val/f", "F-score", ".3f", True)
# Timing
_add("time/total", "Total time", ".1f", None, "Seconds since training started")
_add("time/train", "Training time", ".1f", None, "Seconds spent training")
_add("time/eval", "Evaluation time", ".1f", None, "Seconds spent evaluating")
_add("training_time_seconds", "Training time", ".1f", None)

_LOWER_IS_BETTER = ("loss", "error", "regret", "mse", "mae", "rmse")
_HIGHER_IS_BETTER = ("reward", "return", "accuracy", "acc", "precision", "recall", "auroc", "auc", "f1", "score")


def metric_group(name: str) -> str:
    """The part before the first '/', or 'other' for unprefixed names."""
    return name.split("/", 1)[0] if "/" in name else "other"


def describe(name: str) -> MetricInfo:
    """Display info for any metric column name."""
    if name in _KNOWN:
        return _KNOWN[name]
    leaf = name.rsplit("/", 1)[-1]
    label = leaf.replace("_", " ").strip().capitalize() or name
    words = leaf.lower().replace("-", "_").split("_")
    higher = None
    if any(w in _LOWER_IS_BETTER for w in words):
        higher = False
    elif any(w in _HIGHER_IS_BETTER for w in words):
        higher = True
    return MetricInfo(name, label, ".4g", higher)


def group_sort_key(group: str) -> tuple[int, str]:
    return (GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER), group)
