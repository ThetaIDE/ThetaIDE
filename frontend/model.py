"""Portable run records. This module does not import the training backend."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
from uuid import uuid4

# Backend experiment every builder config is composed from (in/config/experiment/thetaide/_base.yaml).
BASE_EXPERIMENT = "thetaide/_base"
METHOD = "method"


def _num(value):
    """Format a float so both the Hydra override grammar and YAML read it back as a float."""
    text = repr(float(value))
    if "e" in text and "." not in text.split("e")[0]:
        mantissa, exponent = text.split("e")
        text = f"{mantissa}.0e{exponent}"
    return text


@dataclass
class Config:
    name: str = "experiment"
    env: str = ""
    method_name: str = METHOD
    agent: str = ""
    model: str = ""
    seed: int = 42
    total_timesteps: int = 10000
    lr: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    tensorboard: bool = False

    def overrides(self):
        """Hydra overrides applied to BASE_EXPERIMENT, exactly as typed after `run_pipeline.py`."""
        method = f"++methods.{self.method_name}"
        overrides = [f"++experiment_id='{self.name}'", f"seed={self.seed}", f"total_timesteps={self.total_timesteps}"]
        if self.agent:
            overrides.append(f"{method}.agent={self.agent}")
        if self.model:
            overrides.append(f"{method}.model={self.model}")
        overrides.extend([f"{method}.lr={_num(self.lr)}", f"{method}.batch_size={self.batch_size}",
                          f"{method}.gamma={_num(self.gamma)}", f"tensorboard={str(self.tensorboard).lower()}"])
        return overrides

    def command(self):
        return " ".join(["python", "run_pipeline.py", BASE_EXPERIMENT, *self.overrides()])

    def recipe_yaml(self):
        """A standalone experiment recipe equivalent to BASE_EXPERIMENT + overrides()."""
        return (
            "# @package _global_\n"
            "# Exported from ThetaIDE. Save under in/config/experiment/thetaide/ and run:\n"
            "#   python run_pipeline.py thetaide/<file name without .yaml>\n"
            f"defaults:\n- {BASE_EXPERIMENT}\n\n"
            f"experiment_id: {json.dumps(self.name)}\n"
            f"seed: {self.seed}\ntotal_timesteps: {self.total_timesteps}\n"
            f"tensorboard: {str(self.tensorboard).lower()}\n\n"
            f"methods:\n  {self.method_name}:\n"
            + (f"    agent: {self.agent}\n" if self.agent else "")
            + (f"    model: {self.model}\n" if self.model else "")
            + f"    lr: {_num(self.lr)}\n    batch_size: {self.batch_size}\n    gamma: {_num(self.gamma)}\n"
        )


def sample(config, index):
    rng = random.Random(config.seed * 101 + index)
    progress = index / 60
    reward = max(0, 1 - math.exp(-3.6 * progress) + rng.uniform(-0.04, 0.04))
    return {"step": round(config.total_timesteps * progress),
            "reward": round(reward, 2),
            "loss": round(0.8 * math.exp(-4 * progress) + rng.uniform(0.015, 0.055), 4)}


def new_run(config, simulated=True):
    return {"id": uuid4().hex[:10], "created": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "status": "running", "simulated": simulated,
            "metrics": [], "notes": ""}


# Statuses of a backend training job that is still in progress, and those that are final.
LIVE_STATUSES = ("starting", "running")
FINAL_STATUSES = ("completed", "failed", "cancelled", "error", "interrupted")


# Run-record metric keys and the metrics.csv columns they come from.
METRIC_COLUMNS = {
    "reward": "eval/reward",
    "reward_std": "eval/reward_std",
    "loss": "losses/total_loss",
    "policy_loss": "losses/policy_loss",
    "value_loss": "losses/value_loss",
    "entropy": "losses/entropy",
    "approx_kl": "losses/approx_kl",
}


# metrics.csv counters: a point's position ("step"), never a metric of their own.
COUNTER_COLUMNS = ("transitions", "step", "epoch")


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def position_column(rows):
    """The column that places a point on the x-axis, chosen once for the whole run.

    `transitions` wherever a run logs it, which is every RL run. The choice is
    made per run rather than per row on purpose: Lightning writes an epoch-level
    duplicate of each row *without* transitions, and those duplicates have to
    stay out of the chart, so falling back row by row would admit them and
    double every RL curve.

    A supervised or unsupervised run logs no transitions anywhere - there is no
    environment to step through - so it falls back to step, then epoch, and its
    curves become plottable without changing anything for RL.
    """
    for name in COUNTER_COLUMNS:
        if any(_number(row.get(name)) is not None for row in rows):
            return name
    return None


def metric_points(rows):
    """Convert metrics.csv rows into chart points keyed by the run's position column.

    Every numeric column is kept, whatever runtime logged it: the METRIC_COLUMNS under their
    short run-record keys, and any other column under its own metrics.csv name. Evaluation rows carry eval/*
    and training rows carry losses/*; a point keeps None for the short keys its row does not
    report and simply lacks other columns it does not report.
    """
    rows = list(rows)
    position = position_column(rows)
    if position is None:
        return []
    aliased = set(METRIC_COLUMNS.values())
    points = []
    for row in rows:
        if _number(row.get(position)) is None:
            continue
        point = {key: _number(row.get(column)) for key, column in METRIC_COLUMNS.items()}
        extra = {name: value for name, value in row.items()
                 if name not in aliased and name not in COUNTER_COLUMNS and _number(value) is not None}
        if all(value is None for value in point.values()) and not extra:
            continue
        points.append({"step": round(row[position]), **point, **extra})
    return points


def metric_points_from_api(rows):
    """metric_points() for the rows of catalog.run_metrics, whose values arrive as text."""
    numeric_rows = []
    for row in rows:
        numeric = {}
        for key, value in row.items():
            try:
                numeric[key] = float(value)
            except (TypeError, ValueError):
                pass
        numeric_rows.append(numeric)
    return metric_points(numeric_rows)


def available_metrics(run):
    """Metric keys with at least one value in the run (demo runs only have reward and loss)."""
    if run.get("_metrics_deferred"):
        return set(run["_metric_summary"]["latest"]) - {"step"}
    return {key for m in run["metrics"] for key, value in m.items() if key != "step" and value is not None}


def latest(run, key):
    """The most recent non-missing value of a metric, or None."""
    if run.get("_metrics_deferred"):
        return run["_metric_summary"]["latest"].get(key)
    return next((m[key] for m in reversed(run["metrics"]) if m.get(key) is not None), None)


# ── Results browser ──────────────────────────────────────────────────────────

RESULT_COLUMNS = ("Experiment", "Started", "Method", "Environment", "Seed", "Timesteps", "Duration", "Status",
                  "Best reward", "Latest reward", "Source")
SOURCE_LABELS = {"app": "Trained (app)", "cli": "Command line", "simulated": "Simulated", "example": "Example"}
# Results browser filters: key -> label shown in the dropdown.
SOURCE_FILTERS = {"all": "All sources", "real": "Trained (app + command line)", "app": "Trained in the app",
                  "cli": "Command line", "simulated": "Simulated & examples"}
STATUS_FILTERS = {"all": "All statuses", "completed": "Completed", "active": "Queued or running",
                  "stopped": "Failed or stopped"}
STATUS_GROUPS = {"completed": ("completed",), "active": ("queued", "starting", "running"),
                 "stopped": ("failed", "error", "cancelled", "interrupted", "incomplete", "stopped")}


def run_name(run):
    """The name a run is listed under: the experiment ID the backend wrote results to, or the demo name."""
    return run["config"]["name"] if run["simulated"] else run["backend"]["experiment_id"]


def run_source(run):
    """Where a run came from: "app" (trained from the GUI), "cli" (found in results/logs), "simulated" or "example"."""
    if run["status"] == "example":
        return "example"
    if run["simulated"]:
        return "simulated"
    return "cli" if run.get("origin") == "cli" else "app"


def disk_run_id(group, experiment_id):
    """Record ID for a results/logs folder; safe as a run-store folder name."""
    return "cli-" + re.sub(r"[^A-Za-z0-9_-]", "_", f"{group}-{experiment_id}")


def _value(value, kind, fallback):
    try:
        return fallback if value is None else kind(value)
    except (TypeError, ValueError):
        return fallback  # e.g. an unresolved ${...} interpolation in a saved config


def remote_run_is_active(run):
    remote = run.get('remote') or {}
    from src.app.sites.result_state import results_available
    if not remote or results_available(remote):
        return False
    if remote.get('status_error'):
        return False
    state = remote.get('scheduler_state', remote.get('state', run.get('status')))
    return state in ('prepared', 'submitting', 'queued', 'pending', 'configuring',
                     'running', 'completing', 'suspended', 'requeued', 'requeue_hold')


def _remote_status(entry):
    from src.app.sites.execution import TERMINAL
    from src.app.sites.result_state import results_complete

    remote = entry['remote']
    value = remote.get('state', 'unknown')
    if results_complete(remote):
        if value not in TERMINAL:
            value = remote.get('scheduler_state')
        if value not in TERMINAL:
            return 'completed'
    elif value not in ('completed', 'failed', 'cancelled', 'submission_failed'):
        if remote.get('status_error'):
            value = 'unknown'
        elif 'scheduler_state' in remote:
            value = remote['scheduler_state']
    if value in ('configuring', 'requeued', 'requeue_hold'): value = 'pending'
    if value in ('completing', 'suspended'): value = 'running'
    if value in ('pending', 'prepared', 'submitting'): return 'queued'
    if value in ('timeout', 'out_of_memory', 'node_fail', 'preempted', 'boot_fail',
                 'submission_failed', 'deadline', 'revoked'): return 'failed'
    if value in ('completed', 'failed', 'cancelled', 'running'): return value
    return 'incomplete'


def disk_run(entry):
    """Run record for a results/logs folder that no app record links to, such as a command-line run.

    `entry` is one item of catalog.list_runs(). The record is `transient`: it is rebuilt from each listing
    and only written to the run store once something of the user's (such as notes) is attached.
    """
    cfg = entry.get("config") or {}
    methods = cfg.get("methods") or {}
    default = Config()
    method_name, method = next(iter(methods.items()), (METHOD, {}))
    if not isinstance(method, dict):
        method = {}
    config = Config(name=entry["experiment_id"], env=_value(cfg.get("env"), str, default.env),
                    method_name=str(method_name), agent=str(method.get("agent") or ""),
                    model=str(method.get("model") or ""),
                    seed=_value(cfg.get("seed"), int, default.seed),
                    total_timesteps=_value(cfg.get("total_timesteps"), int, default.total_timesteps),
                    lr=_value(method.get("lr"), float, default.lr),
                    batch_size=_value(method.get("batch_size"), int, default.batch_size),
                    gamma=_value(method.get("gamma"), float, default.gamma), tensorboard=False)
    agents = entry.get("agents") or []
    return {
        "id": disk_run_id(entry["group"], entry["experiment_id"]),
        "created": (agents[0].get("started") if agents else None) or "",
        "config": asdict(config), "status": _remote_status(entry) if entry.get('remote') else ("completed" if entry.get("finished") else "incomplete"),
        **({'remote': entry['remote']} if entry.get('remote') else {}),
        "simulated": False, "metrics": [], "notes": "", "origin": "cli", "transient": True,
        "builder_compatible": len(methods) == 1,
        "backend": {"group": entry["group"], "experiment_id": entry["experiment_id"],
                    "agents": [agent["name"] for agent in agents], "total_timesteps": cfg.get("total_timesteps")},
    }


def _timestamp(iso):
    try:
        return datetime.fromisoformat(iso).timestamp() if iso else None
    except ValueError:
        return None


def format_duration(seconds):
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


def result_row(run, disk=None):
    """(text, sort key) for each RESULT_COLUMNS cell of a run.

    `disk` is the run's catalog.list_runs() entry, if one was listed: it supplies timing for
    every trained run and the reward summary until a command-line run's metrics are loaded.
    Values come from the run's first agent, the one whose metrics the Training monitor shows.
    """
    agents = (disk or {}).get("agents") or []
    primary = agents[0] if agents else {}
    rewards = [m["reward"] for m in run["metrics"] if m.get("reward") is not None]
    best = max(rewards) if rewards else primary.get("best_reward")
    last = rewards[-1] if rewards else primary.get("latest_reward")
    steps = max((m["step"] for m in run["metrics"]), default=None) or primary.get("timesteps")
    summary = run.get("_metric_summary") or {}
    if run.get("_metrics_deferred"):
        best = summary.get("best_reward", best)
        last = summary.get("latest", {}).get("reward", last)
        steps = summary.get("step") or steps
    source = run_source(run)
    started = primary.get("started") if source == "cli" else run.get("created")
    duration = primary.get("training_time_seconds")
    methods = ((disk or {}).get("config") or {}).get("methods") or {}
    method = (" + ".join(" / ".join(str(m.get(key)) for key in ("agent", "model") if m.get(key)) or "—"
                          for m in methods.values()) if methods else "—")
    stamp = _timestamp(started)
    name, env, seed = run_name(run), run["config"]["env"], run["config"]["seed"]
    reward_text = lambda value: "—" if value is None else f"{value:.1f}"
    return [
        (name, name.lower()),
        ("—" if stamp is None else datetime.fromtimestamp(stamp).strftime("%Y-%m-%d %H:%M"), stamp),
        (method, method),
        (env, env),
        (str(seed), seed),
        ("—" if steps is None else f"{steps:,}", steps),
        (format_duration(duration), duration),
        (run["status"], run["status"]),
        (reward_text(best), best),
        (reward_text(last), last),
        (SOURCE_LABELS[source], SOURCE_LABELS[source]),
    ]


def matches_filters(run, source="all", status="all", show_examples=True):
    """Whether the Results browser's source/status filters keep a run."""
    kind = run_source(run)
    if kind == "example" and not show_examples:
        return False
    if source == "real" and kind not in ("app", "cli"):
        return False
    if source in ("app", "cli") and kind != source:
        return False
    if source == "simulated" and kind not in ("simulated", "example"):
        return False
    return status == "all" or run["status"] in STATUS_GROUPS[status]


def load_run_metrics(run):
    """Materialize a saved history only when a consumer needs its points."""
    path = run.get("_metrics_deferred")
    if path:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        if record.get("id") != run["id"] or not isinstance(record.get("metrics"), list):
            raise ValueError("Invalid run metric history")
        run["metrics"] = record["metrics"]
        run.pop("_metrics_deferred", None)
        run.pop("_metric_summary", None)
    return run["metrics"]


def _metric_summary(metrics):
    last = {}
    rewards = []
    step = 0
    for point in metrics:
        last.update({key: value for key, value in point.items() if value is not None})
        step = max(step, point.get("step") or 0)
        if isinstance(point.get("reward"), (int, float)):
            rewards.append(point["reward"])
    stride = max(1, len(rewards) / 31)
    trend = [rewards[int(i * stride)] for i in range(min(31, len(rewards)))]
    if rewards:
        trend.append(rewards[-1])
    return {"count": len(metrics), "latest": last, "step": step,
            "best_reward": max(rewards) if rewards else None, "trend": trend}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run):
        load_run_metrics(run)  # Never overwrite a deferred history with an empty list.
        record = {key: value for key, value in run.items() if key not in ("_metrics_deferred", "_metric_summary")}
        folder = self.root / run["id"]
        folder.mkdir(exist_ok=True)
        temporary = folder / "record.tmp"
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        temporary.replace(folder / "record.json")
        try:
            self._save_summary(folder / "record.json", record)
        except OSError:
            pass  # The authoritative record remains usable without this derived cache.
        (folder / "config.yaml").write_text(Config(**run["config"]).recipe_yaml(), encoding="utf-8")
        columns = ["step"] + [key for key in METRIC_COLUMNS if any(key in m for m in run["metrics"])]
        cell = lambda value: "" if value is None else value
        rows = [",".join(columns)] + [",".join(str(cell(m.get(key))) for key in columns) for m in run["metrics"]]
        (folder / "metrics.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _save_summary(self, path, run, stamp=None):
        stamp = stamp or self._stamp(path)
        summary = {key: value for key, value in run.items() if key not in ("metrics", "_metrics_deferred", "_metric_summary")}
        summary["metrics"] = []
        summary["_metric_summary"] = _metric_summary(run["metrics"])
        payload = {"version": 1, "stamp": stamp, "run": summary}
        temporary = path.with_name(f"summary-{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            temporary.replace(path.with_name("summary.json"))
        finally:
            temporary.unlink(missing_ok=True)
        return summary

    @staticmethod
    def _stamp(path):
        stat = path.stat()
        return [stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino]

    def _cached_summary(self, path, stamp):
        try:
            cached = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
            if not isinstance(cached, dict) or cached.get("version") != 1 or cached.get("stamp") != stamp:
                return None
            run = cached["run"]
            if not isinstance(run, dict) or run.get("id") != path.parent.name or run.get("metrics") != []:
                return None
            Config(**run["config"])
            for key in ("status", "notes", "created", "simulated"):
                run[key]
            summary = run["_metric_summary"]
            if not isinstance(summary.get("latest"), dict) or not isinstance(summary.get("trend"), list):
                return None
            return cached
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None

    def load_summaries(self):
        return self.load(summaries_only=True)

    def load(self, *, summaries_only=False):
        runs, errors = [], []
        for path in self.root.glob("*/record.json"):
            try:
                stamp = self._stamp(path)
                cached = self._cached_summary(path, stamp) if summaries_only else None
                run = cached["run"] if cached else json.loads(path.read_text(encoding="utf-8"))
                Config(**run["config"])
                if run["id"] != path.parent.name or not isinstance(run["metrics"], list):
                    raise ValueError("Invalid run record")
                for key in ("status", "notes", "created", "simulated"):
                    run[key]
                if summaries_only:
                    if not cached:
                        try:
                            summary = self._save_summary(path, run, stamp)
                        except OSError:
                            summary = {key: value for key, value in run.items() if key != "metrics"}
                            summary.update(metrics=[], _metric_summary=_metric_summary(run["metrics"]))
                        run = summary
                    run["_metrics_deferred"] = str(path)
                if run["status"] == "running" and run["simulated"]:
                    run["status"] = "interrupted"  # live runs are reconnected to the backend instead
                runs.append(run)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f"Could not load {path}: {exc}")
        return sorted(runs, key=lambda r: r["created"], reverse=True), errors
