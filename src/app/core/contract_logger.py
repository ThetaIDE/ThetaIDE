"""Universal Contract Logger for Black-Box Training Engines.

Pure-Python, zero-dependency logger conforming to Theta-IDE's execution contract:
  1. Writes standard metric rows to `results/logs/<group>/<exp>/<agent>/version_X/metrics.csv`
  2. Emits standard columns (`step`, `transitions`, `eval/reward`, `val/loss`, etc.)
  3. Writes standard `runtime.json` metadata for reproducibility.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List


class ContractLogger:
    """Manages telemetry output adhering to Theta-IDE's universal metrics contract.

    Can be used by any framework (Stable-Baselines3, CleanRL, TensorFlow, JAX, Scikit-Learn)
    without requiring PyTorch or PyTorch Lightning.
    """

    def __init__(
        self,
        log_dir: str | Path | None = None,
        group: str | None = None,
        experiment_id: str | None = None,
        method_name: str | None = None,
        base_dir: str | Path = "results/logs",
        version: str | None = None,
        flush_interval: int = 1,
    ):
        """Initialize logger.

        Args:
            log_dir: Direct path to log directory. If provided, overrides group/exp/method.
            group: Experiment group name.
            experiment_id: Unique experiment name.
            method_name: Algorithm/Agent identifier.
            base_dir: Root logs directory (defaults to 'results/logs').
            version: Explicit version directory name (e.g. 'version_0'). If None, auto-resolved.
            flush_interval: Flush file to disk every N logged rows.
        """
        if log_dir is not None:
            self.target_dir = Path(log_dir)
        elif group and experiment_id and method_name:
            method_clean = method_name.replace("/", "_")
            self.target_dir = Path(base_dir) / group / experiment_id / method_clean
        else:
            raise ValueError("Either 'log_dir' or all of ('group', 'experiment_id', 'method_name') must be provided.")

        self.version_dir = self._resolve_version_dir(self.target_dir, version)
        self.version_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv_path = self.version_dir / "metrics.csv"
        self.flush_interval = max(1, flush_interval)
        self.flush_seconds = float(os.environ.get('THETA_STREAM_FLUSH_SECONDS', '0'))
        self._last_flush = time.monotonic()

        self._fieldnames: List[str] = []
        self._rows: List[Dict[str, Any]] = []
        self._file = None
        self._writer = None
        self._step_counter = 0
        self._start_time = time.time()
        self._unwritten_count = 0

        self._init_csv()

    @staticmethod
    def _resolve_version_dir(target_dir: Path, explicit_version: str | None = None) -> Path:
        """Find or create appropriate version_X directory."""
        if explicit_version:
            return target_dir / explicit_version

        if not target_dir.exists():
            return target_dir / "version_0"

        # Check existing version_X folders
        existing_versions = []
        for p in target_dir.glob("version_*"):
            if p.is_dir():
                stem = p.name.replace("version_", "")
                if stem.isdigit():
                    existing_versions.append(int(stem))

        if not existing_versions:
            return target_dir / "version_0"

        # Default to latest if empty, or next available version
        next_ver = max(existing_versions) + 1
        return target_dir / f"version_{next_ver}"

    def _init_csv(self) -> None:
        """Read existing CSV header if resuming, else initialize empty fieldnames."""
        if self.metrics_csv_path.exists() and self.metrics_csv_path.stat().st_size > 0:
            try:
                with open(self.metrics_csv_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        self._fieldnames = list(header)
            except Exception:
                self._fieldnames = []

    def _rewrite_csv_with_new_fields(self, new_fields: List[str]) -> None:
        """Rewrite existing CSV file with expanded fieldnames when new metrics appear."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

        existing_rows: List[Dict[str, Any]] = []
        if self.metrics_csv_path.exists():
            try:
                with open(self.metrics_csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    existing_rows = list(reader)
            except Exception:
                pass

        self._fieldnames = new_fields
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)

    def log(
        self,
        metrics: Dict[str, Any],
        step: int | None = None,
        transitions: int | float | None = None,
        epoch: int | None = None,
    ) -> None:
        """Log a dictionary of metrics adhering to standard Theta-IDE columns.

        Args:
            metrics: Key-value metric pairs (e.g. {'eval/reward': 150.0, 'val/loss': 0.05}).
            step: Global optimizer step. If None, auto-increments.
            transitions: Cumulative environment transitions/samples.
            epoch: Optional training epoch number.
        """
        if step is not None:
            self._step_counter = step
        else:
            self._step_counter += 1

        row: Dict[str, Any] = {"step": self._step_counter}

        if transitions is not None:
            row["transitions"] = transitions
        elif "transitions" in metrics:
            row["transitions"] = metrics.pop("transitions")

        if epoch is not None:
            row["epoch"] = epoch
        elif "epoch" in metrics:
            row["epoch"] = metrics.pop("epoch")

        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                row[k] = v
            elif hasattr(v, "item") and callable(v.item):
                row[k] = v.item()
            else:
                try:
                    row[k] = float(v)
                except (ValueError, TypeError):
                    row[k] = str(v)

        # Check if new fields appeared that aren't in header
        current_keys = list(row.keys())
        missing_keys = [k for k in current_keys if k not in self._fieldnames]
        if missing_keys:
            # Preserve standard ordering: epoch, step, transitions, eval/reward, then others
            base_order = ["epoch", "step", "transitions", "eval/reward", "eval/reward_std"]
            all_keys = list(self._fieldnames)
            for k in missing_keys:
                if k not in all_keys:
                    all_keys.append(k)
            # Sort prioritized fields first
            sorted_fields = [k for k in base_order if k in all_keys] + [k for k in all_keys if k not in base_order]
            self._rewrite_csv_with_new_fields(sorted_fields)

        # Append row to CSV
        if self._file is None:
            write_header = not self.metrics_csv_path.exists() or self.metrics_csv_path.stat().st_size == 0
            self._file = open(self.metrics_csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            if write_header:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._unwritten_count += 1
        if (not self.flush_seconds and self._unwritten_count >= self.flush_interval) or time.monotonic() - self._last_flush >= self.flush_seconds or self._unwritten_count >= 256:
            self.flush()

    def log_eval(
        self,
        reward_mean: float,
        reward_std: float = 0.0,
        step: int | None = None,
        transitions: int | float | None = None,
        epoch: int | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Convenience method for logging evaluation rollout performance."""
        payload: Dict[str, Any] = {
            "eval/reward": float(reward_mean),
            "eval/reward_std": float(reward_std),
        }
        if extra:
            payload.update(extra)
        self.log(payload, step=step, transitions=transitions, epoch=epoch)

    def flush(self) -> None:
        """Flush internal buffers to disk."""
        if self._file:
            self._file.flush()
            self._last_flush = time.monotonic()
            self._unwritten_count = 0

    def close(self) -> None:
        """Close logger file handles."""
        self.flush()
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None

    def save_runtime_metadata(
        self,
        config: Dict[str, Any] | None = None,
        training_time_seconds: float | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> Path:
        """Save runtime.json metadata using the Theta-IDE run contract."""
        return write_runtime_metadata(
            self.version_dir, self._start_time, self._step_counter, config, training_time_seconds, extra
        )

    def __enter__(self) -> ContractLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def write_runtime_metadata(
    version_dir: Path,
    start_time: float,
    total_steps: int,
    config: Dict[str, Any] | None = None,
    training_time_seconds: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> Path:
    """Write <version_dir>/runtime.json: when the run started and ended, how long it trained, its config."""
    end_time = time.time()
    duration = training_time_seconds or (end_time - start_time)

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time)),
        "training_time_seconds": round(duration, 3),
        "total_steps": total_steps,
        "version_dir": str(version_dir),
    }
    if config:
        meta["config"] = config
    if extra:
        meta.update(extra)

    meta_file = Path(version_dir) / "runtime.json"
    meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta_file
