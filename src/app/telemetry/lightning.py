"""Lightning bridge: every self.log(...) of a LightningModule also reaches a telemetry run.

ThetaLogger sits next to Lightning's CSVLogger (which keeps writing metrics.csv) and writes to the same
version_N folder, so agents need no changes. Checkpoints arrive through after_save_checkpoint, the end
of a failed fit through finalize; TelemetryCallback adds the exception message. A successful fit only
marks the run completed: the pipeline logs a few more values afterwards (training time, GPU peaks), so
finish_telemetry(trainer) ends it once those are written, or the process exit does.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .run import Run
from .selection import MetricSelection
from .sinks import Sink
from .system import DEFAULT_INTERVAL

# Lightning's finalize() status → the run's finish status
FINISH_STATUS = {"success": "completed", "failed": "failed", "finished": "completed"}


class ThetaLogger(Logger):
    """A Lightning logger that writes to a telemetry Run in `version_dir()` (resolved when training starts)."""

    def __init__(
        self,
        version_dir: Callable[[], str | Path],
        sinks: list[str | Sink],
        config: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        system_interval: float = DEFAULT_INTERVAL,
        metrics: MetricSelection | None = None,
    ):
        super().__init__()
        self._metrics = metrics
        self._version_dir = version_dir
        self._sinks = sinks
        self._config = config
        self._meta = meta
        self._system_interval = system_interval
        self._run: Run | None = None

    @property
    def run(self) -> Run:
        """The telemetry run, started on first use (the CSVLogger's version folder is known by then)."""
        if self._run is None:
            self._run = Run(
                self._version_dir(), self._sinks, self._config, self._meta, self._system_interval, self._metrics
            )
        return self._run

    @property
    def name(self) -> str:
        return "telemetry"

    @property
    def version(self) -> str:
        return Path(self._version_dir()).name

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.run.log(dict(metrics), step=step)

    @rank_zero_only
    def log_hyperparams(self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        if isinstance(params, Namespace):
            params = vars(params)
        try:
            self.run.update_config(dict(params))
        except (TypeError, ValueError):
            pass

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback) -> None:
        path = getattr(checkpoint_callback, "_last_checkpoint_saved", None) or getattr(
            checkpoint_callback, "last_model_path", None
        )
        if not path:
            return
        best = getattr(checkpoint_callback, "best_model_path", "") == path
        score = getattr(checkpoint_callback, "best_model_score", None)
        self.run.artifact(path, kind="checkpoint", best=best)
        detail = f" (best {checkpoint_callback.monitor} {float(score):.4g})" if best and score is not None else ""
        self.run.event("checkpoint", f"Saved {Path(path).name}{detail}", path=str(path), best=best)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is None or self._run.finished:
            return
        status = FINISH_STATUS.get(status, status)
        if status == "failed":
            self._run.finish(status=status)
        else:
            self._run.exit_status = status  # values logged after fit still reach the stream


def finish_telemetry(trainer, status: str | None = None) -> None:
    """Finish the telemetry run of every ThetaLogger on `trainer` (as completed, unless fit failed)."""
    for lg in getattr(trainer, "loggers", None) or []:
        if isinstance(lg, ThetaLogger) and lg._run is not None and not lg._run.finished:
            lg._run.finish(status=status or lg._run.exit_status)


class TelemetryCallback(Callback):
    """Starts the run (and its system sampling) when fitting starts, and records why training failed."""

    def __init__(self, logger: ThetaLogger):
        self.logger = logger

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.logger.run.event("stage", "Training started", stage="training")

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        if trainer.is_global_zero:
            if getattr(exception, "theta_trial_status", None) == "pruned":
                self.logger.run.event("stage", str(exception), stage="pruned")
                self.logger.run.finish(status="pruned")
                return
            self.logger.run.event("error", f"{type(exception).__name__}: {exception}", level="error")
