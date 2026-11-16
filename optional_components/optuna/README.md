# Optuna tuner for ThetaIDE

Install **Optuna tuner** from **Components → Hub → All Components**. This package
uses native storage v3 and the Theta SDK; use the IDE's current layer-boundary
implementation, which supports those contracts. Older IDE builds that support
only storage v2 cannot install it.

The installer supplies the declared Optuna and SQLAlchemy dependencies to the
IDE's Python environment. Select `tuning.component: optuna` in an experiment;
saved search spaces live under `methods.<method>.tune`, and tuner options under
`tuning.options.optuna`. The provider contributes settings, parameter sweep
controls and the study monitor through the host's generic extension slots.

Version 1.1.1 is the first production Hub release. It packages the current
maintained implementation, including the `hydra/sweeper/theta_optuna` profile,
workspace-owned result paths, public SDK imports, and explicit tuning selection.
Version 1.1.0 existed only as an unpublished local preparation.

This package's maintained source is `optional_components/optuna` in ThetaIDE.
Build it with `scripts/package_optional_components.py`; managed installed copies
are read-only and are replaced through normal package installation.
