# 1.2.0

The `theta_optuna` sweeper profile targets the IDE's generic tuner adapter,
`hydra_plugins.theta_sweeper.ThetaSweeper` with `tuner: optuna`, which forwards to this
package's `hydra_sweeper` export. Requires an IDE with Theta SDK 3.1; the Optuna-specific
adapter `theta_optuna_sweeper` no longer exists in the IDE.

# 1.1.1

Native storage-v3 package with public SDK imports and workspace-owned results paths.
