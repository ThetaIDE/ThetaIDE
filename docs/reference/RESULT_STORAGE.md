# Saved result layout

Unique saved runs live inside their experiment folder:

```text
results/<artifact-root>/<group>/<experimentID>/<run>/...
```

`experimentID` retains recipe subdirectories. The IDE's **Timestamp experiment** option creates a timestamped run folder on every launch, with a counter for collisions. Repeated-seed executions use a timestamp plus a UUID suffix; cluster submissions use their execution UUID. A repeated-seed execution launched with an already unique IDE or cluster result ID reuses that run folder. Logs, checkpoints, plots, and other artifacts share the same result ID. Recovery keeps its selected result ID.

The Results panel groups these saves under one experiment. Older sibling folders such as `experiment_20261112-120000`, repeated-seed suffixes, and cluster UUID suffixes remain readable and are grouped with new saves. Existing folders are not automatically moved.

Runs with timestamping disabled and ordinary command-line launches still write directly to the configured experiment ID. Replacing, archiving, viewing, or deleting such a legacy run excludes nested saved runs. Experiment renames move the enclosing folder and update execution references; individual saved runs can also be renamed.
