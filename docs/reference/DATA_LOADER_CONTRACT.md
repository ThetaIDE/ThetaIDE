# Data loader ownership and selection

An environment describes the task and data source. Experiments bind a training data module to methods in the **Data Loaders** box. A package may supply a model and a loader as separate capabilities; ownership of a loader does not make it responsible for constructing the model.

`data_module` provides training batches, splits and sampling through the Lightning data-module interface.

## Configuration

```yaml
data_module:
  name: replay_buffer
  params:
    num_workers: 4

methods:
  baseline:
    agent: d3rlpy/cql
  tabular:
    agent: themes-dev/apprenticeship
    data_module: themes-dev/csv_trajectories
```

Selections name a registered provider or an installed `data/` profile. Each method inherits the experiment selection unless it overrides it. Parameters merge for the same selector; a different selector replaces the previous selection and its parameters, including when selecting another profile of the same implementation. Training modules retain the paradigm fallback. Environments do not select loaders. CLI provider selections replace inherited provider settings; apply parameter edits after the provider selection.

Dataset identity, split membership and run seeds remain shared experiment settings. Each run creates its own data module instance.

## Declarations and compatibility

A data module is a `datamodule` capability. Algorithms and models state which data modules they accept with [compatibility requirements](COMPATIBILITY_REQUIREMENTS.md): a `kind: datamodule` rule with `capability` or `contract`, or a `config: data_module` rule (for example `present: false` for an algorithm that trains only from the live environment). Data modules state the contracts they satisfy in `provides`. The Data Loaders box appears when any method sets a data module and lists only candidates without block findings.

Components a model consumes, such as BlendRL's symbolic state builder, are [component-reference parameters](COMPATIBILITY_REQUIREMENTS.md#component-references) of that model (`model.symbolic_state`), not experiment slots. The model loads its selection with `load_selected`.

## Numerical trajectory providers

The apprenticeship runner consumes `themes.trajectories/v1` from a training
provider through `load(path, patients, seed)` and `prepare(data, seed)`, rather
than requesting Lightning batches. The loader owns source schema validation,
visit grouping, ordering, action/feature selection, splits, lagging and fitted
preprocessing. The runner receives the resulting arrays and fitted state and
passes them to the numerical models; it contains no dataset-specific parsing.

`themes-dev/csv_trajectories` is the custom loader for downloaded THEMES CSVs.
Its editable profile exposes visit, time, action, feature and optional actual
person columns, plus held-out fraction. It defaults to binary `a_2` and the
14 physiological columns; `Action`, `Shock`, identifiers, timestamps and other
intervention flags cannot become policy features. It preserves original CSV row
indices and timestamps. Missing unused Shock annotations are ignored. Configured
actual person IDs keep repeated visits together; otherwise splitting is by visit.
Scalers fit training visits only, and persisted loader schema plus source checksum
support independent lineage verification and frozen inference after reload.

CSV files remain external local datasets. The provider payload contains the
loader and recipe references, never clinical rows. See the [AL guide](../guides/THEMES_AL.md)
for runnable local CSV recipes and the fixed-row solver/time-window conventions.


Dataset source paths are workspace references. Relative paths (including `../`
for external files) resolve from the Theta project root, independent of process
working directory or temporary preview configuration directories. The shared
file picker saves relative paths when possible; explicit absolute and home-relative
paths remain valid. Package URIs retain provider-relative resolution. An explicit
missing path fails with the actual file reference and `dataset_path` guidance,
without silently choosing other data or suggesting component installation.
