# d3rlpy-dev

Editable local execution-contract-v1 backend. IDs are `d3rlpy-dev/bc`, `/cql`,
`/iql`, `/td3_bc` and encoder model `d3rlpy-dev/vector`. Production `d3rlpy/*`
selectors are not redirected. Parent provenance is d3rlpy-suite 1.0.3; this
implementation replaces its script dispatch with a static-data fitting adapter.

Use `Execution/d3rlpy_smoke` with a configured `dataset_path` pointing to a local
Theta `DatasetWriter` transition directory. The backend loads each transition
once, performs the selected learner's updates and records training losses. It
never calls a simulator for offline validation. Training losses are not held-out
scientific evaluation. Native `.d3` bundles can be reloaded with
`d3rlpy.load_learnable`; exact training resume is rejected. Tune/promotion support
is a remaining gate. No clinical data is distributed.
