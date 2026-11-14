#!/usr/bin/env python3
"""Write example .npz datasets for the tabular NPZ data modules.

Materializes real datasets for the unsupervised paradigm, which needs an offline
dataset but has no access to MIMIC. Everything here ships inside scikit-learn,
so nothing is downloaded and the result is reproducible on any machine.

Writes <output>/<name>/<name>.npz, each holding:
    features  float32  (n_samples, n_features)        flat form
    sequences float32  (n_samples, steps, per_step)    sequence form, when the
                                                       feature count factorises
    labels    int64    (n_samples,)                    held back from training;
                                                       for scoring a latent space

<output> defaults to the selected workspace's dataset root (src/usr/datasets).

Usage:
    python -m src.app.core.paradigm_impls.base.prepare_tabular                 # every dataset
    python -m src.app.core.paradigm_impls.base.prepare_tabular digits          # just one
    python -m src.app.core.paradigm_impls.base.prepare_tabular --output DIR    # elsewhere
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# name -> (sklearn loader, sequence shape or None)
DATASETS = {
    "breast_cancer": ("load_breast_cancer", None),
    "digits": ("load_digits", (8, 8)),
}

COMMAND = "python -m src.app.core.paradigm_impls.base.prepare_tabular"


def build(name: str, output: Path) -> Path:
    import sklearn.datasets as skd

    loader_name, seq_shape = DATASETS[name]
    bunch = getattr(skd, loader_name)()
    features = np.asarray(bunch.data, dtype=np.float32)
    labels = np.asarray(bunch.target, dtype=np.int64)

    # Standardise once, here: an autoencoder on raw clinical ranges is dominated
    # by whichever feature happens to have the largest units.
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    features = (features - mean) / std

    arrays = {"features": features, "labels": labels}
    if seq_shape is not None:
        steps, per_step = seq_shape
        if steps * per_step == features.shape[1]:
            arrays["sequences"] = features.reshape(-1, steps, per_step)

    out_dir = Path(output) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez_compressed(out_path, **arrays)
    shapes = ", ".join(f"{k}{v.shape}" for k, v in arrays.items())
    print(f"wrote {out_path}  ({shapes})")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write example .npz datasets for the tabular data modules.")
    parser.add_argument("names", nargs="*", help=f"datasets to build (default: all of {', '.join(sorted(DATASETS))})")
    parser.add_argument("--output", type=Path, help="dataset root to write into (default: the workspace's src/usr/datasets)")
    args = parser.parse_args(argv)
    requested = args.names or sorted(DATASETS)
    unknown = [n for n in requested if n not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(sorted(DATASETS))}")
        return 1
    if args.output is None:
        from src.app.workspace import WorkspacePaths
        args.output = WorkspacePaths.for_process().datasets
    for name in requested:
        build(name, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
