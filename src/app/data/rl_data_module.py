from typing import Any, Dict, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from src.app.core.interfaces import BaseDataModule
from src.app.dataset_utils import DatasetReader


class OfflineDataset(Dataset):
    def __init__(self, reader: DatasetReader):
        self.reader = reader

    def __len__(self):
        return self.reader.limit

    def __getitem__(self, idx):
        return idx


class RLDataModule(L.LightningDataModule, BaseDataModule):
    def __init__(self, cfg: Any = None):
        super().__init__()
        self.cfg = cfg
        self.reader: DatasetReader | None = None
        self.val_reader: DatasetReader | None = None
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None, cfg: Any = None) -> None:
        if cfg is not None and not isinstance(cfg, str):
            self.cfg = cfg
        elif stage is not None and not isinstance(stage, str) and cfg is None:
            self.cfg = stage
        if self.cfg is None:
            return

        paradigm = (
            self.cfg.get("paradigm", "online_rl")
            if hasattr(self.cfg, "get")
            else getattr(self.cfg, "paradigm", "online_rl")
        )

        if paradigm == "offline_rl":
            import os
            from pathlib import Path

            dataset_path = Path(self.cfg.dataset_path)
            if self.cfg.env.get("preprocess_on_load", False) and (
                not dataset_path.exists() or not list(dataset_path.glob("*.pkl"))
            ):
                preprocess_script = self.cfg.env.get("preprocess_script", None)
                if not preprocess_script:
                    raise ValueError('preprocess_on_load requires an explicit env.preprocess_script package URI or path')

                if preprocess_script:
                    from src.app.core.provider_loader import resolve_resource
                    preprocess_script = str(resolve_resource(preprocess_script))
                if preprocess_script and Path(preprocess_script).exists():
                    print(f"[RLDataModule] Dataset missing at '{dataset_path}'. Auto-running {preprocess_script}...")
                    import subprocess
                    import sys

                    subprocess.run([sys.executable, "-m", "src.app.pipeline.resource_worker", str(preprocess_script)], check=True)

            full_reader = DatasetReader(self.cfg.dataset_path)

            # Determine if validation split should be enabled
            val_split = self.cfg.get("val_split", None)
            if val_split is None:
                val_split = 0.1

            if val_split > 0 and len(full_reader) > 10:
                self.reader, self.val_reader = full_reader.split(val_ratio=val_split, seed=self.cfg.seed)
            else:
                self.reader = full_reader
                self.val_reader = None

            self.train_dataset = OfflineDataset(self.reader)
            if self.val_reader is not None:
                self.val_dataset = OfflineDataset(self.val_reader)
        else:
            # Online mode dummy dataset
            self.train_dataset = torch.utils.data.TensorDataset(torch.zeros(1))
            self.val_dataset = None

    def train_dataloader(self):
        paradigm = (
            self.cfg.get("paradigm", "online_rl")
            if hasattr(self.cfg, "get")
            else getattr(self.cfg, "paradigm", "online_rl")
        )
        is_offline = paradigm == "offline_rl"
        agent_cfg = getattr(self.cfg, "agent", {}) if hasattr(self.cfg, "agent") else {}
        if agent_cfg is None:
            agent_cfg = {}
        batch_size = agent_cfg.get("batch_size", 1024) if is_offline else agent_cfg.get("batch_size", 1)
        default_workers = 0 if is_offline else (2 if torch.cuda.is_available() else 0)
        num_workers = (
            self.cfg.get("num_workers", agent_cfg.get("num_workers", default_workers))
            if hasattr(self.cfg, "get")
            else default_workers
        )
        pin_memory = (
            self.cfg.get("pin_memory", torch.cuda.is_available() and num_workers > 0)
            if hasattr(self.cfg, "get")
            else False
        )
        persistent_workers = num_workers > 0
        if self.reader is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda idxs: self.reader.get_batch(idxs, device="cpu"),
            )
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def val_dataloader(self):
        paradigm = (
            self.cfg.get("paradigm", "online_rl")
            if hasattr(self.cfg, "get")
            else getattr(self.cfg, "paradigm", "online_rl")
        )
        is_offline = paradigm == "offline_rl"
        agent_cfg = getattr(self.cfg, "agent", {}) if hasattr(self.cfg, "agent") else {}
        if agent_cfg is None:
            agent_cfg = {}
        if self.val_dataset is not None and self.val_reader is not None:
            batch_size = agent_cfg.get("batch_size", 1024)
            default_workers = 0 if is_offline else (2 if torch.cuda.is_available() else 0)
            num_workers = (
                self.cfg.get("num_workers", agent_cfg.get("num_workers", default_workers))
                if hasattr(self.cfg, "get")
                else default_workers
            )
            pin_memory = self.cfg.get("pin_memory", torch.cuda.is_available() and num_workers > 0)
            persistent_workers = num_workers > 0
            return DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=lambda idxs: self.val_reader.get_batch(idxs, device="cpu"),
            )
        return None
