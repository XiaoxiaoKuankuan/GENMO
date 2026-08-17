"""Minimal BUMI music/qpos DataModule without human/camera placeholder fields."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch.utils.data import ConcatDataset, DataLoader, Subset, default_collate

from gem.utils.pylogger import Log


def bumi_collate_fn(batch: list[dict], mode: str = "train") -> dict:
    if not batch:
        raise ValueError("BUMI collate received an empty batch")
    allowed = {
        "qpos",
        "music_embed",
        "music_beats",
        "foot_contact",
        "length",
        "fps",
        "mask",
        "meta",
    }
    for item_index, item in enumerate(batch):
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(
                f"BUMI {mode} item {item_index} contains unsupported fields {sorted(unknown)}"
            )
        required = allowed - {"foot_contact"}
        missing = required - set(item)
        if missing:
            raise ValueError(f"BUMI {mode} item {item_index} is missing {sorted(missing)}")
    result = {
        "B": len(batch),
        "qpos": default_collate([item["qpos"] for item in batch]),
        "music_embed": default_collate([item["music_embed"] for item in batch]),
        "music_beats": default_collate([item["music_beats"] for item in batch]),
        "length": default_collate([item["length"] for item in batch]),
        "fps": default_collate([item["fps"] for item in batch]),
        "mask": default_collate([item["mask"] for item in batch]),
        "meta": [item["meta"] for item in batch],
    }
    has_contact = ["foot_contact" in item for item in batch]
    if any(has_contact):
        template = next(item["foot_contact"] for item in batch if "foot_contact" in item)
        contacts = []
        available = []
        for item, present in zip(batch, has_contact, strict=True):
            if present:
                contact = item["foot_contact"]
                if contact.shape != template.shape:
                    raise ValueError(
                        f"mixed BUMI foot_contact shapes: {contact.shape} != {template.shape}"
                    )
                contacts.append(contact)
                available.append(item["mask"]["valid"].bool())
            else:
                # The value is never treated as a label: availability=False
                # tells BumiEndecoder to derive contact from this sample's GT qpos FK.
                contacts.append(torch.zeros_like(template))
                available.append(torch.zeros_like(item["mask"]["valid"], dtype=torch.bool))
        result["foot_contact"] = default_collate(contacts)
        result["foot_contact_available"] = default_collate(available)
    return result


class DataModule(pl.LightningDataModule):
    """Concat multiple BUMI training roots and keep evaluation loaders sequential."""

    def __init__(
        self,
        dataset_opts,
        loader_opts,
        limit_each_trainset: int | None = None,
        train_subset_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.loader_opts = loader_opts
        self.limit_each_trainset = limit_each_trainset
        self.train_subset_ratio = train_subset_ratio
        if train_subset_ratio is not None and not 0.0 < float(train_subset_ratio) <= 1.0:
            raise ValueError("train_subset_ratio must be in (0,1]")
        if "train" in dataset_opts:
            self.trainset = self._build_train(dataset_opts["train"])
        for split in ("val", "test"):
            if split in dataset_opts:
                datasets = [instantiate(value) for value in dataset_opts[split].values()]
                setattr(self, f"{split}sets", datasets)
                for index, dataset in enumerate(datasets):
                    Log.info(f"[BUMI {split.title()} Dataset][{index + 1}]: size={len(dataset)}")

    def _build_train(self, split_opts) -> ConcatDataset:
        datasets = []
        records = []
        for index, (name, config) in enumerate(split_opts.items()):
            dataset = instantiate(config)
            summary = dict(getattr(dataset, "sampling_summary", {}))
            if self.limit_each_trainset is not None:
                count = min(int(self.limit_each_trainset), len(dataset))
                dataset = Subset(dataset, list(range(count)))
            if self.train_subset_ratio is not None:
                count = max(1, int(len(dataset) * float(self.train_subset_ratio)))
                dataset = Subset(dataset, list(range(count)))
            datasets.append(dataset)
            records.append((name, summary, len(dataset)))
            Log.info(
                f"[BUMI Train Dataset][{index + 1}/{len(split_opts)}]: "
                f"name={name}, size={len(dataset)}"
            )
        if not datasets:
            raise ValueError("BUMI DataModule requires at least one training dataset")
        effective_total = sum(len(dataset) for dataset in datasets)
        for name, summary, effective_len in records:
            if summary:
                fraction = effective_len / effective_total if effective_total else 0.0
                Log.info(
                    "[BUMI Train Sampling] "
                    f"name={name}, raw_sequences={summary['raw_sequences']}, "
                    f"hours={summary['hours']:.6f}, effective_len={effective_len}, "
                    f"sampling_fraction={fraction:.6%}, "
                    f"duration_aware={summary['duration_aware_sampling']}"
                )
        result = ConcatDataset(datasets)
        Log.info(f"[BUMI Train Dataset][All]: ConcatDataset size={len(result)}")
        return result

    @staticmethod
    def _options(config) -> dict:
        if isinstance(config, Mapping):
            return dict(config)
        return {key: config[key] for key in config}

    def train_dataloader(self):
        if not hasattr(self, "trainset"):
            return super().train_dataloader()
        options = self._options(self.loader_opts.train)
        workers = int(options.pop("num_workers", 0))
        return DataLoader(
            self.trainset,
            shuffle=True,
            num_workers=workers,
            persistent_workers=workers > 0,
            drop_last=True,
            collate_fn=partial(bumi_collate_fn, mode="train"),
            **options,
        )

    def _sequential_loaders(self, split: str):
        datasets = getattr(self, f"{split}sets", None)
        if datasets is None:
            return None
        options = self._options(self.loader_opts[split])
        workers = int(options.pop("num_workers", 0))
        loaders = [
            DataLoader(
                dataset,
                shuffle=False,
                num_workers=workers,
                persistent_workers=workers > 0,
                collate_fn=partial(bumi_collate_fn, mode=split),
                **options,
            )
            for dataset in datasets
        ]
        return CombinedLoader(loaders, mode="sequential")

    def val_dataloader(self):
        return self._sequential_loaders("val")

    def test_dataloader(self):
        return self._sequential_loaders("test")


__all__ = ["DataModule", "bumi_collate_fn"]
