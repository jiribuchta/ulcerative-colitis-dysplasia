from collections.abc import Iterable
from typing import Any

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dysplasia.typing import (
    TileMetadata,
    TilesInput,
    TilesPredictInput,
)


class DataModule(LightningDataModule):
    def __init__(
        self, batch_size: int, num_workers: int = 0, datasets: DictConfig | None = None
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets if datasets is not None else {}

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
            case "validate":
                self.val = instantiate(self.datasets["val"])
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[TilesInput]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=tile_collate,
        )

    def val_dataloader(self) -> Iterable[TilesInput]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=tile_collate,
        )

    def test_dataloader(self) -> Iterable[TilesInput]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=tile_collate,
        )

    def predict_dataloader(self) -> Iterable[TilesPredictInput]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=tile_collate,
        )


def tile_collate(
    batch: list[Any],
) -> (
    tuple[torch.Tensor, torch.Tensor, list[TileMetadata]]
    | tuple[torch.Tensor, list[TileMetadata]]
):
    """Custom collate function for tile batches."""
    if len(batch[0]) == 3:
        images = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        metadata = [item[2] for item in batch]
        return images, labels, metadata
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return images, metadata
