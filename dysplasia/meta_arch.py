from copy import deepcopy

from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from dysplasia.typing import TilesInput, TilesPredictInput


class MetaArch(LightningModule):
    """Lightning module for tile-level dysplasia classification.

    Combines a *backbone* (e.g. VGG16, ResNet) with a *head* (e.g. MLP or
    linear layer) and trains with BCE-with-logits loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        lr: float | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        metrics: dict[str, Metric | MetricCollection] = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "kappa": BinaryCohenKappa(),
        }

        self.train_metrics = MetricCollection(dict(deepcopy(metrics)), prefix="train/")
        self.val_metrics = MetricCollection(dict(deepcopy(metrics)), prefix="validation/")
        self.test_metrics = MetricCollection(dict(deepcopy(metrics)), prefix="test/")

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits.sigmoid()

    def training_step(self, batch: TilesInput, batch_idx: int) -> Tensor:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self.head(self.backbone(x)).squeeze(1)
        labels = labels.float()

        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        self.train_metrics.update(logits.sigmoid(), labels)
        self.log_dict(self.train_metrics, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch: TilesInput, batch_idx: int) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self.head(self.backbone(x)).squeeze(1)
        labels = labels.float()

        loss = self.criterion(logits, labels)
        self.log("validation/loss", loss, prog_bar=True)

        self.val_metrics.update(logits.sigmoid(), labels)
        self.log_dict(self.val_metrics, on_epoch=True, on_step=False)

    def test_step(self, batch: TilesInput, batch_idx: int) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        logits = self.head(self.backbone(x)).squeeze(1)

        self.test_metrics.update(logits.sigmoid(), labels.float())
        self.log_dict(self.test_metrics, on_epoch=True, on_step=False)

    def predict_step(self, batch: TilesPredictInput, batch_idx: int) -> Tensor:  # pylint: disable=arguments-differ
        x, _ = batch
        return self.forward(x)

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)
