from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.adam import Adam
from torch.optim.optimizer import Optimizer

from dysplasia.modeling import MLP
from dysplasia.typing import (
    Output,
    SlideEmbeddingsInput,
    SlideEmbeddingsPredictInput,
)


class DysplasiaEmbeddingsPercentagesModel(LightningModule):
    """MLP regressor that predicts per-tile dysplasia percentages from embeddings.

    Expects the dataset to return a target tensor shaped ``(num_targets,)`` with
    values in ``[0, 1]`` (e.g. LG% and HG% as fractions).
    """

    def __init__(
        self,
        dims: list[int] | tuple[int, ...] | None = None,
        lr: float | None = None,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        dims = tuple(dims) if dims is not None else (768, 256, 128, 2)
        self.regressor = MLP(*dims)
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.clamp_output = clamp_output

    def forward(self, x: Tensor) -> Output:
        y = self.regressor(x)
        return y.sigmoid() if self.clamp_output else y

    def training_step(self, batch: SlideEmbeddingsInput) -> Tensor:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        labels = labels.float()
        preds = self(x)

        loss = self.criterion(preds, labels)
        self.log("train/loss", loss, on_step=True, prog_bar=True)

        mae = (preds - labels).abs().mean()
        self.log("train/mae", mae, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: SlideEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        labels = labels.float()
        preds = self(x)

        loss = self.criterion(preds, labels)
        self.log("validation/loss", loss, prog_bar=True)

        mae = (preds - labels).abs().mean()
        self.log("validation/mae", mae, on_step=False, on_epoch=True)

    def test_step(self, batch: SlideEmbeddingsInput) -> None:  # pylint: disable=arguments-differ
        x, labels, _ = batch
        labels = labels.float()
        preds = self(x)

        mae = (preds - labels).abs().mean()
        self.log("test/mae", mae, on_step=False, on_epoch=True)

    def predict_step(self, batch: SlideEmbeddingsPredictInput) -> Output:  # pylint: disable=arguments-differ
        return self(batch[0])

    def configure_optimizers(self) -> Optimizer:
        if self.lr is None:
            raise ValueError("Learning rate must be set for training.")
        return Adam(self.parameters(), lr=self.lr)
