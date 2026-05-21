from typing import TypedDict

from torch import Tensor


type Output = Tensor


class TileMetadata(TypedDict):
    slide_id: str
    x: int
    y: int


type TilesSample = tuple[Tensor, Tensor, TileMetadata]
type TilesPredictSample = tuple[Tensor, TileMetadata]
type TilesInput = tuple[Tensor, Tensor, list[TileMetadata]]
type TilesPredictInput = tuple[Tensor, list[TileMetadata]]


type SlideEmbeddingsSample = tuple[Tensor, Tensor, dict[str, str]]
type SlideEmbeddingsPredictSample = tuple[Tensor, dict[str, str]]
type SlideEmbeddingsInput = tuple[Tensor, Tensor, list[dict[str, str]]]
type SlideEmbeddingsPredictInput = tuple[Tensor, list[dict[str, str]]]
