from typing import Any

from datasets import Dataset as HFDataset


def filter_tiles(
    tiles: HFDataset,
    thresholds: dict[str, float] | None = None,
    min_thresholds: dict[str, float] | None = None,
) -> HFDataset:
    """Filter tiles by per-column value thresholds.

    ``thresholds`` (upper bounds) keep tiles where ``column <= value``, e.g.
    QC columns like blur/artifacts must be low. ``min_thresholds`` (lower
    bounds) keep tiles where ``column >= value``, e.g. epithelium coverage
    must be high. A tile is kept only if it satisfies every bound.
    """
    thresholds = thresholds or {}
    min_thresholds = min_thresholds or {}

    def keep(tile: dict[str, Any]) -> bool:
        return all(tile[col] <= thr for col, thr in thresholds.items()) and all(
            tile[col] >= thr for col, thr in min_thresholds.items()
        )

    return tiles.filter(keep)
