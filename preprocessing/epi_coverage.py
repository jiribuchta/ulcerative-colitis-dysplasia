import tempfile
import warnings
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def compute_epithelium_coverage(
    mask_path: Path,
    tiles_df: pd.DataFrame,
) -> pd.Series:
    """Compute epithelium coverage for each tile from a slide mask.

    Tiles are expected to be in the same coordinate system as the mask
    (i.e. same MPP/level). Coverage is the fraction of nonzero mask pixels
    inside the tile region.
    """
    mask_image = pyvips.Image.new_from_file(str(mask_path))
    mask_width = mask_image.width
    mask_height = mask_image.height

    coverages: list[float] = []
    for _, row in tiles_df.iterrows():
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        w = int(row["tile_extent_x"])
        h = int(row["tile_extent_y"])

        # Clip to mask bounds
        x = max(0, min(x, mask_width - 1))
        y = max(0, min(y, mask_height - 1))
        w = min(w, mask_width - x)
        h = min(h, mask_height - y)

        if w <= 0 or h <= 0:
            coverages.append(0.0)
            continue

        region = mask_image.crop(x, y, w, h)
        region_array = np.ndarray(
            buffer=region.write_to_memory(),
            dtype=np.uint8,
            shape=[h, w],
        )

        tile_area = int(row["tile_extent_x"]) * int(row["tile_extent_y"])
        coverage = (
            float(np.count_nonzero(region_array)) / tile_area if tile_area > 0 else 0.0
        )
        coverages.append(coverage)

    return pd.Series(coverages, index=tiles_df.index)


def process_slide_tiles(
    slide_id: str,
    group: pd.DataFrame,
    mask_dir: Path,
) -> pd.DataFrame:
    """Add ``epi_coverage`` column to all tiles of one slide."""
    mask_path = mask_dir / f"{slide_id}.tiff"
    if not mask_path.exists():
        warnings.warn(
            f"Mask file not found for slide {slide_id} at {mask_path}. "
            "Setting epi_coverage to NaN."
        )
        group = group.copy()
        group["epi_coverage"] = np.nan
        return group

    group = group.copy()
    group["epi_coverage"] = compute_epithelium_coverage(mask_path, group)
    return group


@with_cli_args(["+preprocessing=epi_coverage"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        for split in config.splits:
            input_uri = config.mlflow_uris[config.input_tiles_uri_key][split]
            input_dir = Path(download_artifacts(artifact_uri=input_uri))

            slides = pd.read_parquet(input_dir / "slides.parquet")
            tiles = pd.read_parquet(input_dir / "tiles.parquet")

            tiles = tiles.merge(
                slides[["id", "tile_extent_x", "tile_extent_y"]].rename(
                    columns={"id": "slide_id"}
                ),
                on="slide_id",
                how="left",
            )

            mask_dir = Path(config.epi_masks_path) / split / config.epi_masks_folder
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

            tiles_with_coverage = (
                tiles.groupby("slide_id", group_keys=False)
                .apply(
                    lambda group: process_slide_tiles(
                        slide_id=group.name,
                        group=group,
                        mask_dir=mask_dir,
                    ),
                    include_groups=False,
                )
                .reset_index(drop=True)
            )

            save_dir = tmpdir_path / split
            save_dir.mkdir(parents=True, exist_ok=True)
            slides.to_parquet(save_dir / "slides.parquet", index=False)
            tiles_with_coverage.to_parquet(save_dir / "tiles.parquet", index=False)

        mlflow.log_artifacts(str(tmpdir_path), config.mlflow_artifact_path)


if __name__ == "__main__":
    main()
