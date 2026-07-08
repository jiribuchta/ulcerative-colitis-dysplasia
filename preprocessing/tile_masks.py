"""Script for creating outlines of the tiles and also masks based on tiling percentages."""

from pathlib import Path
from typing import Any

import hydra
import mlflow.artifacts
import numpy as np
import pandas as pd
import pyvips
import ray
import torch
from omegaconf import DictConfig
from ratiopath.openslide import OpenSlide
from rationai.masks import process_items, tile_mask, write_big_tiff
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote
def process_slide(
    slide: pd.Series,
    percentage_cols: list[str],
    output_path: Path,
    tiles: pd.DataFrame,
    level: int,
) -> None:
    slide_tiles = tiles[tiles["slide_id"] == slide.id].copy()
    with OpenSlide(slide["path"]) as slide_wsi:
        mask_extent_x, mask_extent_y = slide_wsi.level_dimensions[level]
        mpp_x, mpp_y = slide_wsi.slide_resolution(level)

    slide_tiles["x"] = (slide_tiles["x"] / mpp_x * slide.mpp_x).astype(int)
    slide_tiles["y"] = (slide_tiles["y"] / mpp_y * slide.mpp_y).astype(int)

    for percentage_col in [*percentage_cols]:
        filename = f"{Path(slide.path).stem}.tiff"
        save_dir = output_path / percentage_col

        builder = ScalarMaskBuilder(
            save_dir,
            filename,
            mask_extent_x,
            mask_extent_y,
            mpp_x,
            mpp_y,
            slide.tile_extent_x,
            slide.stride_x,
        )

        xs = torch.tensor(slide_tiles["x"].values)
        ys = torch.tensor(slide_tiles["y"].values)
        data = torch.tensor(slide_tiles[percentage_col].values)
        builder.update(data, xs, ys)
        builder.save()

    # Outlines
    mask = tile_mask(
        slide_tiles,
        tile_extent=(slide.tile_extent_x, slide.tile_extent_y),
        size=(mask_extent_x, mask_extent_y),
    )

    mask_path = output_path / "outlines" / f"{Path(slide.path).stem}.tiff"
    write_big_tiff(
        pyvips.Image.new_from_array(np.array(mask)),
        mask_path,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
    )


@with_cli_args(["+preprocessing=tile_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    for percentage_col in [*config.percentage_cols, "outlines"]:
        (output_path / str(percentage_col)).mkdir(parents=True, exist_ok=True)

    for name, uri in config.dataset.mlflow_uris.tiling_filtered.items():
        local_path = Path(mlflow.artifacts.download_artifacts(uri))

        slides = pd.read_parquet(local_path / "slides")
        tiles = pd.read_parquet(local_path / "tiles")


        process_items(
            (slide for _, slide in slides.iterrows()),
            process_slide,
            fn_kwargs={
                "percentage_cols": config.percentage_cols,
                "output_path": output_path,
                "tiles": tiles,
                "level": config.level,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(
            str(output_path), artifact_path=config.mlflow_artifact_path
        )


if __name__ == "__main__":
    main()
