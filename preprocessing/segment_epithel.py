import asyncio
import traceback
from pathlib import Path
from typing import Any

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
from mlflow.artifacts import download_artifacts
from numpy.lib.format import open_memmap
from omegaconf import DictConfig
from rationai.client import AsyncClient
from rationai.masks import slide_resolution, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm.asyncio import tqdm


openslide: Any
try:
    import openslide
except ImportError:
    openslide = None


def extract_tile(
    slide: Any,
    row: pd.Series,
    level: int,
    tile_extent_x: int,
    tile_extent_y: int,
    downsample: float,
) -> np.ndarray:
    """Extract a single RGB uint8 tile from a whole-slide image.

    The ``x`` and ``y`` columns in the tiles DataFrame are stored in the
    coordinate system of the tiling level. OpenSlide's ``read_region`` expects
    level-0 coordinates, so we multiply by the level downsample.
    """
    x = int(round(row["x"] * downsample))
    y = int(round(row["y"] * downsample))
    tile_rgba = slide.read_region((x, y), level, (tile_extent_x, tile_extent_y))
    return np.array(tile_rgba.convert("RGB"), dtype=np.uint8)


def extract_epithelium_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    """Extract and threshold the epithelium channel from the model output.

    The model returns a (num_classes, H, W) float16 array. Channel 1 is used
    for multi-class outputs; channel 0 is used for single-channel outputs.
    The result is a binary uint8 mask with values 0 or 255.
    """
    channel = mask[1] if mask.shape[0] > 1 else mask[0]
    return ((channel > threshold) * 255).astype(np.uint8)


async def process_slide(
    slide_id: str,
    slide_path: str,
    level: int,
    tile_extent_x: int,
    tile_extent_y: int,
    tiles_df: pd.DataFrame,
    masks_dir: Path,
    client: AsyncClient,
    model_name: str,
    request_timeout: int,
    mask_threshold: float,
    request_semaphore: asyncio.Semaphore,
) -> str:
    """Segment all tiles of a single slide and save one TIFF mask.

    The mask is written incrementally to a temporary memory-mapped file so
    that the full slide mask never has to be held in RAM. Once complete, the
    mask is saved as ``masks/<slide_id>.tiff``. If the TIFF already exists,
    the slide is skipped.
    """
    if openslide is None:
        raise ImportError(
            "The 'openslide-python' library must be installed to extract tiles from WSI."
        )

    if not Path(slide_path).exists():
        raise FileNotFoundError(f"Slide file not found at: {slide_path}")

    tiff_file = masks_dir / f"{slide_id}.tiff"
    if tiff_file.exists():
        return slide_id

    temp_file = masks_dir / f"{slide_id}.npy.tmp"
    temp_file.unlink(missing_ok=True)

    with openslide.OpenSlide(slide_path) as slide:
        slide_width, slide_height = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        mpp_x, mpp_y = slide_resolution(slide, level)

        mask = open_memmap(
            str(temp_file),
            mode="w+",
            dtype=np.uint8,
            shape=(slide_height, slide_width),
        )
        mask[:] = 0
        mask.flush()

        for _, row in tiles_df.iterrows():
            try:
                tile = extract_tile(
                    slide, row, level, tile_extent_x, tile_extent_y, downsample
                )
                if client.is_closed:
                    print(
                        f"⚠️ Client is closed before tile ({row['x']}, {row['y']}) "
                        f"of slide {slide_id}"
                    )
                    break
                async with request_semaphore:
                    tile_mask = await client.models.segment_image(
                        model_name, tile, timeout=request_timeout
                    )
                binary_mask = extract_epithelium_mask(tile_mask, mask_threshold)
                tile_h, tile_w = binary_mask.shape

                x_l = min(
                    max(int(round(row["x"])), 0),
                    slide_width - tile_w,
                )
                y_l = min(
                    max(int(round(row["y"])), 0),
                    slide_height - tile_h,
                )

                mask[y_l : y_l + tile_h, x_l : x_l + tile_w] = binary_mask
                mask.flush()
            except Exception as e:
                print(
                    f"⚠️ Error processing tile ({row['x']}, {row['y']}) "
                    f"from slide {slide_id}: {e}"
                )
                traceback.print_exc()

    del mask
    memmap_for_write = np.load(str(temp_file), mmap_mode="r")
    vips_image = pyvips.Image.new_from_array(np.array(memmap_for_write))
    write_big_tiff(vips_image, path=tiff_file, mpp_x=mpp_x, mpp_y=mpp_y)
    temp_file.unlink()

    return slide_id


async def process_split(
    split: str,
    split_uri: str,
    output_dir: Path,
    client: AsyncClient,
    config: DictConfig,
    request_semaphore: asyncio.Semaphore,
    slide_semaphore: asyncio.Semaphore,
) -> None:
    """Download one split and save one TIFF mask per slide."""
    local_dir = Path(download_artifacts(artifact_uri=split_uri))

    slides = pd.read_parquet(local_dir / "slides.parquet")
    tiles = pd.read_parquet(local_dir / "tiles.parquet")

    masks_dir = output_dir / split / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    slide_meta = {
        row["id"]: {
            "path": row["path"],
            "level": int(row["level"]),
            "tile_extent_x": int(row["tile_extent_x"]),
            "tile_extent_y": int(row["tile_extent_y"]),
            "mpp_x": float(row["mpp_x"]),
        }
        for _, row in slides.iterrows()
    }

    slide_groups = list(tiles.groupby("slide_id"))
    slide_groups = [
        (slide_id, group) for slide_id, group in slide_groups if slide_id in slide_meta
    ]

    mpp_groups: dict[float, list[tuple[str, pd.DataFrame]]] = {}
    for slide_id, group in slide_groups:
        mpp = slide_meta[slide_id]["mpp_x"]
        mpp_groups.setdefault(mpp, []).append((slide_id, group))

    async def process_one_slide(
        slide_id: str,
        group: pd.DataFrame,
        meta: dict[str, Any],
        masks_dir: Path,
    ) -> str:
        async with slide_semaphore:
            return await process_slide(
                slide_id=slide_id,
                slide_path=meta["path"],
                level=meta["level"],
                tile_extent_x=meta["tile_extent_x"],
                tile_extent_y=meta["tile_extent_y"],
                tiles_df=group,
                masks_dir=masks_dir,
                client=client,
                model_name=config.model_name,
                request_timeout=config.request_timeout,
                mask_threshold=config.mask_threshold,
                request_semaphore=request_semaphore,
            )

    total_processed = 0
    for mpp, mpp_slide_groups in mpp_groups.items():
        masks_dir = output_dir / split / f"masks_mpp_{mpp:.6f}"
        masks_dir.mkdir(parents=True, exist_ok=True)

        tasks = [
            asyncio.create_task(
                process_one_slide(slide_id, group, slide_meta[slide_id], masks_dir)
            )
            for slide_id, group in mpp_slide_groups
        ]

        if not tasks:
            continue

        processed_ids: list[Any] = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            desc=f"Processing {split} mpp={mpp}",
            total=len(tasks),
        ):
            try:
                processed_ids.append(await coro)
            except Exception as exc:
                processed_ids.append(exc)

        successes = [r for r in processed_ids if isinstance(r, str)]
        failures = [r for r in processed_ids if not isinstance(r, str)]
        total_processed += len(successes)

        print(f"Split '{split}' mpp={mpp}: {len(successes)} slides processed")
        if failures:
            print(f"Split '{split}' mpp={mpp}: {len(failures)} slides failed")
            for i, failure in enumerate(failures[:5]):
                print(f"  Failure {i + 1}: {failure}")
                if isinstance(failure, BaseException):
                    traceback.print_exception(
                        type(failure), failure, failure.__traceback__
                    )

    if total_processed == 0:
        raise RuntimeError(f"No slides were successfully processed for split '{split}'")


async def epi_segmentation_main(config: DictConfig, logger: MLFlowLogger) -> None:
    """Async entry point matching the structure of filter_tiles.py."""
    output_dir = Path(config.project_path) / config.mlflow_artifact_path
    output_dir.mkdir(parents=True, exist_ok=True)

    request_semaphore = asyncio.Semaphore(config.max_concurrent)
    slide_semaphore = asyncio.Semaphore(config.get("max_slide_concurrency", 2))

    async with AsyncClient() as client:
        for split in config.splits:
            split_uri = config.mlflow_uris.tiling[split]
            await process_split(
                split=split,
                split_uri=split_uri,
                output_dir=output_dir,
                client=client,
                config=config,
                request_semaphore=request_semaphore,
                slide_semaphore=slide_semaphore,
            )

            mlflow.log_artifacts(str(output_dir), config.mlflow_artifact_path)


# run with +dataset=processed_w_tiling
@with_cli_args(["+preprocessing=epi_segmentation"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    asyncio.run(epi_segmentation_main(config, logger))


if __name__ == "__main__":
    main()
