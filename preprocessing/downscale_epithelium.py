import tempfile
from pathlib import Path
from typing import cast

import hydra
import mlflow.artifacts
import pyvips
import ray
from omegaconf import DictConfig
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


@ray.remote(memory=4 * 1024**3)
def process_slide(slide_path: Path, downscale: int, output_path: Path) -> None:
    image = cast("pyvips.Image", pyvips.Image.new_from_file(slide_path))
    mpp_x, mpp_y = 1000 / image.xres, 1000 / image.yres

    mask = image.shrink(hshrink=1 / downscale, vshrink=1 / downscale, ceil=False)

    write_big_tiff(
        mask,
        path=output_path / slide_path.name,
        mpp_x=mpp_x * downscale,
        mpp_y=mpp_y * downscale,
    )


@with_cli_args(["+preprocessing=downscale_epithelium"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    epithelium_folder = Path(
        mlflow.artifacts.download_artifacts(artifact_uri=config.mlflow_uris.epithelium)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        process_items(
            epithelium_folder.glob("*.tiff"),
            process_item=process_slide,
            fn_kwargs={
                "downscale": config.downscale,
                "output_path": tmpdir_path,
            },
            max_concurrent=config.max_concurrent,
        )

        logger.log_artifacts(str(tmpdir_path), config.mlflow_artifact_path)


if __name__ == "__main__":
    with ray.init(runtime_env={"excludes": [".git", ".venv"]}):  # type: ignore[call-arg]
        main()

