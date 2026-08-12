from pathlib import Path
from random import randint

import hydra
import mlflow
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import Trainer, autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml._mlflow_compat import apply_mlflow_compat_patch
from ml.data import DataModule
from ml.predict_output import save_predictions


OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    apply_mlflow_compat_patch()

    seed_everything(config.seed, workers=True)

    data = hydra.utils.instantiate(
        config.datamodule,
        _recursive_=False,  # to avoid instantiating all the datasets
        _target_=DataModule,
    )
    model = hydra.utils.instantiate(config.model)

    trainer = hydra.utils.instantiate(config.trainer, _target_=Trainer, logger=logger)
    ckpt = config.checkpoint
    if ckpt is not None and str(ckpt).startswith("mlflow-artifacts:/"):
        ckpt = mlflow.artifacts.download_artifacts(str(ckpt))

    if config.mode == "predict":
        model.predict_metadata = []
        outputs = trainer.predict(model, datamodule=data, ckpt_path=ckpt)
        if "predict_output" in config and config.predict_output is not None:
            save_predictions(
                outputs,
                model.predict_metadata,
                config.predict_output,
                config.label_mode,
            )
            logger.log_artifact(config.predict_output, artifact_path="predictions")
            run = logger.experiment.get_run(logger.run_id)
            uri = (
                f"mlflow-artifacts:/{run.info.experiment_id}/{logger.run_id}"
                f"/artifacts/predictions/{Path(config.predict_output).name}"
            )
            print(f"Predictions stored in MLflow: {uri}", flush=True)
    else:
        getattr(trainer, config.mode)(model, datamodule=data, ckpt_path=ckpt)
    mlflow.end_run()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
