"""End-to-end provenance demo: create → register → verify + train.

Everything logged to MLflow is what the external provenance tool consumes:
user/dataset registries, environment, job limits (on the cluster), the
flattened Hydra config, dataset verification results and metrics.

Steps (``demo.step``):

    create    write a tiny two-class dataset to ``data_path``
    register  record file sizes/mtimes in MLflow's Dataset_Registry (once)
    train     verify the data on disk matches MLflow, then fit a tiny model

    # once (locally with the share mounted, or as a job):
    uv run python -m provenance_demo +experiment=provenance_demo demo.step=create
    uv run python -m provenance_demo +experiment=provenance_demo demo.step=register

    # then verify + train (locally, or via scripts/provenance_demo.py):
    uv run python -m provenance_demo +experiment=provenance_demo demo.step=train
"""

import os
import random

import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.provenance import log_provenance, register_dataset, verify_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score


def _create_dataset(data_path: str, files_per_class: int) -> None:
    """Write manifest.csv (wsi_path, cancer) + dummy slide files.

    Positive-class files are ~10x larger, so the demo model has a learnable
    signal from file stats alone. Replace with your real dataset folder
    (same manifest format) to demo on actual data.
    """
    rows = []
    rng = random.Random(0)
    for label, sub, size in ((0, "neg", 1_000), (1, "pos", 10_000)):
        os.makedirs(os.path.join(data_path, sub), exist_ok=True)
        for i in range(files_per_class):
            # shortcut: unique basenames required — registry keys file sizes by name
            name = f"{sub}_{i:03d}.dat"
            with open(os.path.join(data_path, sub, name), "wb") as f:
                f.write(rng.randbytes(size + i))
            rows.append({"wsi_path": f"{sub}/{name}", "cancer": label})
    pd.DataFrame(rows).to_csv(
        os.path.join(data_path, "manifest.csv"), index=False
    )
    print(f"[provenance_demo] wrote {len(rows)} files + manifest.csv → {data_path}")


@hydra.main(config_path="configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    data_path = str(cfg.data_path)
    manifest = os.path.join(data_path, "manifest.csv")

    if cfg.demo.step == "create":
        _create_dataset(data_path, int(cfg.demo.files_per_class))

    elif cfg.demo.step == "register":
        register_dataset(
            dataset_dir=data_path,
            dataset_name=str(cfg.demo.dataset_name),
            version=str(cfg.demo.dataset_version),
        )

    elif cfg.demo.step == "train":
        # 1) Does the data on disk still match what we registered in MLflow?
        verification = verify_dataset(
            manifest_path=manifest, data_root=data_path
        )

        mlflow.set_experiment(str(cfg.metadata.experiment_name))
        with mlflow.start_run(run_name=str(cfg.metadata.run_name)):
            # 2) Environment + job limits + the whole Hydra config as params.
            log_provenance(config=cfg, name=str(cfg.demo.dataset_name))

            mlflow.set_tags(
                {
                    "dataset_verified": str(verification["verified"]),
                    "dataset_run_id": str(verification["dataset_run_id"]),
                }
            )
            mlflow.log_params(
                {
                    "dataset_files_total": verification["files_total"],
                    "dataset_files_missing": verification["files_missing"],
                }
            )

            # 3) Simple ML: logistic regression on file size.
            samples = pd.read_csv(manifest)
            features, labels = [], []
            for _, row in samples.iterrows():
                path = str(row["wsi_path"])
                path = path if os.path.isabs(path) else os.path.join(data_path, path)
                features.append([os.stat(path).st_size])
                labels.append(int(row["cancer"]))
            X, y = np.asarray(features), np.asarray(labels)

            model = LogisticRegression().fit(X, y)
            mlflow.log_metrics(
                {
                    "train_accuracy": float(accuracy_score(y, model.predict(X))),
                    "train_auc": float(
                        roc_auc_score(y, model.predict_proba(X)[:, 1])
                    ),
                    "cv_accuracy": float(cross_val_score(model, X, y, cv=3).mean()),
                }
            )
            mlflow.sklearn.log_model(model, name="model")
            print(
                f"[provenance_demo] dataset_verified={verification['verified']} "
                f"→ run in experiment '{cfg.metadata.experiment_name}'"
            )


if __name__ == "__main__":
    main()
