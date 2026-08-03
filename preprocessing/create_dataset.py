"""Create the dysplasia dataset and register provenance."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.mlkit.provenance import (
    build_dataset_prov,
    capture_environment,
)


def extract_case_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid slide/annotation name format: {stem}")
    return f"{parts[0]}/{parts[1]}"


def get_df(
    folder_path: Path, pattern: re.Pattern[str], key: str, ext: str
) -> pd.DataFrame:
    data = []
    for slide_path in folder_path.glob(ext):
        if not pattern.search(slide_path.name):
            continue

        case_id = extract_case_id(slide_path.stem)
        data.append(
            {
                "slide_id": slide_path.stem,
                "case_id": case_id,
                f"{key}_path": str(slide_path.absolute()),
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["slide_id", "case_id", f"{key}_path"]).set_index(
            "slide_id"
        )
    return df.set_index("slide_id")


def load_selected_cases(excel_path: str, label: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, header=None)
    cases = df.iloc[:, 1].dropna().astype(str).str.strip()
    cases = cases[cases != "Bs"].str.replace("_", "/", regex=False).tolist()
    return pd.DataFrame({"case_id": cases, "clarity": label})


def create_dataset(
    slides_path: str,
    annot_path: str,
    selected_clear: str,
    selected_unclear: str,
    pattern_str: str,
) -> tuple[pd.DataFrame, list[str]]:
    slides_df = get_df(
        Path(slides_path), re.compile(pattern_str), key="slide", ext="*.czi"
    )
    annot_df = get_df(
        Path(annot_path), re.compile(pattern_str), key="annot", ext="*.json"
    )

    clear_cases = load_selected_cases(selected_clear, "clear")
    unclear_cases = load_selected_cases(selected_unclear, "unclear")
    selected_cases = pd.concat([clear_cases, unclear_cases], ignore_index=True)

    dataset_df = slides_df.join(annot_df, how="outer", rsuffix="_drop")
    dataset_df["case_id"] = dataset_df["case_id"].fillna(dataset_df["case_id_drop"])

    dataset_df = dataset_df.reset_index().merge(
        selected_cases, on="case_id", how="right"
    )

    missing_slides = (
        dataset_df[dataset_df["slide_path"].isna()]["case_id"].dropna().to_list()
    )

    dataset_df = dataset_df.dropna(subset=["slide_path"])

    dataset_df["annot_path"] = dataset_df["annot_path"].fillna("NEGATIVE")

    dataset_df = dataset_df.set_index("slide_id")
    dataset_df = dataset_df[["case_id", "slide_path", "annot_path", "clarity"]]

    return dataset_df, missing_slides


@with_cli_args(["+preprocessing=create_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    # ── Environment + user provenance ────────────────────────
    capture_environment(snapshot_env=True)

    dataset, missing_slides = create_dataset(
        config.data_path,
        config.dataset.annot_path,
        config.dataset.selected_clear_cases,
        config.dataset.selected_unclear_cases,
        config.dataset.regex_pattern,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        output_path = tmpdir_path / "dataset.csv"
        dataset.to_csv(output_path, index=True)
        logger.log_artifact(str(output_path))

        def _log_missing_items(items: list[str], filename: str) -> None:
            file_path = tmpdir_path / filename
            file_path.write_text("\n".join(items) + "\n")
            logger.log_artifact(str(file_path))

        _log_missing_items(missing_slides, "missing_slides.txt")

        # ── Log dataset metadata as tags ─────────────────────
        num_samples = len(dataset)
        num_positive = int((dataset["annot_path"] != "NEGATIVE").sum())
        num_negative = num_samples - num_positive

        file_sizes: dict[str, int] = {}
        for _, row in dataset.iterrows():
            basename = os.path.basename(row["slide_path"])
            fpath = Path(row["slide_path"])
            file_sizes[basename] = int(fpath.stat().st_size) if fpath.exists() else -1

        mlflow.log_params(
            {
                "num_samples": num_samples,
                "num_positive": num_positive,
                "num_negative": num_negative,
            }
        )
        mlflow.set_tags(
            {
                "dataset_name": "ulcerative-colitis-dysplasia",
                "version": "1.0.0",
                "file_sizes": json.dumps(file_sizes),
            }
        )

        # ── Build and log PROV document ──────────────────────
        active_run = mlflow.active_run()
        assert active_run is not None
        run_id = active_run.info.run_id

        prov_path = tmpdir_path / "provenance" / "prov.json"
        prov_path.parent.mkdir(exist_ok=True)
        prov_doc = build_dataset_prov(
            run_id=run_id,
            dataset_name="ulcerative-colitis-dysplasia",
            version="1.0.0",
            dataset_root=str(config.data_path),
            num_samples=num_samples,
            num_positive=num_positive,
            num_negative=num_negative,
            file_sizes=file_sizes,
            manifest_path=str(output_path),
        )
        prov_path.write_text(json.dumps(prov_doc, indent=2))
        logger.log_artifact(str(prov_path), artifact_path="provenance")


if __name__ == "__main__":
    main()
