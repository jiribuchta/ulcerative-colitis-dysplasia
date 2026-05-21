import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dysplasia.typing import (
    SlideEmbeddingsSample,
)


class DysplasiaEmbeddingsDataset(Dataset[SlideEmbeddingsSample]):
    def __init__(
        self,
        uri: str | None = None,
        embeddings_dir: str | Path = "",
        target_labels: list[str] | None = None,
        threshold: float = 0.0,
        label_mode: str = "binary_or",
        label_scale: float | None = None,
        clamp_labels: bool = True,
        cache_size: int = 8,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.label_mode = label_mode
        self.label_scale = label_scale
        self.clamp_labels = clamp_labels
        self.embeddings_dir = Path(embeddings_dir)
        self._cache_size = cache_size

        self.target_labels = target_labels

        if uri is not None:
            local_path = Path(self._resolve_uri(uri))

            tiles_df = pd.read_parquet(local_path / "tiles.parquet")
            slides_df = pd.read_parquet(local_path / "slides.parquet")
            self.data = tiles_df.merge(
                slides_df, left_on="slide_id", right_on="id", how="left"
            )

        embedding_files: dict[str, Path] = {
            p.stem: p for p in self.embeddings_dir.glob("*.parquet")
        }

        required_slides = set(self.data["slide_id"].astype(str).unique())

        stem_by_slide_id: dict[str, str] = {}
        if "path" in self.data.columns:
            meta = (
                self.data[["slide_id", "path"]]
                .dropna(subset=["path"])
                .drop_duplicates(subset=["slide_id"])
            )
            for _, r in meta.iterrows():
                try:
                    stem_by_slide_id[str(r["slide_id"])] = Path(str(r["path"])).stem
                except (TypeError, ValueError):
                    continue

        self._slide_paths: dict[str, Path] = {}
        missing_slide_ids: set[str] = set()
        for slide_id in required_slides:
            if slide_id in embedding_files:
                self._slide_paths[slide_id] = embedding_files[slide_id]
                continue

            stem = stem_by_slide_id.get(slide_id)
            if stem is not None and stem in embedding_files:
                self._slide_paths[slide_id] = embedding_files[stem]
                continue

            missing_slide_ids.add(slide_id)

        # FOR QUICK TEST UNCOMMENT WHATEVER THIS IS, FILTERS THE CREATED EMBEDDINGS

        # if missing_slide_ids:
        #     # TEMP (2026-05-21): We are generating embeddings incrementally.
        #     # For now, filter the dataset to *only* slides that already have an
        #     # embeddings parquet on disk.
        #     # DELETE THIS BLOCK once embeddings exist for all slides.
        #     keep_slide_ids = set(self._slide_paths.keys())
        #     if not keep_slide_ids:
        #         # TEMP (2026-05-21): Allow empty splits (typically val/test)
        #         # while embeddings are still being generated.
        #         # DELETE THIS BLOCK once embeddings exist for all slides.
        #         warnings.warn(
        #             "TEMP: no embeddings found for any slide in this split; "
        #             "returning an empty dataset. "
        #             f"Embeddings dir: {self.embeddings_dir}",
        #             stacklevel=2,
        #         )
        #         self.data = self.data.iloc[0:0].copy()
        #         keep_slide_ids = set()
        #     if keep_slide_ids:
        #         before_rows = len(self.data)
        #         self.data = self.data[
        #             self.data["slide_id"].astype(str).isin(keep_slide_ids)
        #         ].reset_index(drop=True)
        #         warnings.warn(
        #             "TEMP: filtering to processed slides only. "
        #             f"Dropped {len(missing_slide_ids)} slide(s) without embeddings. "
        #             f"Tiles: {before_rows} -> {len(self.data)}",
        #             stacklevel=2,
        #         )

        #     # TEMP (2026-05-21): Also filter out tile coords that are missing
        #     # from the slide's embeddings parquet to avoid runtime KeyErrors.
        #     # DELETE THIS BLOCK once embeddings are generated from the exact
        #     # same tiling/filtered-tiles output.
        #     if keep_slide_ids and len(self.data) > 0:
        #         before_rows = len(self.data)
        #         filtered_parts: list[pd.DataFrame] = []
        #         for sid in sorted(keep_slide_ids):
        #             part = self.data[self.data["slide_id"].astype(str) == sid]
        #             emb_xy = pd.read_parquet(self._slide_paths[sid], columns=["x", "y"])
        #             emb_xy = emb_xy.drop_duplicates(subset=["x", "y"])
        #             part = part.merge(emb_xy, on=["x", "y"], how="inner")
        #             filtered_parts.append(part)

        #         if filtered_parts:
        #             self.data = pd.concat(filtered_parts, ignore_index=True)
        #         warnings.warn(
        #             "TEMP: filtered tiles to coordinates with available embeddings. "
        #             f"Tiles: {before_rows} -> {len(self.data)}",
        #             stacklevel=2,
        #         )
        #         if len(self.data) == 0:
        #             raise ValueError(
        #                 "No tiles remain after filtering to available embeddings. "
        #                 "This usually means the embeddings were computed with a different tiling grid."
        #             )

        # LRU cache: slide_id -> {(x, y): np.ndarray}
        self._cache: dict[str, dict[tuple[int, int], np.ndarray]] = {}

    @staticmethod
    def _resolve_uri(uri: str) -> Path:
        if uri.startswith("mlflow-artifacts:/"):
            import mlflow

            return Path(mlflow.artifacts.download_artifacts(uri))
        return Path(uri)

    def _load_slide(self, slide_id: str) -> dict[tuple[int, int], np.ndarray]:
        if slide_id in self._cache:
            return self._cache[slide_id]

        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))

        df = pd.read_parquet(self._slide_paths[slide_id])
        slide_map = {
            (int(r["x"]), int(r["y"])): np.array(r["embedding"], dtype=np.float32)
            for _, r in df.iterrows()
        }
        self._cache[slide_id] = slide_map
        return slide_map

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> SlideEmbeddingsSample:
        row = self.data.iloc[idx]
        slide_id = str(row["slide_id"])
        x = int(row["x"])
        y = int(row["y"])

        slide_embs = self._load_slide(slide_id)
        emb_array = slide_embs.get((x, y))
        embedding = torch.from_numpy(emb_array)

        metadata: dict[str, str] = {"slide_id": slide_id, "x": str(x), "y": str(y)}

        values: list[float] = []
        for col in self.target_labels:
            v = float(row[col])
            values.append(v)

        label = torch.tensor(values, dtype=torch.float32)
        if self.clamp_labels:
            label = label.clamp(0.0, 1.0)

        return embedding, label, metadata
