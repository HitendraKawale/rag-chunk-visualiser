from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)


class ProjectionError(RuntimeError):
    pass


def validate_projection_inputs(matrix: np.ndarray | None, method: str) -> None:
    if matrix is None:
        raise ProjectionError("No embedding matrix available for projection.")

    if matrix.ndim != 2:
        raise ProjectionError(f"Expected a 2D embedding matrix, but got shape {matrix.shape}.")

    if matrix.shape[0] == 0:
        raise ProjectionError("Embedding matrix has no rows.")

    if method not in {"pca", "umap"}:
        raise ProjectionError(
            f"Unsupported projection method '{method}'. Expected 'pca' or 'umap'."
        )


def project_single_point() -> tuple[np.ndarray, dict, dict]:
    coordinates = np.array([[0.0, 0.0]], dtype=np.float32)
    summary = {
        "method": "single-point",
        "point_count": 1,
        "x_label": "X",
        "y_label": "Y",
        "note": "Only one chunk available, so the point is placed at the origin.",
    }
    projection_model = {"kind": "single-point"}
    return coordinates, summary, projection_model


def fit_pca_projector(
    matrix: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, dict, PCA]:
    projector = PCA(n_components=2, random_state=random_state)
    coordinates = projector.fit_transform(matrix).astype(np.float32)

    explained = projector.explained_variance_ratio_.tolist()

    summary = {
        "method": "pca",
        "point_count": int(matrix.shape[0]),
        "x_label": "PC1",
        "y_label": "PC2",
        "explained_variance_ratio_pc1": round(float(explained[0]), 6),
        "explained_variance_ratio_pc2": round(float(explained[1]), 6),
    }

    return coordinates, summary, projector


def fit_umap_projector(
    matrix: np.ndarray,
    random_state: int,
    n_neighbors: int,
    min_dist: float,
):
    try:
        import umap
    except ImportError as exc:
        raise ProjectionError(
            "UMAP is not installed. Install 'umap-learn' or switch to PCA."
        ) from exc

    effective_n_neighbors = min(n_neighbors, max(2, matrix.shape[0] - 1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )

    coordinates = reducer.fit_transform(matrix).astype(np.float32)

    summary = {
        "method": "umap",
        "point_count": int(matrix.shape[0]),
        "x_label": "UMAP-1",
        "y_label": "UMAP-2",
        "umap_n_neighbors": int(effective_n_neighbors),
        "umap_min_dist": float(min_dist),
        "umap_metric": "cosine",
    }

    return coordinates, summary, reducer


def fit_projection_model(
    matrix: np.ndarray,
    method: str | None = None,
    random_state: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
):
    resolved_method = method or settings.projection_method
    resolved_random_state = (
        settings.projection_random_state if random_state is None else random_state
    )
    resolved_umap_n_neighbors = (
        settings.umap_n_neighbors if umap_n_neighbors is None else umap_n_neighbors
    )
    resolved_umap_min_dist = settings.umap_min_dist if umap_min_dist is None else umap_min_dist

    validate_projection_inputs(matrix, resolved_method)

    if matrix.shape[0] == 1:
        return project_single_point()

    if resolved_method == "umap" and matrix.shape[0] < 4:
        logger.info("Falling back to PCA because the dataset is too small for UMAP.")
        coordinates, summary, projector = fit_pca_projector(
            matrix=matrix,
            random_state=resolved_random_state,
        )
        summary["fallback_from"] = "umap"
        summary["fallback_reason"] = "too_few_points"
        return coordinates, summary, projector

    if resolved_method == "pca":
        coordinates, summary, projector = fit_pca_projector(
            matrix=matrix,
            random_state=resolved_random_state,
        )
    else:
        coordinates, summary, projector = fit_umap_projector(
            matrix=matrix,
            random_state=resolved_random_state,
            n_neighbors=resolved_umap_n_neighbors,
            min_dist=resolved_umap_min_dist,
        )

    logger.info(
        "Projected embedding matrix | method=%s | points=%s",
        summary["method"],
        summary["point_count"],
    )

    return coordinates, summary, projector


def transform_query_vector(
    query_vector: np.ndarray,
    projection_model,
    projection_summary: dict,
) -> dict:
    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.ndim != 1:
        raise ProjectionError(f"Expected a 1D query vector, but got shape {query_vector.shape}.")

    method = projection_summary.get("method")
    if method == "single-point":
        return {
            "x": 0.0,
            "y": 0.0,
        }

    if projection_model is None:
        raise ProjectionError("No fitted projection model available for query projection.")

    try:
        coordinates = projection_model.transform(query_vector.reshape(1, -1))
    except Exception as exc:
        raise ProjectionError("Failed to project query vector into the existing 2D space.") from exc

    coordinates = np.asarray(coordinates, dtype=np.float32)

    return {
        "x": round(float(coordinates[0, 0]), 8),
        "y": round(float(coordinates[0, 1]), 8),
    }


def build_projection_dataframe(
    chunks: list[dict],
    coordinates: np.ndarray,
) -> pd.DataFrame:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ProjectionError(
            f"Expected 2D coordinates with shape (n, 2), got {coordinates.shape}."
        )

    if len(chunks) != coordinates.shape[0]:
        raise ProjectionError("Chunk count does not match projected coordinate count.")

    rows = []
    for plot_index, (chunk, coord) in enumerate(zip(chunks, coordinates, strict=True)):
        rows.append(
            {
                "plot_index": plot_index,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"],
                "preview": chunk["preview"],
                "text": chunk["text"],
                "x": round(float(coord[0]), 8),
                "y": round(float(coord[1]), 8),
            }
        )

    return pd.DataFrame(rows)


def project_chunks(
    chunks: list[dict],
    embedding_matrix: np.ndarray,
    method: str | None = None,
):
    if not chunks:
        raise ProjectionError("No chunks available for projection.")

    coordinates, summary, projection_model = fit_projection_model(
        matrix=embedding_matrix,
        method=method,
    )
    projection_df = build_projection_dataframe(
        chunks=chunks,
        coordinates=coordinates,
    )

    return projection_df, summary, projection_model
