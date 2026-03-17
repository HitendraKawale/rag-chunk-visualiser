from __future__ import annotations

from pathlib import Path

import lancedb
import numpy as np
import pandas as pd

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)


class VectorStoreError(RuntimeError):
    pass


def resolve_db_path(db_path: str | Path | None = None) -> str:
    return str(Path(db_path) if db_path is not None else settings.lancedb_dir)


def validate_chunks_and_embeddings(
    chunks: list[dict],
    embedding_matrix: np.ndarray | None,
) -> None:
    if not chunks:
        raise VectorStoreError("No chunks available for LanceDB storage.")

    if embedding_matrix is None:
        raise VectorStoreError("No embedding matrix available for LanceDB storage.")

    if embedding_matrix.ndim != 2:
        raise VectorStoreError(
            f"Expected a 2D embedding matrix, but got shape {embedding_matrix.shape}."
        )

    if len(chunks) != embedding_matrix.shape[0]:
        raise VectorStoreError("Chunk count does not match embedding row count.")


def get_db_connection(db_path: str | Path | None = None):
    resolved_path = resolve_db_path(db_path)
    try:
        return lancedb.connect(resolved_path)
    except Exception as exc:
        raise VectorStoreError(f"Failed to connect to LanceDB at '{resolved_path}'.") from exc


def build_chunk_dataframe(
    chunks: list[dict],
    embedding_matrix: np.ndarray,
) -> pd.DataFrame:
    validate_chunks_and_embeddings(chunks, embedding_matrix)

    rows = []
    for chunk, vector in zip(chunks, embedding_matrix, strict=True):
        rows.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "char_count": chunk["char_count"],
                "word_count": chunk["word_count"],
                "preview": chunk["preview"],
                "text": chunk["text"],
                "vector": vector.astype(np.float32).tolist(),
            }
        )

    return pd.DataFrame(rows)


def write_chunk_embeddings(
    chunks: list[dict],
    embedding_matrix: np.ndarray,
    table_name: str | None = None,
    db_path: str | Path | None = None,
) -> dict:
    resolved_table_name = table_name or settings.lancedb_table_name
    resolved_db_path = resolve_db_path(db_path)

    dataframe = build_chunk_dataframe(chunks, embedding_matrix)
    db = get_db_connection(resolved_db_path)

    try:
        db.create_table(
            resolved_table_name,
            data=dataframe,
            mode="overwrite",
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to write embeddings into LanceDB table '{resolved_table_name}'."
        ) from exc

    summary = {
        "table_name": resolved_table_name,
        "db_path": resolved_db_path,
        "row_count": int(len(dataframe)),
        "vector_dim": int(embedding_matrix.shape[1]),
        "search_metric": "dot",
        "write_mode": "overwrite",
    }

    logger.info(
        "Stored chunk embeddings in LanceDB | table=%s | rows=%s | dim=%s",
        summary["table_name"],
        summary["row_count"],
        summary["vector_dim"],
    )

    return summary


def open_chunks_table(
    table_name: str | None = None,
    db_path: str | Path | None = None,
):
    resolved_table_name = table_name or settings.lancedb_table_name
    db = get_db_connection(db_path)

    try:
        return db.open_table(resolved_table_name)
    except Exception as exc:
        raise VectorStoreError(f"Failed to open LanceDB table '{resolved_table_name}'.") from exc


def fetch_table_preview(
    limit: int = 10,
    table_name: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    if limit <= 0:
        raise VectorStoreError("Preview limit must be greater than 0.")

    table = open_chunks_table(table_name=table_name, db_path=db_path)

    try:
        dataframe = table.to_pandas()
    except Exception as exc:
        raise VectorStoreError("Failed to read LanceDB table preview.") from exc

    preferred_columns = [
        "chunk_id",
        "filename",
        "chunk_index",
        "char_count",
        "word_count",
        "preview",
    ]
    available_columns = [column for column in preferred_columns if column in dataframe.columns]

    return dataframe.loc[:, available_columns].head(limit).reset_index(drop=True)


def search_similar_chunks(
    query_vector: np.ndarray,
    top_k: int,
    exclude_chunk_id: str | None = None,
    table_name: str | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    if query_vector.ndim != 1:
        raise VectorStoreError(f"Expected a 1D query vector, but got shape {query_vector.shape}.")

    if top_k <= 0:
        raise VectorStoreError("top_k must be greater than 0.")

    table = open_chunks_table(table_name=table_name, db_path=db_path)
    search_limit = top_k + 1 if exclude_chunk_id else top_k

    try:
        results = (
            table.search(query_vector.astype(np.float32))
            .distance_type("dot")
            .limit(search_limit)
            .to_pandas()
        )
    except Exception as exc:
        raise VectorStoreError("LanceDB vector search failed.") from exc

    if exclude_chunk_id and "chunk_id" in results.columns:
        results = results[results["chunk_id"] != exclude_chunk_id]

    results = results.head(top_k).reset_index(drop=True)

    logger.info(
        "LanceDB similarity search complete | results=%s | top_k=%s | excluded=%s",
        len(results),
        top_k,
        exclude_chunk_id,
    )

    return results
