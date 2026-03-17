from __future__ import annotations

import numpy as np
import pandas as pd

from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)


class RetrievalError(RuntimeError):
    pass


def validate_query_text(query_text: str) -> None:
    if not isinstance(query_text, str) or not query_text.strip():
        raise RetrievalError("Enter a non-empty query before running retrieval.")


def build_embedding_lookup(
    chunks: list[dict],
    embedding_matrix: np.ndarray | None,
) -> dict[str, np.ndarray]:
    if embedding_matrix is None:
        raise RetrievalError("No embedding matrix available for retrieval enrichment.")

    if embedding_matrix.ndim != 2:
        raise RetrievalError(
            f"Expected a 2D embedding matrix, but got shape {embedding_matrix.shape}."
        )

    if len(chunks) != embedding_matrix.shape[0]:
        raise RetrievalError("Chunk count does not match embedding row count.")

    lookup: dict[str, np.ndarray] = {}
    for chunk, vector in zip(chunks, embedding_matrix, strict=True):
        lookup[chunk["chunk_id"]] = np.asarray(vector, dtype=np.float32)

    return lookup


def enrich_query_results(
    results_df: pd.DataFrame,
    chunks: list[dict],
    embedding_matrix: np.ndarray | None,
    query_vector: np.ndarray,
) -> list[dict]:
    if results_df.empty:
        return []

    query_vector = np.asarray(query_vector, dtype=np.float32)
    if query_vector.ndim != 1:
        raise RetrievalError(f"Expected a 1D query vector, but got shape {query_vector.shape}.")

    embedding_lookup = build_embedding_lookup(chunks, embedding_matrix)

    enriched_rows: list[dict] = []
    for rank, row in enumerate(results_df.to_dict(orient="records"), start=1):
        chunk_id = row.get("chunk_id")
        if chunk_id not in embedding_lookup:
            continue

        similarity_score = float(np.dot(query_vector, embedding_lookup[chunk_id]))

        enriched_rows.append(
            {
                **row,
                "rank": rank,
                "similarity_score": round(similarity_score, 6),
            }
        )

    logger.info("Enriched retrieval results | count=%s", len(enriched_rows))
    return enriched_rows


def get_retrieved_chunk_ids(retrieval_results: list[dict]) -> list[str]:
    return [result["chunk_id"] for result in retrieval_results if "chunk_id" in result]
