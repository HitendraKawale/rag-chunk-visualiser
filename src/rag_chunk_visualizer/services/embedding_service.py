from __future__ import annotations

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


def validate_embedding_inputs(texts: list[str], batch_size: int) -> None:
    if not texts:
        raise EmbeddingError("No texts provided for embedding.")

    if batch_size <= 0:
        raise EmbeddingError("Embedding batch size must be greater than 0.")

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(f"Text at position {index} is empty or invalid.")


@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    logger.info(
        "Loading embedding model | model=%s | device=%s",
        model_name,
        device,
    )

    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to load embedding model '{model_name}' on device '{device}'."
        ) from exc


def _encode_documents(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    encode_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }

    if hasattr(model, "encode_document"):
        embeddings = model.encode_document(texts, **encode_kwargs)
    else:
        embeddings = model.encode(texts, **encode_kwargs)

    return np.asarray(embeddings, dtype=np.float32)


def _encode_query(
    model: SentenceTransformer,
    query_text: str,
) -> np.ndarray:
    encode_kwargs = {
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }

    if hasattr(model, "encode_query"):
        embedding = model.encode_query(query_text, **encode_kwargs)
    else:
        embedding = model.encode([query_text], **encode_kwargs)[0]

    return np.asarray(embedding, dtype=np.float32)


def embed_texts(
    texts: list[str],
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    resolved_model_name = model_name or settings.embedding_model
    resolved_device = device or settings.embedding_device
    resolved_batch_size = batch_size or settings.embedding_batch_size

    validate_embedding_inputs(texts, resolved_batch_size)

    model = load_embedding_model(
        model_name=resolved_model_name,
        device=resolved_device,
    )

    try:
        embeddings = _encode_documents(
            model=model,
            texts=texts,
            batch_size=resolved_batch_size,
        )
    except Exception as exc:
        raise EmbeddingError("Embedding generation failed.") from exc

    matrix = np.asarray(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise EmbeddingError(f"Expected a 2D embedding matrix, but got shape {matrix.shape}.")

    logger.info(
        "Generated embeddings | texts=%s | dim=%s",
        matrix.shape[0],
        matrix.shape[1],
    )

    return matrix


def embed_chunks(chunks: list[dict]) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]
    return embed_texts(texts=texts)


def embed_query_text(query_text: str) -> np.ndarray:
    validate_embedding_inputs([query_text], batch_size=1)

    model = load_embedding_model(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )

    try:
        query_vector = _encode_query(model=model, query_text=query_text)
    except Exception as exc:
        raise EmbeddingError("Query embedding generation failed.") from exc

    if query_vector.ndim != 1:
        raise EmbeddingError(f"Expected a 1D query embedding, but got shape {query_vector.shape}.")

    logger.info("Generated query embedding | dim=%s", query_vector.shape[0])

    return query_vector


def summarize_embedding_matrix(matrix: np.ndarray) -> dict:
    if matrix.ndim != 2:
        raise EmbeddingError(f"Expected a 2D embedding matrix, but got shape {matrix.shape}.")

    norms = np.linalg.norm(matrix, axis=1)

    return {
        "count": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "dtype": str(matrix.dtype),
        "mean_norm": round(float(norms.mean()), 6),
        "min_norm": round(float(norms.min()), 6),
        "max_norm": round(float(norms.max()), 6),
    }
