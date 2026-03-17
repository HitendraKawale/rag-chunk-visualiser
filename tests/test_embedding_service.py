import numpy as np
import pytest

from rag_chunk_visualizer.services.embedding_service import (
    EmbeddingError,
    summarize_embedding_matrix,
    validate_embedding_inputs,
)


def test_validate_embedding_inputs_rejects_empty_list() -> None:
    with pytest.raises(EmbeddingError):
        validate_embedding_inputs([], batch_size=32)


def test_validate_embedding_inputs_rejects_blank_text() -> None:
    with pytest.raises(EmbeddingError):
        validate_embedding_inputs(["hello", "   "], batch_size=32)


def test_validate_embedding_inputs_rejects_bad_batch_size() -> None:
    with pytest.raises(EmbeddingError):
        validate_embedding_inputs(["hello"], batch_size=0)


def test_summarize_embedding_matrix_returns_expected_shape_info() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    summary = summarize_embedding_matrix(matrix)

    assert summary["count"] == 2
    assert summary["dimension"] == 3
    assert summary["dtype"] == "float32"
    assert summary["mean_norm"] == 1.0
    assert summary["min_norm"] == 1.0
    assert summary["max_norm"] == 1.0


def test_summarize_embedding_matrix_rejects_non_2d_input() -> None:
    with pytest.raises(EmbeddingError):
        summarize_embedding_matrix(np.array([1.0, 2.0, 3.0], dtype=np.float32))
