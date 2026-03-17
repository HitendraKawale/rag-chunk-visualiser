import numpy as np
import pytest

from rag_chunk_visualizer.visualization.projection import (
    ProjectionError,
    build_projection_dataframe,
    project_chunks,
    project_single_point,
    transform_query_vector,
)


def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "doc_a_chunk_000",
            "doc_id": "doc_a",
            "filename": "a.txt",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 20,
            "char_count": 20,
            "word_count": 4,
            "preview": "alpha beta gamma",
            "text": "alpha beta gamma delta",
        },
        {
            "chunk_id": "doc_a_chunk_001",
            "doc_id": "doc_a",
            "filename": "a.txt",
            "chunk_index": 1,
            "char_start": 20,
            "char_end": 40,
            "char_count": 20,
            "word_count": 4,
            "preview": "epsilon zeta eta",
            "text": "epsilon zeta eta theta",
        },
    ]


def test_project_single_point_returns_origin() -> None:
    coords, summary, model = project_single_point()

    assert coords.shape == (1, 2)
    assert coords[0, 0] == 0.0
    assert coords[0, 1] == 0.0
    assert summary["method"] == "single-point"
    assert model["kind"] == "single-point"


def test_project_chunks_with_pca_returns_projection_model() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    chunks = sample_chunks() * 2
    projection_df, summary, model = project_chunks(chunks, matrix, method="pca")

    assert len(projection_df) == 4
    assert summary["method"] == "pca"
    assert hasattr(model, "transform")


def test_build_projection_dataframe_rejects_mismatch() -> None:
    chunks = sample_chunks()
    coords = np.array([[0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ProjectionError):
        build_projection_dataframe(chunks, coords)


def test_transform_query_vector_returns_2d_coordinates() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    chunks = sample_chunks() * 2
    _, summary, model = project_chunks(chunks, matrix, method="pca")

    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query_point = transform_query_vector(query_vector, model, summary)

    assert "x" in query_point
    assert "y" in query_point
