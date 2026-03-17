import numpy as np

from rag_chunk_visualizer.storage.lancedb_store import (
    build_chunk_dataframe,
    fetch_table_preview,
    search_similar_chunks,
    write_chunk_embeddings,
)


def normalized_matrix() -> np.ndarray:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


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
            "char_start": 15,
            "char_end": 35,
            "char_count": 20,
            "word_count": 4,
            "preview": "alpha beta variant",
            "text": "alpha beta variant delta",
        },
        {
            "chunk_id": "doc_b_chunk_000",
            "doc_id": "doc_b",
            "filename": "b.txt",
            "chunk_index": 0,
            "char_start": 0,
            "char_end": 20,
            "char_count": 20,
            "word_count": 4,
            "preview": "theta lambda omega",
            "text": "theta lambda omega sigma",
        },
    ]


def test_build_chunk_dataframe_returns_expected_columns() -> None:
    chunks = sample_chunks()
    matrix = normalized_matrix()

    dataframe = build_chunk_dataframe(chunks, matrix)

    assert len(dataframe) == 3
    assert "vector" in dataframe.columns
    assert "chunk_id" in dataframe.columns
    assert isinstance(dataframe.iloc[0]["vector"], list)
    assert len(dataframe.iloc[0]["vector"]) == 3


def test_write_and_search_lancedb(tmp_path) -> None:
    chunks = sample_chunks()
    matrix = normalized_matrix()

    summary = write_chunk_embeddings(
        chunks=chunks,
        embedding_matrix=matrix,
        table_name="test_chunks",
        db_path=tmp_path,
    )

    assert summary["row_count"] == 3
    assert summary["vector_dim"] == 3
    assert summary["search_metric"] == "dot"

    results = search_similar_chunks(
        query_vector=matrix[0],
        top_k=2,
        exclude_chunk_id="doc_a_chunk_000",
        table_name="test_chunks",
        db_path=tmp_path,
    )

    assert len(results) == 2
    assert "chunk_id" in results.columns
    assert results.iloc[0]["chunk_id"] == "doc_a_chunk_001"


def test_fetch_table_preview_returns_rows(tmp_path) -> None:
    chunks = sample_chunks()
    matrix = normalized_matrix()

    write_chunk_embeddings(
        chunks=chunks,
        embedding_matrix=matrix,
        table_name="preview_chunks",
        db_path=tmp_path,
    )

    preview = fetch_table_preview(
        limit=2,
        table_name="preview_chunks",
        db_path=tmp_path,
    )

    assert len(preview) == 2
    assert "chunk_id" in preview.columns
    assert "preview" in preview.columns
