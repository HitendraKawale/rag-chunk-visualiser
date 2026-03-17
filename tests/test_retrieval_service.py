import numpy as np
import pandas as pd
import pytest

from rag_chunk_visualizer.services.retrieval_service import (
    RetrievalError,
    enrich_query_results,
    get_retrieved_chunk_ids,
    validate_query_text,
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


def sample_embedding_matrix() -> np.ndarray:
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return matrix


def test_validate_query_text_rejects_blank_query() -> None:
    with pytest.raises(RetrievalError):
        validate_query_text("   ")


def test_enrich_query_results_adds_rank_and_similarity_score() -> None:
    results_df = pd.DataFrame(
        [
            {
                "chunk_id": "doc_a_chunk_000",
                "doc_id": "doc_a",
                "filename": "a.txt",
                "chunk_index": 0,
                "preview": "alpha beta gamma",
                "text": "alpha beta gamma delta",
            },
            {
                "chunk_id": "doc_b_chunk_000",
                "doc_id": "doc_b",
                "filename": "b.txt",
                "chunk_index": 0,
                "preview": "theta lambda omega",
                "text": "theta lambda omega sigma",
            },
        ]
    )

    query_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    enriched = enrich_query_results(
        results_df=results_df,
        chunks=sample_chunks(),
        embedding_matrix=sample_embedding_matrix(),
        query_vector=query_vector,
    )

    assert len(enriched) == 2
    assert enriched[0]["rank"] == 1
    assert "similarity_score" in enriched[0]
    assert enriched[0]["similarity_score"] == 1.0
    assert enriched[1]["similarity_score"] == 0.0


def test_get_retrieved_chunk_ids_returns_only_chunk_ids() -> None:
    retrieval_results = [
        {"chunk_id": "doc_a_chunk_000", "rank": 1},
        {"chunk_id": "doc_b_chunk_000", "rank": 2},
    ]

    assert get_retrieved_chunk_ids(retrieval_results) == [
        "doc_a_chunk_000",
        "doc_b_chunk_000",
    ]
