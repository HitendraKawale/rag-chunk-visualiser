import pytest

from rag_chunk_visualizer.services.rag_service import (
    RAGGenerationError,
    build_context_block,
    build_grounded_prompt,
    format_source_label,
    validate_query_text,
    validate_retrieval_results,
)


def sample_retrieval_results() -> list[dict]:
    return [
        {
            "rank": 1,
            "chunk_id": "doc_a_chunk_000",
            "doc_id": "doc_a",
            "filename": "a.txt",
            "chunk_index": 0,
            "similarity_score": 0.912345,
            "preview": "alpha beta gamma",
            "text": "alpha beta gamma delta",
        },
        {
            "rank": 2,
            "chunk_id": "doc_b_chunk_000",
            "doc_id": "doc_b",
            "filename": "b.txt",
            "chunk_index": 1,
            "similarity_score": 0.812345,
            "preview": "theta lambda omega",
            "text": "theta lambda omega sigma",
        },
    ]


def test_validate_query_text_rejects_blank_query() -> None:
    with pytest.raises(RAGGenerationError):
        validate_query_text("   ")


def test_validate_retrieval_results_rejects_empty_list() -> None:
    with pytest.raises(RAGGenerationError):
        validate_retrieval_results([])


def test_format_source_label_contains_rank_filename_and_chunk() -> None:
    label = format_source_label(sample_retrieval_results()[0])

    assert "[Source 1]" in label
    assert "a.txt" in label
    assert "chunk 0" in label


def test_build_context_block_includes_all_sources() -> None:
    context_block = build_context_block(sample_retrieval_results())

    assert "[Source 1]" in context_block
    assert "[Source 2]" in context_block
    assert "alpha beta gamma delta" in context_block
    assert "theta lambda omega sigma" in context_block


def test_build_grounded_prompt_includes_question_context_and_instructions() -> None:
    prompt = build_grounded_prompt(
        query_text="What does the text say?",
        retrieval_results=sample_retrieval_results(),
    )

    assert "User question:" in prompt
    assert "Retrieved context:" in prompt
    assert "Instructions:" in prompt
    assert "[Source 1]" in prompt
    assert "What does the text say?" in prompt
