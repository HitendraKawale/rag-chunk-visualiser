from rag_chunk_visualizer.services.chunking_service import (
    ChunkingParameterError,
    build_chunks_from_documents,
    chunk_document,
    make_preview,
    validate_chunk_params,
)


def test_validate_chunk_params_rejects_bad_overlap() -> None:
    try:
        validate_chunk_params(chunk_size=100, chunk_overlap=100)
        raise AssertionError("Expected ChunkingParameterError")
    except ChunkingParameterError:
        assert True


def test_make_preview_truncates_long_text() -> None:
    text = "word " * 60
    preview = make_preview(text, max_chars=40)
    assert len(preview) <= 40
    assert preview.endswith("...")


def test_chunk_document_creates_multiple_chunks() -> None:
    raw_document = {
        "doc_id": "doc123",
        "filename": "notes.txt",
        "text": "This is a test sentence. " * 50,
    }

    chunks = chunk_document(raw_document, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "doc123_chunk_000"
    assert chunks[0]["doc_id"] == "doc123"
    assert chunks[0]["filename"] == "notes.txt"
    assert chunks[0]["char_count"] > 0
    assert chunks[0]["word_count"] > 0
    assert chunks[1]["char_start"] < chunks[0]["char_end"]


def test_build_chunks_from_multiple_documents() -> None:
    raw_documents = [
        {
            "doc_id": "doc_a",
            "filename": "a.txt",
            "text": "Alpha beta gamma delta. " * 30,
        },
        {
            "doc_id": "doc_b",
            "filename": "b.txt",
            "text": "One two three four five. " * 30,
        },
    ]

    chunks = build_chunks_from_documents(
        raw_documents=raw_documents,
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) >= 2
    assert any(chunk["doc_id"] == "doc_a" for chunk in chunks)
    assert any(chunk["doc_id"] == "doc_b" for chunk in chunks)
