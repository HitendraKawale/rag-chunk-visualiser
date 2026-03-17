from rag_chunk_visualizer.services.document_service import (
    DocumentValidationError,
    decode_text_bytes,
    get_extension,
    normalize_text,
    sanitize_filename,
    validate_uploaded_file,
)


def test_sanitize_filename() -> None:
    assert sanitize_filename("my notes!!.md") == "my_notes_.md"


def test_get_extension() -> None:
    assert get_extension("notes.md") == "md"
    assert get_extension("paper.txt") == "txt"


def test_validate_uploaded_file_rejects_bad_extension() -> None:
    try:
        validate_uploaded_file("report.pdf", 100)
        raise AssertionError("Expected DocumentValidationError")
    except DocumentValidationError:
        assert True


def test_decode_text_bytes_utf8() -> None:
    text = decode_text_bytes("hello world".encode("utf-8"), "hello.txt")
    assert text == "hello world"


def test_normalize_text() -> None:
    raw = "line1\r\n\r\n\r\nline2\rline3"
    normalized = normalize_text(raw)
    assert normalized == "line1\n\nline2\nline3"
