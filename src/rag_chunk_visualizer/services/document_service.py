from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.core.logging import get_logger
from rag_chunk_visualizer.models.document import RawDocument

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {"txt", "md"}


class DocumentValidationError(ValueError):
    pass


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return sanitized or "document.txt"


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def validate_uploaded_file(filename: str, size_bytes: int) -> None:
    extension = get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise DocumentValidationError(
            f"Unsupported file type for '{filename}'. Allowed types: .txt, .md"
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise DocumentValidationError(
            f"File '{filename}' is too large ({size_bytes} bytes). "
            f"Maximum allowed size is {settings.max_upload_mb} MB."
        )


def decode_text_bytes(raw_bytes: bytes, filename: str) -> str:
    encodings_to_try = ("utf-8-sig", "utf-8")
    for encoding in encodings_to_try:
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise DocumentValidationError(
        f"Could not decode '{filename}' as UTF-8 text. "
        "Please upload a UTF-8 encoded .txt or .md file."
    )


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_uploaded_file(raw_bytes: bytes, doc_id: str, filename: str) -> Path:
    safe_name = sanitize_filename(filename)
    output_path = settings.documents_dir / f"{doc_id}_{safe_name}"
    output_path.write_bytes(raw_bytes)
    return output_path


def build_raw_document(filename: str, raw_bytes: bytes) -> RawDocument:
    validate_uploaded_file(filename, len(raw_bytes))

    decoded_text = decode_text_bytes(raw_bytes, filename)
    normalized_text = normalize_text(decoded_text)

    if not normalized_text:
        raise DocumentValidationError(f"File '{filename}' is empty after text extraction.")

    doc_id = uuid4().hex[:8]
    saved_path = save_uploaded_file(raw_bytes, doc_id, filename)

    document = RawDocument(
        doc_id=doc_id,
        filename=filename,
        extension=get_extension(filename),
        source_path=str(saved_path),
        char_count=len(normalized_text),
        word_count=len(normalized_text.split()),
        text=normalized_text,
    )

    logger.info(
        "Built raw document | doc_id=%s | filename=%s | chars=%s | words=%s",
        document.doc_id,
        document.filename,
        document.char_count,
        document.word_count,
    )

    return document


def process_uploaded_files(uploaded_files: list) -> tuple[list[dict], list[str]]:
    processed_documents: list[dict] = []
    errors: list[str] = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        raw_bytes = uploaded_file.getvalue()

        try:
            document = build_raw_document(filename=filename, raw_bytes=raw_bytes)
            processed_documents.append(document.to_dict())
        except DocumentValidationError as exc:
            logger.warning("Document validation failed | filename=%s | error=%s", filename, exc)
            errors.append(str(exc))

    return processed_documents, errors
