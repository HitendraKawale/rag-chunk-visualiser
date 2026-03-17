from __future__ import annotations

import re

from rag_chunk_visualizer.core.logging import get_logger
from rag_chunk_visualizer.models.chunk import ChunkRecord

logger = get_logger(__name__)


class ChunkingParameterError(ValueError):
    pass


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ChunkingParameterError("Chunk size must be greater than 0.")

    if chunk_overlap < 0:
        raise ChunkingParameterError("Chunk overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ChunkingParameterError("Chunk overlap must be smaller than chunk size.")


def make_preview(text: str, max_chars: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def choose_chunk_end(text: str, target_end: int, lookahead: int = 80) -> int:
    if target_end >= len(text):
        return len(text)

    boundary_window = text[target_end : min(len(text), target_end + lookahead)]
    match = re.search(r"[\n.!?;:, ]", boundary_window)

    if match:
        return target_end + match.start() + 1

    return target_end


def chunk_document(raw_document: dict, chunk_size: int, chunk_overlap: int) -> list[dict]:
    validate_chunk_params(chunk_size, chunk_overlap)

    text = raw_document["text"]
    if not text.strip():
        return []

    doc_id = raw_document["doc_id"]
    filename = raw_document["filename"]

    step = chunk_size - chunk_overlap
    start = 0
    chunk_index = 0
    chunks: list[dict] = []

    while start < len(text):
        target_end = min(len(text), start + chunk_size)
        end = choose_chunk_end(text, target_end)

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk = ChunkRecord(
                chunk_id=f"{doc_id}_chunk_{chunk_index:03d}",
                doc_id=doc_id,
                filename=filename,
                chunk_index=chunk_index,
                char_start=start,
                char_end=end,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                preview=make_preview(chunk_text),
                text=chunk_text,
            )
            chunks.append(chunk.to_dict())

        if end >= len(text):
            break

        start += step
        chunk_index += 1

    logger.info(
        "Chunked document | doc_id=%s | filename=%s | chunks=%s | chunk_size=%s | overlap=%s",
        doc_id,
        filename,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )

    return chunks


def build_chunks_from_documents(
    raw_documents: list[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    validate_chunk_params(chunk_size, chunk_overlap)

    all_chunks: list[dict] = []

    for raw_document in raw_documents:
        document_chunks = chunk_document(
            raw_document=raw_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(document_chunks)

    logger.info(
        "Built chunks across documents | documents=%s | total_chunks=%s",
        len(raw_documents),
        len(all_chunks),
    )

    return all_chunks
