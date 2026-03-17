from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    char_start: int
    char_end: int
    char_count: int
    word_count: int
    preview: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)
