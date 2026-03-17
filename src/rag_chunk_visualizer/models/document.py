from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RawDocument:
    doc_id: str
    filename: str
    extension: str
    source_path: str
    char_count: int
    word_count: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)
