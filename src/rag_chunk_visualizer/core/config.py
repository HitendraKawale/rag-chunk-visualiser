from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "RAG Chunk Visualizer")
    app_env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu").lower()
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")

    projection_method: str = os.getenv("PROJECTION_METHOD", "pca").lower()
    projection_random_state: int = int(os.getenv("PROJECTION_RANDOM_STATE", "42"))
    umap_n_neighbors: int = int(os.getenv("UMAP_N_NEIGHBORS", "15"))
    umap_min_dist: float = float(os.getenv("UMAP_MIN_DIST", "0.1"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "75"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "5"))

    lancedb_table_name: str = os.getenv("LANCEDB_TABLE_NAME", "rag_chunks")

    data_dir: Path = ROOT_DIR / "data"
    documents_dir: Path = ROOT_DIR / "data" / "documents"
    lancedb_dir: Path = ROOT_DIR / "data" / "lancedb"
    exports_dir: Path = ROOT_DIR / "data" / "exports"
    docs_dir: Path = ROOT_DIR / "docs"

    def validate(self) -> None:
        if self.projection_method not in {"pca", "umap"}:
            raise ValueError(
                f"Invalid PROJECTION_METHOD='{self.projection_method}'. Expected 'pca' or 'umap'."
            )

        if self.embedding_device not in {"cpu", "mps"}:
            raise ValueError(
                f"Invalid EMBEDDING_DEVICE='{self.embedding_device}'. Expected 'cpu' or 'mps'."
            )

        if self.embedding_batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be greater than 0.")

        if self.ollama_temperature < 0:
            raise ValueError("OLLAMA_TEMPERATURE cannot be negative.")

        if self.ollama_num_predict <= 0:
            raise ValueError("OLLAMA_NUM_PREDICT must be greater than 0.")

        if self.ollama_timeout_seconds <= 0:
            raise ValueError("OLLAMA_TIMEOUT_SECONDS must be greater than 0.")

        if not self.ollama_keep_alive.strip():
            raise ValueError("OLLAMA_KEEP_ALIVE cannot be empty.")

        if self.umap_n_neighbors <= 1:
            raise ValueError("UMAP_N_NEIGHBORS must be greater than 1.")

        if not (0.0 <= self.umap_min_dist <= 0.99):
            raise ValueError("UMAP_MIN_DIST must be between 0.0 and 0.99.")

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0.")

        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.top_k <= 0:
            raise ValueError("TOP_K must be greater than 0.")

        if self.max_upload_mb <= 0:
            raise ValueError("MAX_UPLOAD_MB must be greater than 0.")

        if not self.lancedb_table_name.strip():
            raise ValueError("LANCEDB_TABLE_NAME cannot be empty.")


settings = Settings()
settings.validate()


def ensure_directories() -> None:
    for path in (
        settings.data_dir,
        settings.documents_dir,
        settings.lancedb_dir,
        settings.exports_dir,
        settings.docs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
