from __future__ import annotations

import importlib
import shutil
from pathlib import Path

from rag_chunk_visualizer.core.config import ensure_directories, settings
from rag_chunk_visualizer.core.logging import configure_logging, get_logger

MODULES_TO_CHECK = [
    "streamlit",
    "plotly",
    "pandas",
    "numpy",
    "sentence_transformers",
    "lancedb",
    "sklearn",
    "umap",
    "ollama",
]


def main() -> None:
    configure_logging()
    logger = get_logger("smoke_check")

    logger.info("Starting smoke check")
    ensure_directories()

    for module_name in MODULES_TO_CHECK:
        importlib.import_module(module_name)
        logger.info("Imported module successfully: %s", module_name)

    project_root = Path(__file__).resolve().parents[1]

    print("\n=== SMOKE CHECK SUMMARY ===")
    print(f"Project root      : {project_root}")
    print(f"App name          : {settings.app_name}")
    print(f"Embedding model   : {settings.embedding_model}")
    print(f"Embedding device  : {settings.embedding_device}")
    print(f"Embedding batch   : {settings.embedding_batch_size}")
    print(f"Projection method : {settings.projection_method}")
    print(f"Projection seed   : {settings.projection_random_state}")
    print(f"UMAP neighbors    : {settings.umap_n_neighbors}")
    print(f"UMAP min_dist     : {settings.umap_min_dist}")
    print(f"LanceDB table     : {settings.lancedb_table_name}")
    print(f"Ollama base URL   : {settings.ollama_base_url}")
    print(f"Ollama model      : {settings.ollama_model}")
    print(f"Ollama temp       : {settings.ollama_temperature}")
    print(f"Ollama predict    : {settings.ollama_num_predict}")
    print(f"Ollama timeout    : {settings.ollama_timeout_seconds}")
    print(f"Ollama keep_alive : {settings.ollama_keep_alive}")
    print(f"Chunk size        : {settings.chunk_size}")
    print(f"Chunk overlap     : {settings.chunk_overlap}")
    print(f"Top-k             : {settings.top_k}")
    print(f"Documents dir     : {settings.documents_dir}")
    print(f"LanceDB dir       : {settings.lancedb_dir}")
    print(f"Exports dir       : {settings.exports_dir}")

    ollama_cli_path = shutil.which("ollama")
    if ollama_cli_path:
        print(f"Ollama CLI        : FOUND at {ollama_cli_path}")
    else:
        print("Ollama CLI        : NOT FOUND on PATH")

    print("\nSmoke check passed.")


if __name__ == "__main__":
    main()
