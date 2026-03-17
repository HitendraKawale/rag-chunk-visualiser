from rag_chunk_visualizer.core.config import settings


def test_projection_method_is_valid() -> None:
    assert settings.projection_method in {"pca", "umap"}


def test_projection_random_state_is_int() -> None:
    assert isinstance(settings.projection_random_state, int)


def test_umap_n_neighbors_is_valid() -> None:
    assert settings.umap_n_neighbors > 1


def test_umap_min_dist_is_valid() -> None:
    assert 0.0 <= settings.umap_min_dist <= 0.99


def test_embedding_device_is_valid() -> None:
    assert settings.embedding_device in {"cpu", "mps"}


def test_embedding_batch_size_is_positive() -> None:
    assert settings.embedding_batch_size > 0


def test_ollama_temperature_is_non_negative() -> None:
    assert settings.ollama_temperature >= 0


def test_ollama_num_predict_is_positive() -> None:
    assert settings.ollama_num_predict > 0


def test_ollama_timeout_seconds_is_positive() -> None:
    assert settings.ollama_timeout_seconds > 0


def test_ollama_keep_alive_is_not_empty() -> None:
    assert settings.ollama_keep_alive.strip()


def test_chunk_values_are_valid() -> None:
    assert settings.chunk_size > 0
    assert settings.chunk_overlap >= 0
    assert settings.chunk_overlap < settings.chunk_size


def test_top_k_is_positive() -> None:
    assert settings.top_k > 0


def test_max_upload_mb_is_positive() -> None:
    assert settings.max_upload_mb > 0


def test_lancedb_table_name_is_not_empty() -> None:
    assert settings.lancedb_table_name.strip()
