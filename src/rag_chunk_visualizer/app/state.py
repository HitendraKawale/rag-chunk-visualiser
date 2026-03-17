from __future__ import annotations

import streamlit as st

DEFAULT_SESSION_STATE = {
    "uploaded_documents": [],
    "raw_documents": [],
    "document_errors": [],
    "chunk_errors": [],
    "embedding_errors": [],
    "vector_store_errors": [],
    "projection_errors": [],
    "retrieval_errors": [],
    "generation_errors": [],
    "selected_document_id": None,
    "chunks": [],
    "selected_chunk_id": None,
    "embedding_matrix": None,
    "embedding_summary": None,
    "query_embedding": None,
    "vector_store_summary": None,
    "vector_search_results": [],
    "projection_df": None,
    "projection_summary": None,
    "projection_model": None,
    "query_projection": None,
    "query_text": "",
    "grounded_prompt": None,
    "retrieval_results": [],
    "generated_answer": None,
    "app_status": "Ready",
}


def initialize_session_state() -> None:
    for key, default_value in DEFAULT_SESSION_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
