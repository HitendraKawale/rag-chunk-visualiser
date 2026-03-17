from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.services.chunking_service import (
    ChunkingParameterError,
    build_chunks_from_documents,
)
from rag_chunk_visualizer.services.document_service import process_uploaded_files
from rag_chunk_visualizer.services.embedding_service import (
    EmbeddingError,
    embed_chunks,
    embed_query_text,
    summarize_embedding_matrix,
)
from rag_chunk_visualizer.services.rag_service import (
    RAGGenerationError,
    build_grounded_prompt,
    generate_grounded_answer,
)
from rag_chunk_visualizer.services.retrieval_service import (
    RetrievalError,
    enrich_query_results,
    get_retrieved_chunk_ids,
    validate_query_text,
)
from rag_chunk_visualizer.storage.lancedb_store import (
    VectorStoreError,
    fetch_table_preview,
    search_similar_chunks,
    write_chunk_embeddings,
)
from rag_chunk_visualizer.visualization.plotting import (
    PlotBuildError,
    create_embedding_scatter_plot,
)
from rag_chunk_visualizer.visualization.projection import (
    ProjectionError,
    project_chunks,
    transform_query_vector,
)


def toast_success(message: str) -> None:
    st.toast(message, icon="✅")


def reset_downstream_state() -> None:
    st.session_state["embedding_errors"] = []
    st.session_state["vector_store_errors"] = []
    st.session_state["projection_errors"] = []
    st.session_state["retrieval_errors"] = []
    st.session_state["generation_errors"] = []
    st.session_state["embedding_matrix"] = None
    st.session_state["embedding_summary"] = None
    st.session_state["query_embedding"] = None
    st.session_state["vector_store_summary"] = None
    st.session_state["vector_search_results"] = []
    st.session_state["projection_df"] = None
    st.session_state["projection_summary"] = None
    st.session_state["projection_model"] = None
    st.session_state["query_projection"] = None
    st.session_state["grounded_prompt"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["generated_answer"] = None


def handle_document_uploads(uploaded_files: list) -> None:
    if not uploaded_files:
        st.session_state["document_errors"] = ["No files selected."]
        st.session_state["chunk_errors"] = []
        reset_downstream_state()
        st.session_state["app_status"] = "No files uploaded"
        return

    processed_documents, errors = process_uploaded_files(uploaded_files)

    st.session_state["raw_documents"] = processed_documents
    st.session_state["document_errors"] = errors
    st.session_state["chunk_errors"] = []
    st.session_state["selected_document_id"] = (
        processed_documents[0]["doc_id"] if processed_documents else None
    )
    st.session_state["chunks"] = []
    st.session_state["selected_chunk_id"] = None
    reset_downstream_state()

    if processed_documents:
        st.session_state["app_status"] = f"Loaded {len(processed_documents)} document(s)"
        toast_success(f"Loaded {len(processed_documents)} document(s)")
    else:
        st.session_state["app_status"] = "Upload failed"


def handle_chunk_build(chunk_size: int, chunk_overlap: int) -> None:
    raw_documents = st.session_state["raw_documents"]

    if not raw_documents:
        st.session_state["chunk_errors"] = ["Upload at least one document before chunking."]
        st.session_state["app_status"] = "Chunking failed"
        return

    try:
        chunks = build_chunks_from_documents(
            raw_documents=raw_documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ChunkingParameterError as exc:
        st.session_state["chunk_errors"] = [str(exc)]
        st.session_state["chunks"] = []
        st.session_state["selected_chunk_id"] = None
        reset_downstream_state()
        st.session_state["app_status"] = "Chunking failed"
        return

    st.session_state["chunk_errors"] = []
    st.session_state["chunks"] = chunks
    st.session_state["selected_chunk_id"] = chunks[0]["chunk_id"] if chunks else None
    reset_downstream_state()

    if chunks:
        st.session_state["app_status"] = f"Built {len(chunks)} chunk(s)"
        toast_success(f"Built {len(chunks)} chunk(s)")
    else:
        st.session_state["app_status"] = "No chunks produced"


def handle_embedding_build() -> None:
    chunks = st.session_state["chunks"]

    if not chunks:
        st.session_state["embedding_errors"] = ["Build chunks before generating embeddings."]
        st.session_state["app_status"] = "Embedding failed"
        return

    try:
        matrix = embed_chunks(chunks)
        summary = summarize_embedding_matrix(matrix)
    except EmbeddingError as exc:
        st.session_state["embedding_errors"] = [str(exc)]
        st.session_state["embedding_matrix"] = None
        st.session_state["embedding_summary"] = None
        st.session_state["query_embedding"] = None
        st.session_state["vector_store_summary"] = None
        st.session_state["vector_store_errors"] = []
        st.session_state["vector_search_results"] = []
        st.session_state["projection_df"] = None
        st.session_state["projection_summary"] = None
        st.session_state["projection_model"] = None
        st.session_state["query_projection"] = None
        st.session_state["projection_errors"] = []
        st.session_state["retrieval_errors"] = []
        st.session_state["generation_errors"] = []
        st.session_state["grounded_prompt"] = None
        st.session_state["retrieval_results"] = []
        st.session_state["generated_answer"] = None
        st.session_state["app_status"] = "Embedding failed"
        return

    st.session_state["embedding_errors"] = []
    st.session_state["vector_store_errors"] = []
    st.session_state["projection_errors"] = []
    st.session_state["retrieval_errors"] = []
    st.session_state["generation_errors"] = []
    st.session_state["embedding_matrix"] = matrix
    st.session_state["embedding_summary"] = summary
    st.session_state["query_embedding"] = None
    st.session_state["vector_store_summary"] = None
    st.session_state["vector_search_results"] = []
    st.session_state["projection_df"] = None
    st.session_state["projection_summary"] = None
    st.session_state["projection_model"] = None
    st.session_state["query_projection"] = None
    st.session_state["grounded_prompt"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["generated_answer"] = None
    st.session_state["app_status"] = f"Embedded {matrix.shape[0]} chunk(s)"
    toast_success(f"Embedded {matrix.shape[0]} chunk(s)")


def handle_projection_build(method: str) -> None:
    chunks = st.session_state["chunks"]
    embedding_matrix = st.session_state["embedding_matrix"]

    if not chunks or embedding_matrix is None:
        st.session_state["projection_errors"] = [
            "Generate embeddings before building a 2D projection."
        ]
        st.session_state["app_status"] = "Projection failed"
        return

    try:
        projection_df, projection_summary, projection_model = project_chunks(
            chunks=chunks,
            embedding_matrix=embedding_matrix,
            method=method,
        )
    except ProjectionError as exc:
        st.session_state["projection_errors"] = [str(exc)]
        st.session_state["projection_df"] = None
        st.session_state["projection_summary"] = None
        st.session_state["projection_model"] = None
        st.session_state["query_projection"] = None
        st.session_state["retrieval_results"] = []
        st.session_state["generated_answer"] = None
        st.session_state["grounded_prompt"] = None
        st.session_state["app_status"] = "Projection failed"
        return

    st.session_state["projection_errors"] = []
    st.session_state["projection_df"] = projection_df
    st.session_state["projection_summary"] = projection_summary
    st.session_state["projection_model"] = projection_model
    st.session_state["query_projection"] = None
    st.session_state["grounded_prompt"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["generated_answer"] = None
    st.session_state["app_status"] = f"Projected {len(projection_df)} chunk(s) to 2D"
    toast_success(f"Projected {len(projection_df)} chunk(s) to 2D")


def handle_store_in_lancedb() -> None:
    chunks = st.session_state["chunks"]
    embedding_matrix = st.session_state["embedding_matrix"]

    if not chunks or embedding_matrix is None:
        st.session_state["vector_store_errors"] = [
            "Generate embeddings before storing vectors in LanceDB."
        ]
        st.session_state["app_status"] = "Vector DB sync failed"
        return

    try:
        summary = write_chunk_embeddings(
            chunks=chunks,
            embedding_matrix=embedding_matrix,
        )
    except VectorStoreError as exc:
        st.session_state["vector_store_errors"] = [str(exc)]
        st.session_state["vector_store_summary"] = None
        st.session_state["vector_search_results"] = []
        st.session_state["app_status"] = "Vector DB sync failed"
        return

    st.session_state["vector_store_errors"] = []
    st.session_state["vector_store_summary"] = summary
    st.session_state["vector_search_results"] = []
    st.session_state["app_status"] = f"Stored {summary['row_count']} row(s) in LanceDB"
    toast_success(f"Stored {summary['row_count']} row(s) in LanceDB")


def get_selected_document() -> dict | None:
    raw_documents = st.session_state["raw_documents"]
    selected_document_id = st.session_state["selected_document_id"]

    for document in raw_documents:
        if document["doc_id"] == selected_document_id:
            return document

    return raw_documents[0] if raw_documents else None


def get_selected_chunk() -> dict | None:
    chunks = st.session_state["chunks"]
    selected_chunk_id = st.session_state["selected_chunk_id"]

    for chunk in chunks:
        if chunk["chunk_id"] == selected_chunk_id:
            return chunk

    return chunks[0] if chunks else None


def get_selected_chunk_vector():
    selected_chunk = get_selected_chunk()
    embedding_matrix = st.session_state["embedding_matrix"]
    chunks = st.session_state["chunks"]

    if selected_chunk is None or embedding_matrix is None:
        return None, None

    for index, chunk in enumerate(chunks):
        if chunk["chunk_id"] == selected_chunk["chunk_id"]:
            return selected_chunk, embedding_matrix[index]

    return None, None


def handle_lancedb_neighbor_search(top_k: int) -> None:
    vector_store_summary = st.session_state["vector_store_summary"]

    if vector_store_summary is None:
        st.session_state["vector_store_errors"] = [
            "Store vectors in LanceDB before running nearest-neighbor preview."
        ]
        st.session_state["app_status"] = "Vector DB search failed"
        return

    selected_chunk, query_vector = get_selected_chunk_vector()
    if selected_chunk is None or query_vector is None:
        st.session_state["vector_store_errors"] = [
            "Select a chunk and generate embeddings before searching LanceDB."
        ]
        st.session_state["app_status"] = "Vector DB search failed"
        return

    try:
        results_df = search_similar_chunks(
            query_vector=query_vector,
            top_k=top_k,
            exclude_chunk_id=selected_chunk["chunk_id"],
        )
    except VectorStoreError as exc:
        st.session_state["vector_store_errors"] = [str(exc)]
        st.session_state["vector_search_results"] = []
        st.session_state["app_status"] = "Vector DB search failed"
        return

    st.session_state["vector_store_errors"] = []
    st.session_state["vector_search_results"] = results_df.to_dict(orient="records")
    st.session_state["app_status"] = f"Retrieved {len(results_df)} neighbor(s) from LanceDB"
    toast_success(f"Retrieved {len(results_df)} neighbor(s) from LanceDB")


def handle_query_retrieval(query_text: str, top_k: int) -> None:
    try:
        validate_query_text(query_text)
    except RetrievalError as exc:
        st.session_state["retrieval_errors"] = [str(exc)]
        st.session_state["app_status"] = "Retrieval failed"
        return

    if st.session_state["vector_store_summary"] is None:
        st.session_state["retrieval_errors"] = [
            "Store vectors in LanceDB before running query retrieval."
        ]
        st.session_state["app_status"] = "Retrieval failed"
        return

    if st.session_state["projection_summary"] is None:
        st.session_state["retrieval_errors"] = [
            "Project embeddings to 2D before running query retrieval."
        ]
        st.session_state["app_status"] = "Retrieval failed"
        return

    try:
        query_vector = embed_query_text(query_text)
        results_df = search_similar_chunks(
            query_vector=query_vector,
            top_k=top_k,
            exclude_chunk_id=None,
        )
        retrieval_results = enrich_query_results(
            results_df=results_df,
            chunks=st.session_state["chunks"],
            embedding_matrix=st.session_state["embedding_matrix"],
            query_vector=query_vector,
        )
        query_projection = transform_query_vector(
            query_vector=query_vector,
            projection_model=st.session_state["projection_model"],
            projection_summary=st.session_state["projection_summary"],
        )
        grounded_prompt = build_grounded_prompt(query_text, retrieval_results)
    except (
        EmbeddingError,
        VectorStoreError,
        RetrievalError,
        ProjectionError,
        RAGGenerationError,
    ) as exc:
        st.session_state["retrieval_errors"] = [str(exc)]
        st.session_state["query_embedding"] = None
        st.session_state["query_projection"] = None
        st.session_state["grounded_prompt"] = None
        st.session_state["retrieval_results"] = []
        st.session_state["generated_answer"] = None
        st.session_state["app_status"] = "Retrieval failed"
        return

    st.session_state["retrieval_errors"] = []
    st.session_state["generation_errors"] = []
    st.session_state["query_embedding"] = query_vector
    st.session_state["query_projection"] = {
        **query_projection,
        "query_text": query_text.strip(),
    }
    st.session_state["grounded_prompt"] = grounded_prompt
    st.session_state["retrieval_results"] = retrieval_results
    st.session_state["generated_answer"] = None

    if retrieval_results:
        top_result = retrieval_results[0]
        st.session_state["selected_chunk_id"] = top_result["chunk_id"]
        st.session_state["selected_document_id"] = top_result["doc_id"]
        st.session_state["app_status"] = f"Retrieved {len(retrieval_results)} chunk(s) for query"
        toast_success(f"Retrieved {len(retrieval_results)} chunk(s) for query")
    else:
        st.session_state["app_status"] = "No retrieval results found"


def handle_answer_generation(query_text: str) -> None:
    retrieval_results = st.session_state["retrieval_results"]

    if not retrieval_results:
        st.session_state["generation_errors"] = [
            "Run retrieval before generating a grounded answer."
        ]
        st.session_state["app_status"] = "Answer generation failed"
        return

    try:
        generated_answer = generate_grounded_answer(
            query_text=query_text,
            retrieval_results=retrieval_results,
        )
    except RAGGenerationError as exc:
        st.session_state["generation_errors"] = [str(exc)]
        st.session_state["generated_answer"] = None
        st.session_state["app_status"] = "Answer generation failed"
        return

    st.session_state["generation_errors"] = []
    st.session_state["grounded_prompt"] = generated_answer["prompt"]
    st.session_state["generated_answer"] = generated_answer
    st.session_state["app_status"] = f"Generated grounded answer with {generated_answer['model']}"
    toast_success(f"Generated grounded answer with {generated_answer['model']}")


def handle_projection_selection(event) -> None:
    if event is None:
        return

    if hasattr(event, "selection"):
        selection = event.selection
    elif isinstance(event, dict):
        selection = event.get("selection")
    else:
        selection = None

    if not selection:
        return

    points = selection.get("points", [])
    if not points:
        return

    point = points[0]
    custom_data = point.get("customdata", [])

    if len(custom_data) < 2:
        return

    chunk_id = custom_data[0]
    doc_id = custom_data[1]

    if chunk_id == "__query__":
        return

    st.session_state["selected_chunk_id"] = chunk_id
    st.session_state["selected_document_id"] = doc_id


def clear_documents() -> None:
    st.session_state["uploaded_documents"] = []
    st.session_state["raw_documents"] = []
    st.session_state["document_errors"] = []
    st.session_state["chunk_errors"] = []
    st.session_state["embedding_errors"] = []
    st.session_state["vector_store_errors"] = []
    st.session_state["projection_errors"] = []
    st.session_state["retrieval_errors"] = []
    st.session_state["generation_errors"] = []
    st.session_state["selected_document_id"] = None
    st.session_state["chunks"] = []
    st.session_state["selected_chunk_id"] = None
    st.session_state["embedding_matrix"] = None
    st.session_state["embedding_summary"] = None
    st.session_state["query_embedding"] = None
    st.session_state["vector_store_summary"] = None
    st.session_state["vector_search_results"] = []
    st.session_state["projection_df"] = None
    st.session_state["projection_summary"] = None
    st.session_state["projection_model"] = None
    st.session_state["query_projection"] = None
    st.session_state["query_text"] = ""
    st.session_state["grounded_prompt"] = None
    st.session_state["retrieval_results"] = []
    st.session_state["generated_answer"] = None
    st.session_state["app_status"] = "Ready"
    toast_success("Cleared current session data")


def get_selected_chunk_embedding_preview(max_dims: int = 8) -> pd.DataFrame | None:
    selected_chunk = get_selected_chunk()
    embedding_matrix = st.session_state["embedding_matrix"]
    chunks = st.session_state["chunks"]

    if selected_chunk is None or embedding_matrix is None:
        return None

    chunk_position = None
    for index, chunk in enumerate(chunks):
        if chunk["chunk_id"] == selected_chunk["chunk_id"]:
            chunk_position = index
            break

    if chunk_position is None:
        return None

    vector = embedding_matrix[chunk_position]
    preview_rows = []

    for dimension in range(min(max_dims, len(vector))):
        preview_rows.append(
            {
                "dimension": dimension,
                "value": round(float(vector[dimension]), 6),
            }
        )

    return pd.DataFrame(preview_rows)


def render_sidebar_pipeline_health() -> None:
    checks = [
        ("Documents", bool(st.session_state["raw_documents"])),
        ("Chunks", bool(st.session_state["chunks"])),
        ("Embeddings", st.session_state["embedding_summary"] is not None),
        ("Projection", st.session_state["projection_summary"] is not None),
        ("Vector DB", st.session_state["vector_store_summary"] is not None),
        ("Retrieval", bool(st.session_state["retrieval_results"])),
        ("Answer", st.session_state["generated_answer"] is not None),
    ]

    with st.container(border=True):
        st.subheader("Pipeline Health")
        for label, ok in checks:
            icon = "✅" if ok else "⚪"
            st.write(f"{icon} {label}")


def render_sidebar() -> dict:
    with st.sidebar:
        st.title("RAG Chunk Visualizer")
        st.caption("Visual debugging and teaching tool for RAG systems")

        st.divider()

        with st.form("upload_form"):
            st.subheader("Document Upload")

            uploaded_files = st.file_uploader(
                "Upload text documents",
                type=["txt", "md"],
                accept_multiple_files=True,
                help="Stage 11 keeps the same UTF-8 .txt and .md support.",
            )

            process_uploads = st.form_submit_button(
                "Process Uploads",
                type="primary",
                width="stretch",
            )

        if process_uploads:
            handle_document_uploads(uploaded_files)

        if st.button("Clear Documents", width="stretch"):
            clear_documents()

        st.divider()

        st.subheader("Pipeline Controls")

        chunk_size = st.slider(
            "Chunk size",
            min_value=100,
            max_value=1200,
            value=settings.chunk_size,
            step=50,
            help="Target character window for chunking.",
        )

        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=300,
            value=settings.chunk_overlap,
            step=25,
            help="Character overlap between consecutive chunks.",
        )

        top_k = st.slider(
            "Top-k retrieval",
            min_value=1,
            max_value=10,
            value=settings.top_k,
            step=1,
            help="Number of chunks to retrieve for the query.",
        )

        projection_method = st.selectbox(
            "Projection method",
            options=["pca", "umap"],
            index=0 if settings.projection_method == "pca" else 1,
            help="2D projection method for embedding visualization.",
        )

        build_chunks = st.button(
            "Build Chunks",
            type="primary",
            width="stretch",
            disabled=not st.session_state["raw_documents"],
        )

        if build_chunks:
            handle_chunk_build(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        build_embeddings = st.button(
            "Generate Embeddings",
            type="primary",
            width="stretch",
            disabled=not st.session_state["chunks"],
        )

        if build_embeddings:
            with st.spinner("Loading model and generating chunk embeddings..."):
                handle_embedding_build()

        build_projection = st.button(
            "Project to 2D",
            type="primary",
            width="stretch",
            disabled=st.session_state["embedding_matrix"] is None,
        )

        if build_projection:
            with st.spinner(f"Projecting embeddings with {projection_method.upper()}..."):
                handle_projection_build(method=projection_method)

        store_vectors = st.button(
            "Store in LanceDB",
            type="primary",
            width="stretch",
            disabled=st.session_state["embedding_matrix"] is None,
        )

        if store_vectors:
            with st.spinner("Writing chunk embeddings to LanceDB..."):
                handle_store_in_lancedb()

        preview_neighbors = st.button(
            "Preview Similar Chunks",
            type="primary",
            width="stretch",
            disabled=st.session_state["vector_store_summary"] is None,
        )

        if preview_neighbors:
            with st.spinner("Searching nearest chunks in LanceDB..."):
                handle_lancedb_neighbor_search(top_k=top_k)

        st.divider()

        st.subheader("Query")
        query_text = st.text_input(
            "Enter a query",
            value=st.session_state.get("query_text", ""),
            placeholder="e.g. What does the document say about vector search?",
        )

        enable_generation = st.toggle(
            "Enable answer generation",
            value=False,
            help="Use your local Ollama model to answer from retrieved chunks.",
        )

        run_query = st.button(
            "Run Retrieval",
            type="primary",
            width="stretch",
            disabled=(
                st.session_state["vector_store_summary"] is None
                or st.session_state["projection_summary"] is None
                or not query_text.strip()
            ),
        )

        if run_query:
            with st.spinner("Embedding query and retrieving relevant chunks..."):
                handle_query_retrieval(query_text=query_text, top_k=top_k)

        generate_answer = st.button(
            "Generate Answer",
            type="primary",
            width="stretch",
            disabled=(
                not enable_generation
                or not st.session_state["retrieval_results"]
                or not query_text.strip()
            ),
        )

        if generate_answer:
            with st.spinner("Generating grounded answer with Ollama..."):
                handle_answer_generation(query_text=query_text)

        st.divider()

        render_sidebar_pipeline_health()

        st.divider()

        st.subheader("Model Settings")
        st.text_input(
            "Embedding model",
            value=settings.embedding_model,
            disabled=True,
        )
        st.text_input(
            "Embedding device",
            value=settings.embedding_device,
            disabled=True,
        )
        st.text_input(
            "LanceDB table",
            value=settings.lancedb_table_name,
            disabled=True,
        )
        st.text_input(
            "Ollama model",
            value=settings.ollama_model,
            disabled=True,
        )

        st.divider()

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "top_k": top_k,
        "projection_method": projection_method,
        "query_text": query_text,
        "enable_generation": enable_generation,
        "run_query": run_query,
    }


def render_header() -> None:
    st.title("RAG Chunk Visualizer")
    st.markdown(
        "Inspect how documents become chunks, how chunks become vectors, "
        "and how retrieval selects grounded context."
    )


def render_overview_banner() -> None:
    docs = len(st.session_state["raw_documents"])
    chunks = len(st.session_state["chunks"])
    retrieved = len(st.session_state["retrieval_results"])
    answer_ready = st.session_state["generated_answer"] is not None

    with st.container(border=True):
        st.markdown(
            f"**Overview** — Docs: `{docs}` | "
            f"Chunks: `{chunks}` | "
            f"Retrieved: `{retrieved}` | "
            f"Answer ready: `{answer_ready}`"
        )


def render_status_metrics(sidebar_values: dict) -> None:
    embedding_matrix = st.session_state["embedding_matrix"]
    vector_store_summary = st.session_state["vector_store_summary"]
    projection_df = st.session_state["projection_df"]
    projection_summary = st.session_state["projection_summary"]
    generated_answer = st.session_state["generated_answer"]

    embedded_count = 0 if embedding_matrix is None else int(embedding_matrix.shape[0])
    embedding_dim = "—" if embedding_matrix is None else int(embedding_matrix.shape[1])
    stored_count = 0 if vector_store_summary is None else vector_store_summary["row_count"]
    projected_count = 0 if projection_df is None else int(len(projection_df))
    retrieved_count = len(st.session_state["retrieval_results"])
    answer_ready = "Yes" if generated_answer is not None else "No"
    method_label = "—"
    if projection_summary is not None:
        method_label = str(projection_summary.get("method", "—")).upper()

    metric_cols = st.columns(10)

    metric_cols[0].metric("Documents", len(st.session_state["raw_documents"]))
    metric_cols[1].metric("Chunks", len(st.session_state["chunks"]))
    metric_cols[2].metric("Embedded", embedded_count)
    metric_cols[3].metric("Stored", stored_count)
    metric_cols[4].metric("Projected", projected_count)
    metric_cols[5].metric("Retrieved", retrieved_count)
    metric_cols[6].metric("Answer", answer_ready)
    metric_cols[7].metric("Dim", embedding_dim)
    metric_cols[8].metric("Method", method_label)
    metric_cols[9].metric("Status", st.session_state["app_status"])


def render_document_summary() -> None:
    st.subheader("Uploaded Documents")

    raw_documents = st.session_state["raw_documents"]
    document_errors = st.session_state["document_errors"]

    if document_errors:
        for error in document_errors:
            st.error(error)

    if not raw_documents:
        st.info("No documents uploaded yet.")
        return

    summary_df = pd.DataFrame(
        [
            {
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "extension": doc["extension"],
                "chars": doc["char_count"],
                "words": doc["word_count"],
            }
            for doc in raw_documents
        ]
    )

    st.dataframe(summary_df, width="stretch", hide_index=True)


def render_document_preview_panel() -> None:
    st.subheader("Document Preview")

    raw_documents = st.session_state["raw_documents"]

    if not raw_documents:
        with st.container(border=True):
            st.write("No document available for preview yet.")
        return

    doc_ids = [doc["doc_id"] for doc in raw_documents]
    if st.session_state["selected_document_id"] not in doc_ids:
        st.session_state["selected_document_id"] = doc_ids[0]

    st.selectbox(
        "Select document",
        options=doc_ids,
        key="selected_document_id",
        format_func=lambda doc_id: next(
            doc["filename"] for doc in raw_documents if doc["doc_id"] == doc_id
        ),
    )

    selected_document = get_selected_document()
    if selected_document is None:
        return

    with st.container(border=True):
        st.markdown(f"**Document ID:** `{selected_document['doc_id']}`")
        st.markdown(f"**Filename:** {selected_document['filename']}")
        st.markdown(f"**Characters:** {selected_document['char_count']}")
        st.markdown(f"**Words:** {selected_document['word_count']}")
        st.markdown(f"**Saved to:** `{selected_document['source_path']}`")
        st.markdown("**Extracted text preview:**")
        st.code(selected_document["text"][:3000], language="text")


def render_chunk_summary_panel() -> None:
    st.subheader("Chunk Summary")

    chunk_errors = st.session_state["chunk_errors"]
    chunks = st.session_state["chunks"]

    if chunk_errors:
        for error in chunk_errors:
            st.error(error)

    if not chunks:
        st.info("No chunks built yet. Upload documents, then click 'Build Chunks'.")
        return

    chunk_df = pd.DataFrame(
        [
            {
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "chars": chunk["char_count"],
                "words": chunk["word_count"],
                "preview": chunk["preview"],
            }
            for chunk in chunks
        ]
    )

    st.dataframe(chunk_df, width="stretch", hide_index=True)


def render_chunk_detail_panel() -> None:
    st.subheader("Selected Chunk Details")

    chunks = st.session_state["chunks"]
    selected_document = get_selected_document()

    if not chunks:
        with st.container(border=True):
            st.code(
                "Chunking has not been run yet.\n\n"
                "Build chunks, generate embeddings, and project them in this stage.",
                language="text",
            )
        return

    if selected_document is not None:
        visible_chunks = [c for c in chunks if c["doc_id"] == selected_document["doc_id"]]
    else:
        visible_chunks = chunks

    if not visible_chunks:
        with st.container(border=True):
            st.write("No chunks available for the selected document.")
        return

    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in visible_chunks}
    chunk_ids = list(chunk_lookup.keys())

    if st.session_state["selected_chunk_id"] not in chunk_ids:
        st.session_state["selected_chunk_id"] = chunk_ids[0]

    st.selectbox(
        "Select chunk",
        options=chunk_ids,
        key="selected_chunk_id",
        format_func=lambda chunk_id: (
            f"{chunk_lookup[chunk_id]['filename']} | chunk {chunk_lookup[chunk_id]['chunk_index']}"
        ),
    )

    selected_chunk = get_selected_chunk()
    if selected_chunk is None:
        return

    with st.container(border=True):
        st.markdown(f"**Chunk ID:** `{selected_chunk['chunk_id']}`")
        st.markdown(f"**Document:** {selected_chunk['filename']}")
        st.markdown(f"**Chunk Index:** {selected_chunk['chunk_index']}")
        st.markdown(
            f"**Character Range:** {selected_chunk['char_start']} → {selected_chunk['char_end']}"
        )
        st.markdown(f"**Characters:** {selected_chunk['char_count']}")
        st.markdown(f"**Words:** {selected_chunk['word_count']}")
        st.markdown("**Chunk Text:**")
        st.code(selected_chunk["text"], language="text")

        embedding_preview_df = get_selected_chunk_embedding_preview()
        if embedding_preview_df is not None:
            st.markdown("**Embedding Preview (first dimensions):**")
            st.dataframe(
                embedding_preview_df,
                width="stretch",
                hide_index=True,
            )


def render_embedding_summary_panel() -> None:
    st.subheader("Embedding Summary")

    embedding_errors = st.session_state["embedding_errors"]
    embedding_summary = st.session_state["embedding_summary"]

    if embedding_errors:
        for error in embedding_errors:
            st.error(error)

    if embedding_summary is None:
        st.info("No embeddings generated yet. Build chunks, then click 'Generate Embeddings'.")
        return

    summary_df = pd.DataFrame([embedding_summary])
    st.dataframe(summary_df, width="stretch", hide_index=True)


def render_projection_summary_panel() -> None:
    st.subheader("Projection Summary")

    projection_errors = st.session_state["projection_errors"]
    projection_summary = st.session_state["projection_summary"]

    if projection_errors:
        for error in projection_errors:
            st.error(error)

    if projection_summary is None:
        st.info("No 2D projection yet. Generate embeddings, then click 'Project to 2D'.")
        return

    summary_df = pd.DataFrame([projection_summary])
    st.dataframe(summary_df, width="stretch", hide_index=True)


def render_vector_store_panel() -> None:
    st.subheader("LanceDB Summary")

    vector_store_errors = st.session_state["vector_store_errors"]
    vector_store_summary = st.session_state["vector_store_summary"]

    if vector_store_errors:
        for error in vector_store_errors:
            st.error(error)

    if vector_store_summary is None:
        st.info("No LanceDB table yet. Generate embeddings, then click 'Store in LanceDB'.")
        return

    summary_df = pd.DataFrame([vector_store_summary])
    st.dataframe(summary_df, width="stretch", hide_index=True)

    try:
        preview_df = fetch_table_preview(limit=8)
    except VectorStoreError as exc:
        st.warning(str(exc))
        return

    st.caption("Local LanceDB row preview")
    st.dataframe(preview_df, width="stretch", hide_index=True)


def render_embedding_map_panel() -> None:
    st.subheader("Embedding Map")

    projection_df = st.session_state["projection_df"]
    projection_summary = st.session_state["projection_summary"]
    query_projection = st.session_state["query_projection"]
    retrieval_results = st.session_state["retrieval_results"]

    if projection_df is None or projection_df.empty:
        placeholder_df = pd.DataFrame(
            [
                {"x": -1.2, "y": 0.8, "label": "chunk_001", "document": "demo_doc_a"},
                {"x": -0.4, "y": 0.2, "label": "chunk_002", "document": "demo_doc_a"},
                {"x": 0.3, "y": -0.1, "label": "chunk_003", "document": "demo_doc_b"},
                {"x": 1.1, "y": -0.6, "label": "chunk_004", "document": "demo_doc_b"},
            ]
        )

        st.caption("Project embeddings to 2D to replace this placeholder with the real map.")
        st.dataframe(placeholder_df, width="stretch", hide_index=True)
        return

    try:
        figure = create_embedding_scatter_plot(
            projection_df=projection_df,
            projection_summary=projection_summary,
            selected_chunk_id=st.session_state["selected_chunk_id"],
            retrieved_chunk_ids=get_retrieved_chunk_ids(retrieval_results),
            query_projection=query_projection,
            query_text=st.session_state["query_text"],
        )
    except PlotBuildError as exc:
        st.error(str(exc))
        return

    st.caption(
        "Retrieved chunks are outlined, the selected chunk is highlighted, "
        "and the query appears as its own point."
    )

    event = st.plotly_chart(
        figure,
        width="stretch",
        height=560,
        key="embedding_map_chart",
        on_select="rerun",
        selection_mode=("points", "box"),
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )

    handle_projection_selection(event)


def render_query_results_panel() -> None:
    st.subheader("Retrieved Sources")

    retrieval_errors = st.session_state["retrieval_errors"]
    retrieval_results = st.session_state["retrieval_results"]

    if retrieval_errors:
        for error in retrieval_errors:
            st.error(error)

    if not retrieval_results:
        with st.container(border=True):
            st.write("No query retrieval results yet.")
            st.caption("Store vectors in LanceDB, project to 2D, then run a query.")
        return

    results_df = pd.DataFrame(
        [
            {
                "rank": result["rank"],
                "filename": result["filename"],
                "chunk_index": result["chunk_index"],
                "similarity_score": result["similarity_score"],
                "preview": result["preview"],
            }
            for result in retrieval_results
        ]
    )

    st.dataframe(results_df, width="stretch", hide_index=True)

    for result in retrieval_results:
        expander_title = (
            f"#{result['rank']} | {result['filename']} | "
            f"chunk {result['chunk_index']} | score {result['similarity_score']}"
        )
        with st.expander(expander_title, expanded=result["rank"] == 1):
            st.markdown(f"**Chunk ID:** `{result['chunk_id']}`")
            st.markdown(f"**Document:** {result['filename']}")
            st.markdown(f"**Similarity Score:** {result['similarity_score']}")
            st.code(result["text"], language="text")


def render_debug_neighbor_panel() -> None:
    st.subheader("LanceDB Neighbor Preview")

    results = st.session_state["vector_search_results"]

    if not results:
        with st.container(border=True):
            st.write("No chunk-to-chunk neighbor preview yet.")
            st.caption("This remains available as a debug view from Stage 7.")
        return

    results_df = pd.DataFrame(results)

    preferred_columns = [
        "chunk_id",
        "filename",
        "chunk_index",
        "_distance",
        "preview",
    ]
    available_columns = [column for column in preferred_columns if column in results_df.columns]

    with st.container(border=True):
        st.caption("Raw LanceDB nearest-neighbor debug view.")
        st.dataframe(
            results_df.loc[:, available_columns],
            width="stretch",
            hide_index=True,
        )


def render_prompt_and_answer_panel() -> None:
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Grounded Prompt")
        with st.container(border=True):
            grounded_prompt = st.session_state["grounded_prompt"]
            if grounded_prompt:
                st.code(grounded_prompt, language="text")
            else:
                st.code(
                    "Run retrieval first. The grounded prompt will be built from the "
                    "query and retrieved chunks.",
                    language="text",
                )

    with right_col:
        st.subheader("Answer Panel")

        generation_errors = st.session_state["generation_errors"]
        generated_answer = st.session_state["generated_answer"]

        if generation_errors:
            for error in generation_errors:
                st.error(error)

        with st.container(border=True):
            if generated_answer is None:
                st.write("No answer generated yet.")
                st.caption("Enable generation and click 'Generate Answer' after retrieval.")
            else:
                st.markdown("**Answer**")
                st.write(generated_answer["answer"])

                st.markdown("**Generation Metadata**")
                metadata_df = pd.DataFrame([generated_answer["metadata"]])
                st.dataframe(metadata_df, width="stretch", hide_index=True)

                st.markdown("**Sources Used**")
                for result in st.session_state["retrieval_results"]:
                    st.markdown(
                        f"- [Source {result['rank']}] "
                        f"{result['filename']} | "
                        f"chunk {result['chunk_index']}"
                    )


def render_pipeline_tab() -> None:
    upper_left, upper_right = st.columns([1.25, 1])

    with upper_left:
        render_document_summary()
        st.divider()
        render_document_preview_panel()

    with upper_right:
        render_chunk_detail_panel()

    st.divider()

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    with summary_col_1:
        render_embedding_summary_panel()
    with summary_col_2:
        render_projection_summary_panel()
    with summary_col_3:
        render_vector_store_panel()

    st.divider()
    render_chunk_summary_panel()
    st.divider()
    render_embedding_map_panel()


def render_retrieval_tab() -> None:
    left_col, right_col = st.columns([1.25, 0.75])

    with left_col:
        render_embedding_map_panel()
        st.divider()
        render_query_results_panel()

    with right_col:
        with st.container(border=True):
            st.subheader("Retrieval Notes")
            st.write("Base map: projected chunk embeddings")
            st.write("Green X: current query point")
            st.write("Orange outlines: retrieved chunks")
            st.write("White diamond: selected chunk")
        st.divider()
        with st.expander("Debug: Chunk-to-Chunk Neighbor Preview", expanded=False):
            render_debug_neighbor_panel()


def render_generation_tab() -> None:
    with st.container(border=True):
        st.subheader("Generation Checklist")
        st.write("1. Run retrieval")
        st.write("2. Review retrieved chunks")
        st.write("3. Enable answer generation")
        st.write("4. Generate the grounded answer")

    st.divider()
    render_prompt_and_answer_panel()


def render_footer_notes(sidebar_values: dict) -> None:
    st.divider()
    with st.expander("Current UI State", expanded=False):
        st.json(
            {
                "documents_loaded": len(st.session_state["raw_documents"]),
                "chunks_built": len(st.session_state["chunks"]),
                "embeddings_ready": st.session_state["embedding_summary"] is not None,
                "projection_ready": st.session_state["projection_summary"] is not None,
                "vector_store_ready": st.session_state["vector_store_summary"] is not None,
                "retrieval_count": len(st.session_state["retrieval_results"]),
                "answer_ready": st.session_state["generated_answer"] is not None,
                "chunk_size": sidebar_values["chunk_size"],
                "chunk_overlap": sidebar_values["chunk_overlap"],
                "top_k": sidebar_values["top_k"],
                "projection_method": sidebar_values["projection_method"],
                "query_text": sidebar_values["query_text"],
                "enable_generation": sidebar_values["enable_generation"],
            }
        )


def render_app() -> None:
    render_header()
    sidebar_values = render_sidebar()

    st.session_state["query_text"] = sidebar_values["query_text"]

    render_status_metrics(sidebar_values)
    render_overview_banner()

    pipeline_tab, retrieval_tab, generation_tab = st.tabs(
        ["🧩 Pipeline", "🔎 Retrieval", "🧠 Generation"],
        width="stretch",
        key="main_workspace_tabs",
        on_change="rerun",
    )

    if pipeline_tab.open:
        with pipeline_tab:
            render_pipeline_tab()

    if retrieval_tab.open:
        with retrieval_tab:
            render_retrieval_tab()

    if generation_tab.open:
        with generation_tab:
            render_generation_tab()

    render_footer_notes(sidebar_values)
