from __future__ import annotations

import streamlit as st

from rag_chunk_visualizer.app.layout import render_app
from rag_chunk_visualizer.app.state import initialize_session_state
from rag_chunk_visualizer.core.config import ensure_directories, settings
from rag_chunk_visualizer.core.logging import configure_logging, get_logger

st.set_page_config(
    page_title=settings.app_name,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

configure_logging()
logger = get_logger("streamlit_app")


def main() -> None:
    ensure_directories()
    initialize_session_state()

    logger.info("Starting Streamlit app shell")
    render_app()


if __name__ == "__main__":
    main()
