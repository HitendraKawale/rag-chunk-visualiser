from __future__ import annotations

import pandas as pd
import plotly.express as px

from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)


class PlotBuildError(RuntimeError):
    pass


def build_plot_dataframe(projection_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        raise PlotBuildError("Projection dataframe is empty.")

    plot_df = projection_df.copy()

    # Keep a small, fixed point size so one document doesn't visually swallow others.
    plot_df["point_size"] = 10.0

    # Add tiny deterministic jitter only for exact duplicate coordinates.
    # This makes overlapping points visible without changing the underlying structure much.
    duplicate_rank = plot_df.groupby(["x", "y"]).cumcount()
    plot_df["x_plot"] = plot_df["x"] + (duplicate_rank * 0.01)
    plot_df["y_plot"] = plot_df["y"] + (duplicate_rank * 0.01)

    return plot_df


def create_embedding_scatter_plot(
    projection_df: pd.DataFrame,
    projection_summary: dict | None = None,
    selected_chunk_id: str | None = None,
    retrieved_chunk_ids: list[str] | None = None,
    query_projection: dict | None = None,
    query_text: str | None = None,
):
    plot_df = build_plot_dataframe(projection_df)
    retrieved_chunk_ids = retrieved_chunk_ids or []

    method_label = "2D Projection"
    if projection_summary is not None:
        method_label = str(projection_summary.get("method", "2D Projection")).upper()

    fig = px.scatter(
        plot_df,
        x="x_plot",
        y="y_plot",
        color="filename",
        symbol="filename",
        size="point_size",
        hover_name="chunk_id",
        hover_data={
            "filename": True,
            "chunk_index": True,
            "char_count": True,
            "word_count": True,
            "preview": True,
            "point_size": False,
            "x_plot": False,
            "y_plot": False,
            "x": ":.4f",
            "y": ":.4f",
        },
        custom_data=["chunk_id", "doc_id", "plot_index"],
        render_mode="svg",
        title=f"{method_label} embedding map",
    )

    fig.update_traces(
        marker={
            "size": 10,
            "line": {"width": 1},
            "opacity": 0.6,
        },
    )

    if retrieved_chunk_ids:
        retrieved_df = plot_df[plot_df["chunk_id"].isin(retrieved_chunk_ids)]
        if not retrieved_df.empty:
            fig.add_scatter(
                x=retrieved_df["x_plot"],
                y=retrieved_df["y_plot"],
                mode="markers",
                name="Retrieved",
                customdata=retrieved_df[["chunk_id", "doc_id", "plot_index"]].values,
                marker={
                    "symbol": "circle-open",
                    "size": 22,
                    "line": {"width": 3, "color": "#F59E0B"},
                },
                hovertemplate="Retrieved: %{customdata[0]}<extra></extra>",
            )

    if selected_chunk_id:
        selected_df = plot_df[plot_df["chunk_id"] == selected_chunk_id]
        if not selected_df.empty:
            fig.add_scatter(
                x=selected_df["x_plot"],
                y=selected_df["y_plot"],
                mode="markers",
                name="Selected",
                customdata=selected_df[["chunk_id", "doc_id", "plot_index"]].values,
                marker={
                    "symbol": "diamond-open",
                    "size": 24,
                    "line": {"width": 3, "color": "#FFFFFF"},
                },
                hovertemplate="Selected: %{customdata[0]}<extra></extra>",
            )

    if query_projection is not None:
        query_label = query_text.strip() if isinstance(query_text, str) else "Query"
        fig.add_scatter(
            x=[query_projection["x"]],
            y=[query_projection["y"]],
            mode="markers+text",
            text=["Query"],
            textposition="top center",
            name="Query",
            customdata=[["__query__", "__query__", -1]],
            marker={
                "symbol": "x",
                "size": 18,
                "color": "#22C55E",
                "line": {"width": 2},
            },
            hovertemplate=f"Query: {query_label}<extra></extra>",
        )

    fig.update_layout(
        height=560,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
        xaxis_title="Projection X",
        yaxis_title="Projection Y",
        legend_title_text="Legend",
        dragmode="select",
        clickmode="event+select",
    )

    return fig
