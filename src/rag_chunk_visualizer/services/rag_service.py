from __future__ import annotations

from typing import Any

from ollama import Client, ResponseError

from rag_chunk_visualizer.core.config import settings
from rag_chunk_visualizer.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a grounded RAG assistant. "
    "Answer only from the provided context. "
    "Do not invent facts. "
    "If the context is insufficient, explicitly say that the answer cannot be "
    "determined from the retrieved context. "
    "When possible, cite the source labels exactly as provided, such as [Source 1]."
)


class RAGGenerationError(RuntimeError):
    pass


def validate_retrieval_results(retrieval_results: list[dict]) -> None:
    if not retrieval_results:
        raise RAGGenerationError("Run retrieval before generating a grounded answer.")


def validate_query_text(query_text: str) -> None:
    if not isinstance(query_text, str) or not query_text.strip():
        raise RAGGenerationError("Enter a non-empty query before generating an answer.")


def format_source_label(result: dict) -> str:
    rank = result.get("rank", "?")
    filename = result.get("filename", "unknown")
    chunk_index = result.get("chunk_index", "?")
    similarity_score = result.get("similarity_score", "n/a")

    return f"[Source {rank}] {filename} | chunk {chunk_index} | score {similarity_score}"


def build_context_block(retrieval_results: list[dict]) -> str:
    validate_retrieval_results(retrieval_results)

    sections: list[str] = []
    for result in retrieval_results:
        label = format_source_label(result)
        text = str(result.get("text", "")).strip()

        sections.append(f"{label}\n{text}")

    return "\n\n".join(sections)


def build_grounded_prompt(query_text: str, retrieval_results: list[dict]) -> str:
    validate_query_text(query_text)
    validate_retrieval_results(retrieval_results)

    context_block = build_context_block(retrieval_results)

    return (
        "User question:\n"
        f"{query_text.strip()}\n\n"
        "Retrieved context:\n"
        f"{context_block}\n\n"
        "Instructions:\n"
        "1. Answer only from the retrieved context.\n"
        "2. If the context is insufficient, say so clearly.\n"
        "3. Keep the answer concise and factual.\n"
        "4. End with a short 'Sources used:' line that references the source labels.\n"
    )


def get_ollama_client(
    base_url: str | None = None,
    timeout_seconds: float | None = None,
) -> Client:
    resolved_base_url = base_url or settings.ollama_base_url
    resolved_timeout = (
        settings.ollama_timeout_seconds if timeout_seconds is None else timeout_seconds
    )

    return Client(
        host=resolved_base_url,
        timeout=resolved_timeout,
    )


def _response_field(response: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(response, dict):
        return response.get(field_name, default)

    return getattr(response, field_name, default)


def _extract_message_content(response: Any) -> str:
    message = _response_field(response, "message", {})
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()

    return str(getattr(message, "content", "")).strip()


def _ns_to_seconds(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return round(float(value) / 1_000_000_000, 4)
    except (TypeError, ValueError):
        return None


def generate_grounded_answer(
    query_text: str,
    retrieval_results: list[dict],
    model_name: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    num_predict: int | None = None,
    timeout_seconds: float | None = None,
    keep_alive: str | None = None,
) -> dict:
    prompt = build_grounded_prompt(query_text, retrieval_results)

    resolved_model_name = model_name or settings.ollama_model
    resolved_temperature = settings.ollama_temperature if temperature is None else temperature
    resolved_num_predict = settings.ollama_num_predict if num_predict is None else num_predict
    resolved_keep_alive = keep_alive or settings.ollama_keep_alive

    client = get_ollama_client(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        response = client.chat(
            model=resolved_model_name,
            messages=messages,
            stream=False,
            options={
                "temperature": resolved_temperature,
                "num_predict": resolved_num_predict,
            },
            keep_alive=resolved_keep_alive,
        )
    except ResponseError as exc:
        error_text = getattr(exc, "error", str(exc))
        status_code = getattr(exc, "status_code", None)

        if status_code == 404:
            raise RAGGenerationError(
                f"Ollama model '{resolved_model_name}' was not found. "
                f"Pull it first with: ollama pull {resolved_model_name}"
            ) from exc

        raise RAGGenerationError(f"Ollama request failed: {error_text}") from exc
    except Exception as exc:
        raise RAGGenerationError(
            "Failed to contact the local Ollama server. Make sure Ollama is running and reachable."
        ) from exc

    answer_text = _extract_message_content(response)
    if not answer_text:
        raise RAGGenerationError("Ollama returned an empty answer.")

    result = {
        "answer": answer_text,
        "prompt": prompt,
        "model": resolved_model_name,
        "metadata": {
            "done_reason": _response_field(response, "done_reason"),
            "prompt_eval_count": _response_field(response, "prompt_eval_count"),
            "eval_count": _response_field(response, "eval_count"),
            "total_duration_s": _ns_to_seconds(_response_field(response, "total_duration")),
            "load_duration_s": _ns_to_seconds(_response_field(response, "load_duration")),
            "keep_alive": resolved_keep_alive,
            "temperature": resolved_temperature,
            "num_predict": resolved_num_predict,
        },
    }

    logger.info(
        "Generated grounded answer | model=%s | sources=%s",
        resolved_model_name,
        len(retrieval_results),
    )

    return result
