# -----------------------------------------------------------
# Simple RAG Demo - Generator
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import logging

from google import genai

from simple_rag.rag.utils.generation_helpers import build_prompt
from simple_rag.settings import settings
from simple_rag.utils.llm_helpers import generate_text_gemini_async

log = logging.getLogger(__name__)


async def generate_answer(
    query: str,
    chunks: list[dict],
    client: genai.Client,
    model: str = settings.GEMINI_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Genera una respuesta fundamentada a partir de los chunks recuperados, usando Gemini.

    Args:
        query: Pregunta original del usuario.
        chunks: Dicts de chunks recuperados de retrieval.retrieve().
        client: Cliente de Gemini configurado.
        model: Nombre del modelo de Gemini para la generación.
        max_tokens: Máximo de tokens de salida.

    Returns:
        String de la respuesta generada, fundamentada en los chunks provistos.
    """
    if not chunks:
        return "No se encontró información relevante para responder a tu pregunta."

    prompt = build_prompt(query, chunks)
    answer = await generate_text_gemini_async(
        client=client,
        prompt=prompt,
        model_name=model,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    log.debug("Respuesta generada (%d caracteres)", len(answer))
    return answer
