# -----------------------------------------------------------
# Simple RAG Demo - Generation Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Lógica pura de armado de prompt para el generador de respuestas."""

_ANSWER_SYSTEM_PROMPT = """Eres un asistente de preguntas y respuestas en español. \
Responde de forma precisa y concisa basándote ÚNICAMENTE en los fragmentos de contexto proporcionados. \
Comienza tu respuesta directamente con la información relevante disponible en el contexto — evita abrir \
la respuesta con frases negativas como "el contexto no especifica..." o "el contexto no menciona...". \
Si algún detalle concreto de la pregunta no aparece en el contexto, indícalo en una frase breve al final \
de la respuesta, no al principio. No inventes ni asumas información que no esté en el contexto. \
Al final de tu respuesta incluye las fuentes utilizadas en formato: Fuentes: [dominio1, dominio2, ...]"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Arma el prompt de respuesta fundamentada a partir de los chunks recuperados.

    Args:
        query: Pregunta original del usuario.
        chunks: Dicts de chunks recuperados, cada uno con una clave 'text' y
            opcionalmente 'source_domain'/'link' para la cita.

    Returns:
        String del prompt completo, listo para enviar al modelo generador.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source_domain") or chunk.get("link") or "desconocido"
        context_blocks.append(f"[{i}] (Fuente: {source})\n{chunk['text']}")
    context_text = "\n\n".join(context_blocks)
    return f"{_ANSWER_SYSTEM_PROMPT}\n\nContexto:\n{context_text}\n\nPregunta: {query}\n\nRespuesta:"
