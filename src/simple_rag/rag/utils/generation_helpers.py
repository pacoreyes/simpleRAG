# -----------------------------------------------------------
# Simple RAG Demo - Generation Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""Lógica pura de armado de prompt para el generador de respuestas."""

_ANSWER_SYSTEM_PROMPT = """Eres un asistente de preguntas y respuestas en español.

Instrucciones:
1. Responde de forma precisa y concisa basándote ÚNICAMENTE en los fragmentos de contexto proporcionados.
2. Comienza tu respuesta directamente con la información relevante disponible en el contexto — evita abrir la respuesta con frases negativas como "el contexto no especifica..." o "el contexto no menciona...".
3. Si algún detalle concreto de la pregunta no aparece en el contexto, indícalo en una frase breve al final de la respuesta, no al principio.
4. No inventes ni asumas información que no esté en el contexto.
5. REGLA OBLIGATORIA sobre la línea de fuentes — se aplica siempre, sin excepciones:
   - Si usaste al menos un fragmento del contexto para responder → termina tu respuesta con una línea: Fuentes: [dominio1, dominio2, ...]
   - Si NINGÚN fragmento del contexto es relevante para la pregunta → tu respuesta debe decir que no hay información disponible, y tu respuesta NUNCA debe incluir la línea "Fuentes:". Esto aplica incluso si el contexto tiene fragmentos sobre otros temas — no los cites si no responden la pregunta.

Ejemplo INCORRECTO cuando el contexto no es relevante (nunca respondas así):
  "El contexto no contiene información sobre cómo se hace un ceviche.
  Fuentes: [www.japan.travel, japonpedia.com]"
  ← MAL: no se debe agregar "Fuentes:" porque ningún fragmento respondió la pregunta.

Ejemplo CORRECTO en ese mismo caso:
  "El contexto no contiene información sobre cómo se hace un ceviche."
  ← BIEN: sin línea de fuentes."""


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
