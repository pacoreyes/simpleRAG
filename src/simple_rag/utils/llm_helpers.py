# -----------------------------------------------------------
# Simple RAG Demo - LLM and NLP Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""
Helpers genéricos de inferencia LLM usando Google Gemini (SDK nuevo) y Tiktoken.

Cubre: tokenización, configuración del cliente de Gemini, generación de texto
síncrona y asíncrona, y parsing de la salida del LLM. Sin lógica de dominio.
"""

import re
from typing import Any, Optional

import tiktoken
from google import genai
from google.genai import types


class TiktokenTokenizer:
    """Wrapper para hacer que tiktoken sea compatible con la interfaz de tokenizer de transformers."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Codifica texto en tokens, ignorando argumentos extra por compatibilidad."""
        return self.encoding.encode(text)


def load_tokenizer_only(encoding_name: str = "cl100k_base") -> TiktokenTokenizer:
    """
    Carga el tokenizer de Tiktoken.

    Args:
        encoding_name: Nombre del encoding de Tiktoken (p. ej., "cl100k_base").

    Returns:
        La instancia de TiktokenTokenizer.
    """
    return TiktokenTokenizer(encoding_name)


# ---------------------------------------------------------------------------
# Helpers de Gemini (usando el nuevo SDK google-genai)
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: str) -> genai.Client:
    """
    Devuelve un cliente de Google Gemini configurado.

    Args:
        api_key: Clave de API de Google Gemini.

    Returns:
        Una instancia de genai.Client.
    """
    if not api_key:
        raise ValueError("Se requiere una clave de API de Gemini.")
    return genai.Client(api_key=api_key)


def generate_text_gemini(
    client: genai.Client,
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
    thinking_budget: Optional[int] = None,
) -> str:
    """
    Genera texto usando la API de Google Gemini (SDK nuevo).

    Args:
        client: Cliente de Gemini configurado.
        prompt: El prompt de entrada.
        model_name: Nombre del modelo de Gemini (default: "gemini-2.0-flash").
        max_tokens: Máximo de tokens de salida.
        temperature: Temperatura de sampling.
        mime_type: Tipo MIME de respuesta opcional (p. ej., "application/json").
        response_schema: Schema de respuesta opcional (modelo Pydantic o dict).
        thinking_budget: Budget de tokens de "thinking" de Gemini, opcional; se
            omite de la request cuando es None.

    Returns:
        Contenido de texto generado.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=mime_type,
        response_schema=response_schema,
        thinking_config=(
            types.ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        ),
        # Deshabilita los filtros de seguridad para tareas de extracción sobre datos públicos
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ]
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        return response.text
    except Exception as e:
        raise RuntimeError(f"Falló la generación con Gemini: {e}") from e


async def generate_text_gemini_async(
    client: genai.Client,
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
    thinking_budget: Optional[int] = None,
    retry_count: int = 8,
    backoff_factor: float = 2.0,
) -> str:
    """Genera texto usando la API de Google Gemini de forma asíncrona (async nativo).

    Usa ``client.aio.models.generate_content`` para que la llamada sea no
    bloqueante sin necesidad de ``asyncio.to_thread``.

    Args:
        client: Cliente de Gemini configurado (misma instancia devuelta por
            ``get_gemini_client``).
        prompt: El prompt de entrada.
        model_name: Nombre del modelo de Gemini (default: "gemini-2.0-flash").
        max_tokens: Máximo de tokens de salida.
        temperature: Temperatura de sampling.
        mime_type: Tipo MIME de respuesta opcional (p. ej., "application/json").
        response_schema: Schema de respuesta opcional (modelo Pydantic o dict).
        thinking_budget: Budget de tokens de "thinking" de Gemini, opcional; se
            omite de la request cuando es None.

    Returns:
        Contenido de texto generado.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=mime_type,
        response_schema=response_schema,
        thinking_config=(
            types.ThinkingConfig(thinking_budget=thinking_budget)
            if thinking_budget is not None
            else None
        ),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ]
    )

    import asyncio as _asyncio

    for attempt in range(retry_count):
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response.text is None:
                raise RuntimeError("Gemini devolvió una respuesta vacía (posiblemente filtrada).")
            return response.text
        except Exception as e:
            err = str(e)
            is_retryable = any(code in err for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if is_retryable and attempt < retry_count - 1:
                wait = min((backoff_factor ** attempt) * 3.0, 120.0)
                await _asyncio.sleep(wait)
                continue
            raise RuntimeError(f"Falló la generación con Gemini: {e}") from e
    raise RuntimeError("Falló la generación con Gemini: se superó el máximo de reintentos")


def strip_json_fences(text: str) -> str:
    """
    Elimina fences de código markdown de la salida del LLM y extrae el primer valor JSON.

    Escanea el primer bloque balanceado ``{...}`` o ``[...]``, llevando la
    cuenta de la profundidad de anidación y saltando caracteres tipo brace
    dentro de literales de string. Esto se detiene al final del primer valor
    JSON completo en lugar de abarcar de forma greedy hasta el último brace
    del texto, así que no se traga contenido siguiente cuando un modelo emite
    más de un objeto JSON en una sola respuesta.

    Args:
        text: Respuesta cruda del LLM que puede venir envuelta en fences ```json ... ```.

    Returns:
        String JSON limpio.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return text

    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return text[start:].strip()
