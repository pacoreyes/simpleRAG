# -----------------------------------------------------------
# LLM and Tokenizer Helpers
# simple_rag — shared utilities
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

"""
Generic LLM inference helpers using Google Gemini (new SDK) and Tiktoken.

Covers: tokenization, Gemini client setup, synchronous and asynchronous
text generation, and LLM output parsing. No domain-specific logic.
"""

import re
from typing import Any, Optional

import tiktoken
from google import genai
from google.genai import types


class TiktokenTokenizer:
    """Wrapper to make tiktoken compatible with the transformers tokenizer interface."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encodes text into tokens, ignoring extra arguments for compatibility."""
        return self.encoding.encode(text)


def load_tokenizer_only(encoding_name: str = "cl100k_base") -> TiktokenTokenizer:
    """
    Loads the Tiktoken tokenizer.

    Args:
        encoding_name: Tiktoken encoding name (e.g., "cl100k_base").

    Returns:
        The TiktokenTokenizer instance.
    """
    return TiktokenTokenizer(encoding_name)


# ---------------------------------------------------------------------------
# Gemini Helpers (using new google-genai SDK)
# ---------------------------------------------------------------------------

def get_gemini_client(api_key: str) -> genai.Client:
    """
    Returns a configured Google Gemini client.

    Args:
        api_key: Google Gemini API key.

    Returns:
        A genai.Client instance.
    """
    if not api_key:
        raise ValueError("Gemini API key is required.")
    return genai.Client(api_key=api_key)


def generate_text_gemini(
    client: genai.Client,
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
) -> str:
    """
    Generates text using Google Gemini API (new SDK).

    Args:
        client: Configured Gemini client.
        prompt: The input prompt.
        model_name: Gemini model name (default: "gemini-2.0-flash").
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        mime_type: Optional response MIME type (e.g., "application/json").
        response_schema: Optional response schema (Pydantic model or dict).

    Returns:
        Generated text content.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=mime_type,
        response_schema=response_schema,
        # Disable safety filters for extraction tasks on public data
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
        raise RuntimeError(f"Gemini generation failed: {e}") from e


async def generate_text_gemini_async(
    client: genai.Client,
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    mime_type: Optional[str] = None,
    response_schema: Optional[Any] = None,
) -> str:
    """Generates text using Google Gemini API asynchronously (native async).

    Uses ``client.aio.models.generate_content`` so the call is non-blocking
    without requiring ``asyncio.to_thread``.

    Args:
        client: Configured Gemini client (same instance returned by
            ``get_gemini_client``).
        prompt: The input prompt.
        model_name: Gemini model name (default: "gemini-2.0-flash").
        max_tokens: Max output tokens.
        temperature: Sampling temperature.
        mime_type: Optional response MIME type (e.g., "application/json").
        response_schema: Optional response schema (Pydantic model or dict).

    Returns:
        Generated text content.
    """
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type=mime_type,
        response_schema=response_schema,
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
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config
        )
        if response.text is None:
            raise RuntimeError("Gemini returned empty response (possibly filtered).")
        return response.text
    except Exception as e:
        raise RuntimeError(f"Gemini generation failed: {e}") from e


def get_device() -> "torch.device":
    """
    Detects the best available compute device.

    Returns:
        torch.device: CUDA if available, MPS for Apple Silicon, otherwise CPU.
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def strip_json_fences(text: str) -> str:
    """
    Removes markdown code fences from LLM output and extracts the outermost JSON content.

    Args:
        text: Raw LLM response that may be wrapped in ```json ... ``` fences.

    Returns:
        Clean JSON string.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    brace_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()
    return text.strip()
