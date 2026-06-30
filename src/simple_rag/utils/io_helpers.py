# -----------------------------------------------------------
# I/O and Serialization Helpers
# simple_rag — shared utilities
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import msgspec

JSONDecodeError = msgspec.DecodeError


async def async_read_json_file(path: Path) -> Optional[Any]:
    """
    Reads and decodes a JSON file asynchronously using msgspec.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded data, or None if the file does not exist or decoding fails.
    """
    if not await asyncio.to_thread(path.exists):
        return None

    try:
        def read_bytes():
            with open(path, "rb") as f:
                return f.read()
        
        data = await asyncio.to_thread(read_bytes)
        return msgspec.json.decode(data)
    except (OSError, msgspec.DecodeError):
        return None


async def async_write_json_file(path: Path, data: Any) -> None:
    """
    Encodes and writes data to a JSON file asynchronously using msgspec.

    Ensures parent directories exist.

    Args:
        path: Path to the JSON file to write.
        data: Data to be JSON-encoded and written.
    """
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)

    def write_bytes():
        with open(path, "wb") as f:
            f.write(msgspec.json.encode(data))

    await asyncio.to_thread(write_bytes)


async def async_read_text_file(path: Path) -> Optional[str]:
    """
    Reads a text file asynchronously.

    Args:
        path: Path to the text file.

    Returns:
        The file content as a string, or None if the file does not exist or on error.
    """
    if not await asyncio.to_thread(path.exists):
        return None

    try:
        def read_text():
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        
        return await asyncio.to_thread(read_text)
    except OSError:
        return None


async def async_write_text_file(path: Path, content: str) -> None:
    """
    Writes content to a text file asynchronously.

    Ensures parent directories exist.

    Args:
        path: Path to the text file to write.
        content: String content to write.
    """
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)

    def write_text():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(write_text)


def generate_cache_key(text: str) -> str:
    """
    Creates a SHA256 hash of a string to use as a cache key.

    Args:
        text: Input string to hash.

    Returns:
        SHA256 hex digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_url_domain(url: str | None) -> str | None:
    """Extracts the network location (domain) from a URL.

    Args:
        url: Full URL string, or None / empty string.

    Returns:
        The netloc component (e.g. ``"airhelp.com"``), or None if the URL
        is absent, empty, or has no parseable host.
    """
    if not url:
        return None
    return urlparse(url).netloc or None


def decode_json(data: bytes) -> Any:
    """
    Decodes JSON bytes using msgspec.

    Args:
        data: JSON bytes to decode.

    Returns:
        The decoded data.
    """
    return msgspec.json.decode(data)
