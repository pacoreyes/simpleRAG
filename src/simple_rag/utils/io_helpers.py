# -----------------------------------------------------------
# Simple RAG Demo - I/O and Serialization Helpers
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
    Lee y decodifica un archivo JSON de forma asíncrona usando msgspec.

    Args:
        path: Ruta al archivo JSON.

    Returns:
        Los datos decodificados, o None si el archivo no existe o falla la decodificación.
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
    Codifica y escribe datos en un archivo JSON de forma asíncrona usando msgspec.

    Asegura que los directorios padre existan.

    Args:
        path: Ruta del archivo JSON a escribir.
        data: Datos a codificar en JSON y escribir.
    """
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)

    def write_bytes():
        with open(path, "wb") as f:
            f.write(msgspec.json.encode(data))

    await asyncio.to_thread(write_bytes)


async def async_read_text_file(path: Path) -> Optional[str]:
    """
    Lee un archivo de texto de forma asíncrona.

    Args:
        path: Ruta al archivo de texto.

    Returns:
        El contenido del archivo como string, o None si el archivo no existe o hay un error.
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
    Escribe contenido en un archivo de texto de forma asíncrona.

    Asegura que los directorios padre existan.

    Args:
        path: Ruta del archivo de texto a escribir.
        content: Contenido en string a escribir.
    """
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)

    def write_text():
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    await asyncio.to_thread(write_text)


def generate_cache_key(text: str) -> str:
    """
    Crea un hash SHA256 de un string para usar como clave de caché.

    Args:
        text: String de entrada a hashear.

    Returns:
        Hex digest del SHA256.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_url_domain(url: str | None) -> str | None:
    """Extrae el network location (dominio) de una URL.

    Args:
        url: String de URL completa, o None / string vacío.

    Returns:
        El componente netloc (p. ej. ``"airhelp.com"``), o None si la URL
        está ausente, vacía, o no tiene un host parseable.
    """
    if not url:
        return None
    return urlparse(url).netloc or None


def decode_json(data: bytes) -> Any:
    """
    Decodifica bytes JSON usando msgspec.

    Args:
        data: Bytes JSON a decodificar.

    Returns:
        Los datos decodificados.
    """
    return msgspec.json.decode(data)
