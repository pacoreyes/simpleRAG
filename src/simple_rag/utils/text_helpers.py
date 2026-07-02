# -----------------------------------------------------------
# Simple RAG Demo - Generic Text and String Helpers
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from typing import Any, Optional, Sequence


def format_list_natural_language(items: Optional[Sequence[Any]]) -> str:
    """
    Formatea una secuencia como string en lenguaje natural con coma de Oxford.

    Args:
        items: Secuencia de elementos a formatear.

    Returns:
        String legible para humanos, p. ej. "a, b, and c".
    """
    if not items:
        return ""

    clean_items = []
    seen: set = set()
    for x in items:
        if x and x not in seen:
            clean_items.append(str(x))
            seen.add(x)

    if not clean_items:
        return ""
    if len(clean_items) == 1:
        return clean_items[0]
    if len(clean_items) == 2:
        return f"{clean_items[0]} and {clean_items[1]}"
    return f"{', '.join(clean_items[:-1])}, and {clean_items[-1]}"
