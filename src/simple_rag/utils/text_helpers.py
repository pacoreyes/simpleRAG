# -----------------------------------------------------------
# Generic Text and String Helpers
# simple_rag — shared utilities
#
# (C) 2026 Juan-Francisco Reyes, Essen, Germany
# Released under MIT License
# email pacoreyes@protonmail.com
# -----------------------------------------------------------

from typing import Any, Optional, Sequence


def format_list_natural_language(items: Optional[Sequence[Any]]) -> str:
    """
    Formats a sequence into a natural-language string with Oxford comma.

    Args:
        items: Sequence of items to format.

    Returns:
        Human-readable string, e.g. "a, b, and c".
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
