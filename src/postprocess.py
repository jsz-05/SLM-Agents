from __future__ import annotations

import re


def clean_answer(answer: str) -> str:
    """Generic formatting cleanup applied equally to all methods."""
    cleaned = str(answer).strip().strip("\"'`")
    cleaned = re.sub(r"^\s*the\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[.。]+$", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned
