from __future__ import annotations

from typing import Any

SAMPLE_DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc_001",
        "text": "Blue resume template modern professional",
        "tokens": ["blue", "resume", "template", "modern", "professional"],
        "clicks": 120,
        "age_days": 10,
    },
    {
        "id": "doc_002",
        "text": "Wedding invitation floral theme",
        "tokens": ["wedding", "invitation", "floral", "theme"],
        "clicks": 80,
        "age_days": 5,
    },
    {
        "id": "doc_003",
        "text": "Business card minimalist design",
        "tokens": ["business", "card", "minimalist", "design"],
        "clicks": 50,
        "age_days": 20,
    },
    {
        "id": "doc_004",
        "text": "Resume template clean layout",
        "tokens": ["resume", "template", "clean", "layout"],
        "clicks": 200,
        "age_days": 60,
    },
    {
        "id": "doc_005",
        "text": "Birthday invitation fun colorful",
        "tokens": ["birthday", "invitation", "fun", "colorful"],
        "clicks": 30,
        "age_days": 2,
    },
]


def get_sample_documents() -> list[dict[str, Any]]:
    """Return a shallow copy of the in-memory sample documents.

    Returns
    -------
    list[dict[str, Any]]
        A copy of SAMPLE_DOCUMENTS so callers do not mutate the module constant.
    """
    return list(SAMPLE_DOCUMENTS)
