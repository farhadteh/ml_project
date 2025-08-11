from __future__ import annotations

from typing import Any

from data.sample_data import get_sample_documents


def test_get_sample_documents_basic_shape() -> None:
    docs = get_sample_documents()
    assert isinstance(docs, list)
    assert len(docs) >= 3
    first: dict[str, Any] = docs[0]
    assert "id" in first and isinstance(first["id"], str)
    assert "tokens" in first and isinstance(first["tokens"], list)
    assert all(isinstance(tok, str) for tok in first["tokens"])
