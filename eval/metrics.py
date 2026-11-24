"""Retrieval metrics. Pure functions: no files, no command line."""

from collections.abc import Collection, Sequence


def recall_at_k(expected: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """Share of the expected entities found in the top k. Ignores their order.

    Duplicates are not removed before cutting: two chunks of the same entity take up two
    slots, exactly as they will in the prompt.
    """
    _check(expected, k)
    wanted = set(expected)
    return len(wanted & set(retrieved[:k])) / len(wanted)


def reciprocal_rank(expected: Collection[str], retrieved: Sequence[str], k: int) -> float:
    """1 / position of the first hit in the top k, or 0. Ignores any later hit."""
    _check(expected, k)
    for rank, entity in enumerate(retrieved[:k], start=1):
        if entity in expected:
            return 1 / rank
    return 0.0


def _check(expected: Collection[str], k: int) -> None:
    if not expected:
        raise ValueError("expected entities cannot be empty")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
