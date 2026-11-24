"""Retrieval metrics against values worked out by hand."""

import pytest

from eval.metrics import recall_at_k, reciprocal_rank

RETRIEVED = ["X", "A", "Y", "Z", "W", "B"]


@pytest.mark.parametrize(
    ("expected", "k", "value"),
    [
        pytest.param(["A", "B"], 5, 1 / 2, id="one-of-two"),
        pytest.param(["A", "B"], 6, 1.0, id="both-with-larger-k"),
        pytest.param(["A"], 2, 1.0, id="last-position-counts"),
        pytest.param(["A"], 1, 0.0, id="just-outside-k"),
        pytest.param(["A", "Y", "B"], 5, 2 / 3, id="two-of-three"),
        pytest.param(["M"], 5, 0.0, id="none"),
    ],
)
def test_recall_at_k(expected: list[str], k: int, value: float) -> None:
    assert recall_at_k(expected, RETRIEVED, k) == pytest.approx(value)


@pytest.mark.parametrize(
    ("expected", "k", "value"),
    [
        pytest.param(["A", "B"], 5, 1 / 2, id="first-hit-at-rank-2"),
        pytest.param(["X"], 5, 1.0, id="rank-1"),
        pytest.param(["Z", "Y"], 5, 1 / 3, id="best-rank-wins"),
        pytest.param(["W"], 5, 1 / 5, id="last-position-counts"),
        pytest.param(["B"], 5, 0.0, id="just-outside-k"),
        pytest.param(["M"], 5, 0.0, id="none"),
    ],
)
def test_reciprocal_rank(expected: list[str], k: int, value: float) -> None:
    assert reciprocal_rank(expected, RETRIEVED, k) == pytest.approx(value)


def test_duplicates_take_up_slots() -> None:
    # Two chunks of A leave B in position 3, outside the top 2
    assert recall_at_k(["A", "B"], ["A", "A", "B"], 2) == pytest.approx(1 / 2)
    assert reciprocal_rank(["B"], ["A", "A", "B"], 3) == pytest.approx(1 / 3)


def test_a_duplicated_hit_does_not_count_twice() -> None:
    assert recall_at_k(["A", "B"], ["A", "A"], 5) == pytest.approx(1 / 2)


def test_nothing_retrieved_scores_zero() -> None:
    assert recall_at_k(["A"], [], 5) == 0.0
    assert reciprocal_rank(["A"], [], 5) == 0.0


@pytest.mark.parametrize("metric", [recall_at_k, reciprocal_rank])
def test_metrics_reject_meaningless_input(metric) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="expected"):
        metric([], ["A"], 5)
    with pytest.raises(ValueError, match="k must be"):
        metric(["A"], ["A"], 0)
