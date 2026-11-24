"""Rules of the evaluation dataset: one test per rule agreed in US0.2.1."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from eval.schema import Difficulty, EvalQuestion, QuestionType, Unit

QUANTITATIVE: dict[str, Any] = {
    "id": "Q001",
    "question": "¿Cuántos Constructores necesito para 30 de Hormigón por minuto?",
    "question_type": "quantitative",
    "expected_answer": "2 Constructores.",
    "expected_entities": ["Recipe_Concrete_C"],
    "expected_numbers": [{"value": 2, "unit": "machines", "entity": "Build_ConstructorMk1_C"}],
    "tier": 0,
    "difficulty": "easy",
    "notes": "15/min por Constructor; 30 / 15 = 2",
}

IN_SCOPE: dict[str, Any] = {**QUANTITATIVE, "expected_numbers": [], "notes": ""}

OUT_OF_SCOPE: dict[str, Any] = {
    "id": "Q002",
    "question": "¿Qué mod me recomiendas?",
    "question_type": "out_of_scope",
    "expected_answer": "No lo sé: los mods están fuera de alcance.",
    "expected_entities": [],
    "expected_numbers": [],
    "tier": None,
    "difficulty": "easy",
    "notes": "",
}


def parse(base: dict[str, Any], **changes: Any) -> EvalQuestion:
    """Validate through JSON, exactly as a line of dataset.jsonl is read."""
    return EvalQuestion.model_validate_json(json.dumps({**base, **changes}))


def number(**changes: Any) -> dict[str, Any]:
    return {**QUANTITATIVE["expected_numbers"][0], **changes}


# --- Valid questions ---------------------------------------------------------


def test_quantitative_question_with_all_fields_is_valid() -> None:
    question = parse(QUANTITATIVE)

    assert question.question_type is QuestionType.QUANTITATIVE
    assert question.difficulty is Difficulty.EASY
    assert question.expected_entities == ("Recipe_Concrete_C",)
    assert question.expected_numbers[0].value == 2
    assert question.expected_numbers[0].unit is Unit.MACHINES


@pytest.mark.parametrize("question_type", ["conceptual", "lookup", "guide"])
def test_numbers_and_notes_are_optional_for_other_in_scope_types(question_type: str) -> None:
    assert parse(IN_SCOPE, question_type=question_type).expected_numbers == ()


def test_out_of_scope_question_is_valid() -> None:
    assert parse(OUT_OF_SCOPE).tier is None


@pytest.mark.parametrize("tier", [0, 4])
def test_tier_limits_are_inclusive(tier: int) -> None:
    assert parse(QUANTITATIVE, tier=tier).tier == tier


@pytest.mark.parametrize(
    "entity",
    ["Desc_Rotor_C", "Recipe_Alternate_Screw_2_C", "Build_SmelterMk1_C", "Schematic_3-4_C"],
)
def test_real_entity_ids_are_accepted(entity: str) -> None:
    assert parse(QUANTITATIVE, expected_entities=[entity]).expected_entities == (entity,)


def test_rates_may_be_fractional() -> None:
    question = parse(QUANTITATIVE, expected_numbers=[number(value=12.5, unit="per_minute")])

    assert question.expected_numbers[0].value == 12.5


# --- Rules by question type --------------------------------------------------


def test_quantitative_question_without_numbers_fails_with_a_clear_message() -> None:
    with pytest.raises(ValidationError, match="Q001: quantitative requires expected_numbers"):
        parse(QUANTITATIVE, expected_numbers=[])


@pytest.mark.parametrize("notes", ["", "   "])
def test_quantitative_question_requires_the_calculation_in_notes(notes: str) -> None:
    with pytest.raises(ValidationError, match="Q001: quantitative requires notes"):
        parse(QUANTITATIVE, notes=notes)


@pytest.mark.parametrize("question_type", ["conceptual", "lookup", "quantitative", "guide"])
def test_in_scope_question_requires_entities(question_type: str) -> None:
    with pytest.raises(ValidationError, match=f"{question_type} requires expected_entities"):
        parse(QUANTITATIVE, question_type=question_type, expected_entities=[])


@pytest.mark.parametrize("question_type", ["conceptual", "lookup", "quantitative", "guide"])
def test_in_scope_question_requires_tier(question_type: str) -> None:
    with pytest.raises(ValidationError, match=f"{question_type} requires tier"):
        parse(QUANTITATIVE, question_type=question_type, tier=None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_entities", ["Desc_Rotor_C"]),
        ("expected_numbers", [number()]),
        ("tier", 0),
    ],
)
def test_out_of_scope_question_rejects_expected_data(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match=f"Q002: out_of_scope forbids {field}"):
        parse(OUT_OF_SCOPE, **{field: value})


def test_all_broken_rules_are_reported_together() -> None:
    with pytest.raises(ValidationError) as error:
        parse(QUANTITATIVE, expected_entities=[], expected_numbers=[], notes="", tier=None)

    for field in ("expected_entities", "expected_numbers", "notes", "tier"):
        assert f"requires {field}" in str(error.value)


# --- Field format ------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("id", "Q1", id="short-id"),
        pytest.param("id", "quantitative-001", id="id-with-type"),
        pytest.param("question", "  ", id="blank-question"),
        pytest.param("expected_answer", "", id="empty-answer"),
        pytest.param("question_type", "math", id="unknown-type"),
        pytest.param("difficulty", "extreme", id="unknown-difficulty"),
        pytest.param("tier", -1, id="tier-below-range"),
        pytest.param("tier", 5, id="tier-above-promised-scope"),
        pytest.param("tier", "0", id="tier-as-text"),
        pytest.param("expected_entities", ["Rotor"], id="display-name-as-entity"),
        pytest.param("expected_entities", ["Desc_Rotor"], id="entity-without-suffix"),
        pytest.param("expected_entities", ["Desc_Rotor_C\n"], id="entity-with-newline"),
        pytest.param("expected_entities", ["Desc_Rotor_C"] * 2, id="duplicated-entity"),
        pytest.param("expected_numbers", [number(value=0)], id="zero"),
        pytest.param("expected_numbers", [number(value=-3)], id="negative"),
        pytest.param("expected_numbers", [number(value="2")], id="number-as-text"),
        pytest.param("expected_numbers", [number(value=float("nan"))], id="nan"),
        pytest.param("expected_numbers", [number(value=float("inf"), unit="per_minute")], id="inf"),
        pytest.param("expected_numbers", [number(value=2.5)], id="fractional-machines"),
        pytest.param("expected_numbers", [number(unit="gallons")], id="unknown-unit"),
        pytest.param("expected_numbers", [number(entity="Constructor")], id="bad-number-entity"),
        pytest.param("expected_numbers", [2], id="number-without-unit"),
        pytest.param("expected_number", [number()], id="misspelled-field"),
    ],
)
def test_invalid_field_is_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError) as error:
        parse(QUANTITATIVE, **{field: value})

    # The error must point at the offending field, not just fail for any reason
    assert {e["loc"][0] for e in error.value.errors()} == {field}


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "question",
        "question_type",
        "expected_answer",
        "expected_entities",
        "tier",
        "difficulty",
    ],
)
def test_missing_field_is_rejected(field: str) -> None:
    line = json.dumps({k: v for k, v in QUANTITATIVE.items() if k != field})

    with pytest.raises(ValidationError) as error:
        EvalQuestion.model_validate_json(line)

    assert [(e["type"], e["loc"]) for e in error.value.errors()] == [("missing", (field,))]


def test_questions_are_immutable() -> None:
    question = parse(QUANTITATIVE)

    with pytest.raises(ValidationError, match="frozen"):
        question.tier = 1
