"""Schema of an evaluation question: one line of ``eval/dataset.jsonl``."""

from enum import StrEnum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# Quality is only promised up to this tier (docs/scope.md)
MAX_TIER = 4

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Format only: CI has no game data to check that the entity exists.
# Hyphens are valid: every milestone has one (Schematic_3-4_C).
EntityId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z]+_[A-Za-z0-9_-]+_C$")]


class QuestionType(StrEnum):
    CONCEPTUAL = "conceptual"
    LOOKUP = "lookup"
    QUANTITATIVE = "quantitative"
    GUIDE = "guide"
    OUT_OF_SCOPE = "out_of_scope"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Unit(StrEnum):
    """Says how a number is compared: machines exactly, rates with a tolerance."""

    MACHINES = "machines"
    PER_MINUTE = "per_minute"
    AMOUNT = "amount"
    SECONDS = "seconds"


class _DatasetModel(BaseModel):
    # forbid: a misspelled field must fail instead of being silently ignored
    # strict: "2" is not a number
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExpectedNumber(_DatasetModel):
    value: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    unit: Unit
    entity: EntityId

    @model_validator(mode="after")
    def _machines_are_whole(self) -> Self:
        if self.unit is Unit.MACHINES and not self.value.is_integer():
            raise ValueError(f"machines must be a whole number, got {self.value}")
        return self


class EvalQuestion(_DatasetModel):
    """A question with its verified answer.

    ``expected_entities`` holds the minimum set of documents without which the question
    cannot be answered: "what do I need for a Rotor?" expects ``Recipe_Rotor_C`` only.
    """

    # The id does not encode the type, so reclassifying a question keeps its history
    id: Annotated[str, StringConstraints(pattern=r"^Q\d{3}$")]
    question: NonBlankStr
    question_type: QuestionType
    expected_answer: NonBlankStr
    expected_entities: tuple[EntityId, ...]
    expected_numbers: tuple[ExpectedNumber, ...] = ()
    tier: Annotated[int, Field(ge=0, le=MAX_TIER)] | None
    difficulty: Difficulty
    notes: Annotated[str, StringConstraints(strip_whitespace=True)] = ""

    @field_validator("expected_entities")
    @classmethod
    def _entities_are_unique(cls, entities: tuple[str, ...]) -> tuple[str, ...]:
        # A repeated entity would count twice in recall@k
        if len(set(entities)) != len(entities):
            raise ValueError("expected_entities has duplicates")
        return entities

    @model_validator(mode="after")
    def _fields_match_question_type(self) -> Self:
        present = {
            "expected_entities": bool(self.expected_entities),
            "expected_numbers": bool(self.expected_numbers),
            "tier": self.tier is not None,
            "notes": bool(self.notes),
        }
        required: set[str] = set()
        forbidden: set[str] = set()
        if self.question_type is QuestionType.OUT_OF_SCOPE:
            forbidden = {"expected_entities", "expected_numbers", "tier"}
        else:
            required = {"expected_entities", "tier"}
            if self.question_type is QuestionType.QUANTITATIVE:
                required |= {"expected_numbers", "notes"}

        # Report every broken rule at once: fixing a dataset one error at a time is slow
        problems = [f"requires {f}" for f in sorted(required) if not present[f]]
        problems += [f"forbids {f}" for f in sorted(forbidden) if present[f]]
        if problems:
            raise ValueError("; ".join(f"{self.id}: {self.question_type} {p}" for p in problems))
        return self
