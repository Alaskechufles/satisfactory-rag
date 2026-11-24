"""Loading of the evaluation dataset (JSONL, one question per line)."""

import hashlib
from pathlib import Path

from pydantic import ValidationError

from eval.schema import EvalQuestion, QuestionType

TARGET_COUNTS = {
    QuestionType.CONCEPTUAL: 9,
    QuestionType.LOOKUP: 11,
    QuestionType.QUANTITATIVE: 9,
    QuestionType.GUIDE: 7,
    QuestionType.OUT_OF_SCOPE: 4,
}


class DatasetError(Exception):
    """The dataset cannot be used. Holds every problem found, not just the first."""

    def __init__(self, *problems: str) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


def load_dataset(path: Path) -> tuple[EvalQuestion, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"cannot read {path}: {error}") from error

    questions: list[EvalQuestion] = []
    first_line: dict[str, int] = {}
    problems: list[str] = []
    # Not splitlines(): it also cuts at separators that are valid inside a JSON string
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate_json(line)
        except ValidationError as error:
            problems += [
                f"line {number}: {_describe(error, i)}" for i in range(error.error_count())
            ]
            continue
        if question.id in first_line:
            problems.append(
                f"line {number}: duplicated id {question.id}, "
                f"first seen on line {first_line[question.id]}"
            )
            continue
        first_line[question.id] = number
        questions.append(question)

    if problems:
        raise DatasetError(*problems)
    return tuple(questions)


def fingerprint(path: Path) -> str:
    """Short hash of the dataset: metrics are only comparable between equal fingerprints."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _describe(error: ValidationError, index: int) -> str:
    detail = error.errors()[index]
    field = ".".join(str(part) for part in detail["loc"])
    return f"{field}: {detail['msg']}" if field else detail["msg"]
