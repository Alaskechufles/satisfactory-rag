"""Evaluation runner.

uv run python -m eval.run_eval --dry-run
uv run python -m eval.run_eval --retriever module:object --note "what changed"
"""

import argparse
import importlib
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import fsum
from pathlib import Path
from typing import Protocol, runtime_checkable

from eval.dataset import TARGET_COUNTS, DatasetError, fingerprint, load_dataset
from eval.metrics import recall_at_k, reciprocal_rank
from eval.schema import EvalQuestion, QuestionType

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval" / "dataset.jsonl"
DEFAULT_LOG = ROOT / "docs" / "eval-log.md"
LOG_TITLE = "# Registro de evaluación\n"

# docs/scope.md sets its thresholds for recall@5
THRESHOLD_K = 5


@runtime_checkable
class Retriever(Protocol):
    """Anything with this method can be evaluated; it does not need to import eval."""

    def retrieve(self, question: str, k: int) -> list[str]:
        """Return up to k entity ids, best first."""
        ...


@dataclass(frozen=True)
class Group:
    name: str
    types: frozenset[QuestionType]
    threshold: float


# Quantitative questions are answered by the calculator, not by retrieval
GROUPS = (
    Group(
        "Conceptual + consulta",
        frozenset({QuestionType.CONCEPTUAL, QuestionType.LOOKUP}),
        0.85,
    ),
    Group("Guía", frozenset({QuestionType.GUIDE}), 0.70),
)


@dataclass(frozen=True)
class GroupReport:
    group: Group
    count: int
    recall: float
    mrr: float
    incomplete: tuple[str, ...]
    """Ids of the questions whose recall is below 1."""


def evaluate(
    questions: Sequence[EvalQuestion], retriever: Retriever, k: int
) -> tuple[GroupReport, ...]:
    return tuple(_evaluate_group(group, questions, retriever, k) for group in GROUPS)


def _evaluate_group(
    group: Group, questions: Sequence[EvalQuestion], retriever: Retriever, k: int
) -> GroupReport:
    recalls: dict[str, float] = {}
    ranks: list[float] = []
    for question in questions:
        if question.question_type not in group.types:
            continue
        retrieved = retriever.retrieve(question.question, k)
        recalls[question.id] = recall_at_k(question.expected_entities, retrieved, k)
        ranks.append(reciprocal_rank(question.expected_entities, retrieved, k))
    count = len(recalls)
    return GroupReport(
        group=group,
        count=count,
        recall=fsum(recalls.values()) / count if count else 0.0,
        mrr=fsum(ranks) / count if count else 0.0,
        incomplete=tuple(id_ for id_, recall in recalls.items() if recall < 1),
    )


def meets_threshold(value: float, threshold: float) -> bool:
    # An average such as 0.85 may be stored as 0.8499999999999999
    return value >= threshold - 1e-9


def render_report(reports: Sequence[GroupReport], k: int) -> str:
    rows = [
        f"| Grupo | Preguntas | recall@{k} | MRR@{k} | Umbral | Cumple | Recall incompleto |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        recall = mrr = threshold = met = "-"
        if report.count:
            recall, mrr = f"{report.recall:.3f}", f"{report.mrr:.3f}"
        if k == THRESHOLD_K:
            threshold = f"≥ {report.group.threshold:.2f}"
            if report.count:
                met = "sí" if meets_threshold(report.recall, report.group.threshold) else "no"
        incomplete = ", ".join(report.incomplete) or "-"
        rows.append(
            f"| {report.group.name} | {report.count} | {recall} | {mrr} "
            f"| {threshold} | {met} | {incomplete} |"
        )
    return "\n".join(rows) + "\n"


def render_counts(questions: Sequence[EvalQuestion]) -> str:
    counts = Counter(question.question_type for question in questions)
    rows = [f"{t:<14}{counts[t]}/{target}" for t, target in TARGET_COUNTS.items()]
    rows.append(f"{'total':<14}{len(questions)}/{sum(TARGET_COUNTS.values())}")
    return "\n".join(rows)


def append_to_log(path: Path, entry: str) -> None:
    title = "" if path.exists() else LOG_TITLE
    with path.open("a", encoding="utf-8") as log:
        log.write(f"{title}\n{entry}")


def describe_commit() -> str:
    def git(*args: str) -> str:
        command = ["git", "-C", str(ROOT), *args]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        # The log itself does not count: the previous run has just changed it
        dirty = git("status", "--porcelain", "--", ".", ":!docs/eval-log.md")
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return f"{commit} (con cambios sin commit)" if dirty else commit


def load_retriever(spec: str) -> Retriever:
    """Import "module:object", like uvicorn does. A class is instantiated."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected module:object, got {spec!r}")
    try:
        candidate = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"cannot load {spec!r}: {error}") from error
    if isinstance(candidate, type):
        candidate = candidate()
    if not isinstance(candidate, Retriever):
        raise ValueError(f"{spec!r} has no retrieve(question, k) method")
    return candidate


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise ValueError
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_eval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dry-run", action="store_true", help="validate and count questions only")
    parser.add_argument("--retriever", metavar="MODULE:OBJECT")
    parser.add_argument("--k", type=positive_int, default=THRESHOLD_K)
    parser.add_argument("--note", default="", help="what changed; required to write the log")
    parser.add_argument("--no-log", action="store_true", help="print the report without logging it")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args(argv)
    note = " ".join(args.note.split())

    retriever: Retriever | None = None
    if not args.dry_run:
        # Checked before evaluating: a forgotten note must not waste a slow run
        if not args.retriever:
            parser.error("--retriever is required unless --dry-run is used")
        if not note and not args.no_log:
            parser.error("--note is required to write the log (or use --no-log)")
        try:
            retriever = load_retriever(args.retriever)
        except ValueError as error:
            parser.error(f"--retriever: {error}")

    try:
        questions = load_dataset(args.dataset)
    except DatasetError as error:
        print(error, file=sys.stderr)
        return 1

    if retriever is None:
        print(render_counts(questions))
        return 0

    reports = evaluate(questions, retriever, args.k)
    if not any(report.count for report in reports):
        print("no retrieval questions in the dataset", file=sys.stderr)
        return 1

    table = render_report(reports, args.k)
    print(table, end="")
    if not args.no_log:
        entry = (
            f"## {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · {note}\n\n"
            f"- **Commit:** {describe_commit()}\n"
            f"- **Retriever:** `{args.retriever}`, k={args.k}\n"
            f"- **Dataset:** {len(questions)} preguntas, huella `{fingerprint(args.dataset)}`\n\n"
            f"{table}"
        )
        append_to_log(args.log, entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
