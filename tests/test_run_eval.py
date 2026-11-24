"""Evaluation runner: scoring, report, log and command line."""

import subprocess
from pathlib import Path

import pytest

from eval import run_eval
from eval.dataset import load_dataset
from eval.run_eval import GROUPS, GroupReport, evaluate, main, meets_threshold
from tests.fakes import SAMPLE_ANSWERS, FixedRetriever

SAMPLE = Path(__file__).parent / "fixtures" / "sample_dataset.jsonl"
RETRIEVER = "tests.fakes:sample_retriever"


def reports(k: int = 5) -> dict[str, GroupReport]:
    results = evaluate(load_dataset(SAMPLE), FixedRetriever(SAMPLE_ANSWERS), k)
    return {r.group.name: r for r in results}


def cli(*args: object) -> int:
    return main(["--dataset", str(SAMPLE), *map(str, args)])


def usage_error(capsys: pytest.CaptureFixture[str], *args: object) -> str:
    with pytest.raises(SystemExit) as error:
        cli(*args)
    assert error.value.code == 2
    return capsys.readouterr().err


# --- Scoring (expected values worked out in tests/fakes.py) --------------------


def test_metrics_match_the_values_worked_out_by_hand() -> None:
    main_group, guide = reports().values()

    assert (main_group.count, main_group.recall, main_group.mrr) == (2, 0.75, 0.75)
    assert (guide.count, guide.recall, guide.mrr) == (1, 1.0, pytest.approx(1 / 3))


def test_questions_with_incomplete_recall_are_listed() -> None:
    main_group, guide = reports().values()

    assert (main_group.incomplete, guide.incomplete) == (("Q001",), ())


def test_k_reaches_the_metric() -> None:
    main_group, _ = reports(k=6).values()

    assert main_group.recall == 1.0


def test_only_retrieval_questions_are_asked() -> None:
    retriever = FixedRetriever(SAMPLE_ANSWERS)
    evaluate(load_dataset(SAMPLE), retriever, 7)

    assert retriever.calls == [(question, 7) for question in SAMPLE_ANSWERS]


def test_thresholds_are_the_ones_in_the_scope_document() -> None:
    assert [(g.name, g.threshold) for g in GROUPS] == [
        ("Conceptual + consulta", 0.85),
        ("Guía", 0.70),
    ]


@pytest.mark.parametrize(
    ("value", "met"), [(0.85, True), (0.85 - 1e-12, True), (0.849, False), (1.0, True)]
)
def test_threshold_tolerates_float_noise_only(value: float, met: bool) -> None:
    assert meets_threshold(value, 0.85) is met


# --- Dry run -------------------------------------------------------------------


def test_dry_run_counts_questions_by_type(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli("--dry-run") == 0

    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows == [
        ["conceptual", "1/9"],
        ["lookup", "1/11"],
        ["quantitative", "1/9"],
        ["guide", "1/7"],
        ["out_of_scope", "1/4"],
        ["total", "5/40"],
    ]


def test_dry_run_neither_asks_the_retriever_nor_writes_the_log(tmp_path: Path) -> None:
    log = tmp_path / "log.md"

    assert cli("--dry-run", "--retriever", "missing:nothing", "--note", "x", "--log", log) == 0
    assert not log.exists()


def test_invalid_dataset_fails_showing_every_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("{bad\n{worse\n", encoding="utf-8")

    assert main(["--dataset", str(path), "--dry-run"]) == 1
    err = capsys.readouterr().err
    assert "line 1:" in err
    assert "line 2:" in err


# --- Full run ------------------------------------------------------------------


def test_finished_run_adds_an_entry_to_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_eval, "describe_commit", lambda: "abc1234")
    log = tmp_path / "log.md"

    assert cli("--retriever", RETRIEVER, "--note", "línea  base\nsimulada", "--log", log) == 0

    text = log.read_text(encoding="utf-8")
    assert text.startswith("# Registro de evaluación\n")
    assert "UTC · línea base simulada\n" in text
    assert "- **Commit:** abc1234\n" in text
    assert f"`{RETRIEVER}`, k=5" in text
    assert "5 preguntas, huella `" in text
    assert "| Conceptual + consulta | 2 | 0.750 | 0.750 | ≥ 0.85 | no | Q001 |" in text
    assert "| Guía | 1 | 1.000 | 0.333 | ≥ 0.70 | sí | - |" in text
    assert "| Guía | 1 | 1.000 | 0.333 |" in capsys.readouterr().out


def test_log_is_append_only_with_a_single_title(tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    cli("--retriever", RETRIEVER, "--note", "primera", "--log", log)
    before = log.read_text(encoding="utf-8")
    cli("--retriever", RETRIEVER, "--note", "segunda", "--log", log)
    after = log.read_text(encoding="utf-8")

    assert after.startswith(before)
    assert after.count("# Registro de evaluación\n") == 1
    assert after.count("\n## ") == 2


def test_no_log_prints_the_report_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "log.md"

    assert cli("--retriever", RETRIEVER, "--no-log", "--log", log) == 0
    assert not log.exists()
    assert "| Conceptual + consulta | 2 | 0.750 |" in capsys.readouterr().out


def test_thresholds_only_apply_to_recall_at_5(capsys: pytest.CaptureFixture[str]) -> None:
    cli("--retriever", RETRIEVER, "--no-log", "--k", "3")

    assert "| Guía | 1 | 1.000 | 0.333 | - | - | - |" in capsys.readouterr().out


def test_group_without_questions_is_not_a_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(SAMPLE.read_text(encoding="utf-8").split("\n")[1], encoding="utf-8")

    assert main(["--dataset", str(path), "--retriever", RETRIEVER, "--no-log"]) == 0
    assert "| Guía | 0 | - | - | ≥ 0.70 | - | - |" in capsys.readouterr().out


def test_dataset_without_retrieval_questions_is_not_logged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, log = tmp_path / "d.jsonl", tmp_path / "log.md"
    path.write_text(SAMPLE.read_text(encoding="utf-8").split("\n")[4], encoding="utf-8")

    args = ["--dataset", str(path), "--retriever", RETRIEVER, "--note", "x", "--log", str(log)]
    assert main(args) == 1
    assert not log.exists()
    assert "no retrieval questions" in capsys.readouterr().err


# --- Commit in the log ----------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean git repository with one commit, used as the project root."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "eval-log.md").write_text("log\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    for args in (["init", "-q"], ["add", "."], [*identity, "commit", "-q", "-m", "first"]):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True)
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)
    return tmp_path


def head(repo: Path) -> str:
    command = ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def test_clean_repository_reports_its_commit(repo: Path) -> None:
    assert run_eval.describe_commit() == head(repo)


def test_uncommitted_changes_are_flagged(repo: Path) -> None:
    (repo / "code.py").write_text("x = 2\n")

    assert run_eval.describe_commit() == f"{head(repo)} (con cambios sin commit)"


def test_a_previous_log_entry_is_not_an_uncommitted_change(repo: Path) -> None:
    (repo / "docs" / "eval-log.md").write_text("log\nentry\n")

    assert run_eval.describe_commit() == head(repo)


def test_outside_git_the_commit_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_eval, "ROOT", tmp_path)

    assert run_eval.describe_commit() == "desconocido"


# --- Command line --------------------------------------------------------------


def test_run_requires_a_retriever(capsys: pytest.CaptureFixture[str]) -> None:
    assert "--retriever" in usage_error(capsys, "--note", "x")


@pytest.mark.parametrize("note", [[], ["--note", "  "]])
def test_logging_requires_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], note: list[str]
) -> None:
    log = tmp_path / "log.md"

    assert "--note" in usage_error(capsys, "--retriever", RETRIEVER, "--log", log, *note)
    assert not log.exists()


@pytest.mark.parametrize("k", ["0", "-1", "five"])
def test_k_must_be_positive(capsys: pytest.CaptureFixture[str], k: str) -> None:
    assert "--k" in usage_error(capsys, "--retriever", RETRIEVER, "--no-log", "--k", k)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("tests.fakes", "module:object"),
        ("missing_module:thing", "missing_module"),
        ("tests.fakes:missing", "missing"),
        ("tests.fakes:SAMPLE_ANSWERS", "retrieve"),
    ],
)
def test_bad_retriever_is_explained(
    capsys: pytest.CaptureFixture[str], spec: str, message: str
) -> None:
    assert message in usage_error(capsys, "--retriever", spec, "--no-log")


def test_a_retriever_class_is_instantiated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "class_retriever.py").write_text(
        "class Empty:\n    def retrieve(self, question, k):\n        return []\n"
    )
    monkeypatch.syspath_prepend(tmp_path)

    assert cli("--retriever", "class_retriever:Empty", "--no-log") == 0
    assert "| Guía | 1 | 0.000 | 0.000 |" in capsys.readouterr().out
