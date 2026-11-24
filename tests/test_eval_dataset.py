"""Loading of the evaluation dataset."""

import json
from pathlib import Path

import pytest

from eval.dataset import DatasetError, fingerprint, load_dataset

SAMPLE = Path(__file__).parent / "fixtures" / "sample_dataset.jsonl"
LINES = SAMPLE.read_text(encoding="utf-8").strip().split("\n")


def write(path: Path, *lines: str) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_every_question_in_order() -> None:
    assert [q.id for q in load_dataset(SAMPLE)] == ["Q001", "Q002", "Q003", "Q004", "Q005"]


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    assert len(load_dataset(write(tmp_path / "d.jsonl", LINES[0], "", "  ", LINES[1]))) == 2


def test_unicode_line_separator_does_not_split_a_question(tmp_path: Path) -> None:
    line = json.dumps({**json.loads(LINES[0]), "notes": "a\u2028b"}, ensure_ascii=False)

    assert load_dataset(write(tmp_path / "d.jsonl", line))[0].notes == "a\u2028b"


def test_every_problem_is_reported_with_its_line_number(tmp_path: Path) -> None:
    no_numbers = json.dumps({**json.loads(LINES[3]), "expected_numbers": []})
    bad_tier = json.dumps({**json.loads(LINES[1]), "tier": 9})
    path = write(tmp_path / "d.jsonl", LINES[0], "", "{not json", no_numbers, bad_tier)

    with pytest.raises(DatasetError) as error:
        load_dataset(path)

    problems = error.value.problems
    assert [p.split(":")[0] for p in problems] == ["line 3", "line 4", "line 5"]
    assert "Invalid JSON" in problems[0]
    assert "Q004: quantitative requires expected_numbers" in problems[1]
    assert "tier" in problems[2]


def test_duplicated_id_is_rejected_pointing_at_both_lines(tmp_path: Path) -> None:
    path = write(tmp_path / "d.jsonl", LINES[0], LINES[1], LINES[0])

    with pytest.raises(DatasetError) as error:
        load_dataset(path)

    assert error.value.problems == ("line 3: duplicated id Q001, first seen on line 1",)


def test_missing_file_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="cannot read"):
        load_dataset(tmp_path / "missing.jsonl")


def test_wrong_encoding_is_a_dataset_error(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_bytes(LINES[0].encode("utf-16"))

    with pytest.raises(DatasetError, match="cannot read"):
        load_dataset(path)


def test_fingerprint_changes_with_the_content(tmp_path: Path) -> None:
    one = write(tmp_path / "one.jsonl", LINES[0])
    same = write(tmp_path / "same.jsonl", LINES[0])
    other = write(tmp_path / "other.jsonl", LINES[1])

    assert fingerprint(one) == fingerprint(same) != fingerprint(other)
