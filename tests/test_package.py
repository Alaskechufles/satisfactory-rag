"""Smoke tests: the project layout is importable and typed."""

import importlib
from importlib.resources import files

import pytest

MODULES = [
    "satisrag",
    "satisrag.agent",
    "satisrag.api",
    "satisrag.documents",
    "satisrag.indexing",
    "satisrag.ingestion",
    "satisrag.llm",
    "satisrag.retrieval",
    "satisrag.tools",
    "eval",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_is_importable(name: str) -> None:
    assert importlib.import_module(name).__doc__


def test_package_ships_type_information() -> None:
    assert files("satisrag").joinpath("py.typed").is_file()
