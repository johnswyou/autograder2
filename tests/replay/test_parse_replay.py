"""Parsing the fixture blanks with recorded calls to the real parser model.

Re-record (paid) with `REDPEN_RECORD=1 uv run pytest tests/replay/test_parse_replay.py`.
The `prepare` test always replays the recording of `blank.pdf`, so it spends nothing.
"""

import asyncio
import json
import shutil
import sqlite3
from collections.abc import Callable
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pytest
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Config, Models, read_lock
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway
from redpen.llm.models import Catalog, ModelInfo
from redpen.prepare.key import load_key
from redpen.prepare.parse import parse
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES

# ~anthropic/claude-sonnet-latest as `pin` resolved it with strict privacy on
# 12 November 2026.
MODEL = ModelInfo(
    slug="anthropic/claude-sonnet-5.5",
    supported_parameters=frozenset(
        {
            "include_reasoning",
            "max_completion_tokens",
            "max_tokens",
            "reasoning",
            "reasoning_effort",
            "response_format",
            "stop",
            "structured_outputs",
            "tool_choice",
            "tools",
            "verbosity",
        }
    ),
    accepts_images=True,
    prompt_price=Decimal("0.000002"),
    completion_price=Decimal("0.00001"),
    providers=frozenset(
        {"google-vertex/europe", "google-vertex/global", "google-vertex/us"}
    ),
)
TRUTH: list[dict[str, Any]] = json.loads((FIXTURES / "key.json").read_text())["parts"]


def assert_matches_the_truth(spec: Spec) -> None:
    assert [p.id for p in spec.parts] == [t["id"] for t in TRUTH]
    for found, truth in zip(spec.parts, TRUTH, strict=True):
        assert (found.id, found.label, found.response_type, found.home_page) == (
            truth["id"],
            truth["label"],
            truth["response_type"],
            truth["home_page"],
        )
        if found.response_type == "multiple_choice":
            assert found.options == ["A", "B", "C", "D", "E"]
    # The part on page 3 under question 4 is part of question 4, introduction and all.
    [continued] = [p for p in spec.parts if p.id.startswith("Q4") and p.home_page == 3]
    assert continued.id == "Q4.c"
    assert "4.0 kg" in continued.prompt


@pytest.mark.parametrize("blank", ["blank.pdf", "blank_classkick.pdf"])
def test_the_fixture_blank_parses_into_the_parts_of_its_key(
    replay: Callable[[str], Gateway], tmp_path: Path, blank: str
) -> None:
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / blank, workspace.blank_pdf)

    async def main() -> Spec:
        async with replay(f"parse_{blank.removesuffix('.pdf')}") as gateway:
            return await parse(workspace, gateway, MODEL, Config().image_budget)

    spec = asyncio.run(main())

    assert_matches_the_truth(spec)
    assert read_model(workspace.spec_json, Spec) == spec


def catalog() -> Catalog:
    """OpenRouter listing MODEL behind every default role."""
    entry = {
        "id": MODEL.slug,
        "architecture": {"input_modalities": ["text", "image"]},
        "supported_parameters": sorted(MODEL.supported_parameters),
        "pricing": {
            "prompt": str(MODEL.prompt_price),
            "completion": str(MODEL.completion_price),
        },
    }
    aliases = {
        name: {"id": name, "alias_target": {"slug": MODEL.slug}}
        for name in Models().model_dump().values()
    }
    endpoints = [
        {
            "model_id": MODEL.slug,
            "tag": tag,
            "safety_retention": False,
            "supported_parameters": ["structured_outputs"],
        }
        for tag in MODEL.providers
    ]
    return Catalog({**aliases, MODEL.slug: entry}, endpoints)


def recorded(db_path: Path, api_key: str, **options: Any) -> Gateway:
    """The gateway `prepare` makes, answering offline from the blank.pdf recording."""
    with closing(sqlite3.connect(db_path, autocommit=True)) as connection:
        CallCache(connection).load(
            Path(__file__).parents[1] / "recordings" / "parse_blank.jsonl"
        )
    return Gateway(
        db_path, api_key, offline=True, transport=httpx.MockTransport(refuse)
    )


def refuse(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("prepare reached for OpenRouter")


TRUTH_KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())


async def truth_key(*args: object) -> Key:
    """Stands in for the solve and rubric stages, whose replay tests start from the
    fixtures' spec rather than from this parse."""
    return TRUTH_KEY


def test_prepare_writes_the_key_files_and_a_second_run_calls_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fetch_catalog(api_key: str, **options: object) -> Catalog:
        return catalog()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.config.fetch_catalog", fetch_catalog)
    monkeypatch.setattr("redpen.cli.Gateway", recorded)
    monkeypatch.setattr("redpen.cli._key", truth_key)
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    runner = CliRunner()

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == 0, result.output
    workspace = Workspace(Path("ws"))
    assert workspace.blank_pdf.read_bytes() == Path("blank.pdf").read_bytes()
    assert all(workspace.key_page(n).exists() for n in (1, 2, 3, 4))
    assert not workspace.key_page(5).exists()
    assert read_lock(workspace.config_lock_toml).models["parser"] == MODEL
    spec = read_model(workspace.spec_json, Spec)
    assert_matches_the_truth(spec)
    assert "Pages: 4\nProblems: 5\nParts: 9\n" in result.output
    assert load_key(workspace.key_yaml) == TRUTH_KEY
    assert workspace.solutions_md.exists() and workspace.rubric_md.exists()
    assert "Total points: 16\n  page 1: 3\n" in result.output
    written = workspace.key_yaml.read_bytes()

    monkeypatch.setattr("redpen.config.fetch_catalog", refuse)
    monkeypatch.setattr("redpen.cli.Gateway", refuse)
    again = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert again.exit_code == 0, again.output
    assert again.output.startswith(f"Reusing {workspace.spec_json}\n")
    assert read_model(workspace.spec_json, Spec) == spec
    assert workspace.key_yaml.read_bytes() == written


def test_prepare_after_a_file_it_cannot_open_works_in_the_same_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fetch_catalog(api_key: str, **options: object) -> Catalog:
        return catalog()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.config.fetch_catalog", fetch_catalog)
    monkeypatch.setattr("redpen.cli.Gateway", recorded)
    monkeypatch.setattr("redpen.cli._key", truth_key)
    Path("blank.pdf").write_bytes(b"Questions 1 to 5, typed in a text file.")
    runner = CliRunner()

    refused = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert refused.exit_code == 1
    assert result.exit_code == 0, result.output
    assert_matches_the_truth(read_model(Workspace(Path("ws")).spec_json, Spec))
