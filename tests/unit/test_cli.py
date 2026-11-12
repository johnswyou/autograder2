import functools
import logging
import shutil
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas
from rich.text import Text
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Models
from redpen.llm.models import Catalog, recheck_providers
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES

runner = CliRunner()


def first_column(help_text: str) -> set[str]:
    """Names in the first column of the help's option and command tables."""
    # Typer forces colour when GITHUB_ACTIONS or FORCE_COLOR is set.
    plain = Text.from_ansi(help_text).plain
    return {
        line[2:].split()[0]
        for line in plain.splitlines()
        if line.startswith("│ ") and line[2] != " "
    }


@pytest.mark.parametrize(
    ("group", "commands"),
    [
        (
            [],
            {"run", "prepare", "key", "grade", "regrade", "report", "review"}
            | {"override", "eval", "models"},
        ),
        (["key"], {"check", "render"}),
        (["models"], {"check"}),
    ],
)
def test_help_lists_every_command(group: list[str], commands: set[str]) -> None:
    result = runner.invoke(app, [*group, "--help"])

    assert result.exit_code == 0
    assert commands <= first_column(result.output)


run_options = [
    "--total", "50", "--page-points", "10", "--profile", "thorough",
    "--roster", "roster.csv", "--dry-run", "--workspace", "ws",
]  # fmt: skip


@pytest.mark.parametrize(
    "args",
    [
        ["run", "blank.pdf", "submissions", *run_options],
        ["grade", "submissions"],
        ["regrade", "--part", "Q7.b", "--student", "s1"],
        ["report"],
        ["review"],
        ["override", "s1", "Q7.b-1", "2"],
        ["eval", "--gold", "gold.csv"],
    ],
)
def test_stub_accepts_its_arguments_and_reports_not_implemented(
    args: list[str],
) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert result.stderr.strip() == "not implemented yet"


@pytest.mark.parametrize(
    ("args", "flag"),
    [
        (["--total", "50"], "--total"),
        (["--page-points", "10"], "--page-points"),
        (["--dry-run"], "--dry-run"),
    ],
)
def test_prepare_refuses_what_it_does_not_support_yet(
    tmp_path: Path, args: list[str], flag: str
) -> None:
    result = runner.invoke(
        app, ["prepare", "blank.pdf", "--workspace", str(tmp_path / "ws"), *args]
    )

    assert result.exit_code == 1
    assert result.stderr.strip() == f"{flag} is not supported yet"
    assert not (tmp_path / "ws").exists()


def test_prepare_refuses_a_workspace_prepared_from_another_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    first = (FIXTURES / "blank_classkick.pdf").read_bytes()
    workspace.blank_pdf.write_bytes(first)
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == 1
    assert "holds the key of another blank" in result.stderr
    assert workspace.blank_pdf.read_bytes() == first


def write_unreadable(path: Path, kind: str) -> None:
    if kind == "not a PDF":
        path.write_bytes(b"Questions 1 to 5, typed in a text file.")
    elif kind == "encrypted":
        pdf = canvas.Canvas(str(path), encrypt="secret")
        pdf.drawString(72, 720, "1. Find x.")
        pdf.save()
    else:
        pdfium.PdfDocument.new().save(path)


@pytest.mark.parametrize("kind", ["not a PDF", "encrypted", "no pages"])
def test_prepare_refuses_a_file_it_cannot_open_and_leaves_no_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    write_unreadable(Path("blank.pdf"), kind)

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == 1
    assert len(result.stderr.strip().splitlines()) == 1
    assert result.stderr.startswith("blank.pdf ")
    assert not Path("ws").exists()


def reused_spec(home_of_q3: int) -> Spec:
    return Spec(
        parts=[
            Part(
                id=id,
                label=id[1:],
                prompt="Find x.",
                response_type="numeric",
                home_page=home,
                band=[0.1, 0.2],
                figures=[],
                depends_on=[],
                options=[],
            )
            for id, home in (("Q1", 1), ("Q3", home_of_q3))
        ]
    )


@pytest.mark.parametrize(
    ("home_of_q3", "exit_code", "message"),
    [
        (9, 1, "fails its checks: Q3 has home page 9 of 4"),
        (2, 0, "Warning: no problem numbered 2 was found."),
    ],
    ids=["failing", "gap"],
)
def test_prepare_checks_a_spec_it_reuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    home_of_q3: int,
    exit_code: int,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy("blank.pdf", workspace.blank_pdf)
    workspace.config_lock_toml.write_text(LOCK)
    write_model(workspace.spec_json, reused_spec(home_of_q3))
    workspace.key_yaml.write_text("")  # so that prepare stops after the spec

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == exit_code
    assert message in result.stderr


def refuse(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("prepare reached for OpenRouter")


def test_prepare_leaves_an_existing_key_as_it_is_and_calls_no_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.config.fetch_catalog", refuse)
    monkeypatch.setattr("redpen.cli.Gateway", refuse)
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy("blank.pdf", workspace.blank_pdf)
    workspace.config_lock_toml.write_text(LOCK)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    edited = b"# Edited by hand.\nparts: []\n"
    workspace.key_yaml.write_bytes(edited)

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == 0, result.output
    assert "ws/key/key.yaml exists, so it is left as it is." in result.output
    assert workspace.key_yaml.read_bytes() == edited
    assert not workspace.solutions_md.exists()


@pytest.mark.parametrize(
    ("flags", "level"),
    [([], logging.INFO), (["-v"], logging.DEBUG), (["--verbose"], logging.DEBUG)],
)
def test_verbose_flag_sets_log_level(flags: list[str], level: int) -> None:
    runner.invoke(app, [*flags, "report"])

    assert logging.getLogger("redpen").level == level


def vision(slug: str, inputs: list[str]) -> dict[str, Any]:
    return {
        "id": slug,
        "architecture": {"input_modalities": inputs},
        "supported_parameters": ["structured_outputs"],
        "pricing": {"prompt": "0", "completion": "0"},
    }


CATALOG = Catalog(
    {
        "vendor/vision-1": vision("vendor/vision-1", ["text", "image"]),
        "vendor/text-1": vision("vendor/text-1", ["text"]),
    },
    [
        {
            "model_id": slug,
            "tag": "zero",
            "safety_retention": False,
            "supported_parameters": ["structured_outputs"],
        }
        for slug in ("vendor/vision-1", "vendor/text-1")
    ],
)


def configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str) -> None:
    """A redpen.toml with every role on vendor/vision-1 but the reader, and
    OpenRouter listing CATALOG."""

    async def fetch_catalog(api_key: str) -> Catalog:
        return CATALOG

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("redpen.cli.fetch_catalog", fetch_catalog)
    roles = dict.fromkeys(Models.model_fields, "vendor/vision-1") | {"reader": reader}
    toml = "".join(f'{role} = "{name}"\n' for role, name in roles.items())
    Path("redpen.toml").write_text("[models]\n" + toml)


@pytest.mark.parametrize(
    ("reader", "exit_code", "problem"),
    [
        ("vendor/vision-1", 0, None),
        ("vendor/text-1", 1, "reader: vendor/text-1 does not accept image input"),
    ],
)
def test_models_check_prints_a_table_and_fails_on_a_problem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: str,
    exit_code: int,
    problem: str | None,
) -> None:
    configure(tmp_path, monkeypatch, reader)

    result = runner.invoke(app, ["models", "check"])

    assert result.exit_code == exit_code
    assert "Role" in result.output and "zero" in result.output
    output = result.output.splitlines()
    assert [line for line in output if line.startswith("reader:")] == (
        [problem] if problem else []
    )


LOCK = """profile = "balanced"

[models.reader]
slug = "vendor/vision-1"
supported_parameters = ["structured_outputs"]
accepts_images = true
prompt_price = "0"
completion_price = "0"
providers = ["zero"]
"""
RETAINING_BELOW = {
    "model_id": "vendor/vision-1",
    "tag": "zero/eu",
    "safety_retention": True,
    "supported_parameters": ["structured_outputs"],
}


@pytest.mark.parametrize(
    ("zdr", "exit_code", "problem"),
    [
        (CATALOG.zdr_endpoints, 0, None),
        (
            [*CATALOG.zdr_endpoints, RETAINING_BELOW],
            1,
            "reader: vendor/vision-1 pinned endpoint zero now admits zero/eu,"
            " which may retain data",
        ),
    ],
    ids=["unchanged", "retaining below"],
)
def test_models_check_reverifies_the_lock_in_a_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    zdr: Any,
    exit_code: int,
    problem: str | None,
) -> None:
    configure(tmp_path, monkeypatch, "vendor/vision-1")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": zdr})
    )
    monkeypatch.setattr(
        "redpen.cli.recheck_providers",
        functools.partial(recheck_providers, transport=transport),
    )
    lock = Workspace(tmp_path / "ws").config_lock_toml
    lock.parent.mkdir()
    lock.write_text(LOCK)

    result = runner.invoke(app, ["models", "check", "--workspace", str(lock.parent)])

    assert result.exit_code == exit_code
    assert [line for line in result.output.splitlines() if "pinned" in line] == (
        [f"{lock}: {problem}"] if problem else []
    )
