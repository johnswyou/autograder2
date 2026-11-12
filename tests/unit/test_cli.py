import logging

import pytest
from rich.text import Text
from typer.testing import CliRunner

from redpen.cli import app

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
        ["prepare", "blank.pdf", *run_options],
        ["key", "check"],
        ["key", "render"],
        ["grade", "submissions"],
        ["regrade", "--part", "Q7.b", "--student", "s1"],
        ["report"],
        ["review"],
        ["override", "s1", "Q7.b-1", "2"],
        ["eval", "--gold", "gold.csv"],
        ["models", "check"],
    ],
)
def test_stub_accepts_its_arguments_and_reports_not_implemented(
    args: list[str],
) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert result.stderr.strip() == "not implemented yet"


@pytest.mark.parametrize(
    ("flags", "level"),
    [([], logging.INFO), (["-v"], logging.DEBUG), (["--verbose"], logging.DEBUG)],
)
def test_verbose_flag_sets_log_level(flags: list[str], level: int) -> None:
    runner.invoke(app, [*flags, "report"])

    assert logging.getLogger("redpen").level == level
