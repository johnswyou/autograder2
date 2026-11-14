"""The README and the user guide held to the code. Every command, option, flag code,
setting and profile they name exists; every command, flag code and setting the code
has is in the guide, each flag with its severity; every message the guide quotes
(a `> ` line) is one the code prints; and the guide's workspace tree is the tree of
the fixture class replayed into a workspace: its blank parsed, its spec corrected
by hand, its key prepared and then replaced by the fixture's own, three students
graded, one review list entry accepted and one award corrected. Everything is
derived from the code or that workspace; the test keeps no list of its own."""

import ast
import asyncio
import csv
import functools
import re
import shlex
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import get_args

import httpx
import pytest
import typer
from typer._click.core import Command  # what typer.main.get_command returns
from typer.core import TyperGroup
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Config, Lock, Models, Profile, write_lock
from redpen.grade.ingest import IngestSummary
from redpen.llm.cache import CallCache
from redpen.llm.client import Gateway
from redpen.llm.models import recheck_providers
from redpen.prepare.key import save_key
from redpen.prepare.parse import parse
from redpen.schemas import key as key_schema
from redpen.schemas.flags import SEVERITY, FlagCode
from redpen.schemas.key import Key
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, truth_of
from tests.pinned import GRADE_MODELS, KEY_MODELS, PARSER
from tests.replay.test_run_replay import CLASSES, openrouter

ROOT = Path(__file__).parents[2]
SRC = ROOT / "src" / "redpen"
RECORDINGS = ROOT / "tests" / "recordings"
README = (ROOT / "README.md").read_text(encoding="utf-8")
GUIDE = (ROOT / "docs" / "USER_GUIDE.md").read_text(encoding="utf-8")
DOCS = {"README.md": README, "docs/USER_GUIDE.md": GUIDE}
COMMAND = typer.main.get_command(app)
assert isinstance(COMMAND, TyperGroup)  # an app of commands
PROFILES = get_args(Profile)
SETTINGS = {*Config.model_fields, *Models.model_fields}
SNAKE = re.compile(r"[a-z]+(?:_[a-z0-9]+)+")


# --- what the documents say ----------------------------------------------------------


def blocks(text: str, kind: str = "") -> list[str]:
    """The fenced code blocks of `text` whose info string starts with `kind`."""
    return re.findall(rf"^```{kind}[^\n]*\n(.*?)^```", text, re.M | re.S)


def spans(text: str) -> list[str]:
    """The inline code spans of `text`, outside its fenced blocks."""
    prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    return re.findall(r"`([^`\n]+)`", prose)


def command_lines(text: str) -> list[list[str]]:
    """Each `redpen` command in a code span or a line of a code block, as words."""
    lines = [line.strip() for b in blocks(text) for line in b.splitlines()]
    return [
        shlex.split(line, comments=True)
        for line in [*lines, *spans(text)]
        if line == "redpen" or line.startswith("redpen ")
    ]


def resolve(words: list[str]) -> tuple[list[str], list[str]]:
    """The command path `words` name and the problems with them: a command or an
    option the code does not have."""
    command: Command = COMMAND
    path: list[str] = []
    problems: list[str] = []
    for word in words[1:]:
        if word.startswith("-"):
            option = word.split("=")[0]
            if option not in options(command):
                problems.append(f"{' '.join(words)}: no option {option}")
        elif isinstance(command, TyperGroup):
            if word not in command.commands:
                problems.append(f"{' '.join(words)}: no command {word}")
                break
            command = command.commands[word]
            path.append(word)
    return path, problems


def options(command: Command) -> set[str]:
    return {"--help"} | {
        o for p in command.params for o in [*p.opts, *p.secondary_opts]
    }


def commands(group: TyperGroup = COMMAND, path: tuple[str, ...] = ()) -> set[str]:
    """Every command the code has, as `redpen` and its words."""
    found: set[str] = set()
    for name, command in group.commands.items():
        if isinstance(command, TyperGroup):
            found |= commands(command, (*path, name))
        else:
            found.add(" ".join(("redpen", *path, name)))
    return found


def every_option(group: TyperGroup = COMMAND) -> set[str]:
    found = options(group)
    for command in group.commands.values():
        found |= (
            every_option(command)
            if isinstance(command, TyperGroup)
            else options(command)
        )
    return found


def quotes(text: str) -> list[str]:
    """Each message quoted in `text`: a run of `> ` lines, joined."""
    return [
        " ".join(line.removeprefix(">").strip() for line in quote.splitlines())
        for quote in re.findall(r"(?:^>.*\n?)+", text, re.M)
    ]


# --- what the code says --------------------------------------------------------------


def messages() -> tuple[list[re.Pattern[str]], set[str]]:
    """Every string the code writes, as a pattern whose gaps are its replacement
    fields, and every plain string it holds."""
    found: list[re.Pattern[str]] = []
    plain: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                plain.add(node.value)
                pieces = re.split(r"(\{[a-z_]*\})", node.value)
                found.append(
                    re.compile(
                        "".join(
                            "(.+?)" if re.fullmatch(r"\{[a-z_]*\}", p) else re.escape(p)
                            for p in pieces
                        ),
                        re.S,
                    )
                )
            elif isinstance(node, ast.JoinedStr):
                pattern = "".join(
                    re.escape(v.value)
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    else "(.+?)"
                    for v in node.values
                )
                found.append(re.compile(pattern, re.S))
    return found, plain


def printed(quote: str, patterns: list[re.Pattern[str]], plain: set[str]) -> bool:
    """Whether the code prints `quote`: a string of it, with each gap filled by one
    word (a path, a count) or by another string of the code."""
    for pattern in patterns:
        if (match := pattern.fullmatch(quote)) and all(
            not re.search(r"\s", gap) or gap in plain for gap in match.groups()
        ):
            return True
    return False


# --- the documents against the code --------------------------------------------------


@pytest.mark.parametrize("name", DOCS)
def test_every_command_and_option_named_exists(name: str) -> None:
    text = DOCS[name]
    problems = [p for words in command_lines(text) for p in resolve(words)[1]]
    known = every_option()
    for span in spans(text):
        if span.startswith("-") and (option := span.split()[0]) not in known:
            problems.append(f"no command has the option {option}")
    assert not problems


def test_the_guide_names_every_command() -> None:
    named = {
        " ".join(["redpen", *path])
        for path, problems in map(resolve, command_lines(GUIDE))
        if not problems
    }
    assert commands() <= named, commands() - named


@pytest.mark.parametrize("name", DOCS)
def test_every_profile_named_exists(name: str) -> None:
    text = DOCS[name]
    named = [
        w[i + 1]
        for w in command_lines(text)
        for i, word in enumerate(w[:-1])
        if word == "--profile"
    ]
    named += re.findall(r'^profile\s*=\s*"([^"]*)"', "".join(blocks(text)), re.M)
    assert set(named) <= set(PROFILES)


def test_the_guide_names_every_profile_setting_and_flag_code() -> None:
    named = set(spans(GUIDE))
    assert {*PROFILES, *SETTINGS, *FlagCode} <= named, {
        *PROFILES,
        *SETTINGS,
        *FlagCode,
    } - named


def test_every_setting_in_an_example_exists() -> None:
    for text in DOCS.values():
        for block in blocks(text, "toml"):
            for line in block.splitlines():
                if match := re.match(r"\s*\[(.*)\]", line):
                    assert match[1] == "models", line
                elif match := re.match(r"\s*([a-z_]+)\s*=", line):
                    assert match[1] in SETTINGS, line


def test_the_flag_table_gives_each_code_its_severity() -> None:
    rows = re.findall(r"^\| `([a-z_]+)` \| (\w+) \|", GUIDE, re.M)
    assert {code: severity for code, severity in rows} == {
        str(code): severity for code, severity in SEVERITY.items()
    }


@pytest.mark.parametrize("name", DOCS)
def test_every_name_in_code_is_one_the_tool_has(name: str, replayed: Workspace) -> None:
    """A word with an underscore in a code span is a flag code, a setting, a field
    of the key or the lock, or a column of an output file."""
    columns: set[str] = set()
    for path in (replayed.root / "out").glob("*.csv"):
        with path.open(encoding="utf-8-sig") as file:
            columns |= set(next(csv.reader(file)))
    fields = {
        f
        for model in vars(key_schema).values()
        if isinstance(model, type) and hasattr(model, "model_fields")
        for f in model.model_fields
    }
    known = {*FlagCode, *SETTINGS, *Lock.model_fields, *fields, *columns}
    named = {s for s in spans(DOCS[name]) if SNAKE.fullmatch(s)}
    assert named <= known, named - known


def test_every_message_quoted_is_printed_by_the_code() -> None:
    patterns, plain = messages()
    found = quotes(GUIDE)
    assert found
    assert [q for q in found if not printed(q, patterns, plain)] == []


# --- the workspace tree --------------------------------------------------------------


@pytest.fixture(scope="module")
def replayed(tmp_path_factory: pytest.TempPathFactory) -> Workspace:
    """The fixture class in a workspace, from recorded calls: the blank parsed (a
    recording that does not give the fixture's spec, which is then put in its place
    as an instructor corrects one), the key prepared from that spec and then
    replaced by the fixture's own (the recorded grades are of it), three students
    graded under the balanced profile, the one review list entry accepted and one
    award corrected."""
    root = tmp_path_factory.mktemp("docs")
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "blank.pdf", workspace.blank_pdf)
    submissions = root / "submissions"
    submissions.mkdir()
    for pdf in SUBMISSIONS:
        if truth_of(pdf).layout_class in CLASSES:
            shutil.copy(pdf, submissions / pdf.name)
    models = {"parser": PARSER, **KEY_MODELS, **GRADE_MODELS}
    write_lock(workspace.config_lock_toml, Lock(profile="balanced", models=models))
    with closing(sqlite3.connect(workspace.cache_sqlite, autocommit=True)) as db:
        for name in ("parse_blank", "prepare_blank", "run_chain"):
            CallCache(db).load(RECORDINGS / f"{name}.jsonl")
    transport = httpx.MockTransport(openrouter)
    offline = functools.partial(Gateway, offline=True, transport=transport)

    async def parsed() -> None:
        async with offline(workspace.cache_sqlite, "replay") as gateway:
            await parse(workspace, gateway, PARSER, Config().image_budget)

    asyncio.run(parsed())
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    runner = CliRunner()
    at = ["--workspace", str(workspace.root)]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("redpen.cli.Gateway", offline)
        patch.setattr("redpen.orchestrator.Gateway", offline)
        patch.setattr(
            "redpen.orchestrator.recheck_providers",
            functools.partial(recheck_providers, transport=transport),
        )
        patch.setenv("OPENROUTER_API_KEY", "replay")
        patch.chdir(root)  # no redpen.toml: the defaults, as recorded
        steps: list[tuple[list[str], str | None]] = [
            (["prepare", str(FIXTURES / "blank.pdf"), *at], None),
            (["grade", str(submissions), *at], None),
            (["review", *at], "accept\n"),
        ]
        for args, given in steps:
            if args[0] == "grade":
                key = Key.model_validate_json((FIXTURES / "key.json").read_text())
                save_key(workspace.key_yaml, key)
            result = runner.invoke(app, args, input=given)
            assert result.exit_code == 0, result.output
        student = read_model(workspace.ingest_summary_json, IngestSummary).students[0]
        result = runner.invoke(
            app, ["override", student.student_key, "Q1.c1", "1", *at]
        )
        assert result.exit_code == 0, result.output
    return workspace


def generic(workspace: Workspace) -> Iterator[str]:
    """Each file and folder of `workspace` as the guide's tree names it: a student
    key as `<student>`, a page as p01, a problem or part as Q1, an image's number
    as 1."""
    keys = [
        s.student_key
        for s in read_model(workspace.ingest_summary_json, IngestSummary).students
    ]
    for path in workspace.root.rglob("*"):
        name = path.relative_to(workspace.root).as_posix()
        for key in keys:
            name = re.sub(rf"(?<![^/]){re.escape(key)}(?=/|\.md$|$)", "<student>", name)
        name = re.sub(r"(?<=/)p\d+(?=[.-])", "p01", name)
        name = re.sub(r"(?<=/)Q\d+(?:\.[a-z0-9]+)*(?=\.json$|-\d+\.png$)", "Q1", name)
        name = re.sub(r"-\d+\.png$", "-1.png", name)
        yield name + ("/" if path.is_dir() else "")


def tree(text: str) -> Iterator[str]:
    """Each file and folder the guide's workspace tree names, below its root: the
    first code block under the heading "The workspace", indented two spaces a
    level, a note after each name."""
    section = text.split("\n## The workspace\n", 1)[1]
    stack: list[str] = []
    for line in blocks(section)[0].splitlines()[1:]:
        depth = (len(line) - len(line.lstrip(" "))) // 2
        stack[depth - 1 :] = [line.split()[0]]  # a folder's name ends in "/"
        yield "".join(stack)


def test_the_workspace_tree_is_a_replayed_workspace(replayed: Workspace) -> None:
    named = list(tree(GUIDE))
    assert len(named) == len(set(named))
    assert set(named) == set(generic(replayed))
