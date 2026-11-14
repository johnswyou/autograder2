import functools
import logging
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any, NoReturn

import httpx
import pypdfium2 as pdfium
import pytest
from reportlab.pdfgen import canvas
from rich.text import Text
from typer.testing import CliRunner

from redpen.cli import _point_value, app
from redpen.config import Models, read_lock
from redpen.llm.models import Catalog, recheck_providers
from redpen.prepare.key import load_key, part_hash, save_key
from redpen.prepare.verify import Verification
from redpen.schemas.key import Criterion, Key, KeyPart, PointConstraint, Policies
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES, answer

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
        (["key"], {"check", "render", "verify"}),
        (["models"], {"check"}),
    ],
)
def test_help_lists_every_command(group: list[str], commands: set[str]) -> None:
    result = runner.invoke(app, [*group, "--help"])

    assert result.exit_code == 0
    assert commands <= first_column(result.output)


@pytest.mark.parametrize(
    "args",
    [
        ["review"],
        ["override", "s1", "Q7.b-1", "2.5"],
    ],
)
def test_stub_accepts_its_arguments_and_reports_not_implemented(
    args: list[str],
) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert result.stderr.strip() == "not implemented yet"


# --total and --page-points are supported since task 4.5a (tests/unit/test_points.py).
@pytest.mark.parametrize(("args", "flag"), [(["prepare", "--dry-run"], "--dry-run")])
def test_prepare_and_run_refuse_what_they_do_not_support_yet(
    tmp_path: Path, args: list[str], flag: str
) -> None:
    command, *rest = args
    result = runner.invoke(
        app, [command, "blank.pdf", "--workspace", str(tmp_path / "ws"), *rest]
    )

    assert result.exit_code == 1
    assert result.stderr.strip() == f"{flag} is not supported yet"
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["run", "blank.pdf", "subs", "--total", "fifty"],
        ["prepare", "blank.pdf", "--page-points", "nan"],
        ["prepare", "blank.pdf", "--total", "-3"],
        ["override", "s1", "Q7.b.c1", "Infinity"],
        ["override", "s1", "Q7.b.c1", "0x10"],
    ],
)
def test_points_that_are_not_a_number_are_refused_with_a_clear_message(
    args: list[str],
) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 2
    # Typer draws the error in a box, wrapped to the terminal's width.
    text = " ".join(Text.from_ansi(result.stderr).plain.replace("│", " ").split())
    assert "is not a number of points, such as 50 or 2.5" in text


def test_points_are_read_as_decimals_never_through_float() -> None:
    assert _point_value("0.10000000000000000001") == Decimal("0.10000000000000000001")
    assert _point_value(" 2.50 ") == Decimal("2.50")


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
    save_key(workspace.key_yaml, key_of(reused_spec(2)))  # prepare stops at the key

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == exit_code
    assert message in result.stderr


def refuse(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("prepare reached for OpenRouter")


def key_of(spec: Spec) -> Key:
    """A valid key for `spec`: one criterion per part."""
    return Key(
        policies=Policies(),
        parts=[
            KeyPart(
                id=p.id,
                label=p.label,
                home_page=p.home_page,
                response_type=p.response_type,
                depends_on=[],
                state="unverified",
                prompt=p.prompt,
                answer=answer(value=1.0),
                solution="x = 1",
                criteria=[
                    Criterion(
                        id=f"{p.id}.c1",
                        points=Decimal(1),
                        kind="final_answer",
                        requirement="States x = 1.",
                        partial=[],
                        carry_forward=False,
                    )
                ],
                common_errors=[],
            )
            for p in spec.parts
        ],
    )


@pytest.mark.parametrize(
    ("parts", "exit_code", "lines"),
    [
        (5, 0, ["ws/key/key.yaml has no problem and is left as it is."]),
        (
            4,
            1,
            [
                "Q5 is in the spec but not in the key",
                "ws/key/key.yaml has problems and is left as it is.",
            ],
        ),
    ],
    ids=["no problem", "a problem"],
)
def test_prepare_checks_an_existing_key_leaves_it_as_it_is_and_calls_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parts: int,
    exit_code: int,
    lines: list[str],
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
    first = reused_spec(2).parts[0]
    spec = Spec(
        parts=[
            first.model_copy(update={"id": f"Q{n}", "label": str(n)})
            for n in range(1, 6)
        ]
    )
    write_model(workspace.spec_json, spec)
    save_key(workspace.key_yaml, key_of(Spec(parts=spec.parts[:parts])))
    edited = b"# Edited by hand.\n" + workspace.key_yaml.read_bytes()
    workspace.key_yaml.write_bytes(edited)

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == exit_code, result.output
    assert result.output.splitlines()[-len(lines) :] == lines
    assert workspace.key_yaml.read_bytes() == edited
    assert not workspace.solutions_md.exists()


def prepared_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    """A workspace prepared from the fixture blank, with a valid key and LOCK; any
    reach for OpenRouter fails."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("redpen.config.fetch_catalog", refuse)
    monkeypatch.setattr("redpen.cli.Gateway", refuse)
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(tmp_path / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy("blank.pdf", workspace.blank_pdf)
    workspace.config_lock_toml.write_text(KEY_LOCK)
    spec = reused_spec(2)
    write_model(workspace.spec_json, spec)
    save_key(workspace.key_yaml, key_of(spec))
    return workspace


def test_a_profile_the_lock_does_not_pin_is_named_and_the_lock_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)

    result = runner.invoke(
        app, ["prepare", "blank.pdf", "--workspace", "ws", "--profile", "thorough"]
    )

    assert result.exit_code == 0, result.output
    assert (
        "The workspace's lock pins the profile balanced, and the lock wins over"
        " --profile thorough." in result.stderr.splitlines()
    )
    assert read_lock(workspace.config_lock_toml).profile == "balanced"


MARKER = "MARKER-7f3a written by a model"


@pytest.mark.parametrize(
    ("stored", "text"),
    [
        ("key/key.yaml", f"policies: {{}}\nparts:\n  - id: {MARKER}\n"),
        ("key/spec.json", f'{{"parts": [{{"id": "{MARKER}"}}]}}'),
        ("config.lock.toml", f'profile = "{MARKER}"\n'),
        ("config.lock.toml", f"profile = {MARKER}\n"),
    ],
    ids=["key", "spec", "lock schema", "lock TOML"],
)
def test_prepare_puts_no_stored_text_on_the_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stored: str, text: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)
    (workspace.root / stored).write_text(text)

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])

    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "MARKER" not in result.output + result.stderr


def test_key_check_asks_for_no_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    prepared_key(tmp_path, monkeypatch)

    result = runner.invoke(app, ["key", "check", "--workspace", "ws"])

    assert result.exit_code == 0, result.output
    assert "no problem found" in result.output


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
# LOCK with the roles a fresh prepare verifies with: a checker of another vendor,
# and the grader that tries the rubric.
KEY_LOCK = LOCK + "".join(
    f'\n[models.{role}]\nslug = "{slug}"\nsupported_parameters = ["structured_outputs"]'
    '\naccepts_images = true\nprompt_price = "0"\ncompletion_price = "0"'
    '\nproviders = ["zero"]\n'
    for role, slug in (
        ("solver", "vendor/vision-1"),
        ("checker", "maker/vision-2"),
        ("referee", "vendor/vision-1"),
        ("grader", "vendor/vision-1"),
    )
)
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


# --- --total and --page-points (task 4.5a) -------------------------------------------


@pytest.mark.parametrize(
    "command",
    [["prepare", "blank.pdf"], ["run", "blank.pdf", "subs"]],
    ids=["prepare", "run"],
)
def test_a_total_and_page_maximums_together_are_refused(
    tmp_path: Path, command: list[str]
) -> None:
    both = ["--total", "4", "--page-points", "2,2"]
    result = runner.invoke(app, [*command, "--workspace", str(tmp_path / "ws"), *both])

    assert result.exit_code == 2
    text = " ".join(Text.from_ansi(result.stderr).plain.replace("│", " ").split())
    assert (
        "Invalid value for '--page-points': cannot be given together with --total"
        in text
    )
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize(
    "args",
    [
        ["prepare", "blank.pdf", "--page-points", "3,x"],
        ["run", "blank.pdf", "subs", "--page-points", "1,,2"],
        ["prepare", "blank.pdf", "--page-points", "3,-1"],
    ],
)
def test_page_maximums_that_are_not_numbers_are_refused_as_points_are(
    args: list[str],
) -> None:
    result = runner.invoke(app, args)

    assert result.exit_code == 2
    text = " ".join(Text.from_ansi(result.stderr).plain.replace("│", " ").split())
    assert "is not a number of points, such as 50 or 2.5" in text


def test_a_fresh_prepare_writes_the_points_given_into_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)
    workspace.key_yaml.unlink()
    spec = reused_spec(2)

    async def stages(*args: object) -> tuple[Key, Verification]:  # replaced
        return key_of(spec), Verification(
            checker="maker/vision-2", keyed=False, parts=[]
        )

    monkeypatch.setattr("redpen.cli._key", stages)

    result = runner.invoke(
        app, ["prepare", "blank.pdf", "--workspace", "ws", "--total", "4"]
    )

    assert result.exit_code == 0, result.output
    key = load_key(workspace.key_yaml)
    assert key.points == PointConstraint(total=Decimal(4))
    assert key.model_copy(update={"points": None}) == key_of(spec)
    # 16 steps of 0.25 for two parts of one point each: 2 and 2.
    assert result.output.splitlines()[-3:] == [
        "Total points: 4",
        "  page 1: 2",
        "  page 2: 2",
    ]
    assert "## Q1: 1, page 1, 2 points\n" in workspace.rubric_md.read_text()


def hashes_of(workspace: Workspace) -> dict[str, str]:
    key = load_key(workspace.key_yaml)
    return {p.id: part_hash(p, key.policies) for p in key.parts}


def test_on_an_existing_key_only_the_points_change_and_its_comments_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)
    edited = b"# Edited by hand.\n" + workspace.key_yaml.read_bytes()
    workspace.key_yaml.write_bytes(edited)
    before, hashes = load_key(workspace.key_yaml), hashes_of(workspace)

    def prepare(*points: str) -> list[str]:
        result = runner.invoke(
            app, ["prepare", "blank.pdf", "--workspace", "ws", *points]
        )
        assert result.exit_code == 0, result.output
        assert hashes_of(workspace) == hashes
        return result.output.splitlines()

    lines = prepare("--page-points", "3,1,0,0")

    assert lines[-4:] == [
        "Set the points of ws/key/key.yaml to page maximums of 3, 1, 0, 0, and wrote"
        " solutions.md and rubric.md again.",
        "Total points: 4",
        "  page 1: 3",
        "  page 2: 1",
    ]
    pages = b"points:\n  pages:\n    - 3\n    - 1\n    - 0\n    - 0\n"
    assert workspace.key_yaml.read_bytes() == edited + pages
    assert load_key(workspace.key_yaml) == before.model_copy(
        update={"points": PointConstraint(pages=[Decimal(n) for n in (3, 1, 0, 0)])}
    )
    rubric = workspace.rubric_md.read_text()
    assert "Points follow page maximums of 3, 1, 0, 0. A part's score is" in rubric
    assert "## Q3: 3, page 2, 1 point\n" in rubric

    # Given neither, the key's points stay as they are.
    assert prepare()[-1] == "ws/key/key.yaml has no problem and is left as it is."
    assert workspace.key_yaml.read_bytes() == edited + pages

    prepare("--total", "2")
    assert workspace.key_yaml.read_bytes() == edited + b"points:\n  total: 2\n"


def test_points_that_cannot_be_met_leave_an_existing_key_as_it_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)
    before = workspace.key_yaml.read_bytes()

    result = runner.invoke(
        app, ["prepare", "blank.pdf", "--workspace", "ws", "--page-points", "1,1"]
    )

    assert result.exit_code == 1
    assert result.output.splitlines()[-2:] == [
        "the key gives 2 page maximums and the blank has 4 pages",
        "ws/key/key.yaml has problems and is left as it is.",
    ]
    assert workspace.key_yaml.read_bytes() == before
    assert not workspace.rubric_md.exists()


@pytest.mark.parametrize(
    ("points", "line"),
    [
        (
            ["--page-points", "1,1"],
            "the key gives 2 page maximums and the blank has 4 pages",
        ),
        (
            ["--page-points", "0,1,0,0"],
            "page 1's maximum is 0: 0 steps of 0.25 for 1 part, and each part needs"
            " one",
        ),
        (
            ["--total", "0.25"],
            "the total is 0.25: 1 step of 0.25 for 2 parts, and each part needs one",
        ),
    ],
    ids=["wrong number of pages", "no step for a part", "total too small"],
)
def test_a_fresh_prepare_refuses_points_that_cannot_be_met_before_any_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, points: list[str], line: str
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)
    workspace.key_yaml.unlink()
    for stage in ("_key", "solve", "write_rubric"):  # each fails the test if called
        monkeypatch.setattr(f"redpen.cli.{stage}", refuse)

    result = runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws", *points])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # not the fakes' AssertionError
    # The spec of Q1 and Q3 gives its warning first, as every prepare does.
    gap = "Warning: no problem numbered 2 was found."
    assert result.stderr.splitlines() == [gap, line]
    assert not workspace.key_yaml.exists()


def test_run_with_a_total_sets_it_in_the_key_and_reaches_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    workspace = prepared_key(tmp_path, monkeypatch)  # a key; OpenRouter refused
    reached: list[tuple[Path, PointConstraint | None, Path, bool]] = []

    def grade(
        ws: Workspace, submissions: Path, roster: Path | None, dry_run: bool
    ) -> None:
        points = load_key(ws.key_yaml).points  # as grading would find it
        reached.append((ws.key_yaml.resolve(), points, submissions, dry_run))

    monkeypatch.setattr("redpen.cli._grade", grade)

    result = runner.invoke(
        app, ["run", "blank.pdf", "subs", "--workspace", "ws", "--total", "4"]
    )

    assert result.exit_code == 0, result.output
    total = PointConstraint(total=Decimal(4))
    assert reached == [(workspace.key_yaml.resolve(), total, Path("subs"), False)]
