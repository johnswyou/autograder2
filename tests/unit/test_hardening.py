"""Clear messages for common mistakes (task 5.4a, rule 10): one line that says what
is wrong and what to do, exit code 1, no traceback, and nothing a student or a model
wrote; and the commands that need no API key run without one.

No model is called: OpenRouter is a mock that answers the ZDR listing and refuses
every chat request, and the map stage is a fake that finds no work."""

import functools
import shutil
import zipfile
from pathlib import Path
from typing import Any

import httpx
import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
import pytest
from reportlab.pdfgen.canvas import Canvas
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Lock, read_lock, write_lock
from redpen.llm.client import Gateway
from redpen.llm.models import recheck_providers
from redpen.prepare.key import save_key
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from redpen.schemas.submission import Map, MapPage, MapRecord
from redpen.workspace import Workspace, write_model
from tests.fixture_truth import FIXTURES, NOBODY
from tests.pinned import GRADE_MODELS, ZDR

runner = CliRunner()
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
NO_KEY = "OPENROUTER_API_KEY is not set. Export your OpenRouter API key in it."
WS = Workspace(Path("ws"))  # relative: each test runs in its own tmp_path


def pdf(path: Path, text: str = "1. x = 4", encrypt: str | None = None) -> Path:
    canvas = Canvas(str(path), encrypt=encrypt)
    canvas.drawString(72, 720, text)
    canvas.save()
    return path


def openrouter(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/endpoints/zdr"):
        return httpx.Response(200, json={"data": ZDR})
    return httpx.Response(400, json={"error": {"message": "refused by the test"}})


async def no_work(workspace: Workspace, student: str, spec: Spec, *_: object) -> None:
    found = Map(entries=[], unanswered=[p.id for p in spec.parts], set_aside=[])
    page = MapPage(number=1, rotation=0, photo=False)
    record = MapRecord(provenance=NOBODY, pages=[page], map=found)
    write_model(workspace.map_json(student), record)


@pytest.fixture
def class_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """In `tmp_path`: a prepared `ws`, a folder `subs` of two students' PDFs, a
    made-up API key, a mock OpenRouter and a map stage that finds no work."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "made-up")
    WS.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", WS.blank_pdf)
    shutil.copyfile(FIXTURES / "spec.json", WS.spec_json)
    save_key(WS.key_yaml, KEY)
    write_lock(WS.config_lock_toml, Lock(profile="balanced", models=GRADE_MODELS))
    subs = Path("subs")
    subs.mkdir()
    for name in ("Ann Example", "Ben Example"):
        pdf(subs / f"HW 1 - {name}.pdf")
    transport = httpx.MockTransport(openrouter)
    monkeypatch.setattr(
        "redpen.orchestrator.Gateway", functools.partial(Gateway, transport=transport)
    )
    monkeypatch.setattr(
        "redpen.orchestrator.recheck_providers",
        functools.partial(recheck_providers, transport=transport),
    )
    monkeypatch.setattr("redpen.orchestrator.map_student", no_work)
    return subs


def said(result: Any) -> list[str]:
    """The lines a command printed, but the stage progress."""
    return [
        line
        for line in result.output.splitlines()
        if line.strip() and not line.rstrip().endswith("$0.0000")
    ]


def assert_one_line(result: Any, line: str, code: int = 1) -> None:
    assert result.exit_code == code, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert said(result) == [line]


# --- the API key ------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        ["prepare", str(FIXTURES / "blank.pdf"), "--workspace", "new"],
        ["run", str(FIXTURES / "blank.pdf"), "subs", "--workspace", "new"],
        ["grade", "subs", "--workspace", "ws"],
        ["regrade", "--workspace", "ws"],
        ["key", "verify", "--workspace", "ws"],
        ["models", "check"],
    ],
    ids=lambda c: " ".join(c[:2]) if c[0] in ("key", "models") else c[0],
)
@pytest.mark.parametrize("given", [None, ""], ids=["unset", "empty"])
def test_a_command_that_needs_the_api_key_says_so_in_one_line(
    class_folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: list[str],
    given: str | None,
) -> None:
    if given is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", given)

    assert_one_line(runner.invoke(app, command), NO_KEY)


def test_the_commands_that_need_no_api_key_run_without_one(
    class_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runner.invoke(app, ["grade", "subs", "--workspace", "ws"]).exit_code == 0
    [student, *_] = sorted(p.name for p in WS.students_dir.iterdir())
    gold = "student,page 1,page 2,page 3,page 4\nAnn Example,1,0,0,0\n"
    Path("gold.csv").write_text(gold)
    monkeypatch.delenv("OPENROUTER_API_KEY")
    blank = str(FIXTURES / "blank.pdf")
    commands = [
        ["report", "--workspace", "ws"],
        ["review", "--list", "--workspace", "ws"],
        ["override", student, "Q1.c1", "1", "--workspace", "ws"],
        ["override", student, "Q1.c1", "--remove", "--workspace", "ws"],
        ["eval", "--gold", "gold.csv", "--workspace", "ws"],
        ["grade", "subs", "--dry-run", "--workspace", "ws"],
        ["regrade", "--dry-run", "--workspace", "ws"],
        ["run", blank, "subs", "--dry-run", "--workspace", "ws"],
        ["--help"],
        ["grade", "--help"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.output)
        assert "OPENROUTER_API_KEY" not in result.output, command


# --- the paths given ----------------------------------------------------------------


GIVE_BLANK = "Give BLANK.pdf as a PDF file that opens without a password."


def blanks(folder: Path) -> dict[str, Path]:
    """A blank of each kind that cannot be used."""
    folder.mkdir()
    data = (FIXTURES / "blank.pdf").read_bytes()
    (folder / "truncated.pdf").write_bytes(data[: len(data) // 2])
    (folder / "corrupt.pdf").write_bytes(b"%PDF-1.4 and then nothing")
    (folder / "notes.txt").write_text("not a PDF")
    (folder / "folder.pdf").mkdir()
    pdfium.PdfDocument.new().save(folder / "zero.pdf")
    return {
        "missing": folder / "missing.pdf",
        "folder": folder / "folder.pdf",
        "text": folder / "notes.txt",
        "encrypted": pdf(folder / "encrypted.pdf", encrypt="secret"),
        "corrupt": folder / "corrupt.pdf",
        "truncated": folder / "truncated.pdf",
        "no pages": folder / "zero.pdf",
    }


@pytest.mark.parametrize(
    "kind",
    ["missing", "folder", "text", "encrypted", "corrupt", "truncated", "no pages"],
)
@pytest.mark.parametrize("command", ["prepare", "run", "run --dry-run"])
def test_a_blank_that_cannot_be_opened_stops_the_command_in_one_line(
    class_folder: Path, command: str, kind: str
) -> None:
    blank = blanks(Path("blanks"))[kind]
    # The dry run takes the prepared workspace, whose lock prices it: a new one
    # would list OpenRouter's models first, over the network.
    args = {
        "prepare": ["prepare", str(blank), "--workspace", "new"],
        "run": ["run", str(blank), "subs", "--workspace", "new"],
        "run --dry-run": ["run", str(blank), "subs", "--dry-run", "--workspace", "ws"],
    }[command]

    result = runner.invoke(app, args)

    assert_one_line(
        result, f"{blank} cannot be opened as a PDF: it {WHY[kind]}. {GIVE_BLANK}"
    )
    assert not Path("new").exists() or not Workspace(Path("new")).blank_pdf.exists()


def test_the_blank_s_reason_is_its_own_after_an_encrypted_pdf(
    class_folder: Path,
) -> None:
    found = blanks(Path("blanks"))
    with pytest.raises(pdfium.PdfiumError):  # pdfium's last error: a password
        pdfium.PdfDocument(found["encrypted"])
    blank = found["no pages"]

    result = runner.invoke(app, ["prepare", str(blank), "--workspace", "new"])

    assert_one_line(
        result, f"{blank} cannot be opened as a PDF: it {DAMAGED}. {GIVE_BLANK}"
    )


# pdfium opens no PDF of no pages: one saved with none fails as a damaged file.
DAMAGED = "is not a PDF, or is damaged or cut short"
WHY = {
    "missing": "does not exist",
    "folder": "is a folder",
    "text": DAMAGED,
    "encrypted": "is encrypted",
    "corrupt": DAMAGED,
    "truncated": DAMAGED,
    "no pages": DAMAGED,
}


GIVE_SUBMISSIONS = "Give SUBMISSIONS as a folder or ZIP of the students' PDFs."


def submissions(folder: Path) -> dict[str, Path]:
    """A submissions path of each kind that holds no student PDF."""
    folder.mkdir()
    (folder / "empty").mkdir()
    (folder / "other").mkdir()
    (folder / "other" / "notes.txt").write_text("x")
    with zipfile.ZipFile(folder / "empty.zip", "w"):
        pass
    with zipfile.ZipFile(folder / "system.zip", "w") as zipped:
        zipped.writestr("class/", "")
        zipped.writestr("__MACOSX/class/._HW 1 - Ann.pdf", "x")
        zipped.writestr("class/.DS_Store", "x")
    (folder / "notes.txt").write_text("x")
    return {
        "missing": folder / "missing",
        "a PDF": pdf(folder / "one.pdf"),
        "a text file": folder / "notes.txt",
        "an empty folder": folder / "empty",
        "a folder of other files": folder / "other",
        "an empty ZIP": folder / "empty.zip",
        "a ZIP of system files": folder / "system.zip",
    }


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "a PDF",
        "a text file",
        "an empty folder",
        "a folder of other files",
        "an empty ZIP",
        "a ZIP of system files",
    ],
)
@pytest.mark.parametrize("dry_run", [False, True], ids=["grade", "dry run"])
def test_submissions_with_no_pdf_stop_the_command_in_one_line_with_nothing_sent(
    class_folder: Path, kind: str, dry_run: bool
) -> None:
    path = submissions(Path("given"))[kind]
    dry = ["--dry-run"] if dry_run else []

    result = runner.invoke(app, ["grade", str(path), *dry, "--workspace", "ws"])

    [line] = said(result)
    assert result.exit_code == 1
    assert line.startswith(("No student PDF", f"{path} cannot be read")), line
    assert line.endswith(GIVE_SUBMISSIONS), line
    assert not WS.students_dir.exists()


@pytest.mark.parametrize("kind", ["encrypted", "corrupt", "truncated", "no pages"])
def test_one_unreadable_submission_is_listed_and_the_others_are_graded(
    class_folder: Path, kind: str
) -> None:
    bad = blanks(Path("blanks"))[kind]
    shutil.copyfile(bad, class_folder / "HW 1 - Cal Example.pdf")

    result = runner.invoke(app, ["grade", "subs", "--workspace", "ws"])

    assert result.exit_code == 1
    assert said(result)[-1] == f"Files not ingested: 1. {WS.run_report_md} lists them."
    assert "Students: 2." in WS.run_report_md.read_text()
    assert (
        "- `HW 1 - Cal Example.pdf`: It could not be ingested, so nothing in it is"
        " graded." in WS.run_report_md.read_text().splitlines()
    )


def out_files() -> dict[str, bytes]:
    out = WS.root / "out"
    return {
        p.relative_to(out).as_posix(): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }


def test_submissions_with_no_pdf_leave_the_stored_class_as_it_was(
    class_folder: Path,
) -> None:
    assert runner.invoke(app, ["grade", "subs", "--workspace", "ws"]).exit_code == 0
    before = out_files()
    summary = WS.ingest_summary_json.read_bytes()
    Path("empty").mkdir()

    refused = runner.invoke(app, ["grade", "empty", "--workspace", "ws"])

    assert refused.exit_code == 1
    assert WS.ingest_summary_json.read_bytes() == summary
    assert runner.invoke(app, ["report", "--workspace", "ws"]).exit_code == 0
    assert out_files() == before


# --- the workspace ------------------------------------------------------------------


def no_prepared_key(root: str) -> str:
    return (
        f"{root} holds no prepared key. Run `redpen prepare BLANK.pdf --workspace"
        f" {root}` first."
    )


@pytest.mark.parametrize(
    "command",
    [
        ["grade", "subs", "--workspace", "other"],
        ["grade", "subs", "--dry-run", "--workspace", "other"],
        ["regrade", "--workspace", "other"],
        ["regrade", "--dry-run", "--workspace", "other"],
        ["report", "--workspace", "other"],
        ["review", "--workspace", "other"],
    ],
    ids=lambda c: " ".join(c[:-2]),
)
@pytest.mark.parametrize("state", ["not a workspace", "before a key"])
def test_a_folder_with_no_key_stops_the_command_in_one_line(
    class_folder: Path, command: list[str], state: str
) -> None:
    other = Workspace(Path("other"))
    if state == "before a key":  # prepare stopped once the blank and lock were in
        other.blank_pdf.parent.mkdir(parents=True)
        shutil.copyfile(FIXTURES / "blank.pdf", other.blank_pdf)
        shutil.copyfile(FIXTURES / "spec.json", other.spec_json)
        lock = Lock(profile="balanced", models=GRADE_MODELS)
        write_lock(other.config_lock_toml, lock)
    else:
        Path("other").mkdir()

    assert_one_line(runner.invoke(app, command), no_prepared_key("other"))


@pytest.mark.parametrize(
    "command",
    [["regrade", "--workspace", "ws"], ["report", "--workspace", "ws"]],
    ids=lambda c: c[0],
)
def test_a_workspace_with_no_class_stops_the_command_in_one_line(
    class_folder: Path, command: list[str]
) -> None:
    assert_one_line(
        runner.invoke(app, command),
        "ws holds no ingest of a class. Run `redpen grade SUBMISSIONS --workspace"
        " ws` first.",
    )


def test_two_submissions_of_one_student_are_said_in_one_line(
    class_folder: Path,
) -> None:
    (class_folder / "late").mkdir()
    late = class_folder / "late" / "Ann Example.pdf"
    shutil.copyfile(class_folder / "HW 1 - Ann Example.pdf", late)

    result = runner.invoke(app, ["grade", "subs", "--workspace", "ws"])

    assert result.exit_code == 1
    assert said(result)[-1] == (
        f"Students with more than one PDF: 1. The newest of each was graded, and"
        f" {WS.ingest_summary_json} names the others; remove those not meant to be"
        " graded."
    )


# --- files edited into something that does not parse --------------------------------


def test_a_key_that_does_not_parse_is_named_with_its_line(class_folder: Path) -> None:
    lines = WS.key_yaml.read_text().splitlines()
    WS.key_yaml.write_text("\n".join([*lines[:9], "  - : [", *lines[9:]]))

    for command in (["grade", "subs"], ["regrade"], ["grade", "subs", "--dry-run"]):
        assert_one_line(
            runner.invoke(app, [*command, "--workspace", "ws"]),
            "ws/key/key.yaml does not parse as YAML at line 11. `redpen key check`"
            " lists its problems.",
        )


@pytest.mark.parametrize(
    ("name", "text", "line"),
    [
        ("overrides.yaml", "- student: [\n", "ws/overrides.yaml does not parse"),
        ("reviews.yaml", "- student: [\n", "ws/reviews.yaml does not parse"),
        ("redpen.toml", "profile = \n", "redpen.toml: Invalid value (at line 1"),
        ("redpen.toml", "profil = 'x'\n", "redpen.toml: profil: Extra inputs"),
    ],
)
def test_an_edited_file_that_does_not_parse_is_named_with_its_place(
    class_folder: Path, name: str, text: str, line: str
) -> None:
    (Path("ws") / name if name.endswith(".yaml") else Path(name)).write_text(text)

    for command in (["grade", "subs"], ["grade", "subs", "--dry-run"]):
        result = runner.invoke(app, [*command, "--workspace", "ws"])
        [found] = said(result)
        assert result.exit_code == 1
        assert found.startswith(line), found


# --- stored files damaged (task 5.4a, fix round 1) -----------------------------------


def damaged_line(path: Path, again: str = "grade") -> str:
    return (
        f"{path} is damaged (not JSON). Delete it and run `redpen {again}` again to"
        " have it made again."
    )


def students_damaged(count: int) -> str:
    return (
        f"Stored files of students that are damaged (not JSON): {count}. ws/run.log"
        " names them. Delete them and run `redpen grade` again to have them made"
        " again."
    )


def logged_damaged(workspace: Workspace) -> list[str]:
    """The files the run log's last entry names as damaged."""
    entries = workspace.run_log.read_text().split("\n--- ")
    [heading, *paths] = entries[-1].removeprefix("--- ").splitlines()
    assert heading.endswith(" damaged (not JSON)")
    return paths


def test_a_damaged_coverage_file_is_counted_named_in_the_log_and_stops_the_report(
    class_folder: Path,
) -> None:
    assert runner.invoke(app, ["grade", "subs", "--workspace", "ws"]).exit_code == 0
    [first, *_] = sorted(WS.students_dir.iterdir())
    coverage = WS.coverage_json(first.name)
    coverage.write_bytes(coverage.read_bytes()[:20])  # cut short, as by a full disk

    result = runner.invoke(app, ["report", "--workspace", "ws"])

    assert_one_line(result, students_damaged(1))
    assert logged_damaged(WS) == [str(coverage)]


def test_a_damaged_audit_file_is_named_and_stops_the_command(
    class_folder: Path,
) -> None:
    assert runner.invoke(app, ["grade", "subs", "--workspace", "ws"]).exit_code == 0
    [student, *_] = sorted(p.name for p in WS.students_dir.iterdir())
    audit = WS.audit_json("Q4.c")
    audit.write_text("{")

    for command in (
        ["report"],
        ["review", "--list"],
        ["override", student, "Q1.c1", "1"],
    ):
        result = runner.invoke(app, [*command, "--workspace", "ws"])
        assert result.exit_code == 1, (command, result.output)
        assert said(result)[-1] == damaged_line(audit), command


# --- the lock (task 5.4a, fix round 1) -----------------------------------------------


class Killed(BaseException):
    """The process dies: no handler of the code under test runs."""


def test_a_lock_whose_write_is_cut_short_stays_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.lock.toml"
    old = Lock(profile="balanced", models=GRADE_MODELS)
    new = old.model_copy(update={"prompts": {"read/read": "2"}})
    write_lock(path, old)
    write_text = Path.write_text

    def killed_halfway(self: Path, text: str, *args: Any, **kwargs: Any) -> int:
        write_text(self, text[: len(text) // 2], *args, **kwargs)
        raise Killed

    monkeypatch.setattr(Path, "write_text", killed_halfway)
    with pytest.raises(Killed):
        write_lock(path, new)
    monkeypatch.undo()

    assert read_lock(path) == old
    write_lock(path, new)  # the next write takes the `.partial` left behind
    assert read_lock(path) == new


LOCK_DAMAGED = (
    "ws/config.lock.toml is damaged (not a lock). Delete it and run `redpen prepare`"
    " again to have the models pinned again from redpen.toml."
)


@pytest.mark.parametrize("text", ["", "profile = [\n"], ids=["empty", "garbage"])
def test_a_lock_that_does_not_load_is_called_damaged_by_each_command(
    class_folder: Path, text: str
) -> None:
    assert runner.invoke(app, ["grade", "subs", "--workspace", "ws"]).exit_code == 0
    [student, *_] = sorted(p.name for p in WS.students_dir.iterdir())
    WS.config_lock_toml.write_text(text)
    blank = str(FIXTURES / "blank.pdf")
    commands = [
        ["prepare", blank],
        ["run", blank, "subs"],
        ["run", blank, "subs", "--dry-run"],
        ["grade", "subs"],
        ["grade", "subs", "--dry-run"],
        ["regrade"],
        ["regrade", "--dry-run"],
        ["key", "verify"],
        ["report"],
        ["review"],
        ["override", student, "Q1.c1", "1"],
    ]
    for command in commands:
        result = runner.invoke(app, [*command, "--workspace", "ws"], input="quit\n")
        assert result.exit_code == 1, (command, result.output)
        assert result.exception is None or isinstance(result.exception, SystemExit)
        assert said(result)[-1] == LOCK_DAMAGED, command
        # Before it, review shows its entries; override records the correction.
        if command[0] not in ("review", "override"):
            assert said(result) == [LOCK_DAMAGED], command
    # A command that does not read the lock runs as before.
    assert runner.invoke(app, ["review", "--list", "--workspace", "ws"]).exit_code == 0
