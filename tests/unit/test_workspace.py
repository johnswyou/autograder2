import shutil
from pathlib import Path
from typing import NoReturn

import pytest
from typer.testing import CliRunner

from autograder2.cli import app
from autograder2.llm.models import ModelError
from autograder2.schemas.submission import IngestPage, IngestRecord
from autograder2.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES


def test_paths_follow_the_design_layout() -> None:
    ws = Workspace(Path("unit4-forces.autograder2"))

    assert ws.config_lock_toml.as_posix() == "unit4-forces.autograder2/config.lock.toml"
    assert ws.key_yaml.as_posix() == "unit4-forces.autograder2/key/key.yaml"
    assert ws.key_page(1).as_posix() == "unit4-forces.autograder2/key/pages/p01.png"
    assert (
        ws.student_page("s1", 12).as_posix()
        == "unit4-forces.autograder2/students/s1/pages/p12.png"
    )
    assert (
        ws.crop_png("s1", "Q7.b", 1).as_posix()
        == "unit4-forces.autograder2/students/s1/crops/Q7.b-1.png"
    )
    assert (
        ws.grade_json("s1", "Q7").as_posix()
        == "unit4-forces.autograder2/students/s1/grades/Q7.json"
    )
    assert (
        ws.audit_json("Q7.b").as_posix() == "unit4-forces.autograder2/audit/Q7.b.json"
    )
    assert ws.cache_sqlite.as_posix() == "unit4-forces.autograder2/cache.sqlite"
    assert (
        ws.scores_by_page_csv.as_posix()
        == "unit4-forces.autograder2/out/scores_by_page.csv"
    )


def test_default_workspace_is_named_after_the_blank_and_sits_beside_it() -> None:
    ws = Workspace.for_blank(Path("/term/unit4/ch4.2-forces.pdf"))

    assert ws.root == Path("/term/unit4/ch4.2-forces.autograder2")


def test_workspace_option_overrides_the_default() -> None:
    ws = Workspace.for_blank(Path("/term/unit4/forces.pdf"), Path("/elsewhere/ws"))

    assert ws.root == Path("/elsewhere/ws")


def test_a_model_written_to_json_reads_back_equal(tmp_path: Path) -> None:
    page = IngestPage(
        number=1, width_px=10, height_px=20, slide_number=None, native_ppi=None
    )
    record = IngestRecord(
        student_name="Zoë Ng", student_key="Zoe_Ng", source_kind="other", pages=[page]
    )
    path = tmp_path / "students" / "Zoe_Ng" / "ingest.json"

    write_model(path, record)
    write_model(path, record)  # replaces the earlier file

    assert read_model(path, IngestRecord) == record
    assert [p.name for p in path.parent.iterdir()] == ["ingest.json"]


# --- the workspace is kept out of git ------------------------------------------------

runner = CliRunner()


def refused(*args: object, **kwargs: object) -> NoReturn:
    raise ModelError("the catalog is not reachable in this test")


def test_prepare_gives_the_workspace_it_makes_a_gitignore_holding_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("autograder2.config.fetch_catalog", refused)
    blank = tmp_path / "blank.pdf"
    shutil.copyfile(FIXTURES / "blank.pdf", blank)

    result = runner.invoke(app, ["prepare", str(blank), "--workspace", "ws"])

    assert result.exit_code == 1  # it stops at the catalog, the folder made
    assert Workspace(Path("ws")).gitignore.read_text() == "*\n"


@pytest.mark.parametrize(
    "args",
    [
        ["report"],
        ["regrade"],
        ["key", "check"],
        ["key", "render"],
        ["key", "verify"],
        ["grade", "subs"],
        ["eval", "--gold", "gold.csv"],
        ["override", "x", "Q1.c1", "1"],
        ["models", "check"],
        ["run", "blank.pdf", "subs", "--dry-run"],
        ["prepare", "not-a.pdf"],  # it stops before it copies the blank
    ],
)
def test_any_command_that_finds_a_workspace_with_no_gitignore_writes_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = Workspace(Path("ws"))
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")
    Path("not-a.pdf").write_text("text\n")
    Path("subs").mkdir()

    runner.invoke(app, [*args, "--workspace", "ws"])

    assert workspace.gitignore.read_text() == "*\n"


def test_an_existing_gitignore_is_never_changed_and_a_folder_with_no_blank_gets_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    kept = Workspace(Path("kept"))
    kept.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", kept.blank_pdf)
    kept.gitignore.write_text("out/\n")
    linked = Workspace(Path("linked"))
    linked.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", linked.blank_pdf)
    linked.gitignore.symlink_to(tmp_path / "nowhere")
    Path("other").mkdir()

    for folder in ("kept", "linked", "other"):
        runner.invoke(app, ["report", "--workspace", folder])

    assert kept.gitignore.read_text() == "out/\n"
    assert linked.gitignore.is_symlink() and not (tmp_path / "nowhere").exists()
    assert list(Path("other").iterdir()) == []
