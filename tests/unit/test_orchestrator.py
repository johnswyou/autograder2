"""`redpen run`, `grade` and `report`: the stages in order over the ingested class,
the privacy gate, one student's failure, the exit code, the dry run, the report
rebuilt from stored files, and the lock's prompt versions.

No model is called. OpenRouter is a mock transport that answers the ZDR listing and
refuses every chat request, and some tests replace stages with fakes."""

import asyncio
import csv
import functools
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import httpx
import pytest
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

import redpen.prompts
from redpen.cli import app
from redpen.config import Lock, read_lock, write_lock
from redpen.grade.ingest import IngestSummary
from redpen.llm.client import Gateway
from redpen.llm.models import Catalog, recheck_providers
from redpen.orchestrator import PER_PAGE, estimate, image_tokens, shape
from redpen.prepare.key import save_key
from redpen.report.report import STRICT_OFF_RUN, UNGRADED
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from redpen.schemas.submission import (
    IngestPage,
    IngestRecord,
    Map,
    MapPage,
    MapRecord,
)
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY
from tests.pinned import GRADE_MODELS, GRADER, MAPPER, READER, ZDR

runner = CliRunner()
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
SPEC = read_model(FIXTURES / "spec.json", Spec)
NAMES = ("Ann Example", "Ben Example", "Cal Example")
ROLES = GRADE_MODELS


class OpenRouter:
    """A mock OpenRouter that keeps every request: the ZDR listing answers as
    `zdr` says, and every chat request is refused with HTTP 400."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.zdr: Callable[[httpx.Request], httpx.Response] = lambda _: httpx.Response(
            200, json={"data": ZDR}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/endpoints/zdr"):
            return self.zdr(request)
        return httpx.Response(400, json={"error": {"message": "refused by the test"}})

    @property
    def chats(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/chat/completions")]


def write_pdf(path: Path, text: str) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, text)
    pdf.save()


def prepared(root: Path, lock: Lock) -> Workspace:
    """A workspace holding the fixture assignment's key and `lock`."""
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(FIXTURES / "spec.json", workspace.spec_json)
    save_key(workspace.key_yaml, KEY)
    write_lock(workspace.config_lock_toml, lock)
    return workspace


@pytest.fixture
def router(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OpenRouter:
    """In `tmp_path`: a prepared `ws`, a folder `subs` of one-page PDFs for NAMES,
    an API key, and OpenRouter replaced for the gateway and the privacy check."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    prepared(tmp_path, Lock(profile="balanced", models=ROLES))
    Path("subs").mkdir()
    for name in NAMES:
        write_pdf(Path("subs") / f"HW 1 - {name}.pdf", "1. x = 4")
    openrouter = OpenRouter()
    transport = httpx.MockTransport(openrouter)
    monkeypatch.setattr(
        "redpen.orchestrator.Gateway", functools.partial(Gateway, transport=transport)
    )
    monkeypatch.setattr(
        "redpen.orchestrator.recheck_providers",
        functools.partial(recheck_providers, transport=transport),
    )
    return openrouter


def grade() -> Any:
    return runner.invoke(app, ["grade", "subs", "--workspace", "ws"])


def students() -> list[str]:
    return [
        s.student_key
        for s in read_model(WS.ingest_summary_json, IngestSummary).students
    ]


WS = Workspace(Path("ws"))  # relative: every test runs in its own tmp_path


def unanswered_map(fails: str | None = None) -> Callable[..., Any]:
    """A map stage that finds no work (so read and grade make no call), and raises
    for the student whose key starts with `fails`."""

    async def fake(workspace: Workspace, student: str, spec: Spec, *_: object) -> None:
        if fails and student.startswith(fails):
            raise RuntimeError("the map broke")
        found = Map(entries=[], unanswered=[p.id for p in spec.parts], set_aside=[])
        page = MapPage(number=1, rotation=0, photo=False)
        write_model(
            workspace.map_json(student),
            MapRecord(provenance=NOBODY, pages=[page], map=found),
        )

    return fake


def out_files(workspace: Workspace) -> dict[Path, bytes]:
    return {
        p: p.read_bytes() for p in (workspace.root / "out").rglob("*") if p.is_file()
    }


def by_problem(workspace: Workspace) -> list[dict[str, str]]:
    with workspace.scores_by_problem_csv.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_run_prepares_then_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr("redpen.cli.prepare", lambda *a: calls.append(("prepare", a)))
    monkeypatch.setattr("redpen.cli._grade", lambda *a, **k: calls.append(("grade", a)))
    args = ["run", "b.pdf", "subs", "--workspace", "ws", "--profile", "economy"]

    result = runner.invoke(app, [*args, "--roster", "r.csv"])

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            "prepare",
            (Path("b.pdf"), None, None, "economy", Path("r.csv"), False, Path("ws")),
        ),
        ("grade", (Workspace(Path("ws")), Path("subs"), Path("r.csv"))),
    ]


def test_run_grades_nothing_when_prepare_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*args: object) -> None:
        raise SystemExit(1)

    monkeypatch.setattr("redpen.cli.prepare", failing)
    monkeypatch.setattr("redpen.cli._grade", lambda *a, **k: pytest.fail("graded"))

    assert runner.invoke(app, ["run", "b.pdf", "subs"]).exit_code == 1


def test_grade_runs_each_stage_over_the_ingested_class_in_order(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    for stage in ("map", "read", "grade"):

        async def fake(
            _: Workspace, student: str, *args: object, stage: str = stage
        ) -> None:
            calls.append((stage, student))

        monkeypatch.setattr(f"redpen.orchestrator.{stage}_student", fake)

    result = grade()

    assert result.exit_code == 0, result.output
    assert [stage for stage, _ in calls] == ["map"] * 3 + ["read"] * 3 + ["grade"] * 3
    assert [student for _, student in calls] == students() * 3
    assert len(students()) == len(NAMES)
    assert [row["student"] for row in by_problem(WS)] == list(NAMES)
    assert WS.run_report_md.exists() and len(list(WS.feedback_dir.iterdir())) == 3
    assert router.chats == []


def test_no_more_students_run_at_once_in_a_stage_than_the_gateway_s_limit(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("Dee Example", "Eve Example"):
        write_pdf(Path("subs") / f"HW 1 - {name}.pdf", "1. x = 4")
    transport = httpx.MockTransport(router)
    monkeypatch.setattr(
        "redpen.orchestrator.Gateway",
        functools.partial(Gateway, transport=transport, max_concurrency=2),
    )
    running: list[str] = []
    peak = 0
    mapped = unanswered_map()

    async def fake(workspace: Workspace, student: str, *args: Any) -> None:
        nonlocal peak
        running.append(student)
        peak = max(peak, len(running))
        await asyncio.sleep(0.01)  # the others get their turn here
        running.remove(student)
        await mapped(workspace, student, *args)

    monkeypatch.setattr("redpen.orchestrator.map_student", fake)

    assert grade().exit_code == 0
    assert len(students()) == 5 and peak == 2


def test_a_folder_in_students_not_in_the_ingest_summary_is_in_no_output(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = IngestRecord(
        student_name="Old Spelling",
        student_key="Old_Spelling-0a1b2c",
        source_kind="other",
        pages=[],
    )
    write_model(WS.ingest_json(stale.student_key), stale)
    seen: list[str] = []
    mapped = unanswered_map()

    async def fake(workspace: Workspace, student: str, *args: Any) -> None:
        seen.append(student)
        await mapped(workspace, student, *args)

    monkeypatch.setattr("redpen.orchestrator.map_student", fake)

    assert grade().exit_code == 0
    assert stale.student_key not in seen and len(seen) == 3
    for path, data in out_files(WS).items():
        assert b"Old_Spelling" not in data and b"Old Spelling" not in data, path


def test_one_student_s_failure_is_recorded_and_the_class_goes_on(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map("Ben"))

    result = grade()

    assert result.exit_code == 1
    assert [line for line in result.stderr.splitlines() if "failed" in line] == [
        f"{stage}: 1 of 3 students failed." for stage in ("map", "read", "grade")
    ]
    assert "Ben" not in result.output + result.stderr
    entries = WS.run_log.read_text().split("--- ")[1:]
    assert [e.split("\n")[0].split()[1:] for e in entries] == [
        [stage, next(s for s in students() if s.startswith("Ben"))]
        for stage in ("map", "read", "grade")
    ]
    assert all("Traceback (most recent call last)" in e for e in entries)
    assert "RuntimeError: the map broke" in entries[0]
    assert "FileNotFoundError" in entries[1] and "map.json" in entries[1]
    assert "FileNotFoundError" in entries[2] and "readings" in entries[2]
    rows = {row["student"]: row for row in by_problem(WS)}
    for name in ("Ann Example", "Cal Example"):
        assert (rows[name]["total"], rows[name]["status"]) == ("0", "final")
    ben = rows["Ben Example"]
    assert [ben[q] for q in ("Q1", "Q2", "Q3", "Q4", "Q5")] == [""] * 5
    assert (ben["status"], ben["open_flags"]) == ("incomplete", str(len(KEY.parts)))
    [key] = [s for s in students() if s.startswith("Ben")]
    parts = ", ".join(p.id for p in KEY.parts)
    assert f"- `{key}`: {parts}. {UNGRADED}" in WS.run_report_md.read_text()
    assert f"| incomplete | high | {len(KEY.parts)} |" in WS.run_report_md.read_text()


MARKER = "MARKER-7f3a written by a student"


def quiet(result: Any) -> None:
    """`result` ended by the CLI with exit code 1, no traceback, and nothing of
    MARKER on the terminal."""
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert "MARKER" not in result.output + result.stderr


@pytest.mark.parametrize("stored", ["readings", "grades"])
def test_report_puts_no_stored_text_on_the_terminal(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, stored: str
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    assert grade().exit_code == 0
    path = WS.student_dir(students()[0]) / stored / "Q4.json"
    path.write_text(f'{{"records": [{{"reading": "{MARKER}"}}]}}')

    result = runner.invoke(app, ["report", "--workspace", "ws"])

    quiet(result)
    assert result.stderr.splitlines() == [
        f"The report could not be written. The traceback is in {WS.run_log}."
    ]
    assert "ValidationError" in WS.run_log.read_text()


@pytest.mark.parametrize(
    ("stored", "text"),
    [
        ("key/key.yaml", f"policies: {{}}\nparts:\n  - id: {MARKER}\n"),
        ("key/key.yaml", f"parts: [{MARKER}\n"),
        ("key/spec.json", f'{{"parts": [{{"id": "{MARKER}"}}]}}'),
        ("config.lock.toml", f'profile = "{MARKER}"\n'),
        ("config.lock.toml", f"profile = {MARKER}\n"),
        ("cache.sqlite", MARKER * 100),
    ],
    ids=["key schema", "key YAML", "spec", "lock schema", "lock TOML", "cache"],
)
def test_grade_puts_no_stored_text_on_the_terminal(
    router: OpenRouter, stored: str, text: str
) -> None:
    (WS.root / stored).write_text(text)

    quiet(grade())


@pytest.mark.parametrize(
    ("stored", "text"),
    [
        ("key/spec.json", f'{{"parts": [{{"id": "{MARKER}"}}]}}'),
        ("config.lock.toml", f'profile = "{MARKER}"\n'),
    ],
    ids=["spec", "lock"],
)
def test_a_dry_run_of_run_puts_no_stored_text_on_the_terminal(
    router: OpenRouter, stored: str, text: str
) -> None:
    (WS.root / stored).write_text(text)
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")

    quiet(
        runner.invoke(
            app, ["run", "blank.pdf", "subs", "--workspace", "ws", "--dry-run"]
        )
    )


def test_the_failures_are_named_even_when_the_report_cannot_be_written(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError(MARKER)

    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map("Ben"))
    monkeypatch.setattr("redpen.cli.write_report", broken)

    result = grade()

    quiet(result)
    assert result.stderr.splitlines()[-5:] == [  # after the progress lines
        *(f"{stage}: 1 of 3 students failed." for stage in ("map", "read", "grade")),
        f"Each traceback is in {WS.run_log}.",
        f"The report could not be written. The traceback is in {WS.run_log}.",
    ]


def test_report_rebuilds_out_byte_for_byte_with_no_api_key(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map("Ben"))
    assert grade().exit_code == 1
    before = out_files(WS)
    shutil.rmtree(WS.root / "out")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    requests = len(router.requests)

    result = runner.invoke(app, ["report", "--workspace", "ws"])

    assert result.exit_code == 0, result.output
    assert out_files(WS) == before
    assert len(router.requests) == requests


def test_a_file_that_cannot_be_ingested_fails_the_run_after_the_reports(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    Path("subs", "HW 1 - Dee `x`.pdf").write_bytes(b"%PDF-1.7 not really")

    result = grade()

    assert result.exit_code == 1
    assert len(by_problem(WS)) == 3
    assert "- ``HW 1 - Dee `x`.pdf``: " in WS.run_report_md.read_text()
    assert f"Files not ingested: 1. {WS.run_report_md} lists them." in (
        result.stderr.splitlines()
    )
    assert "Dee" not in result.output + result.stderr


def test_no_pdf_gives_exit_code_1_with_one_line(router: OpenRouter) -> None:
    for pdf in Path("subs").iterdir():
        pdf.unlink()

    result = grade()

    assert result.exit_code == 1
    assert result.stderr.splitlines() == ["No student PDF was found in subs."]
    assert not (WS.root / "out").exists() and router.requests == []


def no_endpoint(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route to host", request=request)


RETAINING = {**ZDR[0], "tag": f"{ZDR[0]['tag']}/priority", "safety_retention": None}


@pytest.mark.parametrize(
    "zdr",
    [
        lambda _: httpx.Response(200, json={"data": [*ZDR, RETAINING]}),
        lambda _: httpx.Response(500, json={"error": {"message": "down"}}),
        no_endpoint,
        lambda _: httpx.Response(200, json={"data": [1, 2]}),
        lambda _: httpx.Response(200, json={"models": []}),
        lambda _: httpx.Response(200, content=b"<html>"),
    ],
    ids=[
        "retaining",
        "server error",
        "unreachable",
        "not endpoints",
        "no data",
        "html",
    ],
)
def test_a_failed_privacy_check_sends_nothing(
    router: OpenRouter, zdr: Callable[[httpx.Request], httpx.Response]
) -> None:
    router.zdr = zdr

    result = grade()

    assert result.exit_code == 1
    assert result.stderr.splitlines()[-1] == (
        "Nothing was sent: the pinned endpoints are not verified to keep nothing."
    )
    assert [r.url.path for r in router.requests] == ["/api/v1/endpoints/zdr"]
    assert not any(b"image_url" in r.content for r in router.requests)
    assert not any(WS.map_json(s).exists() for s in students())
    assert not (WS.root / "out").exists()


def test_a_lock_with_no_model_for_a_grade_role_sends_nothing(
    router: OpenRouter, tmp_path: Path
) -> None:
    shutil.rmtree("ws")
    prepared(
        tmp_path, Lock(profile="balanced", models={"mapper": MAPPER, "grader": GRADER})
    )

    result = grade()

    assert result.exit_code == 1
    assert "reader: no model is pinned in the lock" in result.stderr.splitlines()
    assert router.requests == []


def test_a_check_that_leaves_a_role_out_sends_nothing(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def partial_answer(models: dict[str, Any], *args: object, **kwargs: object):
        return {"mapper": []}

    monkeypatch.setattr("redpen.orchestrator.recheck_providers", partial_answer)

    result = grade()

    assert result.exit_code == 1
    assert "reader: its endpoints were not verified" in result.stderr.splitlines()
    assert router.chats == []


def test_with_the_privacy_check_passing_the_map_stage_starts(
    router: OpenRouter,
) -> None:
    result = grade()

    paths = [r.url.path for r in router.requests]
    assert paths[0] == "/api/v1/endpoints/zdr"
    assert router.chats and all(b"image_url" in r.content for r in router.chats)
    # Each page goes to the lock's model for its role, on the endpoints that keep
    # nothing and nowhere else.
    role = {"Inventory": "mapper", "Resolution": "mapper", "Readings": "reader"}
    role |= {"Grades": "grader"}
    for request in router.chats:
        body = json.loads(request.content)
        model = ROLES[role[body["response_format"]["json_schema"]["name"]]]
        assert body["model"] == model.slug
        assert body["provider"] == {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": sorted(model.providers),
        }
    assert result.exit_code == 1  # every chat request is refused
    assert "map: 3 of 3 students failed." in result.stderr.splitlines()
    assert (WS.root / "out" / "run_report.md").exists()


def test_a_dry_run_sends_nothing_and_writes_no_output(router: OpenRouter) -> None:
    result = runner.invoke(app, ["grade", "subs", "--workspace", "ws", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert router.requests == []
    assert not (WS.root / "out").exists()
    lines = result.output.splitlines()
    assert [line.split()[0] for line in lines[2:7]] == [
        "prepare", "map", "read", "grade", "total"
    ]  # fmt: skip
    assert lines[2].split()[1:] == ["0", "0", "0", "0.0000"]  # the key exists
    assert "- The key exists, so prepare makes no call." in lines


def test_a_dry_run_over_another_folder_leaves_the_stored_class_alone(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    assert grade().exit_code == 0
    before = out_files(WS)
    summary = WS.ingest_summary_json.read_bytes()
    folders = sorted(WS.students_dir.iterdir())
    Path("other").mkdir()
    write_pdf(Path("other") / "HW 1 - Zed Example.pdf", "1. x = 5")

    dry = runner.invoke(app, ["grade", "other", "--workspace", "ws", "--dry-run"])
    result = runner.invoke(app, ["report", "--workspace", "ws"])

    assert dry.exit_code == 0 and result.exit_code == 0, dry.output + result.output
    assert "- 1 students, 1 pages in all;" in " ".join(dry.output.split())
    assert WS.ingest_summary_json.read_bytes() == summary
    assert sorted(WS.students_dir.iterdir()) == folders
    assert out_files(WS) == before


def test_a_dry_run_of_run_on_a_new_workspace_lists_models_and_spends_nothing(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fetch_catalog(api_key: str) -> Catalog:
        entry = {
            "id": MAPPER.slug,
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["structured_outputs"],
            "pricing": {"prompt": "0.000001", "completion": "0.000002"},
        }
        return Catalog({MAPPER.slug: entry}, ZDR)

    monkeypatch.setattr("redpen.cli.fetch_catalog", fetch_catalog)
    toml = "".join(f'{role} = "{MAPPER.slug}"\n' for role in ("parser", "solver"))
    Path("redpen.toml").write_text(
        "[models]\n" + toml + "".join(f'{role} = "{MAPPER.slug}"\n' for role in ROLES)
    )
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")
    args = ["run", "blank.pdf", "subs", "--workspace", "new", "--dry-run"]

    result = runner.invoke(app, args)

    assert result.exit_code == 0, result.output
    assert router.requests == []
    new = Workspace(Path("new"))
    assert not new.config_lock_toml.exists() and not (new.root / "key").exists()
    assert not (new.root / "out").exists()
    assert "configured, not pinned yet" in result.output
    assert any(line.startswith("- No spec yet:") for line in result.output.splitlines())


def test_the_lock_holds_the_version_of_every_prompt_file(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    folder = Path(redpen.prompts.__file__).parent
    files = sorted(folder.rglob("*.md"))  # every prompt file on disk

    assert grade().exit_code == 0

    lock = read_lock(WS.config_lock_toml)
    assert files and lock.prompts == {
        f.relative_to(folder).with_suffix("").as_posix(): hashlib.sha256(
            f.read_bytes()
        ).hexdigest()[:12]
        for f in files
    }
    assert lock.models == ROLES


STRICT_OFF = (
    "strict_privacy is off: the endpoints are not checked, and student pages may go"
    " to providers that keep them."
)


@pytest.mark.parametrize("strict", [True, False])
def test_strict_privacy_switched_off_is_said_and_recorded(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, strict: bool
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    if not strict:
        Path("redpen.toml").write_text("strict_privacy = false\n")

    result = grade()

    assert result.exit_code == 0, result.output
    assert (STRICT_OFF in result.stderr.splitlines()) is not strict
    assert read_lock(WS.config_lock_toml).strict_privacy is strict
    report = WS.run_report_md.read_text()
    assert (STRICT_OFF_RUN in report.splitlines()) is not strict
    if strict:  # the run report as it was before the setting was recorded
        assert "strict" not in report.lower()


def test_the_shape_per_page_is_the_fixture_assignment_s() -> None:
    found, pages = shape(SPEC), 4
    assert {f.name: getattr(found, f.name) for f in fields(found)} == {
        f.name: getattr(PER_PAGE, f.name) * pages for f in fields(PER_PAGE)
    }
    # So without a spec, the grade stages are estimated as with it.
    slides = [
        IngestPage(
            number=n, width_px=1004, height_px=2008, slide_number=n, native_ppi=None
        )
        for n in range(1, pages + 1)
    ]
    student = IngestRecord(
        student_name="A", student_key="A", source_kind="classkick", pages=slides
    )
    models = {**ROLES, "parser": READER, "solver": READER}
    blank = [(1700, 2200)] * pages
    with_spec = estimate(models, blank, SPEC, False, [student], 1536)
    without = estimate(models, blank, None, False, [student], 1536)
    for stage in ("map", "read", "grade"):
        assert with_spec[stage] == without[stage]
    assert with_spec["prepare"].usd < without["prepare"].usd  # parse is still to do


def test_an_image_costs_a_token_per_patch_once_fit_to_the_budget() -> None:
    # As the fixture's calls were billed: a letter page at 200 DPI and a slide.
    assert image_tokens((1700, 2200), 1536) == 2365
    assert image_tokens((1004, 2008), 1536) == 1540
    assert image_tokens((200, 100), 1536) == 8 * 4
