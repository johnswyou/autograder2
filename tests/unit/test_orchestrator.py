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
import re
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from reportlab.pdfgen import canvas
from typer.testing import CliRunner

import redpen.prompts
from redpen.cli import app
from redpen.config import Lock, read_lock, write_lock
from redpen.grade.grade import Grades
from redpen.grade.ingest import IngestSummary
from redpen.llm.client import Gateway
from redpen.llm.models import Catalog, recheck_providers
from redpen.orchestrator import PER_PAGE, TOKENS, estimate, image_tokens, shape
from redpen.prepare.key import load_key, part_hash, save_key
from redpen.report.report import OTHER_READING, STRICT_OFF_RUN, UNGRADED
from redpen.schemas.common import Provenance
from redpen.schemas.grade import CriterionGrade, Grade, GradeRecord, ProblemGrades
from redpen.schemas.key import Key
from redpen.schemas.reading import FinalAnswer, ProblemReadings, Reading, ReadingRecord
from redpen.schemas.spec import Spec
from redpen.schemas.submission import (
    IngestPage,
    IngestRecord,
    Map,
    MapEntry,
    MapPage,
    MapRecord,
)
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY
from tests.grader import grades
from tests.pinned import (
    GRADE_MODELS,
    GRADER,
    KEY_MODELS,
    MAPPER,
    READER,
    SECOND,
    ZDR,
)

runner = CliRunner()
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
SPEC = read_model(FIXTURES / "spec.json", Spec)
NAMES = ("Ann Example", "Ben Example", "Cal Example")
ROLES = GRADE_MODELS


class OpenRouter:
    """A mock OpenRouter that keeps every request: the ZDR listing answers as
    `zdr` says; with `reads`, a read request gets a valid reading of each part it
    names, handwritten with its line marked uncertain when `unsure`; with
    `decides`, the class audit's request gets full points for every case; with
    `grades`, a grade request gets what it gives; every other chat request is
    refused with HTTP 400."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.zdr: Callable[[httpx.Request], httpx.Response] = lambda _: httpx.Response(
            200, json={"data": ZDR}
        )
        self.reads = False
        self.unsure = False
        self.decides = False
        self.grades: Callable[[dict[str, Any]], Grades] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/endpoints/zdr"):
            return self.zdr(request)
        body = json.loads(request.content)
        if self.reads and kind(body) == "Readings":
            # The second reader is sure of what the reader was unsure of.
            unsure = self.unsure and body["model"] != SECOND.slug
            return httpx.Response(200, json=answer(body, readings(body, unsure)))
        if self.decides and kind(body) == "Decisions":
            cases = json.loads(body["messages"][1]["content"][2]["text"])["cases"]
            found = [{"case": c["case"], "reason": "r", "awarded": 1} for c in cases]
            return httpx.Response(200, json=answer(body, {"decisions": found}))
        if self.grades and kind(body) == "Grades":
            found = self.grades(body).model_dump(mode="json")
            return httpx.Response(200, json=answer(body, found))
        return httpx.Response(400, json={"error": {"message": "refused by the test"}})

    @property
    def chats(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/chat/completions")]


def kind(body: dict[str, Any]) -> str:
    """The response model a chat request asks for."""
    return body["response_format"]["json_schema"]["name"]


def readings(body: dict[str, Any], unsure: bool = False) -> dict[str, Any]:
    """A reading of each part in a read request's outline (its first text part):
    one typed line, and the first printed option of a choice part; or, `unsure`,
    the line handwritten and marked uncertain."""
    outline = json.loads(body["messages"][1]["content"][0]["text"])
    return {
        "readings": [
            {
                "part_id": part["id"],
                "status": "answered",
                "medium": "handwritten" if unsure else "typed",
                "lines": ["x = 4"],
                "final_answer": {
                    "choices": part["options"][:1],
                    "value": None,
                    "unit": None,
                    "expression": None,
                },
                "elements": [],
                "uncertain": [1] if unsure else [],
                "addressed_to_grader": [],
                "legibility": "clear",
            }
            for part in outline["parts"]
        ]
    }


def answer(body: dict[str, Any], content: dict[str, Any]) -> dict[str, Any]:
    """An OpenRouter reply to `body` carrying `content`."""
    return {
        "id": "test",
        "model": body["model"],
        "choices": [
            {
                "message": {"role": "assistant", "content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
    }


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
    # The grade stage finds no reading of Ben's to grade: nothing more fails.
    assert [line for line in result.stderr.splitlines() if "failed" in line] == [
        f"{stage}: 1 of 3 students failed." for stage in ("map", "read")
    ]
    assert "Ben" not in result.output + result.stderr
    entries = WS.run_log.read_text().split("--- ")[1:]
    assert [e.split("\n")[0].split()[1:] for e in entries] == [
        [stage, next(s for s in students() if s.startswith("Ben"))]
        for stage in ("map", "read")
    ]
    assert all("Traceback (most recent call last)" in e for e in entries)
    assert "RuntimeError: the map broke" in entries[0]
    assert "FileNotFoundError" in entries[1] and "map.json" in entries[1]
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
    # One entry for each of Ben's parts, on the terminal as in the run report.
    review = f"Entries in the review list, out/review.md: {len(KEY.parts)}."
    assert review in result.stdout.splitlines()
    assert review in WS.run_report_md.read_text().splitlines()


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
    assert result.stderr.splitlines()[-4:] == [  # after the progress lines
        *(f"{stage}: 1 of 3 students failed." for stage in ("map", "read")),
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
    assert restricted(router.chats) == {"mapper"}
    assert result.exit_code == 1  # every chat request is refused
    assert "map: 3 of 3 students failed." in result.stderr.splitlines()
    assert (WS.root / "out" / "run_report.md").exists()


ROLE = {"Inventory": "mapper", "Resolution": "mapper", "Readings": "reader"}
ROLE |= {"Grades": "grader"}


def role(body: dict[str, Any]) -> str:
    """The role a chat request is made for: by the response model it asks for, and
    a reading by the model it goes to, the reader's or the second reader's."""
    if kind(body) == "Readings" and body["model"] == ROLES["second"].slug:
        return "second"
    return ROLE[kind(body)]


def restricted(chats: list[httpx.Request]) -> set[str]:
    """The roles of `chats`, once each request is seen to carry a page and to go
    to the lock's model for its role, on the endpoints that keep nothing and
    nowhere else."""
    assert chats
    for request in chats:
        body = json.loads(request.content)
        model = ROLES[role(body)]
        assert b"image_url" in request.content
        assert body["model"] == model.slug
        assert body["provider"] == {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": sorted(model.providers),
        }
    return {role(json.loads(r.content)) for r in chats}


async def mapped_everywhere(
    workspace: Workspace, student: str, spec: Spec, *_: object
) -> None:
    """A map stage that finds every part's work in one box on page 1."""
    entries = [
        MapEntry(
            part_id=p.id,
            page=1,
            box=[0.1, 0.1, 0.6, 0.3],
            basis="position",
            role="work",
            confidence="high",
        )
        for p in spec.parts
    ]
    found = Map(entries=entries, unanswered=[], set_aside=[])
    page = MapPage(number=1, rotation=0, photo=False)
    write_model(
        workspace.map_json(student),
        MapRecord(provenance=NOBODY, pages=[page], map=found),
    )


def test_with_the_privacy_check_passing_read_and_grade_send_crops_restricted(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    router.reads = True
    transport = httpx.MockTransport(router)
    monkeypatch.setattr(
        "redpen.orchestrator.Gateway",
        functools.partial(Gateway, transport=transport, requests_per_second=1e6),
    )
    monkeypatch.setattr("redpen.orchestrator.map_student", mapped_everywhere)

    result = grade()

    assert restricted(router.chats) == {"reader", "grader"}
    per_role = Counter(role(json.loads(r.content)) for r in router.chats)
    # Three read units, each student's the same pages, so later ones may come from
    # the cache; two problems go to the grader for each student, and are refused.
    assert per_role["reader"] >= 3 and per_role["grader"] == 3 * 2
    assert result.exit_code == 1  # the grader's requests are refused
    assert "grade: 3 of 3 students failed." in result.stderr.splitlines()
    assert "read:" not in result.stderr


def fast(router: OpenRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every part mapped in one box, and the gateway on `router` with no wait."""
    transport = httpx.MockTransport(router)
    monkeypatch.setattr(
        "redpen.orchestrator.Gateway",
        functools.partial(Gateway, transport=transport, requests_per_second=1e6),
    )
    monkeypatch.setattr("redpen.orchestrator.map_student", mapped_everywhere)


def test_a_part_in_doubt_is_read_again_by_the_second_model_as_the_reader_was_asked(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    router.reads = router.unsure = True
    fast(router, monkeypatch)

    result = grade()

    assert restricted(router.chats) == {"reader", "second", "grader"}
    asked = [json.loads(r.content) for r in router.chats]
    reads = [b for b in asked if role(b) == "reader"]
    again = [b for b in asked if role(b) == "second"]

    def sent(body: dict[str, Any]) -> str:  # all but the model and its endpoints
        return json.dumps({k: v for k, v in body.items() if k not in WHERE})

    # Every part is in doubt, so each unit is read again with the read stage's own
    # request: its system text, parts, images, parameters and session.
    assert again and {sent(b) for b in again} == {sent(b) for b in reads}
    # After the class's reading and before its grading.
    order = ["reader", "second", "grader"]
    roles = [role(b) for b in asked]
    assert roles == sorted(roles, key=order.index)
    for request in router.chats:
        for text in [*NAMES, *students()]:
            assert text.encode() not in request.content
    assert "second:" not in result.stderr  # the grader's requests are refused
    report = WS.run_report_md.read_text()
    parts = len(KEY.parts) * len(NAMES)
    assert (
        "Parts the reader was unsure of, read a second time by another vendor's"
        f" model: {parts}. The same: {parts}. Different: 0. Not decided: 0. Failed:"
        " 0."
    ) in report.splitlines()
    assert "hard_to_read" not in report
    cost = report.split("## Cost")[1].splitlines()
    assert any(re.fullmatch(r"\| second \| \d+ \| \d+ \| [\d.]+ \|", r) for r in cost)


WHERE = ("model", "provider")


def refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"a request was made: {request.url}")


def refusing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every gateway and the privacy check on a transport that fails any request."""
    transport = httpx.MockTransport(refuse)
    for where in ("redpen.orchestrator.Gateway", "redpen.cli.Gateway"):
        monkeypatch.setattr(where, functools.partial(Gateway, transport=transport))
    monkeypatch.setattr(
        "redpen.orchestrator.recheck_providers",
        functools.partial(recheck_providers, transport=transport),
    )


# A second reader of the reader's vendor, but another model of it.
SAME_VENDOR = (
    f"ws/config.lock.toml: the second reader {MAPPER.slug} is of the reader's"
    f" vendor, so it cannot check the reader {READER.slug}; pin a model of another"
    " vendor."
)


def test_a_second_reader_of_the_reader_s_vendor_is_refused_before_anything_is_sent(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    refusing(monkeypatch)
    write_lock(
        WS.config_lock_toml,
        Lock(profile="balanced", models={**ROLES, "second": MAPPER}),
    )

    result = grade()

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert result.stderr.splitlines() == [SAME_VENDOR]
    assert not WS.students_dir.exists()


@pytest.mark.parametrize("command", ["run", "prepare"])
def test_a_fresh_key_is_not_paid_for_when_its_lock_could_never_grade(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    refusing(monkeypatch)
    WS.key_yaml.unlink()  # so that prepare would solve and verify a key
    models = {**ROLES, **KEY_MODELS, "second": MAPPER}
    write_lock(WS.config_lock_toml, Lock(profile="balanced", models=models))
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")
    args = ["blank.pdf", *(["subs"] if command == "run" else []), "--workspace", "ws"]

    result = runner.invoke(app, [command, *args])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # not the transport's assertion
    assert result.stderr.splitlines() == [SAME_VENDOR]
    assert router.requests == [] and not WS.key_yaml.exists()


@pytest.mark.parametrize("command", ["grade", "run"])
def test_a_dry_run_says_its_second_reader_would_be_refused(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    refusing(monkeypatch)
    write_lock(
        WS.config_lock_toml,
        Lock(profile="balanced", models={**ROLES, "second": MAPPER}),
    )
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")
    blank = ["blank.pdf"] if command == "run" else []

    result = runner.invoke(
        app, [command, *blank, "subs", "--workspace", "ws", "--dry-run"]
    )

    assert result.exit_code == 0, result.output
    assert result.stderr.splitlines() == [SAME_VENDOR]
    assert "Estimate, with nothing sent to a model:" in result.stdout


@pytest.mark.parametrize("how", ["not pinned", "its endpoints are not listed"])
def test_the_privacy_gate_covers_the_second_reader(
    router: OpenRouter, tmp_path: Path, how: str
) -> None:
    if how == "not pinned":
        shutil.rmtree("ws")
        models = {r: m for r, m in ROLES.items() if r != "second"}
        prepared(tmp_path, Lock(profile="balanced", models=models))
        line = "second: no model is pinned in the lock"
    else:
        listed = [z for z in ZDR if z["model_id"] != SECOND.slug]
        router.zdr = lambda _: httpx.Response(200, json={"data": listed})
        line = f"second: {SECOND.slug} pinned endpoint azure is no longer zero data"
        line += " retention"

    result = grade()

    assert result.exit_code == 1
    assert line in result.stderr.splitlines()
    assert router.chats == []


@pytest.mark.parametrize("how", ["dry run", "gate fails"])
def test_no_part_is_read_again_under_a_dry_run_or_past_a_failed_gate(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    router.reads = router.unsure = True
    fast(router, monkeypatch)
    grade()  # stores readings in doubt, and reads them again
    assert any(role(json.loads(r.content)) == "second" for r in router.chats)
    WS.cache_sqlite.unlink()  # so that no answer could come from the cache
    router.requests.clear()
    if how == "gate fails":
        router.zdr = lambda _: httpx.Response(500, json={"error": {"message": "down"}})
    args = ["grade", "subs", "--workspace", "ws"]

    result = runner.invoke(app, [*args, "--dry-run"] if how == "dry run" else args)

    assert result.exit_code == (0 if how == "dry run" else 1), result.output
    assert router.chats == []


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


def test_a_dry_run_with_a_lock_that_cannot_verify_says_so_in_one_line(
    router: OpenRouter,
) -> None:
    WS.key_yaml.unlink()  # so the estimate prices a solve and its verification
    write_lock(
        WS.config_lock_toml,
        Lock(profile="balanced", models={**ROLES, "solver": READER}),
    )
    shutil.copyfile(FIXTURES / "blank.pdf", "blank.pdf")

    result = runner.invoke(
        app, ["run", "blank.pdf", "subs", "--workspace", "ws", "--dry-run"]
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)  # not a KeyError
    assert result.stderr.splitlines() == [
        "ws/config.lock.toml: no model is pinned for checker, referee."
    ]
    assert router.requests == []


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
    prepare = ("parser", "solver", "checker", "referee")
    toml = "".join(f'{role} = "{MAPPER.slug}"\n' for role in prepare)
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


@pytest.mark.parametrize("gate_passes", [False, True], ids=["stopped", "passed"])
def test_strict_privacy_once_off_stays_recorded(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, gate_passes: bool
) -> None:
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    Path("redpen.toml").write_text("strict_privacy = false\n")
    assert grade().exit_code == 0
    Path("redpen.toml").unlink()  # strict privacy on again
    lock = WS.config_lock_toml.read_bytes()
    if not gate_passes:
        router.zdr = lambda _: httpx.Response(500, json={"error": {"message": "down"}})

    again = grade()
    report = runner.invoke(app, ["report", "--workspace", "ws"])

    assert again.exit_code == (0 if gate_passes else 1), again.output
    if not gate_passes:  # nothing sent, and the lock as the earlier run left it
        assert "Nothing was sent" in again.stderr.splitlines()[-1]
        assert WS.config_lock_toml.read_bytes() == lock
    assert read_lock(WS.config_lock_toml).strict_privacy is False
    assert report.exit_code == 0, report.output
    assert STRICT_OFF_RUN in WS.run_report_md.read_text().splitlines()


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
    models = {
        **ROLES,
        **dict.fromkeys(("parser", "solver", "checker", "referee"), READER),
    }
    blank = [(1700, 2200)] * pages
    with_spec = estimate(models, blank, SPEC, False, [student], 1536)
    without = estimate(models, blank, None, False, [student], 1536)
    for stage in ("map", "read", "grade"):
        assert with_spec[stage] == without[stage]
    assert with_spec["prepare"].usd < without["prepare"].usd  # parse is still to do


def test_the_estimate_counts_a_checker_call_per_problem_and_the_referee_s() -> None:
    models = {
        **ROLES,
        **dict.fromkeys(("parser", "solver", "checker", "referee"), READER),
    }
    blank = [(1700, 2200)] * 4

    costs = estimate(models, blank, SPEC, False, [], 1536)

    # Five problems solved twice, two rubrics, a referee for Q4 and Q5, which have
    # a part code cannot compare (a drawing, an explanation, a graph), and the
    # self-test's grader call for each problem with a part a model grades.
    assert costs["prepare"].calls == 5 + 5 + 2 + 2 + 2
    dearer = GRADER.model_copy(update={"prompt_price": GRADER.prompt_price + 1})
    priced = estimate({**models, "grader": dearer}, blank, SPEC, False, [], 1536)
    assert priced["prepare"].usd - costs["prepare"].usd == 2 * TOKENS["test"][0]
    assert (shape(SPEC).referees, shape(SPEC).referee_pages) == (2, 4)
    assert estimate(models, blank, SPEC, True, [], 1536)["prepare"].calls == 0


def test_an_image_costs_a_token_per_patch_once_fit_to_the_budget() -> None:
    # As the fixture's calls were billed: a letter page at 200 DPI and a slide.
    assert image_tokens((1700, 2200), 1536) == 2365
    assert image_tokens((1004, 2008), 1536) == 1540
    assert image_tokens((200, 100), 1536) == 8 * 4


@pytest.mark.parametrize("gate", ["passes", "fails"])
def test_the_class_audit_runs_in_grade_past_the_gate_on_the_grader_s_endpoints(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    # Every student wrote the same 4(c), and the grader gave one of them 0 for c1.
    async def nothing(*args: object) -> None:
        pass

    async def graded(
        workspace: Workspace, student: str, key: Key, gateway: Gateway, model: Any
    ) -> None:
        part = next(p for p in key.parts if p.id == "Q4.c")
        line = ["It slows down and stops."]
        reading = Reading(
            part_id=part.id,
            status="answered",
            medium="typed",
            lines=line,
            final_answer=FinalAnswer(
                choices=[], value=None, unit=None, expression=None
            ),
            elements=[],
            uncertain=[],
            addressed_to_grader=[],
            legibility="clear",
        )
        record = ReadingRecord(provenance=NOBODY, reading=reading, crops=[])
        write_model(
            workspace.reading_json(student, "Q4"), ProblemReadings(records=[record])
        )
        award = 0 if student == students()[0] else 1
        criteria = [
            CriterionGrade(
                id=c.id,
                rationale="r",
                evidence_line=1,
                evidence_element=None,
                awarded=Decimal(award if c.id == "Q4.c.c1" else 1),
                feedback="f",
            )
            for c in part.criteria
        ]
        grade_ = Grade(
            part_id=part.id,
            criteria=criteria,
            transcript_dispute=None,
            belongs_elsewhere=None,
            code_overruled=None,
        )
        provenance = Provenance(
            key_part_hash=part_hash(part, key.policies),
            prompt_version="v",
            model_slug=model.slug,
        )
        write_model(
            workspace.grade_json(student, "Q4"),
            ProblemGrades(
                records=[
                    GradeRecord(
                        provenance=provenance,
                        grade=grade_,
                        check=None,
                        reading_digest=None,
                    )
                ]
            ),
        )

    for stage in ("map", "read"):
        monkeypatch.setattr(f"redpen.orchestrator.{stage}_student", nothing)
    monkeypatch.setattr("redpen.orchestrator.grade_student", graded)
    router.decides = True
    if gate == "fails":
        router.zdr = lambda _: httpx.Response(500, json={"error": {"message": "down"}})

    result = grade()

    if gate == "fails":
        assert result.exit_code == 1 and router.chats == []
        assert not (WS.root / "audit").exists()
        return
    assert result.exit_code == 0, result.output
    [request] = router.chats
    body = json.loads(request.content)
    assert kind(body) == "Decisions"
    assert body["model"] == GRADER.slug
    assert body["provider"] == {
        "require_parameters": True,
        "data_collection": "deny",
        "zdr": True,
        "only": sorted(GRADER.providers),
    }
    assert b"image_url" not in request.content
    for text in [*NAMES, *students()]:
        assert text.encode() not in request.content
    with WS.scores_detail_csv.open(newline="", encoding="utf-8-sig") as file:
        rows = [r for r in csv.DictReader(file) if r["criterion"] == "Q4.c.c1"]
    assert [r["awarded"] for r in rows] == ["1"] * len(NAMES)


@pytest.mark.parametrize("how", ["dry run", "gate fails", "dies before the audit"])
def test_an_earlier_audit_is_removed_only_once_pages_may_be_sent(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    async def nothing(*args: object) -> None:
        pass

    async def dies(*args: object) -> None:
        raise RuntimeError("the run died between grading and the audit")

    for stage in ("map", "read", "grade"):
        monkeypatch.setattr(f"redpen.orchestrator.{stage}_student", nothing)
    monkeypatch.setattr("redpen.orchestrator.audit_class", dies)
    if how == "gate fails":
        router.zdr = lambda _: httpx.Response(500, json={"error": {"message": "down"}})
    earlier = WS.audit_json("Q4.c")
    earlier.parent.mkdir(parents=True)
    earlier.write_text("{}")

    args = ["grade", "subs", "--workspace", "ws"]
    result = runner.invoke(app, [*args, "--dry-run"] if how == "dry run" else args)

    assert result.exit_code == (0 if how == "dry run" else 1)
    assert earlier.exists() == (how != "dies before the audit")


# --- redpen regrade ------------------------------------------------------------------


def regrade(*options: str) -> Any:
    return runner.invoke(app, ["regrade", "--workspace", "ws", *options])


def graded_once(router: OpenRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    """The class mapped, read and graded with every part earned in full; then the
    router's requests forgotten and reads refused. Each student's page is their
    own, so that no student's grade request is another's."""
    for n, name in enumerate(NAMES, 1):
        pdf = canvas.Canvas(str(Path("subs") / f"HW 1 - {name}.pdf"))
        pdf.drawString(100, 650, f"{n}. x = {n}")
        pdf.save()
    router.reads, router.grades = True, grades
    fast(router, monkeypatch)
    assert grade().exit_code == 0
    crops = {WS.crop_png(s, "Q4.b", 1).read_bytes() for s in students()}
    assert len(crops) == len(NAMES)
    router.reads = False
    router.requests.clear()


def edit_criterion(part_id: str) -> None:
    """Edit the requirement of the first criterion of `part_id` in key.yaml."""
    key = load_key(WS.key_yaml)
    parts = [
        p.model_copy(
            update={
                "criteria": [
                    p.criteria[0].model_copy(update={"requirement": "Edited."}),
                    *p.criteria[1:],
                ]
            }
        )
        if p.id == part_id
        else p
        for p in key.parts
    ]
    save_key(WS.key_yaml, key.model_copy(update={"parts": parts}))


def grade_files() -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in WS.students_dir.glob("*/grades/*.json")}


def grade_requests(router: OpenRouter) -> list[list[str]]:
    """The part ids each chat request asks to grade, every one a grade request."""
    found: list[list[str]] = []
    for request in router.chats:
        body = json.loads(request.content)
        assert kind(body) == "Grades"
        content = body["messages"][1]["content"]
        texts = [json.loads(c["text"]) for c in content if c["type"] == "text"]
        found.append([t["grade"]["part_id"] for t in texts if "grade" in t])
    return sorted(found)


Q4_PARTS = ["Q4.a", "Q4.b", "Q4.c"]


def test_regrade_asks_only_for_the_problem_whose_criterion_was_edited(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graded_once(router, monkeypatch)
    before = grade_files()
    shutil.rmtree(WS.root / "out")
    edit_criterion("Q4.b")

    result = regrade()

    assert result.exit_code == 0, result.output
    assert grade_requests(router) == [Q4_PARTS] * len(NAMES)
    others = [r.url.path for r in router.requests if r not in router.chats]
    assert others == ["/api/v1/endpoints/zdr"]
    after = grade_files()
    assert sorted(after) == sorted(before)
    assert {p for p in after if after[p] != before[p]} == {
        p for p in before if p.name == "Q4.json"
    }
    assert {p.name for p in out_files(WS)} >= {"scores_detail.csv", "review.md"}
    assert result.stdout.splitlines()[-2] == f"Wrote {WS.root / 'out'}"


def test_regrade_with_nothing_changed_asks_nothing_and_writes_the_same(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graded_once(router, monkeypatch)
    files, out = grade_files(), out_files(WS)

    dry = regrade("--dry-run")
    result = regrade()

    assert dry.stdout == "Grade requests the call cache would not answer: 0.\n"
    assert result.exit_code == 0, result.output
    assert router.chats == []
    assert grade_files() == files
    assert out_files(WS) == out


@pytest.mark.parametrize("how", ["gate fails", "key with problems", "dry run"])
def test_regrade_sends_nothing_past_a_failed_gate_or_a_bad_key_or_in_a_dry_run(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    graded_once(router, monkeypatch)
    edit_criterion("Q4.b")  # so that each student's Q4 would be asked again
    # A file stored before the reading's digest was kept, which any grading of it
    # would write again.
    old = WS.grade_json(students()[0], "Q1")
    old.write_text(re.sub(r',\s*"reading_digest": "\w+"', "", old.read_text()))
    files, out = grade_files(), out_files(WS)
    cache = WS.cache_sqlite.read_bytes()
    if how == "gate fails":
        router.zdr = lambda _: httpx.Response(500, json={"error": {"message": "down"}})
    if how == "key with problems":
        text = WS.key_yaml.read_text()
        WS.key_yaml.write_text(text.replace("points: 2", "points: 2.1", 1))

    result = regrade("--dry-run") if how == "dry run" else regrade()

    assert router.chats == []
    assert grade_files() == files and out_files(WS) == out
    if how == "dry run":
        assert result.exit_code == 0, result.output
        assert router.requests == []
        assert result.stdout.splitlines() == [
            f"Grade requests the call cache would not answer: {len(NAMES)}."
        ]
        assert WS.cache_sqlite.read_bytes() == cache
        assert not WS.run_log.exists()
        return
    quiet(result)
    last = result.stderr.splitlines()[-1]
    if how == "gate fails":
        assert last.startswith("Nothing was sent")
    else:
        assert last == f"{WS.key_yaml} has problems, so nothing was graded."


def test_a_student_with_no_stored_reading_of_a_problem_is_skipped_and_named(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graded_once(router, monkeypatch)
    first = students()[0]
    WS.reading_json(first, "Q4").unlink()
    kept = WS.grade_json(first, "Q4").read_bytes()
    edit_criterion("Q4.b")

    result = regrade()

    assert result.exit_code == 0, result.output
    assert grade_requests(router) == [Q4_PARTS] * (len(NAMES) - 1)
    assert WS.grade_json(first, "Q4").read_bytes() == kept
    assert (
        "Problems not graded, with no stored reading: 1, of 1 students."
        f" {WS.run_report_md} names them."
    ) in result.stderr.splitlines()
    assert "Example" not in result.output + result.stderr
    named = f"- `{first}`: Q4.a, Q4.b, Q4.c. {UNGRADED}"
    assert named in WS.run_report_md.read_text().splitlines()


def test_a_problem_whose_new_grade_fails_keeps_its_file_and_is_logged_and_counted(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graded_once(router, monkeypatch)
    first = students()[0]
    # The first student's Q4.a and Q5.a read again, differently.
    for problem in ("Q4", "Q5"):
        path = WS.reading_json(first, problem)
        records = read_model(path, ProblemReadings).records
        read = records[0].reading.model_copy(update={"lines": ["y = 5"]})
        records[0] = records[0].model_copy(update={"reading": read})
        write_model(path, ProblemReadings(records=records))
    q4, q5 = (WS.grade_json(first, q).read_bytes() for q in ("Q4", "Q5"))

    def no_q4c_for_the_new_reading(body: dict[str, Any]) -> Grades:
        found = grades(body)
        if "y = 5" not in json.dumps(body):
            return found
        return Grades(grades=[g for g in found.grades if g.part_id != "Q4.c"])

    router.grades = no_q4c_for_the_new_reading

    result = regrade()

    assert result.exit_code == 1
    asked = ["Q5.a", "Q5.b", "Q5.c"]
    assert grade_requests(router) == sorted([Q4_PARTS, Q4_PARTS, asked])
    assert WS.grade_json(first, "Q4").read_bytes() == q4
    assert WS.grade_json(first, "Q5").read_bytes() != q5
    assert "grade: 1 of 3 students failed." in result.stderr.splitlines()
    [entry] = WS.run_log.read_text().split("--- ")[1:]
    assert entry.split("\n")[0].split()[1:] == ["grade", first]
    assert "StageError" in entry and "y = 5" not in entry
    with WS.scores_detail_csv.open(newline="", encoding="utf-8-sig") as file:
        mine = [r for r in csv.DictReader(file) if r["student"] == NAMES[0]]
    shown = {r["part"]: r["rationale"] for r in mine}
    assert shown["Q4.a"] == OTHER_READING
    assert shown["Q4.b"] != OTHER_READING


# --- redpen override -----------------------------------------------------------------


def override(*args: str) -> Any:
    return runner.invoke(app, ["override", *args, "--workspace", "ws"])


def awarded(criterion: str) -> dict[str, str]:
    """Each student's award on `criterion` in scores_detail.csv, by name."""
    with WS.scores_detail_csv.open(newline="", encoding="utf-8-sig") as file:
        found = [r for r in csv.DictReader(file) if r["criterion"] == criterion]
    return {r["student"]: r["awarded"] for r in found}


def test_a_correction_survives_regrade_and_grade_and_its_removal_restores_the_award(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    graded_once(router, monkeypatch)
    first = students()[0]

    recorded = override(first, "Q4.b.c1", "0.5", "--note", MARKER)

    assert recorded.exit_code == 0, recorded.output
    assert recorded.stdout.splitlines()[0] == f"Recorded in {WS.overrides_yaml}"
    assert awarded("Q4.b.c1") == {NAMES[0]: "0.5", NAMES[1]: "2", NAMES[2]: "2"}
    assert MARKER in WS.feedback_md(first).read_text()
    assert router.chats == []
    edit_criterion("Q5.a")  # so that the regrade asks the grader again
    again = regrade()
    assert again.exit_code == 0 and len(router.chats) == len(NAMES)
    assert awarded("Q4.b.c1")[NAMES[0]] == "0.5"
    router.reads = True
    assert grade().exit_code == 0
    assert awarded("Q4.b.c1")[NAMES[0]] == "0.5"

    removed = override(first, "Q4.b.c1", "--remove")

    assert removed.exit_code == 0, removed.output
    assert removed.stdout.splitlines()[0] == f"Removed from {WS.overrides_yaml}"
    assert awarded("Q4.b.c1")[NAMES[0]] == "2"
    assert MARKER not in WS.feedback_md(first).read_text()
    for result in (recorded, again, removed):
        assert "MARKER" not in result.output + result.stderr
    assert not WS.run_log.exists()


def ungraded_class(router: OpenRouter, monkeypatch: pytest.MonkeyPatch) -> None:
    """The class graded with nothing found to read: its files, and no request."""
    monkeypatch.setattr("redpen.orchestrator.map_student", unanswered_map())
    assert grade().exit_code == 0
    router.requests.clear()


@pytest.mark.parametrize(
    ("args", "line"),
    [
        (["nobody-000000", "Q4.b.c1", "1"], "No student of the latest ingest has that"),
        (["{first}", "Q9.c1", "1"], "The key has no criterion with that id."),
        (["{first}", "Q4.b.c1", "-1"], "POINTS is a number from 0 to 2,"),
        (["{first}", "Q4.b.c1", "2.25"], "POINTS is a number from 0 to 2,"),
        (["{first}", "Q4.b.c1", "two"], "POINTS is a number from 0 to 2,"),
        (["{first}", "Q4.b.c1"], "POINTS is a number from 0 to 2,"),
        (["{first}", "Q4.b.c1", "--remove"], "No correction is recorded for that"),
        (["{first}", "Q4.b.c1", "1", "--remove"], "--remove takes no POINTS"),
    ],
    ids=[
        "unknown student",
        "unknown criterion",
        "below zero",
        "above full points",
        "not a number",
        "no points",
        "nothing to remove",
        "points to remove",
    ],
)
def test_a_correction_that_cannot_be_recorded_is_one_line_and_writes_nothing(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, args: list[str], line: str
) -> None:
    ungraded_class(router, monkeypatch)
    first = students()[0]
    assert override(first, "Q4.a.c1", "1").exit_code == 0
    kept, out = WS.overrides_yaml.read_bytes(), out_files(WS)

    result = override(*(a.replace("{first}", first) for a in args))

    quiet(result)
    [said] = result.stderr.splitlines()
    assert said.startswith(line)
    assert result.stdout == ""
    assert WS.overrides_yaml.read_bytes() == kept and out_files(WS) == out
    assert router.requests == []


BAD_FILES = {
    "not YAML": ("- [unclosed\n", "does not parse as YAML"),
    "not a correction": ("- student: x\n", ", entry 1 is not a correction"),
    "unknown student": (
        "- {{student: nobody-000000, criterion: Q4.b.c1, points: 1, part_hash: h}}\n",
        ", entry 1: no student of the latest ingest has its key.",
    ),
    "unknown criterion": (
        "- {{student: {first}, criterion: Q9.c1, points: 1, part_hash: h}}\n",
        ", entry 1: the key has no criterion Q9.c1.",
    ),
    "above full points": (
        "- {{student: {first}, criterion: Q4.b.c1, points: 3, part_hash: {hashed}}}\n",
        ", entry 1: its points are above the 2 that Q4.b.c1 is worth.",
    ),
}


COMMANDS = ("report", "regrade", "grade", "override")


@pytest.mark.parametrize(
    ("command", "bad"),
    [
        *((c, "unknown student") for c in COMMANDS),
        *((c, "not YAML") for c in COMMANDS),
        *(("report", b) for b in ("not a correction", "unknown criterion")),
        ("report", "above full points"),
    ],
)
def test_an_overrides_file_that_cannot_be_used_stops_each_command_with_one_line(
    router: OpenRouter, monkeypatch: pytest.MonkeyPatch, command: str, bad: str
) -> None:
    ungraded_class(router, monkeypatch)
    first = students()[0]
    hashed = part_hash(next(p for p in KEY.parts if p.id == "Q4.b"), KEY.policies)
    text, line = BAD_FILES[bad]
    WS.overrides_yaml.write_text(text.format(first=first, hashed=hashed))
    out, files = out_files(WS), grade_files()
    summary = WS.ingest_summary_json.read_bytes()
    args = {
        "report": ["report", "--workspace", "ws"],
        "regrade": ["regrade", "--workspace", "ws"],
        "grade": ["grade", "subs", "--workspace", "ws"],
        "override": ["override", first, "Q4.a.c1", "1", "--workspace", "ws"],
    }[command]

    result = runner.invoke(app, args)

    quiet(result)
    [said] = result.stderr.splitlines()
    assert said.startswith(str(WS.overrides_yaml)) and line in said
    assert router.requests == []
    assert out_files(WS) == out and grade_files() == files
    assert WS.ingest_summary_json.read_bytes() == summary
    assert WS.overrides_yaml.read_text() == text.format(first=first, hashed=hashed)
