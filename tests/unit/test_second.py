"""The second read (src/autograder2/grade/second.py): which parts are read again, the
comparison of two readings, a failure, and a run again, on a fixture student's
ingested pages and truth map, with readings stored by hand and a mock transport."""

import asyncio
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from autograder2.grade.ingest import IngestSummary
from autograder2.grade.read import Readings, digest
from autograder2.grade.second import compared, read_again
from autograder2.llm.client import Gateway, StopError
from autograder2.prepare.key import part_hash
from autograder2.report.report import flags, load, write_report
from autograder2.schemas.flags import FlagCode
from autograder2.schemas.key import Key
from autograder2.schemas.reading import (
    ProblemReadings,
    Reading,
    SecondReadings,
)
from autograder2.schemas.submission import IngestRecord
from autograder2.workspace import Workspace, read_model, write_model
from tests.fixture_truth import (
    FIXTURES,
    SPEC,
    SUBMISSIONS,
    map_of,
    store_truth_readings,
    truth_of,
)
from tests.pinned import SECOND

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
BUDGET = 1536


def stored(
    fixture_workspace: tuple[Workspace, IngestSummary],
    root: Path,
    doubt: dict[str, dict[str, Any]],
) -> tuple[Workspace, IngestRecord]:
    """A workspace holding the ink-in-place student, its truth map and its truth
    readings, each part in `doubt` changed as given, as the read stage stores them."""
    shared, summary = fixture_workspace
    [pdf] = [p for p in SUBMISSIONS if truth_of(p).layout_class == "ink-in-place"]
    truth = truth_of(pdf)
    [student] = [s for s in summary.students if s.student_name == truth.student_name]
    who = student.student_key
    workspace = Workspace(root / "ws")
    shutil.copytree(shared.student_dir(who), workspace.student_dir(who))
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "blank.pdf", workspace.blank_pdf)
    write_model(workspace.map_json(who), map_of(truth))
    store_truth_readings(workspace, who, truth)
    for problem in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        path = workspace.reading_json(who, problem)
        records = [
            r.model_copy(
                update={
                    "reading": r.reading.model_copy(update=doubt[r.reading.part_id])
                }
            )
            if r.reading.part_id in doubt
            else r
            for r in read_model(path, ProblemReadings).records
        ]
        write_model(path, ProblemReadings(records=records))
    return workspace, student


def firsts(workspace: Workspace, student: str) -> dict[str, Reading]:
    return {
        r.reading.part_id: r.reading
        for problem in ("Q1", "Q2", "Q3", "Q4", "Q5")
        for r in read_model(
            workspace.reading_json(student, problem), ProblemReadings
        ).records
    }


class Second:
    """A mock OpenRouter for the second reader: each request is answered with
    `answer(the outline's part ids)`, by default the stored first readings."""

    def __init__(self, answer: Callable[[list[str]], Readings | None]) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        ids = [p["id"] for p in json.loads(content[0]["text"])["parts"]]
        self.requests.append({"ids": ids, "json": body})
        value = self.answer(ids)
        if value is None:
            return httpx.Response(400, json={"error": {"message": "refused"}})
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": value.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )


def same_as(workspace: Workspace, student: str) -> Callable[[list[str]], Readings]:
    """The first reader's readings, its doubt included."""
    first = firsts(workspace, student)
    return lambda ids: Readings(readings=[first[i] for i in ids])


CLEAR = {"uncertain": [], "legibility": "clear"}


def clear_as(workspace: Workspace, student: str) -> Callable[[list[str]], Readings]:
    """What the first reader wrote, read with no doubt."""
    first = firsts(workspace, student)
    return lambda ids: Readings(
        readings=[first[i].model_copy(update=CLEAR) for i in ids]
    )


def again(workspace: Workspace, student: str, second: Second) -> list[Any]:
    async def main() -> list[Any]:
        async with Gateway(
            workspace.root.parent / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(second),
        ) as gateway:
            return await read_again(
                workspace, student, SPEC, KEY, gateway, SECOND, BUDGET
            )

    return asyncio.run(main())


def reading_files(workspace: Workspace, student: str) -> dict[Path, bytes]:
    folder = workspace.reading_json(student, "Q1").parent
    return {p: p.read_bytes() for p in folder.glob("Q*.json")}


# --- when a part is read again ------------------------------------------------------


@pytest.mark.parametrize(
    ("doubt", "asked"),
    [
        ({"uncertain": [1]}, True),
        ({"legibility": "poor"}, True),
        ({"medium": "typed", "uncertain": [1]}, False),
        ({}, False),
        ({"status": "blank", "lines": [], "legibility": "poor"}, False),
        ({"status": "unreadable", "lines": [], "legibility": "poor"}, False),
    ],
    ids=["uncertain line", "poor legibility", "typed", "clear", "blank", "unreadable"],
)
def test_a_part_is_read_again_only_when_the_reader_was_unsure_of_it(
    fixture_workspace: tuple[Workspace, IngestSummary],
    tmp_path: Path,
    doubt: dict[str, Any],
    asked: bool,
) -> None:
    workspace, student = stored(fixture_workspace, tmp_path, {"Q4.b": doubt})
    who = student.student_key
    second = Second(same_as(workspace, who))

    found = again(workspace, who, second)

    assert [r["ids"] for r in second.requests] == ([["Q4.a", "Q4.b", "Q4.c"]] * asked)
    # Every part of the unit is compared, and only Q4.b was in doubt.
    assert [(r.part_id, r.doubted) for r in found] == [
        ("Q4.a", False),
        ("Q4.b", True),
        ("Q4.c", False),
    ] * asked
    assert workspace.second_json(who).exists() == asked


def test_one_request_per_unit_however_many_of_its_parts_are_in_doubt(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    doubt = {"legibility": "poor"}  # the drawing has no line to mark
    workspace, student = stored(
        fixture_workspace, tmp_path, dict.fromkeys(["Q4.a", "Q4.b", "Q4.c"], doubt)
    )
    who = student.student_key
    before = reading_files(workspace, who)
    crop = workspace.crop_png(who, "Q4.b", 1)
    crop.write_bytes(b"marker")  # the second read writes no file of the read stage
    second = Second(clear_as(workspace, who))

    found = again(workspace, who, second)

    assert [r["ids"] for r in second.requests] == [["Q4.a", "Q4.b", "Q4.c"]]
    [request] = second.requests
    assert request["json"]["model"] == SECOND.slug
    assert [(r.part_id, r.doubted, r.result) for r in found] == [
        ("Q4.a", True, "same"),
        ("Q4.b", True, "same"),
        ("Q4.c", True, "same"),
    ]
    assert read_model(workspace.second_json(who), SecondReadings).records == found
    first = {
        r.reading.part_id: r
        for r in read_model(workspace.reading_json(who, "Q4"), ProblemReadings).records
    }
    for record in found:
        assert record.first == digest(first[record.part_id])
        assert record.part_hash == part_hash(PARTS[record.part_id], KEY.policies)
        assert record.second is not None
        assert record.second.provenance.model_slug == SECOND.slug
        assert record.second.crops == first[record.part_id].crops
    assert reading_files(workspace, who) == before  # the first reading is graded
    assert crop.read_bytes() == b"marker"


def test_a_second_run_makes_no_request_and_writes_the_same_file(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    workspace, student = stored(
        fixture_workspace, tmp_path, {"Q5.a": {"uncertain": [1]}}
    )
    who = student.student_key
    second = Second(same_as(workspace, who))
    again(workspace, who, second)
    written = workspace.second_json(who).read_bytes()

    again(workspace, who, second)

    assert len(second.requests) == 1
    assert workspace.second_json(who).read_bytes() == written


def test_a_run_that_finds_nothing_in_doubt_leaves_no_earlier_result(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    workspace, student = stored(
        fixture_workspace, tmp_path, {"Q5.a": {"uncertain": [1]}}
    )
    who = student.student_key
    second = Second(clear_as(workspace, who))
    again(workspace, who, second)
    assert workspace.second_json(who).exists()
    path = workspace.reading_json(who, "Q5")
    records = read_model(path, ProblemReadings).records
    cleared = [
        r.model_copy(update={"reading": r.reading.model_copy(update=CLEAR)})
        for r in records
    ]
    write_model(path, ProblemReadings(records=cleared))  # read again, with no doubt

    found = again(workspace, who, second)

    assert found == [] and len(second.requests) == 1
    assert not workspace.second_json(who).exists()


# --- a failure ----------------------------------------------------------------------


class Stopping(Second):
    """OpenRouter refusing the key: a failure no retry cures, which stops the run."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": 401}})


def test_a_stop_during_the_second_read_logs_no_unit(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path
) -> None:
    doubt = {"uncertain": [1]}
    workspace, student = stored(fixture_workspace, tmp_path, {"Q4.b": doubt})

    with pytest.raises(StopError):  # the command says it, and logs it once
        again(workspace, student.student_key, Stopping(lambda ids: None))

    assert not workspace.run_log.exists()


MARKER = "MARKER-2c9e written by a student"


@pytest.mark.parametrize("how", ["refused", "fails its checks twice"])
def test_a_failed_second_read_changes_no_flag_and_is_logged_and_counted(
    fixture_workspace: tuple[Workspace, IngestSummary], tmp_path: Path, how: str
) -> None:
    doubt = {"uncertain": [1]}
    workspace, student = stored(
        fixture_workspace, tmp_path, {"Q4.b": doubt, "Q5.a": doubt}
    )
    who = student.student_key
    first = firsts(workspace, who)
    before = reading_files(workspace, who)
    marked = first["Q4.a"].model_copy(update={"lines": [MARKER]})

    clear = clear_as(workspace, who)

    def answer(ids: list[str]) -> Readings | None:
        if ids[0] != "Q4.a":
            return clear(ids)
        if how == "refused":
            return None
        return Readings(readings=[marked, first["Q4.b"]])  # Q4.c is left out

    found = again(workspace, who, Second(answer))

    # The failed unit's part in doubt is counted; the other unit's parts compared.
    assert [(r.part_id, r.result, r.second is None) for r in found] == [
        ("Q4.b", "failed", True),
        ("Q5.a", "same", False),
        ("Q5.b", "same", False),
        ("Q5.c", "same", False),
    ]
    log = workspace.run_log.read_text()
    assert log.count("--- ") == 1
    assert f"second {who}: Q4 was not read a second time" in log
    assert MARKER not in log
    assert reading_files(workspace, who) == before
    raised = flags(KEY, [load(workspace, KEY, student)])
    # No grade is stored, so every part is also incomplete.
    assert {(f.part_id, f.code) for f in raised if f.code != "incomplete"} == {
        ("Q4.b", FlagCode.HARD_TO_READ),
        ("Q4.b", FlagCode.LOW_RESOLUTION),
    }
    write_report(workspace, KEY, [student])
    lines = workspace.run_report_md.read_text().splitlines()
    assert (
        "Parts the reader was unsure of, read a second time by another vendor's"
        " model: 2. The same: 1. Different: 0. Not decided: 0. Failed: 1."
    ) in lines
    assert (
        "The other parts of those units: 2. The same: 2. Different: 0. Not decided: 0."
    ) in lines


# --- the comparison -----------------------------------------------------------------


def reading(part_id: str, **changes: Any) -> Reading:
    """A clear answered reading of `part_id`, with `changes`; `answer` holds the
    final answer's fields."""
    stated = changes.pop("answer", {})
    fields: dict[str, Any] = {
        "part_id": part_id,
        "status": "answered",
        "medium": "handwritten",
        "lines": ["a = 1"],
        "final_answer": {"choices": [], "value": None, "unit": None, "expression": None}
        | stated,
        "elements": [],
        "uncertain": [],
        "addressed_to_grader": [],
        "legibility": "clear",
    }
    return Reading.model_validate(fields | changes)


def force(label: str, direction: str) -> dict[str, Any]:
    return {"label": label, "kind": "force", "direction": direction, "position": None}


def says(**answer: Any) -> dict[str, Any]:
    return {"answer": answer}


def ms2(value: float) -> dict[str, Any]:
    return says(value=value, unit="m/s^2")


C, A = says(choices=["C"]), ms2(2.5)
X = says(expression="(x - 3)^2 - 4")
SAID = {"lines": ["It slows down and stops."]}
FOLDED = {"lines": [" it slows  down AND stops."]}
T = {"elements": [force("T", "right")]}
BLANK, UNREAD = {"status": "blank", "lines": []}, {"status": "unreadable", "lines": []}
EMPTY = {"lines": []}
POOR = {"legibility": "poor"}
UNSURE_LINE = {"lines": ["It sl?ws down."]}
UNSURE_LABEL = {"elements": [force("?", "right")]}
UNSURE_UNIT = says(value=2.5, unit="m/s?")
LABELLED = UNREAD | {"elements": [force("??", "up")]}
# Each row: what it shows, the part, the first reading's changes (it is also marked
# uncertain, as a reading read again is), the second's, and the result.
TABLE: list[tuple[str, str, dict[str, Any], dict[str, Any], str]] = [
    ("equal options", "Q1", C, C, "same"),
    ("unequal options", "Q1", C, says(choices=["A"]), "differ"),
    ("an option on one side only", "Q1", C, {}, "differ"),
    ("equal numbers", "Q4.b", A, A, "same"),
    ("2.5 against 2.52", "Q4.b", A, ms2(2.52), "differ"),
    ("2.55 against 2.6, a 2% key", "Q4.b", ms2(2.55), ms2(2.6), "differ"),
    ("outside tolerance", "Q4.b", A, ms2(25), "differ"),
    ("in another unit", "Q4.b", A, says(value=250, unit="cm/s^2"), "same"),
    ("2.5 against 2.50", "Q4.b", A, ms2(2.50), "same"),
    ("one unit missing", "Q4.b", A, says(value=2.5), "not decided"),
    ("a value on one side only", "Q4.b", A, {}, "differ"),
    ("no value, equal lines", "Q4.b", {}, {"lines": ["A = 1 "]}, "same"),
    ("equal expressions", "Q5.a", X, says(expression="x^2 - 6*x + 5"), "same"),
    ("different expressions", "Q5.a", X, says(expression="(x - 3)^2 + 4"), "differ"),
    ("unread expression", "Q5.a", X, says(expression="(x-3)^2 - ?"), "not decided"),
    ("lines equal but for case and spaces", "Q4.c", SAID, FOLDED, "same"),
    ("unequal lines", "Q4.c", SAID, {"lines": ["It speeds up."]}, "not decided"),
    ("equal elements", "Q4.a", T, {"elements": [force("t", "Right")]}, "same"),
    ("unequal elements", "Q4.a", T, {"elements": [force("T", "left")]}, "not decided"),
    ("both readings empty", "Q4.c", EMPTY, EMPTY, "not decided"),
    ("a ? in the second's line", "Q4.c", UNSURE_LINE, UNSURE_LINE, "not decided"),
    ("a ? in the second's label", "Q4.a", UNSURE_LABEL, UNSURE_LABEL, "not decided"),
    ("a ? in the second's answer", "Q4.b", A, UNSURE_UNIT, "not decided"),
    ("a ? in the second's option", "Q1", C, says(choices=["?"]), "not decided"),
    ("poor legibility, the same value", "Q4.b", A, A | POOR, "not decided"),
    ("poor legibility, another value", "Q4.b", A, ms2(25) | POOR, "not decided"),
    ("an uncertain line", "Q4.c", SAID, SAID | {"uncertain": [1]}, "not decided"),
    ("one side blank", "Q4.c", SAID, BLANK, "differ"),
    ("one side unreadable", "Q4.b", A, UNREAD, "differ"),
    ("unreadable, a label of ?", "Q4.a", T, LABELLED, "differ"),
    ("the other side blank", "Q4.b", BLANK, A, "differ"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("part", "first", "second", "found"),
    [row[1:] for row in TABLE],
    ids=[row[0] for row in TABLE],
)
def test_two_readings_are_compared_in_code(
    part: str, first: dict[str, Any], second: dict[str, Any], found: str
) -> None:
    one, other = reading(part, uncertain=[1], **first), reading(part, **second)
    assert compared(PARTS[part], KEY.policies, one, other) == found


@pytest.mark.parametrize("penalised", [False, True])
def test_equal_numbers_decide_nothing_when_significant_figures_count(
    penalised: bool,
) -> None:
    # A reading's value keeps no trailing zeros, so 2.5 and 2.50 look alike to code.
    policies = KEY.policies.model_copy(update={"penalise_sig_figs": penalised})
    one = reading("Q4.b", uncertain=[1], answer={"value": 2.5, "unit": "m/s^2"})
    other = reading("Q4.b", answer={"value": 2.50, "unit": "m/s^2"})
    wanted = "not decided" if penalised else "same"
    assert compared(PARTS["Q4.b"], policies, one, other) == wanted
    third = reading("Q4.b", answer={"value": 3.5, "unit": "m/s^2"})
    assert compared(PARTS["Q4.b"], policies, one, third) == "differ"
