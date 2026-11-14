"""The grade stage with stored readings and a mock transport
(src/redpen/grade/grade.py)."""

import asyncio
import base64
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from redpen import prompts
from redpen.grade import grade as stage
from redpen.grade.grade import (
    Grades,
    grade_student,
    graded_from,
    part_points,
    problems,
)
from redpen.grade.read import digest
from redpen.grade.second_grade import grade_again
from redpen.llm import client
from redpen.llm.client import Gateway, ResponseValidationError, StageError
from redpen.llm.models import ModelInfo
from redpen.prepare.key import grading_content, number, part_hash
from redpen.report.report import audited, load
from redpen.schemas.audit import Change, PartAudit
from redpen.schemas.common import Provenance
from redpen.schemas.grade import (
    AnswerCheck,
    CriterionGrade,
    Grade,
    GradeRecord,
    ProblemGrades,
    SecondGrade,
)
from redpen.schemas.key import Key, KeyPart
from redpen.schemas.reading import (
    Crop,
    FinalAnswer,
    ProblemReadings,
    Reading,
    ReadingRecord,
)
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, answer
from tests.unit.test_gateway import base_cache_key

MODEL = ModelInfo(
    slug="vendor/vision-1",
    supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
    accepts_images=True,
    prompt_price=Decimal(0),
    completion_price=Decimal(0),
    providers=frozenset({"zero"}),
)
KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
NOTHING = FinalAnswer(choices=[], value=None, unit=None, expression=None)
NOBODY = Provenance(key_part_hash=None, prompt_version=None, model_slug=None)
STUDENT = "s"
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgAB4iG8MwAAAABJRU5ErkJggg=="
)


def key_of(*parts: KeyPart) -> Key:
    return Key(policies=KEY.policies, parts=list(parts))


def changed(part_id: str, **fields: Any) -> KeyPart:
    return KeyPart.model_validate(PARTS[part_id].model_dump() | fields)


def reading(part_id: str, **changes: Any) -> Reading:
    fields: dict[str, Any] = {
        "part_id": part_id,
        "status": "answered",
        "medium": "handwritten",
        "lines": ["a line", "another"],
        "final_answer": NOTHING,
        "elements": [],
        "uncertain": [],
        "addressed_to_grader": [],
        "legibility": "clear",
    }
    return Reading.model_validate(fields | changes)


def blank(part_id: str) -> Reading:
    return reading(part_id, status="blank", medium=None, lines=[], legibility=None)


def unreadable(part_id: str) -> Reading:
    return reading(part_id, status="unreadable", lines=[], legibility="poor")


def store(
    workspace: Workspace,
    readings: list[Reading],
    crops: int = 0,
    student: str = STUDENT,
) -> None:
    """Write `readings` as the read stage would, each with `crops` crop images."""
    files: defaultdict[str, list[ReadingRecord]] = defaultdict(list)
    for r in readings:
        found = [Crop(page=1, box=[0.1, 0.1, 0.5, 0.5], scale=2.0)] * crops
        for n in range(1, crops + 1):
            path = workspace.crop_png(student, r.part_id, n)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PNG)
        record = ReadingRecord(provenance=NOBODY, reading=r, crops=found)
        files[r.part_id.split(".")[0]].append(record)
    for problem, records in files.items():
        write_model(
            workspace.reading_json(student, problem), ProblemReadings(records=records)
        )


def full(key: Key, part_id: str) -> Grade:
    """Every criterion of `part_id` earned, citing line 1. Each has feedback, so that
    a test may lower an award without breaking the feedback rule."""
    [part] = [p for p in key.parts if p.id == part_id]
    return Grade(
        part_id=part_id,
        criteria=[
            CriterionGrade(
                id=c.id,
                awarded=c.points,
                evidence_line=1,
                evidence_element=None,
                rationale="Meets it.",
                feedback="Check the units.",
            )
            for c in part.criteria
        ],
        transcript_dispute=None,
        belongs_elsewhere=None,
        code_overruled=None,
    )


class Grader:
    """Answers each request with `answer(the part ids it asks to grade, how many
    requests asking for those came before)`, after `delay(ids)` seconds. By default
    every criterion is earned."""

    def __init__(
        self,
        key: Key,
        answer: Callable[[list[str], int], Grades] | None = None,
        delay: Callable[[list[str]], float] = lambda ids: 0.0,
    ) -> None:
        self.answer = answer or (
            lambda ids, n: Grades(grades=[full(key, i) for i in ids])
        )
        self.delay = delay
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        texts = [c["text"] for c in content if c["type"] == "text"]
        ids = [json.loads(t)["grade"]["part_id"] for t in texts if '"grade"' in t]
        before = sum(r["ids"] == ids for r in self.requests)
        self.requests.append({"ids": ids, "json": body})
        await asyncio.sleep(self.delay(ids))
        value = self.answer(ids, before)
        # The model writes points as JSON numbers.
        return reply(json.dumps(value.model_dump(), default=float))


def reply(text: str) -> httpx.Response:
    """The model's answer `text`, as OpenRouter returns it."""
    return httpx.Response(
        200,
        json={
            "model": MODEL.slug,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
        },
    )


def refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError("a part code decides reached the model")


def run(
    workspace: Workspace,
    key: Key,
    grader: Callable[[httpx.Request], Any],
    student: str = STUDENT,
) -> dict[str, GradeRecord]:
    async def main() -> dict[str, GradeRecord]:
        async with Gateway(
            workspace.root.parent / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(grader),
        ) as gateway:
            return await grade_student(workspace, student, key, gateway, MODEL)

    return asyncio.run(main())


def awards(record: GradeRecord) -> list[Decimal]:
    return [c.awarded for c in record.grade.criteria]


def by_code(part: KeyPart) -> Provenance:
    return Provenance(
        key_part_hash=part_hash(part, KEY.policies),
        prompt_version=None,
        model_slug=None,
    )


# --- parts code decides ------------------------------------------------------------


@pytest.mark.parametrize(
    ("response_type", "key_choices", "read", "earned"),
    [
        ("multiple_choice", ["C"], ["C"], True),
        ("multiple_choice", ["C"], ["B"], False),
        ("multiple_choice", ["C"], ["c"], True),
        ("multiple_choice", ["C"], [" C "], True),
        ("multiple_choice", ["C"], ["C", "D"], False),
        ("multiple_select", ["A", "C"], ["C", "A"], True),
        ("multiple_select", ["A", "C"], ["A"], False),
    ],
    ids=[
        "right",
        "wrong",
        "lower-case letter",
        "surrounding spaces",
        "two letters where one is expected",
        "multiple select in another order",
        "multiple select missing one",
    ],
)
def test_a_choice_part_earns_full_points_only_for_the_key_s_set_of_options(
    tmp_path: Path,
    response_type: str,
    key_choices: list[str],
    read: list[str],
    earned: bool,
) -> None:
    worth = Decimal(2)
    criteria = [
        PARTS["Q1"].criteria[0].model_dump() | {"points": worth},
    ]
    q1 = changed(
        "Q1",
        response_type=response_type,
        answer=answer(choices=key_choices),
        criteria=criteria,
    )
    workspace = Workspace(tmp_path / "ws")
    chosen = FinalAnswer(choices=read, value=None, unit=None, expression=None)
    store(workspace, [reading("Q1", lines=[], final_answer=chosen)])

    records = run(workspace, key_of(q1), refuse)

    assert awards(records["Q1"]) == [worth if earned else Decimal(0)]
    assert records["Q1"].provenance == by_code(q1)
    stored = read_model(workspace.grade_json(STUDENT, "Q1"), ProblemGrades)
    assert stored.records == [records["Q1"]]


@pytest.mark.parametrize(
    ("response_type", "key_choices", "read", "earned", "why"),
    [
        ("multiple_choice", ["C"], ["C"], True, "Chooses C, the answer."),
        ("multiple_choice", ["C"], ["B"], False, "Chooses B; the answer is C."),
        (
            "multiple_choice",
            ["C"],
            ["C", "D"],
            False,
            "Two or more options are marked;",
        ),
        ("multiple_choice", ["C"], ["B", "C", "D"], False, "Two or more options are"),
        (
            "multiple_select",
            ["C"],
            ["C", "D"],
            False,
            "Two or more options are marked;",
        ),
        ("multiple_choice", ["C"], ["C", " c"], True, "Chooses C,  c, the answer."),
        ("multiple_select", ["A", "C"], ["C", "A"], True, "Chooses C, A, the answer."),
        ("multiple_select", ["A", "C"], ["A"], False, "Chooses A; the answer is A, C."),
    ],
    ids=[
        "one option right",
        "one option wrong",
        "two marked for a single answer",
        "three marked for a single answer",
        "a select part with one answer",
        "the same option twice",
        "the full set",
        "a partial set",
    ],
)
def test_two_options_marked_where_the_key_has_one_score_zero_by_code(
    response_type: str, key_choices: list[str], read: list[str], earned: bool, why: str
) -> None:
    part = changed(
        "Q1", response_type=response_type, answer=answer(choices=key_choices)
    )
    chosen = FinalAnswer(choices=read, value=None, unit=None, expression=None)

    grade = stage.by_code(part, reading("Q1", final_answer=chosen))

    assert [c.awarded for c in grade.criteria] == [Decimal(1) if earned else Decimal(0)]
    assert all(c.rationale.startswith(why) for c in grade.criteria)


def test_a_choice_part_with_no_option_read_is_withheld_and_a_blank_one_is_zero(
    tmp_path: Path,
) -> None:
    key = key_of(PARTS["Q1"], PARTS["Q2"], PARTS["Q3"])
    workspace = Workspace(tmp_path / "ws")
    chosen = FinalAnswer(choices=["B"], value=None, unit=None, expression=None)
    # Q1: something is written beside the question, and no option is selected.
    store(
        workspace,
        [
            reading("Q1", lines=["a note"]),
            blank("Q2"),
            reading("Q3", final_answer=chosen),
        ],
    )

    records = run(workspace, key, refuse)

    assert list(records) == ["Q2", "Q3"]
    assert awards(records["Q2"]) == [Decimal(0)]
    assert awards(records["Q3"]) == [Decimal(1)]
    q1 = read_model(workspace.grade_json(STUDENT, "Q1"), ProblemGrades)
    assert q1.records == []


def test_a_blank_part_scores_zero_on_every_criterion_and_unreadable_gets_no_grade(
    tmp_path: Path,
) -> None:
    key = key_of(PARTS["Q2"], PARTS["Q4.a"], PARTS["Q4.b"], PARTS["Q4.c"])
    workspace = Workspace(tmp_path / "ws")
    store(
        workspace,
        [unreadable("Q2"), blank("Q4.a"), unreadable("Q4.b"), blank("Q4.c")],
    )

    records = run(workspace, key, refuse)

    assert list(records) == ["Q4.a", "Q4.c"]
    assert awards(records["Q4.a"]) == [Decimal(0), Decimal(0)]
    assert awards(records["Q4.c"]) == [Decimal(0), Decimal(0)]
    assert records["Q4.a"].provenance == by_code(PARTS["Q4.a"])
    q4 = read_model(workspace.grade_json(STUDENT, "Q4"), ProblemGrades)
    assert q4.records == [records["Q4.a"], records["Q4.c"]]
    # A problem with nothing to grade is still written, with no record.
    q2 = read_model(workspace.grade_json(STUDENT, "Q2"), ProblemGrades)
    assert q2.records == []


# --- the checks --------------------------------------------------------------------

Q4 = key_of(PARTS["Q4.a"], PARTS["Q4.b"], PARTS["Q4.c"])
# Q4 asked for, with a part of another problem in the key besides.
SMALL = key_of(*Q4.parts, PARTS["Q5.a"])
ASKED = {i: reading(i) for i in ("Q4.a", "Q4.b", "Q4.c")}


def q4_with(**changes: Any) -> Grades:
    """Full marks on Q4, with Q4.b's first criterion changed as `changes` say."""
    grades = [full(Q4, i) for i in ASKED]
    first = grades[1].criteria[0].model_copy(update=changes)
    grades[1].criteria[0] = first
    return Grades(grades=grades)


def q4_criteria(*criteria: CriterionGrade) -> Grades:
    grades = [full(Q4, i) for i in ASKED]
    grades[1] = grades[1].model_copy(update={"criteria": list(criteria)})
    return Grades(grades=grades)


C1, C2 = full(Q4, "Q4.b").criteria
VIOLATIONS: list[tuple[str, Grades, list[str]]] = [
    (
        "unknown criterion id",
        q4_criteria(C1, C2.model_copy(update={"id": "Q4.b.c3"})),
        ["Q4.b: its criteria are Q4.b.c1, Q4.b.c2, each once"],
    ),
    (
        "missing criterion",
        q4_criteria(C1),
        ["Q4.b: its criteria are Q4.b.c1, Q4.b.c2, each once"],
    ),
    (
        "criterion twice",
        q4_criteria(C1, C2, C2),
        ["Q4.b: its criteria are Q4.b.c1, Q4.b.c2, each once"],
    ),
    (
        "award between levels",
        q4_with(awarded=Decimal("0.75")),
        ["Q4.b.c1: the award is one of 0, 1, 2"],
    ),
    (
        "award above the maximum",
        q4_with(awarded=Decimal(3)),
        ["Q4.b.c1: the award is one of 0, 1, 2"],
    ),
    (
        "award above zero with no evidence",
        q4_with(evidence_line=None),
        ["Q4.b.c1: an award above zero cites a line or an element of the reading"],
    ),
    (
        "line past the end",
        q4_with(evidence_line=3),
        ["Q4.b.c1: it cites line 3, and the reading of Q4.b has no line 3"],
    ),
    (
        "element past the end",
        q4_with(evidence_line=None, evidence_element=1),
        ["Q4.b.c1: it cites element 1, and the reading of Q4.b has no element 1"],
    ),
    (
        "an award below full with no feedback",
        q4_with(awarded=Decimal(1), feedback=""),
        ["Q4.b.c1: an award below its full points gives feedback to the student"],
    ),
    (
        "part missing",
        Grades(grades=[full(Q4, "Q4.a"), full(Q4, "Q4.b")]),
        ["Q4.c has no grade"],
    ),
    (
        "part twice",
        Grades(grades=[full(Q4, i) for i in ("Q4.a", "Q4.b", "Q4.c", "Q4.a")]),
        ["Q4.a is graded 2 times"],
    ),
    (
        "a part of the key not asked",
        Grades(grades=[*q4_with().grades, full(SMALL, "Q5.a")]),
        ["Q5.a is not to be graded in this request"],
    ),
    (
        "a part not in the key",
        Grades(
            grades=[
                *q4_with().grades,
                full(Q4, "Q4.a").model_copy(update={"part_id": "Q9"}),
            ]
        ),
        ["a grade is for a part that is not in the key"],
    ),
]


@pytest.mark.parametrize(
    ("found", "wrong"),
    [(found, wrong) for _, found, wrong in VIOLATIONS]
    + [(q4_with(), []), (q4_with(feedback=""), [])],
    ids=[name for name, _, _ in VIOLATIONS]
    + ["valid", "a full award with no feedback"],
)
def test_each_check_catches_its_failure(found: Grades, wrong: list[str]) -> None:
    assert problems(SMALL, ASKED, found) == wrong


def test_a_zero_award_may_cite_a_line_and_an_element_that_exist() -> None:
    q4a = reading(
        "Q4.a",
        elements=[{"label": "x", "kind": "arrow", "direction": "up", "position": None}],
    )
    found = q4_with(awarded=Decimal(0), evidence_line=None, evidence_element=None)
    found.grades[0].criteria[0] = (
        found.grades[0]
        .criteria[0]
        .model_copy(
            update={"awarded": Decimal(0), "evidence_line": 2, "evidence_element": 1}
        )
    )

    assert problems(SMALL, ASKED | {"Q4.a": q4a}, found) == []


@pytest.mark.parametrize(
    ("bad", "wrong"),
    [(found, wrong) for _, found, wrong in VIOLATIONS],
    ids=[name for name, _, _ in VIOLATIONS],
)
def test_a_violation_is_asked_again_once_then_raises_and_writes_nothing(
    tmp_path: Path, bad: Grades, wrong: list[str]
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, [*ASKED.values(), blank("Q5.a")])
    earlier = workspace.grade_json(STUDENT, "Q4")
    write_model(earlier, ProblemGrades(records=[]))
    before = earlier.read_bytes()
    grader = Grader(SMALL, lambda ids, n: bad)

    with pytest.raises(StageError) as raised:
        run(workspace, SMALL, grader)

    assert str(raised.value) == (
        f"the grade of Q4 of {STUDENT} fails its checks: {'; '.join(wrong)}"
    )
    assert len(grader.requests) == 2
    last = grader.requests[1]["json"]["messages"][1]["content"][-1]
    assert last == {"type": "text", "text": json.dumps({"problems": wrong})}
    assert earlier.read_bytes() == before


def test_a_violation_put_right_on_the_second_answer_is_stored(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    grader = Grader(
        Q4, lambda ids, n: q4_with(awarded=Decimal("0.75")) if n == 0 else q4_with()
    )

    records = run(workspace, Q4, grader)

    assert len(grader.requests) == 2
    assert awards(records["Q4.b"]) == [Decimal(2), Decimal(1)]


# --- held to the answer check ------------------------------------------------------


def stated(part_id: str, **final: Any) -> Reading:
    return reading(part_id, final_answer=NOTHING.model_copy(update=final))


def q4b_final(award: int, overruled: str | None = None) -> Grades:
    """Full marks on Q4, with Q4.b.c2 (its final answer, worth 1) at `award` and
    `overruled` as Q4.b's reason."""
    grades = q4_with().grades
    c2 = grades[1].criteria[1].model_copy(update={"awarded": Decimal(award)})
    grades[1] = grades[1].model_copy(
        update={"criteria": [grades[1].criteria[0], c2], "code_overruled": overruled}
    )
    return Grades(grades=grades)


AGAINST = "the award goes against the answer check, so code_overruled gives the reason"
RIGHT, WRONG, NONE = 2.5, 5.0, None  # Q4.b's key is 2.5 m/s^2


@pytest.mark.parametrize(
    ("value", "found", "wrong"),
    [
        (RIGHT, q4b_final(0), [f"Q4.b.c2: {AGAINST}"]),
        (WRONG, q4b_final(1), [f"Q4.b.c2: {AGAINST}"]),
        (RIGHT, q4b_final(0, " "), [f"Q4.b.c2: {AGAINST}"]),
        (RIGHT, q4b_final(0, "a reason"), []),
        (WRONG, q4b_final(1, "a reason"), []),
        (RIGHT, q4b_final(1), []),
        (WRONG, q4b_final(0), []),
        (NONE, q4b_final(0), []),
        (NONE, q4b_final(1), []),
        (RIGHT, q4_with(awarded=Decimal(1)), []),
    ],
    ids=[
        "less than full on a match",
        "full on a mismatch",
        "white space for a reason",
        "less than full on a match, with a reason",
        "full on a mismatch, with a reason",
        "full on a match",
        "zero on a mismatch",
        "zero when it cannot tell",
        "full when it cannot tell",
        "a method criterion below full on a match",
    ],
)
def test_a_final_answer_award_against_the_answer_check_needs_a_reason(
    value: float | None, found: Grades, wrong: list[str]
) -> None:
    asked = ASKED | {"Q4.b": stated("Q4.b", value=value, unit="m/s^2")}

    assert problems(SMALL, asked, found) == wrong


@pytest.mark.parametrize(("award", "wrong"), [(1, []), (2, [f"Q5.a.c1: {AGAINST}"])])
def test_a_partial_level_agrees_with_a_mismatch_and_full_points_do_not(
    award: int, wrong: list[str]
) -> None:
    # Q5.a's key is (x - 3)^2 - 4; its final answer, worth 2, lists a partial level 1.
    key = key_of(PARTS["Q5.a"])
    found = full(key, "Q5.a")
    found.criteria[0].awarded = Decimal(award)
    asked = {"Q5.a": stated("Q5.a", expression="(x - 3)^2 + 4")}

    assert problems(key, asked, Grades(grades=[found])) == wrong


def test_an_overrule_is_asked_for_kept_and_one_that_overrules_nothing_cleared(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    wrong = stated("Q4.b", value=WRONG, unit="m/s^2")
    store(workspace, [ASKED["Q4.a"], wrong, ASKED["Q4.c"]])

    def answer(ids: list[str], n: int) -> Grades:
        grades = q4b_final(1, "carried" if n else None).grades
        # Q4.a has no check, so nothing to overrule.
        grades[0] = grades[0].model_copy(update={"code_overruled": "o"})
        return Grades(grades=grades)

    grader = Grader(Q4, answer)

    records = run(workspace, Q4, grader)

    assert len(grader.requests) == 2
    last = grader.requests[1]["json"]["messages"][1]["content"][-1]
    assert last == {
        "type": "text",
        "text": json.dumps({"problems": [f"Q4.b.c2: {AGAINST}"]}),
    }
    assert records["Q4.b"].grade.code_overruled == "carried"
    assert records["Q4.a"].grade.code_overruled is None


TWO_FINAL = key_of(
    PARTS["Q4.a"],
    changed(
        "Q4.b",
        criteria=[
            PARTS["Q4.b"].criteria[0].model_dump() | {"kind": "final_answer"},
            PARTS["Q4.b"].criteria[1],
        ],
    ),
    PARTS["Q4.c"],
)


def test_a_part_with_two_final_answer_criteria_gets_the_check_and_is_not_held(
    tmp_path: Path,
) -> None:
    # A match, and the second final-answer criterion at zero with no reason.
    workspace = Workspace(tmp_path / "ws")
    right = stated("Q4.b", value=RIGHT, unit="m/s^2")
    store(workspace, [ASKED["Q4.a"], right, ASKED["Q4.c"]])
    grader = Grader(TWO_FINAL, lambda ids, n: q4b_final(0))

    records = run(workspace, TWO_FINAL, grader)

    assert problems(TWO_FINAL, ASKED | {"Q4.b": right}, q4b_final(0)) == []
    assert len(grader.requests) == 1
    content = grader.requests[0]["json"]["messages"][1]["content"]
    texts = [json.loads(c["text"]) for c in content if c["type"] == "text"]
    [item] = [t for t in texts if t.get("grade", {}).get("part_id") == "Q4.b"]
    assert item["check"]["verdict"] == "matches"
    assert awards(records["Q4.b"]) == [Decimal(2), Decimal(0)]


@pytest.mark.parametrize(
    ("first", "kept"),
    [(0, "a reason"), (2, None)],
    ids=["every final answer against: kept", "one of two against: cleared"],
)
def test_on_a_part_with_two_final_answers_a_reason_is_kept_only_against_both(
    tmp_path: Path, first: int, kept: str | None
) -> None:
    # A match; the second final-answer criterion at zero, and the first at `first`.
    workspace = Workspace(tmp_path / "ws")
    right = stated("Q4.b", value=RIGHT, unit="m/s^2")
    store(workspace, [ASKED["Q4.a"], right, ASKED["Q4.c"]])

    def answer(ids: list[str], n: int) -> Grades:
        grades = q4b_final(0, "a reason").grades
        c1 = grades[1].criteria[0].model_copy(update={"awarded": Decimal(first)})
        grades[1].criteria[0] = c1
        return Grades(grades=grades)

    grader = Grader(TWO_FINAL, answer)

    records = run(workspace, TWO_FINAL, grader)

    assert len(grader.requests) == 1
    assert awards(records["Q4.b"]) == [Decimal(first), Decimal(0)]
    assert records["Q4.b"].grade.code_overruled == kept


def test_the_check_given_to_the_grader_is_stored_with_the_grade(
    tmp_path: Path,
) -> None:
    key = key_of(PARTS["Q1"], *Q4.parts)
    workspace = Workspace(tmp_path / "ws")
    chosen = FinalAnswer(choices=["C"], value=None, unit=None, expression=None)
    wrong = stated("Q4.b", value=WRONG, unit="m/s^2")
    readings = [ASKED["Q4.a"], wrong, ASKED["Q4.c"]]
    store(workspace, [reading("Q1", final_answer=chosen), *readings])

    records = run(workspace, key, Grader(Q4, lambda ids, n: q4b_final(0)))

    found = AnswerCheck(
        verdict="does not match",
        reason="The value is outside the tolerance of every value the key accepts.",
    )
    assert records["Q4.b"].check == found
    assert records["Q4.a"].check is None  # no check for a drawing
    assert records["Q1"].check is None  # decided by code
    stored = read_model(workspace.grade_json(STUDENT, "Q4"), ProblemGrades)
    assert [r.check for r in stored.records] == [None, found, None]
    # A record written before the check was stored still loads.
    old = records["Q4.a"].model_dump(mode="json")
    del old["check"]
    assert GradeRecord.model_validate(old).check is None


def test_a_reason_given_where_no_award_goes_against_the_check_is_cleared(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    right = stated("Q4.b", value=RIGHT, unit="m/s^2")
    store(workspace, [ASKED["Q4.a"], right, ASKED["Q4.c"]])
    grader = Grader(Q4, lambda ids, n: q4b_final(1, "o"))

    records = run(workspace, Q4, grader)

    assert len(grader.requests) == 1
    assert records["Q4.b"].grade.code_overruled is None


def test_a_negative_award_fails_the_schema_and_is_repaired_once_then_raises(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    grader = Grader(Q4, lambda ids, n: q4_with(awarded=Decimal(-1)))

    with pytest.raises(ResponseValidationError):
        run(workspace, Q4, grader)

    # The gateway's one repair: the same request with the invalid answer and the
    # error, which the mock counts as a second request for the same parts.
    assert len(grader.requests) == 2
    assert not workspace.grade_json(STUDENT, "Q4").exists()


def test_the_grader_writes_its_reasoning_before_the_award() -> None:
    # A model writes its answer in the schema's order.
    order = list(Grade.model_json_schema()["$defs"]["CriterionGrade"]["properties"])
    assert order == [
        "id",
        "rationale",
        "evidence_line",
        "evidence_element",
        "awarded",
        "feedback",
    ]


# --- calls -------------------------------------------------------------------------


def test_points_never_pass_through_float(tmp_path: Path) -> None:
    c1, c2 = PARTS["Q4.b"].criteria
    q4b = changed(
        "Q4.b",
        criteria=[
            c1.model_dump()
            | {"points": "0.2", "partial": [{"points": "0.1", "when": "w"}]},
            c2.model_dump() | {"points": "0.1"},
        ],
    )
    key = key_of(q4b)
    workspace = Workspace(tmp_path / "ws")
    store(workspace, [reading("Q4.b")])

    def tenths(ids: list[str], n: int) -> Grades:
        grade = full(key, "Q4.b")
        for c in grade.criteria:
            c.awarded = Decimal("0.1")
        return Grades(grades=[grade])

    records = run(workspace, key, Grader(key, tenths))

    assert awards(records["Q4.b"]) == [Decimal("0.1"), Decimal("0.1")]
    # 0.1 + 0.2 in float is 0.30000000000000004.
    assert part_points(records["Q4.b"], key) == (Decimal("0.2"), Decimal("0.3"))


def test_an_award_is_stored_as_the_key_writes_its_level(tmp_path: Path) -> None:
    key = key_of(PARTS["Q4.b"])  # Q4.b.c1 is worth 2, Q4.b.c2 is worth 1
    workspace = Workspace(tmp_path / "ws")
    store(workspace, [reading("Q4.b")])
    text = json.dumps({"grades": [full(key, "Q4.b").model_dump(mode="json")]})
    # Points given as strings reach Decimal as written; a number such as 2.00 does
    # not, since Pydantic reads it through float first.
    text = text.replace('"awarded": "2"', '"awarded": "2.00"')
    text = text.replace('"awarded": "1"', '"awarded": "-0"')

    records = run(workspace, key, lambda request: reply(text))

    assert [str(a) for a in awards(records["Q4.b"])] == ["2", "0"]
    stored = workspace.grade_json(STUDENT, "Q4").read_text()
    assert '"awarded": "2"' in stored and '"awarded": "0"' in stored


def test_a_part_s_points_are_summed_by_code_from_the_record_and_the_key() -> None:
    grade = full(KEY, "Q4.b")
    grade.criteria[0].awarded = Decimal(1)
    record = GradeRecord(
        provenance=by_code(PARTS["Q4.b"]), grade=grade, check=None, reading_digest=None
    )

    assert part_points(record, KEY) == (Decimal(2), Decimal(3))


C1_EDITED = changed(
    "Q4.b",
    criteria=[
        PARTS["Q4.b"].criteria[0].model_dump() | {"requirement": "New."},
        PARTS["Q4.b"].criteria[1],
    ],
)


@pytest.mark.parametrize(
    "edited",
    [
        key_of(C1_EDITED),
        Key(
            policies=KEY.policies.model_copy(update={"rel_tol": 0.05}),
            parts=[PARTS["Q4.b"]],
        ),
    ],
    ids=["a criterion edited", "only the policies edited"],
)
def test_a_record_graded_under_another_key_is_never_summed(edited: Key) -> None:
    record = GradeRecord(
        provenance=by_code(PARTS["Q4.b"]),
        grade=full(KEY, "Q4.b"),
        check=None,
        reading_digest=None,
    )

    with pytest.raises(ValueError) as raised:
        part_points(record, edited)

    assert str(raised.value) == (
        "Q4.b was graded under another version of its key part: grade it again"
    )


def test_dependencies_are_sent_as_context_and_are_not_graded(tmp_path: Path) -> None:
    # Q5.c is made to depend on Q4.b too, a part of another problem.
    q5c = changed("Q5.c", depends_on=["Q4.b"])
    key = key_of(PARTS["Q4.b"], PARTS["Q5.a"], PARTS["Q5.b"], q5c)
    workspace = Workspace(tmp_path / "ws")
    q4b = reading("Q4.b", lines=["the earlier work"])
    q5b = reading("Q5.b", final_answer=NOTHING.model_copy(update={"value": -4.0}))
    store(workspace, [q4b], crops=1)
    store(workspace, [blank("Q5.a"), q5b, reading("Q5.c")], crops=1)
    grader = Grader(key)

    records = run(workspace, key, grader)

    q5 = [r for r in grader.requests if r["ids"] == ["Q5.b", "Q5.c"]]
    assert len(q5) == 1
    content = q5[0]["json"]["messages"][1]["content"]
    texts = [json.loads(c["text"]) for c in content if c["type"] == "text"]
    context = [t["context"] for t in texts if "context" in t]
    assert context == [
        blank("Q5.a").model_dump(mode="json"),
        q4b.model_dump(mode="json"),
    ]
    assert not any("grade" in t and t["grade"]["part_id"] == "Q5.a" for t in texts)
    # Q5.a is read blank, so code scored it.
    assert records["Q5.a"].provenance == by_code(PARTS["Q5.a"])
    # Each record names every reading it was graded from: its own, and its
    # dependencies', in this problem or another.
    stored = {
        r.reading.part_id: r
        for q in ("Q4", "Q5")
        for r in read_model(workspace.reading_json(STUDENT, q), ProblemReadings).records
    }
    own = digest(stored["Q5.a"])
    assert records["Q5.a"].reading_digest == graded_from(PARTS["Q5.a"], stored) == own
    for part in (PARTS["Q5.b"], q5c):
        digests = " ".join(digest(stored[i]) for i in [part.id, *part.depends_on])
        made = hashlib.sha256(digests.encode()).hexdigest()[:12]
        assert records[part.id].reading_digest == graded_from(part, stored) == made
    # Each part's crops follow its reading, each after a line saying what it is.
    # The answer check goes beside the reading of a part that has one.
    url = f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
    kinds = [
        json.loads(c["text"]) if c["type"] == "text" else c["image_url"]["url"]
        for c in content
    ]
    check = {
        "verdict": "matches",
        "reason": "The value is within tolerance of the key's value.",
    }
    start = kinds.index({"grade": q5b.model_dump(mode="json"), "check": check})
    assert kinds[start : start + 6] == [
        {"grade": q5b.model_dump(mode="json"), "check": check},
        {"crop": 1, "part": "Q5.b"},
        url,
        {"grade": reading("Q5.c").model_dump(mode="json")},
        {"crop": 1, "part": "Q5.c"},
        url,
    ]


def test_the_shared_content_is_the_same_for_every_student(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()), student="s1")
    store(
        workspace,
        [reading("Q4.a", lines=["other"]), reading("Q4.b"), blank("Q4.c")],
        crops=1,
        student="s2",
    )
    grader = Grader(Q4)

    run(workspace, Q4, grader, "s1")
    run(workspace, Q4, grader, "s2")

    shared: list[list[dict[str, Any]]] = []
    for request in grader.requests:
        content = request["json"]["messages"][1]["content"]
        end = next(n for n, c in enumerate(content) if "cache_control" in c)
        shared.append(content[: end + 1])
    assert [r["ids"] for r in grader.requests] == [
        ["Q4.a", "Q4.b", "Q4.c"],
        ["Q4.a", "Q4.b"],
    ]
    assert shared[0] == shared[1]
    policies = json.dumps(KEY.policies.model_dump(), default=number)
    assert [c["text"] for c in shared[0]] == [
        *(grading_content(p) for p in Q4.parts),
        policies,
    ]


def test_grades_are_stored_per_problem_with_their_provenance(tmp_path: Path) -> None:
    key = key_of(PARTS["Q1"], *Q4.parts)
    workspace = Workspace(tmp_path / "ws")
    chosen = FinalAnswer(choices=["C"], value=None, unit=None, expression=None)
    store(workspace, [reading("Q1", final_answer=chosen), *ASKED.values()])

    def disputed(ids: list[str], n: int) -> Grades:
        grades = q4_with().grades
        grades[0] = grades[0].model_copy(
            update={"transcript_dispute": "d", "code_overruled": "o"}
        )
        return Grades(grades=grades)

    grader = Grader(Q4, disputed)

    records = run(workspace, key, grader)

    assert {r["json"]["session_id"] for r in grader.requests} == {"ws:grade"}
    _, version = prompts.load("grade", "problem")
    for part in Q4.parts:
        assert records[part.id].provenance == Provenance(
            key_part_hash=part_hash(part, KEY.policies),
            prompt_version=version,
            model_slug=MODEL.slug,
        )
    assert records["Q1"].provenance == by_code(PARTS["Q1"])
    # Stored as returned, except that Q4.a has no answer check to overrule.
    assert records["Q4.a"].grade.transcript_dispute == "d"
    assert records["Q4.a"].grade.code_overruled is None
    for problem, ids in (("Q1", ["Q1"]), ("Q4", ["Q4.a", "Q4.b", "Q4.c"])):
        stored = read_model(workspace.grade_json(STUDENT, problem), ProblemGrades)
        assert stored.records == [records[i] for i in ids]
        # Each record names the reading it was graded from, by code or by model.
        read = read_model(workspace.reading_json(STUDENT, problem), ProblemReadings)
        assert [r.reading_digest for r in stored.records] == [
            digest(r) for r in read.records
        ]


def test_a_grade_request_keeps_the_cache_key_it_had_before_draws_existed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()), crops=1)
    keyed: list[tuple[tuple[Any, ...], str]] = []
    real = client.cache_key

    def kept(*args: Any) -> str:
        keyed.append((args, found := real(*args)))
        return found

    monkeypatch.setattr(client, "cache_key", kept)

    run(workspace, Q4, Grader(Q4))

    [(args, key)] = keyed
    *material, draw = args
    assert draw == 1 and key == base_cache_key(*material)


def test_a_record_stored_before_the_reading_s_digest_or_the_check_loads() -> None:
    record = GradeRecord(
        provenance=by_code(PARTS["Q1"]),
        grade=full(KEY, "Q1"),
        check=None,
        reading_digest="0123456789ab",
    )
    stored = record.model_dump(mode="json")
    del stored["reading_digest"], stored["check"]

    old = GradeRecord.model_validate_json(json.dumps(stored))

    assert (old.reading_digest, old.check) == (None, None)
    assert old.grade == record.grade


def test_a_problem_that_fails_twice_keeps_its_earlier_file_and_the_others_are_replaced(
    tmp_path: Path,
) -> None:
    key = key_of(PARTS["Q1"], *Q4.parts, PARTS["Q5.a"])
    workspace = Workspace(tmp_path / "ws")
    chosen = FinalAnswer(choices=["C"], value=None, unit=None, expression=None)
    store(
        workspace,
        [reading("Q1", final_answer=chosen), *ASKED.values(), reading("Q5.a")],
    )
    earlier = ProblemGrades(records=[])
    for problem in ("Q1", "Q5"):
        write_model(workspace.grade_json(STUDENT, problem), earlier)
    # A grade made before, under an older prompt: the only one Q4 has.
    old = GradeRecord(
        provenance=by_code(PARTS["Q4.a"]).model_copy(
            update={"prompt_version": "older", "model_slug": MODEL.slug}
        ),
        grade=full(key, "Q4.a"),
        check=None,
        reading_digest=None,
    )
    write_model(workspace.grade_json(STUDENT, "Q4"), ProblemGrades(records=[old]))
    before = workspace.grade_json(STUDENT, "Q4").read_bytes()

    def answer(ids: list[str], n: int) -> Grades:
        return Grades(grades=[full(key, i) for i in ids if i != "Q4.c"])

    # Q5 is still being graded when Q4's second answer fails.
    grader = Grader(key, answer, delay=lambda ids: 0.3 if ids == ["Q5.a"] else 0.0)

    with pytest.raises(StageError) as raised:
        run(workspace, key, grader)

    assert str(raised.value) == (
        f"the grade of Q4 of {STUDENT} fails its checks: Q4.c has no grade"
    )
    assert workspace.grade_json(STUDENT, "Q4").read_bytes() == before
    for problem in ("Q1", "Q5"):
        stored = read_model(workspace.grade_json(STUDENT, problem), ProblemGrades)
        assert stored != earlier


# --- a stored grade kept when nothing it was graded from changed --------------------


def drawn(key: Key, tag: str, lowered: str | None = None) -> Grades:
    """One draw of the grader: full marks, every rationale `tag`, but zero on the
    first criterion of the part `lowered`."""
    grades: list[Grade] = []
    for g in (full(key, p.id) for p in key.parts if p.id in ASKED):
        criteria = [c.model_copy(update={"rationale": tag}) for c in g.criteria]
        if g.part_id == lowered:
            criteria[0] = criteria[0].model_copy(update={"awarded": Decimal(0)})
        grades.append(g.model_copy(update={"criteria": criteria}))
    return Grades(grades=grades)


def tags(records: dict[str, GradeRecord]) -> dict[str, str]:
    """The rationale of each record, by part id: which draw it came from."""
    return {i: r.grade.criteria[0].rationale for i, r in records.items()}


def requirement(key: Key, criterion: str, text: str) -> Key:
    """`key` with the requirement of `criterion` edited to `text`."""
    parts = [
        p.model_copy(
            update={
                "criteria": [
                    c.model_copy(update={"requirement": text})
                    if c.id == criterion
                    else c
                    for c in p.criteria
                ]
            }
        )
        for p in key.parts
    ]
    return key.model_copy(update={"parts": parts})


# Q4.c made to depend on Q4.b.
DEPENDENT = key_of(PARTS["Q4.a"], PARTS["Q4.b"], changed("Q4.c", depends_on=["Q4.b"]))


def test_an_edited_criterion_grades_its_part_and_dependents_again_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    first = run(workspace, DEPENDENT, Grader(DEPENDENT, lambda ids, n: drawn(Q4, "1")))
    edited = requirement(DEPENDENT, "Q4.b.c2", "States the acceleration.")

    after = run(workspace, edited, Grader(edited, lambda ids, n: drawn(Q4, "2")))

    assert after["Q4.a"].model_dump_json() == first["Q4.a"].model_dump_json()
    assert tags(after) == {"Q4.a": "1", "Q4.b": "2", "Q4.c": "2"}
    stored = read_model(workspace.grade_json(STUDENT, "Q4"), ProblemGrades)
    assert stored.records == list(after.values())


@pytest.mark.parametrize(
    ("change", "new"),
    [("nothing", set[str]()), ("a reading", {"Q4.a"}), ("a check", {"Q4.b"})],
)
def test_a_part_is_graded_again_only_when_what_it_was_graded_from_changed(
    tmp_path: Path, change: str, new: set[str]
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    run(workspace, Q4, Grader(Q4, lambda ids, n: drawn(Q4, "1")))
    path = workspace.grade_json(STUDENT, "Q4")
    before = path.read_bytes()
    if change == "a reading":
        store(workspace, [reading("Q4.a", lines=["other"]), *list(ASKED.values())[1:]])
    if change == "a check":  # the verdict the check gave before it was put right
        file = read_model(path, ProblemGrades)
        older = AnswerCheck(verdict="does not match", reason="An older verdict.")
        file.records[1] = file.records[1].model_copy(update={"check": older})
        write_model(path, file)
    (tmp_path / "cache.sqlite").unlink()  # so that the grader draws again

    after = run(workspace, Q4, Grader(Q4, lambda ids, n: drawn(Q4, "2")))

    assert {i for i, tag in tags(after).items() if tag == "2"} == new
    if not new:
        assert path.read_bytes() == before


def test_the_second_grade_pairs_with_a_kept_grade_and_is_compared_with_it(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    first = run(workspace, Q4, Grader(Q4, lambda ids, n: drawn(Q4, "1")))
    edited = requirement(Q4, "Q4.b.c2", "States the acceleration.")
    # The first draw grades, the second lowers Q4.a.
    grader = Grader(edited, lambda ids, n: drawn(Q4, "2", "Q4.a" if n else None))
    run(workspace, edited, grader)

    async def again() -> list[SecondGrade]:
        async with Gateway(
            tmp_path / "cache.sqlite",
            "test-key",
            requests_per_second=1000,
            transport=httpx.MockTransport(grader),
        ) as gateway:
            return await grade_again(workspace, STUDENT, edited, gateway, MODEL)

    found = {s.part_id: s for s in asyncio.run(again())}

    assert found["Q4.a"].first == digest(first["Q4.a"])
    assert found["Q4.a"].result == "differ"
    assert found["Q4.b"].result == "same"


def test_an_audit_change_decided_on_a_kept_grade_still_applies(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path / "ws")
    store(workspace, list(ASKED.values()))
    run(workspace, Q4, Grader(Q4, lambda ids, n: drawn(Q4, "1")))
    [criterion, _] = PARTS["Q4.a"].criteria
    change = Change(
        student_key=STUDENT,
        criterion=criterion.id,
        part_hash=part_hash(PARTS["Q4.a"], KEY.policies),
        before=criterion.points,
        after=Decimal(0),
        reason="Decided again.",
    )
    audit = PartAudit(
        part_id="Q4.a",
        part_hash=change.part_hash,
        groups=1,
        changed=1,
        undecided=0,
        split=0,
        changes=[change],
        key_suspect=None,
    )
    write_model(workspace.audit_json("Q4.a"), audit)
    edited = requirement(Q4, "Q4.b.c2", "States the acceleration.")
    # A fresh draw would give Q4.a another award than the one the audit changed.
    run(workspace, edited, Grader(edited, lambda ids, n: drawn(Q4, "2", "Q4.a")))
    student = IngestRecord(
        student_name="S", student_key=STUDENT, source_kind="other", pages=[]
    )

    found = audited(load(workspace, edited, student), PARTS["Q4.a"])

    assert found.grade.criteria[0].awarded == 0
    assert found.grade.criteria[0].rationale.startswith("Changed by the class audit")
