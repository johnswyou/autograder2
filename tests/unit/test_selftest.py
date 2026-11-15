"""The rubric's self-test (design 6.5): the key's own solution built into a response
in code and graded by the grade stage itself, the state it leaves, the rubric
written once more on a fresh prepare, and the two measurements beside it. No model
is called: the gateway talks to a mock transport, and solve and verify are fakes."""

import asyncio
import functools
import json
import shutil
import sqlite3
from collections.abc import Awaitable, Callable
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from autograder2 import prompts
from autograder2.cli import app
from autograder2.grade.grade import Grades, grade_student
from autograder2.llm.client import Gateway
from autograder2.llm.models import ModelInfo
from autograder2.prepare.key import load_key
from autograder2.prepare.rubric import PartRubric, ProposedCriterion, Rubric
from autograder2.prepare.selftest import NotEarned, response, self_test, trial
from autograder2.prepare.solve import PartSolution
from autograder2.prepare.verify import (
    AGREE,
    Lost,
    Trial,
    Verdict,
    Verification,
    summary,
    verification,
    verification_md,
)
from autograder2.schemas.common import DiagramElement
from autograder2.schemas.grade import AnswerCheck, CriterionGrade, Grade
from autograder2.schemas.key import Key, KeyPart
from autograder2.schemas.reading import ProblemReadings, ReadingRecord
from autograder2.schemas.spec import Part, Spec
from autograder2.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, NOBODY, answer
from tests.grader import Loses, grades

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
FREE = ["Q4.a", "Q4.b", "Q4.c", "Q5.a", "Q5.b", "Q5.c"]  # the parts a model grades
runner = CliRunner()


def model(slug: str) -> ModelInfo:
    return ModelInfo(
        slug=slug,
        supported_parameters=frozenset({"structured_outputs", "max_tokens"}),
        accepts_images=True,
        prompt_price=Decimal(0),
        completion_price=Decimal(0),
        providers=frozenset({"zero"}),
    )


GRADER = model("vendor/sense-1")
LOCK = "\n".join(
    ['profile = "balanced"']
    + [
        f'\n[models.{role}]\nslug = "{slug}"\nsupported_parameters ='
        ' ["structured_outputs", "max_tokens"]\naccepts_images = true\n'
        'prompt_price = "0"\ncompletion_price = "0"\nproviders = ["zero"]'
        for role, slug in (
            ("solver", "vendor/vision-1"),
            ("checker", "maker/vision-2"),
            ("referee", "vendor/vision-1"),
            ("grader", GRADER.slug),
        )
    ]
)


class Router:
    """A mock OpenRouter that keeps every request's body. The grader gives every
    criterion its full points, except where `lose` gives a rationale; with `fail`
    "call" it answers HTTP 400, and with "checks" it grades no part. The rubric
    writer gives each part one criterion, "States <id>." and, once told what was
    not earned, "States <id> again."; with `fail` "rewrite" that second call
    answers HTTP 400, and with "rewrite checks" it gives no part."""

    def __init__(self, lose: Loses = lambda r, c: None, fail: str = "") -> None:
        self.lose, self.fail = lose, fail
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        texts = [
            json.loads(c["text"])
            for c in body["messages"][1]["content"]
            if c["type"] == "text"
        ]
        value: Any
        if self.named("Grades", body):
            if self.fail == "call":
                return httpx.Response(400, json={"error": {"message": "refused"}})
            checks = self.fail == "checks"
            value = Grades(grades=[]) if checks else grades(body, self.lose)
        else:
            assert self.named("Rubric", body)
            again = " again" if any("not_earned" in t for t in texts) else ""
            if again and self.fail == "rewrite":
                return httpx.Response(400, json={"error": {"message": "refused"}})
            if again and self.fail == "rewrite checks":
                texts[0]["parts"] = []
            value = Rubric(
                parts=[
                    PartRubric(
                        part_id=p["id"],
                        criteria=[
                            ProposedCriterion(
                                kind="final_answer",
                                requirement=f"States {p['id']}{again}.",
                                partial=[],
                            )
                        ],
                        common_errors=[],
                    )
                    for p in texts[0]["parts"]
                ]
            )
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [{"message": {"content": value.model_dump_json()}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )

    @staticmethod
    def named(name: str, body: dict[str, Any]) -> bool:
        return body["response_format"]["json_schema"]["name"] == name

    def of(self, name: str) -> list[dict[str, Any]]:
        return [b for b in self.bodies if self.named(name, b)]


def solved(part: KeyPart, text: str | None = None) -> PartSolution:
    return PartSolution(
        part_id=part.id,
        solution=part.solution if text is None else text,
        answer=part.answer,
        computation="",
        uses=[],
        alternatives=[],
    )


def verdicts(key: Key, second: str | None = None, **change: Any) -> dict[str, Verdict]:
    """Each part verified, the second solution agreeing in code (its text `second`,
    or the key's own), with the fields of `change[part id]` changed."""
    return {
        p.id: Verdict(
            part_id=p.id,
            state="verified",
            compared=AGREE,
            checks=(None, None),
            ruling=None,
            reason=None,
            first=solved(p),
            second=solved(p, second),
            stands=solved(p),
        ).model_copy(update=change.get(p.id.replace(".", "_"), {}))
        for p in key.parts
    }


def run(
    root: Path,
    found: dict[str, Verdict],
    router: Router,
    key: Key = KEY,
    rewrite: Callable[[NotEarned], Awaitable[Key]] | None = None,
) -> tuple[Key, dict[str, Verdict]]:
    workspace = Workspace(root / "ws")
    workspace.root.mkdir(parents=True, exist_ok=True)

    async def main() -> tuple[Key, dict[str, Verdict]]:
        async with Gateway(
            root / "cache.sqlite",
            "test-key",
            transport=httpx.MockTransport(router),
            requests_per_second=1e6,
        ) as gateway:
            return await self_test(workspace, gateway, GRADER, key, found, rewrite)

    return asyncio.run(main())


def items(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(c["text"])
        for c in body["messages"][1]["content"]
        if c["type"] == "text"
    ]


# --- the response built in code ---------------------------------------------------

FORCE = DiagramElement(label="N", kind="force", direction="up", position=None)


@pytest.mark.parametrize(
    ("solution", "fields", "lines"),
    [
        ("a = 10 / 4.0 = 2.5 m/s^2", {"value": 2.5, "unit": "m/s^2"}, 1),
        ("(x - 3)^2 - 4", {"expression": "(x - 3)^2 - 4"}, 1),
        ("Friction acts.", {"must_include": ["It slows down."]}, 1),
        ("Four forces.", {"elements": [FORCE]}, 1),
        ("  f = 10 N\n\nF = 10 N\n   \na = 2.5 m/s^2  \n", {"value": 2.5}, 3),
    ],
    ids=["a number with a unit", "an expression", "an explanation", "a drawing",
         "several lines"],
)  # fmt: skip
def test_the_response_is_the_solution_and_the_key_s_answer(
    solution: str, fields: dict[str, Any], lines: int
) -> None:
    key = answer(**fields)

    found = response("Q4.b", solution, key)

    assert found.lines == [x.strip() for x in solution.splitlines() if x.strip()]
    assert len(found.lines) == lines
    stated = found.final_answer
    assert (stated.choices, stated.value, stated.unit, stated.expression) == (
        key.choices,
        key.value,
        key.unit,
        key.expression,
    )
    assert found.elements == key.elements
    assert (found.part_id, found.status, found.medium) == ("Q4.b", "answered", "typed")
    assert (found.uncertain, found.addressed_to_grader) == ([], [])
    assert found.legibility == "clear"


def test_every_criterion_in_full_keeps_every_state_and_code_s_parts_are_not_tried(
    tmp_path: Path,
) -> None:
    router = Router()
    found = verdicts(KEY, "Other work.", Q4_c={"state": "refereed"})

    key, tried = run(tmp_path, found, router)

    # One request per problem a model grades, and none for the second solutions.
    assert len(router.bodies) == 2
    asked = [x["grade"] for b in router.bodies for x in items(b) if "grade" in x]
    assert [r["part_id"] for r in asked] == FREE
    for reading in asked:
        part = PARTS[reading["part_id"]]
        expected = response(part.id, part.solution, part.answer).model_dump(mode="json")
        assert reading == expected
    assert all("crop" not in x for b in router.bodies for x in items(b))
    assert all(
        c["type"] == "text" for b in router.bodies for c in b["messages"][1]["content"]
    )
    for p in key.parts:
        assert p.state == tried[p.id].state == found[p.id].state
        passed = [Trial(outcome="passed", lost=[])]
        assert tried[p.id].tried == (passed if p.id in FREE else [])
        assert tried[p.id].state_before_test == found[p.id].state
    assert key.model_copy(update={"parts": KEY.parts}) == KEY


def test_a_criterion_below_its_full_points_fails_the_part(tmp_path: Path) -> None:
    # A method criterion: a final answer held below full against a match would be
    # refused by the grade stage's own check.
    def lose(reading: dict[str, Any], c: dict[str, Any]) -> str | None:
        return "Not shown." if c["id"] == "Q4.b.c1" else None

    key, tried = run(tmp_path, verdicts(KEY), Router(lose))

    states = {p.id: p.state for p in key.parts}
    assert states == {**dict.fromkeys(states, "verified"), "Q4.b": "unverified"}
    part = PARTS["Q4.b"]
    c1 = part.criteria[0]
    lost = Lost(
        id="Q4.b.c1",
        kind="method",
        requirement=c1.requirement,
        awarded=Decimal(0),
        points=c1.points,
        rationale="Not shown.",
    )
    assert tried["Q4.b"].tried == [Trial(outcome="failed", lost=[lost])]
    # A partial level is below full points too.
    partly = Grade(
        part_id=part.id,
        criteria=[
            CriterionGrade(
                id=c.id,
                rationale="Partly.",
                evidence_line=1,
                evidence_element=None,
                awarded=c.partial[0].points if c.id == "Q4.b.c1" else c.points,
                feedback="",
            )
            for c in part.criteria
        ],
        transcript_dispute=None,
        belongs_elsewhere=None,
        code_overruled=None,
    )
    found = trial(part, partly)
    assert (found.outcome, [c.id for c in found.lost]) == ("failed", ["Q4.b.c1"])


@pytest.mark.parametrize("state", ["unverified", "refereed"])
def test_a_state_verification_set_is_never_raised_by_a_pass(
    tmp_path: Path, state: str
) -> None:
    found = verdicts(KEY, Q4_b={"state": state}, Q2={"state": state})

    key, tried = run(tmp_path, found, Router())

    assert {p.id: p.state for p in key.parts if p.state != "verified"} == {
        "Q2": state,
        "Q4.b": state,
    }
    assert tried["Q4.b"].tried == [Trial(outcome="passed", lost=[])]


@pytest.mark.parametrize("fail", ["call", "checks"])
def test_a_test_call_that_fails_leaves_its_parts_unverified(
    tmp_path: Path, fail: str
) -> None:
    key, tried = run(tmp_path, verdicts(KEY), Router(fail=fail))

    assert {p.id for p in key.parts if p.state == "unverified"} == set(FREE)
    for id in FREE:
        assert tried[id].tried == [Trial(outcome="not graded", lost=[])]
    log = (tmp_path / "ws" / "run.log").read_text()
    assert "self-test Q4" in log and "self-test Q5" in log


# --- the message is the grade stage's ----------------------------------------------


def test_the_message_is_the_grade_stage_s_and_moves_with_its_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tested = Router()
    run(tmp_path / "test", verdicts(KEY), tested)

    # A student whose readings are the same responses, graded by the grade stage.
    student, workspace = Router(), Workspace(tmp_path / "student" / "ws")
    for problem in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        parts = [p for p in KEY.parts if p.id.split(".")[0] == problem]
        records = [
            ReadingRecord(
                provenance=NOBODY,
                reading=response(p.id, p.solution, p.answer),
                crops=[],
            )
            for p in parts
        ]
        path = workspace.reading_json("s", problem)
        write_model(path, ProblemReadings(records=records))

    async def grade() -> None:
        async with Gateway(
            tmp_path / "student" / "cache.sqlite",
            "test-key",
            transport=httpx.MockTransport(student),
        ) as gateway:
            await grade_student(workspace, "s", KEY, gateway, GRADER)

    asyncio.run(grade())

    def sent(router: Router) -> list[str]:
        return sorted(json.dumps(b, sort_keys=True) for b in router.bodies)

    assert len(tested.bodies) == 2 and sent(tested) == sent(student)

    original = prompts.load

    def changed(stage: str, name: str) -> tuple[str, str]:
        text, version = original(stage, name)
        return (text + "\nChanged.", "changed") if stage == "grade" else (text, version)

    monkeypatch.setattr("autograder2.prompts.load", changed)
    again = Router()
    run(tmp_path / "again", verdicts(KEY), again)

    def keys(root: Path) -> set[str]:
        with closing(sqlite3.connect(root / "cache.sqlite")) as db:
            return {k for (k,) in db.execute("SELECT key FROM calls")}

    assert len(keys(tmp_path / "again")) == 2
    assert not keys(tmp_path / "test") & keys(tmp_path / "again")
    assert all(b["messages"][0]["content"].endswith("\nChanged.") for b in again.bodies)


# --- the rubric written once more on a fresh prepare --------------------------------

RATIONALE = "The response never shows Q2.a's claim, as a grader writes."


def part(id: str, response_type: str, options: list[str] | None = None) -> Part:
    return Part.model_validate(
        {
            "id": id,
            "label": id[1:],
            "prompt": "Answer it.",
            "response_type": response_type,
            "home_page": 1,
            "band": [0.1, 0.2],
            "figures": [],
            "depends_on": [],
            "options": options or [],
        }
    )


SPEC = Spec(
    parts=[
        part("Q1", "numeric"),
        part("Q2.a", "explanation"),
        part("Q2.b", "numeric"),
        part("Q3", "multiple_choice", ["A", "B"]),
    ]
)
FIRST = {
    "Q1": ({"value": 1.0}, "One."),
    "Q2.a": ({"must_include": ["It stops."]}, "It stops."),
    "Q2.b": ({"value": 2.0}, "Two."),
    "Q3": ({"choices": ["A"]}, "A."),
}


def fresh_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, router: Router
) -> tuple[Workspace, Any]:
    """`autograder2 prepare` of SPEC with fake solve and verify (every part verified,
    the second solution agreeing) and the rubric writer and grader behind `router`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "autograder2.cli.Gateway",
        functools.partial(
            Gateway, transport=httpx.MockTransport(router), requests_per_second=1e6
        ),
    )
    shutil.copy(FIXTURES / "blank.pdf", "blank.pdf")
    workspace = Workspace(Path("ws"))
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copy("blank.pdf", workspace.blank_pdf)
    workspace.config_lock_toml.write_text(LOCK)
    write_model(workspace.spec_json, SPEC)
    first = {
        id: PartSolution(
            part_id=id,
            solution=text,
            answer=answer(**fields),
            computation="",
            uses=[],
            alternatives=[],
        )
        for id, (fields, text) in FIRST.items()
    }

    async def solve(*args: object) -> dict[str, PartSolution]:
        return first

    async def verify(*args: object) -> dict[str, Verdict]:
        return {
            id: Verdict(
                part_id=id,
                state="verified",
                compared=AGREE,
                checks=(None, None),
                ruling=None,
                reason=None,
                first=s,
                second=s,
                stands=s,
            )
            for id, s in first.items()
        }

    monkeypatch.setattr("autograder2.cli.solve", solve)
    monkeypatch.setattr("autograder2.cli.verify", verify)
    return workspace, runner.invoke(app, ["prepare", "blank.pdf", "--workspace", "ws"])


@pytest.mark.parametrize("again", ["passes", "fails"])
def test_a_failed_problem_s_rubric_is_written_once_more_with_what_it_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, again: str
) -> None:
    def lose(reading: dict[str, Any], c: dict[str, Any]) -> str | None:
        first = c["requirement"] == "States Q2.a."
        return (
            RATIONALE if c["id"] == "Q2.a.c1" and (first or again == "fails") else None
        )

    router = Router(lose)

    workspace, result = fresh_prepare(tmp_path, monkeypatch, router)

    assert result.exit_code == 0, result.output
    rubrics = [items(b) for b in router.of("Rubric")]
    assert sorted(r[0]["parts"][0]["id"] for r in rubrics) == ["Q1", "Q2.a", "Q2.a"]
    first, second = [r for r in rubrics if r[0]["parts"][0]["id"] == "Q2.a"]
    # What it was given the first time, then each criterion it lost and why.
    assert second == [
        first[0],
        {
            "not_earned": [
                {"id": "Q2.a.c1", "requirement": "States Q2.a.", "rationale": RATIONALE}
            ]
        },
    ]
    key = load_key(workspace.key_yaml)
    assert [c.requirement for p in key.parts for c in p.criteria] == [
        "States Q1.",
        "States Q2.a again.",
        "States Q2.b again.",
        "Chooses A.",
    ]
    states = {p.id: p.state for p in key.parts}
    assert states == {
        "Q1": "verified",
        "Q2.a": "verified" if again == "passes" else "unverified",
        "Q2.b": "verified",
        "Q3": "verified",
    }
    record = read_model(workspace.verification_json, Verification)
    tries = {v.part_id: [t.outcome for t in v.tried] for v in record.parts}
    second_try = "passed" if again == "passes" else "failed"
    assert tries == {
        "Q1": ["passed"],
        "Q2.a": ["failed", second_try],
        "Q2.b": ["passed", "passed"],
        "Q3": [],
    }
    report = workspace.verification_md.read_text()
    q2a = report.split("## Q2.a:")[1].split("## Q2.b:")[0]
    assert "Its rubric was written a second time." in q2a
    assert "Self-test on the first rubric: failed" in q2a
    assert f"Self-test on the rubric written again: {second_try}" in q2a
    assert f"- Q2.a.c1 (final answer, 0 of 1): States Q2.a. Grader: {RATIONALE}" in q2a
    assert "Self-test: not made, as code scores this part." in report.split("## Q3:")[1]
    failed = 0 if again == "passes" else 1
    assert (
        f"Rubric self-test: {3 - failed} passed, {failed} failed, 0 not graded; 2 with"
        " the rubric written a second time."
    ) in result.output.splitlines()
    # The lost criterion and the grader's words on it reach the rubric writer and
    # verification.md, and nothing else.
    log = workspace.run_log.read_text() if workspace.run_log.exists() else ""
    for text in (result.output, result.stderr, log):
        assert "Q2.a's claim" not in text and "States Q2.a" not in text


@pytest.mark.parametrize("fail", ["call", "checks"])
def test_a_fresh_prepare_finishes_when_the_test_call_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    router = Router(fail=fail)

    workspace, result = fresh_prepare(tmp_path, monkeypatch, router)

    assert result.exit_code == 0, result.output
    states = {p.id: p.state for p in load_key(workspace.key_yaml).parts}
    assert states == {
        "Q1": "unverified",
        "Q2.a": "unverified",
        "Q2.b": "unverified",
        "Q3": "verified",
    }
    assert len(router.of("Rubric")) == 2  # a call that failed is no rubric's fault
    assert (
        "3 parts are unverified because a call failed; `autograder2 key verify` tries"
        " them again." in result.stderr.splitlines()
    )
    assert "self-test Q2" in workspace.run_log.read_text()


@pytest.mark.parametrize("fail", ["rewrite", "rewrite checks"])
def test_a_rewrite_that_fails_keeps_the_first_rubric_and_prepare_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: str
) -> None:
    def lose(reading: dict[str, Any], c: dict[str, Any]) -> str | None:
        return RATIONALE if c["id"] == "Q2.a.c1" else None

    router = Router(lose, fail)

    workspace, result = fresh_prepare(tmp_path, monkeypatch, router)

    assert result.exit_code == 0, result.output
    key = load_key(workspace.key_yaml)
    assert [c.requirement for p in key.parts for c in p.criteria] == [
        "States Q1.",
        "States Q2.a.",
        "States Q2.b.",
        "Chooses A.",
    ]
    assert {p.id: p.state for p in key.parts} == {
        "Q1": "verified",
        "Q2.a": "unverified",
        "Q2.b": "verified",
        "Q3": "verified",
    }
    record = read_model(workspace.verification_json, Verification)
    tries = {v.part_id: [t.outcome for t in v.tried] for v in record.parts}
    assert tries == {"Q1": ["passed"], "Q2.a": ["failed"], "Q2.b": ["passed"], "Q3": []}
    log = workspace.run_log.read_text()
    assert "self-test Q2: writing the rubric again" in log
    for text in (result.output, result.stderr, log):
        assert "Q2.a's claim" not in text and "States Q2.a" not in text


# --- the answer check of the key's own answer, which sets no state ------------------


def test_the_key_s_answer_the_check_cannot_read_is_named_and_changes_no_state(
    tmp_path: Path,
) -> None:
    parts = [
        p.model_copy(update={"answer": p.answer.model_copy(update={"unit": "blargs"})})
        if p.id == "Q4.b"
        else p
        for p in KEY.parts
    ]
    key = KEY.model_copy(update={"parts": parts})

    tested, found = run(tmp_path, verdicts(key), Router(), key)

    assert {p.state for p in tested.parts} == {"verified"}
    unread = AnswerCheck(
        verdict="cannot tell", reason="A unit is not one the check can read."
    )
    assert found["Q4.b"].key_check == unread
    # A part with a value or an expression the check reads as matching itself.
    for id in ("Q5.a", "Q5.b"):
        check = found[id].key_check
        assert check is not None and check.verdict == "matches"
    assert found["Q4.a"].key_check is None and found["Q1"].key_check is None
    report = verification_md(
        verification(found, tested, "maker/vision-2", True), tested
    )
    line = (
        "The answer check of the key's own answer: cannot tell. A unit is not one the"
        " check can read. So the check will never decide this part for a student."
    )
    assert report.count(line) == 1 and line in report.split("## Q4.b:")[1]
    assert summary(list(found.values()))[1] == (
        "Key answers the answer check does not read as matching: 1."
    )


def test_rule_7_checks_the_key_s_own_answer_whatever_the_second_says(
    tmp_path: Path,
) -> None:
    other = solved(PARTS["Q4.b"]).model_copy(
        update={"answer": answer(value=3.5, unit="m/s^2")}
    )

    _, found = run(tmp_path, verdicts(KEY, Q4_b={"second": other}), Router())

    assert found["Q4.b"].key_check == AnswerCheck(
        verdict="matches", reason="The value is within tolerance of the key's value."
    )


# --- verification.md ---------------------------------------------------------------


def test_the_need_a_look_line_names_the_check_that_left_each_part_unverified(
    tmp_path: Path,
) -> None:
    def lose(reading: dict[str, Any], c: dict[str, Any]) -> str | None:
        return "Not shown." if c["id"] in ("Q4.b.c1", "Q4.c.c2") else None

    found = verdicts(KEY, Q4_a={"state": "unverified"}, Q4_c={"state": "unverified"})

    key, tried = run(tmp_path, found, Router(lose))

    report = verification_md(verification(tried, key, "maker/vision-2", True), key)
    assert (
        "Parts that need a look: Q4.a (the two solutions), Q4.b (the rubric's"
        " self-test), Q4.c (both). For each, correct key.yaml if it is wrong, then"
        " run `autograder2 key verify` again."
    ) in report.splitlines()
    assert (
        "A part a model grades passes the rubric's self-test when the key's own"
        " solution, graded by the grade stage as a student's response, earns every"
        " criterion in full. This shows that the rubric asks for no more than a"
        " complete correct response holds. It does not show that each criterion's"
        " wording agrees with the key's answer."
    ) in report.splitlines()


def test_a_lost_criterion_s_text_is_one_escaped_line(tmp_path: Path) -> None:
    rationale = "Fine.\n\n## Q2: 2, page 1\n```\nState: verified"
    requirement = "Writes it.\n# Heading"
    lost = Lost(
        id="Q4.b.c1",
        kind="method",
        requirement=requirement,
        awarded=Decimal(0),
        points=Decimal(2),
        rationale=rationale,
    )
    found = verdicts(KEY)
    found["Q4.b"] = found["Q4.b"].model_copy(
        update={"tried": [Trial(outcome="failed", lost=[lost])], "state": "unverified"}
    )

    report = verification_md(verification(found, KEY, "maker/vision-2", True), KEY)

    lines = report.splitlines()
    assert (
        "- Q4.b.c1 (method, 0 of 2): Writes it. \\# Heading Grader: Fine. \\#\\# Q2: 2,"
        " page 1 ``` State: verified"
    ) in lines
    assert [line for line in lines if line.startswith("#")] == ["# Verification"] + [
        f"## {p.id}: {p.label}, page {p.home_page}" for p in KEY.parts
    ]
    q4b = report.split("## Q4.b:")[1].split("## Q4.c:")[0].splitlines()
    assert "```" not in lines and "State: verified" not in q4b  # no forged line


def test_a_record_from_before_the_self_test_or_from_its_first_version_loads(
    tmp_path: Path,
) -> None:
    data = verification(verdicts(KEY), KEY, "maker/vision-2", True).model_dump(
        mode="json"
    )
    before = json.loads(json.dumps(data))
    for v in before["parts"]:
        for field in ("tried", "key_check", "state_before_test"):
            del v[field]
    # The first version of the self-test also graded the second solution.
    first = json.loads(json.dumps(data))
    for v in first["parts"]:
        del v["state_before_test"]
        v["second_tried"] = {"outcome": "passed", "lost": []}
    path = tmp_path / "verification.json"

    for record in (before, first):
        path.write_text(json.dumps(record))
        old = read_model(path, Verification)
        assert all(
            (v.tried, v.key_check, v.state_before_test) == ([], None, None)
            for v in old.parts
        )
