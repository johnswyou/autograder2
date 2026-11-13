"""The rubric's self-test (design 6.5, its first half). Nobody reviews the key before
a class is graded, so the key's own solution of each part a model grades is turned,
in code, into a response, and the grade stage grades it against the rubric, text
only, exactly as it grades a student: it has to earn every criterion in full. On a
fresh prepare a problem that fails has its rubric written once more and is tried
again. A part that fails, or whose call fails, is unverified: nothing here makes a
part verified.

Two things are measured beside it, and set no state: the answer check of the key's
own answer, and the second solver's solution graded the same way where the two
solvers agreed."""

import tempfile
from collections import defaultdict
from collections.abc import Awaitable, Callable, Collection, Mapping
from pathlib import Path

from redpen.grade.check import check_answer
from redpen.grade.grade import grade_student
from redpen.grade.read import problem_of
from redpen.llm.client import Gateway, finish_all
from redpen.llm.models import ModelInfo
from redpen.prepare.verify import DIFFER, Lost, Trial, Verdict
from redpen.schemas.common import Provenance
from redpen.schemas.grade import AnswerCheck, Grade
from redpen.schemas.key import Answer, Key, KeyPart
from redpen.schemas.reading import FinalAnswer, ProblemReadings, Reading, ReadingRecord
from redpen.schemas.spec import CHOICE_TYPES
from redpen.workspace import Workspace, log_failure, write_model

STUDENT = "key"  # whose readings the responses are, in a scratch workspace
NOBODY = Provenance(key_part_hash=None, prompt_version=None, model_slug=None)

# By problem id, each criterion a try did not earn, as the rubric writer is told.
type NotEarned = dict[str, list[dict[str, str]]]


def response(part_id: str, solution: str, answer: Answer) -> Reading:
    """A solution as a reader would transcribe a typed response holding it: its
    lines as the lines, the answer's choices, value, unit and expression as the
    final answer, its drawing's elements as the elements; answered, clear, and
    nothing uncertain."""
    return Reading(
        part_id=part_id,
        status="answered",
        medium="typed",
        lines=[line.strip() for line in solution.splitlines() if line.strip()],
        final_answer=FinalAnswer(
            choices=answer.choices,
            value=answer.value,
            unit=answer.unit,
            expression=answer.expression,
        ),
        elements=answer.elements,
        uncertain=[],
        addressed_to_grader=[],
        legibility="clear",
    )


async def graded(
    workspace: Workspace,
    gateway: Gateway,
    model: ModelInfo,
    key: Key,
    responses: Mapping[str, Reading],
    problems: Collection[str],
) -> dict[str, Trial]:
    """Each part of `problems` that a model grades, graded against `key` by the
    grade stage itself, `grade_student`, as a student whose readings are
    `responses` and who has no crops: one call per problem, the same message. A
    call that fails, or whose answer fails the stage's checks twice, is written to
    the run log and leaves its problem's parts "not graded"."""
    groups: defaultdict[str, list[KeyPart]] = defaultdict(list)
    for part in key.parts:
        groups[problem_of(part.id)].append(part)
    with tempfile.TemporaryDirectory() as folder:
        # Named as the workspace, so that the grader's session is the grade stage's.
        scratch = Workspace(Path(folder) / workspace.root.name)
        for problem, parts in groups.items():  # every problem: a part's context
            records = [
                ReadingRecord(provenance=NOBODY, reading=responses[p.id], crops=[])
                for p in parts
                if p.id in responses
            ]
            path = scratch.reading_json(STUDENT, problem)
            write_model(path, ProblemReadings(records=records))

        async def one(problem: str, asked: list[KeyPart]) -> dict[str, Trial]:
            mine = key.model_copy(update={"parts": groups[problem]})
            try:
                found = await grade_student(scratch, STUDENT, mine, gateway, model)
            except Exception as error:  # any failure: its parts are not graded
                log_failure(workspace, f"self-test {problem}", error)
                return {p.id: Trial(outcome="not graded", lost=[]) for p in asked}
            return {p.id: trial(p, found[p.id].grade) for p in asked}

        asked = {
            problem: [p for p in parts if p.response_type not in CHOICE_TYPES]
            for problem, parts in groups.items()
            if problem in problems
        }
        done = await finish_all(one(p, parts) for p, parts in asked.items() if parts)
    return {id: t for found in done for id, t in found.items()}


def trial(part: KeyPart, grade: Grade) -> Trial:
    """`grade` of `part` as a try: each criterion below its full points is lost."""
    criteria = {c.id: c for c in part.criteria}
    lost = [
        Lost(
            id=c.id,
            kind=criteria[c.id].kind,
            requirement=criteria[c.id].requirement,
            awarded=c.awarded,
            points=criteria[c.id].points,
            rationale=c.rationale,
        )
        for c in grade.criteria
        if c.awarded < criteria[c.id].points
    ]
    return Trial(outcome="failed" if lost else "passed", lost=lost)


async def self_test(
    workspace: Workspace,
    gateway: Gateway,
    model: ModelInfo,
    key: Key,
    found: Mapping[str, Verdict],
    rewrite: Callable[[NotEarned], Awaitable[Key]] | None = None,
) -> tuple[Key, dict[str, Verdict]]:
    """`key` tried with the `grader` `model`, and each part's verdict `found` with
    the test's results and the state they leave: a part a model grades keeps its
    state only when the key's own solution earns every criterion in full on the
    rubric that stands, and is unverified otherwise. With `rewrite`, on a fresh
    prepare, each problem where a criterion was not earned is given to it once, and
    tried again on the key it returns. Without it, no criterion is written again."""

    async def attempt(key: Key, problems: Collection[str]) -> dict[str, Trial]:
        mine = {p.id: response(p.id, p.solution, p.answer) for p in key.parts}
        return await graded(workspace, gateway, model, key, mine, problems)

    every = {problem_of(p.id) for p in key.parts}
    tried = {id: [t] for id, t in (await attempt(key, every)).items()}
    failed = {problem_of(id) for id, t in tried.items() if t[0].outcome == "failed"}
    if rewrite and failed:
        lost: NotEarned = {
            problem: [
                {"id": c.id, "requirement": c.requirement, "rationale": c.rationale}
                for id, t in tried.items()
                if problem_of(id) == problem
                for c in t[0].lost
            ]
            for problem in failed
        }
        key = await rewrite(lost)
        for id, t in (await attempt(key, failed)).items():
            tried[id].append(t)
    graded_by_model = {p.id for p in key.parts if p.response_type not in CHOICE_TYPES}
    agreed = {
        id
        for id, v in found.items()
        if id in graded_by_model
        and v.second
        and (v.ruling is None or (v.ruling == "same" and v.compared != DIFFER))
    }
    seconds = {
        id: response(id, v.second.solution, v.second.answer)
        for id, v in found.items()
        if v.second
    }
    second = await graded(
        workspace, gateway, model, key, seconds, {problem_of(id) for id in agreed}
    )
    verdicts: dict[str, Verdict] = {}
    parts: list[KeyPart] = []
    for part in key.parts:
        mine = tried.get(part.id, [])
        passed = part.id not in graded_by_model or (
            bool(mine) and mine[-1].outcome == "passed"
        )
        state = found[part.id].state if passed else "unverified"
        reference = response(part.id, part.solution, part.answer)
        check = check_answer(part, reference, key.policies)
        verdicts[part.id] = found[part.id].model_copy(
            update={
                "state": state,
                "tried": mine,
                "key_check": AnswerCheck(**check._asdict()) if check else None,
                "second_tried": second.get(part.id) if part.id in agreed else None,
            }
        )
        parts.append(part.model_copy(update={"state": state}))
    return key.model_copy(update={"parts": parts}), verdicts
