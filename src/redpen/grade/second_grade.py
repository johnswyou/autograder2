"""The second grade (design 7.5 and 10.4, as task 4.6b rules them), after the grade
stage and before the class audit, under the profiles in `PROFILES`. Each problem of
a student that the grader was asked about, and whose stored grade is the answer to
the request the grade stage makes now, is asked again: the grade stage's own
request (`grade.request`, `grade.ask_grader`) to the same model, with the same
checks and one retry, its answer kept apart in the call cache by a draw number that
is never sent. Code compares the two awards of each criterion. No referee: the first
grade is the grade, and a part with a differing award gets `graders_disagree` in
the report stage, read from `Workspace.second_grade_json`."""

from typing import Literal

from pydantic import ValidationError

from redpen.grade.grade import ask_grader, by_problem, graded_from, request
from redpen.grade.read import digest
from redpen.llm.client import Gateway, finish_all
from redpen.llm.models import ModelInfo
from redpen.prepare.key import part_hash
from redpen.schemas.common import Provenance
from redpen.schemas.grade import (
    Grade,
    GradeRecord,
    ProblemGrades,
    SecondGrade,
    SecondGrades,
)
from redpen.schemas.key import Key, KeyPart
from redpen.workspace import Workspace, log_failure, read_model, write_model

# The stage's name, and the role its calls are booked under in the ledger: the
# model is the grader's.
ROLE = "second grade"
PROFILES = ("balanced", "thorough")
DRAW = 2


async def grade_again(
    workspace: Workspace, student: str, key: Key, gateway: Gateway, model: ModelInfo
) -> list[SecondGrade]:
    """Grade again, with the grader `model`, each problem of `student` the grader
    was asked about whose stored grade answers the request made now, compare each
    part's awards, and store the results with those stored earlier in
    `Workspace.second_grade_json`, written once when every problem is done (none
    when nothing was ever graded again); return what is stored. A result stored
    earlier is replaced only by a new comparison: a problem not asked again keeps
    it, and so does a problem whose call fails, or whose answer still fails its
    checks, where it is tied to the same first grade; its other parts get "failed".
    A failure writes a fixed line and the traceback to the run log, and the other
    problems go on. No grade file is written."""
    path = workspace.second_grade_json(student)
    try:
        earlier = {r.part_id: r for r in read_model(path, SecondGrades).records}
    except (FileNotFoundError, ValidationError):  # none, or of another shape
        earlier = {}

    async def again(problem: str, parts: list[KeyPart]) -> list[SecondGrade]:
        found = request(workspace, student, key, parts)
        if found is None or not found.asked:
            return []
        file = workspace.grade_json(student, problem)
        stored = read_model(file, ProblemGrades).records if file.exists() else []
        first = {r.grade.part_id: r for r in stored}
        # Each part's record as the grade stage would store it now, but for the
        # grade: none is asked again where that stage failed and left an earlier
        # file.
        made = {
            p.id: (
                Provenance(
                    key_part_hash=part_hash(p, key.policies),
                    prompt_version=found.version,
                    model_slug=model.slug,
                ),
                graded_from(p, found.shown),
            )
            for p in found.asked
        }
        if any(
            (r := first.get(i)) is None or (r.provenance, r.reading_digest) != m
            for i, m in made.items()
        ):
            return []

        def tied(
            part: KeyPart,
            result: Literal["same", "differ", "failed"],
            second: GradeRecord | None,
        ) -> SecondGrade:
            return SecondGrade(
                part_id=part.id,
                part_hash=part_hash(part, key.policies),
                reading_digest=made[part.id][1],
                first=digest(first[part.id]),
                result=result,
                second=second,
            )

        try:
            second = await ask_grader(found, key, gateway, model, ROLE, DRAW)
        except Exception as error:  # any failure: the flags stay as they are
            what = f"{ROLE} {student}: {problem} was not graded a second time"
            log_failure(workspace, what, error)
            failed = [tied(p, "failed", None) for p in found.asked]
            return [
                old
                if (old := earlier.get(f.part_id))
                and (old.part_hash, old.reading_digest, old.first)
                == (f.part_hash, f.reading_digest, f.first)
                else f
                for f in failed
            ]
        return [
            tied(
                p,
                "differ"
                if differing(first[p.id].grade, second[p.id].grade)
                else "same",
                second[p.id],
            )
            for p in found.asked
        ]

    done = await finish_all(again(q, parts) for q, parts in by_problem(key))
    results = earlier | {r.part_id: r for problem in done for r in problem}
    records = [results[p.id] for p in key.parts if p.id in results]
    if records:
        write_model(path, SecondGrades(records=records))
    return records


def differing(first: Grade, second: Grade) -> list[str]:
    """The criteria whose awards differ between two grades of a part, in the first's
    order. Awards are compared as numbers: 1, 1.0 and 1.00 are the same."""
    theirs = {c.id: c.awarded for c in second.criteria}
    return [c.id for c in first.criteria if c.awarded != theirs.get(c.id)]
