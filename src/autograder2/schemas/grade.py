"""The grader's decisions on a part (design 7.5), and the instructor's corrections
(design 11) and reviews (design 12).

`Grade` is what the grader returns. `GradeRecord` is what is stored: the grade
with the provenance, including the key part hash, and the reading it was produced
from. `ProblemGrades` is one stored file, a top-level problem's records.
`SecondGrades` is one student's second grades. `Override` is one entry of
`overrides.yaml`, `Review` one of `reviews.yaml`.
"""

from typing import Any, Literal, cast

from pydantic import ConfigDict, PositiveInt, model_validator

from autograder2.schemas.common import (
    CriterionId,
    PartId,
    Points,
    Provenance,
    Schema,
    StudentKey,
)

# The answer check's verdict on a stated final answer (autograder2.grade.check).
type Verdict = Literal["matches", "does not match", "cannot tell"]


# The evidence is a line number or a diagram element number in the reading,
# counting from 1, or null. A model writes its answer in field order, so the
# rationale and the evidence come before the award they decide.
class CriterionGrade(Schema):
    id: CriterionId
    rationale: str
    evidence_line: PositiveInt | None
    evidence_element: PositiveInt | None
    awarded: Points
    feedback: str


class Grade(Schema):
    part_id: PartId
    criteria: list[CriterionGrade]
    # Each is the grader's reason, or null when it has none to give.
    transcript_dispute: str | None  # the reading is wrong in a way that matters
    belongs_elsewhere: str | None  # the response answers a different part
    code_overruled: str | None  # the computed answer check is wrong


# The answer check the grader was given beside the reading, as it was then.
class AnswerCheck(Schema):
    verdict: Verdict
    reason: str


class GradeRecord(Schema):
    provenance: Provenance
    grade: Grade
    # None for a part with no check, decided by code, or graded before checks.
    # Required, so that the schema passes the strict lint like every other.
    check: AnswerCheck | None
    # The digest of the stored reading record graded (`grade.read.digest`); None in a
    # record stored before it existed.
    reading_digest: str | None

    @model_validator(mode="before")
    @classmethod
    def _stored_before(cls, data: Any) -> Any:
        """A record stored before `check` or `reading_digest` existed has none: it
        loads with None."""
        if not isinstance(data, dict):
            return data
        return {"check": None, "reading_digest": None, **cast(dict[str, Any], data)}


# A part read as unreadable has no record, nor has a choice part answered with no
# option read: its score is withheld, never zero.
class ProblemGrades(Schema):
    records: list[GradeRecord]


# A part the grader graded a second time, with the same request, and compared with
# its first grade in code, award by award: "differ" when one award differs, and
# "failed" when that call failed or its answer failed the grade stage's checks
# twice, and then `second` is null. What ties the result to the first grade it was
# compared with: `part_hash`, the key part's hash then; `reading_digest`, the digest
# of the readings it was graded from (`grade.grade.graded_from`); and `first`, the
# digest of the first grade's stored record (`grade.read.digest`). A result made for
# another version of any of them is ignored. The first grade is the grade.
class SecondGrade(Schema):
    part_id: PartId
    part_hash: str
    reading_digest: str
    first: str
    result: Literal["same", "differ", "failed"]
    second: GradeRecord | None


# One student's parts graded a second time: `Workspace.second_grade_json`.
class SecondGrades(Schema):
    records: list[SecondGrade]


# One of the instructor's corrections: `student` gets `points` on `criterion`, in the
# criterion's own points, while the key part of the criterion has `part_hash`, its
# hash when the correction was recorded. `note` is the instructor's, for the student.
class Override(Schema):
    # A hand-written entry may leave the note out.
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    student: StudentKey
    criterion: CriterionId
    points: Points
    part_hash: str
    note: str | None = None


# One of the instructor's reviews (`autograder2 review`): the review list's entry of
# `student` and `part` was looked at, and closes its flags while the key part still
# has `part_hash` and the entry still shows what has `digest`
# (`autograder2.report.report.shown`). A key part's entry has no student; a whole
# submission's has no part and no part hash.
class Review(Schema):
    # A hand-written entry may leave out what its entry does not have.
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    student: StudentKey | None = None
    part: PartId | None = None
    part_hash: str | None = None
    digest: str
