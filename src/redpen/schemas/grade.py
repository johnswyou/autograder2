"""The grader's decisions on a part (design 7.5).

`Grade` is what the grader returns. `GradeRecord` is what is stored: the grade
with the provenance, including the key part hash, and the reading it was produced
from. `ProblemGrades` is one stored file, a top-level problem's records.
"""

from typing import Any, Literal, cast

from pydantic import PositiveInt, model_validator

from redpen.schemas.common import CriterionId, PartId, Points, Provenance, Schema

# The answer check's verdict on a stated final answer (redpen.grade.check).
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
