"""The grader's decisions on a part (design 7.5).

`Grade` is what the grader returns. `GradeRecord` is what is stored: the grade
with the provenance, including the key part hash, it was produced under.
"""

from pydantic import PositiveInt

from redpen.schemas.common import CriterionId, PartId, Points, Provenance, Schema


# The evidence is a line number or a diagram element number in the reading,
# counting from 1, or null.
class CriterionGrade(Schema):
    id: CriterionId
    awarded: Points
    evidence_line: PositiveInt | None
    evidence_element: PositiveInt | None
    rationale: str
    feedback: str


class Grade(Schema):
    part_id: PartId
    criteria: list[CriterionGrade]
    # Each is the grader's reason, or null when it has none to give.
    transcript_dispute: str | None  # the reading is wrong in a way that matters
    belongs_elsewhere: str | None  # the response answers a different part
    code_overruled: str | None  # the computed answer check is wrong


class GradeRecord(Schema):
    provenance: Provenance
    grade: Grade
