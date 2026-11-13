"""What the class audit found on one key part (design 7.6 and 11): the file
`audit/<part>.json`. The grade files are never rewritten; the report stage applies
each change while the grade it was decided on still stands."""

from redpen.schemas.common import CriterionId, PartId, Points, Schema, StudentKey


# One award the audit changed. It applies while the student's stored grade of the
# part still has `part_hash` and the award `before`. `reason` is the model's.
class Change(Schema):
    student_key: StudentKey
    criterion: CriterionId
    part_hash: str
    before: Points
    after: Points
    reason: str


# The class's most common final answer is not the key's: how many students share
# it, and how many answered.
class Suspect(Schema):
    shared: int
    answered: int


# `groups`: groups of cases with the same evidence and differing awards; `changed`:
# those decided again with an award changed; `undecided`: those whose call failed
# or whose answer failed its checks twice. `part_hash` is the key part's when the
# audit ran.
class PartAudit(Schema):
    part_id: PartId
    part_hash: str
    groups: int
    changed: int
    undecided: int
    changes: list[Change]
    key_suspect: Suspect | None
