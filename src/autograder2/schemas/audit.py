"""What the class audit found on one key part (design 7.6 and 11): the file
`audit/<part>.json`. The grade files are never rewritten; the report stage applies
each change while the grade it was decided on still stands."""

from typing import Annotated, Any, cast

from pydantic import Field, model_validator

from autograder2.schemas.common import CriterionId, PartId, Points, Schema, StudentKey


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
# it, how many answered, and that answer as the first of them stated it. A file
# stored before `answer` existed has none, and its suspect dumps as it did then.
class Suspect(Schema):
    shared: int
    answered: int
    answer: Annotated[str | None, Field(exclude_if=lambda v: v is None)]

    @model_validator(mode="before")
    @classmethod
    def _stored_before(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {"answer": None, **cast(dict[str, Any], data)}


# `groups`: groups of cases with the same evidence and differing awards; `changed`:
# those with an award changed, in any part (a group split in parts may be changed
# and undecided too); `undecided`: those whose call failed or whose answer failed
# its checks twice, in any part; `split`: those too large
# for one request, decided in parts that may differ. `part_hash` is the key part's
# when the audit ran.
class PartAudit(Schema):
    part_id: PartId
    part_hash: str
    groups: int
    changed: int
    undecided: int
    split: int
    changes: list[Change]
    key_suspect: Suspect | None

    @model_validator(mode="before")
    @classmethod
    def _stored_before(cls, data: Any) -> Any:
        """An audit stored before `split` existed split no group: it loads with 0."""
        if not isinstance(data, dict):
            return data
        return {"split": 0, **cast(dict[str, Any], data)}
