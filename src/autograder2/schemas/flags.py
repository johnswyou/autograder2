"""Flags on a student and part, and their severities (design section 9)."""

from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Literal

from autograder2.schemas.common import PartId, Schema, StudentKey


class FlagCode(StrEnum):
    KEY_UNVERIFIED = "key_unverified"
    KEY_SUSPECT = "key_suspect"
    UNREADABLE = "unreadable"
    READERS_DISAGREE = "readers_disagree"
    GRADERS_DISAGREE = "graders_disagree"
    NO_EVIDENCE = "no_evidence"
    ADDRESSED_TO_GRADER = "addressed_to_grader"
    INCOMPLETE = "incomplete"
    UNCLAIMED_WORK = "unclaimed_work"
    WEAK_ATTRIBUTION = "weak_attribution"
    DUPLICATE_RESPONSE = "duplicate_response"
    HARD_TO_READ = "hard_to_read"
    LOW_RESOLUTION = "low_resolution"
    TRANSCRIPT_DISPUTE = "transcript_dispute"
    CODE_OVERRULED = "code_overruled"
    AUDIT_CHANGED = "audit_changed"


Severity = Literal["high", "medium", "low"]

SEVERITY: Final[Mapping[FlagCode, Severity]] = {
    FlagCode.KEY_UNVERIFIED: "high",
    FlagCode.KEY_SUSPECT: "high",
    FlagCode.UNREADABLE: "high",
    FlagCode.READERS_DISAGREE: "high",
    FlagCode.GRADERS_DISAGREE: "high",
    FlagCode.NO_EVIDENCE: "high",
    FlagCode.ADDRESSED_TO_GRADER: "high",
    FlagCode.INCOMPLETE: "high",
    FlagCode.UNCLAIMED_WORK: "medium",
    FlagCode.WEAK_ATTRIBUTION: "medium",
    FlagCode.DUPLICATE_RESPONSE: "medium",
    FlagCode.HARD_TO_READ: "medium",
    FlagCode.LOW_RESOLUTION: "medium",
    FlagCode.TRANSCRIPT_DISPUTE: "medium",
    FlagCode.CODE_OVERRULED: "low",
    FlagCode.AUDIT_CHANGED: "low",
}


# The part id is null for a flag on a whole submission; the student key is null
# for a flag on every student, such as key_unverified.
class Flag(Schema):
    code: FlagCode
    part_id: PartId | None
    student_key: StudentKey | None
    detail: str

    @property
    def severity(self) -> Severity:
        return SEVERITY[self.code]
