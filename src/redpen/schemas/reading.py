"""The reader's transcript of a part (design 7.4).

`Reading` is what the reader returns. `ReadingRecord` is what is stored: the
reading with the provenance it was produced under.
"""

from typing import Literal

from pydantic import PositiveInt

from redpen.schemas.common import DiagramElement, PartId, Provenance, Schema


# The student's final answer, one shape for every response type. All null or empty
# when the student gave none.
class FinalAnswer(Schema):
    choices: list[str]
    value: float | None
    unit: str | None
    expression: str | None


class Reading(Schema):
    part_id: PartId
    status: Literal["answered", "blank", "unreadable"]
    medium: Literal["handwritten", "typed", "mixed"] | None
    lines: list[str]
    final_answer: FinalAnswer
    diagram: list[DiagramElement]
    # Numbers, counting from 1, of the lines the reader is unsure of.
    uncertain: list[PositiveInt]
    legibility: Literal["clear", "poor"] | None


class ReadingRecord(Schema):
    provenance: Provenance
    reading: Reading
