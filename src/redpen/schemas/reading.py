"""The reader's transcript of a part (design 7.4).

`Reading` is what the reader returns. `ReadingRecord` is what is stored: the
reading with the provenance it was produced under and the crops it was read from.
`ProblemReadings` is one stored file, a top-level problem's records.
"""

from typing import Literal

from pydantic import PositiveFloat, PositiveInt

from redpen.schemas.common import Box, DiagramElement, PartId, Provenance, Schema


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
    elements: list[DiagramElement]
    # Numbers, counting from 1, of the lines the reader is unsure of.
    uncertain: list[PositiveInt]
    legibility: Literal["clear", "poor"] | None


# One region image a reading was made from: the page, the box actually rendered (in
# fractions of the page image as ingest wrote it, before any turn) and its pixels
# per native pixel, so above 1 is enlarged past what the source holds. The scale is
# null on a page with no raster ceiling. Crop n of a part, counting from 1, is the
# image at `Workspace.crop_png(student, part, n)`.
class Crop(Schema):
    page: PositiveInt
    box: Box
    scale: PositiveFloat | None


class ReadingRecord(Schema):
    provenance: Provenance
    reading: Reading
    crops: list[Crop]  # empty for a reading code wrote with no model call


class ProblemReadings(Schema):
    records: list[ReadingRecord]
