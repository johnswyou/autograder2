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
    # Numbers, counting from 1, of the lines that speak to a reader or a grader or
    # give instructions (design 14.2). Each is still transcribed as a line.
    addressed_to_grader: list[PositiveInt]
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


# A part of a unit read a second time, by another vendor's model, and compared with
# its first reading in code: "failed" when that call failed or its answer failed the
# read stage's checks twice, and then `second` is null. `doubted` says whether the
# reader was unsure of the first reading. What ties the result to what it was made
# for: `part_hash`, the key part's hash then, and `first`, the digest of the first
# reading's stored record (`grade.read.digest`); a result made for another version of
# either is ignored. The first reading is the one graded.
class SecondReading(Schema):
    part_id: PartId
    part_hash: str
    first: str
    doubted: bool
    result: Literal["same", "differ", "not decided", "failed"]
    second: ReadingRecord | None


# One student's parts read a second time: `Workspace.second_json`.
class SecondReadings(Schema):
    records: list[SecondReading]
