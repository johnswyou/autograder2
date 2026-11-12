"""One student's submission: the ingest record, the fingerprint and the map
(design 7.1 to 7.3)."""

from typing import Annotated, Literal

from pydantic import Field, PositiveFloat, PositiveInt

from redpen.schemas.common import Box, PartId, Provenance, Schema, StudentKey


class IngestPage(Schema):
    number: PositiveInt
    # Size of the page image: the slide image for a Classkick export.
    width_px: PositiveInt
    height_px: PositiveInt
    slide_number: PositiveInt | None
    native_ppi: PositiveFloat | None


class IngestRecord(Schema):
    student_name: str
    student_key: StudentKey
    source_kind: Literal["classkick", "other"]
    pages: list[IngestPage]


class FingerprintPage(Schema):
    number: PositiveInt
    kind: Literal["template", "foreign"]
    # The blank page this page is a copy of; null on a foreign page.
    blank_page: PositiveInt | None
    # Share of the page image the student added to.
    added_fraction: Annotated[float, Field(ge=0, le=1)]
    added_regions: list[Box]


class Fingerprint(Schema):
    pages: list[FingerprintPage]


# Where one part's response sits (design 7.3).
class MapEntry(Schema):
    part_id: PartId
    page: PositiveInt
    box: Box
    basis: Literal["student_label", "position", "content"]
    # `continuation` carries on a response that starts in another entry.
    role: Literal["work", "continuation"]
    confidence: Literal["high", "medium", "low"]


# Added work that no part claims, such as a name or a doodle, and why.
class SetAside(Schema):
    page: PositiveInt
    box: Box
    reason: str


class Map(Schema):
    provenance: Provenance
    entries: list[MapEntry]
    unanswered: list[PartId]
    set_aside: list[SetAside]
