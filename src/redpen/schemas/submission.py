"""One student's submission: the ingest record, the fingerprint and the map
(design 7.1 to 7.3).

`Map` is built from what the resolve call returns. `MapRecord` is what is stored:
the map, the facts found about each page, and the provenance.
"""

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


# What a map entry's part was attributed by, strongest first (design 7.3).
Basis = Literal["student_label", "position", "content"]
# `continuation` carries on a response that starts in another entry.
Role = Literal["work", "continuation"]
Confidence = Literal["high", "medium", "low"]


# Where one part's response sits (design 7.3).
class MapEntry(Schema):
    part_id: PartId
    page: PositiveInt
    box: Box
    basis: Basis
    role: Role
    confidence: Confidence


# Added work that no part claims, such as a name or a doodle, and why.
class SetAside(Schema):
    page: PositiveInt
    box: Box
    reason: str


class Map(Schema):
    entries: list[MapEntry]
    unanswered: list[PartId]
    set_aside: list[SetAside]


# What the mapper found about one page of the submission as a whole.
class MapPage(Schema):
    number: PositiveInt
    # Clockwise degrees that turn the student's work upright.
    rotation: Literal[0, 90, 180, 270]
    # Whether the student's work on the page is a photograph or scan of paper.
    photo: bool


class MapRecord(Schema):
    provenance: Provenance
    pages: list[MapPage]
    map: Map
