"""One student's submission: the ingest record and the map (design 7.1 and 7.3).

`Map` is built from what the resolve call returns. `MapRecord` is what is stored:
the map, the facts found about each page, and the provenance.
"""

from typing import Literal

from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt

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
    # A map stored before names could be masked leaves `names` out.
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    number: PositiveInt
    # Clockwise degrees that turn the student's work upright.
    rotation: Literal[0, 90, 180, 270]
    # Whether the student's work on the page is a photograph or scan of paper.
    photo: bool
    # Where a student's name is, found only with names masked: blanked in every
    # image of the page sent after the map.
    names: list[Box] = Field(default_factory=list[list[float]])
    # Where a name overlaps a mapped response, so is left whole: a mask never
    # covers a response.
    names_on_responses: list[Box] = Field(default_factory=list[list[float]])


class MapRecord(Schema):
    provenance: Provenance
    pages: list[MapPage]
    map: Map


# One page of a submission against its class's background (design 7.3).
class CoveragePage(Schema):
    number: PositiveInt
    # Why the page was not compared with the background; null when it was.
    not_compared: (
        Literal[
            "few comparable",
            "not classkick",
            "other rendering",
            "unreadable",
            "check failed",
        ]
        | None
    )
    # Each patch of dense ink that no region of the map claims.
    patches: list[Box]


# Where a student's ink lies outside every region of the map, for the map record
# whose digest it holds.
class Coverage(Schema):
    map_digest: str
    pages: list[CoveragePage]
