"""Types shared by the schemas: base model, ids, points, boxes and provenance.

Models carry `#` comments, not docstrings: Pydantic copies a class docstring into
the JSON Schema, and model-facing text belongs in `src/autograder2/prompts/`.
"""

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    WithJsonSchema,
)


# Base of every autograder2 schema: unknown fields are rejected.
class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")


# A point value. Pydantic takes an int, a numeric string or a Decimal as is and a
# float as Decimal(str(x)), and writes it to JSON as a string such as "0.5". A model
# is shown a number: offered a string, one returned "".
Points = Annotated[Decimal, Field(ge=0), WithJsonSchema({"type": "number"})]

# `Q3`, `Q7.b`, `Q1.a.i`: letters-only segments below the problem number.
PartId = Annotated[str, StringConstraints(pattern=r"^Q[1-9][0-9]*(\.[a-z]+)*$")]

# `Q7.b.c1`: criterion 1 of part Q7.b.
CriterionId = Annotated[
    str, StringConstraints(pattern=r"^Q[1-9][0-9]*(\.[a-z]+)*\.c[1-9][0-9]*$")
]

# A student's name made safe to use as a directory name.
StudentKey = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]

ResponseType = Literal[
    "multiple_choice",
    "multiple_select",
    "numeric",
    "symbolic",
    "short_text",
    "explanation",
    "derivation",
    "free_body_diagram",
    "graph",
    "other",
]


def _check_box(box: list[float]) -> list[float]:
    if len(box) != 4:
        raise ValueError("a box is [x0, y0, x1, y1]")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("a box needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return box


# [x0, y0, x1, y1] as fractions of the page image. A plain array, not a tuple, so
# the schema has no prefixItems: provider strict modes differ on what they accept.
Box = Annotated[list[float], AfterValidator(_check_box)]


# One labelled element of a drawn response, such as an arrow, a point or a curve.
# `direction` and `position` are free text, null where they do not apply.
class DiagramElement(Schema):
    label: str
    kind: str
    direction: str | None
    position: str | None


# What an artifact was produced under. The key part hash is
# null on artifacts that do not depend on the key. Prompt version and model slug
# are null on artifacts produced by code with no model call.
class Provenance(Schema):
    key_part_hash: str | None
    prompt_version: str | None
    model_slug: str | None
