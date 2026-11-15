"""The spec: every question part found in the blank assignment (design 6.1)."""

from typing import Annotated

from pydantic import AfterValidator, PositiveInt

from autograder2.schemas.common import Box, PartId, ResponseType, Schema


def _check_band(band: list[float]) -> list[float]:
    if len(band) != 2 or not 0 <= band[0] < band[1] <= 1:
        raise ValueError("a band is [top, bottom] with 0 <= top < bottom <= 1")
    return band


# The response types whose answer is a selection among printed options.
CHOICE_TYPES = ("multiple_choice", "multiple_select")


# [top, bottom] of a part on its home page, as fractions of the page height.
Band = Annotated[list[float], AfterValidator(_check_band)]


class Figure(Schema):
    page: PositiveInt
    box: Box


class Part(Schema):
    id: PartId
    label: str
    prompt: str
    response_type: ResponseType
    home_page: PositiveInt
    band: Band
    figures: list[Figure]
    depends_on: list[PartId]
    # Printed option labels such as A to D; empty unless the part is multiple
    # choice or multiple select. Design 6.1 checks there are at least two.
    options: list[str]


class Spec(Schema):
    parts: list[Part]
