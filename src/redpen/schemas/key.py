"""The key: answers, solutions, criteria and policies (design 6.2, 6.3, 6.6)."""

from decimal import Decimal
from typing import Literal, Self

from pydantic import ConfigDict, PositiveInt, model_validator

from redpen.schemas.common import (
    CriterionId,
    DiagramElement,
    PartId,
    Points,
    ResponseType,
    Schema,
)


# The correct answer, one shape for every response type: fields that do not apply
# are null or empty. `elements` lists what a correct drawing shows; `must_include`
# lists the features or claims a correct response must contain.
class Answer(Schema):
    choices: list[str]
    value: float | None
    unit: str | None
    rel_tol: float | None
    accept_also: list[float]
    expression: str | None
    elements: list[DiagramElement]
    must_include: list[str]


_NUMBER = {"value", "unit", "rel_tol", "accept_also"}

# The answer fields each response type may fill.
_ANSWER_FIELDS: dict[ResponseType, set[str]] = {
    "multiple_choice": {"choices"},
    "multiple_select": {"choices"},
    "numeric": _NUMBER,
    "symbolic": {"expression"},
    "short_text": {"must_include"},
    "explanation": {"must_include"},
    "derivation": _NUMBER | {"expression", "must_include"},
    "free_body_diagram": {"elements", "must_include"},
    "graph": {"elements", "must_include"},
    "other": set(Answer.model_fields),
}


def answer_problem(response_type: ResponseType, answer: Answer) -> str | None:
    """Why `answer` does not fit `response_type`, or None if it does."""
    filled = {name for name, value in answer if value not in (None, [])}
    allowed = _ANSWER_FIELDS[response_type]
    if not filled and response_type != "other":
        return f"a {response_type} answer fills none of {sorted(allowed)}"
    if filled - allowed:
        return f"a {response_type} answer cannot fill {sorted(filled - allowed)}"
    return None


class PartialCredit(Schema):
    points: Points
    when: str


class Criterion(Schema):
    id: CriterionId
    points: Points
    kind: Literal["method", "final_answer"]
    requirement: str
    partial: list[PartialCredit]
    carry_forward: bool


class KeyPart(Schema):
    id: PartId
    label: str
    home_page: PositiveInt
    response_type: ResponseType
    depends_on: list[PartId]
    state: Literal["verified", "refereed", "unverified"]
    prompt: str
    answer: Answer
    solution: str
    criteria: list[Criterion]
    common_errors: list[str]

    @model_validator(mode="after")
    def _answer_fits_response_type(self) -> Self:
        if problem := answer_problem(self.response_type, self.answer):
            raise ValueError(problem)
        return self


# The whole-assignment policies of design 6.3, with its defaults.
class Policies(Schema):
    # The key always spells out every policy, so its schema lists them as required.
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    # A wrong result from an earlier part, used correctly in a later part, earns the
    # later points. An error made inside a part is not carried forward within it.
    carry_forward: bool = True
    # Any valid method earns method credit.
    alternative_methods: bool = True
    # Relative tolerance on numeric answers.
    rel_tol: float = 0.02
    # A final-answer criterion requires correct units.
    require_units: bool = True
    # Significant figures are not penalised.
    penalise_sig_figs: bool = False
    # Two different final answers for one part earn no final-answer credit.
    penalise_contradictory_answers: bool = True
    point_step: Points = Decimal("0.25")


# The instructor's points: a total for the assignment, or one maximum per page of the
# blank in page order, never both. It lies outside every part and the policies, so
# that setting it changes no part hash and re-grades nobody. Plain Decimals, so that
# a value below zero is refused by the point check, which names its page.
class PointConstraint(Schema):
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    total: Decimal | None = None
    pages: list[Decimal] | None = None

    @model_validator(mode="after")
    def _a_total_or_pages(self) -> Self:
        if (self.total is None) == (self.pages is None):
            raise ValueError("give either a total or page maximums")
        return self


class Key(Schema):
    model_config = ConfigDict(
        json_schema_mode_override="serialization",
        json_schema_serialization_defaults_required=True,
    )

    policies: Policies
    parts: list[KeyPart]
    # None, as in a key written before it existed: each part is worth its criteria.
    points: PointConstraint | None = None
