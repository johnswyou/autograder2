"""Assemble the key from the spec, the solutions and the rubric (design 6.2 to 6.4,
the first row of 6.4: the proposal used as is)."""

from collections.abc import Mapping
from decimal import Decimal

from redpen.prepare.parse import CHOICE_TYPES
from redpen.prepare.rubric import PartRubric, ProposedCriterion
from redpen.prepare.solve import PartSolution, answer_of, depends_on
from redpen.schemas.key import Criterion, Key, KeyPart, PartialCredit, Policies
from redpen.schemas.spec import Spec

ONE = Decimal(1)  # every criterion's points (design 6.4, the proposal used as is)


def assemble(
    spec: Spec,
    solutions: Mapping[str, PartSolution],
    rubrics: Mapping[str, PartRubric],
) -> Key:
    """The key of `spec` under the default policies: each part's solution and
    answer, and its criteria, each worth one point with ids given in order. A choice
    part's one criterion is written here. Every part is `unverified`, since one
    solver verifies nothing."""
    policies = Policies()
    parts: list[KeyPart] = []
    for part in spec.parts:
        found = solutions[part.id]
        answer = answer_of(found)
        depends = depends_on(spec, part, found)
        if part.response_type in CHOICE_TYPES:
            only = (
                " and no other option"
                if part.response_type == "multiple_select"
                else ""
            )
            requirement = f"Chooses {' and '.join(answer.choices)}{only}."
            proposed = [
                ProposedCriterion(
                    kind="final_answer", requirement=requirement, partial=[]
                )
            ]
            errors: list[str] = []
            carries = False  # fixed options leave no result to carry forward
        else:
            proposed = rubrics[part.id].criteria
            errors = rubrics[part.id].common_errors
            carries = bool(depends)
        criteria = [
            Criterion(
                id=f"{part.id}.c{n}",
                points=ONE,
                kind=c.kind,
                requirement=c.requirement,
                partial=_levels(c.partial, ONE, policies.point_step),
                carry_forward=carries and c.kind == "final_answer",
            )
            for n, c in enumerate(proposed, 1)
        ]
        parts.append(
            KeyPart(
                id=part.id,
                label=part.label,
                home_page=part.home_page,
                response_type=part.response_type,
                depends_on=depends,
                state="unverified",
                prompt=part.prompt,
                answer=answer,
                solution=found.solution,
                criteria=criteria,
                common_errors=errors,
            )
        )
    return Key(policies=policies, parts=parts)


def _levels(
    proposed: list[PartialCredit], points: Decimal, step: Decimal
) -> list[PartialCredit]:
    """Each level rounded down to a multiple of `step`, kept if that leaves it above
    zero and below `points`."""
    kept: list[PartialCredit] = []
    for level in proposed:
        rounded = level.points // step * step
        if 0 < rounded < points:
            kept.append(level.model_copy(update={"points": rounded}))
    return kept
