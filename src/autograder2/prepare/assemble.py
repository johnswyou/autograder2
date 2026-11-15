"""Assemble the key from the spec, the solutions and the rubric (the rubric's
proposal is used as is)."""

from collections.abc import Mapping
from decimal import Decimal

from autograder2.prepare.rubric import PartRubric, ProposedCriterion
from autograder2.prepare.solve import PartSolution, answer_of, depends_on
from autograder2.prepare.verify import State
from autograder2.schemas.key import Criterion, Key, KeyPart, PartialCredit, Policies
from autograder2.schemas.spec import CHOICE_TYPES, Spec

ONE = Decimal(1)  # every criterion's points (the proposal used as is)


def assemble(
    spec: Spec,
    solutions: Mapping[str, PartSolution],
    rubrics: Mapping[str, PartRubric],
    states: Mapping[str, State],
) -> Key:
    """The key of `spec` under the default policies: each part's solution and
    answer, its state as verification left it, and its criteria, each worth one
    point with ids given in order. A choice part's one criterion is written here."""
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
            carries = False  # code scores it and cannot honour the flag
        else:
            proposed = rubrics[part.id].criteria
            errors = rubrics[part.id].common_errors
            # Every criterion of a dependent part, method ones too: the flag alone
            # decides carry-forward in the grader.
            carries = bool(depends)
        criteria = [
            Criterion(
                id=f"{part.id}.c{n}",
                points=ONE,
                kind=c.kind,
                requirement=c.requirement,
                partial=_levels(c.partial, ONE, policies.point_step),
                carry_forward=carries,
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
                state=states[part.id],
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
