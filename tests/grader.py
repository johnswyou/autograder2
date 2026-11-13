"""A mock grader's answer to a request of the grade stage, for the tests whose mock
OpenRouter also meets the rubric's self-test."""

import json
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

from redpen.grade.grade import Grades
from redpen.schemas.grade import CriterionGrade, Grade

# Given a reading and a criterion of its part, as the request has them: the
# rationale of a 0 award, or None for full points.
type Loses = Callable[[dict[str, Any], dict[str, Any]], str | None]


def grades(body: Mapping[str, Any], lose: Loses = lambda reading, c: None) -> Grades:
    """Every criterion of each part the request `body` asks to grade earned in
    full, citing line 1 (element 1 when the reading has no line), except where
    `lose` gives a rationale: 0, with that rationale."""
    texts = [
        json.loads(c["text"])
        for c in body["messages"][1]["content"]
        if c["type"] == "text"
    ]
    keys = {t["id"]: t for t in texts if "criteria" in t}
    found: list[Grade] = []
    for reading in (t["grade"] for t in texts if "grade" in t):
        criteria: list[CriterionGrade] = []
        for c in keys[reading["part_id"]]["criteria"]:
            why = lose(reading, c)
            criteria.append(
                CriterionGrade(
                    id=c["id"],
                    rationale=why or "Meets it.",
                    evidence_line=1 if reading["lines"] else None,
                    evidence_element=None if reading["lines"] else 1,
                    awarded=Decimal(0) if why else Decimal(c["points"]),
                    feedback="Show it." if why else "",
                )
            )
        found.append(
            Grade(
                part_id=reading["part_id"],
                criteria=criteria,
                transcript_dispute=None,
                belongs_elsewhere=None,
                code_overruled=None,
            )
        )
    return Grades(grades=found)
