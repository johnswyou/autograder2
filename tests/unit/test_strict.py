from typing import Any

import pytest
from pydantic import BaseModel

from redpen.grade.audit import Decisions
from redpen.schemas.audit import PartAudit
from redpen.schemas.flags import Flag
from redpen.schemas.grade import Grade, GradeRecord, Override, Review, SecondGrades
from redpen.schemas.key import Key
from redpen.schemas.reading import Reading, ReadingRecord, SecondReadings
from redpen.schemas.spec import Spec
from redpen.schemas.strict import strict_problems
from redpen.schemas.submission import (
    Coverage,
    IngestRecord,
    Map,
    MapRecord,
)

# Each schema carries its nested models under $defs, so these cover every model.
MODELS: list[type[BaseModel]] = [
    Spec,
    Key,
    IngestRecord,
    Map,
    MapRecord,
    Coverage,
    Reading,
    ReadingRecord,
    SecondReadings,
    Grade,
    GradeRecord,
    SecondGrades,
    Override,
    Review,
    Flag,
    PartAudit,
    Decisions,
]


@pytest.mark.parametrize("model", MODELS, ids=lambda model: model.__name__)
def test_every_schema_passes_the_strict_lint(model: type[BaseModel]) -> None:
    assert strict_problems(model.model_json_schema()) == []


def closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("schema", "problem"),
    [
        (
            {**closed({"a": {"type": "string"}}), "required": []},
            "#: property 'a' is not in required",
        ),
        (
            {"type": "array", "items": {"type": "object", "properties": {}}},
            "#/items: additionalProperties is not false",
        ),
        (
            {
                **closed(
                    {
                        "x": {
                            "anyOf": [
                                {"$ref": "#/$defs/A"},
                                closed({"b": {"type": "number"}}),
                            ]
                        }
                    }
                ),
                "$defs": {"A": closed({})},
            },
            "#/properties/x: anyOf has more than one object member",
        ),
    ],
    ids=["optional property", "open object", "union of objects"],
)
def test_lint_reports_each_broken_rule(schema: dict[str, Any], problem: str) -> None:
    assert strict_problems(schema) == [problem]


def test_lint_allows_nullable_values() -> None:
    schema = {
        **closed(
            {
                "text": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "list": {"anyOf": [{"type": "array"}, {"type": "null"}]},
                "object": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]},
            }
        ),
        "$defs": {"A": closed({})},
    }

    assert strict_problems(schema) == []
