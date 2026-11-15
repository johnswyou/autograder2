import json
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from autograder2.schemas.common import CriterionId, PartId, Points, StudentKey
from autograder2.schemas.flags import SEVERITY, Flag, FlagCode
from autograder2.schemas.grade import CriterionGrade, Grade, GradeRecord
from autograder2.schemas.key import Key, KeyPart, Policies
from autograder2.schemas.reading import Reading, ReadingRecord
from autograder2.schemas.spec import Part, Spec
from autograder2.schemas.submission import (
    IngestRecord,
    Map,
    MapEntry,
    MapRecord,
)

# The design's examples, with every field present (no optional keys) and the
# answers flattened.

SPEC_PART: dict[str, Any] = {
    "id": "Q7.b",
    "label": "7(b)",
    "prompt": "Calculate the acceleration of the block.",
    "response_type": "numeric",
    "home_page": 5,
    "band": [0.41, 0.63],
    "figures": [{"page": 5, "box": [0.55, 0.1, 0.95, 0.4]}],
    "depends_on": ["Q7.a"],
    "options": [],
}

KEY_PART: dict[str, Any] = {
    "id": "Q7.b",
    "label": "7(b)",
    "home_page": 5,
    "response_type": "numeric",
    "depends_on": ["Q7.a"],
    "state": "verified",
    "prompt": "Calculate the acceleration of the block down the incline.",
    "answer": {
        "choices": [],
        "value": 3.2,
        "unit": "m/s^2",
        "rel_tol": 0.02,
        "accept_also": [3.3],
        "expression": None,
        "elements": [],
        "must_include": [],
    },
    "solution": (
        "Along the incline: m g sin(theta) - mu m g cos(theta) = m a\n"
        "a = g (sin 30 - 0.20 cos 30) = 9.8 (0.500 - 0.173) = 3.2 m/s^2\n"
    ),
    "criteria": [
        {
            "id": "Q7.b.c1",
            "points": 1,
            "kind": "method",
            "requirement": "Writes Newton's second law along the incline with both "
            "the gravity component and friction.",
            "partial": [
                {
                    "points": 0.5,
                    "when": "Both terms present, friction has the wrong sign.",
                }
            ],
            "carry_forward": False,
        },
        {
            "id": "Q7.b.c2",
            "points": 1,
            "kind": "method",
            "requirement": "Uses f = mu N with N = m g cos(theta).",
            "partial": [],
            "carry_forward": False,
        },
        {
            "id": "Q7.b.c3",
            "points": 1,
            "kind": "final_answer",
            "requirement": "States a = 3.2 m/s^2 with units.",
            "partial": [],
            "carry_forward": True,
        },
    ],
    "common_errors": [
        "Adds friction to the gravity component (gives 6.6 m/s^2).",
        "Uses sin for the normal force.",
    ],
}

MAP_ENTRY: dict[str, Any] = {
    "part_id": "Q7.b",
    "page": 4,
    "box": [0.06, 0.41, 0.94, 0.63],
    "basis": "student_label",
    "role": "work",
    "confidence": "high",
}

READING: dict[str, Any] = {
    "part_id": "Q7.b",
    "status": "answered",
    "medium": "handwritten",
    "lines": [
        "mg sin θ + μ mg cos θ = ma",
        "a = 9.8 (0.5 + 0.2 · 0.866)",
        "a = 6.6 m/s²",
    ],
    "final_answer": {"choices": [], "value": 6.6, "unit": "m/s^2", "expression": None},
    "elements": [],
    "uncertain": [],
    "addressed_to_grader": [],
    "legibility": "clear",
}

GRADE: dict[str, Any] = {
    "part_id": "Q7.b",
    "criteria": [
        {
            "id": "Q7.b.c1",
            "awarded": 0.5,
            "evidence_line": 1,
            "evidence_element": None,
            "rationale": "Both terms present; friction added instead of subtracted.",
            "feedback": "Friction points up the incline while the block slides "
            "down, so it subtracts from the gravity component.",
        },
        {
            "id": "Q7.b.c2",
            "awarded": 1,
            "evidence_line": 1,
            "evidence_element": None,
            "rationale": "Uses mu m g cos(theta) for friction.",
            "feedback": "",
        },
        {
            "id": "Q7.b.c3",
            "awarded": 0,
            "evidence_line": 3,
            "evidence_element": None,
            "rationale": "6.6 m/s^2 is not the correct value. The error starts in "
            "this part, so carry-forward does not apply.",
            "feedback": "With the sign corrected, the acceleration comes out to "
            "3.2 m/s².",
        },
    ],
    "transcript_dispute": None,
    "belongs_elsewhere": None,
    "code_overruled": None,
}


MAP: dict[str, Any] = {
    "entries": [MAP_ENTRY],
    "unanswered": ["Q7.c"],
    "set_aside": [{"page": 1, "box": [0.05, 0.0, 0.4, 0.05], "reason": "A name."}],
}

# A part read as blank, scored in code with no model call.
BLANK_GRADE: dict[str, Any] = {
    "part_id": "Q2",
    "criteria": [
        {
            "id": "Q2.c1",
            "awarded": 0,
            "evidence_line": None,
            "evidence_element": None,
            "rationale": "The part was read as blank.",
            "feedback": "",
        }
    ],
    "transcript_dispute": None,
    "belongs_elsewhere": None,
    "code_overruled": None,
}


def provenance(key_part_hash: str | None) -> dict[str, Any]:
    return {
        "key_part_hash": key_part_hash,
        "prompt_version": "1",
        "model_slug": "vendor/model-1",
    }


CODE_PROVENANCE: dict[str, Any] = {
    "key_part_hash": "7d41b0aa",
    "prompt_version": None,
    "model_slug": None,
}

FLAG = Flag(
    code=FlagCode.KEY_UNVERIFIED,
    part_id="Q7.b",
    student_key=None,
    detail="The referee judged the question ambiguous.",
)

INSTANCES: list[BaseModel] = [
    Spec.model_validate({"parts": [SPEC_PART]}),
    Key.model_validate({"policies": Policies(), "parts": [KEY_PART]}),
    IngestRecord.model_validate(
        {
            "student_name": "Sample Student",
            "student_key": "sample-student",
            "source_kind": "classkick",
            "pages": [
                {
                    "number": 1,
                    "width_px": 1004,
                    "height_px": 2008,
                    "slide_number": 1,
                    "native_ppi": 183.6,
                }
            ],
        }
    ),
    Map.model_validate(MAP),
    MapRecord.model_validate(
        {
            "provenance": provenance(None),
            "pages": [{"number": 1, "rotation": 90, "photo": True}],
            "map": MAP,
        }
    ),
    Reading.model_validate(READING),
    ReadingRecord.model_validate(
        {
            "provenance": provenance(None),
            "reading": READING,
            "crops": [
                {"page": 1, "box": [0.1, 0.2, 0.5, 0.4], "scale": 2.0},
                {"page": 2, "box": [0.0, 0.0, 1.0, 0.5], "scale": None},
            ],
        }
    ),
    Grade.model_validate(GRADE),
    GradeRecord.model_validate({"provenance": provenance("3f2a9c1e"), "grade": GRADE}),
    GradeRecord.model_validate({"provenance": CODE_PROVENANCE, "grade": BLANK_GRADE}),
    FLAG,
]


def instance_id(instance: BaseModel) -> str:
    if isinstance(instance, GradeRecord) and instance.provenance.model_slug is None:
        return "GradeRecord-from-code"
    return type(instance).__name__


@pytest.mark.parametrize("instance", INSTANCES, ids=instance_id)
def test_round_trip_through_json(instance: BaseModel) -> None:
    assert type(instance).model_validate_json(instance.model_dump_json()) == instance


def test_points_keep_the_decimal_a_float_was_written_as() -> None:
    points = TypeAdapter(Points)

    assert str(points.validate_python(0.1)) == "0.1"
    assert points.validate_python(1) == Decimal(1)
    assert points.validate_python("0.25") == Decimal("0.25")
    assert points.dump_json(Decimal("0.5")) == b'"0.5"'


def test_points_show_a_model_a_number() -> None:
    assert TypeAdapter(Points).json_schema() == {"type": "number"}


@pytest.mark.parametrize("awarded", [0.5, "0.5"], ids=["number", "string"])
def test_points_load_from_a_json_number_or_string(awarded: float | str) -> None:
    document = json.dumps(GRADE["criteria"][0] | {"awarded": awarded})

    assert CriterionGrade.model_validate_json(document).awarded == Decimal("0.5")


def test_points_reject_an_empty_string() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Points).validate_json('""')


def test_policies_default_to_the_design() -> None:
    assert Policies().model_dump(mode="json") == {
        "carry_forward": True,
        "alternative_methods": True,
        "rel_tol": 0.02,
        "require_units": True,
        "penalise_sig_figs": False,
        "penalise_contradictory_answers": True,
        "point_step": "0.25",
    }


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (PartId, "Q3"),
        (PartId, "Q7.b"),
        (PartId, "Q12.ii"),
        (PartId, "Q1.a.i"),
        (CriterionId, "Q3.c1"),
        (CriterionId, "Q7.b.c12"),
        (CriterionId, "Q1.a.i.c1"),
    ],
)
def test_id_shapes_are_accepted(kind: Any, value: str) -> None:
    assert TypeAdapter(kind).validate_python(value) == value


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (PartId, "7.b"),
        (PartId, "Q7b"),
        (PartId, "Q7.B"),
        (PartId, "Q07"),
        (PartId, "Q7.b.c1"),
        (CriterionId, "Q7.b"),
        (CriterionId, "Q7.b.1"),
        (StudentKey, "../other"),
        (StudentKey, "a b"),
        (Points, -1),
    ],
)
def test_bad_value_is_rejected(kind: Any, value: object) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(kind).validate_python(value)


@pytest.mark.parametrize(
    "box",
    [
        [0.5, 0.1, 0.5, 0.2],
        [0.6, 0.1, 0.5, 0.2],
        [0.1, 0.3, 0.9, 0.2],
        [0.1, 0.1, 1.2, 0.2],
        [-0.1, 0.1, 0.5, 0.2],
        [0.1, 0.1, 0.5],
    ],
)
def test_bad_box_is_rejected(box: list[float]) -> None:
    with pytest.raises(ValidationError):
        MapEntry.model_validate({**MAP_ENTRY, "box": box})


def test_band_must_run_top_to_bottom() -> None:
    with pytest.raises(ValidationError):
        Part.model_validate({**SPEC_PART, "band": [0.63, 0.41]})


@pytest.mark.parametrize(
    ("response_type", "answer"),
    [
        ("numeric", {"choices": ["B"]}),
        ("multiple_choice", {"choices": ["B"], "value": 3.2}),
        ("numeric", {"value": None, "unit": None, "rel_tol": None, "accept_also": []}),
    ],
    ids=["numeric with a choice", "choice with a value", "numeric with no value"],
)
def test_answer_must_fit_the_response_type(
    response_type: str, answer: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError, match="answer"):
        KeyPart.model_validate(
            {
                **KEY_PART,
                "response_type": response_type,
                "answer": {**KEY_PART["answer"], **answer},
            }
        )


def test_every_flag_code_has_a_severity() -> None:
    assert len(FlagCode) == 16
    assert set(SEVERITY) == set(FlagCode)


def test_flag_severity_is_looked_up_not_stored() -> None:
    assert FLAG.severity == "high"
    assert "severity" not in FLAG.model_dump()
