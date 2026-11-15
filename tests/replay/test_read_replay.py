"""Reading each fixture submission with recorded calls to the real reader model.

The inputs are the fixtures' own: `spec.json`, a fresh ingest of the submission
(the session's `fixture_workspace`) and the map record its truth file gives, never
another stage's recording. Each layout class has its own recording,
`tests/recordings/read_<class>.jsonl`. Re-record (paid) with
`AUTOGRADER2_RECORD=1 uv run pytest tests/replay/test_read_replay.py`.
"""

import asyncio
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from autograder2.config import Config
from autograder2.grade.ingest import IngestSummary
from autograder2.grade.read import read_student
from autograder2.llm.client import Gateway
from autograder2.schemas.reading import ReadingRecord
from autograder2.schemas.spec import CHOICE_TYPES, Spec
from autograder2.workspace import Workspace, read_model, write_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, map_of, truth_of
from tests.pinned import READER

SPEC = read_model(FIXTURES / "spec.json", Spec)
TYPES = {p.id: p.response_type for p in SPEC.parts}
POINTS = [-1.5, 0.0, 2.0, 3.5, 10.0]
TRANSFORMATIONS = (*standard_transformations, implicit_multiplication, convert_xor)


def expression(text: str, letters: str) -> sympy.Expr:
    """`text` parsed, after checking it holds only numbers (a point only between
    digits), arithmetic and `letters`: SymPy's parser evaluates its input as Python
    (design 14.2)."""
    allowed = rf"(?:[0-9]+(?:\.[0-9]+)?|[{re.escape(letters)}]|[-+*/^() ])+"
    assert re.fullmatch(allowed, text), f"not a plain expression in {letters}: {text!r}"
    try:
        return parse_expr(text, transformations=TRANSFORMATIONS)
    except Exception as error:  # any failure to parse is reported with the string
        raise AssertionError(f"cannot parse {text!r}: {error!r}") from None


def agree(found: str, truth: str) -> bool:
    letters = "".join(sorted({c for c in truth if c.isalpha()}))
    want, got = expression(truth, letters), expression(found, letters)
    names = sorted(want.free_symbols | got.free_symbols, key=str)
    try:
        return all(
            abs(
                float(want.subs({n: x for n in names}))
                - float(got.subs({n: x for n in names}))
            )
            < 1e-9
            for x in POINTS
        )
    except Exception as error:  # any failure to evaluate is reported with the string
        raise AssertionError(f"cannot evaluate {found!r}: {error!r}") from None


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_reads_as_its_truth_says(
    pdf: Path,
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    shared, summary = fixture_workspace
    truth = truth_of(pdf)
    [record] = [r for r in summary.students if r.student_name == truth.student_name]
    student = record.student_key
    workspace = Workspace(tmp_path / "ws")
    shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
    write_model(workspace.map_json(student), map_of(truth))

    async def main() -> dict[str, ReadingRecord]:
        async with replay(f"read_{truth.layout_class}") as gateway:
            return await read_student(
                workspace, student, SPEC, gateway, READER, Config().image_budget
            )

    records = asyncio.run(main())

    for want in truth.readings:
        got = records[want.part_id].reading
        kind = TYPES[want.part_id]
        assert got.status == want.status, (want.part_id, got.status)
        if want.status != "answered":
            continue
        if kind in CHOICE_TYPES:
            assert set(got.final_answer.choices) == set(want.final_answer.choices), (
                want.part_id
            )
        if want.legibility != "clear":
            continue
        if kind == "numeric":
            assert got.final_answer.value == want.final_answer.value, want.part_id
            units = [
                (a.unit or "").replace(" ", "")
                for a in (got.final_answer, want.final_answer)
            ]
            assert units[0] == units[1], want.part_id
        if kind == "symbolic":
            assert want.final_answer.expression is not None
            assert got.final_answer.expression is not None, want.part_id
            assert agree(got.final_answer.expression, want.final_answer.expression), (
                want.part_id,
                got.final_answer.expression,
            )
