"""The second read of one fixture submission, with recorded calls to the real
second reader (`tests.pinned.SECOND`, of another vendor than the reader).

No truth reading of the fixtures is one the reader was unsure of, so the test
stores the small-writing student's truth readings as the read stage would, with
their crops, and marks the reader's doubt on some of them by hand: a line
uncertain, or poor legibility on a drawing with no line. The other inputs are the
fixtures' own: `spec.json`, `key.json`, the session's ingest and the truth's map.

The recording, `tests/recordings/second_small-writing.jsonl`, was made by running
`seed` and `read_again` live into a persistent scratch cache and keeping exactly
the calls made (task 4.6a report). Under `REDPEN_RECORD=1` the `replay` fixture
would call the second reader again (paid) and rewrite it once the test passes."""

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from redpen.config import Config
from redpen.grade.ingest import IngestSummary
from redpen.grade.read import digest
from redpen.grade.second import read_again
from redpen.llm.client import Gateway
from redpen.prepare.key import part_hash
from redpen.schemas.key import Key
from redpen.schemas.reading import ProblemReadings, SecondReading
from redpen.workspace import Workspace, read_model, write_model
from tests.fixture_truth import (
    FIXTURES,
    SPEC,
    SUBMISSIONS,
    map_of,
    store_truth_readings,
    truth_of,
)
from tests.pinned import SECOND

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())
PARTS = {p.id: p for p in KEY.parts}
CLASS = "small-writing"
# The reader's doubt, marked by hand on the truth reading of each part.
DOUBT: dict[str, dict[str, Any]] = {
    "Q1": {"uncertain": [1]},
    "Q4.b": {"uncertain": [3]},
    "Q4.c": {"uncertain": [1]},
    "Q5.a": {"uncertain": [2]},
    "Q5.c": {"legibility": "poor"},
}


def seed(
    shared: Workspace, summary: IngestSummary, root: Path
) -> tuple[Workspace, str]:
    """`root/ws` holding the class's student as ingested in `shared`, the truth's
    map, and its truth readings with `DOUBT` marked."""
    [pdf] = [p for p in SUBMISSIONS if truth_of(p).layout_class == CLASS]
    truth = truth_of(pdf)
    [record] = [s for s in summary.students if s.student_name == truth.student_name]
    student = record.student_key
    workspace = Workspace(root / "ws")
    shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
    write_model(workspace.map_json(student), map_of(truth))
    store_truth_readings(workspace, student, truth)
    for problem in dict.fromkeys(i.split(".")[0] for i in DOUBT):
        path = workspace.reading_json(student, problem)
        records = [
            r.model_copy(
                update={
                    "reading": r.reading.model_copy(update=DOUBT[r.reading.part_id])
                }
            )
            if r.reading.part_id in DOUBT
            else r
            for r in read_model(path, ProblemReadings).records
        ]
        write_model(path, ProblemReadings(records=records))
    return workspace, student


def test_each_unit_in_doubt_is_read_again_once_and_compared(
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    workspace, student = seed(*fixture_workspace, tmp_path)

    async def main() -> list[SecondReading]:
        async with replay(f"second_{CLASS}") as gateway:
            return await read_again(
                workspace, student, SPEC, KEY, gateway, SECOND, Config().image_budget
            )

    found = asyncio.run(main())

    # One call for each of the units page 1, Q4 and Q5, and every part of them
    # compared.
    assert [(r.part_id, r.doubted) for r in found] == [
        (p.id, p.id in DOUBT) for p in SPEC.parts
    ]
    first = {
        r.reading.part_id: r
        for problem in ("Q1", "Q2", "Q3", "Q4", "Q5")
        for r in read_model(
            workspace.reading_json(student, problem), ProblemReadings
        ).records
    }
    for r in found:
        assert r.first == digest(first[r.part_id])
        assert r.part_hash == part_hash(PARTS[r.part_id], KEY.policies)
        assert r.second is not None
        assert r.second.provenance.model_slug == SECOND.slug
    # The drawing's and the graph's elements are described in other words, so code
    # cannot tell.
    assert {r.part_id: r.result for r in found} == {
        "Q1": "same",
        "Q2": "same",
        "Q3": "same",
        "Q4.a": "not decided",
        "Q4.b": "same",
        "Q4.c": "same",
        "Q5.a": "same",
        "Q5.b": "same",
        "Q5.c": "not decided",
    }
    # Where the two are the same, the second reader stated the truth's answer.
    for r in found:
        if r.result == "same" and r.second:
            truth = first[r.part_id].reading.final_answer
            assert r.second.reading.final_answer == truth, r.part_id
