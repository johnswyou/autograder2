"""Mapping each fixture submission with recorded calls to the real mapper model.

The inputs are the fixtures' own: `spec.json` and a fresh ingest of the submission
(the session's `fixture_workspace`), never another stage's recording. Each layout
class has its own recording, `tests/recordings/map_<class>.jsonl`. Re-record (paid)
with `REDPEN_RECORD=1 uv run pytest tests/replay/test_map_replay.py`.
"""

import asyncio
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from redpen.config import Config
from redpen.grade.ingest import IngestSummary
from redpen.grade.map import map_student
from redpen.llm.client import Gateway
from redpen.schemas.spec import Spec
from redpen.schemas.submission import MapRecord
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, SubmissionTruth, truth_of
from tests.replay.test_gateway_replay import MODEL as MAPPER

SPEC = read_model(FIXTURES / "spec.json", Spec)


def overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def assert_matches_the_truth(found: MapRecord, truth: SubmissionTruth) -> None:
    """Each answered part has a `work` entry on the truth's page that overlaps the
    truth's box; each unanswered part is unanswered and has no entry; and each
    page's rotation is the truth's. Basis and confidence are not compared."""
    entries = found.map.entries
    for want in (t for t in truth.map.entries if t.role == "work"):
        assert any(
            e.part_id == want.part_id
            and e.role == "work"
            and e.page == want.page
            and overlaps(e.box, want.box)
            for e in entries
        ), (want, [e for e in entries if e.part_id == want.part_id])
    for part in truth.map.unanswered:
        assert part in found.map.unanswered, part
        assert all(e.part_id != part for e in entries), part
    assert [p.rotation for p in found.pages] == truth.rotations


def copy_of(shared: Workspace, student: str, root: Path) -> Workspace:
    """A workspace holding the shared one's key and one student, to write into."""
    workspace = Workspace(root)
    shutil.copytree(shared.root / "key", workspace.root / "key")
    shutil.copytree(shared.student_dir(student), workspace.student_dir(student))
    return workspace


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_maps_as_its_truth_says(
    pdf: Path,
    fixture_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
    tmp_path: Path,
) -> None:
    shared, summary = fixture_workspace
    truth = truth_of(pdf)
    [record] = [r for r in summary.students if r.student_name == truth.student_name]
    student = record.student_key
    workspace = copy_of(shared, student, tmp_path / "ws")

    async def main() -> MapRecord:
        async with replay(f"map_{truth.layout_class}") as gateway:
            return await map_student(
                workspace, student, SPEC, gateway, MAPPER, Config().image_budget
            )

    found = asyncio.run(main())

    assert_matches_the_truth(found, truth)
    assert read_model(workspace.map_json(student), MapRecord) == found
