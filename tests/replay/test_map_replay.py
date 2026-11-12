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
from redpen.grade.read import MARGIN
from redpen.llm.client import Gateway
from redpen.schemas.spec import Spec
from redpen.schemas.submission import MapRecord
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import FIXTURES, SUBMISSIONS, SubmissionTruth, truth_of
from tests.pinned import MAPPER

SPEC = read_model(FIXTURES / "spec.json", Spec)


def coverage(
    found: Sequence[float], truth: Sequence[float], margin: float = 0.0
) -> float:
    """The share of `truth`'s area inside `found` grown by `margin` on each side."""
    width = min(found[2] + margin, truth[2]) - max(found[0] - margin, truth[0])
    height = min(found[3] + margin, truth[3]) - max(found[1] - margin, truth[1])
    area = (truth[2] - truth[0]) * (truth[3] - truth[1])
    return max(width, 0.0) * max(height, 0.0) / area


def assert_matches_the_truth(found: MapRecord, truth: SubmissionTruth) -> None:
    """Each answered part has a `work` entry on the truth's page. Each of the
    truth's entries, work or continuation, is covered to 0.99 of its area by a box
    found for its part on its page, grown by the read stage's margin (`MARGIN`). Each
    unanswered part is unanswered and has no entry, and each page's rotation is the
    truth's. Basis and confidence are not compared."""
    entries = found.map.entries
    for want in truth.map.entries:
        here = [e for e in entries if (e.part_id, e.page) == (want.part_id, want.page)]
        if want.role == "work":
            assert any(e.role == "work" for e in here), (want, here)
        covered = max((coverage(e.box, want.box, MARGIN) for e in here), default=0)
        assert covered >= 0.99, (want, covered, here)
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
